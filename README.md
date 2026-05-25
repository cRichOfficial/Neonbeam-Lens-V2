# Laser Engraver Object Detection System

FastAPI service for a Raspberry Pi 5 laser engraver with CSI camera and Hailo-8L NPU. Provides camera control, AprilTag bed calibration, object detection/segmentation, and debug visualization.

**Deployment target:** `crichards999@neonbeam-lens.richwerks.local:~/object-detection-v2`

## Features

- Camera settings (exposure, gain, mount height), snapshot, and MJPEG stream
- AprilTag homography calibration with mm coordinate mapping
- AprilTag PDF generator (IDs 0–3, to-scale on US Letter)
- YOLOv8 object detection with Hailo NPU (production) and Ultralytics CPU fallback (dev)
- Parallax compensation for object height above the bed
- Debug image overlay (tags, grid, detections)
- Custom model training pipeline scaffold

## Quick Start (neonbeam-lens.richwerks.local)

```bash
cd ~/object-detection-v2
bash deploy/setup-pi.sh
```

Or manually:

```bash
sudo apt install hailo-all python3-picamera2 python3-venv
cd ~/object-detection-v2
python3 -m venv .venv --system-site-packages   # required — picamera2 is a system package
source .venv/bin/activate
pip install -r requirements-pi.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Note:** `picamera2` and Hailo bindings are installed via `apt`, not `pip`. The venv **must** be created with `--system-site-packages` so those modules are visible. Without it, the service falls back to a mock camera.

Open `http://neonbeam-lens.richwerks.local:8000/docs` for interactive API documentation.

### Troubleshooting: mock camera warning

If you see `picamera2 unavailable (No module named 'picamera2'), using mock camera`:

```bash
cd ~/object-detection-v2
rm -rf .venv
bash deploy/setup-pi.sh
sudo systemctl restart laser-detection   # if using systemd
```

Verify with `curl http://localhost:8000/health` — `"camera": { "mode": "picamera2", "mock": false }`.

If `deploy/setup-pi.sh` fails with `$'\r': command not found`, the script has Windows line endings. Redeploy from Windows with `.\deploy\deploy.ps1` (which normalizes `.sh` files), or on the Pi run: `sed -i 's/\r$//' deploy/setup-pi.sh`

If you see `Bad substitution` or `source: not found`, you ran it with `sh` instead of `bash`. Use `bash deploy/setup-pi.sh` (or `./deploy/setup-pi.sh` after `chmod +x`). The same applies to `scripts/start.sh` — always use `bash scripts/start.sh`.

### Troubleshooting: Hailo `HAILO_OUT_OF_PHYSICAL_DEVICES`

The Pi has **one** Hailo NPU — only one process can open it. This error at startup usually means another copy of the app is already running:

```bash
# Check for duplicate uvicorn / systemd service
ps aux | grep uvicorn
sudo systemctl status laser-detection

# Stop the systemd service when testing manually on another port
sudo systemctl stop laser-detection
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

If no duplicate process is running, reboot the Pi to release a stale device handle from a crashed process. The camera and annotation UI still work without Hailo; detection falls back to CPU when a `.pt` model is available.

Check status: `curl http://localhost:8100/health` → `detection.hailo.loaded` and `detection.hailo.last_error`.


### Windows (scp)

Requires [OpenSSH Client](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) (`scp`, `ssh`, and `tar`).

```powershell
# From project root — upload and extract on neonbeam-lens.richwerks.local
.\deploy\deploy.ps1

# Also restart the systemd service after deploy
.\deploy\deploy.ps1 -Restart

# Or via cmd wrapper
deploy\deploy.cmd restart
```

The script creates a tarball (excluding `.venv`, `data/`, `models/`, `__pycache__`, etc.), normalizes shell script line endings to LF, uploads with `scp`, and extracts on the remote host.

The script tests SSH **before** building the web UI or creating the archive, then retries transient SSH failures (timeouts, connection reset) with configurable timeouts in `deploy.config.json`.

**Hostname not resolving?** Prefer the full hostname (not bare `neonbeam-lens`):

```powershell
.\deploy\deploy.ps1 -RemoteHost "crichards999@neonbeam-lens.richwerks.local" -Restart
```

Avoid deploying by IP unless you have added the IP to `~/.ssh/known_hosts` — SSH keys are stored under the hostname, so `crichards999@192.168.1.120` often fails with *Host key verification failed* even when the hostname works.

Or create a local config file (not committed to git):

```powershell
copy deploy\deploy.config.example.json deploy\deploy.config.json
# Tune sshRetryCount / sshConnectTimeout if you see intermittent "Connection reset"
.\deploy\deploy.ps1 -Restart
```

**SSH `Connection reset` from Windows?** The Pi may be busy or sshd overloaded. Verify with `ssh crichards999@neonbeam-lens.richwerks.local "echo ok"`, reboot the Pi if needed, then re-run deploy.

### Linux / macOS

```bash
./deploy/deploy.sh
./deploy/deploy.sh --restart
```

## systemd Service

```bash
sudo cp deploy/laser-detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now laser-detection
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health |
| GET | `/api/v1/camera/settings` | Camera settings (`exposure_ms`, gain, mount height) |
| PUT | `/api/v1/camera/settings` | Update camera settings (`exposure_ms` in milliseconds; converted to µs internally) |
| GET | `/api/v1/camera/snapshot` | JPEG still |
| GET | `/api/v1/camera/stream` | MJPEG stream |
| POST | `/api/v1/calibration/apriltag` | Calibrate from AprilTag specs |
| GET | `/api/v1/calibration/status` | Calibration status |
| POST | `/api/v1/calibration/apriltag/preview` | Tag detection preview image |
| POST | `/api/v1/calibration/apriltag/generate-pdf` | Print-ready AprilTag sheet |
| POST | `/api/v1/detection/detect` | Run object detection |
| POST | `/api/v1/detection/segment` | Run instance segmentation |
| GET | `/api/v1/detection/debug-image` | Annotated debug JPEG |
| GET | `/api/v1/dataset/classes` | List annotation classes |
| PUT | `/api/v1/dataset/classes` | Update class list |
| POST | `/api/v1/dataset/capture` | Capture frame into dataset |
| GET | `/api/v1/dataset/images` | List captured images |
| GET | `/api/v1/dataset/images/{id}` | Image metadata + annotations |
| GET | `/api/v1/dataset/images/{id}/file` | JPEG file (`?variant=thumb\|preview\|full`) |
| GET | `/api/v1/dataset/stats` | Class counts and reviewed image stats |
| PUT | `/api/v1/dataset/images/{id}/annotations` | Save annotations |
| PATCH | `/api/v1/dataset/images/{id}` | Mark reviewed / notes |
| DELETE | `/api/v1/dataset/images/{id}` | Delete image |
| POST | `/api/v1/dataset/export` | Export detection + segmentation datasets |
| GET | `/api/v1/dataset/export/status` | Last export summary |

## Annotation Web UI

Open **`http://neonbeam-lens.richwerks.local:8000/annotate`** in a browser on the same network.

Built with **React + Vite** (`web/`). Production builds go to `web/dist/` and are served by FastAPI.

```bash
# Dev (API + Vite hot reload)
.\scripts\start.ps1          # or bash scripts/start.sh
cd web && npm install && npm run dev

# Production build (also run automatically by deploy/deploy.ps1)
cd web && npm ci && npm run build
```

1. **Capture tab** — live camera preview (stream active only on this tab); capture frames into the dataset
2. **Annotate tab** — draw **boxes** (most workpieces) or **polygons** (irregular parts); assign classes; mark reviewed
3. **Export tab** — one click exports **both** YOLO detection and segmentation datasets (boxes auto-convert to 4-corner polygons for seg)

See [web/README.md](web/README.md) for keyboard shortcuts, dev proxy, and performance notes.

1. `POST /api/v1/calibration/apriltag/generate-pdf` with `size_mm` and `safe_zone_padding_mm`
2. Print at **100% scale** (Actual size — disable "Fit to page")
3. Verify tag edge length with a ruler (±0.5 mm)
4. Cut along dashed borders; align tag **center** to each bed corner

## Calibration Example

Bed coordinates default to **origin bottom-left**, **+x right**, **+y up** (`bed.origin: bottom_left`, `bed.y_axis: up` in `config/default.yaml`). Override for other machines if needed.

| Tag ID | Bed position | Corner |
|--------|--------------|--------|
| 0 | `(0, 0)` | bottom-left |
| 1 | `(400, 0)` | bottom-right |
| 2 | `(0, 400)` | top-left |
| 3 | `(400, 400)` | top-right |

`x_mm` / `y_mm` are the **tag center** (not a corner). `size_mm` is the **printed black square edge** — measure with calipers; do not use the dashed cut-line outer dimension.

Place all tags with the **same orientation as the generated PDF** (use optional `rotation_deg` per tag if rotated).

```json
POST /api/v1/calibration/apriltag
{
  "tags": [
    {"id": 0, "x_mm": 0, "y_mm": 0, "size_mm": 30},
    {"id": 1, "x_mm": 400, "y_mm": 0, "size_mm": 30},
    {"id": 2, "x_mm": 0, "y_mm": 400, "size_mm": 30},
    {"id": 3, "x_mm": 400, "y_mm": 400, "size_mm": 30}
  ]
}
```

Use `POST /api/v1/calibration/apriltag/preview` to verify detections; corner indices `0–3` are drawn on each tag. On failure, the API returns structured JSON with center vs corner error breakdown.

Automated tests in `tests/test_calibration.py` use synthetic tag geometry only (mock detector, no camera). After deploying to the Pi, re-run calibration there to validate with real lens distortion.

### Lens distortion (wide-angle cameras)

The CSI camera defaults to **102° horizontal FOV** (`camera.hfov_deg` in `config/default.yaml`). Wide-angle lenses show barrel distortion — tag **centers** may fit a homography while **corners** do not. When `camera.auto_distortion: true` (default), calibration auto-estimates pinhole distortion coefficients `k1`/`k2` from all 16 tag corners, undistorts image points before fitting homography, and stores intrinsics in `config/calibration.json`.

Successful calibration responses and `GET /api/v1/calibration/status` include a `distortion` block (`fx`, `fy`, `cx`, `cy`, `k1`, `k2`, `hfov_deg`).

| Error field | Meaning |
|-------------|---------|
| `center_error_mm` | Mean residual on tag centers (global homography) |
| `corner_error_mm` | Mean residual on all 16 corners (global homography) |
| `per_tag_errors_mm` | Mean corner residual per tag ID |

If `center_error_mm` is low but `corner_error_mm` is high (~30 mm), lens distortion is the likely cause — ensure `hfov_deg` matches your module and re-run calibration. Optional manual override:

```yaml
camera:
  hfov_deg: 102
  auto_distortion: false
  intrinsics_override:
    dist: [-0.25, 0.05, 0, 0, 0]   # k1, k2, p1, p2, k3
```

## Custom Model Training

See [training/compile_hef.md](training/compile_hef.md) for HEF compilation after training.

### 1. Collect and label data

Use the [annotation UI](http://neonbeam-lens.richwerks.local:8000/annotate) or capture via API:

```bash
curl -X POST http://neonbeam-lens.richwerks.local:8000/api/v1/dataset/capture
```

Define classes in the UI, annotate, mark images reviewed, then **Export datasets**.

### 2. Train detection model

```bash
python training/train.py --data data/dataset/export/detection/dataset.yaml
```

### 3. Train segmentation model (optional)

```bash
python training/train.py --data data/dataset/export/segmentation/dataset.yaml --model yolov8n-seg.pt
```

### 4. Deploy to Pi

```bash
python training/export_onnx.py models/best.pt
# Compile to .hef on x86 — see training/compile_hef.md
# Copy models/detection.hef to neonbeam-lens.richwerks.local
```

## Configuration

Edit [config/default.yaml](config/default.yaml) for bed size, camera height, model paths, and detection thresholds.

## Development (Windows/x86)

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Without picamera2/Hailo hardware, the service uses a mock camera and CPU detection when a model is available.

Run the API with `.\scripts\start.ps1`, then `cd web && npm run dev` for the annotation UI at `http://localhost:5173/annotate/`, or build with `npm run build` and open `http://localhost:8000/annotate`.

# Laser Engraver Object Detection System

FastAPI service for a Raspberry Pi 5 laser engraver with CSI camera and Hailo-8L NPU. Provides camera control, AprilTag bed calibration, object detection/segmentation, and debug visualization.

**Deployment target:** `crichards999@neonbeam-lens:~/object-detection-v2`

## Features

- Camera settings (exposure, gain, mount height), snapshot, and MJPEG stream
- AprilTag homography calibration with mm coordinate mapping
- AprilTag PDF generator (IDs 0–3, to-scale on US Letter)
- YOLOv8 object detection with Hailo NPU (production) and Ultralytics CPU fallback (dev)
- Parallax compensation for object height above the bed
- Debug image overlay (tags, grid, detections)
- Custom model training pipeline scaffold

## Quick Start (neonbeam-lens)

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

Open `http://neonbeam-lens:8000/docs` for interactive API documentation.

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

## Deploy Updates (from dev machine)

### Windows (scp)

Requires [OpenSSH Client](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) (`scp`, `ssh`, and `tar`).

```powershell
# From project root — upload and extract on neonbeam-lens
.\deploy\deploy.ps1

# Also restart the systemd service after deploy
.\deploy\deploy.ps1 -Restart

# Or via cmd wrapper
deploy\deploy.cmd restart
```

The script creates a tarball (excluding `.venv`, `data/`, `models/`, `__pycache__`, etc.), normalizes shell script line endings to LF, uploads with `scp`, and extracts on the remote host.

**Hostname not resolving?** Windows may not resolve `neonbeam-lens` via mDNS. Use the Pi's IP address instead:

```powershell
.\deploy\deploy.ps1 -RemoteHost "crichards999@192.168.1.50" -Restart
```

Or create a local config file (not committed to git):

```powershell
copy deploy\deploy.config.example.json deploy\deploy.config.json
# Edit deploy.config.json and set "remoteHost" to crichards999@<pi-ip-address>
.\deploy\deploy.ps1 -Restart
```

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
| GET | `/api/v1/camera/settings` | Camera settings |
| PUT | `/api/v1/camera/settings` | Update exposure/gain/mount height |
| GET | `/api/v1/camera/snapshot` | JPEG still |
| GET | `/api/v1/camera/stream` | MJPEG stream |
| POST | `/api/v1/calibration/apriltag` | Calibrate from AprilTag specs |
| GET | `/api/v1/calibration/status` | Calibration status |
| POST | `/api/v1/calibration/apriltag/preview` | Tag detection preview image |
| POST | `/api/v1/calibration/apriltag/generate-pdf` | Print-ready AprilTag sheet |
| POST | `/api/v1/detection/detect` | Run object detection |
| POST | `/api/v1/detection/segment` | Run instance segmentation |
| GET | `/api/v1/detection/debug-image` | Annotated debug JPEG |

## AprilTag PDF Printing

1. `POST /api/v1/calibration/apriltag/generate-pdf` with `size_mm` and `safe_zone_padding_mm`
2. Print at **100% scale** (Actual size — disable "Fit to page")
3. Verify tag edge length with a ruler (±0.5 mm)
4. Cut along dashed borders; align tag **center** to each bed corner

## Calibration Example

```json
POST /api/v1/calibration/apriltag
{
  "tags": [
    {"id": 0, "x_mm": 0, "y_mm": 0, "size_mm": 20},
    {"id": 1, "x_mm": 400, "y_mm": 0, "size_mm": 20},
    {"id": 2, "x_mm": 0, "y_mm": 400, "size_mm": 20},
    {"id": 3, "x_mm": 400, "y_mm": 400, "size_mm": 20}
  ]
}
```

## Custom Model Training

See [training/compile_hef.md](training/compile_hef.md) for the full workflow:

1. Collect images via `/api/v1/camera/snapshot`
2. Label objects (future: `/annotate` web UI)
3. `python training/train.py`
4. `python training/export_onnx.py models/best.pt`
5. Compile to `.hef` on x86 with Hailo DFC
6. Copy to `models/detection.hef` on neonbeam-lens

## Configuration

Edit [config/default.yaml](config/default.yaml) for bed size, camera height, model paths, and detection thresholds.

## Development (Windows/x86)

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Without picamera2/Hailo hardware, the service uses a mock camera and CPU detection when a model is available.

## Future: Annotation Web UI

Phase 6 will add a browser-based labeling tool at `/annotate`. See [web/README.md](web/README.md).

# Laser Engraver Object Detection System

FastAPI service for a Raspberry Pi 5 laser engraver with CSI camera and Hailo-8L NPU. Provides camera control, AprilTag bed calibration, object detection/segmentation, and debug visualization.

**Deployment target:** `crichards999@neonbeam-lens.richwerks.local:~/object-detection-v2`

## Features

- Camera settings (exposure, gain, mount height), snapshot, and MJPEG stream
- AprilTag homography calibration with mm coordinate mapping
- AprilTag PDF generator (IDs 0–3, to-scale on US Letter)
- FastSAM instance segmentation with Hailo NPU (production) and Ultralytics CPU fallback (dev)
- Bed-calibrated shape detection (mm geometry, rotation, segmentation polygons)
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
pip install --upgrade pip
pip install torch torchvision --prefer-binary --extra-index-url https://www.piwheels.org/simple
pip install -r requirements-pi.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Note:** `ultralytics` depends on PyTorch. On Pi (`aarch64`), a plain `pip install ultralytics` can pull **NVIDIA CUDA** wheels (~1GB, useless without an NVIDIA GPU). `setup-pi.sh` installs **CPU-only** torch from [piwheels](https://www.piwheels.org/) first.

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

**First check: is the AI HAT installed?** Error 74 with `found: 0` is normal when no Hailo hardware is connected:

```bash
hailortcli fw-control identify
ls -l /dev/hailo*
```

If the HAT is not installed, Hailo FastSAM cannot load — set `detection.fastsam_device: cpu` for Ultralytics **FastSAM-s.pt** on the CPU. Install the HAT on the Pi 5 PCIe connector, reboot, and re-run `hailortcli fw-control identify`.

`/health` → `npu.hardware.present` should be `true` when the device is detected.

When the HAT **is** installed, this app uses **direct VDevice access** (picamera2-style). **`hailort.service` must be stopped**:

```bash
sudo systemctl stop hailort.service
sudo systemctl disable hailort.service
```

Duplicate **app instances** also cause `found: 0`:

```bash
sudo systemctl stop laser-detection
bash scripts/start.sh --stop-service
```

Check status: `curl http://localhost:8100/health` → `npu.hardware`, `npu.hailort_service`, `npu.models.fastsam`.


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

`exposure_ms` is the configured/saved value. `exposure_ms_actual` (when present) is what the camera hardware applied — they should match after the frame-duration fix below. Long exposures (> ~66 ms) automatically lower the capture frame rate so libcamera is not capped by a fixed 15 fps window.
| GET | `/api/v1/camera/snapshot` | JPEG still |
| GET | `/api/v1/camera/stream` | MJPEG stream (`?size=main` 16:9 preview default; `?size=lores` square 640×640 for ML) |
| POST | `/api/v1/calibration/apriltag` | Calibrate from AprilTag specs |
| GET | `/api/v1/calibration/status` | Calibration status |
| POST | `/api/v1/calibration/apriltag/preview` | Tag detection preview image |
| GET | `/api/v1/calibration/apriltag/debug-image` | Calibrated grid + tags overlay with side lengths (no object detection) |
| POST | `/api/v1/calibration/apriltag/generate-pdf` | Print-ready AprilTag sheet |
| GET | `/api/v1/detection/detect` | FastSAM detection (mm geometry + segmentation; query params) |
| POST | `/api/v1/detection/capture-background` | Save empty-bed background reference (requires calibration) |
| GET | `/api/v1/detection/background/status` | Background reference present / stale status |
| DELETE | `/api/v1/detection/background` | Clear stored background reference |
| GET | `/api/v1/detection/work-area-image` | Rectified bed JPEG (AprilTags at corners) |
| GET | `/api/v1/detection/work-area-image/info` | Work-area image scale and mm→px mapping |
| GET | `/api/v1/detection/debug-image` | Pipeline stage JPEG (`stage=all` for tiled mosaic) |
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

Mount AprilTags at the **four corners** of the work area with tag **centers** on the corners. The origin tag center becomes bed `(0, 0)`. Bed coordinates use **origin bottom-left**, **+x right**, **+y up** (`bed.origin: bottom_left`, `bed.y_axis: up` in `config/default.yaml`).

| Tag ID | Typical corner |
|--------|----------------|
| 0 | bottom-left (origin) |
| 1 | bottom-right |
| 2 | top-left |
| 3 | top-right |

`size_mm` is the **printed black square edge** — measure with calipers; do not use the dashed cut-line outer dimension. Neighbor tags (BR, TL, TR) are identified automatically from geometry; IDs need not follow this table if `origin_tag_id` is set correctly.

```json
POST /api/v1/calibration/apriltag
{
  "origin_tag_id": 0,
  "size_mm": 30,
  "tag_ids": [0, 1, 2, 3]
}
```

The response includes a `work_area` block with derived `width_mm` and `height_mm`, plus `tag_size_validation` showing per-tag measured edge lengths after scale refinement. These are saved in `config/calibration.json` and drive the debug grid overlay.

`size_mm` is treated as ground truth. Calibration uses **per-axis scale** from horizontal vs vertical tag edges (important at wide FOV) and iterates width/height independently until tag edges measure the declared size on both axes (within `calibration.scale_refinement_tolerance_mm`, default 0.5 mm). Work-area width/height are re-derived from the final homography before save. If refinement cannot fully converge, calibration still saves but the response `message` includes a warning — check print quality, tag flatness, and detection.

The `tag_size_validation` block includes `mm_per_px_x`, `mm_per_px_y`, `mean_horizontal_mm`, `mean_vertical_mm`, and per-axis iteration counts.

Use `GET /api/v1/calibration/apriltag/debug-image` to overlay per-tag measured sizes (`30 mm (meas 29.8)`) and work-area side lengths.

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
| `tag_size_validation` | Per-tag measured edge length vs `size_mm`; horizontal/vertical means; `mm_per_px_x`/`y`; `converged` |

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

## Detection (FastSAM)

Requires calibration with a derived `work_area`. Detects flat geometric workpieces (coasters, cards, bracelets) on a **white painted bed with dark/black parts**.

`GET /api/v1/detection/detect` returns per-object geometry in **bed millimeters**: `bbox_mm`, `width_mm`, `height_mm`, `rotation_deg`, `oriented_box_mm`, `segmentation_polygon_mm`. Query params:

| Param | Default | Purpose |
|-------|---------|---------|
| `min_confidence` | config `detection.min_confidence` | Drop low-confidence detections |
| `include_work_area_coords` | `false` | When `true`, also include `segmentation_polygon_work_area_px` and `oriented_box_work_area_px` on the warped work-area image (for UI overlays) |
| `use_background_reference` | config `detection.use_background_reference` | Use stored empty-bed reference for bg_subtract post-filter |

Response includes `fastsam_used`, `fastsam_device`, `fastsam_filter_detail`, `background_reference_used`, and `fastsam_error` when FastSAM fails. Responses use `Cache-Control: no-store`.

### Pipeline architecture

```
warp → bg_subtract (foreground hint) → FastSAM (Hailo or CPU)
     → post-filter masks (min area + bg overlap) → geometry → DetectionResponse
```

The bg post-filter drops corner-tag speckles and masks outside bg_subtract foreground. If bg_subtract covers more than ~45% of the frame (stale reference or overexposure), the filter auto-disables and raw FastSAM masks are used.

`GET /api/v1/detection/work-area-image` returns a rectified JPEG: work area warped so AprilTag corners align to the image edges (origin tag at bottom-left). Companion `GET .../work-area-image/info` documents `pixels_per_mm` and mm→px mapping (`y_px = (height_mm - y_mm) * pixels_per_mm`).

`GET /api/v1/detection/debug-image?stage=...` exposes pipeline stages. Valid `stage` values match the tiles below plus `all` for a mosaic.

For `stage=all`, the mosaic includes only stages active on **this run** (minimum: `raw → warp → fastsam → final`). Optional tiles:

- **bg_diff** / **bg_subtract** — when a stored background reference exists and is used
- **fastsam_filtered** — when FastSAM returns masks and filtering keeps at least one

The **bg_diff** tile is grayscale `|current − reference|` on the warped work area (black = no change; bright = pixel difference). Recapture the empty-bed reference if an empty bed is not mostly black.

**fastsam** shows raw FastSAM masks; **fastsam_filtered** shows masks after bg overlap and minimum-area filtering; **final** shows detected geometry with mm labels. Set `show_center_coords=true` on the debug-image request to draw a crosshair and `center x, y mm` label on each object in the **final** tile (bed coordinates, bottom-left origin, Y-up).

### Hailo bring-up (Pi deploy)

Before testing on device:

```bash
hailortcli fw-control identify          # should show device
sudo systemctl stop hailort.service     # required for direct VDevice access
curl localhost:8000/health | jq '.npu, .detection.fastsam'
```

Expect `npu.hardware.present: true`, `detection.fastsam.device: hailo`, `loaded: true`. Default config sets `fastsam_device: hailo`. The Hailo `fast_sam_s.hef` returns raw YOLOv8-seg tensors — the service decodes them on the CPU (`fastsam_hailo_score_threshold`, `fastsam_hailo_nms_iou`, `fastsam_hailo_mask_threshold` in config). Without a HAT, set `fastsam_device: cpu` to use Ultralytics **FastSAM-s.pt** (~3–8 s/frame on Pi 5 at `fastsam_cpu_imgsz: 640`).

**FastSAM debug logs** — each detection run emits `[fastsam]` lines to stdout (INFO level): frame sizes, raw Hailo output tensor shapes, max class score, pre/post-NMS boxes, and per-mask area/bbox. On the Pi:

```bash
journalctl -u laser-detection -f | grep fastsam
```

Or watch uvicorn stdout if running manually. Use these lines to see whether real objects score below threshold or NMS is keeping only corner speckles.

### Empty-bed background reference

After repainting the bed white, **recapture** an empty-bed reference — a stale dark-bed reference will poison `bg_subtract`:

1. Clear the bed, then either `POST /api/v1/calibration/apriltag` with `"capture_empty_background": true`, or `POST /api/v1/detection/capture-background`.
2. Place dark workpieces and run `GET /api/v1/detection/detect` or `GET .../debug-image?stage=all`.

bg_subtract subtracts the stored warp against the current frame. Recapture when lighting, bed surface, camera position, or calibration scale changes — `GET .../background/status` reports `stale_reason` when metadata no longer matches.

Disable background subtraction globally with `use_background_reference: false` in config, or per request:

```bash
curl ".../detection/detect?use_background_reference=false"
```

`GET .../debug-image?use_background_reference=false` accepts the same query parameter.

### Exposure and glare on white paint

White bed paint clips easily. Target **20–40 ms** exposure for preview and detection (`PUT /api/v1/camera/settings` with `exposure_ms`). Long exposures blow out the centre and produce false foreground in debug tiles.

### Detection tuning (`detection:`)

| Key | Purpose |
|-----|---------|
| `bg_subtract_min_diff` / `bg_subtract_blur_kernel_px` | Diff floor and blur before threshold (bg post-filter) |
| `fastsam_bg_filter_enabled` | Post-filter FastSAM masks by bg_subtract overlap |
| `fastsam_bg_filter_min_overlap` | Min fraction of mask pixels inside bg foreground (default 0.25) |
| `fastsam_bg_filter_max_fg_ratio` | Skip bg filter when foreground exceeds this fraction of frame |
| `fastsam_min_mask_area_px` | Drop tiny FastSAM speckle masks (default 800 px) |
| `mask_morph_kernel_px` | Morphological close kernel for mask cleanup |
| `mask_min_component_area_mm2` | Drop speckle blobs below this area before contouring |
| `mask_max_components` | Scoring penalty when too many mask fragments |
| `mask_max_component_area_ratio` | Reject masks where one blob covers too much of the bed (glare bridges) |
| `mask_bridge_break_kernel_px` | Morphological open to break thin glare connections |
| `morph_close_iterations` | Merge fragmented masks |
| `glare_rejection_enabled` | Reject bright specular hot-spots |
| `glare_l_delta` / `glare_l_absolute_min` | LAB L-channel thresholds relative to bed border |
| `glare_suppression_l_cap` | Suppress weak diff speckle on overexposed paint in bg_subtract |
| `fastsam_device` | `hailo` (default), `cpu`, or `auto` |
| `fastsam_cpu_imgsz` | Ultralytics inference size (lower = faster on Pi) |
| `use_background_reference` | Enable bg_subtract when a stored reference exists (overridable per request) |
| `background_storage_path` | PNG + JSON metadata path for empty-bed capture |

Diffuse or angle overhead lighting to reduce glare on white paint.

### Migration from `/shapes`

Replace `POST /api/v1/detection/shapes` with:

```bash
curl ".../detection/detect?min_confidence=0.35"
curl ".../detection/detect?include_work_area_coords=true"
```

Background capture: `POST .../detection/capture-background`  
Debug mosaic: `GET .../detection/debug-image?stage=all`

## Work surface prep

**White painted bed + dark/black workpieces** is the default detection scenario. Recapture the empty-bed background after any repaint or surface change.

- Target matte white paint; avoid glossy finish that creates specular hotspots
- **Angled LED bar**: rim-light edges help segmentation find part outlines
- Keep exposure short (20–40 ms) to avoid clipping white paint
- AprilTags at corners define the ROI; optional tape border inside tags for clipping only
- **Do not use a checkerboard pattern** on the bed — it creates false contours

## Configuration

Edit [config/default.yaml](config/default.yaml) for bed coordinate frame (`origin`, `y_axis`), camera height, model paths, and detection thresholds. Work area size is not configured — it is derived from AprilTag corner placement during calibration.

## Development (Windows/x86)

```bash
pip install -r requirements-dev.txt
pytest
uvicorn app.main:app --reload
```

Without picamera2/Hailo hardware, the service uses a mock camera. FastSAM CPU inference works when Ultralytics and a `.pt` model are available.

Run the API with `.\scripts\start.ps1`, then `cd web && npm run dev` for the annotation UI at `http://localhost:5173/annotate/`, or build with `npm run build` and open `http://localhost:8000/annotate`.

# Annotation Web UI (Future — Phase 6)

Static frontend served at `/annotate` when deployed.

Planned modules under `src/`:

- `camera.js` — embed MJPEG stream from `/api/v1/camera/stream`
- `canvas.js` — click-drag bounding box editor
- `classes.js` — class list management via future dataset API
- `dataset.js` — capture, save annotations, export YOLO format

No build step required for the MVP; vanilla JS + HTML5 Canvas.

# Annotation Web UI (React + Vite)

Browser-based tool for capturing training images and annotating objects on the laser bed.

**URL:** `http://neonbeam-lens.richwerks.local:8000/annotate`

## Workflow

1. **Capture** — live MJPEG preview (only while Capture tab is active); capture frames into the dataset
2. **Annotate** — dual-layer canvas with box/polygon tools; preview-sized images for speed
3. **Export** — generate both YOLO detection and segmentation datasets in one action

## Development

Requires Node.js 18+ on your dev machine.

```bash
cd web
npm install
npm run dev
```

Vite dev server runs on `http://localhost:5173/annotate/` and proxies `/api` and `/health` to the FastAPI backend (default `http://localhost:8100` — edit `vite.config.ts` if needed).

Start the API in another terminal:

```powershell
# Windows
.\scripts\start.ps1

# Linux / Pi
bash scripts/start.sh
```

## Production build

```bash
cd web
npm ci
npm run build
```

Output goes to `web/dist/`. FastAPI serves this folder at `/annotate`. The deploy script (`deploy/deploy.ps1`) runs `npm run build` automatically before packaging.

## Tools

| Tool | Shortcut | Use |
|------|----------|-----|
| Box | `B` | Rectangular workpieces |
| Polygon | `P` | Irregular shapes; double-click or Enter to finish |
| Delete | `Del` | Remove selected annotation |
| Undo | `Ctrl+Z` | Undo last annotation change |
| Class | `1-9` | Select active class |

Keyboard shortcuts are **desktop only**. On mobile, use the bottom toolbar and sheet instead.

## Mobile annotation

On viewports **768px wide or narrower**, the app switches to a mobile layout:

- **Bottom tab bar** — Capture, Annotate, and Export stay thumb-reachable with safe-area padding.
- **Canvas first** — The annotate view fills the screen; no scrolling past a sidebar to reach the image.
- **Bottom toolbar** — Box/Polygon toggle, active class chip, prev/next image, undo/delete when applicable, **Finish polygon** and **Undo last point** while drawing, **Reset view** when zoomed, and **Menu** to open the sheet.
- **Bottom sheet** — Image list, classes (add/select), annotation list, reviewed checkbox, save/delete image.

### Touch interactions

- **Box** — Drag to draw (same as desktop mouse drag).
- **Polygon** — Tap to add vertices; tap **Finish** in the toolbar (or use Enter / double-click on desktop).
- **Select / move** — Tap an annotation or drag a vertex; hit targets are enlarged for touch.
- **Pinch-to-zoom** — Two-finger pinch zooms 1×–5× around the pinch center; annotations stay aligned with the image.
- **Pan when zoomed** — Two-finger drag, or one-finger drag on empty canvas when zoomed in.
- **Reset view** — Toolbar button returns to fit-to-screen (1×, centered).

Images are always **letterboxed** (`object-fit: contain`); zoom scales uniformly and never stretches the aspect ratio.

## Stack

- React 19 + TypeScript + Vite
- TanStack Query for API caching
- Dual-layer HTML5 Canvas (background image + annotation overlay)
- REST client in `src/api/client.ts` → `/api/v1/dataset/*` and `/api/v1/camera/*`

## Performance notes

- MJPEG preview uses `<img>` (not canvas) at **640px lores** via `/api/v1/camera/stream?size=lores`
- Stream unmounts when leaving the Capture tab (reduces camera lock contention)
- Annotation uses a static `<img>` (preview JPEG) plus a single overlay canvas for shapes
- Dragging annotations updates the overlay only; React state commits on mouseup
- Inactive tabs unmount entirely (no hidden canvas or listeners)
- Thumbnails and previews are cached on disk at capture time

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from app.config import get_config_store
from app.schemas.detection import (
    CaptureBackgroundResponse,
    DebugStage,
    WorkAreaBackgroundStatusResponse,
    WorkAreaImageInfo,
)
from app.services.camera_service import CameraService, get_camera_service
from app.services.calibration_service import get_calibration_service
from app.services.shape_pipeline import get_shape_pipeline
from app.services.work_area_background import get_work_area_background_store
from app.services.work_area_renderer import get_work_area_renderer

router = APIRouter(prefix="/api/v1/detection", tags=["detection"])


def _require_calibration() -> None:
    if not get_calibration_service().is_calibrated():
        raise HTTPException(status_code=409, detail="Calibration with work area is required")
    data = get_calibration_service().data
    if data is None or data.work_area is None:
        raise HTTPException(status_code=409, detail="Calibration work area is required")


@router.get("/detect")
def run_detection(
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    include_work_area_coords: bool = Query(default=False),
    use_background_reference: bool | None = Query(
        default=None,
        description="Use stored empty-bed reference for bg_subtract filter; defaults to detection.use_background_reference in config",
    ),
    camera: CameraService = Depends(get_camera_service),
) -> JSONResponse:
    _require_calibration()
    frame = camera.capture_frame()
    result = get_shape_pipeline().run(
        frame,
        min_confidence=min_confidence,
        include_work_area_coords=include_work_area_coords,
        use_background_reference=use_background_reference,
    )
    return JSONResponse(
        content=result.response.model_dump(),
        headers={"Cache-Control": "no-store"},
    )


@router.post("/capture-background", response_model=CaptureBackgroundResponse)
def capture_background(
    camera: CameraService = Depends(get_camera_service),
) -> CaptureBackgroundResponse:
    _require_calibration()
    cfg = get_config_store().config.detection
    frame = camera.capture_frame()
    view = get_work_area_renderer().render(frame, max_edge_px=cfg.max_edge_px)
    metadata = get_work_area_background_store().save(view, max_edge_px=cfg.max_edge_px)
    return CaptureBackgroundResponse(
        captured=True,
        timestamp=metadata.timestamp,
        pixels_per_mm=metadata.pixels_per_mm,
        width_px=metadata.width_px,
        height_px=metadata.height_px,
    )


@router.get("/background/status", response_model=WorkAreaBackgroundStatusResponse)
def background_status(
    pixels_per_mm: float | None = Query(default=None, gt=0),
    max_edge_px: int | None = Query(default=None, ge=64, le=4096),
) -> WorkAreaBackgroundStatusResponse:
    _require_calibration()
    cfg = get_config_store().config.detection
    max_edge = max_edge_px if max_edge_px is not None else cfg.max_edge_px
    width_mm, height_mm, origin_tag_id = get_work_area_renderer().require_work_area()
    from app.services.work_area_renderer import WorkAreaView, resolve_pixels_per_mm

    ppm = resolve_pixels_per_mm(
        width_mm,
        height_mm,
        pixels_per_mm=pixels_per_mm,
        max_edge_px=max_edge,
    )
    width_px = max(1, int(round(width_mm * ppm)))
    height_px = max(1, int(round(height_mm * ppm)))
    view = WorkAreaView(
        image=np.zeros((height_px, width_px, 3), dtype=np.uint8),
        width_mm=width_mm,
        height_mm=height_mm,
        width_px=width_px,
        height_px=height_px,
        pixels_per_mm=ppm,
        origin_tag_id=origin_tag_id,
    )
    status = get_work_area_background_store().get_status(view, max_edge_px=max_edge)
    return WorkAreaBackgroundStatusResponse(
        present=status.present,
        timestamp=status.timestamp,
        pixels_per_mm=status.pixels_per_mm,
        width_px=status.width_px,
        height_px=status.height_px,
        stale_reason=status.stale_reason,
    )


@router.delete("/background", status_code=204)
def delete_background() -> Response:
    get_work_area_background_store().clear()
    return Response(status_code=204)


@router.get("/work-area-image")
def work_area_image(
    pixels_per_mm: float | None = Query(default=None, gt=0),
    max_edge_px: int = Query(default=1024, ge=64, le=4096),
    quality: int = Query(default=85, ge=1, le=100),
    camera: CameraService = Depends(get_camera_service),
) -> Response:
    _require_calibration()
    frame = camera.capture_frame()
    renderer = get_work_area_renderer()
    view = renderer.render(frame, pixels_per_mm=pixels_per_mm, max_edge_px=max_edge_px)
    jpeg = renderer.encode_jpeg(view, quality=quality)
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/work-area-image/info", response_model=WorkAreaImageInfo)
def work_area_image_info(
    pixels_per_mm: float | None = Query(default=None, gt=0),
    max_edge_px: int = Query(default=1024, ge=64, le=4096),
) -> WorkAreaImageInfo:
    _require_calibration()
    renderer = get_work_area_renderer()
    width_mm, height_mm, origin_tag_id = renderer.require_work_area()
    from app.services.work_area_renderer import resolve_pixels_per_mm

    ppm = resolve_pixels_per_mm(
        width_mm,
        height_mm,
        pixels_per_mm=pixels_per_mm,
        max_edge_px=max_edge_px,
    )
    width_px = max(1, int(round(width_mm * ppm)))
    height_px = max(1, int(round(height_mm * ppm)))
    return WorkAreaImageInfo(
        width_mm=width_mm,
        height_mm=height_mm,
        width_px=width_px,
        height_px=height_px,
        pixels_per_mm=ppm,
        origin_tag_id=origin_tag_id,
    )


@router.get("/debug-image")
def debug_image(
    stage: DebugStage = Query(
        default="final",
        description=(
            "Pipeline stage JPEG. Use `all` for a tiled mosaic of every stage active on this run "
            "(typically raw → warp → fastsam → final; adds bg_diff/texture_diff/bg_subtract when a background "
            "reference is stored and used; adds fastsam_filtered when FastSAM returns masks)."
        ),
    ),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    use_background_reference: bool | None = Query(
        default=None,
        description="Use stored empty-bed reference for bg_subtract filter; defaults to detection.use_background_reference in config",
    ),
    pixels_per_mm: float | None = Query(default=None, gt=0),
    max_edge_px: int = Query(default=1024, ge=64, le=4096),
    max_width_px: int = Query(default=1920, ge=320, le=4096),
    max_height_px: int = Query(default=1080, ge=240, le=4096),
    columns: int = Query(default=3, ge=1, le=6),
    quality: int = Query(default=85, ge=1, le=100),
    show_center_coords: bool = Query(
        default=False,
        description=(
            "On the final stage tile, mark each detection center with a crosshair and "
            "label bed coordinates in mm (bottom-left origin, Y-up — same frame as GET /detect)."
        ),
    ),
    camera: CameraService = Depends(get_camera_service),
) -> Response:
    _require_calibration()
    frame = camera.capture_frame()
    pipeline = get_shape_pipeline()
    result = pipeline.run(
        frame,
        min_confidence=min_confidence,
        include_work_area_coords=True,
        pixels_per_mm=pixels_per_mm,
        max_edge_px=max_edge_px,
        use_background_reference=use_background_reference,
        show_center_coords=show_center_coords,
    )
    try:
        jpeg = pipeline.render_debug_stage(
            result,
            stage,
            max_width_px=max_width_px,
            max_height_px=max_height_px,
            columns=columns,
            quality=quality,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=jpeg, media_type="image/jpeg")

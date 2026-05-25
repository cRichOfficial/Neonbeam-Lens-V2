from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.config import get_config_store
from app.schemas.detection import DetectionRequest, DetectionResponse, SegmentationRequest
from app.schemas.shapes import DebugStage, ShapesRequest, ShapesResponse, WorkAreaImageInfo
from app.services.camera_service import CameraService, get_camera_service
from app.services.calibration_service import get_calibration_service
from app.services.debug_renderer import get_debug_renderer, raw_to_detection_items
from app.services.detector_factory import get_detector
from app.services.shape_pipeline import get_shape_pipeline
from app.services.work_area_renderer import get_work_area_renderer

router = APIRouter(prefix="/api/v1/detection", tags=["detection"])


def _resolve_object_height(request_height: float | None) -> float:
    config = get_config_store().config
    if request_height is not None:
        return request_height
    return config.parallax.default_object_height_mm


def _resolve_confidence(request_confidence: float | None) -> float:
    config = get_config_store().config
    if request_confidence is not None:
        return request_confidence
    return config.detection.confidence_threshold


@router.post("/detect", response_model=DetectionResponse)
def run_detection(
    payload: DetectionRequest = DetectionRequest(),
    camera: CameraService = Depends(get_camera_service),
) -> DetectionResponse:
    frame = camera.capture_lores_frame()
    confidence = _resolve_confidence(payload.confidence_threshold)
    object_height_mm = _resolve_object_height(payload.object_height_mm)

    detector = get_detector()
    raw = detector.detect(frame, confidence)
    items = raw_to_detection_items(raw, object_height_mm=object_height_mm)

    return DetectionResponse(
        backend=detector.name,
        count=len(items),
        detections=items,
        calibrated=get_calibration_service().is_calibrated(),
        object_height_mm=object_height_mm,
    )


@router.post("/segment", response_model=DetectionResponse)
def run_segmentation(
    payload: SegmentationRequest = SegmentationRequest(),
    camera: CameraService = Depends(get_camera_service),
) -> DetectionResponse:
    frame = camera.capture_lores_frame()
    confidence = _resolve_confidence(payload.confidence_threshold)
    object_height_mm = _resolve_object_height(payload.object_height_mm)

    detector = get_detector()
    if hasattr(detector, "segment"):
        raw = detector.segment(frame, confidence)
    else:
        raw = detector.detect(frame, confidence)
    items = raw_to_detection_items(raw, object_height_mm=object_height_mm)

    return DetectionResponse(
        backend=detector.name,
        count=len(items),
        detections=items,
        calibrated=get_calibration_service().is_calibrated(),
        object_height_mm=object_height_mm,
    )


@router.get("/debug-image")
def debug_image(
    object_height_mm: float | None = Query(default=None, ge=0),
    confidence_threshold: float | None = Query(default=None, ge=0, le=1),
    camera: CameraService = Depends(get_camera_service),
) -> Response:
    frame = camera.capture_frame()
    confidence = _resolve_confidence(confidence_threshold)
    height_mm = _resolve_object_height(object_height_mm)

    detector = get_detector()
    raw = detector.detect(frame, confidence)
    items = raw_to_detection_items(raw, object_height_mm=height_mm)
    jpeg = get_debug_renderer().render(frame, detections=items)
    return Response(content=jpeg, media_type="image/jpeg")


def _require_calibration() -> None:
    if not get_calibration_service().is_calibrated():
        raise HTTPException(status_code=409, detail="Calibration with work area is required")
    data = get_calibration_service().data
    if data is None or data.work_area is None:
        raise HTTPException(status_code=409, detail="Calibration work area is required")


@router.post("/shapes", response_model=ShapesResponse)
def run_shapes(
    payload: ShapesRequest = ShapesRequest(),
    camera: CameraService = Depends(get_camera_service),
) -> ShapesResponse:
    _require_calibration()
    frame = camera.capture_frame()
    config = get_config_store().config.shapes
    backend = payload.backend or config.backend
    result = get_shape_pipeline().run(
        frame,
        backend=backend,
        min_confidence=payload.min_confidence,
        include_work_area_coords=payload.include_work_area_coords,
    )
    return result.response


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


@router.get("/shapes/debug-image")
def shapes_debug_image(
    stage: DebugStage = Query(default="final"),
    backend: str = Query(default="auto"),
    min_confidence: float | None = Query(default=None, ge=0, le=1),
    pixels_per_mm: float | None = Query(default=None, gt=0),
    max_edge_px: int = Query(default=1024, ge=64, le=4096),
    max_width_px: int = Query(default=1920, ge=320, le=4096),
    max_height_px: int = Query(default=1080, ge=240, le=4096),
    columns: int = Query(default=3, ge=1, le=6),
    quality: int = Query(default=85, ge=1, le=100),
    camera: CameraService = Depends(get_camera_service),
) -> Response:
    _require_calibration()
    frame = camera.capture_frame()
    config = get_config_store().config.shapes
    pipeline = get_shape_pipeline()
    result = pipeline.run(
        frame,
        backend=backend if backend in ("auto", "classical", "fastsam") else config.backend,
        min_confidence=min_confidence,
        include_work_area_coords=True,
        pixels_per_mm=pixels_per_mm,
        max_edge_px=max_edge_px,
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

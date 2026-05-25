from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.config import get_config_store
from app.schemas.detection import DetectionRequest, DetectionResponse, SegmentationRequest
from app.services.camera_service import CameraService, get_camera_service
from app.services.debug_renderer import get_debug_renderer, raw_to_detection_items
from app.services.detector_factory import get_detector
from app.services.calibration_service import get_calibration_service

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

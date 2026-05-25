from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.schemas.calibration import (
    AprilTagCalibrationRequest,
    AprilTagPdfRequest,
    CalibrationResult,
    CalibrationStatusResponse,
    DistortionSummary,
)
from app.services.apriltag_pdf_service import generate_apriltag_pdf
from app.services.apriltag_service import get_apriltag_service
from app.services.calibration_service import CalibrationError, CalibrationService, get_calibration_service
from app.services.camera_service import CameraService, get_camera_service

router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"])


@router.get("/status", response_model=CalibrationStatusResponse)
def calibration_status(
    calibration: CalibrationService = Depends(get_calibration_service),
) -> CalibrationStatusResponse:
    status = calibration.get_status()
    return CalibrationStatusResponse(**status)


@router.post("/apriltag", response_model=CalibrationResult)
def calibrate_apriltags(
    payload: AprilTagCalibrationRequest,
    camera: CameraService = Depends(get_camera_service),
    calibration: CalibrationService = Depends(get_calibration_service),
) -> CalibrationResult:
    frame = camera.capture_frame()
    try:
        result, matched = calibration.calibrate(frame, payload.tags, persist=True)
    except CalibrationError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    distortion = (
        DistortionSummary(**result.intrinsics.summary()) if result.intrinsics else None
    )
    return CalibrationResult(
        success=True,
        timestamp=result.timestamp,
        reprojection_error_mm=result.reprojection_error_mm,
        tags_detected=matched,
        message="Calibration saved",
        distortion=distortion,
    )


@router.post("/apriltag/preview")
def preview_apriltags(
    camera: CameraService = Depends(get_camera_service),
) -> Response:
    frame = camera.capture_frame()
    tags = get_apriltag_service().detect(frame)
    annotated = get_apriltag_service().draw_detections(frame, tags)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode preview image")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@router.post("/apriltag/generate-pdf")
def generate_apriltag_sheet(payload: AprilTagPdfRequest) -> Response:
    pdf_bytes = generate_apriltag_pdf(payload)
    filename = f'apriltags_0-3_{payload.size_mm:g}mm.pdf'
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

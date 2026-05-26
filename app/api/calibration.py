from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.config import get_config_store
from app.schemas.calibration import (
    AprilTagCalibrationRequest,
    AprilTagPdfRequest,
    CalibrationResult,
    CalibrationStatusResponse,
    DistortionSummary,
    TagSizeValidation,
    WorkAreaBackgroundSummary,
    WorkAreaSummary,
)
from app.services.apriltag_pdf_service import generate_apriltag_pdf
from app.services.apriltag_service import get_apriltag_service
from app.services.calibration_service import CalibrationError, CalibrationService, get_calibration_service
from app.services.camera_intrinsics import resolve_camera_intrinsics, undistort_points
from app.services.camera_service import CameraService, get_camera_service
from app.services.debug_renderer import get_debug_renderer
from app.services.work_area import derive_work_area_from_detections
from app.services.work_area_background import get_work_area_background_store
from app.services.work_area_renderer import get_work_area_renderer

router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"])


def _work_area_summary(result) -> WorkAreaSummary | None:
    if result.work_area is None:
        return None
    wa = result.work_area
    return WorkAreaSummary(
        width_mm=wa.width_mm,
        height_mm=wa.height_mm,
        origin_tag_id=wa.origin_tag_id,
        size_mm=wa.size_mm,
    )


def _tag_size_validation_summary(result) -> TagSizeValidation | None:
    if result.tag_size_validation is None:
        return None
    validation = result.tag_size_validation
    return TagSizeValidation(
        expected_mm=validation.expected_mm,
        measured_mm=validation.measured_mm,
        mean_mm=validation.mean_mm,
        max_error_mm=validation.max_error_mm,
        scale_iterations=validation.scale_iterations,
        converged=validation.converged,
        warning=validation.warning,
        mm_per_px_x=validation.mm_per_px_x,
        mm_per_px_y=validation.mm_per_px_y,
        mean_horizontal_mm=validation.mean_horizontal_mm,
        mean_vertical_mm=validation.mean_vertical_mm,
        scale_x_iterations=validation.scale_x_iterations,
        scale_y_iterations=validation.scale_y_iterations,
    )


@router.get("/status", response_model=CalibrationStatusResponse)
def calibration_status(
    calibration: CalibrationService = Depends(get_calibration_service),
) -> CalibrationStatusResponse:
    status = calibration.get_status()
    work_area = status.get("work_area")
    if work_area is not None:
        status = {
            **status,
            "work_area": WorkAreaSummary(
                width_mm=work_area["width_mm"],
                height_mm=work_area["height_mm"],
                origin_tag_id=work_area["origin_tag_id"],
                size_mm=work_area["size_mm"],
            ),
        }
    tag_size_validation = status.get("tag_size_validation")
    if tag_size_validation is not None:
        status = {
            **status,
            "tag_size_validation": TagSizeValidation(
                expected_mm=tag_size_validation["expected_mm"],
                measured_mm={int(k): float(v) for k, v in tag_size_validation["measured_mm"].items()},
                mean_mm=tag_size_validation["mean_mm"],
                max_error_mm=tag_size_validation["max_error_mm"],
                scale_iterations=tag_size_validation["scale_iterations"],
                converged=tag_size_validation["converged"],
                warning=tag_size_validation.get("warning"),
                mm_per_px_x=tag_size_validation.get("mm_per_px_x"),
                mm_per_px_y=tag_size_validation.get("mm_per_px_y"),
                mean_horizontal_mm=tag_size_validation.get("mean_horizontal_mm"),
                mean_vertical_mm=tag_size_validation.get("mean_vertical_mm"),
                scale_x_iterations=int(tag_size_validation.get("scale_x_iterations", 0)),
                scale_y_iterations=int(tag_size_validation.get("scale_y_iterations", 0)),
            ),
        }
    bg_status = get_work_area_background_store().get_status()
    status = {
        **status,
        "background_reference": WorkAreaBackgroundSummary(
            present=bg_status.present,
            timestamp=bg_status.timestamp,
            stale_reason=bg_status.stale_reason,
        ),
    }
    return CalibrationStatusResponse(**status)


@router.post("/apriltag", response_model=CalibrationResult)
def calibrate_apriltags(
    payload: AprilTagCalibrationRequest,
    camera: CameraService = Depends(get_camera_service),
    calibration: CalibrationService = Depends(get_calibration_service),
) -> CalibrationResult:
    frame = camera.capture_frame()
    try:
        detections = get_apriltag_service().detect(frame)
        config = get_config_store().config
        height, width = frame.shape[:2]
        intrinsics = resolve_camera_intrinsics(
            image_width=width,
            image_height=height,
            hfov_deg=config.camera.hfov_deg,
            distortion_model=config.camera.distortion_model,
        )
        zero_dist = intrinsics.dist_coeffs

        def undistort_center(center: np.ndarray) -> np.ndarray:
            return undistort_points(
                center.reshape(1, 2),
                intrinsics.camera_matrix,
                zero_dist,
                intrinsics.distortion_model,
            )[0]

        def undistort_corner(corner: np.ndarray) -> np.ndarray:
            return undistort_points(
                corner.reshape(1, 2),
                intrinsics.camera_matrix,
                zero_dist,
                intrinsics.distortion_model,
            )[0]

        derived = derive_work_area_from_detections(
            detections,
            origin_tag_id=payload.origin_tag_id,
            size_mm=payload.size_mm,
            tag_ids=payload.tag_ids,
            center_transform=undistort_center,
            corner_transform=undistort_corner,
        )
        result, matched = calibration.calibrate(
            frame,
            derived.tag_specs,
            persist=True,
            work_area=derived.work_area,
            mm_per_px_x=derived.mm_per_px_x,
            mm_per_px_y=derived.mm_per_px_y,
        )
    except CalibrationError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    distortion = (
        DistortionSummary(**result.intrinsics.summary()) if result.intrinsics else None
    )
    tag_size_validation = _tag_size_validation_summary(result)
    message = "Calibration saved"
    if tag_size_validation is not None and tag_size_validation.warning:
        message = tag_size_validation.warning

    background_captured = False
    if payload.capture_empty_background:
        shapes_cfg = get_config_store().config.detection
        view = get_work_area_renderer().render(frame, max_edge_px=shapes_cfg.max_edge_px)
        get_work_area_background_store().save(view, max_edge_px=shapes_cfg.max_edge_px)
        background_captured = True

    return CalibrationResult(
        success=True,
        timestamp=result.timestamp,
        reprojection_error_mm=result.reprojection_error_mm,
        tags_detected=matched,
        message=message,
        distortion=distortion,
        work_area=_work_area_summary(result),
        tag_size_validation=tag_size_validation,
        background_captured=background_captured,
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


@router.get("/apriltag/debug-image")
def calibration_debug_image(
    camera: CameraService = Depends(get_camera_service),
    calibration: CalibrationService = Depends(get_calibration_service),
) -> Response:
    if not calibration.is_calibrated():
        raise HTTPException(status_code=400, detail={"message": "Calibration required"})
    frame = camera.capture_frame()
    jpeg = get_debug_renderer().render(
        frame,
        detections=None,
        draw_tags=True,
        draw_grid=True,
        draw_side_lengths=True,
        draw_tag_sizes=True,
    )
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/apriltag/generate-pdf")
def generate_apriltag_sheet(payload: AprilTagPdfRequest) -> Response:
    pdf_bytes = generate_apriltag_pdf(payload)
    filename = f'apriltags_0-3_{payload.size_mm:g}mm.pdf'
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

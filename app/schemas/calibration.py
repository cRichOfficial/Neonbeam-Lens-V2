from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AprilTagSpec(BaseModel):
    """Internal / persisted tag placement after calibration (not used in API requests)."""

    id: int
    x_mm: float = Field(description="Tag center X in bed mm (see bed.origin / bed.y_axis in config)")
    y_mm: float = Field(description="Tag center Y in bed mm (see bed.origin / bed.y_axis in config)")
    size_mm: float = Field(gt=0, description="Printed black square edge length in mm")
    rotation_deg: float = Field(
        default=0.0,
        description="Clockwise rotation of the tag on the bed relative to the PDF print orientation",
    )


class AprilTagCalibrationRequest(BaseModel):
    origin_tag_id: int = Field(
        description="Tag ID whose center is the bed origin (0, 0)",
    )
    size_mm: float = Field(
        gt=0,
        description="Printed black square edge length in mm (measure with calipers)",
    )
    tag_ids: list[int] = Field(
        min_length=4,
        description="AprilTag IDs placed at the four corners of the work area",
    )


class DetectedAprilTag(BaseModel):
    id: int
    center_px: list[float]
    corners_px: list[list[float]]


class DistortionSummary(BaseModel):
    distortion_model: str = "pinhole"
    hfov_deg: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0


class WorkAreaSummary(BaseModel):
    width_mm: float
    height_mm: float
    origin_tag_id: int
    size_mm: float


class TagSizeValidation(BaseModel):
    expected_mm: float
    measured_mm: dict[int, float]
    mean_mm: float
    max_error_mm: float
    scale_iterations: int
    converged: bool
    warning: str | None = None
    mm_per_px_x: float | None = None
    mm_per_px_y: float | None = None
    mean_horizontal_mm: float | None = None
    mean_vertical_mm: float | None = None
    scale_x_iterations: int = 0
    scale_y_iterations: int = 0


class CalibrationStatusResponse(BaseModel):
    calibrated: bool
    bed_frame: str = ""
    timestamp: datetime | None = None
    reprojection_error_mm: float | None = None
    tags_detected: int = 0
    tags_expected: int = 0
    message: str = ""
    distortion: DistortionSummary | None = None
    work_area: WorkAreaSummary | None = None
    tag_size_validation: TagSizeValidation | None = None


class CalibrationResult(BaseModel):
    success: bool
    timestamp: datetime
    reprojection_error_mm: float
    tags_detected: list[DetectedAprilTag]
    message: str = ""
    distortion: DistortionSummary | None = None
    work_area: WorkAreaSummary | None = None
    tag_size_validation: TagSizeValidation | None = None


class AprilTagPdfRequest(BaseModel):
    size_mm: float = Field(gt=0, description="Physical tag edge length in mm")
    safe_zone_padding_mm: float = Field(default=5.0, ge=0)
    family: Literal["tag36h11"] = "tag36h11"

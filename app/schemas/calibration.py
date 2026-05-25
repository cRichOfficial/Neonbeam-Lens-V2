from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AprilTagSpec(BaseModel):
    id: int
    x_mm: float = Field(description="Tag center X in bed mm (see bed.origin / bed.y_axis in config)")
    y_mm: float = Field(description="Tag center Y in bed mm (see bed.origin / bed.y_axis in config)")
    size_mm: float = Field(gt=0, description="Printed black square edge length in mm")
    rotation_deg: float = Field(
        default=0.0,
        description="Clockwise rotation of the tag on the bed relative to the PDF print orientation",
    )


class AprilTagCalibrationRequest(BaseModel):
    tags: list[AprilTagSpec] = Field(min_length=1)


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


class CalibrationStatusResponse(BaseModel):
    calibrated: bool
    bed_frame: str = ""
    timestamp: datetime | None = None
    reprojection_error_mm: float | None = None
    tags_detected: int = 0
    tags_expected: int = 0
    message: str = ""
    distortion: DistortionSummary | None = None


class CalibrationResult(BaseModel):
    success: bool
    timestamp: datetime
    reprojection_error_mm: float
    tags_detected: list[DetectedAprilTag]
    message: str = ""
    distortion: DistortionSummary | None = None


class AprilTagPdfRequest(BaseModel):
    size_mm: float = Field(gt=0, description="Physical tag edge length in mm")
    safe_zone_padding_mm: float = Field(default=5.0, ge=0)
    family: Literal["tag36h11"] = "tag36h11"

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AprilTagSpec(BaseModel):
    id: int
    x_mm: float
    y_mm: float
    size_mm: float = Field(gt=0)


class AprilTagCalibrationRequest(BaseModel):
    tags: list[AprilTagSpec] = Field(min_length=1)


class DetectedAprilTag(BaseModel):
    id: int
    center_px: list[float]
    corners_px: list[list[float]]


class CalibrationStatusResponse(BaseModel):
    calibrated: bool
    timestamp: datetime | None = None
    reprojection_error_mm: float | None = None
    tags_detected: int = 0
    tags_expected: int = 0
    message: str = ""


class CalibrationResult(BaseModel):
    success: bool
    timestamp: datetime
    reprojection_error_mm: float
    tags_detected: list[DetectedAprilTag]
    message: str = ""


class AprilTagPdfRequest(BaseModel):
    size_mm: float = Field(gt=0, description="Physical tag edge length in mm")
    safe_zone_padding_mm: float = Field(default=5.0, ge=0)
    family: Literal["tag36h11"] = "tag36h11"

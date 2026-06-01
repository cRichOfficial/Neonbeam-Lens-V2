from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, Point2D

ShapeKind = Literal["circle", "rect", "rounded_rect"]

DebugStage = Literal[
    "raw",
    "warp",
    "bg_diff",
    "texture_diff",
    "bg_subtract",
    "fastsam",
    "fastsam_filtered",
    "final",
    "all",
]


class WorkAreaSummary(BaseModel):
    width_mm: float
    height_mm: float
    origin_tag_id: int


class WorkAreaImageInfo(BaseModel):
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    pixels_per_mm: float
    coordinate_frame: str = "bottom_left_y_up"
    origin_tag_id: int
    mm_to_work_area_px: dict[str, str] = Field(
        default_factory=lambda: {
            "x_px": "x_mm * pixels_per_mm",
            "y_px": "(height_mm - y_mm) * pixels_per_mm",
        }
    )


class DetectionItem(BaseModel):
    id: int
    shape: ShapeKind
    confidence: float
    bbox_mm: BoundingBox
    center_mm: Point2D
    width_mm: float
    height_mm: float
    rotation_deg: float
    oriented_box_mm: list[Point2D]
    segmentation_polygon_mm: list[Point2D]
    segmentation_polygon_work_area_px: list[Point2D] | None = None
    oriented_box_work_area_px: list[Point2D] | None = None


class DetectionResponse(BaseModel):
    backend: str
    calibrated: bool
    count: int
    detections: list[DetectionItem]
    work_area: WorkAreaSummary | None = None
    work_area_image: WorkAreaImageInfo | None = None
    fastsam_used: bool = False
    fastsam_device: str | None = None
    fastsam_error: str | None = None
    fastsam_filter_detail: str | None = None
    background_reference_used: bool = False
    background_stale_reason: str | None = None


class WorkAreaBackgroundStatusResponse(BaseModel):
    present: bool
    timestamp: str | None = None
    pixels_per_mm: float | None = None
    width_px: int | None = None
    height_px: int | None = None
    stale_reason: str | None = None


class CaptureBackgroundResponse(BaseModel):
    captured: bool
    timestamp: str
    pixels_per_mm: float
    width_px: int
    height_px: int

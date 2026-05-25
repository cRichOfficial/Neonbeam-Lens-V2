from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, Point2D


ShapeKind = Literal["circle", "rect", "rounded_rect"]
ShapeSource = Literal["classical", "fastsam"]
ShapeBackend = Literal["auto", "classical", "fastsam"]
DebugStage = Literal["raw", "warp", "mask", "contours", "shapes", "fastsam", "final", "all"]


class ShapesRequest(BaseModel):
    backend: ShapeBackend = "auto"
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    include_work_area_coords: bool = False


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


class ShapeDetectionItem(BaseModel):
    id: int
    shape: ShapeKind
    confidence: float
    source: ShapeSource
    center_mm: Point2D
    width_mm: float
    height_mm: float
    rotation_deg: float
    bbox_mm: BoundingBox
    oriented_box_mm: list[Point2D]
    segmentation_polygon_mm: list[Point2D]
    segmentation_polygon_work_area_px: list[Point2D] | None = None
    oriented_box_work_area_px: list[Point2D] | None = None


class ShapesResponse(BaseModel):
    backend: str
    calibrated: bool
    work_area: WorkAreaSummary | None = None
    count: int
    objects: list[ShapeDetectionItem]
    work_area_image: WorkAreaImageInfo | None = None
    fastsam_used: bool = False

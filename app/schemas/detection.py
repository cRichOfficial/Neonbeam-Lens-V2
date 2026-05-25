from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, Point2D


class DetectionRequest(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    object_height_mm: float | None = Field(default=None, ge=0)
    debug: bool = False


class SegmentationRequest(BaseModel):
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    object_height_mm: float | None = Field(default=None, ge=0)
    debug: bool = False


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_px: BoundingBox
    bbox_mm: BoundingBox | None = None
    center_mm: Point2D | None = None
    segmentation_polygon_px: list[Point2D] | None = None
    segmentation_polygon_mm: list[Point2D] | None = None


class DetectionResponse(BaseModel):
    backend: str
    count: int
    detections: list[DetectionItem]
    calibrated: bool
    object_height_mm: float


class DebugImageQuery(BaseModel):
    object_height_mm: float | None = Field(default=None, ge=0)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)

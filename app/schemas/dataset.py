from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import BoundingBox, Point2D


class AnnotationShape(BaseModel):
    id: str
    class_id: int = Field(ge=0)
    type: Literal["bbox", "polygon"]
    bbox: BoundingBox
    polygon: list[Point2D] = Field(default_factory=list)


class ImageRecord(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    captured_at: datetime
    reviewed: bool = False
    notes: str = ""
    annotations: list[AnnotationShape] = Field(default_factory=list)


class ImageSummary(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    captured_at: datetime
    reviewed: bool
    annotation_count: int


class ImageListResponse(BaseModel):
    total: int
    items: list[ImageSummary]


class ClassesResponse(BaseModel):
    classes: list[str]


class ClassesUpdateRequest(BaseModel):
    classes: list[str] = Field(min_length=1)


class AnnotationsUpdateRequest(BaseModel):
    annotations: list[AnnotationShape]


class ImagePatchRequest(BaseModel):
    reviewed: bool | None = None
    notes: str | None = None


class ExportRequest(BaseModel):
    reviewed_only: bool = True
    seed: int | None = None


class DatasetExportSummary(BaseModel):
    task: Literal["detection", "segmentation"]
    train_images: int
    val_images: int
    output_path: str
    dataset_yaml: str


class ExportResponse(BaseModel):
    exported_at: datetime
    reviewed_only: bool
    train_val_split: float
    class_counts: dict[str, int]
    detection: DatasetExportSummary
    segmentation: DatasetExportSummary


class ExportStatusResponse(BaseModel):
    last_export_at: datetime | None = None
    last_export: ExportResponse | None = None


ImageVariant = Literal["full", "preview", "thumb"]


class DatasetStatsResponse(BaseModel):
    total_images: int
    reviewed_images: int
    annotated_images: int
    class_counts: dict[str, int]
    train_val_split: float

from __future__ import annotations

import json
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.config import PROJECT_ROOT, get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.schemas.dataset import (
    AnnotationShape,
    DatasetExportSummary,
    DatasetStatsResponse,
    ExportResponse,
    ImageListResponse,
    ImageRecord,
    ImageSummary,
)
from app.services.camera_service import CameraService, get_camera_service

THUMB_MAX_PX = 256
PREVIEW_MAX_PX = 1280


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return resolved.as_posix()


class DatasetService:
    def __init__(self, camera_service: CameraService | None = None) -> None:
        self.camera_service = camera_service or get_camera_service()
        self._ensure_dirs()

    def _config(self):
        return get_config_store().config.dataset

    @property
    def root(self) -> Path:
        return self._config().resolved_storage_path

    @property
    def images_dir(self) -> Path:
        return self.root / "images" / "captured"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def thumbnails_dir(self) -> Path:
        return self.root / "images" / "thumbnails"

    @property
    def previews_dir(self) -> Path:
        return self.root / "images" / "previews"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def export_status_path(self) -> Path:
        return self.export_dir / "status.json"

    def _ensure_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def get_classes(self) -> list[str]:
        return list(self._config().classes)

    def set_classes(self, classes: list[str]) -> list[str]:
        cleaned = [name.strip() for name in classes if name.strip()]
        if not cleaned:
            raise ValueError("At least one class name is required")
        get_config_store().update({"dataset": {"classes": cleaned}})
        self._sync_training_dataset_yaml(cleaned)
        return cleaned

    def _sync_training_dataset_yaml(self, classes: list[str]) -> None:
        training_yaml = PROJECT_ROOT / "training" / "dataset.yaml"
        payload = {
            "path": "../data/dataset/export/detection",
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": classes,
        }
        training_yaml.parent.mkdir(parents=True, exist_ok=True)
        with training_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)

    def capture_from_camera(self) -> ImageRecord:
        if not self.get_classes():
            raise ValueError("Define at least one class before capturing images")

        jpeg = self.camera_service.capture_jpeg()
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode captured JPEG")

        height, width = image.shape[:2]
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.jpg"
        image_path = self.images_dir / filename
        with image_path.open("wb") as handle:
            handle.write(jpeg)

        self._save_thumbnail(image_id, image)
        self._save_preview(image_id, image)

        record = ImageRecord(
            id=image_id,
            filename=filename,
            width=width,
            height=height,
            captured_at=datetime.now(timezone.utc),
            reviewed=False,
            annotations=[],
        )
        self._save_record(record)
        return record

    def list_images(
        self,
        reviewed: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ImageListResponse:
        records = self._load_all_records()
        if reviewed is not None:
            records = [record for record in records if record.reviewed == reviewed]

        records.sort(key=lambda item: item.captured_at, reverse=True)
        total = len(records)
        page = records[offset : offset + limit]
        items = [
            ImageSummary(
                id=record.id,
                filename=record.filename,
                width=record.width,
                height=record.height,
                captured_at=record.captured_at,
                reviewed=record.reviewed,
                annotation_count=len(record.annotations),
            )
            for record in page
        ]
        return ImageListResponse(total=total, items=items)

    def get_image(self, image_id: str) -> ImageRecord:
        return self._load_record(image_id)

    def get_image_bytes(self, image_id: str, variant: str = "full") -> bytes:
        if variant == "full":
            record = self._load_record(image_id)
            path = self.images_dir / record.filename
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {record.filename}")
            return path.read_bytes()

        if variant == "thumb":
            thumb_path = self.thumbnails_dir / f"{image_id}.jpg"
            if thumb_path.exists():
                return thumb_path.read_bytes()
            return self._encode_variant(image_id, THUMB_MAX_PX, quality=75)

        if variant == "preview":
            preview_path = self.previews_dir / f"{image_id}.jpg"
            if preview_path.exists():
                return preview_path.read_bytes()
            return self._encode_variant(image_id, PREVIEW_MAX_PX, quality=85)

        raise ValueError(f"Invalid image variant: {variant}")

    def get_stats(self, reviewed_only: bool = True) -> DatasetStatsResponse:
        classes = self.get_classes()
        records = self._load_all_records()
        if reviewed_only:
            scoped = [record for record in records if record.reviewed]
        else:
            scoped = records

        class_counts = {name: 0 for name in classes}
        annotated = 0
        for record in scoped:
            if record.annotations:
                annotated += 1
            for ann in record.annotations:
                if ann.class_id < len(classes):
                    class_counts[classes[ann.class_id]] += 1

        return DatasetStatsResponse(
            total_images=len(records),
            reviewed_images=sum(1 for record in records if record.reviewed),
            annotated_images=annotated,
            class_counts=class_counts,
            train_val_split=self._config().train_val_split,
        )

    def save_annotations(self, image_id: str, annotations: list[AnnotationShape]) -> ImageRecord:
        classes = self.get_classes()
        record = self._load_record(image_id)
        validated: list[AnnotationShape] = []
        for ann in annotations:
            if ann.class_id >= len(classes):
                raise ValueError(f"Invalid class_id {ann.class_id}")
            normalized = self._normalize_annotation(ann)
            validated.append(normalized)
        record.annotations = validated
        self._save_record(record)
        return record

    def patch_image(
        self,
        image_id: str,
        reviewed: bool | None = None,
        notes: str | None = None,
    ) -> ImageRecord:
        record = self._load_record(image_id)
        if reviewed is not None:
            record.reviewed = reviewed
        if notes is not None:
            record.notes = notes
        self._save_record(record)
        return record

    def delete_image(self, image_id: str) -> None:
        record = self._load_record(image_id)
        image_path = self.images_dir / record.filename
        thumb_path = self.thumbnails_dir / f"{image_id}.jpg"
        preview_path = self.previews_dir / f"{image_id}.jpg"
        meta_path = self._metadata_path(image_id)
        if image_path.exists():
            image_path.unlink()
        if thumb_path.exists():
            thumb_path.unlink()
        if preview_path.exists():
            preview_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def export_datasets(
        self,
        reviewed_only: bool = True,
        seed: int | None = None,
    ) -> ExportResponse:
        classes = self.get_classes()
        if not classes:
            raise ValueError("Define at least one class before exporting")

        records = self._load_all_records()
        if reviewed_only:
            records = [record for record in records if record.reviewed]
        records = [record for record in records if record.annotations]
        if not records:
            raise ValueError("No annotated images available for export")

        split_ratio = self._config().train_val_split
        rng = random.Random(seed)
        rng.shuffle(records)
        split_index = max(1, int(len(records) * split_ratio)) if len(records) > 1 else 1
        if split_index >= len(records):
            split_index = len(records) - 1
        train_records = records[:split_index]
        val_records = records[split_index:]
        if not val_records:
            val_records = train_records[-1:]
            train_records = train_records[:-1]
        if not train_records:
            train_records = val_records

        detection_dir = self.export_dir / "detection"
        segmentation_dir = self.export_dir / "segmentation"
        self._prepare_export_tree(detection_dir)
        self._prepare_export_tree(segmentation_dir)

        class_counts = {name: 0 for name in classes}
        for record in records:
            for ann in record.annotations:
                if ann.class_id < len(classes):
                    class_counts[classes[ann.class_id]] += 1

        for split_name, split_records in (("train", train_records), ("val", val_records)):
            self._write_split(detection_dir, split_name, split_records, seg=False)
            self._write_split(segmentation_dir, split_name, split_records, seg=True)

        det_yaml = self._write_dataset_yaml(detection_dir, classes)
        seg_yaml = self._write_dataset_yaml(segmentation_dir, classes)

        response = ExportResponse(
            exported_at=datetime.now(timezone.utc),
            reviewed_only=reviewed_only,
            train_val_split=split_ratio,
            class_counts=class_counts,
            detection=DatasetExportSummary(
                task="detection",
                train_images=len(train_records),
                val_images=len(val_records),
                output_path=_display_path(detection_dir),
                dataset_yaml=_display_path(det_yaml),
            ),
            segmentation=DatasetExportSummary(
                task="segmentation",
                train_images=len(train_records),
                val_images=len(val_records),
                output_path=_display_path(segmentation_dir),
                dataset_yaml=_display_path(seg_yaml),
            ),
        )
        self._save_export_status(response)
        return response

    def get_export_status(self) -> ExportResponse | None:
        if not self.export_status_path.exists():
            return None
        with self.export_status_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ExportResponse.model_validate(payload)

    def _save_export_status(self, response: ExportResponse) -> None:
        with self.export_status_path.open("w", encoding="utf-8") as handle:
            json.dump(response.model_dump(mode="json"), handle, indent=2)

    def _prepare_export_tree(self, root: Path) -> None:
        if root.exists():
            shutil.rmtree(root)
        for split in ("train", "val"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _write_split(
        self,
        export_root: Path,
        split_name: str,
        records: list[ImageRecord],
        seg: bool,
    ) -> None:
        for record in records:
            src = self.images_dir / record.filename
            stem = Path(record.filename).stem
            dst_image = export_root / "images" / split_name / record.filename
            dst_label = export_root / "labels" / split_name / f"{stem}.txt"
            shutil.copy2(src, dst_image)
            lines = [
                self._annotation_to_yolo_line(ann, seg=seg)
                for ann in record.annotations
            ]
            dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_dataset_yaml(self, export_root: Path, classes: list[str]) -> Path:
        yaml_path = export_root / "dataset.yaml"
        payload = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": classes,
        }
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
        return yaml_path

    @staticmethod
    def bbox_to_yolo_line(class_id: int, bbox: BoundingBox) -> str:
        cx = (bbox.x_min + bbox.x_max) / 2
        cy = (bbox.y_min + bbox.y_max) / 2
        w = bbox.x_max - bbox.x_min
        h = bbox.y_max - bbox.y_min
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def bbox_to_polygon(bbox: BoundingBox) -> list[Point2D]:
        return [
            Point2D(x=bbox.x_min, y=bbox.y_min),
            Point2D(x=bbox.x_max, y=bbox.y_min),
            Point2D(x=bbox.x_max, y=bbox.y_max),
            Point2D(x=bbox.x_min, y=bbox.y_max),
        ]

    def _annotation_to_yolo_line(self, ann: AnnotationShape, seg: bool) -> str:
        if not seg:
            return self.bbox_to_yolo_line(ann.class_id, ann.bbox)
        points = ann.polygon if ann.type == "polygon" and ann.polygon else self.bbox_to_polygon(ann.bbox)
        coords = " ".join(f"{point.x:.6f} {point.y:.6f}" for point in points)
        return f"{ann.class_id} {coords}"

    @staticmethod
    def _normalize_annotation(ann: AnnotationShape) -> AnnotationShape:
        bbox = ann.bbox
        bbox = BoundingBox(
            x_min=min(bbox.x_min, bbox.x_max),
            y_min=min(bbox.y_min, bbox.y_max),
            x_max=max(bbox.x_min, bbox.x_max),
            y_max=max(bbox.y_min, bbox.y_max),
        )
        polygon = ann.polygon
        if ann.type == "polygon":
            if len(polygon) < 3:
                raise ValueError("Polygon annotations require at least 3 points")
            xs = [point.x for point in polygon]
            ys = [point.y for point in polygon]
            bbox = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))
        return AnnotationShape(
            id=ann.id,
            class_id=ann.class_id,
            type=ann.type,
            bbox=bbox,
            polygon=polygon,
        )

    def _metadata_path(self, image_id: str) -> Path:
        return self.metadata_dir / f"{image_id}.json"

    @staticmethod
    def _resize_max_edge(image: np.ndarray, max_px: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(1.0, max_px / max(height, width))
        if scale >= 1.0:
            return image
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _read_image_bgr(self, image_id: str) -> np.ndarray:
        record = self._load_record(image_id)
        path = self.images_dir / record.filename
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {record.filename}")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image: {record.filename}")
        return image

    def _encode_jpeg(self, image: np.ndarray, quality: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode JPEG")
        return encoded.tobytes()

    def _save_thumbnail(self, image_id: str, image_bgr: np.ndarray) -> None:
        thumb = self._resize_max_edge(image_bgr, THUMB_MAX_PX)
        thumb_path = self.thumbnails_dir / f"{image_id}.jpg"
        thumb_path.write_bytes(self._encode_jpeg(thumb, 75))

    def _save_preview(self, image_id: str, image_bgr: np.ndarray) -> None:
        preview = self._resize_max_edge(image_bgr, PREVIEW_MAX_PX)
        preview_path = self.previews_dir / f"{image_id}.jpg"
        preview_path.write_bytes(self._encode_jpeg(preview, 85))

    def _encode_variant(self, image_id: str, max_px: int, quality: int) -> bytes:
        image = self._read_image_bgr(image_id)
        resized = self._resize_max_edge(image, max_px)
        return self._encode_jpeg(resized, quality)

    def _save_record(self, record: ImageRecord) -> None:
        path = self._metadata_path(record.id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, indent=2)

    def _load_record(self, image_id: str) -> ImageRecord:
        path = self._metadata_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Image record not found: {image_id}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ImageRecord.model_validate(payload)

    def _load_all_records(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for path in self.metadata_dir.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records.append(ImageRecord.model_validate(payload))
        return records


_dataset_service: DatasetService | None = None


def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_px: BoundingBox
    segmentation_polygon_px: list[Point2D] | None = None


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def detect(self, frame: np.ndarray, confidence_threshold: float) -> list[RawDetection]: ...

    def segment(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> list[RawDetection]:
        return self.detect(frame, confidence_threshold)


class CPUDetector(BaseDetector):
    name = "cpu"

    def __init__(self) -> None:
        self._model = None
        self._segmentation_model = None
        self._class_names: dict[int, str] = {}

    def _resolve_model_path(self) -> Path | None:
        config = get_config_store().config.detection
        if config.resolved_cpu_model_path.exists():
            return config.resolved_cpu_model_path
        return None

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_path = self._resolve_model_path()
        if model_path is None:
            return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
            names = getattr(self._model, "names", {}) or {}
            self._class_names = {int(k): str(v) for k, v in names.items()}
            logger.info("Loaded CPU model from %s", model_path)
            return self._model
        except Exception as exc:
            logger.warning("Failed to load CPU model: %s", exc)
            return None

    def _load_segmentation_model(self):
        if self._segmentation_model is not None:
            return self._segmentation_model
        config = get_config_store().config.detection
        seg_path = config.resolved_segmentation_model_path
        if not seg_path.exists():
            seg_path = self._resolve_model_path()
        if seg_path is None or not seg_path.exists():
            return None
        try:
            from ultralytics import YOLO

            self._segmentation_model = YOLO(str(seg_path))
            return self._segmentation_model
        except Exception as exc:
            logger.warning("Failed to load segmentation model: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._load_model() is not None

    def detect(self, frame: np.ndarray, confidence_threshold: float) -> list[RawDetection]:
        model = self._load_model()
        if model is None:
            return []
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
        results = model.predict(bgr, conf=confidence_threshold, verbose=False)
        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                detections.append(
                    RawDetection(
                        class_id=class_id,
                        class_name=self._class_names.get(class_id, str(class_id)),
                        confidence=float(box.conf[0]),
                        bbox_px=BoundingBox(
                            x_min=xyxy[0],
                            y_min=xyxy[1],
                            x_max=xyxy[2],
                            y_max=xyxy[3],
                        ),
                    )
                )
        return detections

    def segment(
        self, frame: np.ndarray, confidence_threshold: float
    ) -> list[RawDetection]:
        model = self._load_segmentation_model() or self._load_model()
        if model is None:
            return []
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
        results = model.predict(bgr, conf=confidence_threshold, verbose=False)
        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None:
                continue
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                polygon: list[Point2D] | None = None
                if masks is not None and masks.xy is not None and idx < len(masks.xy):
                    contour = masks.xy[idx]
                    if contour is not None and len(contour) > 0:
                        polygon = [
                            Point2D(x=float(pt[0]), y=float(pt[1])) for pt in contour
                        ]
                detections.append(
                    RawDetection(
                        class_id=class_id,
                        class_name=self._class_names.get(class_id, str(class_id)),
                        confidence=float(box.conf[0]),
                        bbox_px=BoundingBox(
                            x_min=xyxy[0],
                            y_min=xyxy[1],
                            x_max=xyxy[2],
                            y_max=xyxy[3],
                        ),
                        segmentation_polygon_px=polygon,
                    )
                )
        return detections


_cpu_detector: CPUDetector | None = None


def get_cpu_detector() -> CPUDetector:
    global _cpu_detector
    if _cpu_detector is None:
        _cpu_detector = CPUDetector()
    return _cpu_detector

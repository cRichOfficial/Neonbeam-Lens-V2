from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.services.cpu_detector import BaseDetector, RawDetection

logger = logging.getLogger(__name__)

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _resolve_hef_path() -> Path | None:
    config = get_config_store().config.detection
    candidates = [
        config.resolved_model_path,
        Path(config.hailo_fallback_model),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class HailoDetector(BaseDetector):
    name = "hailo"

    def __init__(self) -> None:
        self._hailo = None
        self._model_path: Path | None = None
        self._input_size = (640, 640)
        self._labels = COCO_LABELS

    def _ensure_loaded(self) -> bool:
        if self._hailo is not None:
            return True
        model_path = _resolve_hef_path()
        if model_path is None:
            return False
        try:
            from picamera2.devices import Hailo

            self._hailo = Hailo(str(model_path))
            self._model_path = model_path
            model_h, model_w, _ = self._hailo.get_input_shape()
            self._input_size = (model_w, model_h)
            logger.info("Loaded Hailo model from %s", model_path)
            return True
        except Exception as exc:
            logger.warning("Hailo unavailable: %s", exc)
            self._hailo = None
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def close(self) -> None:
        if self._hailo is not None:
            try:
                self._hailo.close()
            except Exception:
                pass
            self._hailo = None

    def detect(self, frame: np.ndarray, confidence_threshold: float) -> list[RawDetection]:
        if not self._ensure_loaded() or self._hailo is None:
            return []

        resized = cv2.resize(frame, self._input_size)
        if resized.ndim == 3 and resized.shape[2] == 3:
            inference_frame = resized
        else:
            inference_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        results = self._hailo.run(inference_frame)
        return self._parse_hailo_results(
            results,
            frame.shape[1],
            frame.shape[0],
            confidence_threshold,
        )

    def _parse_hailo_results(
        self,
        results,
        frame_w: int,
        frame_h: int,
        confidence_threshold: float,
    ) -> list[RawDetection]:
        detections: list[RawDetection] = []
        if results is None:
            return detections

        try:
            from picamera2.devices import Hailo

            dets = Hailo.get_detections(results) if hasattr(Hailo, "get_detections") else results
        except Exception:
            dets = results

        if not isinstance(dets, (list, tuple)):
            return detections

        scale_x = frame_w / self._input_size[0]
        scale_y = frame_h / self._input_size[1]

        for det in dets:
            if isinstance(det, dict):
                confidence = float(det.get("confidence", det.get("score", 0.0)))
                class_id = int(det.get("class_id", det.get("label", 0)))
                xmin = float(det.get("xmin", 0.0))
                ymin = float(det.get("ymin", 0.0))
                xmax = float(det.get("xmax", 0.0))
                ymax = float(det.get("ymax", 0.0))
            else:
                confidence = float(getattr(det, "confidence", 0.0))
                class_id = int(getattr(det, "class_id", 0))
                bbox = getattr(det, "bbox", None)
                if bbox is None:
                    continue
                xmin, ymin, xmax, ymax = map(float, bbox)

            if confidence < confidence_threshold:
                continue

            detections.append(
                RawDetection(
                    class_id=class_id,
                    class_name=self._labels[class_id] if class_id < len(self._labels) else str(class_id),
                    confidence=confidence,
                    bbox_px=BoundingBox(
                        x_min=xmin * scale_x,
                        y_min=ymin * scale_y,
                        x_max=xmax * scale_x,
                        y_max=ymax * scale_y,
                    ),
                )
            )
        return detections


_hailo_detector: HailoDetector | None = None


def get_hailo_detector() -> HailoDetector:
    global _hailo_detector
    if _hailo_detector is None:
        _hailo_detector = HailoDetector()
    return _hailo_detector

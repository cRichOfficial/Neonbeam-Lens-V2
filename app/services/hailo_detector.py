from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.services.cpu_detector import BaseDetector, RawDetection

logger = logging.getLogger(__name__)

# After a failed init (e.g. device busy), wait before retrying to avoid log spam
# and repeated vdevice open attempts that can disturb the camera pipeline.
_INIT_RETRY_COOLDOWN_S = 60.0

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


def _format_load_error(exc: Exception) -> str:
    msg = str(exc)
    if "HAILO_OUT_OF_PHYSICAL_DEVICES" in msg or "error: 74" in msg:
        return (
            f"{msg} — the Hailo NPU is already in use or unavailable. "
            "Stop duplicate app instances (e.g. `sudo systemctl stop laser-detection` "
            "when running manual uvicorn), or reboot if a crashed process left the device locked."
        )
    return msg


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
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._retry_after = 0.0

    def is_loaded(self) -> bool:
        return self._hailo is not None

    def get_status(self) -> dict[str, str | bool | None]:
        return {
            "loaded": self.is_loaded(),
            "model_path": str(self._model_path) if self._model_path else None,
            "last_error": self._last_error,
        }

    def try_load(self, *, force: bool = False) -> bool:
        with self._lock:
            return self._load_locked(force=force)

    def _load_locked(self, *, force: bool = False) -> bool:
        if self._hailo is not None:
            return True

        now = time.monotonic()
        if not force and self._last_error and now < self._retry_after:
            return False

        model_path = _resolve_hef_path()
        if model_path is None:
            self._last_error = "no HEF model found"
            return False

        hailo = None
        try:
            from picamera2.devices import Hailo

            hailo = Hailo(str(model_path))
            model_h, model_w, _ = hailo.get_input_shape()
            self._hailo = hailo
            self._model_path = model_path
            self._input_size = (model_w, model_h)
            self._last_error = None
            self._retry_after = 0.0
            logger.info("Loaded Hailo model from %s", model_path)
            return True
        except Exception as exc:
            if hailo is not None:
                try:
                    hailo.close()
                except Exception:
                    pass
            self._hailo = None
            self._last_error = _format_load_error(exc)
            self._retry_after = now + _INIT_RETRY_COOLDOWN_S
            logger.warning("Hailo unavailable: %s", self._last_error)
            return False

    def is_available(self) -> bool:
        return self.try_load()

    def close(self) -> None:
        with self._lock:
            if self._hailo is not None:
                try:
                    self._hailo.close()
                except Exception:
                    pass
                self._hailo = None

    def detect(self, frame: np.ndarray, confidence_threshold: float) -> list[RawDetection]:
        with self._lock:
            if not self._load_locked():
                return []
            if self._hailo is None:
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

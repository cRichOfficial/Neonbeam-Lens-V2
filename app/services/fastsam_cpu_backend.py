from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store, expand_path
from app.services.fastsam_hailo_backend import parse_fastsam_results

logger = logging.getLogger(__name__)


def resolve_fastsam_cpu_path() -> Path | None:
    path = expand_path(get_config_store().config.detection.fastsam_cpu_model_path)
    return path if path.exists() else None


class CpuFastSamBackend:
    name = "cpu"

    def __init__(self) -> None:
        self._model = None
        self._last_error: str | None = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_status(self) -> dict[str, str | bool | None]:
        return {
            "device": "cpu",
            "loaded": self.is_loaded(),
            "last_error": self._last_error,
            "model_path": str(resolve_fastsam_cpu_path() or ""),
        }

    def try_load(self, *, force: bool = False) -> bool:
        if self._model is not None and not force:
            return True
        model_path = resolve_fastsam_cpu_path()
        if model_path is None:
            self._last_error = "FastSAM CPU model not found"
            return False
        try:
            from ultralytics import FastSAM

            self._model = FastSAM(str(model_path))
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Failed to load CPU FastSAM: %s", exc)
            self._model = None
            return False

    def segment_masks(self, frame: np.ndarray) -> list[np.ndarray]:
        if not self.try_load():
            return []

        cfg = get_config_store().config.detection
        frame_h, frame_w = frame.shape[:2]
        try:
            results = self._model.predict(
                frame,
                device="cpu",
                retina_masks=True,
                imgsz=cfg.fastsam_cpu_imgsz,
                conf=cfg.fastsam_cpu_confidence,
                verbose=False,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("CPU FastSAM inference failed: %s", exc)
            return []

        if not results:
            return []

        result = results[0]
        if result.masks is None:
            return []

        masks: list[np.ndarray] = []
        data = result.masks.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        else:
            data = np.asarray(data)

        for index in range(data.shape[0]):
            mask = (data[index] > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != (frame_h, frame_w):
                mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(mask) > 0:
                masks.append(mask)
        return masks

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import get_config_store
from app.services.fastsam_cpu_backend import CpuFastSamBackend
from app.services.fastsam_hailo_backend import (
    FASTSAM_MODEL_KEY,
    HailoFastSamBackend,
    resolve_fastsam_hef_path,
)

logger = logging.getLogger(__name__)

# Re-export for main.py startup compatibility
resolve_fastsam_path = resolve_fastsam_hef_path


class FastSamDetector:
    name = "fastsam"

    def __init__(self) -> None:
        self._hailo = HailoFastSamBackend()
        self._cpu = CpuFastSamBackend()
        self._active_device: str | None = None
        self.last_segment_detail: str = ""

    def _resolve_device(self) -> str:
        return get_config_store().config.detection.fastsam_device

    def _select_backend(self):
        device = self._resolve_device()
        if device == "hailo":
            return self._hailo if self._hailo.is_loaded() or self._hailo.try_load() else None
        if device == "cpu":
            return self._cpu if self._cpu.is_loaded() or self._cpu.try_load() else None
        if self._hailo.is_loaded() or self._hailo.try_load():
            return self._hailo
        if self._cpu.is_loaded() or self._cpu.try_load():
            return self._cpu
        return None

    def is_loaded(self) -> bool:
        return self._select_backend() is not None

    def get_status(self) -> dict[str, str | bool | None]:
        backend = self._select_backend()
        hailo_status = self._hailo.get_status()
        cpu_status = self._cpu.get_status()
        device = backend.name if backend is not None else "none"
        loaded = backend is not None
        last_error = None
        if device == "none":
            last_error = cpu_status.get("last_error") or hailo_status.get("last_error")
        return {
            "device": device,
            "loaded": loaded,
            "last_error": last_error,
            "configured_device": self._resolve_device(),
            "hailo": hailo_status,
            "cpu": cpu_status,
        }

    def try_load(self, *, force: bool = False) -> bool:
        device = self._resolve_device()
        if device == "hailo":
            return self._hailo.try_load(force=force)
        if device == "cpu":
            return self._cpu.try_load(force=force)
        if self._hailo.try_load(force=force):
            return True
        return self._cpu.try_load(force=force)

    def close(self) -> None:
        pass

    def segment_masks(self, frame: np.ndarray) -> list[np.ndarray]:
        backend = self._select_backend()
        if backend is None:
            self.last_segment_detail = "no backend available"
            return []
        self._active_device = backend.name
        masks = backend.segment_masks(frame)
        self.last_segment_detail = getattr(backend, "last_segment_detail", "") or ""
        return masks

    @property
    def active_device(self) -> str | None:
        return self._active_device

    def combined_mask(self, frame: np.ndarray) -> np.ndarray:
        masks = self.segment_masks(frame)
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        for index, mask in enumerate(masks, start=1):
            label = min(255, index * 40)
            combined[mask > 0] = label
        return combined

    def render_overlay(self, frame: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        output = frame.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        palette = [
            (0, 255, 128),
            (255, 128, 0),
            (128, 128, 255),
            (255, 0, 255),
            (0, 200, 255),
        ]
        for index, mask in enumerate(masks):
            color = palette[index % len(palette)]
            overlay = output.copy()
            overlay[mask > 0] = color
            output = cv2.addWeighted(output, 0.65, overlay, 0.35, 0)
        return output


_fastsam_detector: FastSamDetector | None = None


def get_fastsam_detector() -> FastSamDetector:
    global _fastsam_detector
    if _fastsam_detector is None:
        _fastsam_detector = FastSamDetector()
    return _fastsam_detector

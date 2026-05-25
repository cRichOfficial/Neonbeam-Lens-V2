from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store
from app.services.hailo_npu import get_hailo_npu

logger = logging.getLogger(__name__)

FASTSAM_MODEL_KEY = "fastsam"


def resolve_fastsam_path() -> Path | None:
    config = get_config_store().config.shapes
    candidates = [
        config.resolved_fastsam_model_path,
        Path(config.fastsam_fallback_model),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class FastSamDetector:
    name = "fastsam"

    def __init__(self) -> None:
        self._npu = get_hailo_npu()

    def is_loaded(self) -> bool:
        return self._npu.is_loaded(FASTSAM_MODEL_KEY)

    def get_status(self) -> dict[str, str | bool | None]:
        return self._npu.get_model_status(FASTSAM_MODEL_KEY)

    def try_load(self, *, force: bool = False) -> bool:
        return self._npu.load_model(FASTSAM_MODEL_KEY, resolve_fastsam_path(), force=force)

    def close(self) -> None:
        pass

    def segment_masks(self, frame: np.ndarray) -> list[np.ndarray]:
        if not self.try_load():
            return []

        input_size = self._npu.get_input_size(FASTSAM_MODEL_KEY)
        if input_size is None:
            return []

        resized = cv2.resize(frame, input_size)
        if resized.ndim == 2:
            inference_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 3:
            inference_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            inference_frame = resized

        results = self._npu.run(FASTSAM_MODEL_KEY, inference_frame)
        return self._parse_masks(results, frame.shape[1], frame.shape[0])

    def _parse_masks(self, results, frame_w: int, frame_h: int) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        if results is None:
            return masks

        def _add_mask(raw_mask: np.ndarray) -> None:
            if raw_mask.ndim > 2:
                raw_mask = raw_mask.squeeze()
            if raw_mask.dtype != np.uint8:
                raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
            if raw_mask.shape[:2] != (frame_h, frame_w):
                raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(raw_mask) > 0:
                masks.append(raw_mask)

        if isinstance(results, dict):
            if "masks" in results:
                for item in results["masks"]:
                    _add_mask(np.asarray(item))
            elif "mask" in results:
                _add_mask(np.asarray(results["mask"]))
        elif isinstance(results, (list, tuple)):
            for item in results:
                if isinstance(item, np.ndarray) and item.ndim >= 2:
                    _add_mask(item)
                elif isinstance(item, dict) and "mask" in item:
                    _add_mask(np.asarray(item["mask"]))
        elif isinstance(results, np.ndarray) and results.ndim >= 3:
            for index in range(results.shape[0]):
                _add_mask(results[index])

        return masks

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

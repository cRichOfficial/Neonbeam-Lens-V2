"""FastSAM Hailo backend — segment masks via YOLOv8-seg postprocess."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import get_config_store
from app.services.fastsam_hailo_postprocess import postprocess_yolov8_seg
from app.services.hailo_npu import get_hailo_npu

logger = logging.getLogger(__name__)

FASTSAM_MODEL_KEY = "fastsam"
_LOG_PREFIX = "[fastsam]"


@dataclass(frozen=True)
class FastSamParseResult:
    masks: list[np.ndarray]
    detail: str = ""


def resolve_fastsam_hef_path() -> Path | None:
    config = get_config_store().config.detection
    candidates = [
        config.resolved_fastsam_model_path,
        Path(config.fastsam_fallback_model),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _describe_outputs(results) -> str:
    if isinstance(results, dict):
        parts = []
        for name, value in results.items():
            arr = np.asarray(value)
            parts.append(f"{name}={tuple(arr.shape)}")
        return ", ".join(parts[:6])
    if isinstance(results, np.ndarray):
        return f"ndarray{tuple(results.shape)}"
    return type(results).__name__


class HailoFastSamBackend:
    name = "hailo"

    def __init__(self) -> None:
        self._npu = get_hailo_npu()
        self.last_segment_detail: str = ""

    def is_loaded(self) -> bool:
        return self._npu.is_loaded(FASTSAM_MODEL_KEY)

    def get_status(self) -> dict[str, str | bool | None]:
        status = self._npu.get_model_status(FASTSAM_MODEL_KEY)
        status["device"] = "hailo"
        status["last_segment_detail"] = self.last_segment_detail or None
        return status

    def try_load(self, *, force: bool = False) -> bool:
        return self._npu.load_model(FASTSAM_MODEL_KEY, resolve_fastsam_hef_path(), force=force)

    def segment_masks(self, frame: np.ndarray) -> list[np.ndarray]:
        self.last_segment_detail = ""
        frame_h, frame_w = frame.shape[:2]
        logger.info("%s segment start warped_frame=%dx%d", _LOG_PREFIX, frame_w, frame_h)
        if not self.try_load():
            self.last_segment_detail = "model not loaded"
            logger.info("%s abort: model not loaded", _LOG_PREFIX)
            return []

        input_size = self._npu.get_input_size(FASTSAM_MODEL_KEY)
        if input_size is None:
            self.last_segment_detail = "input size unavailable"
            logger.info("%s abort: input size unavailable", _LOG_PREFIX)
            return []

        logger.info("%s hailo input_size=%dx%d", _LOG_PREFIX, input_size[0], input_size[1])
        resized = cv2.resize(frame, input_size)
        if resized.ndim == 2:
            inference_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 3:
            inference_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            inference_frame = resized
        if inference_frame.dtype != np.uint8:
            inference_frame = np.clip(inference_frame, 0, 255).astype(np.uint8)

        results = self._npu.run(FASTSAM_MODEL_KEY, inference_frame)
        logger.info(
            "%s inference done output_type=%s outputs=%s",
            _LOG_PREFIX,
            type(results).__name__,
            _describe_outputs(results),
        )
        cfg = get_config_store().config.detection
        parsed = parse_fastsam_results(
            results,
            frame.shape[1],
            frame.shape[0],
            input_size=input_size,
            score_threshold=cfg.fastsam_hailo_score_threshold,
            nms_iou_threshold=cfg.fastsam_hailo_nms_iou,
            mask_threshold=cfg.fastsam_hailo_mask_threshold,
        )
        self.last_segment_detail = parsed.detail
        if parsed.masks:
            for index, mask in enumerate(parsed.masks):
                nz = int(np.count_nonzero(mask))
                ys, xs = np.where(mask > 0) if nz else (np.array([]), np.array([]))
                if nz:
                    logger.info(
                        "%s output %s",
                        _LOG_PREFIX,
                        (
                            f"mask[{index}] area={nz}px "
                            f"bbox=({int(xs.min())},{int(ys.min())})-({int(xs.max())},{int(ys.max())})"
                        ),
                    )
                else:
                    logger.info("%s output mask[%d] empty", _LOG_PREFIX, index)
        else:
            logger.warning(
                "%s produced 0 masks (%s); raw outputs: %s",
                _LOG_PREFIX,
                parsed.detail,
                _describe_outputs(results),
            )
        logger.info(
            "%s segment done masks=%d detail=%s",
            _LOG_PREFIX,
            len(parsed.masks),
            parsed.detail,
        )
        return parsed.masks


def parse_fastsam_results(
    results,
    frame_w: int,
    frame_h: int,
    *,
    input_size: tuple[int, int] | None = None,
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    mask_threshold: float = 0.5,
) -> FastSamParseResult:
    masks: list[np.ndarray] = []
    detail = ""
    if results is None:
        return FastSamParseResult(masks, detail="inference returned None")

    if isinstance(results, dict) and "masks" not in results and "mask" not in results:
        try:
            hailo_masks, detail = postprocess_yolov8_seg(
                results,
                input_size=input_size or (640, 640),
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                mask_threshold=mask_threshold,
            )
            for mask in hailo_masks:
                _append_mask(masks, mask, frame_w, frame_h)
            if masks:
                return FastSamParseResult(masks, detail=detail)
        except Exception as exc:
            detail = f"postprocess error: {exc}"
            logger.warning("FastSAM Hailo YOLOv8-seg postprocess failed: %s", exc)
            logger.warning("FastSAM raw outputs: %s", _describe_outputs(results))

    if not detail:
        detail = "no compatible Hailo output format"

    def _add_mask(raw_mask: np.ndarray) -> None:
        _append_mask(masks, raw_mask, frame_w, frame_h)

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

    if masks and not detail:
        detail = f"{len(masks)} masks (legacy parser)"
    return FastSamParseResult(masks, detail=detail)


def _append_mask(
    masks: list[np.ndarray],
    raw_mask: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> None:
    if raw_mask.ndim > 2:
        raw_mask = raw_mask.squeeze()
    if raw_mask.dtype != np.uint8:
        raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
    if raw_mask.shape[:2] != (frame_h, frame_w):
        raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    if np.count_nonzero(raw_mask) > 0:
        masks.append(raw_mask)

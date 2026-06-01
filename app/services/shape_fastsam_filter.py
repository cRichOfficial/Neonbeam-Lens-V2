"""Post-filter FastSAM masks using bg_subtract foreground and minimum area."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.shape_detector import ShapeDetectorConfig, _estimate_bed_l
from app.services.shape_mask_tracks import track_bg_subtract


def extract_bg_subtract_mask(
    bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: ShapeDetectorConfig,
    *,
    pixels_per_mm: float,
) -> np.ndarray | None:
    """Build bg_subtract foreground mask for FastSAM post-filtering."""
    if not cfg.use_background_reference:
        return None
    if reference_bgr.shape[:2] != bgr.shape[:2]:
        return None

    h, w = bgr.shape[:2]
    margin_px = int(cfg.roi_margin_mm * pixels_per_mm)
    x0 = int(max(0, margin_px))
    y0 = int(max(0, margin_px))
    x1 = int(w - max(0, margin_px))
    y1 = int(h - max(0, margin_px))
    l_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    bed_l = _estimate_bed_l(l_channel, x0, y0, x1, y1)

    track = track_bg_subtract(
        bgr,
        reference_bgr,
        cfg,
        pixels_per_mm=pixels_per_mm,
        bed_l=bed_l,
        mode=cfg.bg_subtract_mode,
    )
    if track is None:
        return None
    return track.mask


def is_bg_mask_sane(mask: np.ndarray, *, max_foreground_ratio: float) -> bool:
    if mask.size == 0:
        return False
    fg_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return fg_ratio <= max_foreground_ratio


def _mask_bg_overlap_ratio(mask: np.ndarray, bg_mask: np.ndarray) -> float:
    mask_fg = mask > 0
    area = int(np.count_nonzero(mask_fg))
    if area == 0:
        return 0.0
    if bg_mask.shape[:2] != mask.shape[:2]:
        bg_mask = cv2.resize(bg_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlap = int(np.count_nonzero(mask_fg & (bg_mask > 0)))
    return float(overlap) / float(area)


def filter_fastsam_masks(
    masks: list[np.ndarray],
    bg_mask: np.ndarray | None,
    *,
    min_overlap: float,
    min_area_px: int,
    bg_filter_enabled: bool = True,
    max_fg_ratio: float = 0.45,
) -> tuple[list[np.ndarray], str]:
    """Drop speckles and masks with insufficient bg_subtract overlap."""
    if not masks:
        return [], "no input masks"

    kept: list[np.ndarray] = []
    dropped_area = 0
    dropped_overlap = 0

    apply_bg = (
        bg_filter_enabled
        and bg_mask is not None
        and np.count_nonzero(bg_mask) > 0
        and is_bg_mask_sane(bg_mask, max_foreground_ratio=max_fg_ratio)
    )
    bg_note = "bg filter on" if apply_bg else "bg filter skipped"

    for mask in masks:
        area = int(np.count_nonzero(mask > 0))
        if area < min_area_px:
            dropped_area += 1
            continue
        if apply_bg:
            overlap = _mask_bg_overlap_ratio(mask, bg_mask)
            if overlap < min_overlap:
                dropped_overlap += 1
                continue
        kept.append(mask)

    detail = (
        f"{len(kept)}/{len(masks)} kept ({bg_note}; "
        f"dropped area<{min_area_px}px={dropped_area}, overlap<{min_overlap:.2f}={dropped_overlap})"
    )
    return kept, detail

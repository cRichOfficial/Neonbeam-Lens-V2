"""Background subtract track for FastSAM post-filter on dark-on-white bed."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.services.shape_detector import ShapeDetectorConfig


def _morph_kernel(size_px: int) -> np.ndarray:
    k = max(3, int(size_px) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px:
            cleaned[labels == label] = 255
    return cleaned


def _clean_mask(
    mask: np.ndarray,
    cfg: ShapeDetectorConfig,
    min_component_area_px: int,
) -> np.ndarray:
    kernel_small = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    close_kernel = _morph_kernel(cfg.mask_morph_kernel_px)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, close_kernel, iterations=cfg.morph_close_iterations
    )
    return _remove_small_components(mask, min_component_area_px)


def _mask_fragmentation_metrics(
    mask: np.ndarray, min_component_area_px: int
) -> tuple[int, float]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fg_components = max(0, num_labels - 1)
    small_pixels = 0
    total_fg = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        total_fg += area
        if area < min_component_area_px:
            small_pixels += area
    speckle_ratio = float(small_pixels) / float(max(1, total_fg))
    return fg_components, speckle_ratio


def _mask_score(
    mask: np.ndarray,
    *,
    min_component_area_px: int,
    max_components: int,
    max_component_area_ratio: float,
) -> tuple[float, int, float]:
    if mask.size == 0:
        return 0.0, 0, 0.0
    fg = float(np.count_nonzero(mask))
    ratio = fg / float(mask.size)
    if ratio < 0.001 or ratio > 0.85:
        return 0.0, 0, 1.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = max(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)),
        default=0,
    )
    largest_ratio = float(max_area) / float(max(1, mask.size))
    if largest_ratio > max_component_area_ratio + 0.15:
        return 0.0, 0, 1.0

    ratio_score = 1.0 - abs(ratio - 0.15)
    component_count, speckle_ratio = _mask_fragmentation_metrics(mask, min_component_area_px)
    if component_count > max_components:
        component_penalty = min(1.0, (component_count - max_components) / float(max_components))
    else:
        component_penalty = 0.0
    fragmentation = min(1.0, 0.4 * component_penalty + 0.6 * speckle_ratio)
    return max(0.0, ratio_score * (1.0 - fragmentation)), component_count, fragmentation


def _apply_glare_exclusion_to_diff(
    diff: np.ndarray,
    bgr: np.ndarray,
    reference_bgr: np.ndarray | None,
    cfg: ShapeDetectorConfig,
    bed_l: float,
) -> np.ndarray:
    cap = max(cfg.glare_suppression_l_cap, bed_l + cfg.glare_l_delta)
    l_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    hot = l_channel > cap
    if reference_bgr is not None and reference_bgr.shape[:2] == bgr.shape[:2]:
        ref_l = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        hot = hot | ((ref_l > cap) & (l_channel > cap - 15))

    floor = int(cfg.bg_subtract_min_diff)
    weak_signal = diff < max(floor * 2, 30)
    suppress = hot & weak_signal
    cleaned = diff.copy()
    cleaned[suppress] = 0
    return cleaned


@dataclass(frozen=True)
class TrackMask:
    name: str
    mask: np.ndarray
    component_count: int
    fragmentation: float


def track_bg_subtract(
    bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: ShapeDetectorConfig,
    *,
    pixels_per_mm: float,
    bed_l: float,
) -> TrackMask | None:
    if reference_bgr.shape[:2] != bgr.shape[:2]:
        return None

    min_component_area_px = int(cfg.mask_min_component_area_mm2 * pixels_per_mm * pixels_per_mm)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, ref_gray)

    k = max(3, int(cfg.bg_subtract_blur_kernel_px) | 1)
    diff = cv2.GaussianBlur(diff, (k, k), 0)
    diff = _apply_glare_exclusion_to_diff(diff, bgr, reference_bgr, cfg, bed_l)

    _, otsu = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = int(cfg.bg_subtract_min_diff)
    _, fixed = cv2.threshold(diff, floor, 255, cv2.THRESH_BINARY)
    raw = cv2.bitwise_or(otsu, fixed)

    cleaned = _clean_mask(raw, cfg, min_component_area_px)
    _, count, frag = _mask_score(
        cleaned,
        min_component_area_px=min_component_area_px,
        max_components=cfg.mask_max_components,
        max_component_area_ratio=cfg.mask_max_component_area_ratio,
    )
    if np.count_nonzero(cleaned) == 0:
        return None
    return TrackMask(name="bg_subtract", mask=cleaned, component_count=count, fragmentation=frag)

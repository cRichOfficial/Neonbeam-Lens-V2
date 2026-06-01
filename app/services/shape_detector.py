from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.schemas.common import BoundingBox

BgSubtractMode = Literal["intensity", "texture", "fused"]
from app.schemas.detection import ShapeKind


@dataclass
class ContourCandidate:
    contour: np.ndarray
    kept: bool = True
    reject_reason: str | None = None


@dataclass
class RawShapeDetection:
    shape: ShapeKind
    confidence: float
    center_px: tuple[float, float]
    width_px: float
    height_px: float
    rotation_deg: float
    bbox_px: BoundingBox
    oriented_box_px: list[tuple[float, float]]
    segmentation_polygon_px: list[tuple[float, float]]


@dataclass
class MaskGeometryResult:
    objects: list[RawShapeDetection]
    mask: np.ndarray
    contours: list[ContourCandidate]
    glare_reject_count: int = 0
    raw_contour_count: int = 0


@dataclass
class ShapeDetectorConfig:
    min_area_mm2: float = 400.0
    max_area_mm2: float = 80000.0
    split_above_area_mm2: float = 20000.0
    max_object_span_ratio: float = 0.45
    min_solidity: float = 0.75
    min_extent: float = 0.35
    circularity_threshold: float = 0.82
    rounded_rect_circularity_min: float = 0.65
    bracelet_min_aspect: float = 6.0
    approx_epsilon_ratio: float = 0.02
    roi_margin_mm: float = 5.0
    mask_morph_kernel_px: int = 15
    mask_min_component_area_mm2: float = 100.0
    mask_max_components: int = 80
    morph_close_iterations: int = 3
    mask_max_component_area_ratio: float = 0.28
    mask_bridge_break_kernel_px: int = 9
    glare_suppression_enabled: bool = True
    glare_suppression_l_cap: float = 220.0
    glare_rejection_enabled: bool = True
    glare_l_delta: float = 40.0
    glare_l_absolute_min: float = 200.0
    use_background_reference: bool = True
    bg_subtract_mode: BgSubtractMode = "intensity"
    bg_subtract_min_diff: int = 15
    bg_subtract_blur_kernel_px: int = 5
    bg_texture_min_diff: int = 12
    bg_texture_blur_kernel_px: int = 5


def _morph_kernel(size_px: int) -> np.ndarray:
    k = max(3, int(size_px) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _largest_component_area_ratio(mask: np.ndarray) -> float:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0.0
    max_area = max(int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels))
    return float(max_area) / float(max(1, mask.size))


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

    largest_ratio = _largest_component_area_ratio(mask)
    if largest_ratio > max_component_area_ratio + 0.15:
        return 0.0, 0, 1.0

    ratio_score = 1.0 - abs(ratio - 0.15)
    component_count, speckle_ratio = _mask_fragmentation_metrics(mask, min_component_area_px)
    if component_count > max_components:
        component_penalty = min(1.0, (component_count - max_components) / float(max_components))
    else:
        component_penalty = 0.0
    blob_penalty = 0.0
    if largest_ratio > max_component_area_ratio:
        blob_penalty = min(
            1.0,
            (largest_ratio - max_component_area_ratio) / max(0.05, max_component_area_ratio),
        )
    fragmentation = min(1.0, 0.4 * component_penalty + 0.3 * speckle_ratio + 0.3 * blob_penalty)
    return max(0.0, ratio_score * (1.0 - fragmentation)), component_count, fragmentation


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


def _score_candidate_mask(
    mask: np.ndarray,
    cfg: ShapeDetectorConfig,
    min_component_area_px: int,
    *,
    weight: float = 1.0,
) -> tuple[float, np.ndarray, int, float]:
    cleaned = _clean_mask(mask, cfg, min_component_area_px)
    score, component_count, fragmentation = _mask_score(
        cleaned,
        min_component_area_px=min_component_area_px,
        max_components=cfg.mask_max_components,
        max_component_area_ratio=cfg.mask_max_component_area_ratio,
    )
    return score * weight, cleaned, component_count, fragmentation


def _break_mask_bridges(mask: np.ndarray, cfg: ShapeDetectorConfig) -> np.ndarray:
    kernel = _morph_kernel(cfg.mask_bridge_break_kernel_px)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    for extra in (4, 8):
        if _largest_component_area_ratio(mask) <= cfg.mask_max_component_area_ratio:
            break
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            _morph_kernel(cfg.mask_bridge_break_kernel_px + extra),
            iterations=1,
        )
    return mask


def _split_contour_by_distance_peaks(
    contour: np.ndarray,
    mask: np.ndarray,
    *,
    min_area_px: float,
) -> list[np.ndarray]:
    x, y, w, h = cv2.boundingRect(contour)
    pad = 12
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + w + pad)
    y1 = min(mask.shape[0], y + h + pad)
    roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour - np.array([[x0, y0]])
    cv2.drawContours(roi, [shifted], -1, 255, -1)
    roi = cv2.bitwise_and(roi, mask[y0:y1, x0:x1])

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() <= 1.0:
        return []

    peak_kernel = _morph_kernel(15)
    eroded = cv2.erode(dist, peak_kernel)
    peaks = np.uint8((dist >= eroded) & (dist > 0.3 * dist.max())) * 255
    num_labels, labels = cv2.connectedComponents(peaks)
    if num_labels < 3:
        return []

    sub_contours: list[np.ndarray] = []
    grow_kernel = _morph_kernel(21)
    for label in range(1, num_labels):
        seed = np.uint8(labels == label) * 255
        grown = cv2.dilate(seed, grow_kernel, iterations=2)
        grown = cv2.bitwise_and(grown, roi)
        cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) >= min_area_px:
                sub_contours.append(cnt + np.array([[x0, y0]]))
    return sub_contours


def _suppress_specular_glare_bgr(
    bgr: np.ndarray,
    bed_l: float,
    cfg: ShapeDetectorConfig,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    if not cfg.glare_suppression_enabled:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    cap = max(cfg.glare_suppression_l_cap, bed_l + cfg.glare_l_delta + 30.0)
    replace = min(255.0, bed_l + 35.0)
    roi_l = l_channel[y0:y1, x0:x1]
    hot = roi_l > cap
    roi_l[hot] = replace
    l_channel[y0:y1, x0:x1] = roi_l
    lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _estimate_bed_l(l_channel: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    h, w = l_channel.shape[:2]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    strip = max(4, int(min(x1 - x0, y1 - y0) * 0.05))
    border_samples: list[np.ndarray] = []
    if y0 + strip < y1:
        border_samples.append(l_channel[y0 : y0 + strip, x0:x1].reshape(-1))
    if y1 - strip > y0:
        border_samples.append(l_channel[y1 - strip : y1, x0:x1].reshape(-1))
    if x0 + strip < x1:
        border_samples.append(l_channel[y0:y1, x0 : x0 + strip].reshape(-1))
    if x1 - strip > x0:
        border_samples.append(l_channel[y0:y1, x1 - strip : x1].reshape(-1))
    if border_samples:
        values = np.concatenate(border_samples)
    else:
        values = l_channel[y0:y1, x0:x1].reshape(-1)
    return float(np.percentile(values, 20))


def _contour_l_stats(contour: np.ndarray, l_channel: np.ndarray) -> tuple[float, float, float]:
    stats_mask = np.zeros(l_channel.shape, dtype=np.uint8)
    cv2.drawContours(stats_mask, [contour], -1, 255, -1)
    values = l_channel[stats_mask > 0]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.max(values))


def _is_specular_glare(
    contour: np.ndarray,
    l_channel: np.ndarray,
    bed_l: float,
    cfg: ShapeDetectorConfig,
) -> bool:
    if not cfg.glare_rejection_enabled:
        return False
    mean_l, std_l, peak_l = _contour_l_stats(contour, l_channel)
    if mean_l < cfg.glare_l_absolute_min:
        return False
    if mean_l < bed_l + cfg.glare_l_delta:
        return False
    area_px = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    box_area = float(rect[1][0] * rect[1][1])
    fill_ratio = float(area_px / box_area) if box_area > 0 else 0.0
    if peak_l >= 245.0:
        return True
    return std_l > 25.0 or fill_ratio < 0.55


def _fill_contour_if_ring(
    contour: np.ndarray,
    cfg: ShapeDetectorConfig,
    solidity: float,
) -> np.ndarray:
    circularity = _circularity(contour)
    if circularity < cfg.circularity_threshold or solidity >= cfg.min_solidity:
        return contour
    fill_mask = np.zeros(
        (int(cv2.boundingRect(contour)[3] + 4), int(cv2.boundingRect(contour)[2] + 4)),
        dtype=np.uint8,
    )
    x, y, w, h = cv2.boundingRect(contour)
    shifted = contour - np.array([[x - 2, y - 2]])
    cv2.drawContours(fill_mask, [shifted], -1, 255, -1)
    filled_contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not filled_contours:
        return contour
    filled = max(filled_contours, key=cv2.contourArea)
    return filled + np.array([[x - 2, y - 2]])


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def _normalize_rect(size_px: tuple[float, float], angle_deg: float) -> tuple[float, float, float]:
    width, height = float(size_px[0]), float(size_px[1])
    angle = float(angle_deg)
    if width < height:
        width, height = height, width
        angle += 90.0
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return width, height, angle


def _oriented_corners(
    center: tuple[float, float], width: float, height: float, angle_deg: float
) -> list[tuple[float, float]]:
    rect = ((center[0], center[1]), (width, height), angle_deg)
    box = cv2.boxPoints(rect)
    return [(float(x), float(y)) for x, y in box]


def _bbox_from_points(points: list[tuple[float, float]]) -> BoundingBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))


def _contour_to_polygon(contour: np.ndarray, max_points: int = 64) -> list[tuple[float, float]]:
    epsilon = 0.005 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    points = [(float(x), float(y)) for x, y in simplified.reshape(-1, 2)]
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return points


class ShapeDetector:
    def __init__(self, config: ShapeDetectorConfig | None = None) -> None:
        self.config = config or ShapeDetectorConfig()

    def _contours_from_mask(
        self,
        warped_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        pixels_per_mm: float,
        width_mm: float,
        height_mm: float,
    ) -> MaskGeometryResult:
        ppm = pixels_per_mm
        cfg = self.config
        margin_px = cfg.roi_margin_mm * ppm
        h, w = warped_bgr.shape[:2]
        x0 = int(max(0, margin_px))
        y0 = int(max(0, margin_px))
        x1 = int(w - max(0, margin_px))
        y1 = int(h - max(0, margin_px))

        lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        bed_l = _estimate_bed_l(l_channel, x0, y0, x1, y1)

        if mask.size <= 1:
            mask = np.zeros(warped_bgr.shape[:2], dtype=np.uint8)
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_contour_count = len(contours)
        min_area_px = cfg.min_area_mm2 * ppm * ppm
        max_area_px = cfg.max_area_mm2 * ppm * ppm
        split_above_area_px = cfg.split_above_area_mm2 * ppm * ppm

        expanded_contours: list[np.ndarray] = []
        pre_rejected: list[ContourCandidate] = []
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px > split_above_area_px or area_px > max_area_px:
                splits = _split_contour_by_distance_peaks(
                    contour, mask, min_area_px=min_area_px
                )
                if len(splits) >= 2:
                    expanded_contours.extend(splits)
                    continue
                if area_px > split_above_area_px:
                    rect = cv2.minAreaRect(contour)
                    rw, rh = float(rect[1][0]), float(rect[1][1])
                    box_w_mm = max(rw, rh) / ppm
                    box_h_mm = min(rw, rh) / ppm
                    if (
                        box_w_mm > width_mm * cfg.max_object_span_ratio
                        or box_h_mm > height_mm * cfg.max_object_span_ratio
                    ):
                        pre_rejected.append(
                            ContourCandidate(contour=contour, kept=False, reject_reason="merged_blob")
                        )
                        continue
            expanded_contours.append(contour)

        candidates: list[ContourCandidate] = list(pre_rejected)
        objects: list[RawShapeDetection] = []
        glare_reject_count = 0

        for contour in expanded_contours:
            area_px = cv2.contourArea(contour)
            candidate = ContourCandidate(contour=contour)
            if area_px < min_area_px:
                candidate.kept = False
                candidate.reject_reason = "too_small"
                candidates.append(candidate)
                continue
            if area_px > max_area_px:
                candidate.kept = False
                candidate.reject_reason = "too_large"
                candidates.append(candidate)
                continue

            if _is_specular_glare(contour, l_channel, bed_l, cfg):
                candidate.kept = False
                candidate.reject_reason = "specular_glare"
                glare_reject_count += 1
                candidates.append(candidate)
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / hull_area) if hull_area > 0 else 0.0
            contour = _fill_contour_if_ring(contour, cfg, solidity)
            area_px = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / hull_area) if hull_area > 0 else 0.0

            rect = cv2.minAreaRect(contour)
            box_area = float(rect[1][0] * rect[1][1])
            extent = float(area_px / box_area) if box_area > 0 else 0.0
            if solidity < cfg.min_solidity:
                candidate.contour = contour
                candidate.kept = False
                candidate.reject_reason = "low_solidity"
                candidates.append(candidate)
                continue
            if extent < cfg.min_extent:
                candidate.contour = contour
                candidate.kept = False
                candidate.reject_reason = "low_extent"
                candidates.append(candidate)
                continue

            circularity = _circularity(contour)
            polygon = _contour_to_polygon(contour)
            (cx, cy), (rw, rh), angle = rect
            width_px, height_px, rotation_deg = _normalize_rect((rw, rh), angle)
            aspect = max(width_px, height_px) / max(1e-6, min(width_px, height_px))

            if circularity >= cfg.circularity_threshold:
                shape: ShapeKind = "circle"
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                diameter = 2.0 * radius
                width_px = height_px = diameter
                rotation_deg = 0.0
                confidence = min(1.0, circularity)
            elif aspect >= cfg.bracelet_min_aspect and circularity >= cfg.rounded_rect_circularity_min:
                shape = "rounded_rect"
                confidence = min(1.0, 0.5 * solidity + 0.5 * extent)
            else:
                shape = "rect"
                approx = cv2.approxPolyDP(
                    contour, cfg.approx_epsilon_ratio * cv2.arcLength(contour, True), True
                )
                vertex_bonus = 0.1 if 4 <= len(approx) <= 6 else 0.0
                confidence = min(1.0, 0.45 * solidity + 0.45 * extent + vertex_bonus)

            oriented = _oriented_corners((cx, cy), width_px, height_px, rotation_deg)
            candidate.contour = contour
            objects.append(
                RawShapeDetection(
                    shape=shape,
                    confidence=float(confidence),
                    center_px=(float(cx), float(cy)),
                    width_px=float(width_px),
                    height_px=float(height_px),
                    rotation_deg=float(rotation_deg),
                    bbox_px=_bbox_from_points(polygon),
                    oriented_box_px=oriented,
                    segmentation_polygon_px=polygon,
                )
            )
            candidates.append(candidate)

        return MaskGeometryResult(
            objects=objects,
            mask=mask,
            contours=candidates,
            glare_reject_count=glare_reject_count,
            raw_contour_count=raw_contour_count,
        )

    @staticmethod
    def from_mask(
        warped_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        pixels_per_mm: float,
        width_mm: float,
        height_mm: float,
        config: ShapeDetectorConfig | None = None,
    ) -> MaskGeometryResult:
        detector = ShapeDetector(config)
        return detector._contours_from_mask(
            warped_bgr,
            mask,
            pixels_per_mm=pixels_per_mm,
            width_mm=width_mm,
            height_mm=height_mm,
        )

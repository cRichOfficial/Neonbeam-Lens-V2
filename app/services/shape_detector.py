from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.common import BoundingBox
from app.schemas.shapes import ShapeKind, ShapeSource


@dataclass
class ContourCandidate:
    contour: np.ndarray
    kept: bool = True
    reject_reason: str | None = None


@dataclass
class RawShapeDetection:
    shape: ShapeKind
    confidence: float
    source: ShapeSource
    center_px: tuple[float, float]
    width_px: float
    height_px: float
    rotation_deg: float
    bbox_px: BoundingBox
    oriented_box_px: list[tuple[float, float]]
    segmentation_polygon_px: list[tuple[float, float]]


@dataclass
class ClassicalDetectionResult:
    objects: list[RawShapeDetection]
    mask: np.ndarray
    contours: list[ContourCandidate]
    mask_method: str
    best_confidence: float


@dataclass
class ShapeDetectorConfig:
    min_area_mm2: float = 400.0
    max_area_mm2: float = 80000.0
    min_solidity: float = 0.75
    min_extent: float = 0.35
    circularity_threshold: float = 0.82
    rounded_rect_circularity_min: float = 0.65
    bracelet_min_aspect: float = 6.0
    approx_epsilon_ratio: float = 0.02
    roi_margin_mm: float = 5.0


def _mask_score(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    fg = float(np.count_nonzero(mask))
    ratio = fg / float(mask.size)
    if ratio < 0.001 or ratio > 0.85:
        return 0.0
    return 1.0 - abs(ratio - 0.15)


def _extract_foreground_mask(bgr: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[str, np.ndarray, float]] = []

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
    )
    candidates.append(("adaptive", adaptive, _mask_score(adaptive)))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("otsu", otsu, _mask_score(otsu)))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, l_mask = cv2.threshold(lab[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("lab_l", l_mask, _mask_score(l_mask)))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 25, 255, cv2.THRESH_BINARY)
    candidates.append(("hsv_sat", sat_mask, _mask_score(sat_mask)))

    edges = cv2.Canny(gray, 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    candidates.append(("canny", closed, _mask_score(closed) * 0.8))

    method, mask, _ = max(candidates, key=lambda item: item[2])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask, method


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

    def detect(
        self,
        warped_bgr: np.ndarray,
        *,
        pixels_per_mm: float,
        width_mm: float,
        height_mm: float,
        source: ShapeSource = "classical",
    ) -> ClassicalDetectionResult:
        ppm = pixels_per_mm
        cfg = self.config
        margin_px = cfg.roi_margin_mm * ppm
        h, w = warped_bgr.shape[:2]
        x0 = int(max(0, margin_px))
        y0 = int(max(0, margin_px))
        x1 = int(w - max(0, margin_px))
        y1 = int(h - max(0, margin_px))

        mask, method = _extract_foreground_mask(warped_bgr)
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_px = cfg.min_area_mm2 * ppm * ppm
        max_area_px = cfg.max_area_mm2 * ppm * ppm

        candidates: list[ContourCandidate] = []
        objects: list[RawShapeDetection] = []

        for contour in contours:
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

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area_px / hull_area) if hull_area > 0 else 0.0
            rect = cv2.minAreaRect(contour)
            box_area = float(rect[1][0] * rect[1][1])
            extent = float(area_px / box_area) if box_area > 0 else 0.0
            if solidity < cfg.min_solidity:
                candidate.kept = False
                candidate.reject_reason = "low_solidity"
                candidates.append(candidate)
                continue
            if extent < cfg.min_extent:
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
            objects.append(
                RawShapeDetection(
                    shape=shape,
                    confidence=float(confidence),
                    source=source,
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

        best_confidence = max((obj.confidence for obj in objects), default=0.0)
        return ClassicalDetectionResult(
            objects=objects,
            mask=mask,
            contours=candidates,
            mask_method=method,
            best_confidence=best_confidence,
        )

    @staticmethod
    def from_mask(
        warped_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        pixels_per_mm: float,
        width_mm: float,
        height_mm: float,
        source: ShapeSource = "fastsam",
        config: ShapeDetectorConfig | None = None,
    ) -> ClassicalDetectionResult:
        detector = ShapeDetector(config)
        ppm = pixels_per_mm
        cfg = detector.config
        min_area_px = cfg.min_area_mm2 * ppm * ppm
        max_area_px = cfg.max_area_mm2 * ppm * ppm

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[ContourCandidate] = []
        objects: list[RawShapeDetection] = []

        for contour in contours:
            area_px = cv2.contourArea(contour)
            candidate = ContourCandidate(contour=contour)
            if area_px < min_area_px or area_px > max_area_px:
                candidate.kept = False
                candidate.reject_reason = "area"
                candidates.append(candidate)
                continue
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
            width_px, height_px, rotation_deg = _normalize_rect((rw, rh), angle)
            polygon = _contour_to_polygon(contour)
            circularity = _circularity(contour)
            aspect = max(width_px, height_px) / max(1e-6, min(width_px, height_px))
            if circularity >= cfg.circularity_threshold:
                shape: ShapeKind = "circle"
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                width_px = height_px = 2.0 * radius
                rotation_deg = 0.0
            elif aspect >= cfg.bracelet_min_aspect:
                shape = "rounded_rect"
            else:
                shape = "rect"
            objects.append(
                RawShapeDetection(
                    shape=shape,
                    confidence=min(1.0, max(0.4, circularity)),
                    source=source,
                    center_px=(float(cx), float(cy)),
                    width_px=float(width_px),
                    height_px=float(height_px),
                    rotation_deg=float(rotation_deg),
                    bbox_px=_bbox_from_points(polygon),
                    oriented_box_px=_oriented_corners((cx, cy), width_px, height_px, rotation_deg),
                    segmentation_polygon_px=polygon,
                )
            )
            candidates.append(candidate)

        best_confidence = max((obj.confidence for obj in objects), default=0.0)
        return ClassicalDetectionResult(
            objects=objects,
            mask=mask,
            contours=candidates,
            mask_method="fastsam",
            best_confidence=best_confidence,
        )

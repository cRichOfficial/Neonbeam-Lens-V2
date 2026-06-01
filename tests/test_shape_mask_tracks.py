"""Mask track tests — bg_subtract for FastSAM post-filter."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.shape_detector import ShapeDetectorConfig
from app.services.shape_mask_tracks import track_bg_subtract


def _white_bed(width_px: int, height_px: int) -> np.ndarray:
    return np.full((height_px, width_px, 3), 240, dtype=np.uint8)


def _synthetic_honeycomb_bed(
    width_px: int,
    height_px: int,
    *,
    cell_px: int = 24,
    base_color: tuple[int, int, int] = (25, 25, 25),
) -> np.ndarray:
    bed = np.full((height_px, width_px, 3), base_color, dtype=np.uint8)
    line_color = (12, 12, 12)
    for x in range(0, width_px, cell_px):
        cv2.line(bed, (x, 0), (x, height_px), line_color, 1)
    for y in range(0, height_px, cell_px):
        cv2.line(bed, (0, y), (width_px, y), line_color, 1)
    return bed


def _place_same_color_object(
    bed: np.ndarray,
    *,
    color: tuple[int, int, int] = (25, 25, 25),
    size_px: int = 180,
) -> np.ndarray:
    current = bed.copy()
    height_px, width_px = current.shape[:2]
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        current,
        (cx - size_px // 2, cy - size_px // 2),
        (cx + size_px // 2, cy + size_px // 2),
        color,
        -1,
    )
    return current


def test_bg_subtract_glare_exclusion_keeps_dark_object() -> None:
    """White-bed reference must not zero strong diff at dark-object pixels."""
    ppm = 2.0
    width_px = height_px = 800
    reference = _white_bed(width_px, height_px)
    current = reference.copy()
    size = int(90 * ppm)
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        current,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (30, 30, 30),
        -1,
    )
    current[10:20, 10:20] = (255, 255, 255)

    cfg = ShapeDetectorConfig()
    track = track_bg_subtract(
        current,
        reference,
        cfg,
        pixels_per_mm=ppm,
        bed_l=200.0,
    )
    assert track is not None
    assert np.count_nonzero(track.mask) > 0


def test_bg_subtract_finds_dark_rectangle_on_white_bed() -> None:
    ppm = 2.0
    width_px = height_px = 800
    reference = _white_bed(width_px, height_px)
    current = reference.copy()
    size = int(90 * ppm)
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        current,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (30, 30, 30),
        -1,
    )

    cfg = ShapeDetectorConfig()
    track = track_bg_subtract(
        current,
        reference,
        cfg,
        pixels_per_mm=ppm,
        bed_l=200.0,
    )
    assert track is not None
    assert np.count_nonzero(track.mask) > 0


def test_intensity_same_color_honeycomb_has_low_interior_diff() -> None:
    from app.services.shape_mask_tracks import compute_intensity_diff

    width_px = height_px = 800
    reference = _synthetic_honeycomb_bed(width_px, height_px)
    current = _place_same_color_object(reference, size_px=200)
    cfg = ShapeDetectorConfig(bg_subtract_mode="intensity")
    diff = compute_intensity_diff(current, reference, cfg, bed_l=25.0)
    cy, cx = height_px // 2, width_px // 2
    half = 70
    interior = diff[cy - half : cy + half, cx - half : cx + half]
    assert int(np.max(interior)) < 12


def test_fused_finds_same_color_object_on_honeycomb() -> None:
    ppm = 2.0
    width_px = height_px = 800
    reference = _synthetic_honeycomb_bed(width_px, height_px)
    current = _place_same_color_object(reference, size_px=200)
    cfg = ShapeDetectorConfig(bg_subtract_mode="fused", bg_texture_min_diff=8)
    track = track_bg_subtract(
        current,
        reference,
        cfg,
        pixels_per_mm=ppm,
        bed_l=25.0,
        mode="fused",
    )
    assert track is not None
    assert track.texture_diff is not None
    assert track.mode == "fused"
    cy, cx = height_px // 2, width_px // 2
    half = 100
    roi = track.mask[cy - half : cy + half, cx - half : cx + half]
    assert int(np.count_nonzero(roi)) > 1000

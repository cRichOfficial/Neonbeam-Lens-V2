"""Mask track tests — bg_subtract for FastSAM post-filter."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.shape_detector import ShapeDetectorConfig
from app.services.shape_mask_tracks import track_bg_subtract


def _white_bed(width_px: int, height_px: int) -> np.ndarray:
    return np.full((height_px, width_px, 3), 240, dtype=np.uint8)


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

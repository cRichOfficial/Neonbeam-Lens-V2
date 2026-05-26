"""FastSAM bg_subtract post-filter tests."""

from __future__ import annotations

import cv2
import numpy as np

from app.services.shape_fastsam_filter import (
    filter_fastsam_masks,
    is_bg_mask_sane,
)


def _mask_at(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def test_is_bg_mask_sane_rejects_full_frame() -> None:
    mask = np.full((100, 100), 255, dtype=np.uint8)
    assert is_bg_mask_sane(mask, max_foreground_ratio=0.45) is False


def test_is_bg_mask_sane_accepts_local_blobs() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    assert is_bg_mask_sane(mask, max_foreground_ratio=0.45) is True


def test_filter_drops_small_speckle_masks() -> None:
    frame_w, frame_h = 640, 640
    speckle = _mask_at(5, 5, 20, 20, frame_w, frame_h)
    object_mask = _mask_at(200, 200, 120, 80, frame_w, frame_h)
    bg = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(bg, (180, 180), (340, 300), 255, -1)

    kept, detail = filter_fastsam_masks(
        [speckle, object_mask],
        bg,
        min_overlap=0.25,
        min_area_px=800,
    )
    assert len(kept) == 1
    assert kept[0].shape == object_mask.shape
    assert np.array_equal(kept[0], object_mask)
    assert "dropped area" in detail


def test_filter_drops_masks_outside_bg_foreground() -> None:
    frame_w, frame_h = 640, 640
    inside = _mask_at(220, 220, 100, 80, frame_w, frame_h)
    outside = _mask_at(20, 20, 100, 80, frame_w, frame_h)
    bg = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(bg, (200, 200), (340, 320), 255, -1)

    kept, detail = filter_fastsam_masks(
        [inside, outside],
        bg,
        min_overlap=0.25,
        min_area_px=100,
    )
    assert len(kept) == 1
    assert np.array_equal(kept[0], inside)
    assert "overlap" in detail


def test_filter_skips_bg_when_mask_insane() -> None:
    frame_w, frame_h = 640, 640
    mask = _mask_at(50, 50, 120, 120, frame_w, frame_h)
    insane_bg = np.full((frame_h, frame_w), 255, dtype=np.uint8)

    kept, detail = filter_fastsam_masks(
        [mask],
        insane_bg,
        min_overlap=0.25,
        min_area_px=100,
    )
    assert len(kept) == 1
    assert "bg filter skipped" in detail

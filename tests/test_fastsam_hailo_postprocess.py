"""FastSAM Hailo YOLOv8-seg postprocess tests."""

from __future__ import annotations

import numpy as np

from app.services.fastsam_hailo_postprocess import (
    _process_masks,
    order_yolov8_seg_endnodes,
    postprocess_yolov8_seg,
)


def _synthetic_endnodes() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for grid in (20, 40, 80):
        outputs[f"box_{grid}"] = np.random.randn(grid, grid, 64).astype(np.float32)
        outputs[f"score_{grid}"] = np.random.randn(grid, grid, 1).astype(np.float32)
        outputs[f"coeff_{grid}"] = np.random.randn(grid, grid, 32).astype(np.float32)
    outputs["proto"] = np.random.randn(160, 160, 32).astype(np.float32)
    return outputs


def test_order_yolov8_seg_endnodes() -> None:
    ordered = order_yolov8_seg_endnodes(_synthetic_endnodes())
    assert len(ordered) == 10
    assert ordered[-1].shape == (160, 160, 32)
    assert ordered[0].shape[2] == 64
    assert ordered[1].shape[2] == 1
    assert ordered[2].shape[2] == 32


def _synthetic_endnodes_chw() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for grid in (20, 40, 80):
        outputs[f"box_{grid}"] = np.random.randn(64, grid, grid).astype(np.float32)
        outputs[f"score_{grid}"] = np.random.randn(1, grid, grid).astype(np.float32)
        outputs[f"coeff_{grid}"] = np.random.randn(32, grid, grid).astype(np.float32)
    outputs["proto"] = np.random.randn(32, 160, 160).astype(np.float32)
    return outputs


def test_order_yolov8_seg_endnodes_chw_layout() -> None:
    ordered = order_yolov8_seg_endnodes(_synthetic_endnodes_chw())
    assert len(ordered) == 10
    assert ordered[0].shape == (20, 20, 64)
    assert ordered[-1].shape == (160, 160, 32)


def test_process_masks_crops_after_upscale() -> None:
    """Large image-space boxes must not be cropped on the 160x160 proto grid."""
    proto = np.zeros((160, 160, 32), dtype=np.float32)
    proto[40:120, 60:100, :] = 2.0
    coeffs = np.zeros((1, 32), dtype=np.float32)
    coeffs[0, :] = 1.0
    boxes = np.array([[240, 160, 400, 480]], dtype=np.float32)
    masks = _process_masks(proto, coeffs, boxes, (640, 640))
    assert masks is not None
    binary = (masks[0] > 0.5).astype(np.uint8)
    assert np.count_nonzero(binary) > 1000
    ys, xs = np.where(binary > 0)
    assert xs.min() >= 240
    assert ys.min() >= 160


def test_postprocess_returns_list_of_masks() -> None:
    outputs = _synthetic_endnodes()
    outputs["score_20"] = np.full((20, 20, 1), 5.0, dtype=np.float32)
    outputs["score_40"] = np.full((40, 40, 1), 5.0, dtype=np.float32)
    outputs["score_80"] = np.full((80, 80, 1), 5.0, dtype=np.float32)
    masks, detail = postprocess_yolov8_seg(
        outputs,
        input_size=(640, 640),
        score_threshold=0.01,
        nms_iou_threshold=0.45,
    )
    assert isinstance(masks, list)
    assert detail
    for mask in masks:
        assert mask.shape == (640, 640)
        assert mask.dtype == np.uint8

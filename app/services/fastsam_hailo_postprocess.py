"""Host-side YOLOv8-seg postprocess for Hailo fast_sam_s.hef raw outputs."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[fastsam]"


def _log_mask_stats(index: int, mask: np.ndarray) -> str:
    nz = int(np.count_nonzero(mask))
    if nz == 0:
        return f"mask[{index}] empty"
    ys, xs = np.where(mask > 0)
    return (
        f"mask[{index}] area={nz}px "
        f"bbox=({int(xs.min())},{int(ys.min())})-({int(xs.max())},{int(ys.max())})"
    )


def _log_box_scores(boxes: np.ndarray, scores: np.ndarray, limit: int = 5) -> None:
    if boxes.size == 0:
        logger.info("%s detections: none", _LOG_PREFIX)
        return
    count = min(limit, len(scores))
    lines = []
    for index in range(count):
        x1, y1, x2, y2 = boxes[index]
        lines.append(
            f"#{index} score={scores[index]:.3f} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
        )
    logger.info("%s top detections: %s", _LOG_PREFIX, "; ".join(lines))


_REG_MAX = 15
_STRIDES = (32, 16, 8)
_NUM_MASKS = 32
_NUM_CLASSES = 1


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _to_hwc(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor after batch squeeze, got {arr.shape}")

    a, b, c = arr.shape
    channel_last = {1, 32, 64, 80, 116}
    # HWC: square spatial grid, channels in last dim (20x20x64)
    if a == b and c in channel_last:
        return arr
    # CHW: channels first on a square grid (64x20x20, 32x160x160)
    if a in channel_last and b == c:
        return np.transpose(arr, (1, 2, 0))
    # Fallback for unexpected layouts
    if a <= 128 and a < min(b, c):
        return np.transpose(arr, (1, 2, 0))
    return arr


def order_yolov8_seg_endnodes(outputs: dict[str, np.ndarray] | list | tuple) -> list[np.ndarray]:
    """Order raw HEF outputs into [box, score, coeff] x3 scales + proto."""
    if isinstance(outputs, (list, tuple)):
        return [_to_hwc(np.asarray(item)) for item in outputs]

    if not isinstance(outputs, dict):
        raise TypeError(f"Unsupported FastSAM Hailo output type: {type(outputs)!r}")

    hwc_tensors = [_to_hwc(value) for value in outputs.values()]
    proto: np.ndarray | None = None
    grouped: dict[tuple[int, int], list[np.ndarray]] = {}

    for tensor in hwc_tensors:
        h, w, channels = tensor.shape
        if max(h, w) == 160 and channels == _NUM_MASKS:
            proto = tensor
            continue
        if channels not in (1, _NUM_MASKS, 64):
            logger.debug("Skipping unexpected FastSAM output shape %s", tensor.shape)
            continue
        grouped.setdefault((h, w), []).append(tensor)

    if proto is None:
        raise ValueError("FastSAM Hailo output missing 160x160 proto tensor")

    ordered: list[np.ndarray] = []
    for grid in sorted(grouped.keys()):
        items = grouped[grid]
        by_channels = {item.shape[2]: item for item in items}
        for channels in (64, 1, _NUM_MASKS):
            if channels not in by_channels:
                raise ValueError(f"Missing FastSAM output for grid {grid} with {channels} channels")
            ordered.append(by_channels[channels])

    ordered.append(proto)
    if len(ordered) != 10:
        raise ValueError(f"Expected 10 FastSAM endnodes, got {len(ordered)}")
    return ordered


def _yolov8_decode_boxes(raw_boxes: list[np.ndarray], image_dims: tuple[int, int]) -> np.ndarray:
    boxes: np.ndarray | None = None
    for raw, stride in zip(raw_boxes, _STRIDES, strict=True):
        shape = [int(image_dims[0] / stride), int(image_dims[1] / stride)]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        reg_range = np.arange(_REG_MAX + 1)
        box_distribute = np.reshape(
            raw, (-1, raw.shape[0] * raw.shape[1], 4, _REG_MAX + 1)
        )
        box_distance = _softmax(box_distribute)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1) * stride

        box_distance = np.concatenate(
            [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1
        )
        decode_box = np.expand_dims(center, axis=0) + box_distance
        xmin = decode_box[:, :, 0]
        ymin = decode_box[:, :, 1]
        xmax = decode_box[:, :, 2]
        ymax = decode_box[:, :, 3]
        xywh = np.transpose(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin],
            (1, 2, 0),
        )
        boxes = xywh if boxes is None else np.concatenate([boxes, xywh], axis=1)
    return boxes


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.copy(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int) -> list[int]:
    if boxes.size == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
        top_k=max_det,
    )
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        return [int(i) for i in indices.reshape(-1)]
    return [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in indices]


def _crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    n_masks, _, _ = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(np.maximum(integer_boxes, 0), 4, axis=1)
    for index in range(n_masks):
        masks[index, : y1[index, 0], :] = 0
        masks[index, y2[index, 0] :, :] = 0
        masks[index, :, : x1[index, 0]] = 0
        masks[index, :, x2[index, 0] :] = 0
    return masks


def _process_masks(
    protos: np.ndarray,
    mask_coeffs: np.ndarray,
    boxes_xyxy: np.ndarray,
    image_dims: tuple[int, int],
) -> np.ndarray | None:
    mh, mw, channels = protos.shape
    masks = _sigmoid(mask_coeffs @ protos.reshape((-1, channels)).T).reshape((-1, mh, mw))
    if masks.shape[0] == 0:
        return None
    # Boxes are in input-image pixels; upscale masks before cropping (proto grid is 160x160).
    resized = cv2.resize(
        np.transpose(masks, (1, 2, 0)),
        (image_dims[1], image_dims[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    if resized.ndim == 2:
        resized = resized[..., np.newaxis]
    resized = np.transpose(resized, (2, 0, 1))
    return _crop_masks(resized, boxes_xyxy)


def postprocess_yolov8_seg(
    outputs: dict[str, np.ndarray] | list | tuple,
    *,
    input_size: tuple[int, int],
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 30,
    mask_threshold: float = 0.5,
) -> tuple[list[np.ndarray], str]:
    """Decode Hailo fast_sam_s / YOLOv8-seg endnodes into binary masks (input_size = W, H)."""
    logger.info(
        "%s postprocess start input=%sx%s score>=%.2f nms=%.2f mask>=%.2f",
        _LOG_PREFIX,
        input_size[0],
        input_size[1],
        score_threshold,
        nms_iou_threshold,
        mask_threshold,
    )
    endnodes = order_yolov8_seg_endnodes(outputs)
    logger.info(
        "%s endnode shapes: %s",
        _LOG_PREFIX,
        ", ".join(f"{node.shape}" for node in endnodes),
    )
    image_dims = (input_size[1], input_size[0])

    raw_boxes = endnodes[0:7:3]
    scores = [
        np.reshape(_sigmoid(score), (-1, score.shape[0] * score.shape[1], _NUM_CLASSES))
        for score in endnodes[1:8:3]
    ]
    scores = np.concatenate(scores, axis=1)
    coeffs = [
        np.reshape(coeff, (-1, coeff.shape[0] * coeff.shape[1], _NUM_MASKS))
        for coeff in endnodes[2:9:3]
    ]
    coeffs = np.concatenate(coeffs, axis=1)
    proto = endnodes[9]

    decoded_boxes = _yolov8_decode_boxes(raw_boxes, image_dims)
    objectness = np.ones((scores.shape[0], scores.shape[1], 1))
    scores_obj = np.concatenate([objectness, scores], axis=-1)
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)

    batch = predictions[0]
    max_score = float(np.max(batch[:, 5])) if batch.size else 0.0
    logger.info(
        "%s raw proposals=%d max_class_score=%.3f",
        _LOG_PREFIX,
        batch.shape[0],
        max_score,
    )
    candidates = batch[batch[:, 4] > score_threshold]
    if candidates.size == 0:
        return [], f"no candidates above {score_threshold:.2f} (max score={max_score:.3f})"

    candidates[:, 5:] *= candidates[:, 4:5]
    boxes = _xywh_to_xyxy(candidates[:, :4])
    conf = candidates[:, 5:6]
    mask_coeffs = candidates[:, 6:]
    class_scores = conf.squeeze(1)
    keep = class_scores > score_threshold
    boxes = boxes[keep]
    class_scores = class_scores[keep]
    mask_coeffs = mask_coeffs[keep]
    if boxes.size == 0:
        return [], f"no class scores above {score_threshold:.2f}"

    order = np.argsort(class_scores)[::-1]
    boxes = boxes[order]
    class_scores = class_scores[order]
    mask_coeffs = mask_coeffs[order]

    logger.info(
        "%s pre-nms candidates=%d (score>=%.2f)",
        _LOG_PREFIX,
        len(class_scores),
        score_threshold,
    )
    _log_box_scores(boxes, class_scores)

    keep_indices = _nms_xyxy(boxes, class_scores, nms_iou_threshold, max_detections)
    if not keep_indices:
        return [], f"NMS removed all {len(class_scores)} candidates"

    boxes = boxes[keep_indices]
    mask_coeffs = mask_coeffs[keep_indices]
    scores_kept = class_scores[keep_indices]
    logger.info("%s post-nms kept=%d", _LOG_PREFIX, len(keep_indices))
    _log_box_scores(boxes, scores_kept)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_dims[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_dims[0])
    masks = _process_masks(proto, mask_coeffs, boxes, image_dims)
    if masks is None:
        return [], f"mask decode failed for {len(keep_indices)} detections"

    binary_masks: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        binary = (mask > mask_threshold).astype(np.uint8) * 255
        nz = int(np.count_nonzero(binary))
        if nz > 0:
            binary_masks.append(binary)
            logger.info("%s %s", _LOG_PREFIX, _log_mask_stats(index, binary))
        else:
            logger.info(
                "%s mask[%d] empty after threshold (max=%.3f box=%s)",
                _LOG_PREFIX,
                index,
                float(np.max(mask)) if mask.size else 0.0,
                tuple(int(v) for v in boxes[index]),
            )
    note = f"{len(binary_masks)} masks from {len(keep_indices)} detections"
    logger.info("%s postprocess done: %s", _LOG_PREFIX, note)
    return binary_masks, note

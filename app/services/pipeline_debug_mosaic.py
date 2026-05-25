from __future__ import annotations

import cv2
import numpy as np

MOSAIC_BG = (26, 26, 26)
HEADER_HEIGHT = 28


def compose_stage_mosaic(
    stages: list[tuple[str, np.ndarray]],
    *,
    max_width_px: int = 1920,
    max_height_px: int = 1080,
    columns: int = 3,
) -> np.ndarray:
    if not stages:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    count = len(stages)
    columns = max(1, columns)
    rows = int(np.ceil(count / columns))
    cell_w = max(1, max_width_px // columns)
    cell_h = max(1, (max_height_px - rows * HEADER_HEIGHT) // rows)
    image_h = max(1, cell_h - HEADER_HEIGHT)

    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    canvas[:] = MOSAIC_BG

    for index, (label, image) in enumerate(stages):
        row = index // columns
        col = index % columns
        x0 = col * cell_w
        y0 = row * cell_h

        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            label,
            (x0 + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        bgr = _ensure_bgr(image)
        ih, iw = bgr.shape[:2]
        scale = min((cell_w - 8) / max(1, iw), image_h / max(1, ih))
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        scaled = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        tile = np.zeros((image_h, cell_w - 8, 3), dtype=np.uint8)
        tile[:] = MOSAIC_BG
        offset_x = (tile.shape[1] - new_w) // 2
        offset_y = (tile.shape[0] - new_h) // 2
        tile[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = scaled
        canvas[y0 + HEADER_HEIGHT : y0 + HEADER_HEIGHT + image_h, x0 + 4 : x0 + 4 + tile.shape[1]] = tile

    return canvas


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", _ensure_bgr(image), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode debug JPEG")
    return encoded.tobytes()


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

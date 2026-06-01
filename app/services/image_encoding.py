from __future__ import annotations

import cv2
import numpy as np


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert in-memory RGB888 frames to OpenCV BGR."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def encode_jpeg_rgb(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an in-memory RGB frame as JPEG (OpenCV imencode expects BGR)."""
    bgr = rgb_to_bgr(frame)
    return encode_jpeg_bgr(bgr, quality=quality)


def encode_jpeg_bgr(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an already-BGR frame as JPEG."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()

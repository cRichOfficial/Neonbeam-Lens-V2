from __future__ import annotations

import cv2
import numpy as np

from app.schemas.calibration import AprilTagPngRequest
from app.services.apriltag_marker import generate_tag36h11_image

PRINT_DPI = 300


def _mm_to_px(mm: float) -> int:
    return max(1, int(round(mm * PRINT_DPI / 25.4)))


def generate_apriltag_png(request: AprilTagPngRequest) -> bytes:
    if request.family != "tag36h11":
        raise ValueError(f"Unsupported AprilTag family: {request.family}")

    total_mm = request.size_mm + (2 * request.safe_zone_mm)
    total_px = _mm_to_px(total_mm)
    tag_px = _mm_to_px(request.size_mm)
    safe_px = _mm_to_px(request.safe_zone_mm)
    border_px = max(1, _mm_to_px(0.2))

    canvas = np.full((total_px, total_px, 3), 255, dtype=np.uint8)
    tag = generate_tag36h11_image(request.tag_id, tag_px)
    canvas[safe_px : safe_px + tag_px, safe_px : safe_px + tag_px] = tag
    cv2.rectangle(canvas, (0, 0), (total_px - 1, total_px - 1), (0, 0, 0), border_px)

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode AprilTag PNG")
    return encoded.tobytes()

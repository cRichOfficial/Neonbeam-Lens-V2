from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException

TAG36H11_MARKER_COUNT = 587


def validate_tag36h11_id(tag_id: int) -> None:
    if tag_id < 0 or tag_id >= TAG36H11_MARKER_COUNT:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"tag_id must be in range 0–{TAG36H11_MARKER_COUNT - 1} for tag36h11",
                "tag_id": tag_id,
            },
        )


def generate_tag36h11_image(tag_id: int, pixel_size: int) -> np.ndarray:
    validate_tag36h11_id(tag_id)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    try:
        image = cv2.aruco.generateImageMarker(dictionary, tag_id, pixel_size, borderBits=1)
    except cv2.error as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": f"Failed to generate tag36h11 marker for id {tag_id}", "error": str(exc)},
        ) from exc
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

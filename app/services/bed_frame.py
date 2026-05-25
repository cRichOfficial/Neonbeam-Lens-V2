from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from app.config import BedConfig, BedFrameConfig

# pupil-apriltags ideal tag corners in tag y-up frame: UL, UR, LR, LL
PUPIL_IDEAL_TAG_CORNERS = np.array(
    [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]],
    dtype=np.float32,
)


def frame_description(bed: BedFrameConfig) -> str:
    return f"origin={bed.origin}, y_axis={bed.y_axis}"


def bed_center_mm(bed: BedConfig) -> tuple[float, float]:
    return bed.width_mm / 2.0, bed.height_mm / 2.0


def bed_boundary_corners_mm(bed: BedConfig) -> np.ndarray:
    w, h = bed.width_mm, bed.height_mm
    return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)


def _ideal_to_bed_offset_unit(x_ideal: float, y_ideal: float, y_axis: Literal["up", "down"]) -> tuple[float, float]:
    if y_axis == "up":
        return x_ideal, y_ideal
    return x_ideal, -y_ideal


def tag_corner_offsets_mm(
    size_mm: float,
    bed: BedConfig,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    half = size_mm / 2.0
    offsets: list[list[float]] = []
    for x_ideal, y_ideal in PUPIL_IDEAL_TAG_CORNERS:
        bed_x, bed_y = _ideal_to_bed_offset_unit(float(x_ideal), float(y_ideal), bed.y_axis)
        offsets.append([bed_x * half, bed_y * half])
    points = np.array(offsets, dtype=np.float32)
    if rotation_deg:
        rad = math.radians(rotation_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        rotation = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        points = (rotation @ points.T).T
    return points


def physical_corners_for_tag(
    x_mm: float,
    y_mm: float,
    size_mm: float,
    bed: BedConfig,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    center = np.array([x_mm, y_mm], dtype=np.float32)
    return center + tag_corner_offsets_mm(size_mm, bed, rotation_deg)


def sort_corners_ccw(corners: np.ndarray, center: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    origin = np.asarray(center, dtype=np.float32).reshape(2)
    relative = points - origin
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    return points[np.argsort(angles)]


def align_tag_corner_pairs(
    image_corners: np.ndarray,
    image_center: np.ndarray,
    physical_corners: np.ndarray,
    physical_center: np.ndarray,
    *,
    score_error_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Match tag corners by angle sort, trying both windings and 90° cyclic shifts."""
    image_sorted = sort_corners_ccw(image_corners, image_center)
    physical_sorted = sort_corners_ccw(physical_corners, physical_center)

    best_image = image_sorted
    best_physical = physical_sorted
    best_error = float("inf")

    for reverse in (False, True):
        physical_winding = physical_sorted[::-1].copy() if reverse else physical_sorted
        for shift in range(4):
            physical_candidate = np.roll(physical_winding, shift, axis=0)
            if len(image_sorted) >= 4:
                homography, _ = cv2.findHomography(
                    image_sorted.astype(np.float32),
                    physical_candidate.astype(np.float32),
                    method=0,
                )
                if homography is None:
                    continue
                reprojected = cv2.perspectiveTransform(
                    image_sorted.reshape(-1, 1, 2).astype(np.float32),
                    homography,
                ).reshape(-1, 2)
                error = float(np.mean(np.linalg.norm(reprojected - physical_candidate, axis=1)))
            else:
                error = float(np.mean(np.linalg.norm(image_sorted - physical_candidate, axis=1)))

            if error < best_error:
                best_error = error
                best_image = image_sorted
                best_physical = physical_candidate

    if score_error_mm is not None:
        best_error = score_error_mm
    return best_image, best_physical, best_error

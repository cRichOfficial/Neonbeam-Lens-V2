from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

DistortionModelName = Literal["pinhole", "fisheye"]


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    distortion_model: DistortionModelName
    image_width: int
    image_height: int
    hfov_deg: float

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.reshape(-1).tolist(),
            "distortion_model": self.distortion_model,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "hfov_deg": self.hfov_deg,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CameraIntrinsics:
        return cls(
            camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            distortion_model=payload.get("distortion_model", "pinhole"),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            hfov_deg=float(payload.get("hfov_deg", 0.0)),
        )

    def summary(self) -> dict:
        flat = self.dist_coeffs.reshape(-1)
        return {
            "distortion_model": self.distortion_model,
            "hfov_deg": self.hfov_deg,
            "fx": float(self.camera_matrix[0, 0]),
            "fy": float(self.camera_matrix[1, 1]),
            "cx": float(self.camera_matrix[0, 2]),
            "cy": float(self.camera_matrix[1, 2]),
            "k1": float(flat[0]) if flat.size > 0 else 0.0,
            "k2": float(flat[1]) if flat.size > 1 else 0.0,
        }


def camera_matrix_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _as_dist_coeffs(dist: np.ndarray | list[float], model: DistortionModelName) -> np.ndarray:
    arr = np.asarray(dist, dtype=np.float64).reshape(-1)
    if model == "fisheye":
        if arr.size < 4:
            arr = np.pad(arr, (0, 4 - arr.size))
        return arr[:4].reshape(4, 1)
    if arr.size < 5:
        arr = np.pad(arr, (0, 5 - arr.size))
    return arr[:5].reshape(5, 1)


def undistort_points(
    points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model: DistortionModelName = "pinhole",
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    dist = _as_dist_coeffs(dist_coeffs, model)
    if model == "fisheye":
        undist = cv2.fisheye.undistortPoints(points, camera_matrix, dist, P=camera_matrix)
    else:
        undist = cv2.undistortPoints(points, camera_matrix, dist, P=camera_matrix)
    return undist.reshape(-1, 2)


def _project_normalized(normalized_xy: np.ndarray, dist: np.ndarray, model: DistortionModelName) -> np.ndarray:
    x = normalized_xy[:, 0]
    y = normalized_xy[:, 1]
    if model == "fisheye":
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        k = dist.reshape(-1)
        theta_d = theta * (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8)
        scale = np.ones_like(r)
        mask = r > 1e-8
        scale[mask] = theta_d[mask] / r[mask]
        return np.column_stack([x * scale, y * scale])

    k1, k2, p1, p2, k3 = dist.reshape(-1)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([x_dist, y_dist])


def distort_points(
    undist_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model: DistortionModelName = "pinhole",
) -> np.ndarray:
    points = np.asarray(undist_points_px, dtype=np.float64).reshape(-1, 2)
    dist = _as_dist_coeffs(dist_coeffs, model)

    if model == "pinhole":
        normalized = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            camera_matrix,
            np.zeros((5, 1), dtype=np.float64),
            P=None,
        ).reshape(-1, 2)
        object_points = np.hstack(
            [normalized, np.ones((len(normalized), 1), dtype=np.float64)]
        )
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
        return projected.reshape(-1, 2).astype(np.float32)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    normalized = np.column_stack([(points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy])
    guess = normalized.copy()
    for _ in range(12):
        projected = _project_normalized(guess, dist, model)
        delta = normalized - projected
        if float(np.max(np.abs(delta))) < 1e-9:
            break
        guess += delta

    distorted_norm = _project_normalized(guess, dist, model)
    return np.column_stack([distorted_norm[:, 0] * fx + cx, distorted_norm[:, 1] * fy + cy]).astype(np.float32)


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4:
        return None
    homography, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), method=0)
    return homography


def mean_corner_reprojection_mm(
    src_px: np.ndarray,
    dst_mm: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model: DistortionModelName = "pinhole",
) -> float:
    return _mean_reprojection_mm(src_px, dst_mm, camera_matrix, dist_coeffs, model)


def _mean_reprojection_mm(
    src_px: np.ndarray,
    dst_mm: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model: DistortionModelName,
) -> float:
    undist = undistort_points(src_px, camera_matrix, dist_coeffs, model)
    homography = _fit_homography(undist, dst_mm)
    if homography is None:
        return float("inf")
    reprojected = cv2.perspectiveTransform(undist.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(reprojected - dst_mm, axis=1)
    if not np.all(np.isfinite(errors)):
        return float("inf")
    return float(np.mean(errors))


def estimate_distortion_from_tags(
    image_corners_px: np.ndarray,
    bed_corners_mm: np.ndarray,
    camera_matrix: np.ndarray,
    model: DistortionModelName = "pinhole",
) -> tuple[np.ndarray, float]:
    src = np.asarray(image_corners_px, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(bed_corners_mm, dtype=np.float32).reshape(-1, 2)

    best_dist = np.zeros((5, 1), dtype=np.float64)
    best_error = _mean_reprojection_mm(src, dst, camera_matrix, best_dist, model)

    k1_values = np.linspace(-1.0, 0.15, 47)
    k2_values = np.linspace(-0.35, 0.45, 33)
    for k1 in k1_values:
        for k2 in k2_values:
            candidate = np.array([[k1], [k2], [0.0], [0.0], [0.0]], dtype=np.float64)
            error = _mean_reprojection_mm(src, dst, camera_matrix, candidate, model)
            if error < best_error:
                best_error = error
                best_dist = candidate.copy()

    best_k1 = float(best_dist[0, 0])
    best_k2 = float(best_dist[1, 0])
    for k1 in np.linspace(best_k1 - 0.08, best_k1 + 0.08, 33):
        for k2 in np.linspace(best_k2 - 0.08, best_k2 + 0.08, 17):
            candidate = np.array([[k1], [k2], [0.0], [0.0], [0.0]], dtype=np.float64)
            error = _mean_reprojection_mm(src, dst, camera_matrix, candidate, model)
            if error < best_error:
                best_error = error
                best_dist = candidate.copy()

    return best_dist, best_error


def resolve_camera_intrinsics(
    *,
    image_width: int,
    image_height: int,
    hfov_deg: float,
    distortion_model: DistortionModelName,
    override_fx: float | None = None,
    override_fy: float | None = None,
    override_cx: float | None = None,
    override_cy: float | None = None,
    override_dist: list[float] | None = None,
) -> CameraIntrinsics:
    default_k = camera_matrix_from_hfov(image_width, image_height, hfov_deg)
    camera_matrix = default_k.copy()
    if override_fx is not None:
        camera_matrix[0, 0] = override_fx
    if override_fy is not None:
        camera_matrix[1, 1] = override_fy
    if override_cx is not None:
        camera_matrix[0, 2] = override_cx
    if override_cy is not None:
        camera_matrix[1, 2] = override_cy

    if override_dist is not None:
        dist = _as_dist_coeffs(np.array(override_dist), distortion_model)
    else:
        dist = np.zeros((5, 1), dtype=np.float64)

    return CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist,
        distortion_model=distortion_model,
        image_width=image_width,
        image_height=image_height,
        hfov_deg=hfov_deg,
    )

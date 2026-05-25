from __future__ import annotations

import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.services.calibration_service import CalibrationService, get_calibration_service
from app.services.camera_intrinsics import (
    CameraIntrinsics,
    distort_points,
    resolve_camera_intrinsics,
    undistort_points,
)


class TransformService:
    def __init__(self, calibration_service: CalibrationService | None = None) -> None:
        self.calibration_service = calibration_service or get_calibration_service()

    def is_ready(self) -> bool:
        return self.calibration_service.is_calibrated()

    def _homography(self) -> np.ndarray:
        data = self.calibration_service.data
        if data is None:
            raise RuntimeError("Calibration required before coordinate transform")
        return data.homography

    def _inverse_homography(self) -> np.ndarray:
        data = self.calibration_service.data
        if data is None:
            raise RuntimeError("Calibration required before coordinate transform")
        return data.inverse_homography

    def _principal_point(self) -> tuple[float, float]:
        data = self.calibration_service.data
        if data is None:
            raise RuntimeError("Calibration required before coordinate transform")
        return data.principal_point_px

    def _intrinsics(self) -> CameraIntrinsics:
        data = self.calibration_service.data
        if data is None:
            raise RuntimeError("Calibration required before coordinate transform")
        if data.intrinsics is not None:
            return data.intrinsics

        config = get_config_store().config.camera
        width, height = config.main_resolution
        return resolve_camera_intrinsics(
            image_width=width,
            image_height=height,
            hfov_deg=config.hfov_deg,
            distortion_model=config.distortion_model,
            override_fx=config.intrinsics_override.fx,
            override_fy=config.intrinsics_override.fy,
            override_cx=config.intrinsics_override.cx,
            override_cy=config.intrinsics_override.cy,
            override_dist=config.intrinsics_override.dist,
        )

    def px_to_mm(self, points_px: np.ndarray, object_height_mm: float = 0.0) -> np.ndarray:
        points = np.asarray(points_px, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        if points.ndim == 2 and points.shape[1] != 2:
            raise ValueError("points_px must have shape (N, 2)")

        intrinsics = self._intrinsics()
        undist = undistort_points(
            points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            intrinsics.distortion_model,
        )
        homography = self._homography()
        transformed = cv2_perspective(homography, undist)

        if object_height_mm > 0:
            config = get_config_store().config
            mount_height = config.camera.mount_height_mm
            cx, cy = self._principal_point()
            for idx, (px, py) in enumerate(points):
                delta_x_px = (px - cx) / mount_height * object_height_mm
                delta_y_px = (py - cy) / mount_height * object_height_mm
                delta_mm = self._pixel_delta_to_mm(delta_x_px, delta_y_px, px, py)
                transformed[idx] -= delta_mm
        return transformed

    def mm_to_px(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        inverse_h = self._inverse_homography()
        undist = cv2_perspective(inverse_h, points)
        intrinsics = self._intrinsics()
        return distort_points(
            undist,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            intrinsics.distortion_model,
        )

    def _pixel_delta_to_mm(
        self, delta_x_px: float, delta_y_px: float, px: float, py: float
    ) -> np.ndarray:
        base = self.px_to_mm(np.array([[px, py]], dtype=np.float32), object_height_mm=0.0)[0]
        shifted = self.px_to_mm(
            np.array([[px + delta_x_px, py + delta_y_px]], dtype=np.float32),
            object_height_mm=0.0,
        )[0]
        return shifted - base

    def bbox_px_to_mm(self, bbox: BoundingBox, object_height_mm: float = 0.0) -> BoundingBox:
        corners = np.array(
            [
                [bbox.x_min, bbox.y_min],
                [bbox.x_max, bbox.y_min],
                [bbox.x_max, bbox.y_max],
                [bbox.x_min, bbox.y_max],
            ],
            dtype=np.float32,
        )
        mm_corners = self.px_to_mm(corners, object_height_mm=object_height_mm)
        return BoundingBox(
            x_min=float(np.min(mm_corners[:, 0])),
            y_min=float(np.min(mm_corners[:, 1])),
            x_max=float(np.max(mm_corners[:, 0])),
            y_max=float(np.max(mm_corners[:, 1])),
        )

    def point_px_to_mm(self, point: Point2D, object_height_mm: float = 0.0) -> Point2D:
        result = self.px_to_mm(
            np.array([[point.x, point.y]], dtype=np.float32),
            object_height_mm=object_height_mm,
        )[0]
        return Point2D(x=float(result[0]), y=float(result[1]))

    def polygon_px_to_mm(
        self, points: list[Point2D], object_height_mm: float = 0.0
    ) -> list[Point2D]:
        if not points:
            return []
        arr = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        transformed = self.px_to_mm(arr, object_height_mm=object_height_mm)
        return [Point2D(x=float(row[0]), y=float(row[1])) for row in transformed]


def cv2_perspective(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    import cv2

    reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, matrix)
    return transformed.reshape(-1, 2)


_transform_service: TransformService | None = None


def get_transform_service() -> TransformService:
    global _transform_service
    if _transform_service is None:
        _transform_service = TransformService()
    return _transform_service

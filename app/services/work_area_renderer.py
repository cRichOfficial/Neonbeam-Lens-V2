from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.common import Point2D
from app.schemas.shapes import WorkAreaImageInfo
from app.services.calibration_service import CalibrationService, get_calibration_service
from app.services.transform_service import TransformService, get_transform_service


@dataclass(frozen=True)
class WorkAreaView:
    image: np.ndarray
    width_mm: float
    height_mm: float
    width_px: int
    height_px: int
    pixels_per_mm: float
    origin_tag_id: int

    def to_info(self) -> WorkAreaImageInfo:
        return WorkAreaImageInfo(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            width_px=self.width_px,
            height_px=self.height_px,
            pixels_per_mm=self.pixels_per_mm,
            origin_tag_id=self.origin_tag_id,
        )


def resolve_pixels_per_mm(
    width_mm: float,
    height_mm: float,
    *,
    pixels_per_mm: float | None = None,
    max_edge_px: int = 1024,
) -> float:
    if pixels_per_mm is not None and pixels_per_mm > 0:
        return float(pixels_per_mm)
    longest_mm = max(width_mm, height_mm)
    if longest_mm <= 0:
        return 1.0
    return float(max_edge_px) / longest_mm


def mm_to_work_area_px(x_mm: float, y_mm: float, height_mm: float, pixels_per_mm: float) -> tuple[float, float]:
    return x_mm * pixels_per_mm, (height_mm - y_mm) * pixels_per_mm


def work_area_px_to_mm(x_px: float, y_px: float, height_mm: float, pixels_per_mm: float) -> tuple[float, float]:
    if pixels_per_mm <= 0:
        return 0.0, 0.0
    return x_px / pixels_per_mm, height_mm - (y_px / pixels_per_mm)


def mm_point_to_work_area_px(point: Point2D, height_mm: float, pixels_per_mm: float) -> Point2D:
    x_px, y_px = mm_to_work_area_px(point.x, point.y, height_mm, pixels_per_mm)
    return Point2D(x=x_px, y=y_px)


class WorkAreaRenderer:
    def __init__(
        self,
        calibration_service: CalibrationService | None = None,
        transform_service: TransformService | None = None,
    ) -> None:
        self.calibration_service = calibration_service or get_calibration_service()
        self.transform_service = transform_service or get_transform_service()

    def require_work_area(self) -> tuple[float, float, int]:
        data = self.calibration_service.data
        if data is None or data.work_area is None:
            raise RuntimeError("Calibration with work area is required")
        work = data.work_area
        return work.width_mm, work.height_mm, work.origin_tag_id

    def render(
        self,
        frame: np.ndarray,
        *,
        pixels_per_mm: float | None = None,
        max_edge_px: int = 1024,
    ) -> WorkAreaView:
        width_mm, height_mm, origin_tag_id = self.require_work_area()
        ppm = resolve_pixels_per_mm(
            width_mm,
            height_mm,
            pixels_per_mm=pixels_per_mm,
            max_edge_px=max_edge_px,
        )
        width_px = max(1, int(round(width_mm * ppm)))
        height_px = max(1, int(round(height_mm * ppm)))

        bed_corners_mm = np.array(
            [[0.0, 0.0], [width_mm, 0.0], [width_mm, height_mm], [0.0, height_mm]],
            dtype=np.float32,
        )
        src_px = self.transform_service.mm_to_px(bed_corners_mm)
        dst_px = np.array(
            [
                [0.0, float(height_px)],
                [float(width_px), float(height_px)],
                [float(width_px), 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_px.astype(np.float32), dst_px)
        if frame.ndim == 2:
            warped = cv2.warpPerspective(frame, matrix, (width_px, height_px))
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        else:
            warped = cv2.warpPerspective(frame, matrix, (width_px, height_px))

        return WorkAreaView(
            image=warped,
            width_mm=width_mm,
            height_mm=height_mm,
            width_px=width_px,
            height_px=height_px,
            pixels_per_mm=ppm,
            origin_tag_id=origin_tag_id,
        )

    def encode_jpeg(self, view: WorkAreaView, quality: int = 85) -> bytes:
        ok, encoded = cv2.imencode(".jpg", view.image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode work area JPEG")
        return encoded.tobytes()


_renderer: WorkAreaRenderer | None = None


def get_work_area_renderer() -> WorkAreaRenderer:
    global _renderer
    if _renderer is None:
        _renderer = WorkAreaRenderer()
    return _renderer

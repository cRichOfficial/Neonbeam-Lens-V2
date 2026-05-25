from __future__ import annotations

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import Point2D
from app.schemas.detection import DetectionItem
from app.services.apriltag_service import get_apriltag_service
from app.services.calibration_service import get_calibration_service
from app.services.cpu_detector import RawDetection
from app.services.transform_service import get_transform_service


class DebugRenderer:
    def render(
        self,
        frame: np.ndarray,
        detections: list[DetectionItem] | None = None,
        draw_tags: bool = True,
        draw_grid: bool = True,
    ) -> bytes:
        output = frame.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        calibration = get_calibration_service()
        transform = get_transform_service()
        config = get_config_store().config

        if draw_tags:
            tags = get_apriltag_service().detect(output)
            output = get_apriltag_service().draw_detections(output, tags)

        if draw_grid and calibration.is_calibrated():
            output = self._draw_grid(output, transform)

        if calibration.is_calibrated():
            output = self._draw_bed_boundary(output, transform, config.bed.width_mm, config.bed.height_mm)
            status = calibration.get_status()
            if status["reprojection_error_mm"] is not None:
                cv2.putText(
                    output,
                    f"Reproj err: {status['reprojection_error_mm']:.2f} mm",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if detections:
            for det in detections:
                bbox = det.bbox_px
                cv2.rectangle(
                    output,
                    (int(bbox.x_min), int(bbox.y_min)),
                    (int(bbox.x_max), int(bbox.y_max)),
                    (0, 255, 0),
                    2,
                )
                label = det.class_name
                if det.center_mm is not None:
                    label += f" ({det.center_mm.x:.1f}, {det.center_mm.y:.1f}) mm"
                cv2.putText(
                    output,
                    label,
                    (int(bbox.x_min), max(20, int(bbox.y_min) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if det.segmentation_polygon_px:
                    pts = np.array([[p.x, p.y] for p in det.segmentation_polygon_px], dtype=np.int32)
                    cv2.polylines(output, [pts], True, (255, 128, 0), 2)

        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode debug image")
        return encoded.tobytes()

    def _draw_bed_boundary(self, frame: np.ndarray, transform, width_mm: float, height_mm: float) -> np.ndarray:
        corners_mm = np.array(
            [[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]],
            dtype=np.float32,
        )
        corners_px = transform.mm_to_px(corners_mm).astype(np.int32)
        cv2.polylines(frame, [corners_px], True, (255, 0, 255), 2)
        return frame

    def _draw_grid(self, frame: np.ndarray, transform, spacing_mm: float = 50.0) -> np.ndarray:
        config = get_config_store().config
        width_mm = config.bed.width_mm
        height_mm = config.bed.height_mm
        x_values = np.arange(0, width_mm + spacing_mm, spacing_mm)
        y_values = np.arange(0, height_mm + spacing_mm, spacing_mm)

        for x_mm in x_values:
            pts_mm = np.array([[x_mm, 0.0], [x_mm, height_mm]], dtype=np.float32)
            pts_px = transform.mm_to_px(pts_mm).astype(np.int32)
            cv2.line(frame, tuple(pts_px[0]), tuple(pts_px[1]), (80, 80, 160), 1)

        for y_mm in y_values:
            pts_mm = np.array([[0.0, y_mm], [width_mm, y_mm]], dtype=np.float32)
            pts_px = transform.mm_to_px(pts_mm).astype(np.int32)
            cv2.line(frame, tuple(pts_px[0]), tuple(pts_px[1]), (80, 80, 160), 1)

        return frame


def raw_to_detection_items(
    raw_detections: list[RawDetection],
    object_height_mm: float,
) -> list[DetectionItem]:
    transform = get_transform_service()
    calibrated = transform.is_ready()
    items: list[DetectionItem] = []

    for raw in raw_detections:
        bbox_mm = None
        center_mm = None
        polygon_mm = None
        if calibrated:
            bbox_mm = transform.bbox_px_to_mm(raw.bbox_px, object_height_mm=object_height_mm)
            center = raw.bbox_px.center
            center_mm = transform.point_px_to_mm(center, object_height_mm=object_height_mm)
            if raw.segmentation_polygon_px:
                polygon_mm = transform.polygon_px_to_mm(
                    raw.segmentation_polygon_px, object_height_mm=object_height_mm
                )

        items.append(
            DetectionItem(
                class_id=raw.class_id,
                class_name=raw.class_name,
                confidence=raw.confidence,
                bbox_px=raw.bbox_px,
                bbox_mm=bbox_mm,
                center_mm=center_mm,
                segmentation_polygon_px=raw.segmentation_polygon_px,
                segmentation_polygon_mm=polygon_mm,
            )
        )
    return items


_debug_renderer: DebugRenderer | None = None


def get_debug_renderer() -> DebugRenderer:
    global _debug_renderer
    if _debug_renderer is None:
        _debug_renderer = DebugRenderer()
    return _debug_renderer

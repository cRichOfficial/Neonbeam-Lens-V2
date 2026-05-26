from __future__ import annotations

import cv2
import numpy as np

from app.config import BedConfig, get_config_store
from app.schemas.calibration import AprilTagSpec
from app.services.apriltag_service import get_apriltag_service
from app.services.bed_frame import bed_boundary_corners_mm
from app.services.calibration_service import get_calibration_service
from app.services.transform_service import get_transform_service
from app.services.work_area import measure_tag_edge_lengths_mm


class DebugRenderer:
    def render(
        self,
        frame: np.ndarray,
        draw_tags: bool = True,
        draw_grid: bool = True,
        draw_side_lengths: bool = False,
        draw_tag_sizes: bool = False,
    ) -> bytes:
        output = frame.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

        calibration = get_calibration_service()
        transform = get_transform_service()
        tag_detections: list[dict] = []

        if draw_tags:
            tag_detections = get_apriltag_service().detect(output)
            output = get_apriltag_service().draw_detections(output, tag_detections)
            if draw_tag_sizes and calibration.is_calibrated():
                output = self._draw_tag_sizes(output, calibration, tag_detections)

        if draw_grid and calibration.is_calibrated():
            bed = calibration.get_effective_bed()
            output = self._draw_grid(output, transform, bed)

        if calibration.is_calibrated():
            bed = calibration.get_effective_bed()
            output = self._draw_bed_boundary(output, transform, bed)
            if draw_side_lengths:
                output = self._draw_side_lengths(output, transform, bed)
            status = calibration.get_status()
            y_offset = 30
            if status["reprojection_error_mm"] is not None:
                cv2.putText(
                    output,
                    f"Reproj err: {status['reprojection_error_mm']:.2f} mm",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 28
            tag_size_validation = status.get("tag_size_validation")
            if tag_size_validation is not None:
                max_err = tag_size_validation["max_error_mm"]
                color = (255, 255, 255) if tag_size_validation["converged"] else (0, 165, 255)
                cv2.putText(
                    output,
                    f"Tag size err: max {max_err:.2f} mm",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode debug image")
        return encoded.tobytes()

    def _draw_tag_sizes(
        self,
        frame: np.ndarray,
        calibration,
        tag_detections: list[dict],
    ) -> np.ndarray:
        data = calibration.data
        if data is None or data.intrinsics is None:
            return frame

        specs_by_id = {item["id"]: AprilTagSpec.model_validate(item) for item in data.tag_specs}
        matched = [
            (det, specs_by_id[det["id"]])
            for det in tag_detections
            if det["id"] in specs_by_id
        ]
        if not matched:
            return frame

        measured_sizes = measure_tag_edge_lengths_mm(
            matched,
            data.homography,
            data.intrinsics,
        )
        config = get_config_store().config
        max_tag_error_mm = config.calibration.max_tag_size_error_mm

        for det in tag_detections:
            tag_id = det["id"]
            if tag_id not in measured_sizes or tag_id not in specs_by_id:
                continue
            expected_mm = specs_by_id[tag_id].size_mm
            measured_mm = measured_sizes[tag_id]
            center = (int(det["center_px"][0]), int(det["center_px"][1]))
            label = f"{expected_mm:.0f} mm (meas {measured_mm:.1f})"
            error = abs(measured_mm - expected_mm)
            color = (255, 255, 0) if error <= max_tag_error_mm else (0, 165, 255)
            self._put_text_outlined(
                frame,
                label,
                (center[0], center[1] + 24),
                font_scale=0.5,
                color=color,
                centered=True,
            )
        return frame

    def _draw_bed_boundary(self, frame: np.ndarray, transform, bed: BedConfig) -> np.ndarray:
        corners_mm = bed_boundary_corners_mm(bed)
        corners_px = transform.mm_to_px(corners_mm).astype(np.int32)
        cv2.polylines(frame, [corners_px], True, (255, 0, 255), 2)
        return frame

    def _draw_side_lengths(self, frame: np.ndarray, transform, bed: BedConfig) -> np.ndarray:
        corners_mm = bed_boundary_corners_mm(bed)
        corners_px = transform.mm_to_px(corners_mm).astype(np.float32)
        center_px = corners_px.mean(axis=0)

        for index in range(4):
            start_mm = corners_mm[index]
            end_mm = corners_mm[(index + 1) % 4]
            start_px = corners_px[index]
            end_px = corners_px[(index + 1) % 4]
            length_mm = float(np.linalg.norm(end_mm - start_mm))

            midpoint_px = (start_px + end_px) / 2.0
            edge = end_px - start_px
            edge_len = float(np.linalg.norm(edge))
            if edge_len > 1e-6:
                normal = np.array([-edge[1], edge[0]], dtype=np.float32) / edge_len
                toward_center = center_px - midpoint_px
                if float(np.dot(normal, toward_center)) < 0:
                    normal = -normal
                midpoint_px = midpoint_px + normal * 14.0

            label = f"{length_mm:.0f} mm"
            self._put_text_outlined(
                frame,
                label,
                (int(midpoint_px[0]), int(midpoint_px[1])),
                font_scale=0.55,
                color=(255, 255, 0),
                centered=True,
            )
        return frame

    def _put_text_outlined(
        self,
        frame: np.ndarray,
        text: str,
        org: tuple[int, int],
        *,
        font_scale: float,
        color: tuple[int, int, int],
        centered: bool = False,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        if centered:
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            org = (
                org[0] - text_width // 2,
                org[1] + text_height // 2,
            )
        cv2.putText(frame, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_grid(
        self,
        frame: np.ndarray,
        transform,
        bed: BedConfig,
        spacing_mm: float = 50.0,
    ) -> np.ndarray:
        width_mm = bed.width_mm
        height_mm = bed.height_mm
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


_debug_renderer: DebugRenderer | None = None


def get_debug_renderer() -> DebugRenderer:
    global _debug_renderer
    if _debug_renderer is None:
        _debug_renderer = DebugRenderer()
    return _debug_renderer

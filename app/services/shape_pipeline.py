from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.schemas.shapes import ShapeBackend, ShapeDetectionItem, ShapesResponse, WorkAreaSummary
from app.services.fastsam_detector import FastSamDetector, get_fastsam_detector
from app.services.pipeline_debug_mosaic import compose_stage_mosaic, encode_jpeg
from app.services.shape_detector import RawShapeDetection, ShapeDetector, ShapeDetectorConfig
from app.services.work_area_renderer import (
    WorkAreaRenderer,
    WorkAreaView,
    get_work_area_renderer,
    mm_to_work_area_px,
    work_area_px_to_mm,
)


@dataclass
class ShapePipelineResult:
    response: ShapesResponse
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    stage_order: list[str] = field(default_factory=list)


class ShapePipeline:
    def __init__(
        self,
        renderer: WorkAreaRenderer | None = None,
        fastsam: FastSamDetector | None = None,
    ) -> None:
        self.renderer = renderer or get_work_area_renderer()
        self.fastsam = fastsam or get_fastsam_detector()

    def _detector_config(self) -> ShapeDetectorConfig:
        cfg = get_config_store().config.shapes
        return ShapeDetectorConfig(
            min_area_mm2=cfg.min_area_mm2,
            max_area_mm2=cfg.max_area_mm2,
            min_solidity=cfg.min_solidity,
            min_extent=cfg.min_extent,
            circularity_threshold=cfg.circularity_threshold,
            rounded_rect_circularity_min=cfg.rounded_rect_circularity_min,
            bracelet_min_aspect=cfg.bracelet_min_aspect,
            roi_margin_mm=cfg.roi_margin_mm,
        )

    def run(
        self,
        frame: np.ndarray,
        *,
        backend: ShapeBackend = "auto",
        min_confidence: float | None = None,
        include_work_area_coords: bool = False,
        pixels_per_mm: float | None = None,
        max_edge_px: int | None = None,
    ) -> ShapePipelineResult:
        cfg = get_config_store().config.shapes
        threshold = min_confidence if min_confidence is not None else cfg.min_confidence
        max_edge = max_edge_px if max_edge_px is not None else cfg.max_edge_px

        raw_stage = frame.copy()
        if raw_stage.ndim == 2:
            raw_stage = cv2.cvtColor(raw_stage, cv2.COLOR_GRAY2BGR)

        view = self.renderer.render(
            frame,
            pixels_per_mm=pixels_per_mm,
            max_edge_px=max_edge,
        )
        detector = ShapeDetector(self._detector_config())

        classical = detector.detect(
            view.image,
            pixels_per_mm=view.pixels_per_mm,
            width_mm=view.width_mm,
            height_mm=view.height_mm,
            source="classical",
        )

        fastsam_used = False
        fastsam_masks: list[np.ndarray] = []
        detection = classical

        use_fastsam = backend == "fastsam"
        if backend == "auto":
            use_fastsam = (
                len(classical.objects) == 0
                or classical.best_confidence < cfg.classical_fallback_confidence
            )
        if use_fastsam and backend != "classical":
            fastsam_masks = self.fastsam.segment_masks(view.image)
            if fastsam_masks:
                combined = np.zeros(view.image.shape[:2], dtype=np.uint8)
                for mask in fastsam_masks:
                    combined = cv2.bitwise_or(combined, (mask > 0).astype(np.uint8) * 255)
                detection = ShapeDetector.from_mask(
                    view.image,
                    combined,
                    pixels_per_mm=view.pixels_per_mm,
                    width_mm=view.width_mm,
                    height_mm=view.height_mm,
                    source="fastsam",
                    config=self._detector_config(),
                )
                fastsam_used = True
            elif backend == "fastsam":
                detection = classical

        objects = [obj for obj in detection.objects if obj.confidence >= threshold]

        items = [
            self._to_item(
                index,
                raw,
                view,
                include_work_area_coords=include_work_area_coords,
            )
            for index, raw in enumerate(objects)
        ]

        backend_name = "fastsam" if fastsam_used else "classical"
        response = ShapesResponse(
            backend=backend_name,
            calibrated=True,
            work_area=WorkAreaSummary(
                width_mm=view.width_mm,
                height_mm=view.height_mm,
                origin_tag_id=view.origin_tag_id,
            ),
            count=len(items),
            objects=items,
            work_area_image=view.to_info(),
            fastsam_used=fastsam_used,
        )

        stages: dict[str, np.ndarray] = {
            "raw": raw_stage,
            "warp": view.image.copy(),
            "mask": self._render_mask_stage(view.image, detection.mask, detection.mask_method),
            "contours": self._render_contours_stage(view.image, detection.contours),
            "shapes": self._render_shapes_stage(view.image, objects),
            "final": self._render_final_stage(view, items),
        }
        order = ["raw", "warp", "mask", "contours", "shapes"]
        if fastsam_used:
            stages["fastsam"] = self.fastsam.render_overlay(view.image, fastsam_masks)
            order.append("fastsam")
        order.append("final")

        return ShapePipelineResult(response=response, stages=stages, stage_order=order)

    def _px_to_mm_point(self, x_px: float, y_px: float, view: WorkAreaView) -> Point2D:
        x_mm, y_mm = work_area_px_to_mm(x_px, y_px, view.height_mm, view.pixels_per_mm)
        return Point2D(x=x_mm, y=y_mm)

    def _to_item(
        self,
        index: int,
        raw: RawShapeDetection,
        view: WorkAreaView,
        *,
        include_work_area_coords: bool,
    ) -> ShapeDetectionItem:
        center_mm = self._px_to_mm_point(raw.center_px[0], raw.center_px[1], view)
        width_mm = raw.width_px / view.pixels_per_mm
        height_mm = raw.height_px / view.pixels_per_mm

        polygon_mm = [self._px_to_mm_point(x, y, view) for x, y in raw.segmentation_polygon_px]
        oriented_mm = [self._px_to_mm_point(x, y, view) for x, y in raw.oriented_box_px]
        xs = [p.x for p in polygon_mm]
        ys = [p.y for p in polygon_mm]
        bbox_mm = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

        seg_px = None
        oriented_px = None
        if include_work_area_coords:
            seg_px = [
                Point2D(x=px, y=py)
                for px, py in (
                    mm_to_work_area_px(p.x, p.y, view.height_mm, view.pixels_per_mm)
                    for p in polygon_mm
                )
            ]
            oriented_px = [
                Point2D(x=px, y=py)
                for px, py in (
                    mm_to_work_area_px(p.x, p.y, view.height_mm, view.pixels_per_mm)
                    for p in oriented_mm
                )
            ]

        return ShapeDetectionItem(
            id=index,
            shape=raw.shape,
            confidence=raw.confidence,
            source=raw.source,
            center_mm=center_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            rotation_deg=raw.rotation_deg,
            bbox_mm=bbox_mm,
            oriented_box_mm=oriented_mm,
            segmentation_polygon_mm=polygon_mm,
            segmentation_polygon_work_area_px=seg_px,
            oriented_box_work_area_px=oriented_px,
        )

    def _render_mask_stage(self, base: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
        overlay = base.copy()
        color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        color[mask > 0] = (0, 220, 255)
        blended = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)
        cv2.putText(
            blended,
            f"mask:{method}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return blended

    def _render_contours_stage(self, base: np.ndarray, contours: list) -> np.ndarray:
        output = base.copy()
        for candidate in contours:
            color = (0, 255, 0) if candidate.kept else (0, 0, 255)
            cv2.drawContours(output, [candidate.contour], -1, color, 2)
            if candidate.reject_reason:
                x, y, _, _ = cv2.boundingRect(candidate.contour)
                cv2.putText(
                    output,
                    candidate.reject_reason,
                    (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return output

    def _render_shapes_stage(self, base: np.ndarray, objects: list[RawShapeDetection]) -> np.ndarray:
        output = base.copy()
        for obj in objects:
            box = np.array(obj.oriented_box_px, dtype=np.int32)
            cv2.polylines(output, [box], True, (255, 200, 0), 2)
            if obj.shape == "circle":
                center = (int(obj.center_px[0]), int(obj.center_px[1]))
                radius = int(max(obj.width_px, obj.height_px) / 2.0)
                cv2.circle(output, center, radius, (0, 255, 255), 2)
            else:
                cv2.arrowedLine(
                    output,
                    (int(obj.center_px[0]), int(obj.center_px[1])),
                    (
                        int(obj.center_px[0] + 40 * np.cos(np.radians(-obj.rotation_deg))),
                        int(obj.center_px[1] + 40 * np.sin(np.radians(-obj.rotation_deg))),
                    ),
                    (255, 255, 0),
                    2,
                    tipLength=0.3,
                )
        return output

    def _render_final_stage(self, view: WorkAreaView, items: list[ShapeDetectionItem]) -> np.ndarray:
        output = view.image.copy()
        for item in items:
            corners = [
                Point2D(x=px, y=py)
                for px, py in (
                    mm_to_work_area_px(p.x, p.y, view.height_mm, view.pixels_per_mm)
                    for p in item.oriented_box_mm
                )
            ]
            box = np.array([[c.x, c.y] for c in corners], dtype=np.int32)
            cv2.polylines(output, [box], True, (0, 255, 128), 2)
            label = (
                f"#{item.id} {item.shape} {item.width_mm:.0f}x{item.height_mm:.0f}mm "
                f"@{item.rotation_deg:.1f}deg"
            )
            anchor = corners[0]
            cv2.putText(
                output,
                label,
                (int(anchor.x), int(max(14, anchor.y - 6))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return output

    def render_debug_stage(
        self,
        result: ShapePipelineResult,
        stage: str,
        *,
        max_width_px: int = 1920,
        max_height_px: int = 1080,
        columns: int = 3,
        quality: int = 85,
    ) -> bytes:
        if stage == "all":
            tiles = [(name, result.stages[name]) for name in result.stage_order if name in result.stages]
            mosaic = compose_stage_mosaic(
                tiles,
                max_width_px=max_width_px,
                max_height_px=max_height_px,
                columns=columns,
            )
            return encode_jpeg(mosaic, quality=quality)
        if stage not in result.stages:
            raise KeyError(f"Unknown debug stage: {stage}")
        return encode_jpeg(result.stages[stage], quality=quality)


_pipeline: ShapePipeline | None = None


def get_shape_pipeline() -> ShapePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ShapePipeline()
    return _pipeline

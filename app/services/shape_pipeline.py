from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.common import BoundingBox, Point2D
from app.schemas.detection import DetectionItem, DetectionResponse, WorkAreaSummary
from app.services.fastsam_detector import FastSamDetector, get_fastsam_detector
from app.services.pipeline_debug_mosaic import compose_stage_mosaic, encode_jpeg, put_text_outlined
from app.services.shape_detector import RawShapeDetection, ShapeDetector, ShapeDetectorConfig
from app.services.shape_fastsam_filter import (
    extract_bg_subtract_mask,
    filter_fastsam_masks,
    is_bg_mask_sane,
)
from app.services.shape_mask_tracks import split_foreground_components, track_bg_subtract
from app.services.work_area_background import get_work_area_background_store
from app.services.work_area_renderer import (
    WorkAreaRenderer,
    WorkAreaView,
    get_work_area_renderer,
    mm_to_work_area_px,
    work_area_px_to_mm,
)

logger = logging.getLogger(__name__)


@dataclass
class ShapePipelineResult:
    response: DetectionResponse
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

    def _detector_config(self, *, use_background_reference: bool | None = None) -> ShapeDetectorConfig:
        cfg = get_config_store().config.detection
        use_bg = (
            use_background_reference
            if use_background_reference is not None
            else cfg.use_background_reference
        )
        return ShapeDetectorConfig(
            min_area_mm2=cfg.min_area_mm2,
            max_area_mm2=cfg.max_area_mm2,
            split_above_area_mm2=cfg.split_above_area_mm2,
            max_object_span_ratio=cfg.max_object_span_ratio,
            min_solidity=cfg.min_solidity,
            min_extent=cfg.min_extent,
            circularity_threshold=cfg.circularity_threshold,
            rounded_rect_circularity_min=cfg.rounded_rect_circularity_min,
            bracelet_min_aspect=cfg.bracelet_min_aspect,
            roi_margin_mm=cfg.roi_margin_mm,
            mask_morph_kernel_px=cfg.mask_morph_kernel_px,
            mask_min_component_area_mm2=cfg.mask_min_component_area_mm2,
            mask_max_components=cfg.mask_max_components,
            morph_close_iterations=cfg.morph_close_iterations,
            mask_max_component_area_ratio=cfg.mask_max_component_area_ratio,
            mask_bridge_break_kernel_px=cfg.mask_bridge_break_kernel_px,
            glare_suppression_enabled=cfg.glare_suppression_enabled,
            glare_suppression_l_cap=cfg.glare_suppression_l_cap,
            glare_rejection_enabled=cfg.glare_rejection_enabled,
            glare_l_delta=cfg.glare_l_delta,
            glare_l_absolute_min=cfg.glare_l_absolute_min,
            use_background_reference=use_bg,
            bg_subtract_mode=cfg.effective_bg_subtract_mode(),
            bg_subtract_min_diff=cfg.bg_subtract_min_diff,
            bg_subtract_blur_kernel_px=cfg.bg_subtract_blur_kernel_px,
            bg_texture_min_diff=cfg.bg_texture_min_diff,
            bg_texture_blur_kernel_px=cfg.bg_texture_blur_kernel_px,
        )

    def run(
        self,
        frame: np.ndarray,
        *,
        min_confidence: float | None = None,
        include_work_area_coords: bool = False,
        pixels_per_mm: float | None = None,
        max_edge_px: int | None = None,
        use_background_reference: bool | None = None,
        show_center_coords: bool = False,
    ) -> ShapePipelineResult:
        cfg = get_config_store().config.detection
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
        background_store = get_work_area_background_store()
        reference_bgr, background_stale_reason = background_store.load_for_view(
            view,
            max_edge_px=max_edge,
        )
        use_bg = (
            use_background_reference
            if use_background_reference is not None
            else cfg.use_background_reference
        )
        background_reference_used = (
            use_bg
            and reference_bgr is not None
            and background_stale_reason is None
        )

        detector_cfg = self._detector_config(use_background_reference=use_bg)

        bg_mask: np.ndarray | None = None
        bg_track = None
        texture_diff: np.ndarray | None = None
        if background_reference_used and reference_bgr is not None:
            bg_mask = extract_bg_subtract_mask(
                view.image,
                reference_bgr,
                detector_cfg,
                pixels_per_mm=view.pixels_per_mm,
            )
            from app.services.shape_detector import _estimate_bed_l

            margin_px = int(detector_cfg.roi_margin_mm * view.pixels_per_mm)
            h, w = view.image.shape[:2]
            l_channel = cv2.cvtColor(view.image, cv2.COLOR_BGR2LAB)[:, :, 0]
            bed_l = _estimate_bed_l(
                l_channel,
                margin_px,
                margin_px,
                w - margin_px,
                h - margin_px,
            )
            bg_track = track_bg_subtract(
                view.image,
                reference_bgr,
                detector_cfg,
                pixels_per_mm=view.pixels_per_mm,
                bed_l=bed_l,
                mode=detector_cfg.bg_subtract_mode,
            )
            if bg_track is not None:
                texture_diff = bg_track.texture_diff

        fastsam_used = False
        texture_fallback_used = False
        fastsam_device: str | None = None
        fastsam_error: str | None = None
        raw_fastsam_masks: list[np.ndarray] = []
        filtered_fastsam_masks: list[np.ndarray] = []
        fastsam_objects: list[RawShapeDetection] = []
        fastsam_filter_detail = ""
        segment_detail = ""

        logger.info(
            "[fastsam] pipeline invoke device=%s warped=%dx%d",
            self.fastsam.active_device or "pending",
            view.width_px,
            view.height_px,
        )
        raw_fastsam_masks = self.fastsam.segment_masks(view.image)
        fastsam_device = self.fastsam.active_device
        segment_detail = getattr(self.fastsam, "last_segment_detail", "") or ""

        if raw_fastsam_masks:
            fastsam_used = True
            filtered_fastsam_masks, fastsam_filter_detail = filter_fastsam_masks(
                raw_fastsam_masks,
                bg_mask,
                min_overlap=cfg.fastsam_bg_filter_min_overlap,
                min_area_px=cfg.fastsam_min_mask_area_px,
                bg_filter_enabled=cfg.fastsam_bg_filter_enabled,
                max_fg_ratio=cfg.fastsam_bg_filter_max_fg_ratio,
            )
            logger.info("[fastsam] filter %s", fastsam_filter_detail)
            for mask_index, mask in enumerate(filtered_fastsam_masks):
                per_mask = ShapeDetector.from_mask(
                    view.image,
                    mask,
                    pixels_per_mm=view.pixels_per_mm,
                    width_mm=view.width_mm,
                    height_mm=view.height_mm,
                    config=detector_cfg,
                )
                logger.info(
                    "[fastsam] geometry mask[%d] objects=%d",
                    mask_index,
                    len(per_mask.objects),
                )
                fastsam_objects.extend(per_mask.objects)
            logger.info(
                "[fastsam] pipeline raw_masks=%d filtered=%d detections=%d device=%s",
                len(raw_fastsam_masks),
                len(filtered_fastsam_masks),
                len(fastsam_objects),
                fastsam_device,
            )
        else:
            status = self.fastsam.get_status()
            fastsam_error = status.get("last_error") if isinstance(status.get("last_error"), str) else None
            if not fastsam_error and segment_detail:
                fastsam_error = f"FastSAM returned no masks ({segment_detail})"
            elif not fastsam_error and status.get("loaded"):
                fastsam_error = "FastSAM returned no masks (check HEF output parsing or input size)"
            elif not fastsam_error:
                hailo_err = status.get("hailo", {}).get("last_error")
                cpu_err = status.get("cpu", {}).get("last_error")
                nested = hailo_err if isinstance(hailo_err, str) else None
                if not nested and isinstance(cpu_err, str):
                    nested = cpu_err
                fastsam_error = nested or "FastSAM not loaded"

        if not fastsam_objects and bg_mask is not None and is_bg_mask_sane(
            bg_mask,
            max_foreground_ratio=cfg.fastsam_bg_filter_max_fg_ratio,
        ):
            min_component_area_px = int(
                detector_cfg.mask_min_component_area_mm2 * view.pixels_per_mm * view.pixels_per_mm
            )
            component_masks = split_foreground_components(bg_mask, min_component_area_px)
            for comp_index, comp_mask in enumerate(component_masks):
                per_mask = ShapeDetector.from_mask(
                    view.image,
                    comp_mask,
                    pixels_per_mm=view.pixels_per_mm,
                    width_mm=view.width_mm,
                    height_mm=view.height_mm,
                    config=detector_cfg,
                )
                logger.info(
                    "[fastsam] texture_fallback comp[%d] objects=%d",
                    comp_index,
                    len(per_mask.objects),
                )
                fastsam_objects.extend(per_mask.objects)
            if fastsam_objects:
                texture_fallback_used = True
                fallback_note = f"texture_fallback {len(component_masks)} blob(s)"
                fastsam_filter_detail = (
                    f"{fastsam_filter_detail}; {fallback_note}".strip("; ")
                    if fastsam_filter_detail
                    else fallback_note
                )

        objects = [obj for obj in fastsam_objects if obj.confidence >= threshold]
        detections = [
            self._to_detection_item(index, raw, view, include_work_area_coords=include_work_area_coords)
            for index, raw in enumerate(objects)
        ]

        if texture_fallback_used:
            backend_name = "texture_fallback"
        else:
            backend_name = fastsam_device or "none"
        response = DetectionResponse(
            backend=backend_name,
            calibrated=True,
            work_area=WorkAreaSummary(
                width_mm=view.width_mm,
                height_mm=view.height_mm,
                origin_tag_id=view.origin_tag_id,
            ),
            count=len(detections),
            detections=detections,
            work_area_image=view.to_info(),
            fastsam_used=fastsam_used,
            fastsam_device=fastsam_device,
            fastsam_error=fastsam_error,
            fastsam_filter_detail=fastsam_filter_detail or None,
            background_reference_used=background_reference_used,
            background_stale_reason=background_stale_reason,
        )

        stages, order = self._build_debug_stages(
            raw_stage=raw_stage,
            view=view,
            background_store=background_store,
            detections=detections,
            bg_track=bg_track,
            bg_mask=bg_mask,
            texture_diff=texture_diff,
            raw_fastsam_masks=raw_fastsam_masks,
            filtered_fastsam_masks=filtered_fastsam_masks,
            fastsam_filter_detail=fastsam_filter_detail,
            fastsam_error=fastsam_error,
            segment_detail=segment_detail,
            show_center_coords=show_center_coords,
        )

        return ShapePipelineResult(response=response, stages=stages, stage_order=order)

    def _build_debug_stages(
        self,
        *,
        raw_stage: np.ndarray,
        view: WorkAreaView,
        background_store,
        detections: list[DetectionItem],
        bg_track,
        bg_mask: np.ndarray | None,
        texture_diff: np.ndarray | None,
        raw_fastsam_masks: list[np.ndarray],
        filtered_fastsam_masks: list[np.ndarray],
        fastsam_filter_detail: str,
        fastsam_error: str | None,
        segment_detail: str,
        show_center_coords: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        stages: dict[str, np.ndarray] = {
            "raw": raw_stage,
            "warp": view.image.copy(),
            "final": self._render_final_stage(
                view, detections, show_center_coords=show_center_coords
            ),
        }
        order = ["raw", "warp"]

        ref_image = background_store.load_image()
        if ref_image is not None:
            stages["bg_diff"] = background_store.render_diff(view.image, ref_image)
            order.append("bg_diff")

        if texture_diff is not None:
            stages["texture_diff"] = self._render_diff_stage(texture_diff, "texture_diff")
            order.append("texture_diff")

        if bg_track is not None:
            stages["bg_subtract"] = self._render_mask_stage(
                view.image,
                bg_track.mask,
                bg_track.name,
                component_count=bg_track.component_count,
                fragmentation=bg_track.fragmentation,
                label_prefix=f"mask:{bg_track.mode}",
            )
        elif bg_mask is not None and np.count_nonzero(bg_mask) > 0:
            stages["bg_subtract"] = self._render_mask_stage(view.image, bg_mask, "bg_subtract")
        if "bg_subtract" in stages:
            order.append("bg_subtract")

        if raw_fastsam_masks:
            stages["fastsam"] = self._render_fastsam_stage(
                view.image,
                raw_fastsam_masks,
                label=f"fastsam:{len(raw_fastsam_masks)} masks",
            )
            if filtered_fastsam_masks:
                stages["fastsam_filtered"] = self._render_fastsam_stage(
                    view.image,
                    filtered_fastsam_masks,
                    label=f"filtered:{len(filtered_fastsam_masks)} ({fastsam_filter_detail[:48]})",
                )
        else:
            stages["fastsam"] = self._render_fastsam_stage(
                view.image,
                [],
                label=f"fastsam:0 ({fastsam_error or segment_detail or 'no masks'})",
            )
        order.append("fastsam")
        if "fastsam_filtered" in stages:
            order.append("fastsam_filtered")
        order.append("final")
        return stages, order

    def _to_detection_item(
        self,
        index: int,
        raw: RawShapeDetection,
        view: WorkAreaView,
        *,
        include_work_area_coords: bool,
    ) -> DetectionItem:
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

        return DetectionItem(
            id=index,
            shape=raw.shape,
            confidence=raw.confidence,
            bbox_mm=bbox_mm,
            center_mm=center_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            rotation_deg=raw.rotation_deg,
            oriented_box_mm=oriented_mm,
            segmentation_polygon_mm=polygon_mm,
            segmentation_polygon_work_area_px=seg_px,
            oriented_box_work_area_px=oriented_px,
        )

    def _px_to_mm_point(self, x_px: float, y_px: float, view: WorkAreaView) -> Point2D:
        x_mm, y_mm = work_area_px_to_mm(x_px, y_px, view.height_mm, view.pixels_per_mm)
        return Point2D(x=x_mm, y=y_mm)

    def _render_fastsam_stage(
        self,
        base: np.ndarray,
        masks: list[np.ndarray],
        *,
        label: str,
    ) -> np.ndarray:
        output = self.fastsam.render_overlay(base, masks) if masks else base.copy()
        put_text_outlined(
            output,
            label[:96],
            (10, 28),
            font_scale=0.65,
            color=(0, 255, 255),
            thickness=2,
        )
        return output

    def _render_mask_stage(
        self,
        base: np.ndarray,
        mask: np.ndarray,
        method: str,
        *,
        component_count: int = 0,
        fragmentation: float = 0.0,
        label_prefix: str = "mask",
    ) -> np.ndarray:
        overlay = base.copy()
        if mask.shape[:2] != base.shape[:2]:
            mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        color[mask > 0] = (0, 220, 255)
        blended = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)
        label = f"{label_prefix}:{method} comps={component_count} frag={fragmentation:.2f}"
        put_text_outlined(blended, label, (10, 24), font_scale=0.55, thickness=2)
        return blended

    def _render_diff_stage(self, diff: np.ndarray, label: str) -> np.ndarray:
        if diff.ndim == 3:
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray = diff
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        max_val = int(np.max(gray)) if gray.size else 0
        put_text_outlined(
            output,
            f"{label} max={max_val} (bright=change)",
            (10, 24),
            font_scale=0.55,
            thickness=2,
        )
        return output

    def _render_final_stage(
        self,
        view: WorkAreaView,
        items: list[DetectionItem],
        *,
        show_center_coords: bool = False,
    ) -> np.ndarray:
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
            label_y = int(max(14, anchor.y - 6))
            put_text_outlined(
                output,
                label,
                (int(anchor.x), label_y),
                font_scale=0.45,
            )
            if show_center_coords:
                cx_px, cy_px = mm_to_work_area_px(
                    item.center_mm.x,
                    item.center_mm.y,
                    view.height_mm,
                    view.pixels_per_mm,
                )
                center = (int(round(cx_px)), int(round(cy_px)))
                cv2.circle(output, center, 5, (0, 128, 255), -1, lineType=cv2.LINE_AA)
                cv2.line(
                    output,
                    (center[0] - 10, center[1]),
                    (center[0] + 10, center[1]),
                    (0, 128, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    output,
                    (center[0], center[1] - 10),
                    (center[0], center[1] + 10),
                    (0, 128, 255),
                    1,
                    cv2.LINE_AA,
                )
                center_label = (
                    f"center {item.center_mm.x:.1f}, {item.center_mm.y:.1f} mm"
                )
                put_text_outlined(
                    output,
                    center_label,
                    (int(anchor.x), label_y + 16),
                    font_scale=0.42,
                    color=(0, 220, 255),
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
            available = ", ".join(result.stage_order)
            raise KeyError(
                f"Debug stage '{stage}' not available for this run (active stages: {available})"
            )
        return encode_jpeg(result.stages[stage], quality=quality)


_pipeline: ShapePipeline | None = None


def get_shape_pipeline() -> ShapePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ShapePipeline()
    return _pipeline

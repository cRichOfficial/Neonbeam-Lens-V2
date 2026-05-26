"""FastSAM detection pipeline tests — synthetic geometry, no Pi hardware."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.pipeline_debug_mosaic import compose_stage_mosaic, encode_jpeg
from app.services.shape_detector import ShapeDetector, ShapeDetectorConfig
from app.services.shape_pipeline import ShapePipeline
from app.services.work_area_renderer import (
    mm_to_work_area_px,
    resolve_pixels_per_mm,
    work_area_px_to_mm,
)


def _synthetic_work_area_view(
    width_mm: float = 400.0,
    height_mm: float = 400.0,
    ppm: float = 2.0,
) -> tuple[np.ndarray, float, float, float]:
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 240, dtype=np.uint8)
    cx = width_px // 2
    cy = height_px // 2
    size = int(90 * ppm)
    cv2.rectangle(
        image,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (30, 30, 30),
        -1,
    )
    return image, width_mm, height_mm, ppm


def _object_mask_from_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return mask


def _geometry_from_image(
    image: np.ndarray,
    *,
    width_mm: float,
    height_mm: float,
    ppm: float,
    config: ShapeDetectorConfig | None = None,
):
    mask = _object_mask_from_bgr(image)
    return ShapeDetector.from_mask(
        image,
        mask,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
        config=config,
    )


def test_mm_work_area_px_roundtrip() -> None:
    height_mm = 405.0
    ppm = 2.0
    x_mm, y_mm = 120.5, 85.2
    x_px, y_px = mm_to_work_area_px(x_mm, y_mm, height_mm, ppm)
    back_x, back_y = work_area_px_to_mm(x_px, y_px, height_mm, ppm)
    assert back_x == pytest.approx(x_mm, abs=0.01)
    assert back_y == pytest.approx(y_mm, abs=0.01)


def test_resolve_pixels_per_mm_from_max_edge() -> None:
    ppm = resolve_pixels_per_mm(406.0, 405.0, max_edge_px=1024)
    assert ppm == pytest.approx(1024.0 / 406.0, rel=0.01)


def test_synthetic_work_area_finds_rectangle() -> None:
    image, width_mm, height_mm, ppm = _synthetic_work_area_view()
    result = _geometry_from_image(image, width_mm=width_mm, height_mm=height_mm, ppm=ppm)
    assert len(result.objects) >= 1


def test_shape_detector_finds_circle() -> None:
    width_mm, height_mm, ppm = 400.0, 400.0, 2.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 240, dtype=np.uint8)
    cv2.circle(image, (width_px // 2, height_px // 2), int(45 * ppm), (25, 25, 25), -1)

    result = _geometry_from_image(image, width_mm=width_mm, height_mm=height_mm, ppm=ppm)
    assert len(result.objects) >= 1
    assert result.objects[0].shape == "circle"
    assert result.objects[0].width_px == pytest.approx(90.0 * ppm, abs=15.0)


def _textured_white_bed(width_px: int, height_px: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.integers(220, 245, (height_px, width_px, 3), dtype=np.uint8)
    speckle = rng.random((height_px, width_px)) < 0.08
    base[speckle] = rng.integers(200, 235, size=(int(speckle.sum()), 3))
    return base


def test_textured_bed_finds_rectangle() -> None:
    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = _textured_white_bed(width_px, height_px)
    size = int(90 * ppm)
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        image,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (30, 30, 30),
        -1,
    )

    result = _geometry_from_image(image, width_mm=width_mm, height_mm=height_mm, ppm=ppm)
    assert len(result.objects) >= 1
    assert result.objects[0].shape in ("rect", "rounded_rect")


def test_specular_glare_rejected() -> None:
    from app.services.shape_detector import _is_specular_glare

    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 25, dtype=np.uint8)
    cv2.circle(image, (350, 350), 35, (255, 255, 255), -1)
    cv2.rectangle(image, (500, 520), (650, 640), (50, 50, 180), -1)

    cfg = ShapeDetectorConfig()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mask = np.zeros(l_channel.shape, dtype=np.uint8)
    cv2.circle(mask, (350, 350), 35, 255, -1)
    glare_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert glare_contours
    assert _is_specular_glare(glare_contours[0], l_channel, bed_l=30.0, cfg=cfg)

    object_mask = np.zeros_like(mask)
    cv2.rectangle(object_mask, (500, 520), (650, 640), 255, -1)
    result = ShapeDetector(ShapeDetectorConfig(glare_suppression_enabled=False)).from_mask(
        image,
        object_mask,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    assert len(result.objects) >= 1


def test_fragmented_rectangle_merges_to_one_object() -> None:
    from app.services.shape_detector import _clean_mask

    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 240, dtype=np.uint8)
    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    for x in range(250, 550, 6):
        cv2.line(mask, (x, 300), (x, 500), 255, 4)

    cfg = ShapeDetectorConfig(morph_close_iterations=4, mask_morph_kernel_px=21)
    min_component = int(cfg.mask_min_component_area_mm2 * ppm * ppm)
    mask = _clean_mask(mask, cfg, min_component)

    result = ShapeDetector(cfg).from_mask(
        image,
        mask,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    assert len(result.objects) >= 1
    assert result.objects[0].shape in ("rect", "rounded_rect")


def test_hollow_circle_ring_detected_after_fill() -> None:
    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 240, dtype=np.uint8)
    radius = int(45 * ppm)
    cv2.circle(image, (width_px // 2, height_px // 2), radius, (30, 30, 30), max(4, radius // 8))

    result = _geometry_from_image(image, width_mm=width_mm, height_mm=height_mm, ppm=ppm)
    assert len(result.objects) >= 1
    assert result.objects[0].shape == "circle"


def test_split_merged_contour_finds_multiple_objects() -> None:
    from app.services.shape_detector import _split_contour_by_distance_peaks

    ppm = 2.0
    width_px = int(400.0 * ppm)
    height_px = int(400.0 * ppm)
    cfg = ShapeDetectorConfig()
    merged_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    cv2.circle(merged_mask, (200, 400), int(45 * ppm), 255, -1)
    cv2.circle(merged_mask, (600, 400), int(45 * ppm), 255, -1)
    cv2.line(merged_mask, (290, 400), (510, 400), 255, 6)

    cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    splits = _split_contour_by_distance_peaks(
        cnts[0],
        merged_mask,
        min_area_px=cfg.min_area_mm2 * ppm * ppm,
    )
    assert len(splits) >= 2


def test_pipeline_fastsam_only(monkeypatch) -> None:
    from app.services import work_area_renderer as war_module

    warped, _, _, ppm = _synthetic_work_area_view()

    def _fake_render(self, frame, *, pixels_per_mm=None, max_edge_px=1024):
        return war_module.WorkAreaView(
            image=warped,
            width_mm=400.0,
            height_mm=400.0,
            width_px=warped.shape[1],
            height_px=warped.shape[0],
            pixels_per_mm=ppm,
            origin_tag_id=0,
        )

    monkeypatch.setattr(war_module.WorkAreaRenderer, "render", _fake_render)

    class _FastSam:
        def segment_masks(self, frame):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            size = int(45 * ppm)
            cv2.rectangle(
                mask,
                (cx - size, cy - size),
                (cx + size, cy + size),
                255,
                -1,
            )
            return [mask]

        def render_overlay(self, frame, masks):
            from app.services.fastsam_detector import FastSamDetector

            return FastSamDetector.render_overlay(None, frame, masks)

        @property
        def active_device(self):
            return "hailo"

        def get_status(self):
            return {"loaded": True, "last_error": None}

    pipeline = ShapePipeline(fastsam=_FastSam())
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = pipeline.run(frame)
    assert result.response.fastsam_used is True
    assert result.response.count >= 1
    assert result.response.detections[0].segmentation_polygon_mm
    assert "classical_fused" not in result.stages
    assert "canny" not in result.stages
    assert "contours" not in result.stages
    assert "fastsam" in result.stages
    assert "fastsam_filtered" in result.stages
    assert "final" in result.stages


def test_mosaic_all_stages() -> None:
    stages = [
        ("raw", np.zeros((200, 300, 3), dtype=np.uint8)),
        ("warp", np.ones((150, 250, 3), dtype=np.uint8) * 40),
        ("mask", np.zeros((150, 250, 3), dtype=np.uint8)),
    ]
    mosaic = compose_stage_mosaic(stages, max_width_px=640, max_height_px=480, columns=2)
    assert mosaic.shape[0] > 0
    assert mosaic.shape[1] > 0
    jpeg = encode_jpeg(mosaic)
    assert jpeg.startswith(b"\xff\xd8")


def test_shape_pipeline_empty_fastsam(monkeypatch) -> None:
    from app.services import work_area_renderer as war_module

    warped, _, _, ppm = _synthetic_work_area_view()

    def _fake_render(self, frame, *, pixels_per_mm=None, max_edge_px=1024):
        return war_module.WorkAreaView(
            image=warped,
            width_mm=400.0,
            height_mm=400.0,
            width_px=warped.shape[1],
            height_px=warped.shape[0],
            pixels_per_mm=ppm,
            origin_tag_id=0,
        )

    monkeypatch.setattr(war_module.WorkAreaRenderer, "render", _fake_render)

    class _FastSam:
        def segment_masks(self, frame):
            return []

        @property
        def active_device(self):
            return None

        def get_status(self):
            return {"loaded": False, "last_error": "not loaded"}

    pipeline = ShapePipeline(fastsam=_FastSam())
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = pipeline.run(frame)
    assert result.response.count == 0
    assert result.response.fastsam_used is False
    assert "final" in result.stages
    jpeg = pipeline.render_debug_stage(result, "all", max_width_px=800, max_height_px=600)
    assert jpeg.startswith(b"\xff\xd8")


def test_final_stage_show_center_coords() -> None:
    from app.schemas.common import BoundingBox, Point2D
    from app.schemas.detection import DetectionItem
    from app.services import work_area_renderer as war_module
    from app.services.shape_pipeline import ShapePipeline

    warped, width_mm, height_mm, ppm = _synthetic_work_area_view()
    view = war_module.WorkAreaView(
        image=warped,
        width_mm=width_mm,
        height_mm=height_mm,
        width_px=warped.shape[1],
        height_px=warped.shape[0],
        pixels_per_mm=ppm,
        origin_tag_id=0,
    )
    item = DetectionItem(
        id=0,
        shape="rect",
        confidence=0.9,
        bbox_mm=BoundingBox(x_min=155.0, y_min=155.0, x_max=245.0, y_max=245.0),
        center_mm=Point2D(x=200.0, y=200.0),
        width_mm=90.0,
        height_mm=90.0,
        rotation_deg=0.0,
        oriented_box_mm=[
            Point2D(x=155.0, y=155.0),
            Point2D(x=245.0, y=155.0),
            Point2D(x=245.0, y=245.0),
            Point2D(x=155.0, y=245.0),
        ],
        segmentation_polygon_mm=[Point2D(x=200.0, y=200.0)],
    )
    pipeline = ShapePipeline()
    without = pipeline._render_final_stage(view, [item], show_center_coords=False)
    with_coords = pipeline._render_final_stage(view, [item], show_center_coords=True)
    assert without.shape == with_coords.shape
    assert not np.array_equal(without, with_coords)


def test_detect_api_requires_calibration() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/detection/detect")
        assert response.status_code == 409


def test_pipeline_uses_cpu_fastsam_when_injected(monkeypatch) -> None:
    from app.services import work_area_renderer as war_module

    warped, _, _, ppm = _synthetic_work_area_view()

    def _fake_render(self, frame, *, pixels_per_mm=None, max_edge_px=1024):
        return war_module.WorkAreaView(
            image=warped,
            width_mm=400.0,
            height_mm=400.0,
            width_px=warped.shape[1],
            height_px=warped.shape[0],
            pixels_per_mm=ppm,
            origin_tag_id=0,
        )

    monkeypatch.setattr(war_module.WorkAreaRenderer, "render", _fake_render)

    class _CpuFastSam:
        name = "cpu"

        def segment_masks(self, frame):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            size = int(45 * ppm)
            cv2.rectangle(
                mask,
                (cx - size, cy - size),
                (cx + size, cy + size),
                255,
                -1,
            )
            return [mask]

        def render_overlay(self, frame, masks):
            from app.services.fastsam_detector import FastSamDetector

            return FastSamDetector.render_overlay(None, frame, masks)

        @property
        def active_device(self):
            return "cpu"

    pipeline = ShapePipeline(fastsam=_CpuFastSam())
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = pipeline.run(frame)
    assert result.response.fastsam_used is True
    assert result.response.fastsam_device == "cpu"


def test_work_area_image_info_requires_calibration() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/detection/work-area-image/info")
        assert response.status_code == 409

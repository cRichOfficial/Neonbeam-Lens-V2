"""Shape detection pipeline tests — synthetic geometry, no Pi hardware."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.pipeline_debug_mosaic import compose_stage_mosaic, encode_jpeg
from app.services.shape_detector import ShapeDetector
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
    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cx = width_px // 2
    cy = height_px // 2
    size = int(90 * ppm)
    cv2.rectangle(
        image,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (220, 220, 220),
        -1,
    )
    return image, width_mm, height_mm, ppm


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


def test_shape_detector_finds_rectangle() -> None:
    image, width_mm, height_mm, ppm = _synthetic_work_area_view()
    detector = ShapeDetector()
    result = detector.detect(
        image,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    assert len(result.objects) >= 1
    obj = result.objects[0]
    assert obj.shape in ("rect", "rounded_rect")
    assert obj.width_px == pytest.approx(90.0 * ppm, abs=15.0)
    assert obj.height_px == pytest.approx(90.0 * ppm, abs=15.0)


def test_shape_detector_finds_circle() -> None:
    width_mm, height_mm, ppm = 400.0, 400.0, 2.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    cv2.circle(image, (width_px // 2, height_px // 2), int(45 * ppm), (200, 200, 200), -1)

    result = ShapeDetector().detect(
        image,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    assert len(result.objects) >= 1
    assert result.objects[0].shape == "circle"
    assert result.objects[0].width_px == pytest.approx(90.0 * ppm, abs=15.0)


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


def test_shape_pipeline_on_synthetic_frame(monkeypatch) -> None:
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

    pipeline = ShapePipeline(fastsam=_FastSam())
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result = pipeline.run(frame, backend="classical")
    assert result.response.count >= 1
    assert result.response.objects[0].width_mm == pytest.approx(90.0, abs=8.0)
    assert "final" in result.stages
    jpeg = pipeline.render_debug_stage(result, "all", max_width_px=800, max_height_px=600)
    assert jpeg.startswith(b"\xff\xd8")


def test_shapes_api_requires_calibration() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/detection/shapes")
        assert response.status_code == 409


def test_work_area_image_info_requires_calibration() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/detection/work-area-image/info")
        assert response.status_code == 409

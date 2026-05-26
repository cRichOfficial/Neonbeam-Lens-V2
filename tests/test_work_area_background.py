"""Work-area background reference tests — synthetic images, no camera."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import ConfigStore
from app.main import create_app
from app.services import work_area_background as bg_module
from app.services.shape_detector import ShapeDetector, ShapeDetectorConfig
from app.services.work_area_background import WorkAreaBackgroundStore
from app.services.work_area_renderer import WorkAreaView


def _write_config(tmp_path: Path, background_path: Path) -> ConfigStore:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
detection:
  background_storage_path: {background_path.as_posix()}
  max_edge_px: 1024
""".strip(),
        encoding="utf-8",
    )
    return ConfigStore(settings=type("S", (), {"config_path": config_path})())


@pytest.fixture
def bg_env(tmp_path, monkeypatch):
    bg_path = tmp_path / "work_area_background.png"
    store = _write_config(tmp_path, bg_path)
    monkeypatch.setattr("app.config.get_config_store", lambda: store)
    monkeypatch.setattr("app.config._config_store", store)
    monkeypatch.setattr("app.services.work_area_background.get_config_store", lambda: store)
    bg_module._store = None
    yield store, bg_path
    bg_module._store = None


def _sample_view(ppm: float = 2.0) -> WorkAreaView:
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    image = np.full((height_px, width_px, 3), 30, dtype=np.uint8)
    return WorkAreaView(
        image=image,
        width_mm=width_mm,
        height_mm=height_mm,
        width_px=width_px,
        height_px=height_px,
        pixels_per_mm=ppm,
        origin_tag_id=0,
    )


def test_render_diff_identical_images_are_black(bg_env) -> None:
    store = WorkAreaBackgroundStore()
    image = np.full((200, 200, 3), 240, dtype=np.uint8)
    diff = store.render_diff(image, image.copy())
    assert diff.shape == image.shape
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    assert int(np.max(gray[40:, :])) <= 2


def test_render_diff_shows_changed_region(bg_env) -> None:
    store = WorkAreaBackgroundStore()
    reference = np.full((200, 200, 3), 240, dtype=np.uint8)
    current = reference.copy()
    cv2.rectangle(current, (50, 50), (150, 150), (30, 30, 30), -1)
    diff = store.render_diff(current, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    assert int(gray[30, 30]) <= 5
    assert int(gray[100, 100]) > 50


def test_save_and_load_background(bg_env) -> None:
    _, bg_path = bg_env
    store = WorkAreaBackgroundStore()
    view = _sample_view()

    metadata = store.save(view, max_edge_px=1024)
    assert bg_path.exists()
    assert bg_path.with_suffix(".json").exists()

    loaded = store.load_image()
    assert loaded is not None
    assert loaded.shape == view.image.shape
    assert metadata.pixels_per_mm == pytest.approx(view.pixels_per_mm)


def test_stale_when_dimensions_change(bg_env) -> None:
    store = WorkAreaBackgroundStore()
    view = _sample_view(ppm=2.0)
    store.save(view, max_edge_px=1024)

    different_view = _sample_view(ppm=2.5)
    stale = store.stale_reason_for_view(different_view, max_edge_px=1024)
    assert stale == "dimension_mismatch"


def test_stale_when_scale_mismatch(bg_env) -> None:
    import json

    store = WorkAreaBackgroundStore()
    view = _sample_view(ppm=2.0)
    store.save(view, max_edge_px=1024)

    _, bg_path = bg_env
    metadata_path = bg_path.with_suffix(".json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["pixels_per_mm"] = 2.5
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    stale = store.stale_reason_for_view(view, max_edge_px=1024)
    assert stale == "scale_mismatch"


def test_bg_subtract_finds_added_rectangle(bg_env) -> None:
    from app.services.shape_fastsam_filter import extract_bg_subtract_mask

    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    reference = np.full((height_px, width_px, 3), 240, dtype=np.uint8)
    current = reference.copy()
    size = int(90 * ppm)
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        current,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (30, 30, 30),
        -1,
    )

    cfg = ShapeDetectorConfig(use_background_reference=True)
    mask = extract_bg_subtract_mask(current, reference, cfg, pixels_per_mm=ppm)
    assert np.count_nonzero(mask) > 0

    result = ShapeDetector(cfg).from_mask(
        current,
        mask,
        pixels_per_mm=ppm,
        width_mm=width_mm,
        height_mm=height_mm,
    )
    assert len(result.objects) >= 1


def test_bg_subtract_disabled_skips_reference(bg_env) -> None:
    from app.services.shape_fastsam_filter import extract_bg_subtract_mask

    ppm = 2.0
    width_mm, height_mm = 400.0, 400.0
    width_px = int(width_mm * ppm)
    height_px = int(height_mm * ppm)
    reference = np.full((height_px, width_px, 3), 30, dtype=np.uint8)
    current = reference.copy()
    size = int(90 * ppm)
    cx, cy = width_px // 2, height_px // 2
    cv2.rectangle(
        current,
        (cx - size // 2, cy - size // 2),
        (cx + size // 2, cy + size // 2),
        (210, 210, 210),
        -1,
    )

    cfg = ShapeDetectorConfig(use_background_reference=False)
    mask = extract_bg_subtract_mask(current, reference, cfg, pixels_per_mm=ppm)
    assert mask is None


def test_capture_background_requires_calibration(monkeypatch) -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/v1/detection/capture-background")
        assert response.status_code == 409


def test_background_status_absent_when_not_saved(bg_env, monkeypatch) -> None:
    from app.services import calibration_service as cal_module

    class _Cal:
        def is_calibrated(self) -> bool:
            return True

        @property
        def data(self):
            return type("D", (), {"work_area": object()})()

    class _Renderer:
        def require_work_area(self):
            return 400.0, 400.0, 0

    monkeypatch.setattr("app.api.detection.get_calibration_service", lambda: _Cal())
    monkeypatch.setattr("app.api.detection.get_work_area_renderer", lambda: _Renderer())

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/detection/background/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["present"] is False

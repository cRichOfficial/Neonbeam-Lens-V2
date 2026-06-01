"""AprilTag service tests — no Pi hardware."""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.apriltag_marker import generate_tag36h11_image
from app.services.apriltag_service import (
    AprilTagService,
    build_detection_failure_hint,
    frame_gray_stats,
    frame_to_gray_variants,
    merge_detections,
)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_frame_to_gray_variants_multi_includes_four_passes() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    passes = frame_to_gray_variants(frame, "multi")
    assert [name for name, _gray in passes] == ["raw", "bgr", "clahe", "sharpen"]
    assert all(gray.shape == (64, 64) for _name, gray in passes)


def test_frame_to_gray_variants_none_single_pass() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    passes = frame_to_gray_variants(frame, "none")
    assert len(passes) == 1
    assert passes[0][0] == "raw"


def test_merge_detections_keeps_best_decision_margin() -> None:
    merged = merge_detections(
        [
            {"id": 0, "decision_margin": 10.0, "center_px": [1, 1], "corners_px": [], "hamming": 0},
            {"id": 0, "decision_margin": 25.0, "center_px": [2, 2], "corners_px": [], "hamming": 0},
            {"id": 1, "decision_margin": 5.0, "center_px": [3, 3], "corners_px": [], "hamming": 0},
        ]
    )
    assert len(merged) == 2
    by_id = {det["id"]: det for det in merged}
    assert by_id[0]["decision_margin"] == 25.0
    assert by_id[0]["center_px"] == [2, 2]


def test_build_detection_failure_hint_high_exposure() -> None:
    hint = build_detection_failure_hint([], expected_ids=[0, 1, 2, 3], exposure_ms=750.0)
    assert "exposure_ms" in hint
    assert "750" in hint


def test_build_detection_failure_hint_partial_detection() -> None:
    hint = build_detection_failure_hint(
        [{"id": 0}, {"id": 2}],
        expected_ids=[0, 1, 2, 3],
        exposure_ms=30.0,
    )
    assert "missing" in hint.lower() or "[1, 3]" in hint


def test_detect_merges_multipass_results(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AprilTagService()
    call_count = {"n": 0}

    def fake_detect(_gray: np.ndarray) -> list:
        call_count["n"] += 1
        if call_count["n"] == 3:
            det = MagicMock()
            det.tag_id = 7
            det.center = (10.0, 20.0)
            det.corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
            det.hamming = 0
            det.decision_margin = 42.0
            return [det]
        return []

    detector = MagicMock()
    detector.detect.side_effect = fake_detect
    monkeypatch.setattr(service, "_get_detector", lambda _family: detector)

    config = type(
        "Cfg",
        (),
        {
            "family": "tag36h11",
            "preprocess": "multi",
        },
    )()
    monkeypatch.setattr(
        "app.services.apriltag_service.get_config_store",
        lambda: type("Store", (), {"config": type("Root", (), {"apriltag": config})()})(),
    )

    results = service.detect(np.zeros((32, 32, 3), dtype=np.uint8))
    assert len(results) == 1
    assert results[0]["id"] == 7
    assert results[0]["decision_margin"] == 42.0


def test_apriltag_detections_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.calibration.get_apriltag_service",
        lambda: type(
            "S",
            (),
            {
                "detect": lambda _frame, family=None: [
                    {
                        "id": 0,
                        "center_px": [100.0, 100.0],
                        "corners_px": [[90, 90], [110, 90], [110, 110], [90, 110]],
                        "hamming": 0,
                        "decision_margin": 55.0,
                    }
                ]
            },
        )(),
    )
    response = client.get("/api/v1/calibration/apriltag/detections")
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert data["expected_ids"] == [0, 1, 2, 3]
    assert "exposure_ms" in data
    assert "frame_size" in data
    assert "frame_stats" in data
    assert "camera_mode" in data


def test_calibration_error_includes_detected_ids(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.calibration.get_apriltag_service",
        lambda: type(
            "S",
            (),
            {
                "detect": lambda _frame, family=None: [
                    {
                        "id": 5,
                        "center_px": [100.0, 100.0],
                        "corners_px": [[90, 90], [110, 90], [110, 110], [90, 110]],
                        "hamming": 0,
                        "decision_margin": 30.0,
                    }
                ]
            },
        )(),
    )

    response = client.post(
        "/api/v1/calibration/apriltag",
        json={
            "origin_tag_id": 0,
            "size_mm": 30,
            "tag_ids": [0, 1, 2, 3],
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["detected_ids"] == [5]
    assert detail["missing_ids"] == [0, 1, 2, 3]
    assert detail["detection_count"] == 1
    assert "hint" in detail


@pytest.mark.skipif(
    not hasattr(cv2.aruco, "generateImageMarker"),
    reason="OpenCV ArUco marker generation unavailable",
)
def test_aruco_fallback_detects_synthetic_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    tag = generate_tag36h11_image(0, 120)
    canvas = np.full((200, 200, 3), 255, dtype=np.uint8)
    canvas[40:160, 40:160] = tag

    service = AprilTagService()
    monkeypatch.setattr(service, "_detect_pupil", lambda _frame, _family: [])

    config = type(
        "Cfg",
        (),
        {
            "family": "tag36h11",
            "preprocess": "multi",
            "aruco_fallback": True,
        },
    )()
    monkeypatch.setattr(
        "app.services.apriltag_service.get_config_store",
        lambda: type("Store", (), {"config": type("Root", (), {"apriltag": config})()})(),
    )

    results = service.detect(canvas)
    assert results
    assert results[0]["id"] == 0


def test_frame_gray_stats_reports_contrast() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:16] = 255
    stats = frame_gray_stats(frame)
    assert stats["gray_std"] > 50

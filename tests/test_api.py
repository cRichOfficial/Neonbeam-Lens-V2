from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.calibration import AprilTagPdfRequest
from app.services.apriltag_pdf_service import generate_apriltag_pdf


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "camera" in payload
    assert "calibration" in payload
    assert "npu" in payload
    assert "fastsam" in payload["detection"]


def test_camera_settings(client: TestClient) -> None:
    from app.config import get_config_store

    response = client.get("/api/v1/camera/settings")
    assert response.status_code == 200
    data = response.json()
    assert "exposure_ms" in data
    expected_ms = get_config_store().config.camera.exposure_us / 1000.0
    assert data["exposure_ms"] == expected_ms
    assert "mount_height_mm" in data


def test_camera_settings_update_exposure_ms(client: TestClient) -> None:
    response = client.put("/api/v1/camera/settings", json={"exposure_ms": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["exposure_ms"] == 20.0

    get_response = client.get("/api/v1/camera/settings")
    assert get_response.json()["exposure_ms"] == 20.0


def test_camera_settings_persist_long_exposure(client: TestClient) -> None:
    response = client.put("/api/v1/camera/settings", json={"exposure_ms": 750})
    assert response.status_code == 200
    data = response.json()
    assert data["exposure_ms"] == 750.0

    get_response = client.get("/api/v1/camera/settings")
    assert get_response.json()["exposure_ms"] == 750.0


def test_camera_snapshot(client: TestClient) -> None:
    response = client.get("/api/v1/camera/snapshot")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_calibration_status(client: TestClient) -> None:
    response = client.get("/api/v1/calibration/status")
    assert response.status_code == 200
    data = response.json()
    assert data["calibrated"] is False


def test_apriltag_preview(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockAprilTagService:
        def detect(self, frame, family=None):
            return []

        def draw_detections(self, frame, tags, label_corners=True):
            return frame.copy()

    monkeypatch.setattr("app.api.calibration.get_apriltag_service", MockAprilTagService)
    response = client.post("/api/v1/calibration/apriltag/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_generate_apriltag_pdf() -> None:
    pdf = generate_apriltag_pdf(AprilTagPdfRequest(size_mm=20, safe_zone_padding_mm=5))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_generate_apriltag_pdf_via_api(client: TestClient) -> None:
    response = client.post(
        "/api/v1/calibration/apriltag/generate-pdf",
        json={"size_mm": 20, "safe_zone_padding_mm": 5},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_detection_endpoint_requires_calibration(client: TestClient) -> None:
    response = client.get("/api/v1/detection/detect")
    assert response.status_code == 409


def test_detection_endpoint_cache_control(client: TestClient, monkeypatch) -> None:
    class _Cal:
        def is_calibrated(self) -> bool:
            return True

        @property
        def data(self):
            return type("D", (), {"work_area": object()})()

    class _Pipeline:
        def run(self, frame, **kwargs):
            from app.schemas.detection import DetectionResponse

            return type(
                "R",
                (),
                {
                    "response": DetectionResponse(
                        backend="cpu",
                        calibrated=True,
                        count=0,
                        detections=[],
                    )
                },
            )()

    monkeypatch.setattr("app.api.detection.get_calibration_service", lambda: _Cal())
    monkeypatch.setattr("app.api.detection.get_shape_pipeline", lambda: _Pipeline())

    response = client.get("/api/v1/detection/detect")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"


def test_debug_image_requires_calibration(client: TestClient) -> None:
    response = client.get("/api/v1/detection/debug-image")
    assert response.status_code == 409

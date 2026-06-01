"""AprilTag PNG generator tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.calibration import AprilTagPngRequest
from app.services.apriltag_png_service import PRINT_DPI, _mm_to_px, generate_apriltag_png


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_mm_to_px_at_300_dpi() -> None:
    assert _mm_to_px(24) == round(24 * PRINT_DPI / 25.4)


def test_generate_apriltag_png_dimensions() -> None:
    png = generate_apriltag_png(AprilTagPngRequest(tag_id=0, size_mm=20, safe_zone_mm=2))
    assert png.startswith(b"\x89PNG")
    expected_px = _mm_to_px(24)
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (expected_px, expected_px, 3)


def test_generate_apriltag_png_invalid_tag_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        generate_apriltag_png(AprilTagPngRequest(tag_id=9999, size_mm=20))
    assert exc_info.value.status_code == 422


def test_generate_apriltag_png_via_api(client: TestClient) -> None:
    response = client.post(
        "/api/v1/calibration/apriltag/generate-png",
        json={"tag_id": 3, "size_mm": 20, "safe_zone_mm": 2},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert 'attachment; filename="apriltag_3_20mm.png"' in response.headers["content-disposition"]
    assert response.content.startswith(b"\x89PNG")


def test_generate_apriltag_png_invalid_tag_id_via_api(client: TestClient) -> None:
    response = client.post(
        "/api/v1/calibration/apriltag/generate-png",
        json={"tag_id": 9999, "size_mm": 20},
    )
    assert response.status_code == 422

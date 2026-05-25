"""Calibration tests — fully synthetic; no camera or Pi hardware required.

Hardware verification (AprilTag cal on the remote Pi) is manual; see README.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import BedConfig, ConfigStore
from app.schemas.calibration import AprilTagSpec
from app.services import calibration_service as calibration_service_module
from app.services.bed_frame import physical_corners_for_tag, tag_corner_offsets_mm
from app.services.calibration_service import (
    CalibrationError,
    CalibrationService,
    _best_corner_permutation,
)
from app.services.camera_intrinsics import (
    camera_matrix_from_hfov,
    estimate_distortion_from_tags,
    mean_corner_reprojection_mm,
    undistort_points,
)
from app.services.transform_service import TransformService


def _write_config(tmp_path: Path, bed_yaml: str) -> ConfigStore:
    calibration_path = tmp_path / "calibration.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bed:
{bed_yaml}
calibration:
  max_reprojection_error_mm: 2.0
  storage_path: {calibration_path.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    return ConfigStore(settings=type("S", (), {"config_path": config_path})())


@pytest.fixture
def calibration_env(tmp_path, monkeypatch):
    store = _write_config(
        tmp_path,
        """
  width_mm: 400
  height_mm: 400
  origin: bottom_left
  y_axis: up
""",
    )
    monkeypatch.setattr("app.config.get_config_store", lambda: store)
    monkeypatch.setattr("app.config._config_store", store)
    monkeypatch.setattr("app.services.calibration_service.get_config_store", lambda: store)
    calibration_service_module._calibration_service = None
    yield store
    calibration_service_module._calibration_service = None


def _mock_apriltag_service(detections: list[dict]):
    return type("S", (), {"detect": lambda _frame, family=None: detections})()


STANDARD_TAGS = [
    AprilTagSpec(id=0, x_mm=0, y_mm=0, size_mm=30),
    AprilTagSpec(id=1, x_mm=400, y_mm=0, size_mm=30),
    AprilTagSpec(id=2, x_mm=0, y_mm=400, size_mm=30),
    AprilTagSpec(id=3, x_mm=400, y_mm=400, size_mm=30),
]


def _bed_to_px_homography() -> np.ndarray:
    bed_pts = np.array([[0, 0], [400, 0], [0, 400], [400, 400]], dtype=np.float32)
    px_pts = np.array([[120, 880], [880, 880], [120, 120], [880, 120]], dtype=np.float32)
    homography, _ = cv2.findHomography(bed_pts, px_pts, method=0)
    assert homography is not None
    return homography


def _bed_points_to_px(points_mm: np.ndarray, bed_to_px: np.ndarray) -> np.ndarray:
    reshaped = points_mm.reshape(-1, 1, 2).astype(np.float32)
    transformed = cv2.perspectiveTransform(reshaped, bed_to_px).reshape(-1, 2)
    return transformed


def _synthetic_detections(
    tag_specs: list[AprilTagSpec],
    bed: BedConfig,
    bed_to_px: np.ndarray,
) -> list[dict]:
    detections: list[dict] = []
    for spec in tag_specs:
        corners_mm = physical_corners_for_tag(
            spec.x_mm,
            spec.y_mm,
            spec.size_mm,
            bed,
            rotation_deg=spec.rotation_deg,
        )
        corners_px = _bed_points_to_px(corners_mm, bed_to_px)
        center_px = corners_px.mean(axis=0)
        detections.append(
            {
                "id": spec.id,
                "center_px": center_px.tolist(),
                "corners_px": corners_px.tolist(),
                "hamming": 0,
                "decision_margin": 100.0,
            }
        )
    return detections


def _apply_pinhole_distortion(
    undist_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Forward pinhole distortion matching cv2.undistortPoints (for synthetic tests)."""
    points = np.asarray(undist_points_px, dtype=np.float64).reshape(-1, 2)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.reshape(-1)[:5]

    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 +  2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([x_dist * fx + cx, y_dist * fy + cy]).astype(np.float32)


def _apply_distortion_to_detections(
    detections: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[dict]:
    distorted: list[dict] = []
    for det in detections:
        corners = np.array(det["corners_px"], dtype=np.float32)
        distorted_corners = _apply_pinhole_distortion(corners, camera_matrix, dist_coeffs)
        distorted.append(
            {
                **det,
                "corners_px": distorted_corners.tolist(),
                "center_px": distorted_corners.mean(axis=0).tolist(),
            }
        )
    return distorted


def _collect_corner_pairs(
    tag_specs: list[AprilTagSpec],
    bed: BedConfig,
    detections: list[dict],
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    det_by_id = {det["id"]: det for det in detections}
    image_corners: list[list[float]] = []
    bed_corners: list[list[float]] = []
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    for spec in tag_specs:
        det = det_by_id[spec.id]
        corners_mm = physical_corners_for_tag(
            spec.x_mm,
            spec.y_mm,
            spec.size_mm,
            bed,
            rotation_deg=spec.rotation_deg,
        )
        image_corners_arr = np.array(det["corners_px"], dtype=np.float32)
        rough_undist = undistort_points(image_corners_arr, camera_matrix, zero_dist, "pinhole")
        aligned_mm = _best_corner_permutation(rough_undist, corners_mm)
        image_corners.extend(image_corners_arr.tolist())
        bed_corners.extend(aligned_mm.tolist())
    return np.array(image_corners, dtype=np.float32), np.array(bed_corners, dtype=np.float32)


def test_tag_corner_offsets_y_up_vs_y_down() -> None:
    bed_up = BedConfig(width_mm=400, height_mm=400, origin="bottom_left", y_axis="up")
    bed_down = BedConfig(width_mm=400, height_mm=400, origin="top_left", y_axis="down")
    up = tag_corner_offsets_mm(20, bed_up)
    down = tag_corner_offsets_mm(20, bed_down)
    assert np.allclose(up[0], [-10, 10])
    assert np.allclose(down[0], [-10, -10])


def test_calibrate_synthetic_bottom_left_y_up(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, matched = service.calibrate(frame, STANDARD_TAGS, persist=False)

    assert len(matched) == 4
    assert result.reprojection_error_mm < 0.5


def test_calibrate_top_left_y_down(tmp_path, monkeypatch) -> None:
    store = _write_config(
        tmp_path,
        """
  width_mm: 400
  height_mm: 400
  origin: top_left
  y_axis: down
""",
    )
    monkeypatch.setattr("app.config.get_config_store", lambda: store)
    monkeypatch.setattr("app.services.calibration_service.get_config_store", lambda: store)
    calibration_service_module._calibration_service = None

    tags = [
        AprilTagSpec(id=0, x_mm=0, y_mm=0, size_mm=30),
        AprilTagSpec(id=1, x_mm=400, y_mm=0, size_mm=30),
        AprilTagSpec(id=2, x_mm=0, y_mm=400, size_mm=30),
        AprilTagSpec(id=3, x_mm=400, y_mm=400, size_mm=30),
    ]
    bed = store.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(tags, bed, bed_to_px)
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(frame, tags, persist=False)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_with_reversed_corner_winding(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    for det in detections:
        det["corners_px"] = list(reversed(det["corners_px"]))

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(frame, STANDARD_TAGS, persist=False)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_with_rotated_tag(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    tags = [
        AprilTagSpec(id=0, x_mm=0, y_mm=0, size_mm=30, rotation_deg=90),
        AprilTagSpec(id=1, x_mm=400, y_mm=0, size_mm=30),
        AprilTagSpec(id=2, x_mm=0, y_mm=400, size_mm=30),
        AprilTagSpec(id=3, x_mm=400, y_mm=400, size_mm=30),
    ]
    detections = _synthetic_detections(tags, bed, bed_to_px)
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(frame, tags, persist=False)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_size_mismatch_reports_corner_error(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    truth_tags = STANDARD_TAGS
    request_tags = [spec.model_copy(update={"size_mm": 45}) for spec in STANDARD_TAGS]
    detections = _synthetic_detections(truth_tags, bed, bed_to_px)
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    with pytest.raises(CalibrationError) as exc_info:
        service.calibrate(frame, request_tags, persist=False)

    detail = exc_info.value.detail
    assert detail["center_error_mm"] < 2.0
    assert detail["corner_error_mm"] > 2.0
    assert "size_mm" in detail.get("hint", "")


def test_calibration_status_includes_bed_frame(calibration_env) -> None:
    service = CalibrationService()
    status = service.get_status()
    assert status["bed_frame"] == "origin=bottom_left, y_axis=up"


def test_calibration_status_includes_distortion_after_calibrate(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    detections = _synthetic_detections(STANDARD_TAGS, bed, _bed_to_px_homography())
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )
    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    service.calibrate(frame, STANDARD_TAGS, persist=True)
    status = service.get_status()
    assert status["distortion"] is not None
    assert "k1" in status["distortion"]
    assert status["distortion"]["hfov_deg"] == calibration_env.config.camera.hfov_deg


def test_estimate_distortion_recovers_barrel_k1(calibration_env) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    hfov_deg = calibration_env.config.camera.hfov_deg
    camera_matrix = camera_matrix_from_hfov(1000, 1000, hfov_deg)
    image_corners, bed_corners = _collect_corner_pairs(
        STANDARD_TAGS, bed, detections, camera_matrix
    )
    true_k1 = -0.18
    true_dist = np.array([[true_k1], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
    distorted_corners = _apply_pinhole_distortion(image_corners, camera_matrix, true_dist)

    estimated_dist, error = estimate_distortion_from_tags(
        distorted_corners,
        bed_corners,
        camera_matrix,
        "pinhole",
    )
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    zero_error = mean_corner_reprojection_mm(
        distorted_corners, bed_corners, camera_matrix, zero_dist, "pinhole"
    )
    assert error < zero_error
    assert float(estimated_dist[0, 0]) < -0.03
    assert abs(float(estimated_dist[0, 0]) - true_k1) < 0.35


def test_calibrate_with_synthetic_barrel_distortion(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    hfov_deg = calibration_env.config.camera.hfov_deg
    camera_matrix = camera_matrix_from_hfov(1000, 1000, hfov_deg)
    true_dist = np.array([[-0.18], [0.02], [0.0], [0.0], [0.0]], dtype=np.float64)
    detections = _apply_distortion_to_detections(detections, camera_matrix, true_dist)

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, matched = service.calibrate(frame, STANDARD_TAGS, persist=False)

    assert len(matched) == 4
    assert result.reprojection_error_mm < 2.0
    assert result.intrinsics is not None
    assert result.intrinsics.dist_coeffs[0, 0] < -0.05


def test_transform_px_mm_round_trip(calibration_env, monkeypatch) -> None:
    bed = calibration_env.config.bed
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    hfov_deg = calibration_env.config.camera.hfov_deg
    camera_matrix = camera_matrix_from_hfov(1000, 1000, hfov_deg)
    dist = np.array([[-0.08], [0.01], [0.0], [0.0], [0.0]], dtype=np.float64)
    detections = _apply_distortion_to_detections(detections, camera_matrix, dist)
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    service.calibrate(frame, STANDARD_TAGS, persist=True)
    transform = TransformService(calibration_service=service)

    raw_pts = np.array([[500, 500], [350, 650], [650, 350], [450, 550]], dtype=np.float32)
    mm_pts = transform.px_to_mm(raw_pts)
    back = transform.mm_to_px(mm_pts)
    assert np.allclose(back, raw_pts, atol=2.0)

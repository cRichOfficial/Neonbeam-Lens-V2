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
from app.services.work_area import WorkArea
from app.services.transform_service import TransformService


def _write_config(tmp_path: Path, bed_yaml: str) -> ConfigStore:
    calibration_path = tmp_path / "calibration.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bed:
{bed_yaml}
calibration:
  max_reprojection_error_mm: 2.5
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

STANDARD_WORK_AREA = WorkArea(400, 400, origin_tag_id=0, size_mm=30)


def _synthetic_bed(frame) -> BedConfig:
    return BedConfig(
        width_mm=400,
        height_mm=400,
        origin=frame.origin,
        y_axis=frame.y_axis,
    )


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


def _detections_from_centers(
    centers: dict[int, tuple[float, float]],
    *,
    half_edge_px: float = 15.0,
) -> list[dict]:
    detections: list[dict] = []
    for tag_id, (cx, cy) in centers.items():
        corners_px = [
            [cx - half_edge_px, cy - half_edge_px],
            [cx + half_edge_px, cy - half_edge_px],
            [cx + half_edge_px, cy + half_edge_px],
            [cx - half_edge_px, cy + half_edge_px],
        ]
        detections.append(
            {
                "id": tag_id,
                "center_px": [cx, cy],
                "corners_px": corners_px,
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
    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, matched = service.calibrate(frame, STANDARD_TAGS, persist=False, work_area=STANDARD_WORK_AREA)

    assert len(matched) == 4
    assert result.reprojection_error_mm < 0.5


def test_calibrate_top_left_y_down(tmp_path, monkeypatch) -> None:
    store = _write_config(
        tmp_path,
        """
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
    bed = BedConfig(width_mm=400, height_mm=400, origin="top_left", y_axis="down")
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(tags, bed, bed_to_px)
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(frame, tags, persist=False, work_area=STANDARD_WORK_AREA)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_with_reversed_corner_winding(calibration_env, monkeypatch) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
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
    result, _ = service.calibrate(frame, STANDARD_TAGS, persist=False, work_area=STANDARD_WORK_AREA)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_with_rotated_tag(calibration_env, monkeypatch) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
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
    result, _ = service.calibrate(frame, tags, persist=False, work_area=STANDARD_WORK_AREA)
    assert result.reprojection_error_mm < 0.5


def test_calibrate_size_mismatch_reports_corner_error(calibration_env, monkeypatch) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
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
        service.calibrate(
            frame,
            request_tags,
            persist=False,
            work_area=WorkArea(400, 400, origin_tag_id=0, size_mm=45),
        )

    detail = exc_info.value.detail
    assert detail["center_error_mm"] < 2.0
    assert detail["corner_error_mm"] > 2.0
    assert "size_mm" in detail.get("hint", "")


def test_calibration_status_includes_bed_frame(calibration_env) -> None:
    service = CalibrationService()
    status = service.get_status()
    assert status["bed_frame"] == "origin=bottom_left, y_axis=up"


def test_calibration_status_includes_distortion_after_calibrate(calibration_env, monkeypatch) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
    detections = _synthetic_detections(STANDARD_TAGS, bed, _bed_to_px_homography())
    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )
    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    service.calibrate(frame, STANDARD_TAGS, persist=True, work_area=STANDARD_WORK_AREA)
    status = service.get_status()
    assert status["distortion"] is not None
    assert "k1" in status["distortion"]
    assert status["distortion"]["hfov_deg"] == calibration_env.config.camera.hfov_deg


def test_estimate_distortion_recovers_barrel_k1(calibration_env) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
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
    bed = _synthetic_bed(calibration_env.config.bed)
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
    result, matched = service.calibrate(frame, STANDARD_TAGS, persist=False, work_area=STANDARD_WORK_AREA)

    assert len(matched) == 4
    assert result.reprojection_error_mm < 2.0
    assert result.intrinsics is not None
    assert result.intrinsics.dist_coeffs[0, 0] < -0.05


def test_transform_px_mm_round_trip(calibration_env, monkeypatch) -> None:
    bed = _synthetic_bed(calibration_env.config.bed)
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
    service.calibrate(frame, STANDARD_TAGS, persist=True, work_area=STANDARD_WORK_AREA)
    transform = TransformService(calibration_service=service)

    raw_pts = np.array([[500, 500], [350, 650], [650, 350], [450, 550]], dtype=np.float32)
    mm_pts = transform.px_to_mm(raw_pts)
    back = transform.mm_to_px(mm_pts)
    assert np.allclose(back, raw_pts, atol=2.0)


def test_derive_work_area_from_synthetic_detections(calibration_env) -> None:
    from app.services.work_area import derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)

    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )

    assert derived.work_area.width_mm == pytest.approx(400, abs=5)
    assert derived.work_area.height_mm == pytest.approx(400, abs=5)
    assert derived.work_area.origin_tag_id == 0
    assert len(derived.tag_specs) == 4
    origin = next(spec for spec in derived.tag_specs if spec.id == 0)
    br = next(spec for spec in derived.tag_specs if spec.id == 1)
    assert origin.x_mm == pytest.approx(0, abs=0.1)
    assert origin.y_mm == pytest.approx(0, abs=0.1)
    assert br.x_mm == pytest.approx(derived.work_area.width_mm, abs=5)
    assert br.y_mm == pytest.approx(0, abs=0.1)


def test_classify_corner_tags_perspective_skew(calibration_env) -> None:
    from app.services.work_area import _classify_corner_tags, derive_work_area_from_detections

    centers = {
        0: (100.0, 650.0),
        1: (900.0, 400.0),
        2: (100.0, 150.0),
        3: (850.0, 200.0),
    }
    center_arrays = {tag_id: np.array(point, dtype=np.float64) for tag_id, point in centers.items()}
    roles = _classify_corner_tags(0, center_arrays)
    assert roles == {"origin": 0, "br": 1, "tl": 2, "tr": 3}

    detections = _detections_from_centers(centers)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )
    br = next(spec for spec in derived.tag_specs if spec.id == 1)
    assert br.x_mm == pytest.approx(derived.work_area.width_mm, abs=5)
    assert br.y_mm == pytest.approx(0, abs=0.1)
    assert derived.work_area.width_mm > 0
    assert derived.work_area.height_mm > 0


def test_calibrate_corner_defined_mode(calibration_env, monkeypatch) -> None:
    from app.services.work_area import derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, matched = service.calibrate(
        frame,
        derived.tag_specs,
        persist=True,
        work_area=derived.work_area,
    )

    assert len(matched) == 4
    assert result.reprojection_error_mm < 0.5
    assert result.work_area is not None
    assert result.work_area.width_mm == pytest.approx(400, abs=5)
    assert result.work_area.height_mm == pytest.approx(400, abs=5)
    assert service.get_effective_bed().width_mm == pytest.approx(400, abs=5)
    status = service.get_status()
    assert status["work_area"] is not None
    assert status["work_area"]["width_mm"] == pytest.approx(400, abs=5)
    assert result.tag_size_validation is not None
    assert result.tag_size_validation.converged
    assert result.tag_size_validation.mean_mm == pytest.approx(30, abs=0.5)
    assert result.tag_size_validation.max_error_mm <= 1.0


def test_measure_tag_edge_lengths_mm_synthetic(calibration_env, monkeypatch) -> None:
    from app.services.work_area import derive_work_area_from_detections, measure_tag_edge_lengths_mm

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(
        frame,
        derived.tag_specs,
        persist=True,
        work_area=derived.work_area,
    )
    det_by_id = {det["id"]: det for det in detections}
    spec_by_id = {spec.id: spec for spec in derived.tag_specs}
    matched = [(det_by_id[tag_id], spec_by_id[tag_id]) for tag_id in [0, 1, 2, 3]]

    sizes = measure_tag_edge_lengths_mm(matched, result.homography, result.intrinsics)
    assert len(sizes) == 4
    for tag_id, edge_mm in sizes.items():
        assert edge_mm == pytest.approx(30, abs=0.5), f"tag {tag_id} measured {edge_mm}"


def test_scale_refinement_recovers_inflated_work_area(calibration_env, monkeypatch) -> None:
    from app.services.work_area import WorkArea, derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )
    scale = 1.05
    inflated_specs = [
        spec.model_copy(update={"x_mm": spec.x_mm * scale, "y_mm": spec.y_mm * scale})
        for spec in derived.tag_specs
    ]
    inflated_work_area = WorkArea(
        width_mm=derived.work_area.width_mm * scale,
        height_mm=derived.work_area.height_mm * scale,
        origin_tag_id=derived.work_area.origin_tag_id,
        size_mm=derived.work_area.size_mm,
    )

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(
        frame,
        inflated_specs,
        persist=True,
        work_area=inflated_work_area,
    )

    assert result.tag_size_validation is not None
    assert result.tag_size_validation.scale_iterations >= 1
    assert result.tag_size_validation.converged
    assert result.tag_size_validation.mean_mm == pytest.approx(30, abs=0.5)
    assert result.work_area.width_mm == pytest.approx(400, abs=5)


def test_derive_work_area_uses_corner_transform(calibration_env) -> None:
    from app.services.work_area import derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    width, height = 1000, 1000
    camera_matrix = camera_matrix_from_hfov(width, height, 102.0)
    dist = np.array([[-0.2, 0.05, 0.0, 0.0, 0.0]], dtype=np.float64).reshape(5, 1)
    distorted = _apply_distortion_to_detections(detections, camera_matrix, dist)

    def undistort_point(point: np.ndarray) -> np.ndarray:
        return undistort_points(point.reshape(1, 2), camera_matrix, dist, "pinhole")[0]

    raw = derive_work_area_from_detections(
        distorted,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )
    corrected = derive_work_area_from_detections(
        distorted,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
        center_transform=undistort_point,
        corner_transform=undistort_point,
    )

    assert corrected.work_area.width_mm == pytest.approx(400, abs=5)
    assert abs(corrected.work_area.width_mm - raw.work_area.width_mm) > 1.0


def _stretch_tag_horizontal_edges(detections: list[dict], scale: float) -> list[dict]:
    stretched: list[dict] = []
    for det in detections:
        corners = np.array(det["corners_px"], dtype=np.float64)
        center = corners.mean(axis=0)
        for index in range(4):
            edge = corners[(index + 1) % 4] - corners[index]
            if abs(edge[0]) >= abs(edge[1]):
                for vertex in (index, (index + 1) % 4):
                    corners[vertex, 0] = center[0] + (corners[vertex, 0] - center[0]) * scale
        stretched.append(
            {
                **det,
                "corners_px": corners.tolist(),
                "center_px": corners.mean(axis=0).tolist(),
            }
        )
    return stretched


def test_per_axis_derive_with_anisotropic_tag_edges(calibration_env) -> None:
    from app.services.work_area import derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _stretch_tag_horizontal_edges(
        _synthetic_detections(STANDARD_TAGS, bed, bed_to_px),
        scale=1.05,
    )

    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )

    assert derived.mm_per_px_x is not None
    assert derived.mm_per_px_y is not None
    assert derived.mm_per_px_x < derived.mm_per_px_y
    assert derived.work_area.height_mm == pytest.approx(400, abs=8)
    assert derived.work_area.width_mm == pytest.approx(
        400 * derived.mm_per_px_x / derived.mm_per_px_y,
        abs=5,
    )


def test_anisotropic_refinement_recovers_width_only_inflation(calibration_env, monkeypatch) -> None:
    from app.services.work_area import WorkArea, derive_work_area_from_detections

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )
    width_scale = 1.05
    inflated_specs = [
        spec.model_copy(
            update={
                "x_mm": spec.x_mm * width_scale,
                "y_mm": spec.y_mm,
            }
        )
        for spec in derived.tag_specs
    ]
    inflated_work_area = WorkArea(
        width_mm=derived.work_area.width_mm * width_scale,
        height_mm=derived.work_area.height_mm,
        origin_tag_id=derived.work_area.origin_tag_id,
        size_mm=derived.work_area.size_mm,
    )

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(
        frame,
        inflated_specs,
        persist=True,
        work_area=inflated_work_area,
        mm_per_px_x=derived.mm_per_px_x,
        mm_per_px_y=derived.mm_per_px_y,
    )

    assert result.tag_size_validation is not None
    assert result.tag_size_validation.scale_x_iterations >= 1
    assert result.tag_size_validation.converged
    assert result.work_area.width_mm == pytest.approx(400, abs=5)
    assert result.work_area.height_mm == pytest.approx(400, abs=5)


def test_persisted_work_area_matches_final_homography(calibration_env, monkeypatch) -> None:
    from app.services.work_area import derive_work_area_from_detections, finalize_work_area_from_homography

    bed = _synthetic_bed(calibration_env.config.bed)
    bed_to_px = _bed_to_px_homography()
    detections = _synthetic_detections(STANDARD_TAGS, bed, bed_to_px)
    derived = derive_work_area_from_detections(
        detections,
        origin_tag_id=0,
        size_mm=30,
        tag_ids=[0, 1, 2, 3],
    )

    monkeypatch.setattr(
        "app.services.calibration_service.get_apriltag_service",
        lambda: _mock_apriltag_service(detections),
    )

    service = CalibrationService()
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    result, _ = service.calibrate(
        frame,
        derived.tag_specs,
        persist=True,
        work_area=derived.work_area,
        mm_per_px_x=derived.mm_per_px_x,
        mm_per_px_y=derived.mm_per_px_y,
    )

    det_by_id = {det["id"]: det for det in detections}
    spec_by_id = {spec.id: spec for spec in derived.tag_specs}
    matched = [(det_by_id[tag_id], spec_by_id[tag_id]) for tag_id in [0, 1, 2, 3]]
    expected = finalize_work_area_from_homography(
        matched,
        result.homography,
        result.intrinsics,
        origin_tag_id=0,
        size_mm=30,
    )
    assert result.work_area is not None
    assert result.work_area.width_mm == pytest.approx(expected.width_mm, abs=0.5)
    assert result.work_area.height_mm == pytest.approx(expected.height_mm, abs=0.5)

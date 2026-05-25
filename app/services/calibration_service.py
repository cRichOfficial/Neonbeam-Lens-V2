from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.config import get_config_store
from app.schemas.calibration import AprilTagSpec, DetectedAprilTag
from app.services.apriltag_service import get_apriltag_service
from app.services.bed_frame import bed_center_mm, frame_description, physical_corners_for_tag
from app.services.camera_intrinsics import (
    CameraIntrinsics,
    distort_points,
    estimate_distortion_from_tags,
    mean_corner_reprojection_mm,
    resolve_camera_intrinsics,
    undistort_points,
)
from app.services.work_area import (
    AxisEdgeLengths,
    TagSizeValidation,
    WorkArea,
    effective_bed,
    finalize_work_area_from_homography,
    measure_tag_axis_edge_lengths_mm,
    measure_tag_edge_lengths_mm,
    rebuild_corner_tag_specs,
    scale_tag_layout_anisotropic,
)


class CalibrationError(Exception):
    def __init__(self, message: str, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass
class _HomographyFitResult:
    homography: np.ndarray
    inverse_homography: np.ndarray
    intrinsics: CameraIntrinsics
    reprojection_error_mm: float
    center_error_mm: float
    per_tag_errors: dict[int, float]


@dataclass
class CalibrationData:
    homography: np.ndarray
    inverse_homography: np.ndarray
    principal_point_px: tuple[float, float]
    reprojection_error_mm: float
    timestamp: datetime
    tags: list[dict[str, Any]]
    tag_specs: list[dict[str, Any]]
    intrinsics: CameraIntrinsics | None = None
    work_area: WorkArea | None = None
    tag_size_validation: TagSizeValidation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "homography": self.homography.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
            "principal_point_px": list(self.principal_point_px),
            "reprojection_error_mm": self.reprojection_error_mm,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "tag_specs": self.tag_specs,
        }
        if self.intrinsics is not None:
            payload["intrinsics"] = self.intrinsics.to_dict()
        if self.work_area is not None:
            payload["work_area"] = self.work_area.to_dict()
        if self.tag_size_validation is not None:
            payload["tag_size_validation"] = self.tag_size_validation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationData:
        intrinsics = None
        if "intrinsics" in payload:
            intrinsics = CameraIntrinsics.from_dict(payload["intrinsics"])
        work_area = None
        if "work_area" in payload:
            work_area = WorkArea.from_dict(payload["work_area"])
        tag_size_validation = None
        if "tag_size_validation" in payload:
            tag_size_validation = TagSizeValidation.from_dict(payload["tag_size_validation"])
        return cls(
            homography=np.array(payload["homography"], dtype=np.float64),
            inverse_homography=np.array(payload["inverse_homography"], dtype=np.float64),
            principal_point_px=tuple(payload["principal_point_px"]),
            reprojection_error_mm=float(payload["reprojection_error_mm"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            tags=payload.get("tags", []),
            tag_specs=payload.get("tag_specs", []),
            intrinsics=intrinsics,
            work_area=work_area,
            tag_size_validation=tag_size_validation,
        )


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4:
        return None
    homography, _ = cv2.findHomography(src, dst, method=0)
    return homography


def _reprojection_errors(src: np.ndarray, dst: np.ndarray, homography: np.ndarray) -> np.ndarray:
    reprojected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return np.linalg.norm(reprojected - dst, axis=1)


def _best_corner_permutation(
    image_corners: np.ndarray,
    physical_corners: np.ndarray,
    *,
    bed_to_image: np.ndarray | None = None,
) -> np.ndarray:
    image = np.asarray(image_corners, dtype=np.float32).reshape(-1, 2)
    physical = np.asarray(physical_corners, dtype=np.float32).reshape(-1, 2)
    if len(image) != len(physical):
        return physical

    if bed_to_image is not None:
        projected = cv2.perspectiveTransform(
            physical.reshape(-1, 1, 2).astype(np.float32),
            bed_to_image.astype(np.float32),
        ).reshape(-1, 2)
        best_corners = physical
        best_error = float("inf")
        for reverse in (False, True):
            base_proj = projected[::-1] if reverse else projected
            base_phys = physical[::-1] if reverse else physical
            for shift in range(len(base_proj)):
                rolled_proj = np.roll(base_proj, shift, axis=0)
                rolled_phys = np.roll(base_phys, shift, axis=0)
                error = float(np.mean(np.linalg.norm(image - rolled_proj, axis=1)))
                if error < best_error:
                    best_error = error
                    best_corners = rolled_phys
        return best_corners

    best_corners = physical
    best_error = float("inf")
    for reverse in (False, True):
        base = physical[::-1] if reverse else physical
        for shift in range(len(base)):
            permuted = np.roll(base, shift, axis=0)
            homography = _fit_homography(image, permuted)
            if homography is None:
                continue
            errors = _reprojection_errors(image, permuted, homography)
            mean_error = float(np.max(errors) - np.min(errors))
            if mean_error < best_error:
                best_error = mean_error
                best_corners = permuted
    return best_corners


def _collect_matched_corner_pairs(
    matched: list[tuple[dict[str, Any], AprilTagSpec]],
    intrinsics: CameraIntrinsics,
    dist: np.ndarray,
    bed: BedConfig,
    bed_to_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_corners: list[list[float]] = []
    bed_corners: list[list[float]] = []
    model = intrinsics.distortion_model
    k = intrinsics.camera_matrix

    for det, spec in matched:
        image_corners = np.array(det["corners_px"], dtype=np.float32)
        physical_corners = physical_corners_for_tag(
            spec.x_mm,
            spec.y_mm,
            spec.size_mm,
            bed,
            rotation_deg=spec.rotation_deg,
        )
        physical_center = np.array([spec.x_mm, spec.y_mm], dtype=np.float32)
        undist_corners = undistort_points(image_corners, k, dist, model)
        aligned_physical = _best_corner_permutation(
            undist_corners,
            physical_corners,
            bed_to_image=bed_to_image,
        )
        raw_corners.extend(image_corners.tolist())
        bed_corners.extend(aligned_physical.tolist())

    return np.array(raw_corners, dtype=np.float32), np.array(bed_corners, dtype=np.float32)


def _build_tag_size_validation(
    expected_mm: float,
    measured_mm: dict[int, float],
    scale_iterations: int,
    tolerance_mm: float,
    max_tag_error_mm: float,
    *,
    mm_per_px_x: float | None = None,
    mm_per_px_y: float | None = None,
    axis_edges: AxisEdgeLengths | None = None,
    scale_x_iterations: int = 0,
    scale_y_iterations: int = 0,
) -> TagSizeValidation:
    if not measured_mm:
        return TagSizeValidation(
            expected_mm=expected_mm,
            measured_mm={},
            mean_mm=0.0,
            max_error_mm=0.0,
            scale_iterations=scale_iterations,
            converged=False,
            warning="No tags available for tag size validation.",
            mm_per_px_x=mm_per_px_x,
            mm_per_px_y=mm_per_px_y,
            scale_x_iterations=scale_x_iterations,
            scale_y_iterations=scale_y_iterations,
        )
    mean_mm = float(np.mean(list(measured_mm.values())))
    max_error_mm = float(max(abs(value - expected_mm) for value in measured_mm.values()))
    mean_horizontal_mm = axis_edges.mean_horizontal if axis_edges else mean_mm
    mean_vertical_mm = axis_edges.mean_vertical if axis_edges else mean_mm
    horizontal_ok = abs(mean_horizontal_mm - expected_mm) <= tolerance_mm
    vertical_ok = abs(mean_vertical_mm - expected_mm) <= tolerance_mm
    converged = horizontal_ok and vertical_ok and max_error_mm <= max_tag_error_mm
    warning = None
    if not converged:
        warning = (
            f"Tag size validation did not fully converge after {scale_iterations} scale iteration(s). "
            f"Expected {expected_mm:.1f} mm; mean measured {mean_mm:.2f} mm; "
            f"horizontal {mean_horizontal_mm:.2f} mm; vertical {mean_vertical_mm:.2f} mm; "
            f"max per-tag error {max_error_mm:.2f} mm."
        )
    return TagSizeValidation(
        expected_mm=expected_mm,
        measured_mm=measured_mm,
        mean_mm=mean_mm,
        max_error_mm=max_error_mm,
        scale_iterations=scale_iterations,
        converged=converged,
        warning=warning,
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
        mean_horizontal_mm=mean_horizontal_mm,
        mean_vertical_mm=mean_vertical_mm,
        scale_x_iterations=scale_x_iterations,
        scale_y_iterations=scale_y_iterations,
    )


class CalibrationService:
    def __init__(self) -> None:
        self._data: CalibrationData | None = None
        self._load()

    def _storage_path(self):
        return get_config_store().config.calibration.resolved_storage_path

    def _load(self) -> None:
        path = self._storage_path()
        if not path.exists():
            self._data = None
            return
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._data = CalibrationData.from_dict(payload)

    def is_calibrated(self) -> bool:
        return self._data is not None

    @property
    def data(self) -> CalibrationData | None:
        return self._data

    def get_effective_bed(self) -> BedConfig:
        if self._data is None or self._data.work_area is None:
            raise RuntimeError(
                "Calibration with work area is required before bed dimensions are available"
            )
        config = get_config_store().config
        return effective_bed(config.bed, self._data.work_area)

    def _physical_corners_for_tag(self, spec: AprilTagSpec, bed: BedConfig) -> np.ndarray:
        return physical_corners_for_tag(
            spec.x_mm,
            spec.y_mm,
            spec.size_mm,
            bed,
            rotation_deg=spec.rotation_deg,
        )

    def _resolve_intrinsics(self, frame: np.ndarray) -> CameraIntrinsics:
        config = get_config_store().config.camera
        height, width = frame.shape[:2]
        override = config.intrinsics_override
        return resolve_camera_intrinsics(
            image_width=width,
            image_height=height,
            hfov_deg=config.hfov_deg,
            distortion_model=config.distortion_model,
            override_fx=override.fx,
            override_fy=override.fy,
            override_cx=override.cx,
            override_cy=override.cy,
            override_dist=override.dist,
        )

    def _raise_calibration_error(
        self,
        message: str,
        *,
        matched_tags: list[DetectedAprilTag],
        tag_specs: list[AprilTagSpec],
        center_error_mm: float | None = None,
        corner_error_mm: float | None = None,
        max_error_mm: float | None = None,
        per_tag_errors: dict[int, float] | None = None,
        intrinsics: CameraIntrinsics | None = None,
        hint: str | None = None,
    ) -> None:
        config = get_config_store().config
        detail: dict[str, Any] = {
            "message": message,
            "bed_frame": frame_description(config.bed),
            "matched_tag_ids": [tag.id for tag in matched_tags],
            "expected_tag_ids": [spec.id for spec in tag_specs],
            "center_error_mm": center_error_mm,
            "corner_error_mm": corner_error_mm,
            "max_error_mm": max_error_mm,
            "per_tag_errors_mm": per_tag_errors or {},
            "limit_mm": config.calibration.max_reprojection_error_mm,
        }
        if intrinsics is not None:
            detail["distortion"] = intrinsics.summary()
        if hint:
            detail["hint"] = hint
        raise CalibrationError(message, detail)

    def _fit_homography_calibration(
        self,
        *,
        matched: list[tuple[dict[str, Any], AprilTagSpec]],
        matched_tags: list[DetectedAprilTag],
        tag_specs: list[AprilTagSpec],
        work_area: WorkArea | None,
        frame: np.ndarray,
    ) -> _HomographyFitResult:
        config = get_config_store().config
        if work_area is None:
            raise ValueError(
                "work_area is required; bed dimensions must be derived from AprilTag corner placement"
            )
        bed = effective_bed(config.bed, work_area)

        intrinsics = self._resolve_intrinsics(frame)
        model = intrinsics.distortion_model
        k = intrinsics.camera_matrix
        dist = intrinsics.dist_coeffs.copy()

        center_src_raw = np.array([det["center_px"] for det, _ in matched], dtype=np.float32)
        center_dst = np.array([[spec.x_mm, spec.y_mm] for _, spec in matched], dtype=np.float32)
        center_src = undistort_points(center_src_raw, k, dist, model)
        center_homography = _fit_homography(center_src, center_dst)
        if center_homography is None:
            raise ValueError("Failed to compute homography from tag centers")

        center_errors = _reprojection_errors(center_src, center_dst, center_homography)
        center_error_mm = float(np.mean(center_errors))
        center_limit_mm = max(10.0, config.calibration.max_reprojection_error_mm * 5)
        if center_error_mm > center_limit_mm:
            self._raise_calibration_error(
                f"Tag center fit error {center_error_mm:.2f} mm exceeds limit {center_limit_mm:.2f} mm",
                matched_tags=matched_tags,
                tag_specs=tag_specs,
                center_error_mm=center_error_mm,
                intrinsics=intrinsics,
                hint="Check tag placement and that each tag ID matches the expected corner.",
            )

        try:
            bed_to_image = np.linalg.inv(center_homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to compute bed-to-image transform from tag centers") from exc

        if config.camera.auto_distortion and config.camera.intrinsics_override.dist is None:
            dist = np.zeros((5, 1), dtype=np.float64)
            raw_corners, bed_corners = _collect_matched_corner_pairs(
                matched, intrinsics, dist, bed, bed_to_image=bed_to_image
            )
            zero_error = mean_corner_reprojection_mm(
                raw_corners, bed_corners, k, dist, model
            )
            est_dist, est_error = estimate_distortion_from_tags(
                raw_corners,
                bed_corners,
                k,
                model,
            )
            if zero_error > 5.0 and est_error < zero_error * 0.85:
                dist = est_dist
                for _ in range(2):
                    raw_corners, bed_corners = _collect_matched_corner_pairs(
                        matched, intrinsics, dist, bed, bed_to_image=bed_to_image
                    )
                    est_dist, est_error = estimate_distortion_from_tags(
                        raw_corners,
                        bed_corners,
                        k,
                        model,
                    )
                    if est_error < mean_corner_reprojection_mm(
                        raw_corners, bed_corners, k, dist, model
                    ):
                        dist = est_dist
            intrinsics = CameraIntrinsics(
                camera_matrix=intrinsics.camera_matrix,
                dist_coeffs=dist,
                distortion_model=intrinsics.distortion_model,
                image_width=intrinsics.image_width,
                image_height=intrinsics.image_height,
                hfov_deg=intrinsics.hfov_deg,
            )
            dist = intrinsics.dist_coeffs
            center_src = undistort_points(center_src_raw, k, dist, model)
            center_homography = _fit_homography(center_src, center_dst)
            if center_homography is None:
                raise ValueError("Failed to recompute homography from tag centers")
            bed_to_image = np.linalg.inv(center_homography)

        src_points: list[list[float]] = []
        dst_points: list[list[float]] = []
        tag_point_indices: dict[int, list[int]] = {}

        for det, spec in matched:
            image_corners = np.array(det["corners_px"], dtype=np.float32)
            physical_corners = self._physical_corners_for_tag(spec, bed=bed)
            undist_corners = undistort_points(image_corners, k, dist, model)
            aligned_corners = _best_corner_permutation(
                undist_corners,
                physical_corners,
                bed_to_image=bed_to_image,
            )
            start_idx = len(src_points)
            for img_pt, phys_pt in zip(undist_corners, aligned_corners):
                src_points.append(img_pt.tolist())
                dst_points.append(phys_pt.tolist())
            tag_point_indices[spec.id] = list(range(start_idx, len(src_points)))

        if len(src_points) < 4:
            raise ValueError(
                f"Need at least 4 point correspondences, got {len(src_points)} "
                f"from {len(matched_tags)} matched tags"
            )

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        homography = _fit_homography(src, dst)
        if homography is None:
            raise ValueError("Failed to compute homography from tag corners")

        try:
            inverse_homography = np.linalg.inv(homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to compute inverse homography") from exc

        errors = _reprojection_errors(src, dst, homography)
        reprojection_error = float(np.mean(errors))
        max_error = float(np.max(errors))

        per_tag_errors: dict[int, float] = {}
        for tag_id, indices in tag_point_indices.items():
            per_tag_errors[tag_id] = float(np.mean(errors[indices]))

        if reprojection_error > config.calibration.max_reprojection_error_mm:
            dist_summary = intrinsics.summary()
            hint = (
                "Verify size_mm matches the printed black square edge (not the dashed cut line). "
                "Measure with calipers."
            )
            if center_error_mm <= config.calibration.max_reprojection_error_mm:
                k1 = dist_summary["k1"]
                k2 = dist_summary["k2"]
                boundary_note = ""
                if k1 >= 0.09 or k2 >= 0.29 or k1 <= -0.99:
                    boundary_note = (
                        " Distortion hit search limits — check camera.hfov_deg or set "
                        "camera.intrinsics_override.dist manually."
                    )
                size_hint = ""
                tag_size = tag_specs[0].size_mm if tag_specs else 0.0
                if tag_size > 0 and abs(reprojection_error - tag_size) < tag_size * 0.25:
                    size_hint = (
                        f" Corner error (~{reprojection_error:.0f} mm) is near tag size "
                        f"({tag_size:.0f} mm) — verify corner order and size_mm. "
                    )
                hint = (
                    size_hint
                    + "Tag centers fit well but corners do not. "
                    "This often indicates lens distortion at wide FOV — "
                    f"estimated k1={k1:.4f}, k2={k2:.4f}."
                    + boundary_note
                    + " "
                    + hint
                    + " Tags should share the same orientation as the generated PDF."
                )
            self._raise_calibration_error(
                f"Reprojection error {reprojection_error:.2f} mm exceeds limit "
                f"{config.calibration.max_reprojection_error_mm} mm",
                matched_tags=matched_tags,
                tag_specs=tag_specs,
                center_error_mm=center_error_mm,
                corner_error_mm=reprojection_error,
                max_error_mm=max_error,
                per_tag_errors=per_tag_errors,
                intrinsics=intrinsics,
                hint=hint,
            )

        return _HomographyFitResult(
            homography=homography,
            inverse_homography=inverse_homography,
            intrinsics=intrinsics,
            reprojection_error_mm=reprojection_error,
            center_error_mm=center_error_mm,
            per_tag_errors=per_tag_errors,
        )

    def calibrate(
        self,
        frame: np.ndarray,
        tag_specs: list[AprilTagSpec],
        persist: bool = True,
        work_area: WorkArea | None = None,
        mm_per_px_x: float | None = None,
        mm_per_px_y: float | None = None,
    ) -> tuple[CalibrationData, list[DetectedAprilTag]]:
        apriltag_service = get_apriltag_service()
        config = get_config_store().config
        detections = apriltag_service.detect(frame)
        spec_by_id = {spec.id: spec for spec in tag_specs}

        matched: list[tuple[dict[str, Any], AprilTagSpec]] = []
        for det in detections:
            tag_id = det["id"]
            if tag_id in spec_by_id:
                matched.append((det, spec_by_id[tag_id]))

        matched_tags: list[DetectedAprilTag] = [
            DetectedAprilTag(
                id=det["id"],
                center_px=det["center_px"],
                corners_px=det["corners_px"],
            )
            for det, _ in matched
        ]

        if len(matched) < 1:
            raise ValueError(
                f"No matching tags found. Detected IDs: {[d['id'] for d in detections]}, "
                f"expected: {list(spec_by_id.keys())}"
            )

        expected_size_mm = tag_specs[0].size_mm if tag_specs else config.apriltag.default_size_mm
        if work_area is None:
            raise ValueError(
                "work_area is required; bed dimensions must be derived from AprilTag corner placement"
            )
        current_specs = list(tag_specs)
        current_work_area = work_area
        scale_x_iterations = 0
        scale_y_iterations = 0
        fit: _HomographyFitResult | None = None

        max_iterations = config.calibration.scale_refinement_max_iterations
        tolerance_mm = config.calibration.scale_refinement_tolerance_mm
        max_tag_error_mm = config.calibration.max_tag_size_error_mm

        def _matched_with_specs(specs: list[AprilTagSpec]) -> list[tuple[dict[str, Any], AprilTagSpec]]:
            specs_by_id = {spec.id: spec for spec in specs}
            pairs: list[tuple[dict[str, Any], AprilTagSpec]] = []
            for det in detections:
                tag_id = det["id"]
                if tag_id in specs_by_id:
                    pairs.append((det, specs_by_id[tag_id]))
            return pairs

        def _image_centers_undistorted(
            matched_pairs: list[tuple[dict[str, Any], AprilTagSpec]],
            intrinsics: CameraIntrinsics,
        ) -> dict[int, np.ndarray]:
            k = intrinsics.camera_matrix
            dist = intrinsics.dist_coeffs
            model = intrinsics.distortion_model
            image_centers: dict[int, np.ndarray] = {}
            for det, spec in matched_pairs:
                if spec.id in image_centers:
                    continue
                center = np.array(det["center_px"], dtype=np.float64)
                image_centers[spec.id] = undistort_points(center.reshape(1, 2), k, dist, model)[0]
            return image_centers

        def _apply_layout_from_homography(
            matched_pairs: list[tuple[dict[str, Any], AprilTagSpec]],
            homography_fit: _HomographyFitResult,
            origin_tag_id: int,
            tag_ids: list[int],
        ) -> tuple[list[AprilTagSpec], WorkArea]:
            finalized = finalize_work_area_from_homography(
                matched_pairs,
                homography_fit.homography,
                homography_fit.intrinsics,
                origin_tag_id=origin_tag_id,
                size_mm=expected_size_mm,
            )
            image_centers = _image_centers_undistorted(matched_pairs, homography_fit.intrinsics)
            specs = rebuild_corner_tag_specs(
                finalized,
                origin_tag_id=origin_tag_id,
                tag_ids=tag_ids,
                image_centers=image_centers,
            )
            return specs, finalized

        for iteration in range(max_iterations + 1):
            matched_current = _matched_with_specs(current_specs)
            fit = self._fit_homography_calibration(
                matched=matched_current,
                matched_tags=matched_tags,
                tag_specs=current_specs,
                work_area=current_work_area,
                frame=frame,
            )
            axis_edges = measure_tag_axis_edge_lengths_mm(
                matched_current,
                fit.homography,
                fit.intrinsics,
            )
            mean_horizontal = axis_edges.mean_horizontal
            mean_vertical = axis_edges.mean_vertical
            horizontal_ok = abs(mean_horizontal - expected_size_mm) <= tolerance_mm
            vertical_ok = abs(mean_vertical - expected_size_mm) <= tolerance_mm
            if horizontal_ok and vertical_ok:
                break
            if iteration >= max_iterations:
                break
            if current_work_area is None or (mean_horizontal <= 0 and mean_vertical <= 0):
                break

            scale_x = expected_size_mm / mean_horizontal if mean_horizontal > 0 and not horizontal_ok else 1.0
            scale_y = expected_size_mm / mean_vertical if mean_vertical > 0 and not vertical_ok else 1.0
            if scale_x != 1.0:
                scale_x_iterations += 1
            if scale_y != 1.0:
                scale_y_iterations += 1
            current_specs, current_work_area = scale_tag_layout_anisotropic(
                current_specs,
                current_work_area,
                scale_x,
                scale_y,
            )

        assert fit is not None
        scale_iterations = max(scale_x_iterations, scale_y_iterations)

        if current_work_area is not None:
            origin_tag_id = current_work_area.origin_tag_id
            tag_ids = [spec.id for spec in current_specs]
            matched_current = _matched_with_specs(current_specs)
            current_specs, current_work_area = _apply_layout_from_homography(
                matched_current,
                fit,
                origin_tag_id,
                tag_ids,
            )
            matched_current = _matched_with_specs(current_specs)
            fit = self._fit_homography_calibration(
                matched=matched_current,
                matched_tags=matched_tags,
                tag_specs=current_specs,
                work_area=current_work_area,
                frame=frame,
            )
            matched_current = _matched_with_specs(current_specs)
            current_specs, current_work_area = _apply_layout_from_homography(
                matched_current,
                fit,
                origin_tag_id,
                tag_ids,
            )
            matched_for_measure = _matched_with_specs(current_specs)
        else:
            matched_for_measure = _matched_with_specs(current_specs)

        measured_sizes = measure_tag_edge_lengths_mm(
            matched_for_measure,
            fit.homography,
            fit.intrinsics,
        )
        final_axis_edges = measure_tag_axis_edge_lengths_mm(
            matched_for_measure,
            fit.homography,
            fit.intrinsics,
        )
        tag_size_validation = _build_tag_size_validation(
            expected_size_mm,
            measured_sizes,
            scale_iterations,
            tolerance_mm,
            max_tag_error_mm,
            mm_per_px_x=mm_per_px_x,
            mm_per_px_y=mm_per_px_y,
            axis_edges=final_axis_edges,
            scale_x_iterations=scale_x_iterations,
            scale_y_iterations=scale_y_iterations,
        )

        bed = effective_bed(config.bed, current_work_area)
        cx, cy = bed_center_mm(bed)
        bed_center = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        k = fit.intrinsics.camera_matrix
        dist = fit.intrinsics.dist_coeffs
        model = fit.intrinsics.distortion_model
        principal_undist = cv2.perspectiveTransform(
            bed_center,
            fit.inverse_homography.astype(np.float32),
        ).reshape(1, 2)
        principal_px = distort_points(principal_undist, k, dist, model)[0]

        calibration = CalibrationData(
            homography=fit.homography,
            inverse_homography=fit.inverse_homography,
            principal_point_px=(float(principal_px[0]), float(principal_px[1])),
            reprojection_error_mm=fit.reprojection_error_mm,
            timestamp=datetime.now(timezone.utc),
            tags=[tag.model_dump() for tag in matched_tags],
            tag_specs=[spec.model_dump() for spec in current_specs],
            intrinsics=fit.intrinsics,
            work_area=current_work_area,
            tag_size_validation=tag_size_validation,
        )

        if persist:
            self._save(calibration)
        return calibration, matched_tags

    def _save(self, calibration: CalibrationData) -> None:
        path = self._storage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(calibration.to_dict(), handle, indent=2)
        self._data = calibration

    def get_status(self, expected_tags: int = 4) -> dict[str, Any]:
        frame = get_config_store().config.bed
        work_area = self._data.work_area if self._data else None
        base: dict[str, Any] = {
            "bed_frame": frame_description(frame),
        }
        work_area_summary = None
        if work_area is not None:
            work_area_summary = {
                "width_mm": work_area.width_mm,
                "height_mm": work_area.height_mm,
                "origin_tag_id": work_area.origin_tag_id,
                "size_mm": work_area.size_mm,
            }
        if self._data is None:
            return {
                **base,
                "calibrated": False,
                "timestamp": None,
                "reprojection_error_mm": None,
                "tags_detected": 0,
                "tags_expected": expected_tags,
                "message": "Not calibrated",
                "distortion": None,
                "work_area": work_area_summary,
                "tag_size_validation": None,
            }
        tag_size_validation = None
        if self._data.tag_size_validation is not None:
            tag_size_validation = self._data.tag_size_validation.to_dict()
        return {
            **base,
            "calibrated": True,
            "timestamp": self._data.timestamp,
            "reprojection_error_mm": self._data.reprojection_error_mm,
            "tags_detected": len(self._data.tags),
            "tags_expected": expected_tags,
            "message": "Calibration loaded",
            "distortion": self._data.intrinsics.summary() if self._data.intrinsics else None,
            "work_area": work_area_summary,
            "tag_size_validation": tag_size_validation,
        }


_calibration_service: CalibrationService | None = None


def get_calibration_service() -> CalibrationService:
    global _calibration_service
    if _calibration_service is None:
        _calibration_service = CalibrationService()
    return _calibration_service

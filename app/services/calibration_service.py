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


class CalibrationError(Exception):
    def __init__(self, message: str, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


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
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationData:
        intrinsics = None
        if "intrinsics" in payload:
            intrinsics = CameraIntrinsics.from_dict(payload["intrinsics"])
        return cls(
            homography=np.array(payload["homography"], dtype=np.float64),
            inverse_homography=np.array(payload["inverse_homography"], dtype=np.float64),
            principal_point_px=tuple(payload["principal_point_px"]),
            reprojection_error_mm=float(payload["reprojection_error_mm"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            tags=payload.get("tags", []),
            tag_specs=payload.get("tag_specs", []),
            intrinsics=intrinsics,
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
            get_config_store().config.bed,
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

    def _physical_corners_for_tag(self, spec: AprilTagSpec) -> np.ndarray:
        bed = get_config_store().config.bed
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

    def calibrate(
        self,
        frame: np.ndarray,
        tag_specs: list[AprilTagSpec],
        persist: bool = True,
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
                hint="Check bed dimensions and that each tag ID is placed at the matching bed corner.",
            )

        try:
            bed_to_image = np.linalg.inv(center_homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to compute bed-to-image transform from tag centers") from exc

        if config.camera.auto_distortion and config.camera.intrinsics_override.dist is None:
            dist = np.zeros((5, 1), dtype=np.float64)
            raw_corners, bed_corners = _collect_matched_corner_pairs(
                matched, intrinsics, dist, bed_to_image
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
                        matched, intrinsics, dist, bed_to_image
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
            physical_corners = self._physical_corners_for_tag(spec)
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

        cx, cy = bed_center_mm(config.bed)
        bed_center = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
        principal_undist = cv2.perspectiveTransform(bed_center, inverse_homography).reshape(1, 2)
        principal_px = distort_points(principal_undist, k, dist, model)[0]

        calibration = CalibrationData(
            homography=homography,
            inverse_homography=inverse_homography,
            principal_point_px=(float(principal_px[0]), float(principal_px[1])),
            reprojection_error_mm=reprojection_error,
            timestamp=datetime.now(timezone.utc),
            tags=[tag.model_dump() for tag in matched_tags],
            tag_specs=[spec.model_dump() for spec in tag_specs],
            intrinsics=intrinsics,
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
        bed = get_config_store().config.bed
        base: dict[str, Any] = {
            "bed_frame": frame_description(bed),
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
            }
        return {
            **base,
            "calibrated": True,
            "timestamp": self._data.timestamp,
            "reprojection_error_mm": self._data.reprojection_error_mm,
            "tags_detected": len(self._data.tags),
            "tags_expected": expected_tags,
            "message": "Calibration loaded",
            "distortion": self._data.intrinsics.summary() if self._data.intrinsics else None,
        }


_calibration_service: CalibrationService | None = None


def get_calibration_service() -> CalibrationService:
    global _calibration_service
    if _calibration_service is None:
        _calibration_service = CalibrationService()
    return _calibration_service

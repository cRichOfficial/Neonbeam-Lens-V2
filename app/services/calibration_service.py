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


@dataclass
class CalibrationData:
    homography: np.ndarray
    inverse_homography: np.ndarray
    principal_point_px: tuple[float, float]
    reprojection_error_mm: float
    timestamp: datetime
    tags: list[dict[str, Any]]
    tag_specs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "homography": self.homography.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
            "principal_point_px": list(self.principal_point_px),
            "reprojection_error_mm": self.reprojection_error_mm,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "tag_specs": self.tag_specs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationData:
        return cls(
            homography=np.array(payload["homography"], dtype=np.float64),
            inverse_homography=np.array(payload["inverse_homography"], dtype=np.float64),
            principal_point_px=tuple(payload["principal_point_px"]),
            reprojection_error_mm=float(payload["reprojection_error_mm"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            tags=payload.get("tags", []),
            tag_specs=payload.get("tag_specs", []),
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

    def _physical_corners_for_tag(self, spec: AprilTagSpec) -> np.ndarray:
        half = spec.size_mm / 2.0
        center = np.array([spec.x_mm, spec.y_mm], dtype=np.float32)
        offsets = np.array(
            [
                [-half, -half],
                [half, -half],
                [half, half],
                [-half, half],
            ],
            dtype=np.float32,
        )
        return center + offsets

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

        src_points: list[list[float]] = []
        dst_points: list[list[float]] = []
        matched_tags: list[DetectedAprilTag] = []

        for det in detections:
            tag_id = det["id"]
            if tag_id not in spec_by_id:
                continue
            spec = spec_by_id[tag_id]
            image_corners = np.array(det["corners_px"], dtype=np.float32)
            physical_corners = self._physical_corners_for_tag(spec)
            for img_pt, phys_pt in zip(image_corners, physical_corners):
                src_points.append(img_pt.tolist())
                dst_points.append(phys_pt.tolist())
            matched_tags.append(
                DetectedAprilTag(
                    id=tag_id,
                    center_px=det["center_px"],
                    corners_px=det["corners_px"],
                )
            )

        if len(src_points) < 4:
            raise ValueError(
                f"Need at least 4 point correspondences, got {len(src_points)} "
                f"from {len(matched_tags)} matched tags"
            )

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if homography is None:
            raise ValueError("Failed to compute homography")

        inverse_homography, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
        if inverse_homography is None:
            raise ValueError("Failed to compute inverse homography")

        reprojected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(reprojected - dst, axis=1)
        reprojection_error = float(np.mean(errors))

        if reprojection_error > config.calibration.max_reprojection_error_mm:
            raise ValueError(
                f"Reprojection error {reprojection_error:.2f} mm exceeds limit "
                f"{config.calibration.max_reprojection_error_mm} mm"
            )

        bed_center_mm = np.array(
            [[config.bed.width_mm / 2.0, config.bed.height_mm / 2.0]], dtype=np.float32
        ).reshape(-1, 1, 2)
        principal_px = cv2.perspectiveTransform(bed_center_mm, inverse_homography).reshape(2)

        calibration = CalibrationData(
            homography=homography,
            inverse_homography=inverse_homography,
            principal_point_px=(float(principal_px[0]), float(principal_px[1])),
            reprojection_error_mm=reprojection_error,
            timestamp=datetime.now(timezone.utc),
            tags=[tag.model_dump() for tag in matched_tags],
            tag_specs=[spec.model_dump() for spec in tag_specs],
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
        if self._data is None:
            return {
                "calibrated": False,
                "timestamp": None,
                "reprojection_error_mm": None,
                "tags_detected": 0,
                "tags_expected": expected_tags,
                "message": "Not calibrated",
            }
        return {
            "calibrated": True,
            "timestamp": self._data.timestamp,
            "reprojection_error_mm": self._data.reprojection_error_mm,
            "tags_detected": len(self._data.tags),
            "tags_expected": expected_tags,
            "message": "Calibration loaded",
        }


_calibration_service: CalibrationService | None = None


def get_calibration_service() -> CalibrationService:
    global _calibration_service
    if _calibration_service is None:
        _calibration_service = CalibrationService()
    return _calibration_service

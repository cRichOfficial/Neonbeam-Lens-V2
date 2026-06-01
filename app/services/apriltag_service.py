from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.config import ApriltagPreprocess, get_config_store

logger = logging.getLogger(__name__)

APRILTAG_FAMILIES = {
    "tag36h11": "tag36h11",
    "tag25h9": "tag25h9",
    "tag16h5": "tag16h5",
}

OPENCV_APRILTAG_DICTIONARIES = {
    "tag36h11": getattr(cv2.aruco, "DICT_APRILTAG_36h11", None),
    "tag25h9": getattr(cv2.aruco, "DICT_APRILTAG_25h9", None),
    "tag16h5": getattr(cv2.aruco, "DICT_APRILTAG_16h5", None),
}


def _detection_to_dict(det: Any, *, pass_name: str = "raw", backend: str = "pupil") -> dict[str, Any]:
    corners = np.array(det.corners, dtype=np.float32)
    center = np.array(det.center, dtype=np.float32)
    return {
        "id": int(det.tag_id),
        "center_px": center.tolist(),
        "corners_px": corners.tolist(),
        "hamming": int(det.hamming),
        "decision_margin": float(det.decision_margin),
        "pass": pass_name,
        "backend": backend,
    }


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def frame_to_gray_variants(
    frame: np.ndarray,
    preprocess: ApriltagPreprocess,
) -> list[tuple[str, np.ndarray]]:
    if frame.ndim == 2:
        gray_rgb = frame
        gray_bgr = frame
    else:
        gray_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if preprocess == "none":
        return [("raw", gray_rgb)]

    if preprocess == "clahe":
        return [("raw", gray_rgb), ("clahe", _apply_clahe(gray_rgb))]

    return [
        ("raw", gray_rgb),
        ("bgr", gray_bgr),
        ("clahe", _apply_clahe(gray_rgb)),
        ("sharpen", _apply_sharpen(gray_rgb)),
    ]


def frame_gray_stats(frame: np.ndarray) -> dict[str, float | int]:
    gray_variants = frame_to_gray_variants(frame, "none")
    gray = gray_variants[0][1]
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_min": int(gray.min()),
        "gray_max": int(gray.max()),
    }


def merge_detections(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[int, dict[str, Any]] = {}
    for det in all_results:
        tag_id = det["id"]
        existing = best_by_id.get(tag_id)
        if existing is None or det["decision_margin"] > existing["decision_margin"]:
            best_by_id[tag_id] = det
    return sorted(best_by_id.values(), key=lambda item: item["id"])


def build_detection_failure_hint(
    detections: list[dict[str, Any]],
    *,
    expected_ids: list[int],
    exposure_ms: float,
    frame_stats: dict[str, float | int] | None = None,
) -> str:
    detected_ids = sorted({det["id"] for det in detections})
    missing_ids = [tag_id for tag_id in expected_ids if tag_id not in detected_ids]
    if not detections:
        hints: list[str] = []
        if exposure_ms > 100:
            hints.append(
                f"Try lowering exposure_ms (current: {exposure_ms:g}). "
                "Long exposures bloom white surfaces and erase tag edges."
            )
        if frame_stats is not None and float(frame_stats.get("gray_std", 99.0)) < 5.0:
            hints.append(
                "Frame grayscale has very low contrast (std < 5). "
                "The camera buffer may be invalid — redeploy and restart the service."
            )
        if not hints:
            hints.append(
                "Check tag placement, lighting, family (tag36h11), "
                "and a white quiet zone around each black square."
            )
        return " ".join(hints)
    if missing_ids:
        return (
            f"Detected tag IDs {detected_ids} but missing {missing_ids}. "
            "Verify corner tag IDs and placement."
        )
    return "Tag detection failed for an unknown reason."


def _detect_aruco(gray: np.ndarray, family: str) -> list[dict[str, Any]]:
    dict_id = OPENCV_APRILTAG_DICTIONARIES.get(family)
    if dict_id is None:
        return []
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return []

    results: list[dict[str, Any]] = []
    for index, tag_id in enumerate(ids.flatten()):
        corner_set = np.array(corners[index][0], dtype=np.float32)
        center = corner_set.mean(axis=0)
        results.append(
            {
                "id": int(tag_id),
                "center_px": center.tolist(),
                "corners_px": corner_set.tolist(),
                "hamming": 0,
                "decision_margin": 100.0,
                "pass": "aruco",
                "backend": "aruco",
            }
        )
    return results


class AprilTagService:
    def __init__(self) -> None:
        self._detectors: dict[str, Any] = {}

    def _get_detector(self, family: str):
        if family not in self._detectors:
            import pupil_apriltags as apriltag

            config = get_config_store().config.apriltag
            tag_family = APRILTAG_FAMILIES.get(family, family)
            self._detectors[family] = apriltag.Detector(
                families=tag_family,
                quad_decimate=config.quad_decimate,
                quad_sigma=config.quad_sigma,
                refine_edges=1,
                decode_sharpening=config.decode_sharpening,
            )
        return self._detectors[family]

    def _detect_pupil(self, frame: np.ndarray, family: str) -> list[dict[str, Any]]:
        config = get_config_store().config.apriltag
        detector = self._get_detector(family)
        passes = frame_to_gray_variants(frame, config.preprocess)
        merged: list[dict[str, Any]] = []
        for pass_name, gray in passes:
            for det in detector.detect(gray):
                merged.append(_detection_to_dict(det, pass_name=pass_name, backend="pupil"))
        return merge_detections(merged)

    def _detect_aruco_fallback(self, frame: np.ndarray, family: str) -> list[dict[str, Any]]:
        config = get_config_store().config.apriltag
        passes = frame_to_gray_variants(frame, config.preprocess)
        merged: list[dict[str, Any]] = []
        for _pass_name, gray in passes:
            merged.extend(_detect_aruco(gray, family))
        return merge_detections(merged)

    def detect(self, frame: np.ndarray, family: str | None = None) -> list[dict[str, Any]]:
        config = get_config_store().config.apriltag
        family = family or config.family
        results = self._detect_pupil(frame, family)
        if not results and config.aruco_fallback:
            results = self._detect_aruco_fallback(frame, family)
        return results

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        *,
        label_corners: bool = True,
    ) -> np.ndarray:
        output = frame.copy()
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        for det in detections:
            corners = np.array(det["corners_px"], dtype=np.int32)
            cv2.polylines(output, [corners], True, (0, 255, 0), 2)
            center = tuple(int(v) for v in det["center_px"])
            cv2.circle(output, center, 4, (255, 0, 0), -1)
            cv2.putText(
                output,
                f"ID:{det['id']}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if label_corners:
                for idx, corner in enumerate(corners):
                    cx, cy = int(corner[0]), int(corner[1])
                    cv2.circle(output, (cx, cy), 5, (0, 200, 255), -1)
                    cv2.putText(
                        output,
                        str(idx),
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
        return output


_apriltag_service: AprilTagService | None = None


def get_apriltag_service() -> AprilTagService:
    global _apriltag_service
    if _apriltag_service is None:
        _apriltag_service = AprilTagService()
    return _apriltag_service

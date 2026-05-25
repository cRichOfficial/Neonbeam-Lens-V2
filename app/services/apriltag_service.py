from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from app.config import get_config_store

logger = logging.getLogger(__name__)

APRILTAG_FAMILIES = {
    "tag36h11": "tag36h11",
    "tag25h9": "tag25h9",
    "tag16h5": "tag16h5",
}


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
                refine_edges=1,
            )
        return self._detectors[family]

    def detect(self, frame: np.ndarray, family: str | None = None) -> list[dict[str, Any]]:
        config = get_config_store().config.apriltag
        family = family or config.family
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        detector = self._get_detector(family)
        detections = detector.detect(gray)
        results: list[dict[str, Any]] = []
        for det in detections:
            corners = np.array(det.corners, dtype=np.float32)
            center = np.array(det.center, dtype=np.float32)
            results.append(
                {
                    "id": int(det.tag_id),
                    "center_px": center.tolist(),
                    "corners_px": corners.tolist(),
                    "hamming": int(det.hamming),
                    "decision_margin": float(det.decision_margin),
                }
            )
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

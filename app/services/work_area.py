from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

from app.config import BedConfig, BedFrameConfig
from app.schemas.calibration import AprilTagSpec
from app.services.camera_intrinsics import undistort_points

if TYPE_CHECKING:
    from app.services.camera_intrinsics import CameraIntrinsics


@dataclass(frozen=True)
class WorkArea:
    width_mm: float
    height_mm: float
    origin_tag_id: int
    size_mm: float
    mode: str = "corner_defined"

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "origin_tag_id": self.origin_tag_id,
            "size_mm": self.size_mm,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkArea:
        return cls(
            width_mm=float(payload["width_mm"]),
            height_mm=float(payload["height_mm"]),
            origin_tag_id=int(payload["origin_tag_id"]),
            size_mm=float(payload["size_mm"]),
            mode=str(payload.get("mode", "corner_defined")),
        )


@dataclass(frozen=True)
class TagSizeValidation:
    expected_mm: float
    measured_mm: dict[int, float]
    mean_mm: float
    max_error_mm: float
    scale_iterations: int
    converged: bool
    warning: str | None = None
    mm_per_px_x: float | None = None
    mm_per_px_y: float | None = None
    mean_horizontal_mm: float | None = None
    mean_vertical_mm: float | None = None
    scale_x_iterations: int = 0
    scale_y_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_mm": self.expected_mm,
            "measured_mm": {str(k): v for k, v in self.measured_mm.items()},
            "mean_mm": self.mean_mm,
            "max_error_mm": self.max_error_mm,
            "scale_iterations": self.scale_iterations,
            "converged": self.converged,
            "warning": self.warning,
            "mm_per_px_x": self.mm_per_px_x,
            "mm_per_px_y": self.mm_per_px_y,
            "mean_horizontal_mm": self.mean_horizontal_mm,
            "mean_vertical_mm": self.mean_vertical_mm,
            "scale_x_iterations": self.scale_x_iterations,
            "scale_y_iterations": self.scale_y_iterations,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TagSizeValidation:
        measured = payload.get("measured_mm", {})
        return cls(
            expected_mm=float(payload["expected_mm"]),
            measured_mm={int(k): float(v) for k, v in measured.items()},
            mean_mm=float(payload["mean_mm"]),
            max_error_mm=float(payload["max_error_mm"]),
            scale_iterations=int(payload["scale_iterations"]),
            converged=bool(payload["converged"]),
            warning=payload.get("warning"),
            mm_per_px_x=(
                float(payload["mm_per_px_x"]) if payload.get("mm_per_px_x") is not None else None
            ),
            mm_per_px_y=(
                float(payload["mm_per_px_y"]) if payload.get("mm_per_px_y") is not None else None
            ),
            mean_horizontal_mm=(
                float(payload["mean_horizontal_mm"])
                if payload.get("mean_horizontal_mm") is not None
                else None
            ),
            mean_vertical_mm=(
                float(payload["mean_vertical_mm"])
                if payload.get("mean_vertical_mm") is not None
                else None
            ),
            scale_x_iterations=int(payload.get("scale_x_iterations", 0)),
            scale_y_iterations=int(payload.get("scale_y_iterations", 0)),
        )


@dataclass(frozen=True)
class DerivedWorkArea:
    work_area: WorkArea
    tag_specs: list[AprilTagSpec]
    mm_per_px_x: float | None = None
    mm_per_px_y: float | None = None


@dataclass(frozen=True)
class AxisEdgeLengths:
    horizontal: list[float]
    vertical: list[float]

    @property
    def mean_horizontal(self) -> float:
        return float(np.mean(self.horizontal)) if self.horizontal else 0.0

    @property
    def mean_vertical(self) -> float:
        return float(np.mean(self.vertical)) if self.vertical else 0.0


def effective_bed(frame: BedFrameConfig, work_area: WorkArea) -> BedConfig:
    return BedConfig(
        width_mm=work_area.width_mm,
        height_mm=work_area.height_mm,
        origin=frame.origin,
        y_axis=frame.y_axis,
    )


def average_tag_edge_px(corners_px: np.ndarray) -> float:
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    edges = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4])) for i in range(4)]
    return float(np.mean(edges))


def _edge_is_horizontal(edge: np.ndarray) -> bool:
    dx = abs(float(edge[0]))
    dy = abs(float(edge[1]))
    return dx >= dy


def _tag_axis_edge_lengths_px(corners: np.ndarray) -> tuple[list[float], list[float]]:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    horizontal: list[float] = []
    vertical: list[float] = []
    for index in range(4):
        edge = corners[(index + 1) % 4] - corners[index]
        length = float(np.linalg.norm(edge))
        if _edge_is_horizontal(edge):
            horizontal.append(length)
        else:
            vertical.append(length)
    return horizontal, vertical


def _mm_per_px_from_tags(
    detections_by_id: dict[int, dict[str, Any]],
    tag_ids: list[int],
    size_mm: float,
    corner_transform: Callable[[np.ndarray], np.ndarray] | None,
) -> tuple[float, float]:
    horizontal_edges: list[float] = []
    vertical_edges: list[float] = []
    for tag_id in tag_ids:
        det = detections_by_id[tag_id]
        corners = _transform_corners(det["corners_px"], corner_transform)
        h_edges, v_edges = _tag_axis_edge_lengths_px(corners)
        horizontal_edges.extend(h_edges)
        vertical_edges.extend(v_edges)

    all_edges = horizontal_edges + vertical_edges
    fallback = size_mm / float(np.mean(all_edges)) if all_edges else 1.0
    mm_per_px_x = (
        size_mm / float(np.mean(horizontal_edges)) if horizontal_edges else fallback
    )
    mm_per_px_y = size_mm / float(np.mean(vertical_edges)) if vertical_edges else fallback
    return mm_per_px_x, mm_per_px_y


def _anisotropic_span_mm(
    start: np.ndarray,
    end: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
) -> float:
    delta = end - start
    return float(np.hypot(delta[0] * mm_per_px_x, delta[1] * mm_per_px_y))


def _classify_corner_tags(origin_tag_id: int, centers: dict[int, np.ndarray]) -> dict[str, int]:
    if origin_tag_id not in centers:
        raise ValueError(f"Origin tag {origin_tag_id} not found in detections")

    median_y = float(np.median([center[1] for center in centers.values()]))
    bottom_ids = [tag_id for tag_id, center in centers.items() if center[1] >= median_y]
    top_ids = [tag_id for tag_id, center in centers.items() if center[1] < median_y]

    br_candidates = [tag_id for tag_id in bottom_ids if tag_id != origin_tag_id]
    if len(br_candidates) != 1:
        raise ValueError(
            "Could not identify the +X corner tag from geometry. "
            f"Expected one bottom-row tag besides origin; found {br_candidates}."
        )
    br_id = br_candidates[0]

    if len(top_ids) != 2:
        raise ValueError(
            "Could not identify the top row tags from geometry. "
            f"Expected two tags above the horizontal midline; found {top_ids}."
        )
    top_sorted = sorted(top_ids, key=lambda tag_id: centers[tag_id][0])
    tl_id, tr_id = top_sorted[0], top_sorted[1]

    return {"origin": origin_tag_id, "br": br_id, "tl": tl_id, "tr": tr_id}


def _transform_corners(
    corners_px: np.ndarray,
    corner_transform: Callable[[np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if corner_transform is None:
        return corners
    return np.array([corner_transform(corners[i]).reshape(2) for i in range(4)], dtype=np.float64)


def derive_work_area_from_detections(
    detections: list[dict[str, Any]],
    *,
    origin_tag_id: int,
    size_mm: float,
    tag_ids: list[int],
    center_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    corner_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> DerivedWorkArea:
    """Derive work area size and tag bed coordinates from corner tag detections."""
    if len(tag_ids) < 4:
        raise ValueError("corner_defined calibration requires at least 4 tag IDs")

    det_by_id = {det["id"]: det for det in detections}
    missing = [tag_id for tag_id in tag_ids if tag_id not in det_by_id]
    if missing:
        raise ValueError(f"Tags not detected: {missing}")
    if origin_tag_id not in tag_ids:
        raise ValueError("origin_tag_id must be included in tag_ids")

    centers: dict[int, np.ndarray] = {}
    for tag_id in tag_ids:
        det = det_by_id[tag_id]
        center = np.array(det["center_px"], dtype=np.float64)
        if center_transform is not None:
            center = center_transform(center).reshape(2)
        centers[tag_id] = center

    mm_per_px_x, mm_per_px_y = _mm_per_px_from_tags(
        det_by_id,
        tag_ids,
        size_mm,
        corner_transform,
    )
    roles = _classify_corner_tags(origin_tag_id, centers)

    origin_center = centers[roles["origin"]]
    br_center = centers[roles["br"]]
    tl_center = centers[roles["tl"]]

    width_mm = _anisotropic_span_mm(origin_center, br_center, mm_per_px_x, mm_per_px_y)
    height_mm = _anisotropic_span_mm(origin_center, tl_center, mm_per_px_x, mm_per_px_y)

    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Derived work area dimensions must be positive")

    bed_coords = {
        roles["origin"]: (0.0, 0.0),
        roles["br"]: (width_mm, 0.0),
        roles["tl"]: (0.0, height_mm),
        roles["tr"]: (width_mm, height_mm),
    }
    tag_specs = [
        AprilTagSpec(
            id=tag_id,
            x_mm=bed_coords[tag_id][0],
            y_mm=bed_coords[tag_id][1],
            size_mm=size_mm,
        )
        for tag_id in tag_ids
    ]
    work_area = WorkArea(
        width_mm=width_mm,
        height_mm=height_mm,
        origin_tag_id=origin_tag_id,
        size_mm=size_mm,
    )
    return DerivedWorkArea(
        work_area=work_area,
        tag_specs=tag_specs,
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
    )


def measure_tag_edge_lengths_mm(
    matched: list[tuple[dict[str, Any], AprilTagSpec]],
    homography: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> dict[int, float]:

    k = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    model = intrinsics.distortion_model
    sizes: dict[int, float] = {}
    for det, spec in matched:
        corners = np.array(det["corners_px"], dtype=np.float32)
        undist = undistort_points(corners, k, dist, model)
        mm_corners = cv2.perspectiveTransform(
            undist.reshape(-1, 1, 2).astype(np.float32),
            homography.astype(np.float32),
        ).reshape(-1, 2)
        edges = [
            float(np.linalg.norm(mm_corners[i] - mm_corners[(i + 1) % 4]))
            for i in range(4)
        ]
        sizes[spec.id] = float(np.mean(edges))
    return sizes


def measure_tag_axis_edge_lengths_mm(
    matched: list[tuple[dict[str, Any], AprilTagSpec]],
    homography: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> AxisEdgeLengths:
    k = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    model = intrinsics.distortion_model
    horizontal: list[float] = []
    vertical: list[float] = []
    for det, _spec in matched:
        corners = np.array(det["corners_px"], dtype=np.float32)
        undist = undistort_points(corners, k, dist, model)
        mm_corners = cv2.perspectiveTransform(
            undist.reshape(-1, 1, 2).astype(np.float32),
            homography.astype(np.float32),
        ).reshape(-1, 2)
        for index in range(4):
            edge = mm_corners[(index + 1) % 4] - mm_corners[index]
            length = float(np.linalg.norm(edge))
            if _edge_is_horizontal(edge):
                horizontal.append(length)
            else:
                vertical.append(length)
    return AxisEdgeLengths(horizontal=horizontal, vertical=vertical)


def centers_mm_from_homography(
    matched: list[tuple[dict[str, Any], AprilTagSpec]],
    homography: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> dict[int, np.ndarray]:
    k = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    model = intrinsics.distortion_model
    centers: dict[int, np.ndarray] = {}
    for det, spec in matched:
        center = np.array(det["center_px"], dtype=np.float32)
        undist = undistort_points(center.reshape(1, 2), k, dist, model)[0]
        mm = cv2.perspectiveTransform(
            undist.reshape(1, 1, 2).astype(np.float32),
            homography.astype(np.float32),
        ).reshape(2)
        centers[spec.id] = mm
    return centers


def finalize_work_area_from_homography(
    matched: list[tuple[dict[str, Any], AprilTagSpec]],
    homography: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    origin_tag_id: int,
    size_mm: float,
) -> WorkArea:
    k = intrinsics.camera_matrix
    dist = intrinsics.dist_coeffs
    model = intrinsics.distortion_model
    image_centers: dict[int, np.ndarray] = {}
    for det, spec in matched:
        if spec.id in image_centers:
            continue
        center = np.array(det["center_px"], dtype=np.float64)
        undist = undistort_points(center.reshape(1, 2), k, dist, model)[0]
        image_centers[spec.id] = undist

    mm_centers = centers_mm_from_homography(matched, homography, intrinsics)
    roles = _classify_corner_tags(origin_tag_id, image_centers)
    origin = mm_centers[roles["origin"]]
    br = mm_centers[roles["br"]]
    tl = mm_centers[roles["tl"]]
    width_mm = float(np.linalg.norm(br - origin))
    height_mm = float(np.linalg.norm(tl - origin))
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Derived work area dimensions from homography must be positive")
    return WorkArea(
        width_mm=width_mm,
        height_mm=height_mm,
        origin_tag_id=origin_tag_id,
        size_mm=size_mm,
    )


def scale_tag_layout(
    tag_specs: list[AprilTagSpec],
    work_area: WorkArea,
    factor: float,
) -> tuple[list[AprilTagSpec], WorkArea]:
    scaled_specs = [
        spec.model_copy(
            update={
                "x_mm": spec.x_mm * factor,
                "y_mm": spec.y_mm * factor,
            }
        )
        for spec in tag_specs
    ]
    scaled_work_area = WorkArea(
        width_mm=work_area.width_mm * factor,
        height_mm=work_area.height_mm * factor,
        origin_tag_id=work_area.origin_tag_id,
        size_mm=work_area.size_mm,
        mode=work_area.mode,
    )
    return scaled_specs, scaled_work_area


def scale_tag_layout_anisotropic(
    tag_specs: list[AprilTagSpec],
    work_area: WorkArea,
    scale_x: float,
    scale_y: float,
) -> tuple[list[AprilTagSpec], WorkArea]:
    scaled_specs = [
        spec.model_copy(
            update={
                "x_mm": spec.x_mm * scale_x,
                "y_mm": spec.y_mm * scale_y,
            }
        )
        for spec in tag_specs
    ]
    scaled_work_area = WorkArea(
        width_mm=work_area.width_mm * scale_x,
        height_mm=work_area.height_mm * scale_y,
        origin_tag_id=work_area.origin_tag_id,
        size_mm=work_area.size_mm,
        mode=work_area.mode,
    )
    return scaled_specs, scaled_work_area


def rebuild_corner_tag_specs(
    work_area: WorkArea,
    origin_tag_id: int,
    tag_ids: list[int],
    image_centers: dict[int, np.ndarray],
) -> list[AprilTagSpec]:
    roles = _classify_corner_tags(origin_tag_id, image_centers)
    width_mm = work_area.width_mm
    height_mm = work_area.height_mm
    bed_coords = {
        roles["origin"]: (0.0, 0.0),
        roles["br"]: (width_mm, 0.0),
        roles["tl"]: (0.0, height_mm),
        roles["tr"]: (width_mm, height_mm),
    }
    return [
        AprilTagSpec(
            id=tag_id,
            x_mm=bed_coords[tag_id][0],
            y_mm=bed_coords[tag_id][1],
            size_mm=work_area.size_mm,
        )
        for tag_id in tag_ids
        if tag_id in bed_coords
    ]

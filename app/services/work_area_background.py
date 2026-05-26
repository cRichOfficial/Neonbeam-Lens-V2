from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.services.pipeline_debug_mosaic import put_text_outlined

from app.config import expand_path, get_config_store
from app.services.work_area_renderer import WorkAreaView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkAreaBackgroundMetadata:
    timestamp: str
    pixels_per_mm: float
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float
    max_edge_px: int
    origin_tag_id: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "pixels_per_mm": self.pixels_per_mm,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "max_edge_px": self.max_edge_px,
            "origin_tag_id": self.origin_tag_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> WorkAreaBackgroundMetadata:
        return cls(
            timestamp=str(payload["timestamp"]),
            pixels_per_mm=float(payload["pixels_per_mm"]),
            width_px=int(payload["width_px"]),
            height_px=int(payload["height_px"]),
            width_mm=float(payload["width_mm"]),
            height_mm=float(payload["height_mm"]),
            max_edge_px=int(payload["max_edge_px"]),
            origin_tag_id=int(payload["origin_tag_id"]),
        )


@dataclass(frozen=True)
class WorkAreaBackgroundStatus:
    present: bool
    timestamp: str | None = None
    pixels_per_mm: float | None = None
    width_px: int | None = None
    height_px: int | None = None
    stale_reason: str | None = None


class WorkAreaBackgroundStore:
    def __init__(self) -> None:
        self._image_path: Path | None = None
        self._metadata_path: Path | None = None

    def _paths(self) -> tuple[Path, Path]:
        cfg = get_config_store().config.detection
        image_path = expand_path(cfg.background_storage_path)
        metadata_path = image_path.with_suffix(".json")
        return image_path, metadata_path

    def is_present(self) -> bool:
        image_path, metadata_path = self._paths()
        return image_path.exists() and metadata_path.exists()

    def clear(self) -> None:
        image_path, metadata_path = self._paths()
        for path in (image_path, metadata_path):
            if path.exists():
                path.unlink()

    def save(self, view: WorkAreaView, *, max_edge_px: int) -> WorkAreaBackgroundMetadata:
        image_path, metadata_path = self._paths()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), view.image):
            raise RuntimeError(f"Failed to write background image to {image_path}")

        metadata = WorkAreaBackgroundMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pixels_per_mm=float(view.pixels_per_mm),
            width_px=int(view.width_px),
            height_px=int(view.height_px),
            width_mm=float(view.width_mm),
            height_mm=float(view.height_mm),
            max_edge_px=int(max_edge_px),
            origin_tag_id=int(view.origin_tag_id),
        )
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        return metadata

    def load_metadata(self) -> WorkAreaBackgroundMetadata | None:
        if not self.is_present():
            return None
        _, metadata_path = self._paths()
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return WorkAreaBackgroundMetadata.from_dict(payload)

    def load_image(self) -> np.ndarray | None:
        if not self.is_present():
            return None
        image_path, _ = self._paths()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return image

    def stale_reason_for_view(self, view: WorkAreaView, *, max_edge_px: int) -> str | None:
        metadata = self.load_metadata()
        if metadata is None:
            return "missing"
        if metadata.width_px != view.width_px or metadata.height_px != view.height_px:
            return "dimension_mismatch"
        if abs(metadata.pixels_per_mm - view.pixels_per_mm) > 1e-3:
            return "scale_mismatch"
        if metadata.max_edge_px != max_edge_px:
            return "max_edge_mismatch"
        if metadata.origin_tag_id != view.origin_tag_id:
            return "origin_mismatch"
        return None

    def load_for_view(
        self,
        view: WorkAreaView,
        *,
        max_edge_px: int,
    ) -> tuple[np.ndarray | None, str | None]:
        stale = self.stale_reason_for_view(view, max_edge_px=max_edge_px)
        if stale is not None:
            return None, stale
        return self.load_image(), None

    def get_status(self, view: WorkAreaView | None = None, *, max_edge_px: int | None = None) -> WorkAreaBackgroundStatus:
        if not self.is_present():
            return WorkAreaBackgroundStatus(present=False)
        metadata = self.load_metadata()
        stale_reason = None
        if view is not None and max_edge_px is not None:
            stale_reason = self.stale_reason_for_view(view, max_edge_px=max_edge_px)
        return WorkAreaBackgroundStatus(
            present=True,
            timestamp=metadata.timestamp if metadata else None,
            pixels_per_mm=metadata.pixels_per_mm if metadata else None,
            width_px=metadata.width_px if metadata else None,
            height_px=metadata.height_px if metadata else None,
            stale_reason=stale_reason,
        )

    @staticmethod
    def render_diff(current_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
        """Grayscale |current − reference| for debug (black = identical pixels)."""
        if current_bgr.shape != reference_bgr.shape:
            reference_bgr = cv2.resize(
                reference_bgr,
                (current_bgr.shape[1], current_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        diff = cv2.absdiff(current_bgr, reference_bgr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        max_val = int(np.max(gray))
        put_text_outlined(
            output,
            f"absdiff max={max_val} (black=same)",
            (10, 24),
            font_scale=0.55,
            thickness=2,
        )
        return output


_store: WorkAreaBackgroundStore | None = None


def get_work_area_background_store() -> WorkAreaBackgroundStore:
    global _store
    if _store is None:
        _store = WorkAreaBackgroundStore()
    return _store

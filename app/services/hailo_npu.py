from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.hailo_runtime import DirectHailoModel

logger = logging.getLogger(__name__)

_INIT_RETRY_COOLDOWN_S = 60.0
NPU_ACCESS_MODE = "direct"


def format_load_error(exc: Exception) -> str:
    msg = str(exc)
    if "HAILO_OUT_OF_PHYSICAL_DEVICES" in msg or "error: 74" in msg:
        from app.services.hailo_diagnostics import format_out_of_devices_error

        return format_out_of_devices_error()
    return msg


@dataclass
class HailoModelHandle:
    key: str
    model_path: Path
    hailo: DirectHailoModel
    input_size: tuple[int, int]


class HailoNpuManager:
    """Process-wide singleton for direct Hailo NPU access (picamera2-compatible)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, HailoModelHandle] = {}
        self._last_error: str | None = None
        self._retry_after = 0.0
        self._blockers: list[str] = []

    def set_preflight_blockers(self, blockers: list[str]) -> None:
        with self._lock:
            self._blockers = list(blockers)
            if blockers:
                self._last_error = (
                    "NPU preflight blocked load — another app instance is running. "
                    "See startup logs or /health npu.blockers for details."
                )

    def get_blockers(self) -> list[str]:
        with self._lock:
            return list(self._blockers)

    def is_loaded(self, key: str) -> bool:
        with self._lock:
            return key in self._models

    def get_input_size(self, key: str) -> tuple[int, int] | None:
        with self._lock:
            handle = self._models.get(key)
            return handle.input_size if handle else None

    def load_model(self, key: str, hef_path: Path | None, *, force: bool = False) -> bool:
        with self._lock:
            return self._load_model_locked(key, hef_path, force=force)

    def _load_model_locked(
        self, key: str, hef_path: Path | None, *, force: bool = False
    ) -> bool:
        if key in self._models:
            return True

        now = time.monotonic()
        if not force and self._last_error and now < self._retry_after:
            return False

        if self._blockers and not force:
            return False

        if hef_path is None or not hef_path.exists():
            self._last_error = f"no HEF model found for {key}"
            return False

        from app.services.hailo_diagnostics import (
            detect_hailort_service_status,
            format_hailort_active_error,
            format_no_hardware_error,
            probe_hailo_hardware,
        )

        hardware = probe_hailo_hardware()
        if hardware.get("present") is False and not force:
            self._last_error = format_no_hardware_error()
            logger.warning("Hailo NPU not present — skipping FastSAM load")
            return False

        if detect_hailort_service_status() == "active" and not force:
            self._last_error = format_hailort_active_error()
            logger.error(self._last_error)
            return False

        hailo = None
        try:
            hailo = DirectHailoModel(str(hef_path))
            model_h, model_w, _ = hailo.get_input_shape()
            self._models[key] = HailoModelHandle(
                key=key,
                model_path=hef_path,
                hailo=hailo,
                input_size=(model_w, model_h),
            )
            self._last_error = None
            self._retry_after = 0.0
            self._blockers = []
            logger.info("Loaded Hailo model %s from %s (direct VDevice)", key, hef_path)
            return True
        except Exception as exc:
            if hailo is not None:
                try:
                    hailo.close()
                except Exception:
                    pass
            self._last_error = format_load_error(exc)
            self._retry_after = now + _INIT_RETRY_COOLDOWN_S
            if "HAILO_OUT_OF_PHYSICAL_DEVICES" in str(exc) or "error: 74" in str(exc):
                from app.services.hailo_diagnostics import detect_npu_blockers, format_out_of_devices_error

                self._last_error = format_out_of_devices_error()
                blockers = detect_npu_blockers(current_pid=os.getpid())
                if blockers:
                    self._blockers = blockers
            logger.warning("Hailo NPU unavailable for %s: %s", key, self._last_error)
            return False

    def run(self, key: str, frame: np.ndarray) -> Any:
        with self._lock:
            handle = self._models.get(key)
            if handle is None:
                return None
            return handle.hailo.run(frame)

    def get_model_status(self, key: str) -> dict[str, str | bool | None]:
        with self._lock:
            handle = self._models.get(key)
            if handle is None:
                return {
                    "loaded": False,
                    "model_path": None,
                    "last_error": self._last_error,
                }
            return {
                "loaded": True,
                "model_path": str(handle.model_path),
                "last_error": None,
            }

    def get_status(self) -> dict[str, Any]:
        from app.services.hailo_diagnostics import (
            detect_hailort_service_status,
            detect_npu_warnings,
            probe_hailo_hardware,
        )

        hardware = probe_hailo_hardware()
        with self._lock:
            models = {
                key: {
                    "loaded": True,
                    "model_path": str(handle.model_path),
                    "input_size": list(handle.input_size),
                }
                for key, handle in self._models.items()
            }
            return {
                "access_mode": NPU_ACCESS_MODE,
                "device_open": bool(self._models),
                "last_error": self._last_error,
                "blockers": list(self._blockers),
                "hardware": hardware,
                "hailort_service": detect_hailort_service_status(),
                "warnings": detect_npu_warnings(),
                "models": models,
            }

    def close_all(self) -> None:
        with self._lock:
            for handle in self._models.values():
                try:
                    handle.hailo.close()
                except Exception:
                    pass
            self._models.clear()


_hailo_npu: HailoNpuManager | None = None


def get_hailo_npu() -> HailoNpuManager:
    global _hailo_npu
    if _hailo_npu is None:
        _hailo_npu = HailoNpuManager()
    return _hailo_npu

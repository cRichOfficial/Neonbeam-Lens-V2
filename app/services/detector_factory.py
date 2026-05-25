from __future__ import annotations

from app.config import get_config_store
from app.services.cpu_detector import BaseDetector, get_cpu_detector


def get_detector() -> BaseDetector:
    config = get_config_store().config.detection
    backend = config.backend
    cpu = get_cpu_detector()

    if backend == "hailo":
        raise RuntimeError(
            "NPU is reserved for the FastSAM shapes pipeline; use detection.backend: cpu or auto"
        )

    if backend == "cpu":
        if cpu.is_available():
            return cpu
        raise RuntimeError("CPU backend requested but no model available")

    if cpu.is_available():
        return cpu
    return cpu

from __future__ import annotations

from app.config import get_config_store
from app.services.cpu_detector import BaseDetector, CPUDetector, get_cpu_detector
from app.services.hailo_detector import HailoDetector, get_hailo_detector


def get_detector() -> BaseDetector:
    config = get_config_store().config.detection
    backend = config.backend

    hailo = get_hailo_detector()
    cpu = get_cpu_detector()

    if backend == "hailo":
        if hailo.is_available():
            return hailo
        raise RuntimeError("Hailo backend requested but unavailable")

    if backend == "cpu":
        if cpu.is_available():
            return cpu
        raise RuntimeError("CPU backend requested but no model available")

    if hailo.is_available():
        return hailo
    if cpu.is_available():
        return cpu
    return cpu

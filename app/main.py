from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.calibration import router as calibration_router
from app.api.camera import router as camera_router
from app.api.detection import router as detection_router
from app.config import PROJECT_ROOT, get_config_store
from app.services.calibration_service import get_calibration_service
from app.services.camera_service import get_camera_service
from app.services.cpu_detector import get_cpu_detector
from app.services.hailo_detector import get_hailo_detector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    camera = get_camera_service()
    camera.start()
    yield
    get_hailo_detector().close()
    camera.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Laser Engraver Object Detection",
        description="Camera, calibration, and object detection API for laser engraver bed",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(camera_router)
    app.include_router(calibration_router)
    app.include_router(detection_router)

    web_dir = PROJECT_ROOT / "web"
    if web_dir.exists():
        app.mount("/annotate", StaticFiles(directory=str(web_dir), html=True), name="annotate")

    @app.get("/health")
    def health_check() -> dict:
        camera = get_camera_service()
        calibration = get_calibration_service()
        hailo = get_hailo_detector()
        cpu = get_cpu_detector()
        config = get_config_store().config

        backend = "none"
        if hailo.is_available():
            backend = "hailo"
        elif cpu.is_available():
            backend = "cpu"

        camera_hint = None
        if camera.is_mock():
            camera_hint = (
                "picamera2 not found in this Python environment. "
                "On Pi: run 'bash deploy/setup-pi.sh' or recreate the venv with "
                "'python3 -m venv .venv --system-site-packages' after "
                "'sudo apt install python3-picamera2'."
            )

        return {
            "status": "ok",
            "camera": {
                "available": camera.is_available(),
                "mode": camera.mode_name(),
                "mock": camera.is_mock(),
                "hint": camera_hint,
            },
            "calibration": calibration.get_status(),
            "detection": {
                "backend": backend,
                "configured_backend": config.detection.backend,
            },
        }

    return app


app = create_app()

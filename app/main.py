from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.calibration import router as calibration_router
from app.api.camera import router as camera_router
from app.api.dataset import router as dataset_router
from app.api.detection import router as detection_router
from app.config import PROJECT_ROOT, get_config_store
from app.services.calibration_service import get_calibration_service
from app.services.camera_service import get_camera_service
from app.services.cpu_detector import get_cpu_detector
from app.services.fastsam_detector import FASTSAM_MODEL_KEY, get_fastsam_detector, resolve_fastsam_path
from app.services.hailo_diagnostics import (
    detect_npu_blockers,
    detect_npu_warnings,
    format_blocker_message,
)
from app.services.hailo_npu import get_hailo_npu

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    camera = get_camera_service()
    npu = get_hailo_npu()
    shapes_cfg = get_config_store().config.shapes

    # picamera2 Hailo examples open the NPU before Picamera2 in the same process.
    if shapes_cfg.backend in ("auto", "fastsam"):
        for warning in detect_npu_warnings():
            logger.warning(warning)

        blockers = detect_npu_blockers()
        if blockers:
            npu.set_preflight_blockers(blockers)
            logger.error(format_blocker_message(blockers))
        elif not npu.load_model(FASTSAM_MODEL_KEY, resolve_fastsam_path()):
            logger.warning(
                "FastSAM not loaded at startup (%s); shapes will use classical CV only "
                "unless FastSAM becomes available later.",
                npu.get_status().get("last_error") or "unknown error",
            )

    camera.start()

    yield
    camera.stop()
    npu.close_all()


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
    app.include_router(dataset_router)

    web_dist = PROJECT_ROOT / "web" / "dist"
    web_dir = web_dist if web_dist.exists() else PROJECT_ROOT / "web"
    if web_dir.exists():
        app.mount("/annotate", StaticFiles(directory=str(web_dir), html=True), name="annotate")

    @app.get("/health")
    def health_check() -> dict:
        camera = get_camera_service()
        calibration = get_calibration_service()
        cpu = get_cpu_detector()
        config = get_config_store().config
        npu_status = get_hailo_npu().get_status()

        backend = "cpu" if cpu.is_available() else "none"

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
                "hailo": {
                    "enabled": False,
                    "reason": "NPU reserved for FastSAM shapes pipeline",
                },
            },
            "npu": npu_status,
            "shapes": {
                "configured_backend": config.shapes.backend,
                "fastsam": get_fastsam_detector().get_status(),
            },
        }

    return app


app = create_app()

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse

from app.schemas.camera import CameraSettingsResponse, CameraSettingsUpdate, CameraStreamSize
from app.services.camera_service import CameraService, get_camera_service

router = APIRouter(prefix="/api/v1/camera", tags=["camera"])


def _to_response(settings: dict) -> CameraSettingsResponse:
    return CameraSettingsResponse(**settings)


@router.get("/settings", response_model=CameraSettingsResponse)
def get_camera_settings(camera: CameraService = Depends(get_camera_service)) -> CameraSettingsResponse:
    return _to_response(camera.get_settings())


@router.put("/settings", response_model=CameraSettingsResponse)
def update_camera_settings(
    payload: CameraSettingsUpdate,
    camera: CameraService = Depends(get_camera_service),
) -> CameraSettingsResponse:
    settings = camera.update_settings(
        exposure_ms=payload.exposure_ms,
        analogue_gain=payload.analogue_gain,
        mount_height_mm=payload.mount_height_mm,
    )
    return _to_response(settings)


@router.get("/snapshot")
def capture_snapshot(camera: CameraService = Depends(get_camera_service)) -> Response:
    jpeg = camera.capture_jpeg()
    return Response(content=jpeg, media_type="image/jpeg")


def _mjpeg_generator(camera: CameraService, stream: str):
    boundary = b"frame"
    for frame in camera.iter_mjpeg_frames(stream=stream):
        yield (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@router.get("/stream")
def camera_stream(
    size: CameraStreamSize = Query(
        default="preview",
        description=(
            "Stream resolution. `preview` (default): hardware MJPEG 960×540 16:9. "
            "`main`: full-res 1920×1080 software encode. `lores`: alias for `preview`."
        ),
    ),
    camera: CameraService = Depends(get_camera_service),
) -> StreamingResponse:
    stream = "preview" if size == "lores" else size
    return StreamingResponse(
        _mjpeg_generator(camera, stream=stream),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, private",
            "Pragma": "no-cache",
            "Age": "0",
        },
    )

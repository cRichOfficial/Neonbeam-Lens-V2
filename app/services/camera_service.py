from __future__ import annotations

import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from app.config import ConfigStore, get_config_store

logger = logging.getLogger(__name__)

# libcamera caps exposure at the frame period; extend frame duration for long exposures.
_FRAME_READOUT_MARGIN_US = 10_000
_MIN_FRAME_DURATION_US = 20_000
_MAX_FRAME_DURATION_US = 10_000_000


def controls_for_manual_exposure(exposure_us: int, analogue_gain: float) -> dict[str, Any]:
    """Build picamera2 controls that allow the requested exposure time."""
    frame_duration = min(
        max(exposure_us + _FRAME_READOUT_MARGIN_US, _MIN_FRAME_DURATION_US),
        _MAX_FRAME_DURATION_US,
    )
    frame_rate = 1_000_000 / frame_duration
    return {
        "AeEnable": False,
        "ExposureTime": exposure_us,
        "AnalogueGain": analogue_gain,
        "FrameDurationLimits": (frame_duration, frame_duration),
        "FrameRate": frame_rate,
    }


class StreamingOutput(io.BufferedIOBase):
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)

    def readable(self) -> bool:
        return False


def iter_streaming_output_frames(output: StreamingOutput, is_active) -> Any:
    """Yield each new hardware-encoded JPEG; wait for notify before every read."""
    while is_active():
        with output.condition:
            output.condition.wait(timeout=5.0)
            frame = output.frame
        if frame is not None:
            yield frame


def _encode_jpeg_rgb(frame: np.ndarray, quality: int) -> bytes:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return encoded.tobytes()


class CameraBackend(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def mode_name(self) -> str: ...

    @abstractmethod
    def capture_jpeg(self) -> bytes: ...

    @abstractmethod
    def capture_frame(self) -> np.ndarray: ...

    @abstractmethod
    def capture_lores_frame(self) -> np.ndarray: ...

    @abstractmethod
    def set_controls(self, exposure_us: int | None, analogue_gain: float | None) -> None: ...

    @abstractmethod
    def get_controls(self) -> dict[str, Any]: ...

    @abstractmethod
    def iter_preview_mjpeg_frames(self): ...

    @abstractmethod
    def iter_main_mjpeg_frames(self): ...


class MockCameraBackend(CameraBackend):
    def __init__(self, config_store: ConfigStore) -> None:
        self.config_store = config_store
        self._started = False
        self._frame_index = 0
        self._exposure_us = config_store.config.camera.exposure_us
        self._analogue_gain = config_store.config.camera.analogue_gain

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def is_available(self) -> bool:
        return self._started

    def mode_name(self) -> str:
        return "mock"

    def _render_frame(self, width: int, height: int) -> np.ndarray:
        self._frame_index += 1
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        cv2.putText(
            frame,
            "Mock camera - no picamera2",
            (40, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {self._frame_index}",
            (40, height // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return frame

    def capture_frame(self) -> np.ndarray:
        cfg = self.config_store.config.camera
        width, height = cfg.main_resolution
        return self._render_frame(width, height)

    def capture_lores_frame(self) -> np.ndarray:
        cfg = self.config_store.config.camera
        width, height = cfg.lores_resolution
        return self._render_frame(width, height)

    def capture_jpeg(self) -> bytes:
        frame = self.capture_frame()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode mock JPEG")
        return encoded.tobytes()

    def set_controls(self, exposure_us: int | None, analogue_gain: float | None) -> None:
        if exposure_us is not None:
            self._exposure_us = exposure_us
        if analogue_gain is not None:
            self._analogue_gain = analogue_gain

    def get_controls(self) -> dict[str, Any]:
        return {
            "ExposureTime": self._exposure_us,
            "AnalogueGain": self._analogue_gain,
        }

    def iter_preview_mjpeg_frames(self):
        cfg = self.config_store.config.camera
        interval = 1.0 / cfg.stream_max_fps
        while self._started:
            frame = self.capture_lores_frame()
            yield _encode_jpeg_rgb(frame, cfg.stream_jpeg_quality)
            time.sleep(interval)

    def iter_main_mjpeg_frames(self):
        cfg = self.config_store.config.camera
        interval = 1.0 / cfg.stream_max_fps
        while self._started:
            frame = self.capture_frame()
            yield _encode_jpeg_rgb(frame, cfg.stream_jpeg_quality)
            time.sleep(interval)


class Picamera2Backend(CameraBackend):
    def __init__(self, config_store: ConfigStore) -> None:
        self.config_store = config_store
        self._picam2 = None
        self._encoder = None
        self._output: StreamingOutput | None = None
        self._io_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput

        cfg = self.config_store.config.camera
        main_size = tuple(cfg.main_resolution)
        lores_size = tuple(cfg.lores_resolution)

        picam2 = Picamera2()
        try:
            config = picam2.create_video_configuration(
                main={"size": main_size, "format": "RGB888"},
                lores={"size": lores_size, "format": "RGB888"},
                buffer_count=4,
                controls=controls_for_manual_exposure(cfg.exposure_us, cfg.analogue_gain),
            )
            picam2.configure(config)
            output = StreamingOutput()
            encoder = JpegEncoder()
            picam2.start_recording(encoder, FileOutput(output), name="lores")
        except Exception:
            try:
                picam2.close()
            except Exception:
                pass
            raise

        self._picam2 = picam2
        self._output = output
        self._encoder = encoder
        self._started = True

    def stop(self) -> None:
        if self._picam2 is None:
            return
        with self._io_lock:
            try:
                self._picam2.stop_recording()
            except Exception:
                pass
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
        self._picam2 = None
        self._started = False

    def is_available(self) -> bool:
        return self._started and self._picam2 is not None

    def mode_name(self) -> str:
        return "picamera2"

    def _capture_array(self, stream: str = "main") -> np.ndarray:
        if self._picam2 is None:
            raise RuntimeError("Camera not started")
        with self._io_lock:
            return self._picam2.capture_array(stream)

    def capture_frame(self) -> np.ndarray:
        frame = self._capture_array("main")
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame

    def capture_lores_frame(self) -> np.ndarray:
        frame = self._capture_array("lores")
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        return frame

    def capture_jpeg(self) -> bytes:
        frame = self.capture_frame()
        return _encode_jpeg_rgb(frame, 90)

    def set_controls(self, exposure_us: int | None, analogue_gain: float | None) -> None:
        if self._picam2 is None:
            return
        cfg = self.config_store.config.camera
        current = self.get_controls()
        resolved_exposure = exposure_us
        if resolved_exposure is None:
            resolved_exposure = int(current.get("ExposureTime") or cfg.exposure_us)
        resolved_gain = analogue_gain
        if resolved_gain is None:
            resolved_gain = float(current.get("AnalogueGain") or cfg.analogue_gain)
        controls = controls_for_manual_exposure(resolved_exposure, resolved_gain)
        with self._io_lock:
            self._picam2.set_controls(controls)

    def get_controls(self) -> dict[str, Any]:
        if self._picam2 is None:
            return {}
        with self._io_lock:
            metadata = self._picam2.capture_metadata()
        return {
            "ExposureTime": metadata.get("ExposureTime"),
            "AnalogueGain": metadata.get("AnalogueGain"),
        }

    def iter_preview_mjpeg_frames(self):
        if self._output is None:
            return
        yield from iter_streaming_output_frames(self._output, lambda: self._started)

    def iter_main_mjpeg_frames(self):
        cfg = self.config_store.config.camera
        interval = 1.0 / cfg.stream_max_fps
        while self._started:
            frame = self.capture_frame()
            yield _encode_jpeg_rgb(frame, cfg.stream_jpeg_quality)
            time.sleep(interval)


class CameraService:
    def __init__(self, config_store: ConfigStore | None = None) -> None:
        self.config_store = config_store or get_config_store()
        self._backend: CameraBackend | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._backend is not None and self._backend.is_available():
                return
            try:
                backend: CameraBackend = Picamera2Backend(self.config_store)
                backend.start()
                self._backend = backend
                logger.info("Started picamera2 backend")
            except Exception as exc:
                hint = ""
                if isinstance(exc, ModuleNotFoundError) and exc.name == "picamera2":
                    hint = (
                        " Fix: sudo apt install python3-picamera2 && "
                        "python3 -m venv .venv --system-site-packages "
                        "(see deploy/setup-pi.sh)"
                    )
                logger.warning("picamera2 unavailable (%s), using mock camera.%s", exc, hint)
                backend = MockCameraBackend(self.config_store)
                backend.start()
                self._backend = backend

    def stop(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.stop()
                self._backend = None

    @property
    def backend(self) -> CameraBackend:
        if self._backend is None:
            raise RuntimeError("Camera service not started")
        return self._backend

    def is_available(self) -> bool:
        return self._backend is not None and self._backend.is_available()

    def mode_name(self) -> str:
        if self._backend is None:
            return "stopped"
        return self._backend.mode_name()

    def is_mock(self) -> bool:
        if self._backend is None:
            return False
        return self._backend.mode_name() == "mock"

    def get_settings(self) -> dict[str, Any]:
        cfg = self.config_store.config.camera
        controls = self.backend.get_controls() if self.is_available() else {}
        exposure_us_configured = cfg.exposure_us
        exposure_us_actual = controls.get("ExposureTime")
        if exposure_us_actual is not None:
            exposure_us_actual = int(exposure_us_actual)
        gain_configured = cfg.analogue_gain
        gain_actual = controls.get("AnalogueGain")
        if gain_actual is not None:
            gain_actual = float(gain_actual)
        return {
            "exposure_ms": exposure_us_configured / 1000.0,
            "exposure_ms_actual": (
                exposure_us_actual / 1000.0 if exposure_us_actual is not None else None
            ),
            "analogue_gain": gain_configured,
            "analogue_gain_actual": gain_actual,
            "mount_height_mm": cfg.mount_height_mm,
            "main_resolution": cfg.main_resolution,
            "lores_resolution": cfg.lores_resolution,
            "camera_available": self.is_available(),
            "camera_mode": self.mode_name(),
        }

    def update_settings(
        self,
        exposure_ms: float | None = None,
        analogue_gain: float | None = None,
        mount_height_mm: float | None = None,
    ) -> dict[str, Any]:
        exposure_us: int | None = None
        if exposure_ms is not None:
            exposure_us = int(round(exposure_ms * 1000))

        patch: dict[str, Any] = {"camera": {}}
        if exposure_us is not None:
            patch["camera"]["exposure_us"] = exposure_us
        if analogue_gain is not None:
            patch["camera"]["analogue_gain"] = analogue_gain
        if mount_height_mm is not None:
            patch["camera"]["mount_height_mm"] = mount_height_mm
        if patch["camera"]:
            self.config_store.update(patch)
        if self.is_available():
            self.backend.set_controls(exposure_us, analogue_gain)
        return self.get_settings()

    def capture_jpeg(self) -> bytes:
        return self.backend.capture_jpeg()

    def capture_frame(self) -> np.ndarray:
        return self.backend.capture_frame()

    def capture_lores_frame(self) -> np.ndarray:
        return self.backend.capture_lores_frame()

    def iter_mjpeg_frames(self, stream: str = "preview"):
        if stream == "main":
            yield from self.backend.iter_main_mjpeg_frames()
        else:
            yield from self.backend.iter_preview_mjpeg_frames()


_camera_service: CameraService | None = None


def get_camera_service() -> CameraService:
    global _camera_service
    if _camera_service is None:
        _camera_service = CameraService()
    return _camera_service

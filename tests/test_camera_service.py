"""Camera service tests — no Pi hardware."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from app.config import get_config_store
from app.services.image_encoding import encode_jpeg_rgb
from app.services.camera_service import (
    MockCameraBackend,
    StreamingOutput,
    controls_for_manual_exposure,
    iter_streaming_output_frames,
)


def test_controls_for_short_exposure() -> None:
    controls = controls_for_manual_exposure(20_000, 1.0)
    assert controls["ExposureTime"] == 20_000
    assert controls["AeEnable"] is False
    assert controls["FrameDurationLimits"][0] >= 30_000
    assert controls["FrameRate"] > 10


def test_controls_for_long_exposure() -> None:
    exposure_us = 750_000
    controls = controls_for_manual_exposure(exposure_us, 1.5)
    frame_duration = controls["FrameDurationLimits"][0]
    assert frame_duration >= exposure_us
    assert controls["FrameRate"] == 1_000_000 / frame_duration
    assert controls["FrameRate"] < 2.0


def test_streaming_output_waits_for_each_notify() -> None:
    output = StreamingOutput()
    gen = iter_streaming_output_frames(output, lambda: True)

    with output.condition:
        output.frame = b"first"
        output.condition.notify_all()
    assert next(gen) == b"first"

    blocked: list[bytes] = []

    def consume_second() -> None:
        blocked.append(next(gen))

    thread = threading.Thread(target=consume_second)
    thread.start()
    time.sleep(0.05)
    assert not blocked
    assert thread.is_alive()

    with output.condition:
        output.frame = b"second"
        output.condition.notify_all()
    thread.join(timeout=1.0)
    assert blocked == [b"second"]


def test_mock_preview_stream_produces_distinct_frames() -> None:
    backend = MockCameraBackend(get_config_store())
    backend.start()
    try:
        frames = [next(backend.iter_preview_mjpeg_frames()) for _ in range(2)]
    finally:
        backend.stop()
    assert frames[0] != frames[1]


def test_encode_jpeg_preserves_red_channel() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)
    jpeg = encode_jpeg_rgb(frame, quality=90)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    b, _g, r = decoded[16, 16]
    assert r > b
    assert r > 200


def test_mock_main_stream_produces_decodable_jpeg() -> None:
    backend = MockCameraBackend(get_config_store())
    backend.start()
    try:
        jpeg = next(backend.iter_main_mjpeg_frames())
    finally:
        backend.stop()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.ndim == 3

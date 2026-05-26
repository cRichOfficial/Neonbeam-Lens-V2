"""Camera exposure control tests — no Pi hardware."""

from __future__ import annotations

from app.services.camera_service import controls_for_manual_exposure


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

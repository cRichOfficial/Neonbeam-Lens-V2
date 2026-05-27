from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CameraStreamSize = Literal["preview", "main", "lores"]


class CameraSettingsResponse(BaseModel):
    exposure_ms: float
    exposure_ms_actual: float | None = None
    analogue_gain: float
    analogue_gain_actual: float | None = None
    mount_height_mm: float
    main_resolution: list[int]
    lores_resolution: list[int]
    camera_available: bool
    camera_mode: str


class CameraSettingsUpdate(BaseModel):
    exposure_ms: float | None = Field(default=None, gt=0)
    analogue_gain: float | None = Field(default=None, gt=0)
    mount_height_mm: float | None = Field(default=None, gt=0)

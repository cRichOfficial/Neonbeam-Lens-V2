from __future__ import annotations

from pydantic import BaseModel, Field


class CameraSettingsResponse(BaseModel):
    exposure_us: int
    analogue_gain: float
    mount_height_mm: float
    main_resolution: list[int]
    lores_resolution: list[int]
    camera_available: bool
    camera_mode: str


class CameraSettingsUpdate(BaseModel):
    exposure_us: int | None = Field(default=None, gt=0)
    analogue_gain: float | None = Field(default=None, gt=0)
    mount_height_mm: float | None = Field(default=None, gt=0)

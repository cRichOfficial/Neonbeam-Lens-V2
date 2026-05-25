from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def expand_path(value: str) -> Path:
    return Path(os.path.expanduser(value)).resolve()


class BedFrameConfig(BaseModel):
    origin: Literal["bottom_left", "top_left"] = "bottom_left"
    y_axis: Literal["up", "down"] = "up"


class BedConfig(BedFrameConfig):
    width_mm: float
    height_mm: float


class CameraIntrinsicsOverride(BaseModel):
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist: list[float] | None = None


class CameraConfig(BaseModel):
    mount_height_mm: float = 360
    exposure_us: int = 10000
    analogue_gain: float = 1.0
    main_resolution: list[int] = Field(default_factory=lambda: [1920, 1080])
    lores_resolution: list[int] = Field(default_factory=lambda: [640, 640])
    hfov_deg: float = 102.0
    distortion_model: Literal["pinhole", "fisheye"] = "pinhole"
    auto_distortion: bool = True
    intrinsics_override: CameraIntrinsicsOverride = Field(default_factory=CameraIntrinsicsOverride)


class ApriltagConfig(BaseModel):
    family: str = "tag36h11"
    default_size_mm: float = 20
    default_safe_zone_padding_mm: float = 5
    pdf_page_size: str = "letter"
    pdf_page_margin_mm: float = 10
    quad_decimate: float = 1.0


class DetectionConfig(BaseModel):
    model_path: str = "~/object-detection-v2/models/detection.hef"
    cpu_model_path: str = "~/object-detection-v2/models/best.pt"
    segmentation_model_path: str = "~/object-detection-v2/models/segmentation.hef"
    hailo_fallback_model: str = "/usr/share/hailo-models/yolov8s_h8l.hef"
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    backend: Literal["auto", "hailo", "cpu"] = "auto"

    @property
    def resolved_model_path(self) -> Path:
        return expand_path(self.model_path)

    @property
    def resolved_cpu_model_path(self) -> Path:
        return expand_path(self.cpu_model_path)

    @property
    def resolved_segmentation_model_path(self) -> Path:
        return expand_path(self.segmentation_model_path)


class ParallaxConfig(BaseModel):
    default_object_height_mm: float = 0


class CalibrationConfig(BaseModel):
    max_reprojection_error_mm: float = 2.0
    scale_refinement_max_iterations: int = 8
    scale_refinement_tolerance_mm: float = 0.5
    max_tag_size_error_mm: float = 1.0
    storage_path: str = "config/calibration.json"

    @property
    def resolved_storage_path(self) -> Path:
        path = Path(self.storage_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


class DatasetConfig(BaseModel):
    storage_path: str = "data/dataset"
    train_val_split: float = 0.8
    classes: list[str] = Field(default_factory=list)

    @property
    def resolved_storage_path(self) -> Path:
        path = Path(self.storage_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


class AppConfig(BaseModel):
    bed: BedFrameConfig = Field(default_factory=BedFrameConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    apriltag: ApriltagConfig = Field(default_factory=ApriltagConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    parallax: ParallaxConfig = Field(default_factory=ParallaxConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LASER_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_path: Path = DEFAULT_CONFIG_PATH
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST", "LASER_HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "LASER_PORT", "port"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)


def save_yaml_config(config: AppConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ConfigStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config_path = self.settings.config_path
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        self._config = load_yaml_config(self.config_path)

    @property
    def config(self) -> AppConfig:
        return self._config

    def reload(self) -> AppConfig:
        self._config = load_yaml_config(self.config_path)
        return self._config

    def update(self, patch: dict[str, Any]) -> AppConfig:
        current = self._config.model_dump()
        merged = _deep_merge(current, patch)
        self._config = AppConfig.model_validate(merged)
        save_yaml_config(self._config, self.config_path)
        return self._config


_config_store: ConfigStore | None = None


def get_config_store() -> ConfigStore:
    global _config_store
    if _config_store is None:
        _config_store = ConfigStore()
    return _config_store

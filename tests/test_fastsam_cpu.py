"""CPU FastSAM backend selection tests — mocked Ultralytics, no model download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import DetectionConfig
from app.services.fastsam_detector import FastSamDetector


def _patch_detection_config(monkeypatch, fastsam_device: str = "auto") -> DetectionConfig:
    detection = DetectionConfig(fastsam_device=fastsam_device)
    store = type("Store", (), {"config": type("Cfg", (), {"detection": detection})()})()
    monkeypatch.setattr("app.services.fastsam_detector.get_config_store", lambda: store)
    return detection


@pytest.fixture
def detector() -> FastSamDetector:
    return FastSamDetector()


def test_auto_prefers_hailo_when_available(detector: FastSamDetector, monkeypatch) -> None:
    _patch_detection_config(monkeypatch, "auto")

    hailo = MagicMock()
    hailo.name = "hailo"
    hailo.is_loaded.return_value = True
    hailo.try_load.return_value = True
    hailo.segment_masks.return_value = [np.ones((64, 64), dtype=np.uint8) * 255]

    cpu = MagicMock()
    cpu.name = "cpu"
    cpu.is_loaded.return_value = False
    cpu.try_load.return_value = False

    detector._hailo = hailo
    detector._cpu = cpu

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    masks = detector.segment_masks(frame)

    assert len(masks) == 1
    hailo.segment_masks.assert_called_once_with(frame)
    cpu.segment_masks.assert_not_called()
    assert detector.get_status()["device"] == "hailo"


def test_auto_falls_back_to_cpu(detector: FastSamDetector, monkeypatch) -> None:
    _patch_detection_config(monkeypatch, "auto")

    hailo = MagicMock()
    hailo.name = "hailo"
    hailo.is_loaded.return_value = False
    hailo.try_load.return_value = False
    hailo.get_status.return_value = {"last_error": "no hailo"}

    cpu = MagicMock()
    cpu.name = "cpu"
    cpu.is_loaded.return_value = True
    cpu.try_load.return_value = True
    cpu.segment_masks.return_value = [np.ones((64, 64), dtype=np.uint8) * 255]
    cpu.get_status.return_value = {"last_error": None}

    detector._hailo = hailo
    detector._cpu = cpu

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    masks = detector.segment_masks(frame)

    assert len(masks) == 1
    cpu.segment_masks.assert_called_once_with(frame)
    assert detector.get_status()["device"] == "cpu"


def test_hailo_only_skips_cpu(detector: FastSamDetector, monkeypatch) -> None:
    _patch_detection_config(monkeypatch, "hailo")

    hailo = MagicMock()
    hailo.name = "hailo"
    hailo.is_loaded.return_value = False
    hailo.try_load.return_value = False
    hailo.get_status.return_value = {"last_error": "missing hef"}

    cpu = MagicMock()
    cpu.name = "cpu"
    cpu.is_loaded.return_value = True
    cpu.try_load.return_value = True
    cpu.get_status.return_value = {"last_error": None}

    detector._hailo = hailo
    detector._cpu = cpu

    assert detector.segment_masks(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    cpu.segment_masks.assert_not_called()
    assert detector.get_status()["device"] == "none"


def test_cpu_only_uses_cpu_backend(detector: FastSamDetector, monkeypatch) -> None:
    _patch_detection_config(monkeypatch, "cpu")

    hailo = MagicMock()
    hailo.name = "hailo"
    hailo.is_loaded.return_value = True
    hailo.try_load.return_value = True

    cpu = MagicMock()
    cpu.name = "cpu"
    cpu.is_loaded.return_value = True
    cpu.try_load.return_value = True
    cpu.segment_masks.return_value = [np.ones((48, 48), dtype=np.uint8) * 255]

    detector._hailo = hailo
    detector._cpu = cpu

    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    assert len(detector.segment_masks(frame)) == 1
    hailo.segment_masks.assert_not_called()
    cpu.segment_masks.assert_called_once_with(frame)


def test_cpu_backend_loads_ultralytics(tmp_path: Path, monkeypatch) -> None:
    from app.services.fastsam_cpu_backend import CpuFastSamBackend

    model_path = tmp_path / "FastSAM-s.pt"
    model_path.write_bytes(b"fake")

    detection = DetectionConfig(fastsam_device="cpu", fastsam_cpu_model_path=str(model_path))
    store = type("Store", (), {"config": type("Cfg", (), {"detection": detection})()})()
    monkeypatch.setattr("app.services.fastsam_cpu_backend.get_config_store", lambda: store)

    mock_model = MagicMock()
    mock_fastsam_cls = MagicMock(return_value=mock_model)

    backend = CpuFastSamBackend()
    with patch.dict("sys.modules", {"ultralytics": MagicMock(FastSAM=mock_fastsam_cls)}):
        assert backend.try_load() is True
        mock_fastsam_cls.assert_called_once_with(str(model_path))

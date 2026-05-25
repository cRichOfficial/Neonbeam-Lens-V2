from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services import hailo_npu
from app.services.hailo_npu import HailoNpuManager, format_load_error, get_hailo_npu

HARDWARE_PRESENT = {"present": True, "device_nodes": ["/dev/hailo0"], "hint": None}


@pytest.fixture(autouse=True)
def reset_npu_singleton() -> None:
    hailo_npu._hailo_npu = None
    yield
    hailo_npu._hailo_npu = None


def test_get_hailo_npu_returns_singleton() -> None:
    assert get_hailo_npu() is get_hailo_npu()


def test_format_load_error_when_no_hardware() -> None:
    with patch(
        "app.services.hailo_diagnostics.probe_hailo_hardware",
        return_value={"present": False, "hint": "No Hailo NPU detected."},
    ):
        with patch("app.services.hailo_diagnostics.detect_npu_blockers", return_value=[]):
            msg = format_load_error(Exception("HAILO_OUT_OF_PHYSICAL_DEVICES(74)"))
    assert "No Hailo NPU detected" in msg


def test_load_model_skips_when_no_hardware(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    manager = HailoNpuManager()
    with patch(
        "app.services.hailo_diagnostics.probe_hailo_hardware",
        return_value={"present": False, "hint": "No Hailo NPU detected."},
    ):
        with patch("app.services.hailo_npu.DirectHailoModel") as model_cls:
            assert manager.load_model("fastsam", hef) is False
            model_cls.assert_not_called()
    assert "No Hailo NPU detected" in (manager.get_status().get("last_error") or "")


def test_load_model_skips_when_hailort_active(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    manager = HailoNpuManager()
    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="active"):
            with patch("app.services.hailo_npu.DirectHailoModel") as model_cls:
                assert manager.load_model("fastsam", hef) is False
                model_cls.assert_not_called()
    assert "hailort.service" in (manager.get_status().get("last_error") or "")


def test_load_model_is_idempotent(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    mock_hailo = MagicMock()
    mock_hailo.get_input_shape.return_value = (640, 640, 3)

    manager = HailoNpuManager()
    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="inactive"):
            with patch("app.services.hailo_npu.DirectHailoModel", return_value=mock_hailo) as model_cls:
                assert manager.load_model("fastsam", hef) is True
                assert manager.load_model("fastsam", hef) is True
                model_cls.assert_called_once_with(str(hef))

    assert manager.is_loaded("fastsam")
    assert manager.get_input_size("fastsam") == (640, 640)


def test_run_returns_none_when_model_not_loaded() -> None:
    manager = HailoNpuManager()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    assert manager.run("fastsam", frame) is None


def test_run_delegates_to_loaded_model(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    mock_hailo = MagicMock()
    mock_hailo.get_input_shape.return_value = (640, 640, 3)
    mock_hailo.run.return_value = {"masks": [np.ones((640, 640), dtype=np.uint8)]}

    manager = HailoNpuManager()
    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="inactive"):
            with patch("app.services.hailo_npu.DirectHailoModel", return_value=mock_hailo):
                manager.load_model("fastsam", hef)

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    result = manager.run("fastsam", frame)
    mock_hailo.run.assert_called_once_with(frame)
    assert "masks" in result


def test_close_all_releases_models(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    mock_hailo = MagicMock()
    mock_hailo.get_input_shape.return_value = (640, 640, 3)

    manager = HailoNpuManager()
    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="inactive"):
            with patch("app.services.hailo_npu.DirectHailoModel", return_value=mock_hailo):
                manager.load_model("fastsam", hef)

    manager.close_all()
    assert not manager.is_loaded("fastsam")
    mock_hailo.close.assert_called_once()


def test_get_status_reports_loaded_model(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    mock_hailo = MagicMock()
    mock_hailo.get_input_shape.return_value = (640, 640, 3)

    manager = HailoNpuManager()
    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="inactive"):
            with patch("app.services.hailo_npu.DirectHailoModel", return_value=mock_hailo):
                manager.load_model("fastsam", hef)

    with patch("app.services.hailo_diagnostics.probe_hailo_hardware", return_value=HARDWARE_PRESENT):
        with patch("app.services.hailo_diagnostics.detect_hailort_service_status", return_value="inactive"):
            with patch("app.services.hailo_diagnostics.detect_npu_warnings", return_value=[]):
                status = manager.get_status()

    assert status["device_open"] is True
    assert status["access_mode"] == "direct"
    assert status["hardware"]["present"] is True
    assert status["hailort_service"] == "inactive"
    assert status["models"]["fastsam"]["loaded"] is True


def test_preflight_blockers_skip_hailo_constructor(tmp_path: Path) -> None:
    hef = tmp_path / "fast_sam_s.hef"
    hef.write_bytes(b"fake")

    manager = HailoNpuManager()
    manager.set_preflight_blockers(["systemd service is active"])
    with patch("app.services.hailo_npu.DirectHailoModel") as model_cls:
        assert manager.load_model("fastsam", hef) is False
        model_cls.assert_not_called()
    assert manager.get_blockers() == ["systemd service is active"]

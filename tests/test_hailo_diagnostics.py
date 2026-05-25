from __future__ import annotations

import subprocess
from unittest.mock import patch

from app.services.hailo_diagnostics import detect_npu_blockers, format_blocker_message


def test_format_blocker_message_empty() -> None:
    assert format_blocker_message([]) == ""


def test_format_blocker_message_lists_blockers() -> None:
    msg = format_blocker_message(["systemd service is active"])
    assert "systemd service is active" in msg
    assert "scripts/start.sh --stop-service" in msg


def test_detect_npu_blockers_finds_systemd_service() -> None:
    with patch(
        "app.services.hailo_diagnostics._run_command",
        side_effect=[
            subprocess.CompletedProcess(args=[], returncode=0, stdout="active\n", stderr=""),
            None,
            None,
        ],
    ):
        blockers = detect_npu_blockers(current_pid=9999)
    assert any("laser-detection" in blocker for blocker in blockers)


def test_detect_npu_blockers_finds_other_uvicorn() -> None:
    with patch(
        "app.services.hailo_diagnostics._run_command",
        side_effect=[
            subprocess.CompletedProcess(args=[], returncode=0, stdout="inactive\n", stderr=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="1234 /home/user/.venv/bin/uvicorn app.main:app --port 8000\n",
                stderr="",
            ),
            None,
        ],
    ):
        blockers = detect_npu_blockers(current_pid=5678)
    assert any("pid 1234" in blocker for blocker in blockers)


def test_detect_npu_blockers_ignores_current_pid() -> None:
    with patch(
        "app.services.hailo_diagnostics._run_command",
        side_effect=[
            subprocess.CompletedProcess(args=[], returncode=0, stdout="inactive\n", stderr=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="7777 uvicorn app.main:app --port 8100\n",
                stderr="",
            ),
            None,
        ],
    ):
        blockers = detect_npu_blockers(current_pid=7777)
    assert blockers == []


def test_detect_npu_blockers_ignores_hailort_service() -> None:
    with patch(
        "app.services.hailo_diagnostics._run_command",
        side_effect=[
            subprocess.CompletedProcess(args=[], returncode=0, stdout="inactive\n", stderr=""),
            None,
            None,
        ],
    ):
        blockers = detect_npu_blockers(current_pid=762)
    assert blockers == []


def test_detect_npu_warnings_when_hailort_active() -> None:
    from app.services.hailo_diagnostics import detect_npu_warnings

    with patch(
        "app.services.hailo_diagnostics.probe_hailo_hardware",
        return_value={"present": True, "hint": None},
    ):
        with patch(
            "app.services.hailo_diagnostics.detect_hailort_service_status",
            return_value="active",
        ):
            warnings = detect_npu_warnings()
    assert any("stop it before this app" in warning for warning in warnings)


def test_detect_npu_warnings_when_no_hardware() -> None:
    from app.services.hailo_diagnostics import detect_npu_warnings

    with patch(
        "app.services.hailo_diagnostics.probe_hailo_hardware",
        return_value={"present": False, "hint": "No Hailo NPU detected."},
    ):
        with patch(
            "app.services.hailo_diagnostics.detect_hailort_service_status",
            return_value="inactive",
        ):
            warnings = detect_npu_warnings()
    assert any("No Hailo NPU detected" in warning for warning in warnings)


def test_probe_hailo_hardware_no_device_nodes() -> None:
    from app.services.hailo_diagnostics import probe_hailo_hardware

    with patch("app.services.hailo_diagnostics._probe_device_nodes", return_value=[]):
        with patch("app.services.hailo_diagnostics._probe_hailortcli_identify", return_value=(False, None)):
            with patch("app.services.hailo_diagnostics._probe_platform_scan", return_value=0):
                result = probe_hailo_hardware()
    assert result["present"] is False
    assert result["hint"]

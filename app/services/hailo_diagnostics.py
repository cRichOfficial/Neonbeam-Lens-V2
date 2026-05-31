from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path
from typing import Any


def _run_command(args: list[str], *, timeout: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _probe_device_nodes() -> list[str]:
    nodes = sorted(glob.glob("/dev/hailo*"))
    if Path("/dev/hailo0").exists() and "/dev/hailo0" not in nodes:
        nodes.insert(0, "/dev/hailo0")
    return nodes


def _probe_hailortcli_identify() -> tuple[bool, str | None]:
    result = _run_command(["hailortcli", "fw-control", "identify"], timeout=8.0)
    if result is None:
        return False, None
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip()
    if result.returncode != 0 or not output:
        return False, output or None
    lowered = output.lower()
    if "hailo" in lowered and ("architecture" in lowered or "device architecture" in lowered):
        return True, output
    return False, output or None


def _probe_platform_scan() -> int | None:
    try:
        from hailo_platform import Device
    except ImportError:
        return None

    for method_name in ("scan", "scan_devices"):
        scan = getattr(Device, method_name, None)
        if callable(scan):
            try:
                devices = scan()
                return len(devices) if devices is not None else 0
            except Exception:
                return None
    return None


def probe_hailo_hardware() -> dict[str, Any]:
    """Best-effort detection of whether a Hailo NPU is present on this host."""
    device_nodes = _probe_device_nodes()
    cli_ok, identify_output = _probe_hailortcli_identify()
    platform_count = _probe_platform_scan()

    present: bool | None
    if cli_ok:
        present = True
    elif platform_count is not None:
        present = platform_count > 0
    elif device_nodes:
        present = True
    elif identify_output and "fail" in identify_output.lower():
        present = False
    else:
        present = False if not device_nodes else None

    hint: str | None = None
    if present is False:
        hint = (
            "No Hailo NPU detected. Install the AI HAT on the Pi 5 PCIe connector, enable PCIe Gen 3 "
            "in raspi-config if needed, reboot, then run: hailortcli fw-control identify"
        )
    elif present is None:
        hint = "Could not confirm Hailo hardware — run: hailortcli fw-control identify"

    architecture = None
    if identify_output:
        for line in identify_output.splitlines():
            if "architecture" in line.lower():
                architecture = line.split(":", 1)[-1].strip()
                break

    return {
        "present": present,
        "device_nodes": device_nodes,
        "platform_device_count": platform_count,
        "architecture": architecture,
        "identify_ok": cli_ok,
        "identify_output": identify_output,
        "hint": hint,
    }


def detect_hailort_service_status() -> str:
    """Return systemd state for hailort.service (active/inactive/unknown)."""
    result = _run_command(["systemctl", "is-active", "hailort.service"], timeout=2.0)
    if result is None:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _systemd_main_pid(unit: str) -> int | None:
    """Return the MainPID of a systemd unit, or None if unavailable."""
    result = _run_command(
        ["systemctl", "show", "-p", "MainPID", "--value", unit],
        timeout=2.0,
    )
    if result is None:
        return None
    text = result.stdout.strip()
    if not text or text == "0":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def detect_npu_blockers(*, current_pid: int | None = None) -> list[str]:
    """Return duplicate app instances that would contend for camera/NPU access."""
    pid = current_pid if current_pid is not None else os.getpid()
    blockers: list[str] = []

    service_result = _run_command(["systemctl", "is-active", "laser-detection"], timeout=2.0)
    if service_result is not None and service_result.stdout.strip() == "active":
        main_pid = _systemd_main_pid("laser-detection")
        if main_pid is None or main_pid != pid:
            blockers.append(
                "systemd service 'laser-detection' is active on port 8000 "
                "(run: sudo systemctl stop laser-detection, or use that instance instead)"
            )

    pgrep_result = _run_command(["pgrep", "-af", "uvicorn app.main:app"], timeout=2.0)
    if pgrep_result is not None and pgrep_result.stdout.strip():
        for line in pgrep_result.stdout.strip().splitlines():
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            try:
                other_pid = int(parts[0])
            except ValueError:
                continue
            if other_pid == pid:
                continue
            detail = parts[1] if len(parts) > 1 else line
            blockers.append(f"another uvicorn instance is running (pid {other_pid}: {detail})")

    hailort_pgrep = _run_command(["pgrep", "-af", "hailort"], timeout=2.0)
    if hailort_pgrep is not None and hailort_pgrep.stdout.strip():
        hailort_status = detect_hailort_service_status()
        if hailort_status == "active":
            for line in hailort_pgrep.stdout.strip().splitlines():
                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                detail = parts[1] if len(parts) > 1 else line
                if "hailort_service" in detail:
                    blockers.append(
                        f"hailort.service daemon is running (pid {parts[0]}: {detail}) — "
                        "stop with: sudo systemctl stop hailort.service"
                    )

    return blockers


def detect_npu_warnings() -> list[str]:
    """Non-fatal NPU setup hints."""
    warnings: list[str] = []
    hardware = probe_hailo_hardware()
    if hardware.get("present") is False and hardware.get("hint"):
        warnings.append(str(hardware["hint"]))

    hailort_status = detect_hailort_service_status()
    if hailort_status == "active":
        warnings.append(
            "hailort.service is active and holds the NPU exclusively — stop it before this app "
            "can use FastSAM: sudo systemctl stop hailort.service"
        )
    elif hailort_status == "failed":
        warnings.append(
            "hailort.service is in failed state — check: sudo systemctl status hailort.service"
        )
    return warnings


def format_blocker_message(blockers: list[str]) -> str:
    if not blockers:
        return ""
    joined = "\n  - ".join(blockers)
    return (
        "Another app instance is already using the camera/NPU:\n"
        f"  - {joined}\n"
        "Stop the other instance before starting this one, or use "
        "`bash scripts/start.sh --stop-service` for manual testing."
    )


def format_hailort_active_error() -> str:
    return (
        "hailort.service holds the Hailo NPU exclusively. This app uses direct VDevice access "
        "(picamera2-style). Stop the daemon, then restart:\n"
        "  sudo systemctl stop hailort.service\n"
        "  sudo systemctl stop laser-detection   # if testing manually\n"
        "  uvicorn app.main:app --host 0.0.0.0 --port 8100"
    )


def format_no_hardware_error() -> str:
    hardware = probe_hailo_hardware()
    hint = hardware.get("hint") or (
        "No Hailo NPU detected. Install the AI HAT, reboot, and run: hailortcli fw-control identify"
    )
    return f"HAILO_OUT_OF_PHYSICAL_DEVICES — {hint} Shapes will use classical CV only until hardware is available."


def format_out_of_devices_error() -> str:
    hardware = probe_hailo_hardware()
    if hardware.get("present") is False:
        return format_no_hardware_error()

    blockers = detect_npu_blockers()
    if blockers:
        return format_blocker_message(blockers)

    if detect_hailort_service_status() == "active":
        return format_hailort_active_error()

    return (
        "HAILO_OUT_OF_PHYSICAL_DEVICES — the Hailo NPU could not be allocated. "
        "Verify the AI HAT is installed (`hailortcli fw-control identify`), stop duplicate app "
        "instances (`sudo systemctl stop laser-detection`), ensure hailort.service is stopped, "
        "or reboot if a crashed process left the device locked."
    )

#!/usr/bin/env bash
# First-time Raspberry Pi setup for object-detection-v2.
# Run on neonbeam-lens.richwerks.local after deploying the project files.
#
# Usage:
#   cd ~/object-detection-v2
#   bash deploy/setup-pi.sh
#
# Requires bash. If invoked as `sh deploy/setup-pi.sh`, re-execs under bash.

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

configure_hailort_for_direct_access() {
  echo ""
  echo "=== Hailo NPU: configure direct access (stop hailort daemon) ==="
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "WARNING: systemctl not found; skipping hailort.service configuration." >&2
    return 0
  fi

  local unit=""
  for candidate in hailort.service hailort; do
    if systemctl cat "${candidate}" &>/dev/null; then
      unit="$candidate"
      break
    fi
  done

  if [[ -z "$unit" ]]; then
    echo "No hailort systemd unit found (hailort.service / hailort) — skipping."
    echo "If the NPU still fails to load, check: ps aux | grep hailort"
    return 0
  fi

  echo "Found unit: ${unit}"
  echo "Before:"
  systemctl is-active "${unit}" 2>/dev/null || echo "  active: unknown"
  systemctl is-enabled "${unit}" 2>/dev/null || echo "  enabled: unknown"

  sudo systemctl stop "${unit}" || echo "WARNING: stop ${unit} returned non-zero (may already be stopped)" >&2
  sudo systemctl disable "${unit}" || echo "WARNING: disable ${unit} returned non-zero" >&2
  sudo systemctl mask "${unit}" 2>/dev/null || echo "NOTE: could not mask ${unit} (disable is still in effect)"

  echo "After:"
  systemctl is-active "${unit}" 2>/dev/null || echo "  active: unknown"
  systemctl is-enabled "${unit}" 2>/dev/null || echo "  enabled: unknown"
  echo "=== Hailo NPU configuration done ==="
  echo ""
}

install_cpu_torch() {
  echo ""
  echo "=== Installing CPU-only PyTorch (skip NVIDIA CUDA wheels on aarch64) ==="
  # piwheels builds torch for Raspberry Pi OS without nvidia-* CUDA packages.
  if pip install torch torchvision \
      --prefer-binary \
      --extra-index-url https://www.piwheels.org/simple; then
    echo "PyTorch installed from piwheels."
  else
    echo "piwheels unavailable for this Python version; trying PyTorch CPU index..."
    pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple
  fi

  if pip list --format=freeze | grep -qi '^nvidia-'; then
    echo "WARNING: NVIDIA CUDA packages detected after torch install." >&2
    echo "         Ultralytics CPU fallback will still work, but ~1GB of unused CUDA libs were installed." >&2
    echo "         Try: pip uninstall -y \$(pip list --format=freeze | grep -i '^nvidia-' | cut -d= -f1)" >&2
  else
    echo "No nvidia-* CUDA packages detected."
  fi

  python -c "import torch; print(f'torch {torch.__version__} (cuda build: {torch.version.cuda})')"
  echo "=== CPU PyTorch install done ==="
  echo ""
}

echo "Installing system packages (picamera2, libcamera, Hailo)..."
sudo apt update
sudo apt install -y hailo-all python3-picamera2 python3-venv python3-dev

configure_hailort_for_direct_access

if [ -d ".venv" ] && ! .venv/bin/python -c "import picamera2" 2>/dev/null; then
  echo "Removing existing venv (picamera2 not visible — venv likely missing --system-site-packages)..."
  rm -rf .venv
fi

if [ ! -d ".venv" ]; then
  echo "Creating venv with --system-site-packages (required for picamera2 + Hailo)..."
  python3 -m venv .venv --system-site-packages
fi

echo "Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip
install_cpu_torch
pip install -r requirements-pi.txt

echo "Verifying picamera2 import..."
python -c "from picamera2 import Picamera2; print('picamera2 OK')"

mkdir -p models config

FASTSAM_HEF="models/fast_sam_s.hef"
FASTSAM_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.18.0/hailo8l/fast_sam_s.hef"
SYSTEM_FASTSAM="/usr/share/hailo-models/fast_sam_s.hef"

if [ -f "$FASTSAM_HEF" ]; then
  echo "FastSAM model already present at $FASTSAM_HEF"
elif [ -f "$SYSTEM_FASTSAM" ]; then
  echo "Linking FastSAM model from $SYSTEM_FASTSAM"
  ln -sf "$SYSTEM_FASTSAM" "$FASTSAM_HEF"
else
  echo "Downloading fast_sam_s.hef from Hailo Model Zoo..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$FASTSAM_HEF" "$FASTSAM_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$FASTSAM_HEF" "$FASTSAM_URL"
  else
    echo "ERROR: curl or wget required to download $FASTSAM_HEF" >&2
    exit 1
  fi
  if [ ! -s "$FASTSAM_HEF" ]; then
    rm -f "$FASTSAM_HEF"
    echo "ERROR: FastSAM download failed or produced an empty file" >&2
    exit 1
  fi
  echo "Saved $FASTSAM_HEF"
fi

FASTSAM_PT="models/FastSAM-s.pt"
FASTSAM_PT_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"

if [ -f "$FASTSAM_PT" ]; then
  echo "CPU FastSAM model already present at $FASTSAM_PT"
else
  echo "Downloading FastSAM-s.pt for CPU fallback..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$FASTSAM_PT" "$FASTSAM_PT_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$FASTSAM_PT" "$FASTSAM_PT_URL"
  else
    echo "ERROR: curl or wget required to download $FASTSAM_PT" >&2
    exit 1
  fi
  if [ ! -s "$FASTSAM_PT" ]; then
    rm -f "$FASTSAM_PT"
    echo "ERROR: FastSAM-s.pt download failed or produced an empty file" >&2
    exit 1
  fi
  echo "Saved $FASTSAM_PT"
fi

configure_hailort_for_direct_access

echo ""
echo "Setup complete. Start manually with:"
echo "  source .venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or install the systemd service:"
echo "  sudo cp deploy/laser-detection.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now laser-detection"

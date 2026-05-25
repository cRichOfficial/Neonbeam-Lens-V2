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

echo "Installing system packages (picamera2, libcamera, Hailo)..."
sudo apt update
sudo apt install -y hailo-all python3-picamera2 python3-venv python3-dev

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
pip install -r requirements-pi.txt

echo "Verifying picamera2 import..."
python -c "from picamera2 import Picamera2; print('picamera2 OK')"

mkdir -p models data/dataset/images data/dataset/labels config

echo ""
echo "Setup complete. Start manually with:"
echo "  source .venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Or install the systemd service:"
echo "  sudo cp deploy/laser-detection.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now laser-detection"

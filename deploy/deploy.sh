#!/usr/bin/env bash
# Deploy object-detection-v2 to neonbeam-lens via scp (Linux/macOS).
#
# Usage:
#   ./deploy/deploy.sh
#   ./deploy/deploy.sh --restart

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-crichards999@neonbeam-lens}"
REMOTE_PATH="${REMOTE_PATH:-/home/crichards999/object-detection-v2}"
RESTART=false

for arg in "$@"; do
  case "$arg" in
    --restart) RESTART=true ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCHIVE_NAME="object-detection-v2-deploy.tar.gz"
ARCHIVE_PATH="/tmp/${ARCHIVE_NAME}"

cleanup() {
  rm -f "$ARCHIVE_PATH"
}
trap cleanup EXIT

echo "Project root: $PROJECT_ROOT"
echo "Creating archive..."

tar -czf "$ARCHIVE_PATH" \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='data' \
  --exclude='models' \
  --exclude='*.pyc' \
  --exclude='*.hef' \
  --exclude='*.pt' \
  --exclude='*.onnx' \
  -C "$PROJECT_ROOT" .

echo "Ensuring remote directory exists: ${REMOTE_PATH} ..."
ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_PATH}'"

echo "Uploading to ${REMOTE_HOST}:${REMOTE_PATH}/${ARCHIVE_NAME} ..."
scp "$ARCHIVE_PATH" "${REMOTE_HOST}:${REMOTE_PATH}/${ARCHIVE_NAME}"

REMOTE_CMD="set -e; cd '${REMOTE_PATH}'; tar -xzf ${ARCHIVE_NAME}; rm -f ${ARCHIVE_NAME}; echo 'Deploy extract complete.'"
if $RESTART; then
  REMOTE_CMD+=" && if systemctl is-active --quiet laser-detection; then sudo systemctl restart laser-detection && echo 'Service restarted: laser-detection'; else echo 'Service laser-detection is not active (skipped restart).'; fi"
fi

echo "Extracting on remote host..."
ssh "$REMOTE_HOST" "$REMOTE_CMD"

echo "Deploy finished successfully."

#!/usr/bin/env bash
# Start the FastAPI service using HOST and PORT from the project .env file.
#
# Usage (from project root):
#   bash scripts/start.sh
#   ./scripts/start.sh
#   ./scripts/start.sh --stop-service   # stop systemd laser-detection first
#
# Do not run with sh — requires bash. If invoked as `sh scripts/start.sh`, re-execs under bash.
# Copy .env.example to .env and edit PORT as needed (e.g. 8100 for manual testing).
#
# The Pi camera and Hailo NPU are exclusive — stop laser-detection before a manual start:
#   sudo systemctl stop laser-detection

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

STOP_SERVICE=false
for arg in "$@"; do
  case "$arg" in
    --stop-service) STOP_SERVICE=true ;;
    -h|--help)
      echo "Usage: $0 [--stop-service]"
      echo "  --stop-service  Stop laser-detection systemd unit before starting (manual testing)"
      exit 0
      ;;
  esac
done

ensure_no_device_conflict() {
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet hailort.service 2>/dev/null; then
      echo "Stopping hailort.service (required for direct NPU access)..."
      sudo systemctl stop hailort.service
    fi

    if systemctl is-active --quiet laser-detection 2>/dev/null; then
      if $STOP_SERVICE; then
        echo "Stopping laser-detection systemd service..."
        sudo systemctl stop laser-detection
      else
        echo "ERROR: laser-detection systemd service is already running (port 8000)." >&2
        echo "The camera and Hailo NPU can only be used by one app instance." >&2
        echo "" >&2
        echo "  sudo systemctl stop laser-detection   # then re-run this script" >&2
        echo "  ./scripts/start.sh --stop-service       # or stop service automatically" >&2
        echo "  curl http://localhost:8000/health       # use the running service instead" >&2
        exit 1
      fi
    fi
  fi

  local uvicorn_pids
  uvicorn_pids="$(pgrep -f '[u]vicorn app\.main:app' || true)"
  if [[ -n "$uvicorn_pids" ]]; then
    echo "ERROR: another uvicorn app.main instance is already running:" >&2
    pgrep -af '[u]vicorn app\.main:app' >&2 || true
    echo "Stop it first, or reboot if a crashed process left the camera locked." >&2
    exit 1
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="0.0.0.0"
PORT="8000"
ENV_FILE="$ROOT/.env"
VENV_ACTIVATE="$ROOT/.venv/bin/activate"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line%"${line##*[![:space:]]}"}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    val="${line#*=}"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    if [[ "$val" == \"*\" && "$val" == *\" ]]; then
      val="${val:1:${#val}-2}"
    elif [[ "$val" == \'*\' && "$val" == *\' ]]; then
      val="${val:1:${#val}-2}"
    fi
    case "$key" in
      HOST) HOST="$val" ;;
      PORT) PORT="$val" ;;
    esac
  done < "$ENV_FILE"
else
  echo "No .env found; using HOST=$HOST PORT=$PORT (copy .env.example to .env to customize)" >&2
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Python venv not found at .venv/" >&2
  echo "Run first: bash deploy/setup-pi.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_ACTIVATE"

if ! python -m uvicorn --help >/dev/null 2>&1; then
  echo "uvicorn is not installed in .venv" >&2
  echo "Run: bash deploy/setup-pi.sh" >&2
  exit 1
fi

ensure_no_device_conflict

echo "Starting uvicorn on http://${HOST}:${PORT}"
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT"

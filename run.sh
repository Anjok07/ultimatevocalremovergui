#!/usr/bin/env bash
# Launch the UVR web service natively (Linux / macOS).
# First run sets up a venv + installs deps + builds the UI; later runs just start.
# Override any setting by exporting it before running, e.g. UVR_PORT=9000 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- configuration
export UVR_HOST="${UVR_HOST:-127.0.0.1}"           # bind address
export UVR_PORT="${UVR_PORT:-8000}"                # port (SPA + API)
export UVR_DATA_DIR="${UVR_DATA_DIR:-$(pwd)/data}" # inputs / jobs / metrics live here
export UVR_OPEN_BROWSER="${UVR_OPEN_BROWSER:-1}"   # 0 to not open a browser
# Separation is CPU/RAM heavy — cap threads (also trims memory). Lower if you OOM.
export UVR_NUM_THREADS="${UVR_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$UVR_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$UVR_NUM_THREADS}"
# GPU: auto-detects CUDA (Linux/NVIDIA) or MPS (Apple Silicon). Force with 0/1.
export UVR_USE_GPU="${UVR_USE_GPU:-auto}"
# Lets the legacy `sklearn` shim (a transitive dep) install.
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

VENV=".venv"
PY="${PYTHON:-python3}"

# ------------------------------------------------------------- first-run set-up
if [ ! -d "$VENV" ]; then
  echo ">> Creating virtualenv and installing dependencies (first run)…"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip wheel
  if [ "$(uname -s)" = "Darwin" ]; then
    # macOS: default PyPI wheels are MPS-capable on Apple Silicon.
    "$VENV/bin/pip" install torch==2.2.2 torchvision==0.17.2
  else
    # Linux: pin the CPU build so we don't pull multi-GB CUDA wheels by default.
    "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
  fi
  "$VENV/bin/pip" install -r requirements.txt -r requirements-api.txt
fi

# --------------------------------------------------------------- build the UI
if [ ! -f "web/dist/index.html" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo ">> Building the web UI…"
    ( cd web && npm ci && npm run build )
  else
    echo "!! web/dist not found and npm is not installed — the UI won't be served."
    echo "   (the API still works; install Node + run 'cd web && npm ci && npm run build')"
  fi
fi

# ---------------------------------------------------------------- data folders
mkdir -p "$UVR_DATA_DIR/inputs" "$UVR_DATA_DIR/jobs" "$UVR_DATA_DIR/uploads"

# -------------------------------------------------------------------------- run
echo ">> UVR web service → http://$UVR_HOST:$UVR_PORT   (data: $UVR_DATA_DIR)"
exec "$VENV/bin/python" run_web.py

#!/usr/bin/env bash
#
# Ultimate Vocal Remover GUI - Linux installation helper
#
# Sets up a Python virtual environment and installs all dependencies,
# choosing the correct PyTorch build for your system (CUDA GPU or CPU).
#
# Usage:
#   ./install_linux.sh                 # auto-detect GPU/CPU
#   ./install_linux.sh --cpu           # force CPU-only PyTorch
#   ./install_linux.sh --cuda cu126    # force a specific CUDA wheel index
#   ./install_linux.sh --venv .venv    # custom virtualenv directory
#
# After installation:
#   source venv/bin/activate
#   python UVR.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration / argument parsing
# ---------------------------------------------------------------------------
VENV_DIR="venv"
FORCE_MODE=""          # "cpu" | "cuda" | "" (auto)
CUDA_INDEX="cu126"     # default CUDA wheel index; matches CUDA 12.x drivers

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            FORCE_MODE="cpu"; shift ;;
        --cuda)
            FORCE_MODE="cuda"; CUDA_INDEX="${2:-cu126}"; shift 2 ;;
        --venv)
            VENV_DIR="${2:?--venv requires a path}"; shift 2 ;;
        -h|--help)
            sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# System dependency check (informational; we don't auto-install with sudo)
# ---------------------------------------------------------------------------
info "Checking system dependencies (ffmpeg, python3, tk, rubberband)..."
MISSING_SYS=()
command -v ffmpeg      >/dev/null 2>&1 || MISSING_SYS+=("ffmpeg")
command -v python3     >/dev/null 2>&1 || MISSING_SYS+=("python3")
command -v rubberband  >/dev/null 2>&1 || MISSING_SYS+=("rubberband (optional: Time-Stretch / Pitch-Shift)")
python3 -c "import tkinter" >/dev/null 2>&1 || MISSING_SYS+=("python3-tk")

if [[ ${#MISSING_SYS[@]} -gt 0 ]]; then
    warn "The following system packages appear to be missing:"
    for pkg in "${MISSING_SYS[@]}"; do echo "      - $pkg"; done
    echo
    echo "  Install them first, for example:"
    echo "    Debian/Ubuntu: sudo apt-get install -y ffmpeg python3-pip python3-tk rubberband-cli"
    echo "    Arch/Endeavour: sudo pacman -S ffmpeg python-pip tk rubberband"
    echo
fi

# ---------------------------------------------------------------------------
# GPU / CUDA detection
# ---------------------------------------------------------------------------
MODE="$FORCE_MODE"
if [[ -z "$MODE" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        MODE="cuda"
        info "NVIDIA GPU detected -> installing CUDA PyTorch (index: $CUDA_INDEX)."
    else
        MODE="cpu"
        info "No NVIDIA GPU detected -> installing CPU-only PyTorch."
    fi
fi

# ---------------------------------------------------------------------------
# Virtual environment + pip front-end (prefer uv if available)
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if command -v uv >/dev/null 2>&1; then
    PIP_INSTALL=(uv pip install)
elif python -m uv --version >/dev/null 2>&1; then
    PIP_INSTALL=(python -m uv pip install)
else
    info "Upgrading pip..."
    python -m pip install --upgrade pip >/dev/null
    PIP_INSTALL=(python -m pip install)
fi
info "Using installer: ${PIP_INSTALL[*]}"

# ---------------------------------------------------------------------------
# Install PyTorch first (correct build), then the rest of requirements
# ---------------------------------------------------------------------------
if [[ "$MODE" == "cuda" ]]; then
    info "Installing torch / torchvision (CUDA $CUDA_INDEX)..."
    "${PIP_INSTALL[@]}" torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
else
    info "Installing torch / torchvision (CPU)..."
    "${PIP_INSTALL[@]}" torch torchvision \
        --index-url "https://download.pytorch.org/whl/cpu"
fi

info "Installing remaining requirements..."
"${PIP_INSTALL[@]}" -r requirements.txt

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
info "Verifying installation..."
python - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} (x{torch.cuda.device_count()})")
PY

echo
info "Done! To run the application:"
echo "    source $VENV_DIR/bin/activate"
echo "    python UVR.py"

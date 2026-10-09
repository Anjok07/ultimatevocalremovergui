#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="Ultimate Vocal Remover"
APP_PATH="$ROOT_DIR/dist/${APP_NAME}.app"
SPEC_PATH="$ROOT_DIR/packaging/macos/UVR-macos-arm64.spec"
PYTHON_BIN="$ROOT_DIR/.venv-build-macos/bin/python"
SIGNING_DIR=""

cleanup() {
  if [[ -n "$SIGNING_DIR" && -d "$SIGNING_DIR" ]]; then
    rm -rf "$SIGNING_DIR"
  fi
}
trap cleanup EXIT

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: macOS arm64 build requires an arm64 shell; got $(uname -m)" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to create the build environment" >&2
  exit 1
fi

uv python install 3.10
rm -rf "$ROOT_DIR/.venv-build-macos"
uv venv "$ROOT_DIR/.venv-build-macos" --python 3.10
"$PYTHON_BIN" -m ensurepip --upgrade
"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
"$PYTHON_BIN" -m pip install playsound==1.2.2
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True "$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip uninstall -y PySoundFile || true
"$PYTHON_BIN" -m pip install onnx2pytorch pyinstaller
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps soundfile==0.13.1

"$PYTHON_BIN" - <<'PY'
import platform
import sys
import torch
import PyInstaller

if platform.machine() != "arm64":
    raise SystemExit(f"error: expected arm64 Python, got {platform.machine()}")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"error: expected Python 3.10, got {sys.version}")
if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    raise SystemExit("error: torch MPS is not available in the build environment")
print("prebuild ok:", sys.executable, platform.platform(), "PyInstaller", PyInstaller.__version__)
PY

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "warning: ffmpeg was not found on PATH; MP3/FLAC export may fail in the packaged app" >&2
fi

if ! command -v rubberband >/dev/null 2>&1; then
  echo "warning: rubberband was not found on PATH; time-stretch/pitch-shift tools are not bundled" >&2
fi

rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"
"$PYTHON_BIN" -m PyInstaller --noconfirm --clean "$SPEC_PATH"

if [[ ! -d "$APP_PATH" ]]; then
  echo "error: expected app bundle was not created at $APP_PATH" >&2
  exit 1
fi

find "$ROOT_DIR/dist" -name '._*' -exec rm -rf {} +
SIGNING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/uvr-macos-signing.XXXXXX")"
SIGNED_APP_PATH="$SIGNING_DIR/${APP_NAME}.app"
ditto --norsrc --noextattr "$APP_PATH" "$SIGNED_APP_PATH"
find "$SIGNED_APP_PATH" -name '._*' -exec rm -rf {} +
rm -rf "$SIGNED_APP_PATH/Contents/_CodeSignature"

codesign --force --deep --sign - "$SIGNED_APP_PATH"
codesign --verify --deep --strict --verbose=2 "$SIGNED_APP_PATH"
rm -rf "$APP_PATH"
ditto --norsrc "$SIGNED_APP_PATH" "$APP_PATH"
find "$ROOT_DIR/dist" -name '._*' -exec rm -rf {} +

echo "Built and ad-hoc signed: $APP_PATH"

# PyInstaller spec for the UVR web service (single-file launcher: server + UI).
#
# Build (after `pip install -r requirements.txt -r requirements-api.txt pyinstaller`
# and `cd web && npm ci && npm run build`):
#
#     pyinstaller packaging/uvr_web.spec
#
# Produces dist/uvr-web/ — run uvr-web(.exe) to start the server and open the
# browser. Heavy native deps (torch, onnxruntime) make a one-dir build the
# pragmatic choice. The launcher (run_web.py) points UVR_DATA_DIR next to the
# executable so models/jobs are written to a writable location, not the bundle.
#
# NOTE: torch/onnxruntime bundling is finicky; this is a working starting point —
# expect to tweak hidden imports / excludes per platform. CI builds it on
# windows-latest (.github/workflows/build-windows.yml).

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
ROOT = os.path.abspath(os.getcwd())

# Bundle the heavy ML libs (binaries + data + submodules).
datas, binaries, hiddenimports = [], [], []
for pkg in ("torch", "torchvision", "onnxruntime", "librosa", "soundfile", "audioread", "demucs"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "api.main", "api.separation", "api.registry", "api.headless",
    "api.jobs", "api.schemas", "separate", "UVR",
    "anyio", "onnx2pytorch",
]

# App data the engine reads at runtime (config JSON, band params, the built UI).
datas += [
    ("web/dist", "web/dist"),
    ("gui_data", "gui_data"),
    ("lib_v5", "lib_v5"),
    ("models", "models"),            # ships the model_data/*.json (not weights)
    ("demucs", "demucs"),
    ("__version__.py", "."),
]

a = Analysis(
    ["run_web.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter.test"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="uvr-web", console=True, icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="uvr-web")

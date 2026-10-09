# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a local macOS Apple Silicon UVR build.

This bundle intentionally includes model metadata and configuration files but
excludes downloaded model weights. Users can download or place model weights in
the app's model directories after launch.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path(SPECPATH).resolve().parents[1]
APP_NAME = "Ultimate Vocal Remover"
ICON = ROOT / "gui_data" / "img" / "UVR.icns"


def add_tree(source: str, dest: str, excludes: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    source_path = ROOT / source
    if not source_path.exists():
        return []

    datas: list[tuple[str, str]] = []
    for path in source_path.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source_path)
        rel_posix = rel.as_posix()
        if any(path.match(pattern) or rel_posix.endswith(pattern) for pattern in excludes):
            continue
        datas.append((str(path), str(Path(dest) / rel.parent)))
    return datas


def add_if_exists(source: Path, dest: str) -> list[tuple[str, str]]:
    return [(str(source), dest)] if source.exists() else []


datas: list[tuple[str, str]] = []
datas += add_tree("gui_data", "gui_data", excludes=("__pycache__",))
datas += add_tree("lib_v5", "lib_v5", excludes=("__pycache__",))
datas += add_tree("demucs", "demucs", excludes=("__pycache__",))
datas += add_tree("models/VR_Models/model_data", "models/VR_Models/model_data")
datas += add_tree("models/MDX_Net_Models/model_data", "models/MDX_Net_Models/model_data")
datas += add_tree("models/Demucs_Models/model_data", "models/Demucs_Models/model_data")
datas += add_if_exists(ROOT / "models" / "Demucs_Models" / "v3_v4_repo" / "demucs_models.txt", "models/Demucs_Models/v3_v4_repo")

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    datas.append((ffmpeg_path, "."))

rubberband_path = shutil.which("rubberband")
if rubberband_path:
    datas.append((rubberband_path, "."))

datas += collect_data_files("librosa")
datas += collect_data_files("pydub")
datas += collect_data_files("sklearn")
datas += collect_data_files("onnxruntime")
datas += collect_data_files("soundfile")
datas += collect_dynamic_libs("onnxruntime")
datas += collect_dynamic_libs("soundfile")

hiddenimports = []
hiddenimports += collect_submodules("demucs")
hiddenimports += collect_submodules("gui_data")
hiddenimports += collect_submodules("lib_v5")
hiddenimports += collect_submodules("onnx2pytorch")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("yaml")
hiddenimports += [
    "PIL._tkinter_finder",
    "audioread",
    "librosa",
    "ml_collections",
    "numpy",
    "pydub",
    "scipy",
    "soundfile",
    "torch",
    "torch.testing._comparison",
    "torch.testing._creation",
    "torchaudio",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "UVR.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ICON),
    bundle_identifier="com.uvr5.local",
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": "5.6-mps-local",
        "CFBundleVersion": "5.6.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)

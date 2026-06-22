"""Model discovery + download-on-demand.

Reads the bundled ``gui_data/model_manual_download.json`` for the catalog of
downloadable models, and refreshes the hash/param mappers from UVR's remote
data links so freshly downloaded weights are recognized by ``ModelData``.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from .headless import (
    DEMUCS_MODELS_DIR,
    DEMUCS_NEWER_REPO_DIR,
    MDX_HASH_JSON,
    MDX_MODELS_DIR,
    VR_HASH_JSON,
    VR_MODELS_DIR,
)
from .schemas import Arch, ModelDetail, ModelInfo

# UVR's public model + metadata repositories (mirrors gui_data/constants.py).
NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
VR_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json"
MDX_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json"
MDX_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_name_mapper.json"
DEMUCS_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/demucs_model_data/model_name_mapper.json"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_LIST_PATH = os.path.join(REPO_ROOT, "gui_data", "model_manual_download.json")

_DATA_LINKS = {
    "vr": (VR_MODEL_DATA_LINK, os.path.join(VR_MODELS_DIR, "model_data", "model_data.json")),
    "mdx": (MDX_MODEL_DATA_LINK, os.path.join(MDX_MODELS_DIR, "model_data", "model_data.json")),
    "mdx_names": (MDX_MODEL_NAME_DATA_LINK, os.path.join(MDX_MODELS_DIR, "model_data", "model_name_mapper.json")),
    "demucs_names": (DEMUCS_MODEL_NAME_DATA_LINK, os.path.join(DEMUCS_MODELS_DIR, "model_data", "model_name_mapper.json")),
}


def _load_download_lists() -> dict:
    with open(DOWNLOAD_LIST_PATH, "r") as f:
        return json.load(f)


def _installed_dir_for(arch: Arch) -> str:
    return {
        Arch.vr: VR_MODELS_DIR,
        Arch.mdx: MDX_MODELS_DIR,
        Arch.demucs: DEMUCS_MODELS_DIR,
    }[arch]


def list_models() -> list[ModelInfo]:
    """All catalog models for the three architectures, with installed flags."""
    data = _load_download_lists()
    out: list[ModelInfo] = []

    # ``name`` is the model basename ModelData resolves at separation time (the
    # file stem); ``download_name`` is the friendly catalog name for downloading.
    for display, filename in data.get("vr_download_list", {}).items():
        download_name = display.split(": ", 1)[-1]
        installed = os.path.isfile(os.path.join(VR_MODELS_DIR, filename))
        out.append(ModelInfo(
            arch=Arch.vr, name=os.path.splitext(filename)[0],
            download_name=download_name, filename=filename, installed=installed,
        ))

    for display, entry in data.get("mdx_download_list", {}).items():
        download_name = display.split(": ", 1)[-1]
        filename = list(entry.keys())[0] if isinstance(entry, dict) else str(entry)
        installed = os.path.isfile(os.path.join(MDX_MODELS_DIR, filename))
        out.append(ModelInfo(
            arch=Arch.mdx, name=os.path.splitext(filename)[0],
            download_name=download_name, filename=filename, installed=installed,
        ))

    for display, files in data.get("demucs_download_list", {}).items():
        download_name = display.split(": ", 1)[-1]
        # Demucs entries are multi-file; the .yaml stem is the model name and
        # its presence marks the model as installed.
        yaml_files = [k for k in files if k.endswith(".yaml")]
        target = yaml_files[0] if yaml_files else next(iter(files))
        is_newer = any(t in display for t in ("v3", "v4"))
        base = DEMUCS_NEWER_REPO_DIR if is_newer else DEMUCS_MODELS_DIR
        installed = os.path.isfile(os.path.join(base, target))
        out.append(ModelInfo(
            arch=Arch.demucs, name=os.path.splitext(target)[0],
            download_name=download_name, filename=target, installed=installed,
        ))

    return out


def refresh_model_data() -> None:
    """Pull the latest hash/name mappers so new downloads are recognized.

    Best-effort: network failures are swallowed (the bundled caches remain).
    """
    for url, dest in _DATA_LINKS.values():
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.load(resp)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                json.dump(payload, f, indent=4)
        except Exception:  # noqa: BLE001 - keep bundled cache on failure
            continue


def _download(url: str, dest: str, job=None, weight=1.0, base=0.0):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if job is not None and total:
                    job.update(progress=base + weight * (read / total))
    os.replace(tmp, dest)


def download_model(arch: Arch, display_name: str, job=None) -> list[str]:
    """Download a catalog model by its display name. Returns saved file paths."""
    data = _load_download_lists()
    saved: list[str] = []

    if arch == Arch.vr:
        listing = data.get("vr_download_list", {})
        match = _find(listing, display_name)
        if not match:
            raise ValueError(f"Unknown VR model: {display_name}")
        filename = match
        _download(NORMAL_REPO + filename, os.path.join(VR_MODELS_DIR, filename), job)
        saved.append(os.path.join(VR_MODELS_DIR, filename))

    elif arch == Arch.mdx:
        listing = data.get("mdx_download_list", {})
        match = _find(listing, display_name)
        if match is None:
            raise ValueError(f"Unknown MDX model: {display_name}")
        filename = list(match.keys())[0] if isinstance(match, dict) else str(match)
        _download(NORMAL_REPO + filename, os.path.join(MDX_MODELS_DIR, filename), job)
        saved.append(os.path.join(MDX_MODELS_DIR, filename))

    elif arch == Arch.demucs:
        listing = data.get("demucs_download_list", {})
        key, files = _find_keyed(listing, display_name)
        if not files:
            raise ValueError(f"Unknown Demucs model: {display_name}")
        # The version tag ("v3"/"v4") is in the catalog KEY ("Demucs v4: …"), not
        # the friendly name the UI sends ("htdemucs_ft"). v3/v4 weights live under
        # v3_v4_repo/ — checking display_name here would misfile them and the
        # installed-check (which uses the key) would never find them.
        is_newer = any(t in key for t in ("v3", "v4"))
        base_dir = DEMUCS_NEWER_REPO_DIR if is_newer else DEMUCS_MODELS_DIR
        items = list(files.items())
        for i, (fname, url) in enumerate(items):
            _download(url, os.path.join(base_dir, fname), job, weight=1.0 / len(items), base=i / len(items))
            saved.append(os.path.join(base_dir, fname))

    # Make sure the just-downloaded weights are recognized on next separation.
    refresh_model_data()
    return saved


# Secondary stem the model produces alongside its primary (mirrors UVR's
# secondary_stem(): Vocals<->Instrumental, else "No <primary>").
_STEM_PAIR = {"Vocals": "Instrumental", "Instrumental": "Vocals"}


def _secondary_stem(primary: str) -> str:
    return _STEM_PAIR.get(primary, f"No {primary}")


def _model_hash(path: str) -> str:
    """Mirror ModelData.get_model_hash: md5 of the last ~10MB (whole file fallback)."""
    try:
        with open(path, "rb") as f:
            f.seek(-10000 * 1024, 2)
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


def _load_hash_mapper(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def model_detail(arch: Arch, name: str) -> ModelDetail:
    """Stems + technical params for a model. Full details need the weights
    installed (we hash the file to look up its model_data)."""
    info = next((m for m in list_models() if m.arch == arch and m.name == name), None)
    if info is None:
        raise ValueError(f"Unknown {arch.value} model: {name}")

    detail = ModelDetail(
        arch=arch, name=info.name, download_name=info.download_name,
        filename=info.filename, installed=info.installed,
    )

    if arch == Arch.demucs:
        # Stem set is fixed by the source count; no per-file model_data to read.
        is_2 = "UVR_Model" in name
        detail.stems = ["Vocals", "Instrumental"] if is_2 else ["Vocals", "Drums", "Bass", "Other"]
        detail.note = "Demucs outputs multiple stems at once."
        if not info.installed:
            detail.note = "Not installed — download to use."
        return detail

    base = VR_MODELS_DIR if arch == Arch.vr else MDX_MODELS_DIR
    path = os.path.join(base, info.filename)
    if not info.installed or not os.path.isfile(path):
        detail.note = "Not installed — download to see stems and parameters."
        return detail

    detail.bytes = os.path.getsize(path)
    mapper = _load_hash_mapper(VR_HASH_JSON if arch == Arch.vr else MDX_HASH_JSON)
    data = mapper.get(_model_hash(path))
    if not data:
        detail.note = "Parameters not recognized for this file."
        return detail

    detail.technical = data
    if data.get("config_yaml"):  # MDX-C / multi-stem
        detail.note = "MDX23C model (config-driven multi-stem)."
        primary = data.get("primary_stem")
    else:
        primary = data.get("primary_stem")
    if primary:
        detail.primary_stem = primary
        detail.secondary_stem = _secondary_stem(primary)
        detail.stems = [primary, detail.secondary_stem]
    return detail


def _find(listing: dict, download_name: str):
    """Resolve a catalog entry by its full key or its friendly (post-': ') name."""
    return _find_keyed(listing, download_name)[1]


def _find_keyed(listing: dict, download_name: str):
    """Like ``_find`` but also returns the matched catalog key (which carries the
    Demucs version tag). Returns ``(key, value)`` or ``(None, None)``."""
    for display, value in listing.items():
        candidate = display.split(": ", 1)[-1]
        if display == download_name or candidate == download_name:
            return display, value
    return None, None

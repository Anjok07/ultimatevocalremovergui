"""Headless stand-ins for the Tkinter ``MainWindow``.

``ModelData`` (UVR.py) and the ``Seperate*`` engines (separate.py) read all of
their settings from a global ``root`` object that, in the desktop app, is the
Tkinter main window.  Every read is either ``root.<name>_var.get()`` or a call
to a helper method on ``root``.

Rather than re-implement ~400 lines of ``ModelData`` config logic, we build a
tiny object that quacks like ``MainWindow`` for exactly the attributes the
separation path touches, then inject it as ``UVR.root``.  This keeps us bug-for-
bug compatible with the GUI and resilient to upstream tweaks of ``ModelData``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Repo root (this file lives in <repo>/api/). Used for locating model dirs
# without importing the heavyweight UVR module just to read a few paths.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(REPO_ROOT, "models")
VR_MODELS_DIR = os.path.join(MODELS_DIR, "VR_Models")
MDX_MODELS_DIR = os.path.join(MODELS_DIR, "MDX_Net_Models")
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, "Demucs_Models")
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, "v3_v4_repo")

VR_HASH_JSON = os.path.join(VR_MODELS_DIR, "model_data", "model_data.json")
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, "model_data", "model_data.json")
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, "model_data", "model_name_mapper.json")
DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, "model_data", "model_name_mapper.json")


class _Var:
    """Mimic a Tkinter ``*Var`` — the engines only ever call ``.get()``/``.set()``."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _load_json(path: str) -> dict:
    import json

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


# --- Default settings -------------------------------------------------------
# Mirrors the GUI's default state for the subset of settings ``ModelData``
# reads.  Anything a request wants to override is merged on top of these.
# Keys are the tkinter var names WITHOUT the trailing ``_var``.
DEFAULT_SETTINGS: dict[str, Any] = {
    # device / general
    "device_set": "Default",
    "is_gpu_conversion": False,
    "is_normalization": False,
    "is_invert_spec": False,
    "is_primary_stem_only": False,
    "is_secondary_stem_only": False,
    "semitone_shift": "0",
    "is_match_frequency_pitch": True,
    # denoise / deverb
    "denoise_option": "None",            # DENOISE_NONE
    "is_deverb_vocals": False,
    "deverb_vocal_opt": "Main Vocals Only",
    # output format
    "save_format": "WAV",
    "mp3_bit_set": "320k",
    # secondary models / ensemble (all disabled in headless single-model mode)
    "vr_is_secondary_model_activate": False,
    "mdx_is_secondary_model_activate": False,
    "demucs_is_secondary_model_activate": False,
    "is_demucs_pre_proc_model_activate": False,
    "is_demucs_pre_proc_model_inst_mix": False,
    "is_save_inst_set_vocal_splitter": False,
    "chosen_process_method": "",
    "ensemble_main_stem": "Choose Stem Pair",
    # VR Arch
    "aggression_setting": "10",
    "is_tta": False,
    "is_post_process": False,
    "window_size": "512",
    "batch_size": "Default",
    "crop_size": "256",
    "is_high_end_process": False,
    "post_process_threshold": "0.2",
    # MDX-Net
    "margin": "44100",
    "mdx_segment_size": "256",
    "mdx_batch_size": "Default",
    "compensate": "Auto",
    "overlap": "Default",
    "overlap_mdx": "Default",
    "overlap_mdx23": "8",
    "is_mdx_c_seg_def": False,
    "is_mdx23_combine_stems": True,
    "mdxnet_stems": "All Stems",
    # Demucs
    "demucs_stems": "All Stems",
    "is_demucs_combine_stems": True,
    "margin_demucs": "44100",
    "shifts": "2",
    "segment": "Default",
    "is_split_mode": True,
    "is_chunk_demucs": False,
    "is_primary_stem_only_Demucs": False,
    "is_secondary_stem_only_Demucs": False,
}


@dataclass
class HeadlessRoot:
    """A duck-typed replacement for the Tkinter ``MainWindow``.

    Only the attributes/methods that ``ModelData`` and the ``Seperate*`` engines
    actually touch are implemented.  Settings provided via ``overrides`` win over
    :data:`DEFAULT_SETTINGS`.
    """

    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        merged = {**DEFAULT_SETTINGS, **self.overrides}
        for name, value in merged.items():
            setattr(self, f"{name}_var", _Var(value))

        # ``wav_type_set`` is a plain attribute on MainWindow, not a Var.
        self.wav_type_set = "PCM_16"

        # Hash / name mappers (model_data lookups). Loaded from the bundled JSON
        # caches; the registry refreshes these from the remote data links.
        self.vr_hash_MAPPER = _load_json(VR_HASH_JSON)
        self.mdx_hash_MAPPER = _load_json(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = _load_json(MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = _load_json(DEMUCS_MODEL_NAME_SELECT)

        # Populated only if ``ModelData`` hits the "unrecognized model" popup
        # path; in headless mode we never recognize via popup, so leave empty.
        self.vr_model_params = None
        self.mdx_model_params = None

    # --- helper methods ModelData calls on root -----------------------------
    def check_only_selection_stem(self, _checktype) -> bool:
        # Controls inst-only / vocal-only splitter behaviour; off in headless.
        return False

    def return_ensemble_stems(self, is_primary=False):
        # Only reached in ensemble mode, which headless single-model never uses.
        return None, None

    def process_determine_secondary_model(self, *_args, **_kwargs):
        # No secondary-model chaining in single-model headless separation.
        return None, None

    def process_determine_demucs_pre_proc_model(self, *_args, **_kwargs):
        return None

    def process_determine_vocal_split_model(self):
        return None

    # Unrecognized-model popups: in the GUI these prompt the user. Headless has
    # no UI, so signal "unknown model" by returning None (=> model_status False).
    def pop_up_vr_param(self, _model_hash):
        self.vr_model_params = None

    def pop_up_mdx_model(self, _model_hash, _model_path):
        self.mdx_model_params = None

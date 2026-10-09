#!/usr/bin/env python3
"""Real-model smoke tests for the UVR inference backends.

This script intentionally avoids importing UVR.py because that module starts the
tkinter application. It drives the separator classes directly with a small
ModelData stand-in and writes JSONL results for later comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_ROOT = REPO_ROOT / ".research-artifacts" / "smoke"

MODELS_DIR = REPO_ROOT / "models"
VR_MODELS_DIR = MODELS_DIR / "VR_Models"
MDX_MODELS_DIR = MODELS_DIR / "MDX_Net_Models"
DEMUCS_MODELS_DIR = MODELS_DIR / "Demucs_Models"
DEMUCS_NEWER_REPO_DIR = DEMUCS_MODELS_DIR / "v3_v4_repo"
VR_HASH_JSON = VR_MODELS_DIR / "model_data" / "model_data.json"
MDX_HASH_JSON = MDX_MODELS_DIR / "model_data" / "model_data.json"
MDX_C_CONFIG_PATH = MDX_MODELS_DIR / "model_data" / "mdx_c_configs"
VR_PARAM_DIR = REPO_ROOT / "lib_v5" / "vr_network" / "modelparams"
MDX_MIXER_PATH = REPO_ROOT / "lib_v5" / "mixer.ckpt"

NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
VR_MODEL_DATA_LINK = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json"
)
MDX_MODEL_DATA_LINK = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json"
)
MDX23_CONFIG_CHECKS = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"
)


def import_constants():
    from gui_data.constants import (  # noqa: PLC0415
        ALL_STEMS,
        BACKEND_AUTO,
        BACKEND_COREML,
        BACKEND_CPU,
        BACKEND_MPS,
        BASS_STEM,
        DEFAULT,
        DEF_OPT,
        DEMUCS_2_SOURCE,
        DEMUCS_2_SOURCE_MAPPER,
        DEMUCS_4_SOURCE,
        DEMUCS_4_SOURCE_MAPPER,
        DEMUCS_ARCH_TYPE,
        DEMUCS_V1,
        DEMUCS_V2,
        DEMUCS_V4,
        DRUM_STEM,
        FLAC,
        INST_STEM,
        LEAD_VOCAL_STEM,
        MDX_ARCH_TYPE,
        MP3,
        OTHER_STEM,
        PRIMARY_STEM,
        SECONDARY_STEM,
        VOCAL_STEM,
        VR_ARCH_TYPE,
        WAV,
        secondary_stem,
    )

    return {
        "ALL_STEMS": ALL_STEMS,
        "BACKEND_AUTO": BACKEND_AUTO,
        "BACKEND_COREML": BACKEND_COREML,
        "BACKEND_CPU": BACKEND_CPU,
        "BACKEND_MPS": BACKEND_MPS,
        "BASS_STEM": BASS_STEM,
        "DEFAULT": DEFAULT,
        "DEF_OPT": DEF_OPT,
        "DEMUCS_2_SOURCE": DEMUCS_2_SOURCE,
        "DEMUCS_2_SOURCE_MAPPER": DEMUCS_2_SOURCE_MAPPER,
        "DEMUCS_4_SOURCE": DEMUCS_4_SOURCE,
        "DEMUCS_4_SOURCE_MAPPER": DEMUCS_4_SOURCE_MAPPER,
        "DEMUCS_ARCH_TYPE": DEMUCS_ARCH_TYPE,
        "DEMUCS_V1": DEMUCS_V1,
        "DEMUCS_V2": DEMUCS_V2,
        "DEMUCS_V4": DEMUCS_V4,
        "DRUM_STEM": DRUM_STEM,
        "FLAC": FLAC,
        "INST_STEM": INST_STEM,
        "LEAD_VOCAL_STEM": LEAD_VOCAL_STEM,
        "MDX_ARCH_TYPE": MDX_ARCH_TYPE,
        "MP3": MP3,
        "OTHER_STEM": OTHER_STEM,
        "PRIMARY_STEM": PRIMARY_STEM,
        "SECONDARY_STEM": SECONDARY_STEM,
        "VOCAL_STEM": VOCAL_STEM,
        "VR_ARCH_TYPE": VR_ARCH_TYPE,
        "WAV": WAV,
        "secondary_stem": secondary_stem,
    }


C = import_constants()


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    path: Path
    url: str


DOWNLOADS = {
    "vr": DownloadSpec("vr", VR_MODELS_DIR / "1_HP-UVR.pth", NORMAL_REPO + "1_HP-UVR.pth"),
    "mdx": DownloadSpec("mdx", MDX_MODELS_DIR / "UVR_MDXNET_Main.onnx", NORMAL_REPO + "UVR_MDXNET_Main.onnx"),
    "mdxc": DownloadSpec("mdxc", MDX_MODELS_DIR / "MDX23C_D1581.ckpt", NORMAL_REPO + "MDX23C_D1581.ckpt"),
    "mdxc_config": DownloadSpec(
        "mdxc_config",
        MDX_C_CONFIG_PATH / "model_2_stem_061321.yaml",
        MDX23_CONFIG_CHECKS + "model_2_stem_061321.yaml",
    ),
    "demucs_weight": DownloadSpec(
        "demucs_weight",
        DEMUCS_NEWER_REPO_DIR / "955717e8-8726e21a.th",
        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    ),
    "demucs_yaml": DownloadSpec(
        "demucs_yaml",
        DEMUCS_NEWER_REPO_DIR / "htdemucs.yaml",
        NORMAL_REPO + "htdemucs.yaml",
    ),
}


class SmokeError(RuntimeError):
    pass


class MissingAsset(SmokeError):
    pass


class ResultWriter:
    def __init__(self, path: Path, append: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a" if append else "w", encoding="utf-8")

    def write(self, event: dict):
        event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
        self.file.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def read_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def remove_appledouble_files(root: Path) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("._*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
    return removed


def read_json_url(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "uvr-smoke/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def model_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        try:
            handle.seek(-10000 * 1024, os.SEEK_END)
        except OSError:
            handle.seek(0)
        digest.update(handle.read())
    return digest.hexdigest()


def resolve_hash_metadata(path: Path, local_json: Path, remote_url: str | None = None) -> tuple[str, dict | None]:
    digest = model_hash(path)
    data = read_json_file(local_json) if local_json.is_file() else {}
    if digest in data:
        return digest, data[digest]

    if remote_url:
        with contextlib.suppress(Exception):
            remote_data = read_json_url(remote_url)
            if digest in remote_data:
                return digest, remote_data[digest]

    return digest, None


def download_file(spec: DownloadSpec, force: bool = False) -> dict:
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    if spec.path.is_file() and not force:
        return {"key": spec.key, "path": str(spec.path), "status": "exists", "bytes": spec.path.stat().st_size}

    tmp_path = spec.path.with_suffix(spec.path.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "uvr-smoke/1.0"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(spec.path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise

    return {
        "key": spec.key,
        "path": str(spec.path),
        "status": "downloaded",
        "bytes": spec.path.stat().st_size,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }


def ensure_assets(download_models: bool, force_download: bool, writer: ResultWriter | None = None):
    downloads = [
        DOWNLOADS["vr"],
        DOWNLOADS["mdx"],
        DOWNLOADS["mdxc"],
        DOWNLOADS["mdxc_config"],
        DOWNLOADS["demucs_weight"],
        DOWNLOADS["demucs_yaml"],
    ]
    for spec in downloads:
        if download_models:
            event = download_file(spec, force=force_download)
            if writer:
                writer.write({"event": "download", **event})
        elif not spec.path.is_file():
            raise MissingAsset(f"Missing {spec.path}. Run: tools/smoke_inference.py prepare")


def configure_external_binaries() -> dict:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        os.environ["PATH"] = str(Path(ffmpeg_path).parent) + os.pathsep + os.environ.get("PATH", "")
        with contextlib.suppress(Exception):
            import pydub  # noqa: PLC0415

            pydub.AudioSegment.converter = ffmpeg_path
            pydub.AudioSegment.ffmpeg = ffmpeg_path
    return {"ffmpeg_path": ffmpeg_path}


def generate_input(
    path: Path,
    duration: float = 8.0,
    sample_rate: int = 44100,
    channels: int = 2,
    save_format: str | None = None,
) -> dict:
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(duration * sample_rate)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    envelope = np.linspace(0.2, 0.9, samples, dtype=np.float32)
    left = 0.18 * np.sin(2 * np.pi * 220.0 * t) + 0.06 * np.sin(2 * np.pi * 880.0 * t)
    right = 0.16 * np.sin(2 * np.pi * 330.0 * t) + 0.05 * np.sin(2 * np.pi * 660.0 * t)
    if channels == 1:
        audio = (left * envelope).astype(np.float32)
    else:
        audio = np.stack([left * envelope, right * envelope], axis=1).astype(np.float32)

    fmt = (save_format or path.suffix.lstrip(".") or "wav").lower()
    if fmt == "mp3":
        import pydub  # noqa: PLC0415

        configure_external_binaries()
        wav_buffer = path.with_suffix(".tmp.wav")
        sf.write(wav_buffer, audio, sample_rate, subtype="PCM_16")
        try:
            pydub.AudioSegment.from_wav(wav_buffer).export(path, format="mp3", bitrate="192k")
        finally:
            with contextlib.suppress(FileNotFoundError):
                wav_buffer.unlink()
    else:
        sf.write(path, audio, sample_rate, subtype="PCM_16", format=fmt.upper())
    return {
        "path": str(path),
        "duration_sec": duration,
        "sample_rate": sample_rate,
        "samples": samples,
        "channels": channels,
        "format": fmt.upper(),
    }


class SmokeModelData:
    """Small stand-in for UVR.ModelData used by separate.py."""

    def __init__(
        self,
        key: str,
        backend_mode: str,
        save_format: str,
        is_secondary_model: bool = False,
        primary_model_primary_stem: str | None = None,
        is_vocal_split_model: bool = False,
        gpu_enabled: bool = True,
        demucs_version: str | None = None,
    ):
        from lib_v5.vr_network.model_param_init import ModelParameters  # noqa: PLC0415
        from ml_collections import ConfigDict  # noqa: PLC0415
        import yaml  # noqa: PLC0415

        secondary_stem = C["secondary_stem"]

        self.key = key
        self.DENOISER_MODEL = str(VR_MODELS_DIR / "UVR-DeNoise-Lite.pth")
        self.DEVERBER_MODEL = str(VR_MODELS_DIR / "UVR-DeEcho-DeReverb.pth")
        self.is_deverb_vocals = False
        self.deverb_vocal_opt = C["VOCAL_STEM"]
        self.is_denoise_model = False
        self.is_gpu_conversion = 0 if gpu_enabled else -1
        self.backend_mode = backend_mode
        self.is_normalization = False
        self.is_use_opencl = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_denoise = False
        self.is_mdx_c_seg_def = True
        self.mdx_batch_size = 1
        self.mdxnet_stem_select = C["VOCAL_STEM"]
        self.overlap = 0.25
        self.overlap_mdx = 0.25
        self.overlap_mdx23 = 8
        self.semitone_shift = 0.0
        self.is_pitch_change = False
        self.is_match_frequency_pitch = True
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = True
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = 1.0
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = "PCM_16"
        self.device_set = C["DEFAULT"]
        self.mp3_bit_set = "320k"
        self.save_format = save_format
        self.is_invert_spec = False
        self.is_mixer_mode = False
        self.demucs_stems = C["ALL_STEMS"]
        self.is_demucs_combine_stems = True
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = str(MDX_MIXER_PATH)
        self.process_method = None
        self.model_status = True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = 0.9
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = False
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_save_vocal_only = False
        self.is_primary_model_primary_stem_only = False
        self.is_primary_model_secondary_stem_only = False

        if key == "vr":
            self.process_method = C["VR_ARCH_TYPE"]
            self.model_path = str(DOWNLOADS["vr"].path)
            digest, metadata = resolve_hash_metadata(Path(self.model_path), VR_HASH_JSON, VR_MODEL_DATA_LINK)
            if not metadata:
                raise MissingAsset(f"VR metadata not found for {self.model_path} hash {digest}.")
            param_path = VR_PARAM_DIR / f"{metadata['vr_model_param']}.json"
            self.model_name = Path(self.model_path).stem
            self.model_basename = Path(self.model_path).stem
            self.primary_stem = metadata["primary_stem"]
            self.primary_stem_native = self.primary_stem
            self.secondary_stem = secondary_stem(self.primary_stem)
            self.vr_model_param = ModelParameters(str(param_path))
            self.model_samplerate = self.vr_model_param.param["sr"]
            self.is_vr_51_model = "nout" in metadata and "nout_lstm" in metadata
            if self.is_vr_51_model:
                self.model_capacity = metadata["nout"], metadata["nout_lstm"]
            self.is_karaoke = bool(metadata.get("is_karaoke", False))
            self.is_bv_model = bool(metadata.get("is_bv_model", False))
            self.bv_model_rebalance = metadata.get("is_bv_model_rebalanced", 0)
            self.aggression_setting = 0.05
            self.is_tta = False
            self.is_post_process = False
            self.window_size = 512
            self.batch_size = 1
            self.crop_size = 256
            self.is_high_end_process = "none"
            self.post_process_threshold = 0.2

        elif key == "mdx":
            self.process_method = C["MDX_ARCH_TYPE"]
            self.model_path = str(DOWNLOADS["mdx"].path)
            digest, metadata = resolve_hash_metadata(Path(self.model_path), MDX_HASH_JSON, MDX_MODEL_DATA_LINK)
            if not metadata:
                raise MissingAsset(f"MDX metadata not found for {self.model_path} hash {digest}.")
            self.model_name = Path(self.model_path).stem
            self.model_basename = Path(self.model_path).stem
            self.compensate = metadata["compensate"]
            self.mdx_dim_f_set = metadata["mdx_dim_f_set"]
            self.mdx_dim_t_set = metadata["mdx_dim_t_set"]
            self.mdx_n_fft_scale_set = metadata["mdx_n_fft_scale_set"]
            self.mdx_segment_size = 256
            self.chunks = 0
            self.margin = 44100
            self.primary_stem = metadata["primary_stem"]
            self.primary_stem_native = self.primary_stem
            self.secondary_stem = secondary_stem(self.primary_stem)

        elif key == "mdxc":
            self.process_method = C["MDX_ARCH_TYPE"]
            self.model_path = str(DOWNLOADS["mdxc"].path)
            _, metadata = resolve_hash_metadata(Path(self.model_path), MDX_HASH_JSON, MDX_MODEL_DATA_LINK)
            if not metadata:
                metadata = {"config_yaml": "model_2_stem_061321.yaml"}
            config_path = MDX_C_CONFIG_PATH / metadata["config_yaml"]
            with config_path.open("r", encoding="utf-8") as handle:
                config = ConfigDict(yaml.load(handle, Loader=yaml.FullLoader))
            self.is_mdx_ckpt = True
            self.is_mdx_c = True
            self.mdx_c_configs = config
            self.model_name = Path(self.model_path).name
            self.model_basename = Path(self.model_path).stem
            self.mdx_segment_size = int(config.inference.dim_t)
            self.chunks = 0
            self.margin = 44100
            self.mdx_n_fft_scale_set = int(config.audio.n_fft)
            if config.training.target_instrument:
                self.mdx_model_stems = [config.training.target_instrument]
                self.primary_stem = config.training.target_instrument
            else:
                self.mdx_model_stems = list(config.training.instruments)
                self.primary_stem = self.mdxnet_stem_select
            self.mdx_stem_count = len(self.mdx_model_stems)
            self.primary_stem_native = self.primary_stem
            self.secondary_stem = secondary_stem(self.primary_stem)

        elif key == "demucs":
            self.process_method = C["DEMUCS_ARCH_TYPE"]
            self.model_path = str(DOWNLOADS["demucs_yaml"].path)
            self.model_name = "v4 | htdemucs"
            self.model_basename = Path(self.model_path).stem
            self.demucs_version = demucs_version or C["DEMUCS_V4"]
            if self.demucs_version in {C["DEMUCS_V1"], C["DEMUCS_V2"]}:
                self.demucs_source_list = C["DEMUCS_2_SOURCE"]
                self.demucs_source_map = C["DEMUCS_2_SOURCE_MAPPER"]
                self.demucs_stem_count = 2
            else:
                self.demucs_source_list = C["DEMUCS_4_SOURCE"]
                self.demucs_source_map = C["DEMUCS_4_SOURCE_MAPPER"]
                self.demucs_stem_count = 4
            self.primary_stem = C["PRIMARY_STEM"]
            self.secondary_stem = C["SECONDARY_STEM"]
            self.shifts = 0
            self.is_split_mode = True
            self.segment = C["DEF_OPT"]
            self.is_chunk_demucs = False
        else:
            raise ValueError(f"Unknown model key: {key}")

        if self.is_vocal_split_model:
            primary = C["LEAD_VOCAL_STEM"] if self.primary_stem_native == C["VOCAL_STEM"] else "backing_only"
            self.primary_stem = primary
            self.secondary_stem = secondary_stem(primary)


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    model_key: str
    backend_mode: str
    save_format: str
    expected_stems: tuple[str, ...]
    secondary_key: str | None = None
    vocal_splitter_key: str | None = None
    gpu_enabled: bool = True
    expected_backend_label: str | None = None
    expected_runner_contains: str | None = None
    demucs_version: str | None = None
    duration: float | None = None
    input_format: str = "wav"
    input_channels: int = 2


def blocking_cases() -> list[SmokeCase]:
    return [
        SmokeCase("vr_auto", "vr", C["BACKEND_AUTO"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"])),
        SmokeCase("vr_cpu", "vr", C["BACKEND_CPU"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"])),
        SmokeCase("mdx_auto", "mdx", C["BACKEND_AUTO"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"])),
        SmokeCase("mdx_cpu", "mdx", C["BACKEND_CPU"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"])),
        SmokeCase("mdx_coreml", "mdx", C["BACKEND_COREML"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"])),
        SmokeCase("mdxc_auto", "mdxc", C["BACKEND_AUTO"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"])),
        SmokeCase("mdxc_cpu", "mdxc", C["BACKEND_CPU"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"])),
        SmokeCase(
            "demucs_auto",
            "demucs",
            C["BACKEND_AUTO"],
            C["WAV"],
            (C["BASS_STEM"], C["DRUM_STEM"], C["OTHER_STEM"], C["VOCAL_STEM"]),
        ),
    ]


def backend_mode_cases() -> list[SmokeCase]:
    return [
        SmokeCase("vr_mps", "vr", C["BACKEND_MPS"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"]), expected_backend_label=C["BACKEND_MPS"]),
        SmokeCase(
            "mdx_mps",
            "mdx",
            C["BACKEND_MPS"],
            C["WAV"],
            (C["VOCAL_STEM"], C["INST_STEM"]),
            expected_backend_label=C["BACKEND_MPS"],
            expected_runner_contains="ONNX converted to PyTorch",
        ),
        SmokeCase("mdxc_mps", "mdxc", C["BACKEND_MPS"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"]), expected_backend_label=C["BACKEND_MPS"]),
        SmokeCase(
            "demucs_mps",
            "demucs",
            C["BACKEND_MPS"],
            C["WAV"],
            (C["BASS_STEM"], C["DRUM_STEM"], C["OTHER_STEM"], C["VOCAL_STEM"]),
            expected_backend_label=C["BACKEND_MPS"],
        ),
        SmokeCase(
            "demucs_cpu",
            "demucs",
            C["BACKEND_CPU"],
            C["WAV"],
            (C["BASS_STEM"], C["DRUM_STEM"], C["OTHER_STEM"], C["VOCAL_STEM"]),
            expected_backend_label=C["BACKEND_CPU"],
        ),
    ]


def gpu_disabled_cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            "gpu_disabled_auto_vr",
            "vr",
            C["BACKEND_AUTO"],
            C["WAV"],
            (C["INST_STEM"], C["VOCAL_STEM"]),
            gpu_enabled=False,
            expected_backend_label=C["BACKEND_CPU"],
        ),
        SmokeCase(
            "gpu_disabled_mps_mdx",
            "mdx",
            C["BACKEND_MPS"],
            C["WAV"],
            (C["VOCAL_STEM"], C["INST_STEM"]),
            gpu_enabled=False,
            expected_backend_label=C["BACKEND_CPU"],
            expected_runner_contains="ONNX Runtime",
        ),
    ]


def demucs_legacy_cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            "demucs_v1_auto_policy",
            "demucs",
            C["BACKEND_AUTO"],
            C["WAV"],
            (C["INST_STEM"], C["VOCAL_STEM"]),
            expected_backend_label=C["BACKEND_CPU"],
            demucs_version=C["DEMUCS_V1"],
        ),
        SmokeCase(
            "demucs_v2_mps_policy",
            "demucs",
            C["BACKEND_MPS"],
            C["WAV"],
            (C["INST_STEM"], C["VOCAL_STEM"]),
            expected_backend_label=C["BACKEND_CPU"],
            demucs_version=C["DEMUCS_V2"],
        ),
    ]


def format_input_cases() -> list[SmokeCase]:
    return [
        SmokeCase("input_flac_vr_auto", "vr", C["BACKEND_AUTO"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"]), input_format="flac"),
        SmokeCase("input_mp3_mdx_cpu", "mdx", C["BACKEND_CPU"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"]), input_format="mp3"),
        SmokeCase("input_mono_vr_cpu", "vr", C["BACKEND_CPU"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"]), input_channels=1),
    ]


def long_audio_cases() -> list[SmokeCase]:
    return [
        SmokeCase("long_60s_mdx_mps", "mdx", C["BACKEND_MPS"], C["WAV"], (C["VOCAL_STEM"], C["INST_STEM"]), duration=60.0),
        SmokeCase("long_180s_vr_cpu", "vr", C["BACKEND_CPU"], C["WAV"], (C["INST_STEM"], C["VOCAL_STEM"]), duration=180.0),
    ]


def extended_cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            "secondary_vr_mdx",
            "vr",
            C["BACKEND_AUTO"],
            C["WAV"],
            (C["INST_STEM"], C["VOCAL_STEM"]),
            secondary_key="mdx",
        ),
        SmokeCase(
            "vocal_split_mdx_vr",
            "mdx",
            C["BACKEND_AUTO"],
            C["WAV"],
            (C["VOCAL_STEM"], C["INST_STEM"]),
            vocal_splitter_key="vr",
        ),
        SmokeCase("format_flac_vr_cpu", "vr", C["BACKEND_CPU"], C["FLAC"], (C["INST_STEM"], C["VOCAL_STEM"])),
        SmokeCase("format_mp3_vr_cpu", "vr", C["BACKEND_CPU"], C["MP3"], (C["INST_STEM"], C["VOCAL_STEM"])),
    ]


def make_process_data(input_path: Path, audio_base: str, export_path: Path, model_names: list[str]) -> tuple[dict, list[str]]:
    console: list[str] = []
    cache: dict[tuple[str, str | None], object] = {}
    progress = {"value": 0, "calls": 0}

    def set_progress_bar(base, value=None):
        progress["calls"] += 1
        progress["value"] = value if value is not None else base

    def write_to_console(text, base_text=None):
        console.append(str(text))

    def cached_source_callback(process_method, model_name=None):
        key = (process_method, model_name)
        if key in cache:
            return model_name, cache[key]
        return None, None

    def cached_model_source_holder(process_method, sources, model_name=None):
        cache[(process_method, model_name)] = sources

    iteration = {"count": 0}

    def process_iteration():
        iteration["count"] += 1

    return (
        {
            "audio_file": str(input_path),
            "audio_file_base": audio_base,
            "export_path": str(export_path),
            "cached_source_callback": cached_source_callback,
            "cached_model_source_holder": cached_model_source_holder,
            "is_4_stem_ensemble": False,
            "list_all_models": model_names,
            "process_iteration": process_iteration,
            "set_progress_bar": set_progress_bar,
            "write_to_console": write_to_console,
            "is_ensemble_master": False,
        },
        console,
    )


def build_separator(model_data: SmokeModelData, process_data: dict):
    from separate import SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR  # noqa: PLC0415

    if model_data.process_method == C["VR_ARCH_TYPE"]:
        return SeperateVR(model_data, process_data)
    if model_data.process_method == C["MDX_ARCH_TYPE"] and model_data.is_mdx_c:
        return SeperateMDXC(model_data, process_data)
    if model_data.process_method == C["MDX_ARCH_TYPE"]:
        return SeperateMDX(model_data, process_data)
    if model_data.process_method == C["DEMUCS_ARCH_TYPE"]:
        return SeperateDemucs(model_data, process_data)
    raise ValueError(f"Unsupported process method: {model_data.process_method}")


def output_extension(save_format: str) -> str:
    if save_format == C["FLAC"]:
        return ".flac"
    if save_format == C["MP3"]:
        return ".mp3"
    return ".wav"


def case_input_path(default_input: Path, case: SmokeCase, artifact_dir: Path) -> Path:
    if case.duration is None and case.input_format == "wav" and case.input_channels == 2:
        return default_input
    suffix = "." + case.input_format.lower()
    name = case.case_id + suffix
    return artifact_dir / "input" / name


def validate_audio(path: Path) -> dict:
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    if not path.is_file():
        raise AssertionError(f"Missing output {path}")
    data, sample_rate = sf.read(path, always_2d=True)
    if sample_rate != 44100:
        raise AssertionError(f"{path} sample rate is {sample_rate}, expected 44100")
    if data.shape[1] != 2:
        raise AssertionError(f"{path} channel count is {data.shape[1]}, expected stereo")
    if not np.isfinite(data).all():
        raise AssertionError(f"{path} contains NaN or inf")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-8:
        raise AssertionError(f"{path} is silent")
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "shape": list(data.shape),
        "peak": peak,
        "duration_sec": round(data.shape[0] / sample_rate, 3),
    }


def parse_model_load_sec(console_text: str) -> float | None:
    match = re.search(r"Model load:\s*([0-9.]+)s", console_text)
    return float(match.group(1)) if match else None


def run_backend_policy_case(case: SmokeCase, writer: ResultWriter) -> bool:
    from inference_backend import plan_backend  # noqa: PLC0415

    model = SmokeModelData(
        case.model_key,
        case.backend_mode,
        case.save_format,
        gpu_enabled=case.gpu_enabled,
        demucs_version=case.demucs_version,
    )
    plan = plan_backend(
        model.backend_mode,
        model.is_gpu_conversion >= 0,
        model.device_set,
        process_method=model.process_method,
        demucs_version=getattr(model, "demucs_version", None),
    )
    result = {
        "event": "case",
        "case_id": case.case_id,
        "model_key": case.model_key,
        "backend_mode": case.backend_mode,
        "save_format": case.save_format,
        "gpu_enabled": case.gpu_enabled,
        "demucs_version": case.demucs_version,
        "status": "failed",
        "backend_label": plan.label,
        "runner_label": None,
        "fallback": plan.fallback_to_cpu,
        "model_load_sec": None,
        "elapsed_sec": 0,
        "outputs": [],
        "console_tail": f"Backend policy: {plan.label}",
    }
    try:
        if case.expected_backend_label and plan.label != case.expected_backend_label:
            raise AssertionError(f"Expected backend {case.expected_backend_label}, got {plan.label}")
        result["status"] = "passed"
        writer.write(result)
        return True
    except Exception as exc:
        result.update({"exception": repr(exc), "traceback": traceback.format_exc()})
        writer.write(result)
        return False


def run_case(case: SmokeCase, input_path: Path, artifact_dir: Path, writer: ResultWriter) -> bool:
    from inference_backend import clear_backend_cache  # noqa: PLC0415

    if case.demucs_version in {C["DEMUCS_V1"], C["DEMUCS_V2"]}:
        return run_backend_policy_case(case, writer)

    export_dir = artifact_dir / "outputs" / case.case_id
    export_dir.mkdir(parents=True, exist_ok=True)
    audio_base = case.case_id

    for old in export_dir.glob(f"{audio_base}_*"):
        if old.is_file():
            old.unlink()

    model = SmokeModelData(
        case.model_key,
        case.backend_mode,
        case.save_format,
        gpu_enabled=case.gpu_enabled,
        demucs_version=case.demucs_version,
    )
    model_names = [model.model_basename]

    if case.secondary_key:
        secondary = SmokeModelData(
            case.secondary_key,
            case.backend_mode,
            case.save_format,
            is_secondary_model=True,
            primary_model_primary_stem=model.primary_stem,
            gpu_enabled=case.gpu_enabled,
        )
        model.secondary_model = secondary
        model.secondary_model_scale = 0.9
        model.is_secondary_model_activated = True
        model_names.append(secondary.model_basename)

    if case.vocal_splitter_key:
        splitter = SmokeModelData(
            case.vocal_splitter_key,
            case.backend_mode,
            case.save_format,
            is_vocal_split_model=True,
            gpu_enabled=case.gpu_enabled,
        )
        model.vocal_split_model = splitter
        model.is_vocal_split_model_activated = True
        model_names.append(splitter.model_basename)

    process_data, console = make_process_data(input_path, audio_base, export_dir, model_names)
    result = {
        "event": "case",
        "case_id": case.case_id,
        "model_key": case.model_key,
        "backend_mode": case.backend_mode,
        "save_format": case.save_format,
        "gpu_enabled": case.gpu_enabled,
        "input_path": str(input_path),
        "input_format": case.input_format,
        "input_channels": case.input_channels,
        "status": "failed",
        "backend_label": None,
        "runner_label": None,
        "fallback": None,
        "model_load_sec": None,
        "elapsed_sec": None,
        "outputs": [],
        "console_tail": "",
    }

    started = time.perf_counter()
    try:
        separator = build_separator(model, process_data)
        separator.seperate()
        elapsed = time.perf_counter() - started
        console_text = "".join(console)
        model_load_sec = parse_model_load_sec(console_text)
        ext = output_extension(case.save_format)
        outputs = [
            validate_audio(export_dir / f"{audio_base}_({stem}){ext}")
            for stem in case.expected_stems
        ]
        if "Backend:" not in console_text:
            raise AssertionError("Console log did not include backend instrumentation")
        if case.expected_backend_label and separator.backend_plan.label != case.expected_backend_label:
            raise AssertionError(f"Expected backend {case.expected_backend_label}, got {separator.backend_plan.label}")
        if case.expected_runner_contains and case.expected_runner_contains not in str(separator.model_runner_label):
            raise AssertionError(
                f"Expected runner containing {case.expected_runner_contains!r}, got {separator.model_runner_label!r}"
            )

        result.update(
            {
                "status": "passed",
                "elapsed_sec": round(elapsed, 3),
                "model_load_sec": model_load_sec,
                "inference_and_save_sec": round(elapsed - model_load_sec, 3) if model_load_sec is not None else None,
                "backend_label": separator.backend_plan.label,
                "runner_label": separator.model_runner_label,
                "fallback": separator.backend_plan.fallback_to_cpu or "falling back to CPU" in console_text,
                "outputs": outputs,
                "console_tail": console_text[-2000:],
            }
        )
        writer.write(result)
        cleanup_appledouble()
        clear_backend_cache()
        return True
    except Exception as exc:
        result.update(
            {
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "console_tail": "".join(console)[-2000:],
            }
        )
        writer.write(result)
        cleanup_appledouble()
        clear_backend_cache()
        return False


def run_ensemble_smoke(artifact_dir: Path, writer: ResultWriter) -> bool:
    from lib_v5 import spec_utils  # noqa: PLC0415

    source_a = artifact_dir / "outputs" / "vr_cpu" / f"vr_cpu_({C['VOCAL_STEM']}).wav"
    source_b = artifact_dir / "outputs" / "mdx_cpu" / f"mdx_cpu_({C['VOCAL_STEM']}).wav"
    output = artifact_dir / "outputs" / "ensemble" / "ensemble_vocals.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"event": "case", "case_id": "ensemble_vocals", "status": "failed"}
    try:
        if not source_a.is_file() or not source_b.is_file():
            raise MissingAsset("Ensemble smoke requires vr_cpu and mdx_cpu outputs from blocking phase.")
        spec_utils.ensemble_inputs(
            [str(source_a), str(source_b)],
            "Average",
            False,
            "PCM_16",
            str(output),
            is_wave=False,
        )
        result.update({"status": "passed", "outputs": [validate_audio(output)]})
        writer.write(result)
        return True
    except Exception as exc:
        result.update({"exception": repr(exc), "traceback": traceback.format_exc()})
        writer.write(result)
        return False


def summarize_results(results_path: Path, writer: ResultWriter | None = None) -> dict:
    latest: dict[str, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip().startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "case":
                latest[event.get("case_id", "")] = event

    cases = []
    performance_pairs = {}
    for case_id, event in sorted(latest.items()):
        cases.append(
            {
                "case_id": case_id,
                "status": event.get("status"),
                "backend_label": event.get("backend_label"),
                "runner_label": event.get("runner_label"),
                "fallback": event.get("fallback"),
                "elapsed_sec": event.get("elapsed_sec"),
                "model_load_sec": event.get("model_load_sec"),
                "exception": event.get("exception"),
            }
        )

    pair_map = {
        "vr": ("vr_auto", "vr_cpu"),
        "mdx": ("mdx_auto", "mdx_cpu"),
        "mdxc": ("mdxc_auto", "mdxc_cpu"),
        "demucs_v4": ("demucs_auto", "demucs_cpu"),
    }
    for name, (gpu_case, cpu_case) in pair_map.items():
        gpu = latest.get(gpu_case)
        cpu = latest.get(cpu_case)
        if gpu and cpu and gpu.get("elapsed_sec") and cpu.get("elapsed_sec"):
            performance_pairs[name] = {
                "gpu_case": gpu_case,
                "cpu_case": cpu_case,
                "gpu_elapsed_sec": gpu.get("elapsed_sec"),
                "cpu_elapsed_sec": cpu.get("elapsed_sec"),
                "gpu_vs_cpu_ratio": round(float(gpu["elapsed_sec"]) / float(cpu["elapsed_sec"]), 3),
            }

    summary = {
        "event": "summary",
        "case_count": len(cases),
        "passed": sum(1 for case in cases if case["status"] == "passed"),
        "failed": [case for case in cases if case["status"] != "passed"],
        "performance_pairs": performance_pairs,
    }
    if writer:
        writer.write(summary)
    return summary


def probe_backends(writer: ResultWriter | None = None) -> dict:
    from inference_backend import cuda_available, mps_available, plan_backend  # noqa: PLC0415

    probes = []
    for mode in [C["BACKEND_AUTO"], C["BACKEND_MPS"], C["BACKEND_COREML"], C["BACKEND_CPU"]]:
        plan = plan_backend(mode, True)
        probes.append(
            {
                "mode": mode,
                "label": plan.label,
                "torch_backend": plan.torch_backend,
                "torch_device": str(plan.torch_device),
                "onnx_providers": [p[0] if isinstance(p, tuple) else p for p in plan.onnx_providers],
                "prefer_onnx2torch": plan.prefer_onnx2torch,
                "fallback_to_cpu": plan.fallback_to_cpu,
                "supports_stft": plan.supports_stft,
                "supports_complex": plan.supports_complex,
            }
        )

    event = {
        "event": "backend_probe",
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "probes": probes,
    }
    if writer:
        writer.write(event)
    return event


def doctor(writer: ResultWriter | None = None) -> dict:
    report = {
        "event": "doctor",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "imports": {},
        "ffmpeg_path": shutil.which("ffmpeg"),
        "repo_ffmpeg_path": str(REPO_ROOT / "ffmpeg") if (REPO_ROOT / "ffmpeg").exists() else None,
    }
    for module in ["torch", "onnxruntime", "onnx2pytorch", "soundfile", "librosa", "ml_collections"]:
        try:
            imported = __import__(module)
            report["imports"][module] = getattr(imported, "__version__", "ok")
        except Exception as exc:
            report["imports"][module] = f"ERROR: {exc!r}"
    with contextlib.suppress(Exception):
        import torch  # noqa: PLC0415

        report["torch_mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
    with contextlib.suppress(Exception):
        import onnxruntime as ort  # noqa: PLC0415

        report["onnxruntime_providers"] = ort.get_available_providers()
    if writer:
        writer.write(report)
    return report


def selected_cases(phase: str) -> list[SmokeCase]:
    if phase == "blocking":
        return blocking_cases()
    if phase == "extended":
        return extended_cases()
    if phase == "backend_modes":
        return backend_mode_cases()
    if phase == "gpu_disabled":
        return gpu_disabled_cases()
    if phase == "demucs_legacy":
        return demucs_legacy_cases()
    if phase == "format_inputs":
        return format_input_cases()
    if phase == "long_audio":
        return long_audio_cases()
    if phase == "all":
        return (
            blocking_cases()
            + extended_cases()
            + backend_mode_cases()
            + gpu_disabled_cases()
            + demucs_legacy_cases()
            + format_input_cases()
            + long_audio_cases()
        )
    raise ValueError(f"Unknown phase {phase}")


def command_prepare(args, writer: ResultWriter) -> int:
    input_info = generate_input(args.input, args.duration)
    writer.write({"event": "input", **input_info})
    ensure_assets(not args.skip_download, args.force_download, writer)
    return 0


def command_probe(args, writer: ResultWriter) -> int:
    print(json.dumps(probe_backends(writer), indent=2, ensure_ascii=False))
    return 0


def command_doctor(args, writer: ResultWriter) -> int:
    cleanup_appledouble(writer)
    writer.write({"event": "external_binaries", **configure_external_binaries()})
    report = doctor(writer)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    failed = [name for name, value in report["imports"].items() if str(value).startswith("ERROR:")]
    return 1 if failed else 0


def command_run(args, writer: ResultWriter) -> int:
    cleanup_appledouble(writer)
    writer.write({"event": "external_binaries", **configure_external_binaries()})
    ensure_assets(False, False, writer)
    if not args.input.is_file():
        generate_input(args.input, args.duration)
    cases = selected_cases(args.phase)
    if args.case:
        wanted = set(args.case.split(","))
        cases = [case for case in cases if case.case_id in wanted]
    ok = True
    for case in cases:
        input_path = case_input_path(args.input, case, args.artifact_dir)
        if not input_path.is_file():
            generate_input(
                input_path,
                duration=case.duration or args.duration,
                channels=case.input_channels,
                save_format=case.input_format,
            )
        case_ok = run_case(case, input_path, args.artifact_dir, writer)
        ok = ok and case_ok
        if not case_ok and not args.continue_on_error:
            break
    if ok and args.phase in {"extended", "all"}:
        ok = run_ensemble_smoke(args.artifact_dir, writer) and ok
    return 0 if ok else 1


def command_all(args, writer: ResultWriter) -> int:
    prepare_status = command_prepare(args, writer)
    if prepare_status:
        return prepare_status
    doctor_status = command_doctor(args, writer)
    if doctor_status and not args.continue_on_error:
        return doctor_status
    command_probe(args, writer)
    return command_run(args, writer)


def command_summary(args, writer: ResultWriter) -> int:
    summary = summarize_results(args.results, writer)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["failed"] else 1


def cleanup_appledouble(writer: ResultWriter | None = None):
    prefix = Path(sys.prefix).resolve()
    venv_removed = remove_appledouble_files(prefix) if prefix.is_relative_to(REPO_ROOT) else 0
    removed = {
        "artifacts": remove_appledouble_files(ARTIFACT_ROOT),
        "models": remove_appledouble_files(MODELS_DIR),
        "venv": venv_removed,
    }
    if writer and any(removed.values()):
        writer.write({"event": "cleanup_appledouble", **removed})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "doctor", "probe", "run", "all", "summary"])
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--input", type=Path, default=ARTIFACT_ROOT / "input" / "smoke_input.wav")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--results", type=Path, default=ARTIFACT_ROOT / "smoke_results.jsonl")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--phase",
        choices=[
            "blocking",
            "extended",
            "backend_modes",
            "gpu_disabled",
            "demucs_legacy",
            "format_inputs",
            "long_audio",
            "all",
        ],
        default="blocking",
    )
    parser.add_argument("--case", help="Comma-separated case IDs to run.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultWriter(args.results, append=args.append or args.command == "summary")
    try:
        if args.command == "prepare":
            return command_prepare(args, writer)
        if args.command == "doctor":
            return command_doctor(args, writer)
        if args.command == "probe":
            return command_probe(args, writer)
        if args.command == "run":
            return command_run(args, writer)
        if args.command == "all":
            return command_all(args, writer)
        if args.command == "summary":
            return command_summary(args, writer)
        parser.error(f"Unknown command {args.command}")
        return 2
    except (SmokeError, urllib.error.URLError) as exc:
        writer.write({"event": "fatal", "exception": repr(exc), "traceback": traceback.format_exc()})
        print(f"smoke error: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import gc
import os
import platform

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass
from typing import Any

import torch

from gui_data.constants import (
    BACKEND_AUTO,
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_MPS,
    BACKEND_COREML,
    BACKEND_MODE_OPTIONS,
    CUDA_DEVICE,
    DEFAULT,
    DEMUCS_ARCH_TYPE,
    DEMUCS_V3,
    DEMUCS_V4,
)

TORCH_CPU = "cpu"
TORCH_CUDA = "cuda"
TORCH_MPS = "mps"

ONNX_CPU_PROVIDER = "CPUExecutionProvider"
ONNX_CUDA_PROVIDER = "CUDAExecutionProvider"
ONNX_COREML_PROVIDER = "CoreMLExecutionProvider"

OPERATING_SYSTEM = platform.system()
is_macos = OPERATING_SYSTEM == "Darwin"
cuda_available = torch.cuda.is_available()
mps_available = bool(
    is_macos
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)
_mps_stft_supported = None


@dataclass(frozen=True)
class BackendPlan:
    mode: str
    torch_device: Any
    torch_backend: str
    onnx_providers: list
    prefer_onnx2torch: bool
    fallback_to_cpu: bool
    supports_stft: bool
    supports_complex: bool
    label: str

    @property
    def is_mps(self) -> bool:
        return self.torch_backend == TORCH_MPS

    @property
    def is_cuda(self) -> bool:
        return self.torch_backend == TORCH_CUDA

    @property
    def is_cpu(self) -> bool:
        return self.torch_backend == TORCH_CPU


def _normalize_backend_mode(mode: str | None) -> str:
    if mode in BACKEND_MODE_OPTIONS:
        return mode
    return BACKEND_AUTO


def _cuda_device(device_set: str | None):
    if device_set and device_set != DEFAULT:
        return f"{CUDA_DEVICE}:{device_set}"
    return CUDA_DEVICE


def _coreml_options(cache_dir: str | None = None) -> tuple[str, dict]:
    options = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "CPUAndGPU",
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "1",
    }
    if cache_dir:
        options["ModelCacheDirectory"] = cache_dir
    return ONNX_COREML_PROVIDER, options


def plan_backend(
    backend_mode: str | None,
    gpu_enabled: bool,
    device_set: str | None = DEFAULT,
    process_method: str | None = None,
    demucs_version: str | None = None,
    coreml_cache_dir: str | None = None,
) -> BackendPlan:
    mode = _normalize_backend_mode(backend_mode)

    if not gpu_enabled:
        mode = BACKEND_CPU

    if process_method == DEMUCS_ARCH_TYPE and demucs_version not in (DEMUCS_V3, DEMUCS_V4):
        if mode in (BACKEND_AUTO, BACKEND_MPS, BACKEND_COREML):
            mode = BACKEND_CPU

    onnx_providers = [ONNX_CPU_PROVIDER]
    torch_device: Any = torch.device(TORCH_CPU)
    torch_backend = TORCH_CPU
    prefer_onnx2torch = False
    fallback_to_cpu = False
    label = BACKEND_CPU

    if mode == BACKEND_CUDA or (mode == BACKEND_AUTO and cuda_available):
        if cuda_available:
            torch_device = _cuda_device(device_set)
            torch_backend = TORCH_CUDA
            onnx_providers = [ONNX_CUDA_PROVIDER, ONNX_CPU_PROVIDER]
            label = BACKEND_CUDA
        else:
            fallback_to_cpu = True
    elif mode == BACKEND_MPS or (mode == BACKEND_AUTO and mps_available):
        if mps_available:
            torch_device = torch.device(TORCH_MPS)
            torch_backend = TORCH_MPS
            prefer_onnx2torch = True
            label = BACKEND_MPS
        else:
            fallback_to_cpu = True
    elif mode == BACKEND_COREML or (mode == BACKEND_AUTO and is_macos):
        if is_macos:
            onnx_providers = [_coreml_options(coreml_cache_dir), ONNX_CPU_PROVIDER]
            label = BACKEND_COREML if mode == BACKEND_COREML else f"{BACKEND_CPU} / {BACKEND_COREML} ONNX"
        else:
            fallback_to_cpu = True

    if fallback_to_cpu:
        torch_device = torch.device(TORCH_CPU)
        torch_backend = TORCH_CPU
        onnx_providers = [ONNX_CPU_PROVIDER]
        prefer_onnx2torch = False
        label = f"{BACKEND_CPU} (fallback)"

    supports_stft = torch_backend != TORCH_MPS or is_mps_stft_supported()
    supports_complex = torch_backend != TORCH_MPS or supports_stft

    return BackendPlan(
        mode=mode,
        torch_device=torch_device,
        torch_backend=torch_backend,
        onnx_providers=onnx_providers,
        prefer_onnx2torch=prefer_onnx2torch,
        fallback_to_cpu=fallback_to_cpu,
        supports_stft=supports_stft,
        supports_complex=supports_complex,
        label=label,
    )


def clear_backend_cache(plan: BackendPlan | None = None):
    gc.collect()
    if plan and plan.is_mps and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif plan and plan.is_cuda:
        torch.cuda.empty_cache()
    elif plan is None:
        if mps_available and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if cuda_available:
            torch.cuda.empty_cache()


def is_mps_stft_supported() -> bool:
    global _mps_stft_supported
    if _mps_stft_supported is not None:
        return _mps_stft_supported

    if not mps_available:
        _mps_stft_supported = False
        return False
    try:
        device = torch.device(TORCH_MPS)
        wave = torch.zeros(1, 2048, device=device)
        window = torch.hann_window(512, device=device)
        spec = torch.stft(wave, 512, 128, window=window, return_complex=True)
        torch.istft(spec, 512, 128, window=window, length=2048)
        _mps_stft_supported = True
    except Exception:
        _mps_stft_supported = False
    return _mps_stft_supported


def enable_mps_fallback_env():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

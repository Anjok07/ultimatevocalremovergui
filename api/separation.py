"""Run a single-model separation headlessly.

Reuses the unmodified ``ModelData`` (UVR.py) and ``Seperate*`` engines
(separate.py) by injecting a :class:`~api.headless.HeadlessRoot` as ``UVR.root``.
"""
from __future__ import annotations

import importlib
import multiprocessing
import os
import re
import time
from queue import Empty
import threading
from typing import Any

from .headless import HeadlessRoot
from .schemas import JobCancelled, OutputFile, SeparationOptions

# Importing UVR runs heavy module-level code (and pulls torch/onnx). Do it once,
# lazily, behind a lock so concurrent first-calls don't race.
_uvr_lock = threading.Lock()
_uvr = None
_consts = None
_separate = None


def _patch_screeninfo():
    """``gui_data.app_size_values`` calls ``screeninfo.get_monitors()`` at import
    time to size the GUI window. With no display (headless/Docker) that raises
    "No enumerators available", which would abort importing UVR. Patch in a
    default monitor so the import succeeds — the value only affects GUI layout,
    which we never render."""
    import screeninfo

    original = screeninfo.get_monitors

    def safe_get_monitors(*args, **kwargs):
        try:
            monitors = original(*args, **kwargs)
            if monitors:
                return monitors
        except Exception:
            pass
        return [screeninfo.Monitor(x=0, y=0, width=1920, height=1080)]

    # app_size_values does `from screeninfo import get_monitors`, so patch the
    # module attribute BEFORE UVR (and thus that module) is imported below.
    screeninfo.get_monitors = safe_get_monitors


def _limit_memory():
    """Cap inference memory/threads. On a small CPU host, MDX separation of a
    full song otherwise peaks at ~4GB and can OOM-kill the process. The biggest
    win is disabling onnxruntime's CPU memory arena (it grows per-chunk and never
    shrinks); capping threads also trims per-thread working buffers."""
    threads = int(os.environ.get("UVR_NUM_THREADS", "2"))

    import torch

    try:
        torch.set_num_threads(threads)
    except Exception:
        pass

    import onnxruntime as ort

    original_session = ort.InferenceSession

    def lean_session(*args, **kwargs):
        if not kwargs.get("sess_options"):
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False   # don't retain a growing arena
            so.enable_mem_pattern = False
            so.intra_op_num_threads = threads
            kwargs["sess_options"] = so
        return original_session(*args, **kwargs)

    # separate.py calls `ort.InferenceSession(...)` at separation time; patching
    # the module attribute now (before any run) makes it pick up the lean opts.
    ort.InferenceSession = lean_session


def _load_engine():
    global _uvr, _consts, _separate
    if _uvr is not None:
        return _uvr, _consts, _separate
    with _uvr_lock:
        if _uvr is None:
            _patch_screeninfo()
            _consts = importlib.import_module("gui_data.constants")
            _separate = importlib.import_module("separate")
            _uvr = importlib.import_module("UVR")
            _limit_memory()
    return _uvr, _consts, _separate


_ARCH_TO_METHOD = {"vr": "VR_ARCH_TYPE", "mdx": "MDX_ARCH_TYPE", "demucs": "DEMUCS_ARCH_TYPE"}


def gpu_status(separate) -> dict[str, bool]:
    """Report what the loaded engine sees. ``separate`` is the imported module."""
    return {
        "cuda": bool(getattr(separate, "cuda_available", False)),
        "mps": bool(getattr(separate, "mps_available", False)),
    }


_device_info_cache: dict[str, Any] | None = None


def device_info() -> dict[str, Any]:
    """Lightweight GPU probe for the UI — imports torch only (not the full
    engine), cached after first call. Reports availability + the device the
    next 'auto' job would pick + the GPU name when present."""
    global _device_info_cache
    if _device_info_cache is not None:
        return _device_info_cache

    info: dict[str, Any] = {"cuda": False, "mps": False, "name": None, "torch": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if info["cuda"]:
            try:
                info["name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["name"] = "CUDA device"
    except Exception as exc:  # noqa: BLE001 - torch missing/broken => CPU
        info["error"] = str(exc)

    # What an auto job resolves to, honoring the UVR_USE_GPU env override.
    env = os.environ.get("UVR_USE_GPU", "auto").strip().lower()
    forced_off = env in ("0", "false", "no", "off", "cpu")
    forced_on = env in ("1", "true", "yes", "on", "gpu")
    detected = info["cuda"] or info["mps"]
    use_gpu = False if forced_off else True if forced_on else detected
    info["device"] = "cuda" if (use_gpu and info["cuda"]) else "mps" if (use_gpu and info["mps"]) else "cpu"
    info["gpu_available"] = detected
    info["mode"] = env

    # Host spec — readable from inside the container without a host bind
    # (/proc/cpuinfo and /proc/meminfo are not namespaced; the cgroup cap lives
    # under /sys/fs/cgroup).
    info.update(_host_spec())

    _device_info_cache = info
    return info


def _host_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "cpu_model": None, "cpu_cores": None, "cpu_mhz": None,
        "ram_total": None, "ram_limit": None,
    }
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and not spec["cpu_model"]:
                spec["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("cpu mhz") and not spec["cpu_mhz"]:
                spec["cpu_mhz"] = round(float(line.split(":", 1)[1].strip()))
    except Exception:
        pass
    try:
        import psutil

        spec["cpu_cores"] = psutil.cpu_count()
        spec["ram_total"] = psutil.virtual_memory().total
        # Prefer the reported max clock; fall back to the /proc current MHz above.
        freq = psutil.cpu_freq()
        if freq and (freq.max or freq.current):
            spec["cpu_mhz"] = round(freq.max or freq.current)
    except Exception:
        pass
    # Container memory cap (cgroup v2 then v1); ignore if effectively unlimited.
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(p).read().strip()
            if raw and raw != "max":
                limit = int(raw)
                if not spec["ram_total"] or limit < spec["ram_total"]:
                    spec["ram_limit"] = limit
            break
        except Exception:
            continue
    return spec


def _resolve_use_gpu(opts: SeparationOptions, separate) -> bool:
    """Decide whether to run on GPU. Precedence: explicit request option >
    UVR_USE_GPU env > auto-detect. "auto" enables GPU iff the engine sees one.

    The engine itself then picks the concrete device (cuda/mps) — UVR.py's
    ``ModelData`` maps a truthy is_gpu_conversion into the ``is_gpu_conversion
    >= 0`` branch in separate.py, which selects CUDA when ``cuda_available``."""
    available = gpu_status(separate)
    detected = available["cuda"] or available["mps"]

    if opts.use_gpu is not None:
        return bool(opts.use_gpu)

    env = os.environ.get("UVR_USE_GPU", "auto").strip().lower()
    if env in ("0", "false", "no", "off", "cpu"):
        return False
    if env in ("1", "true", "yes", "on", "gpu"):
        return True
    return detected  # "auto"


def _build_overrides(opts: SeparationOptions, use_gpu: bool) -> dict[str, Any]:
    """Translate a request into ``HeadlessRoot`` setting overrides."""
    overrides: dict[str, Any] = {
        # GPU: ModelData reads is_gpu_conversion_var; truthy => engine takes the
        # GPU branch and auto-selects cuda/mps when torch reports one available.
        "is_gpu_conversion": use_gpu,
        "is_normalization": opts.normalization,
        "is_primary_stem_only": opts.primary_stem_only,
        "is_secondary_stem_only": opts.secondary_stem_only,
        "is_primary_stem_only_Demucs": opts.primary_stem_only,
        "is_secondary_stem_only_Demucs": opts.secondary_stem_only,
        "save_format": opts.output_format.value,
        "semitone_shift": str(opts.semitone_shift),
        "denoise_option": "Standard" if opts.denoise else "None",
        # VR
        "aggression_setting": str(int(opts.aggression)),
        "is_tta": opts.tta,
        "window_size": str(int(opts.window_size)),
        "is_post_process": opts.post_process,
        "is_high_end_process": opts.high_end_process,
        # MDX
        "mdx_segment_size": str(int(opts.segment_size)),
        # Demucs
        "shifts": str(int(opts.shifts)),
    }
    if opts.overlap is not None:
        # MDX reads overlap_mdx/overlap_mdx23; Demucs reads overlap.
        overrides["overlap"] = str(opts.overlap)
        overrides["overlap_mdx"] = str(opts.overlap)
        overrides["overlap_mdx23"] = str(opts.overlap)
    if opts.demucs_segment is not None:
        overrides["segment"] = str(int(opts.demucs_segment))
    return overrides


def _settings_summary(opts: SeparationOptions, device: str) -> str:
    """Human-readable list of the effective settings for the job log — only the
    knobs that apply to the chosen architecture, so it reflects what took
    effect."""
    lines = [
        "Settings",
        f"  model:        {opts.model_name} ({opts.arch.value.upper()})",
        f"  device:       {device}",
        f"  output:       {opts.output_format.value}",
        f"  stems:        {'primary only' if opts.primary_stem_only else 'secondary only' if opts.secondary_stem_only else 'both'}",
        f"  normalize:    {opts.normalization}",
        f"  denoise:      {opts.denoise}",
        f"  pitch shift:  {opts.semitone_shift} semitones",
    ]
    if opts.arch.value == "mdx":
        lines += [
            f"  segment size: {opts.segment_size}",
            f"  overlap:      {opts.overlap if opts.overlap is not None else 'default'}",
        ]
    elif opts.arch.value == "vr":
        lines += [
            f"  aggression:   {opts.aggression}",
            f"  window size:  {opts.window_size}",
            f"  TTA:          {opts.tta}",
            f"  post-process: {opts.post_process}",
            f"  high-end:     {opts.high_end_process}",
        ]
    elif opts.arch.value == "demucs":
        lines += [
            f"  shifts:       {opts.shifts}",
            f"  overlap:      {opts.overlap if opts.overlap is not None else 'default'}",
            f"  segment:      {opts.demucs_segment if opts.demucs_segment is not None else 'default'}",
        ]
    return "\n".join(lines) + "\n"


def run_separation(audio_path: str, export_path: str, opts: SeparationOptions, job) -> list[OutputFile]:
    """Separate ``audio_path`` into stems written under ``export_path``.

    ``job`` is the :class:`api.jobs.Job`; we push progress/log onto it.
    Returns the list of produced output files.
    """
    uvr, consts, separate = _load_engine()

    # Resolve GPU usage (auto-detect by default) and inject the headless root so
    # ModelData/engines read our settings.
    use_gpu = _resolve_use_gpu(opts, separate)
    uvr.root = HeadlessRoot(overrides=_build_overrides(opts, use_gpu))

    status = gpu_status(separate)
    device = "cuda" if (use_gpu and status["cuda"]) else "mps" if (use_gpu and status["mps"]) else "cpu"
    input_bytes = os.path.getsize(audio_path) if os.path.isfile(audio_path) else 0
    job.update(
        message=f"Running on {device.upper()}",
        device=device,
        input_bytes=input_bytes,
        started_at=time.time(),  # wall-clock, for the live elapsed timer
    )
    job.append_log(_settings_summary(opts, device) + "\n")
    t0 = time.monotonic()

    # Sample mode: replace the audio with a short clip (~1/3 into the track),
    # mirroring the desktop app's create_sample. Quick preview of a model/setting.
    if opts.sample_mode:
        audio_path = _make_sample(audio_path, opts.sample_seconds, export_path, job)

    # Duration of the audio actually processed (sample-aware) — the basis for the
    # s/audio-min throughput metric (format/bitrate-independent, unlike bytes).
    job.update(audio_seconds=_audio_duration(audio_path))

    method = getattr(consts, _ARCH_TO_METHOD[opts.arch.value])
    model = uvr.ModelData(opts.model_name, selected_process_method=method, is_dry_check=True)

    if not model.model_status:
        raise ValueError(
            f"Model '{opts.model_name}' ({opts.arch.value}) is not installed or its "
            f"parameters are unrecognized. Download it first via /api/models/download."
        )

    os.makedirs(export_path, exist_ok=True)
    audio_file_base = re.sub(r"\.[^.]+$", "", os.path.basename(audio_path))

    def set_progress_bar(step, inference_iterations=0.0):
        job.update(progress=max(0.0, min(0.99, float(step) + float(inference_iterations))))

    def write_to_console(text, base_text=""):
        job.append_log(f"{base_text}{text}")

    def cached_source_callback(_arch_type, model_name=None):
        return None, None

    def cached_model_source_holder(_arch_type, _sources, _basename=None):
        return None

    process_data = {
        "model_data": model,
        "export_path": export_path,
        "audio_file_base": audio_file_base,
        "audio_file": audio_path,
        "set_progress_bar": set_progress_bar,
        "write_to_console": write_to_console,
        "process_iteration": lambda: None,
        "cached_source_callback": cached_source_callback,
        "cached_model_source_holder": cached_model_source_holder,
        "list_all_models": [model.model_basename],
        "is_ensemble_master": False,
        "is_4_stem_ensemble": False,
    }

    before = set(os.listdir(export_path))

    if model.process_method == consts.VR_ARCH_TYPE:
        seperator = separate.SeperateVR(model, process_data)
    elif model.process_method == consts.MDX_ARCH_TYPE:
        seperator = (
            separate.SeperateMDXC(model, process_data)
            if model.is_mdx_c
            else separate.SeperateMDX(model, process_data)
        )
    elif model.process_method == consts.DEMUCS_ARCH_TYPE:
        seperator = separate.SeperateDemucs(model, process_data)
    else:
        raise ValueError(f"Unsupported process method: {model.process_method}")

    job.update(message=f"Separating with {model.model_basename}…")
    seperator.seperate()

    # Timing / throughput — track for the per-host performance panel.
    duration = time.monotonic() - t0
    mb = input_bytes / (1024 * 1024)
    per_mb = duration / mb if mb > 0 else 0.0
    job.update(duration_sec=duration)
    job.append_log(
        f"\nTotal time: {duration:.1f}s on {device.upper()} "
        f"({mb:.1f} MB input, {per_mb:.2f} s/MB)\n"
    )

    # Collect newly produced files (handles WAV/FLAC/MP3 naming uniformly).
    produced = sorted(set(os.listdir(export_path)) - before)
    outputs: list[OutputFile] = []
    stem_re = re.compile(r"_\(([^)]+)\)")
    for fname in produced:
        m = stem_re.search(fname)
        stem = m.group(1) if m else fname
        outputs.append(OutputFile(stem=stem, filename=fname, url=f"/api/jobs/{job.id}/files/{fname}"))
    return outputs


def _audio_duration(path: str) -> float:
    """Audio length in seconds (cheap probe; no full decode)."""
    try:
        import audioread

        with audioread.audio_open(path) as f:
            return float(f.duration)
    except Exception:
        try:
            import soundfile as sf

            info = sf.info(path)
            return info.frames / info.samplerate
        except Exception:
            return 0.0


def _make_sample(audio_path: str, seconds: int, dest_dir: str, job) -> str:
    """Write a `seconds`-long clip from ~1/3 into the track (UVR's create_sample)."""
    import librosa
    import soundfile as sf
    import audioread

    try:
        with audioread.audio_open(audio_path) as f:
            track_length = int(f.duration)
    except Exception:
        y, sr = librosa.load(audio_path, mono=False, sr=44100)
        track_length = int(librosa.get_duration(y=y, sr=sr))

    clip = int(seconds)
    if track_length >= clip:
        offset = track_length // 3
        if offset + clip > track_length:
            offset = max(0, track_length - clip)
    else:
        offset, clip = 0, track_length

    sample = librosa.load(audio_path, offset=offset, duration=clip, mono=False, sr=44100)[0].T
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"_sample_{clip}s.wav")
    sf.write(out, sample, 44100)
    job.append_log(f"Sample mode: {clip}s clip from {offset}s into the track.\n")
    return out


# --- Subprocess isolation ---------------------------------------------------
# Separation runs in a child process so an OOM (or any hard crash) kills only
# that child — the API stays up and the job fails cleanly. The kernel's OOM
# killer targets the fat separation child over the small parent (uvicorn).
#
# The running child is tracked so a cancel request can terminate it.

import threading as _threading  # local alias; module already uses time/os

_proc_lock = _threading.Lock()
_procs: dict[str, "multiprocessing.Process"] = {}
_cancelled: set[str] = set()


def request_cancel(job_id: str) -> bool:
    """Terminate the running separation child for ``job_id``. Returns True if a
    live process was signalled."""
    with _proc_lock:
        _cancelled.add(job_id)
        proc = _procs.get(job_id)
    if proc is not None and proc.is_alive():
        proc.terminate()
        return True
    return False


class _ChildJob:
    """Stand-in for a Job inside the child — forwards updates to the parent over
    a queue instead of mutating shared state. Only the methods/attrs that
    ``run_separation`` touches are implemented."""

    def __init__(self, job_id: str, q):
        self.id = job_id
        self._q = q
        self.status = "running"

    def update(self, **kwargs):
        self._q.put(("update", kwargs))

    def append_log(self, text: str):
        self._q.put(("log", text))


def _separation_child(q, audio_path, export_path, opts_dict, job_id):
    """Child entrypoint (spawned): run the real separation, stream events back."""
    try:
        opts = SeparationOptions(**opts_dict)
        outputs = run_separation(audio_path, export_path, opts, _ChildJob(job_id, q))
        q.put(("done", [o.model_dump() for o in outputs]))
    except BaseException as exc:  # noqa: BLE001 - report anything to the parent
        import traceback

        q.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_separation_isolated(audio_path: str, export_path: str, opts: SeparationOptions, job):
    """Run :func:`run_separation` in a spawned child process, pumping its
    progress/log/result back onto ``job``. Raises on error or OOM."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_separation_child,
        args=(q, audio_path, export_path, opts.model_dump(mode="json"), job.id),
        daemon=True,
    )
    proc.start()
    with _proc_lock:
        _procs[job.id] = proc
        _cancelled.discard(job.id)

    # Sample the child's peak RSS (works even if it's later OOM-killed: we keep
    # the last reading before death). Includes any grandchildren to be safe.
    try:
        import psutil

        ps = psutil.Process(proc.pid)
    except Exception:
        ps = None
    peak_rss = 0

    def _sample_mem():
        nonlocal peak_rss
        if ps is None:
            return
        try:
            rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
            peak_rss = max(peak_rss, rss)
        except Exception:
            pass

    result = None
    error = None
    try:
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except Empty:
                _sample_mem()
                if not proc.is_alive():
                    break  # died without reporting (killed/cancelled)
                continue
            _sample_mem()
            if kind == "log":
                job.append_log(payload)
            elif kind == "update":
                job.update(**payload)
            elif kind == "done":
                result = payload
                break
            elif kind == "error":
                error = payload
                break

        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
    finally:
        if peak_rss:
            job.update(peak_mem_bytes=peak_rss)
            job.append_log(f"Peak memory: {peak_rss / (1024 * 1024):.0f} MB\n")
        with _proc_lock:
            _procs.pop(job.id, None)
            was_cancelled = job.id in _cancelled
            _cancelled.discard(job.id)

    if was_cancelled or proc.exitcode == -15:  # SIGTERM => user cancel
        raise JobCancelled("Cancelled by user")
    if error is not None:
        raise RuntimeError(error)
    if result is None:
        # No done/error and the process is gone => it was killed.
        if proc.exitcode == -9:  # SIGKILL — on this host that means OOM
            raise MemoryError(
                "Separation ran out of memory and was killed. Try a lighter model "
                "(e.g. 'htdemucs' instead of 'htdemucs_ft'), reduce the Demucs "
                "segment, or use a shorter file."
            )
        raise RuntimeError(f"Separation process exited unexpectedly (code {proc.exitcode}).")
    return [OutputFile(**o) for o in result]

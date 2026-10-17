"""FastAPI application exposing UVR separation as a web service."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import registry
from .jobs import store
from .schemas import (
    Arch,
    DownloadRequest,
    InputInfo,
    JobInfo,
    ModelDetail,
    ModelInfo,
    OutputFormat,
    SeparationOptions,
    StatRow,
    StatsInfo,
    StorageInfo,
)
from .separation import run_separation_isolated

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("UVR_DATA_DIR", os.path.join(REPO_ROOT, "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")  # legacy per-job uploads (pre-inputs)
INPUTS_DIR = os.path.join(DATA_DIR, "inputs")  # reusable, persistent input files
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.jsonl")  # append-only perf log
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# Persist job records to the data volume and recover prior history on startup so
# the files list survives restarts/rebuilds.
store.set_persist_dir(JOBS_DIR)
store.load_from_disk()

app = FastAPI(title="Ultimate Vocal Remover API", version="1.0.0")

# Frontend is served from a different origin in dev; allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("UVR_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/system")
def system():
    """Device/GPU info for the UI (cached lightweight torch probe)."""
    from .separation import device_info

    return device_info()


# --- Reusable input files ---------------------------------------------------
# An input is a single audio file under INPUTS_DIR/<id>/<filename>. It persists
# independently of jobs so one upload can feed many separations.

def _input_resolve(input_id: str):
    """Return (path, filename) for an input id, or None."""
    d = os.path.join(INPUTS_DIR, os.path.basename(input_id))
    if not os.path.isdir(d):
        return None
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    if not files:
        return None
    return os.path.join(d, files[0]), files[0]


def _input_info(input_id: str) -> InputInfo | None:
    resolved = _input_resolve(input_id)
    if not resolved:
        return None
    path, filename = resolved
    st = os.stat(path)
    return InputInfo(id=input_id, filename=filename, bytes=st.st_size, created_at=st.st_mtime)


def _save_input(file: UploadFile) -> InputInfo:
    input_id = uuid.uuid4().hex[:12]
    d = os.path.join(INPUTS_DIR, input_id)
    os.makedirs(d, exist_ok=True)
    safe_name = os.path.basename(file.filename or "input")
    with open(os.path.join(d, safe_name), "wb") as out:
        shutil.copyfileobj(file.file, out)
    return _input_info(input_id)  # type: ignore[return-value]


@app.get("/api/inputs", response_model=list[InputInfo])
def list_inputs():
    infos = [_input_info(e.name) for e in os.scandir(INPUTS_DIR) if e.is_dir()]
    return sorted((i for i in infos if i), key=lambda i: i.created_at, reverse=True)


@app.post("/api/inputs", response_model=InputInfo)
async def upload_input(file: UploadFile = File(...)):
    return _save_input(file)


@app.delete("/api/inputs/{input_id}")
def delete_input(input_id: str):
    d = os.path.join(INPUTS_DIR, os.path.basename(input_id))
    if not os.path.isdir(d):
        raise HTTPException(404, "Input not found")
    shutil.rmtree(d, ignore_errors=True)
    return {"deleted": input_id}


@app.get("/api/help")
def help_texts():
    """Param help strings lifted from the desktop app's constants (source of truth)."""
    from gui_data import constants as c

    g = lambda name: getattr(c, name, "")
    return {
        # general
        "output_format": g("FORMAT_SETTING_HELP"),
        "primary_stem_only": g("SAVE_STEM_ONLY_HELP"),
        "secondary_stem_only": g("SAVE_STEM_ONLY_HELP"),
        "normalization": g("IS_NORMALIZATION_HELP"),
        "denoise": g("IS_DENOISE_HELP"),
        "pitch_shift": g("PITCH_SHIFT_HELP"),
        "sample_mode": g("MODEL_SAMPLE_MODE_HELP"),
        # MDX
        "segment_size": g("MDX_SEGMENT_SIZE_HELP"),
        "overlap": g("MDX_OVERLAP_HELP") or g("OVERLAP_HELP"),
        # VR
        "aggression": g("AGGRESSION_SETTING_HELP"),
        "window_size": g("WINDOW_SIZE_HELP"),
        "tta": g("IS_TTA_HELP"),
        "post_process": g("IS_POST_PROCESS_HELP"),
        "high_end_process": g("IS_HIGH_END_PROCESS_HELP"),
        # Demucs
        "shifts": g("SHIFTS_HELP"),
        "demucs_segment": g("SEGMENT_HELP"),
    }


@app.get("/api/models", response_model=list[ModelInfo])
def get_models():
    return registry.list_models()


@app.get("/api/models/detail", response_model=ModelDetail)
def model_detail(arch: Arch, name: str):
    try:
        return registry.model_detail(arch, name)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/models/download", response_model=JobInfo)
def download(req: DownloadRequest):
    job = store.create("download", message=f"Downloading {req.name}…", options=req.model_dump())

    def _task(j):
        registry.download_model(req.arch, req.name, j)
        j.update(message=f"Downloaded {req.name}")

    store.submit(job, _task)
    return job.to_info()


@app.post("/api/separate", response_model=JobInfo)
async def separate(
    file: UploadFile | None = File(None),
    input_id: str | None = Form(None),
    arch: Arch = Form(...),
    model_name: str = Form(...),
    primary_stem_only: bool = Form(False),
    secondary_stem_only: bool = Form(False),
    output_format: OutputFormat = Form(OutputFormat.wav),
    normalization: bool = Form(False),
    denoise: bool = Form(False),
    semitone_shift: float = Form(0.0),
    aggression: int = Form(10),
    tta: bool = Form(False),
    window_size: int = Form(512),
    post_process: bool = Form(False),
    high_end_process: bool = Form(False),
    segment_size: int = Form(256),
    overlap: float | None = Form(None),
    shifts: int = Form(2),
    demucs_segment: int | None = Form(None),
    use_gpu: bool | None = Form(None),
    sample_mode: bool = Form(False),
    sample_seconds: int = Form(15),
):
    opts = SeparationOptions(
        arch=arch,
        model_name=model_name,
        primary_stem_only=primary_stem_only,
        secondary_stem_only=secondary_stem_only,
        output_format=output_format,
        normalization=normalization,
        denoise=denoise,
        semitone_shift=semitone_shift,
        aggression=aggression,
        tta=tta,
        window_size=window_size,
        post_process=post_process,
        high_end_process=high_end_process,
        segment_size=segment_size,
        overlap=overlap,
        shifts=shifts,
        demucs_segment=demucs_segment,
        use_gpu=use_gpu,
        sample_mode=sample_mode,
        sample_seconds=sample_seconds,
    )

    # One job at a time: reject if a separation is already running/queued.
    busy = store.active_separation()
    if busy:
        raise HTTPException(409, f"A separation is already running (job {busy.id}). Wait for it to finish.")

    # Resolve the audio source: reuse a saved input, or save a new upload as a
    # reusable input (so it's retained for further jobs). The input is NOT tied
    # to the job — deleting the job leaves the input in place.
    if input_id:
        resolved = _input_resolve(input_id)
        if not resolved:
            raise HTTPException(404, f"Input '{input_id}' not found")
        audio_path, input_filename = resolved
    elif file is not None:
        info = _save_input(file)
        resolved = _input_resolve(info.id)
        audio_path, input_filename = resolved  # type: ignore[misc]
    else:
        raise HTTPException(400, "Provide either a file upload or an input_id")

    job = store.create(
        "separation",
        input_filename=input_filename,
        options=opts.model_dump(mode="json"),
        message="Queued",
    )

    export_path = os.path.join(JOBS_DIR, job.id)

    def _task(j):
        outputs = run_separation_isolated(audio_path, export_path, opts, j)
        j.update(outputs=outputs, message="Done")

    store.submit(job, _task, on_finish=lambda j: _record_metric(j, opts))
    return job.to_info()


def _record_metric(job, opts: SeparationOptions):
    """Append a performance record for every finished separation (completed,
    failed, or cancelled) — captures device, size, time, and peak memory."""
    if job.status not in ("completed", "failed", "cancelled"):
        return
    rec = {
        "ts": job.updated_at,
        "job_id": job.id,
        "status": job.status,
        "model": opts.model_name,
        "arch": opts.arch.value,
        "device": job.device or "cpu",
        "input_bytes": job.input_bytes,
        "audio_seconds": job.audio_seconds,
        "duration_sec": job.duration_sec,
        "peak_mem_bytes": job.peak_mem_bytes,
        "sample_mode": opts.sample_mode,
    }
    try:
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _dir_size(path: str) -> int:
    total = 0
    if os.path.isdir(path):
        for entry in os.scandir(path):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                pass
    return total


def _upload_files(job_id: str) -> list[str]:
    prefix = f"{job_id}_"
    if not os.path.isdir(UPLOAD_DIR):
        return []
    return [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR)
        if f.startswith(prefix)
    ]


def _tree_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _job_bytes(job_id: str) -> int:
    """Disk used by a job: its output files (minus job.json) + any legacy
    per-job upload. Reusable inputs are shared and accounted separately."""
    out = _dir_size(os.path.join(JOBS_DIR, job_id))
    meta = os.path.join(JOBS_DIR, job_id, "job.json")
    if os.path.isfile(meta):
        out -= os.path.getsize(meta)
    inp = sum(os.path.getsize(f) for f in _upload_files(job_id) if os.path.isfile(f))
    return max(0, out) + inp


def _with_bytes(info: JobInfo) -> JobInfo:
    info.bytes = _job_bytes(info.id)
    return info


@app.get("/api/jobs", response_model=list[JobInfo])
def list_jobs():
    return [_with_bytes(j.to_info()) for j in store.list()]


@app.get("/api/storage", response_model=StorageInfo)
def storage():
    jobs = store.list()
    # Inputs: reusable inputs + any legacy per-job uploads.
    inputs = _tree_size(INPUTS_DIR) + _tree_size(UPLOAD_DIR)
    outputs = 0
    for j in jobs:
        d = _dir_size(os.path.join(JOBS_DIR, j.id))
        meta = os.path.join(JOBS_DIR, j.id, "job.json")
        if os.path.isfile(meta):
            d -= os.path.getsize(meta)
        outputs += max(0, d)
    return StorageInfo(
        total_bytes=inputs + outputs,
        uploads_bytes=inputs,
        outputs_bytes=outputs,
        job_count=len(jobs),
    )


@app.get("/api/stats", response_model=StatsInfo)
def stats():
    """Per-(model, device) processing throughput accumulated on this host."""
    from .separation import device_info

    agg: dict[tuple, dict] = {}
    if os.path.isfile(METRICS_FILE):
        with open(METRICS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                key = (r.get("model", "?"), r.get("arch", "?"), r.get("device", "cpu"))
                a = agg.setdefault(key, {
                    "runs": 0, "completed": 0, "failed": 0, "cancelled": 0,
                    "audio_sec": 0.0, "sec": 0.0, "peaks": [], "last": 0.0,
                })
                status = r.get("status", "completed")
                a["runs"] += 1
                a[status] = a.get(status, 0) + 1
                # Throughput only from full (non-sample) completed runs that have a
                # known audio duration (format-independent, unlike bytes).
                if (
                    status == "completed" and not r.get("sample_mode")
                    and r.get("duration_sec") and r.get("audio_seconds")
                ):
                    a["audio_sec"] += r.get("audio_seconds", 0.0)
                    a["sec"] += r.get("duration_sec", 0.0)
                if r.get("peak_mem_bytes"):
                    a["peaks"].append(r["peak_mem_bytes"])
                a["last"] = max(a["last"], r.get("ts", 0.0))

    mib = 1024 * 1024
    rows = []
    for (model, arch, device), a in agg.items():
        audio_min = a["audio_sec"] / 60
        peaks_mb = [p / mib for p in a["peaks"]]
        rows.append(StatRow(
            model=model, arch=arch, device=device,
            runs=a["runs"], completed=a.get("completed", 0),
            failed=a.get("failed", 0), cancelled=a.get("cancelled", 0),
            total_audio_min=round(audio_min, 1), total_sec=round(a["sec"], 1),
            sec_per_audio_min=round(a["sec"] / audio_min, 1) if audio_min > 0 else 0.0,
            avg_peak_mb=round(sum(peaks_mb) / len(peaks_mb), 0) if peaks_mb else 0.0,
            max_peak_mb=round(max(peaks_mb), 0) if peaks_mb else 0.0,
            last_run=a["last"],
        ))
    rows.sort(key=lambda r: r.last_run, reverse=True)
    return StatsInfo(host_device=device_info().get("device", "cpu"), rows=rows)


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _with_bytes(job.to_info())


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("running", "queued"):
        raise HTTPException(409, "Job is not running")
    from .separation import request_cancel

    if not request_cancel(job_id):
        # No live child (e.g. between stages) — mark it cancelled directly.
        job.update(status="cancelled", message="Cancelled")
        store.persist(job)
    return {"cancelled": job_id}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Remove output dir (includes the persisted job.json) and the input upload.
    shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)
    for f in _upload_files(job_id):
        try:
            os.remove(f)
        except OSError:
            pass
    store.delete(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_job_file(job_id: str, filename: str, request: Request):
    safe = os.path.basename(filename)
    path = os.path.join(JOBS_DIR, job_id, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")

    size = os.path.getsize(path)
    media = mimetypes.guess_type(safe)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    # Honor byte-range requests so the player can stream + seek instead of
    # waiting for the whole file (Starlette's FileResponse ignores Range).
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end:
                raise HTTPException(416, "Requested range not satisfiable")
            length = end - start + 1

            def stream():
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(262144, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                stream(),
                status_code=206,
                media_type=media,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(length),
                },
            )

    # No range: full file, but advertise range support for seeking.
    return FileResponse(path, filename=safe, media_type=media, headers={"Accept-Ranges": "bytes"})


@app.get("/api/jobs/{job_id}/peaks/{filename}")
def job_file_peaks(job_id: str, filename: str, points: int = 800):
    """Precomputed waveform peaks (cached) so the player can draw a waveform
    without downloading the whole file — enabling streaming playback."""
    safe = os.path.basename(filename)
    path = os.path.join(JOBS_DIR, job_id, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    cache = path + ".peaks.json"
    if os.path.isfile(cache):
        try:
            with open(cache) as f:
                return json.load(f)
        except ValueError:
            pass
    result = _compute_peaks(path, points)
    try:
        with open(cache, "w") as f:
            json.dump(result, f)
    except OSError:
        pass
    return result


def _compute_peaks(path: str, n: int) -> dict:
    import numpy as np

    n = max(100, min(4000, n))
    try:
        import soundfile as sf

        info = sf.info(path)
        block = max(1, info.frames // n)
        peaks: list[float] = []
        with sf.SoundFile(path) as f:
            while True:
                data = f.read(block, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break
                peaks.append(round(float(np.max(np.abs(data))), 4) if len(data) else 0.0)
        return {"peaks": peaks, "duration": info.frames / info.samplerate}
    except Exception:
        # MP3 (or anything soundfile can't block-read) via librosa.
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True)
        block = max(1, len(y) // n)
        peaks = [round(float(np.max(np.abs(y[i : i + block]))), 4) for i in range(0, len(y), block)]
        return {"peaks": peaks, "duration": float(len(y)) / sr}


@app.delete("/api/jobs/{job_id}/files/{filename}")
def delete_job_file(job_id: str, filename: str):
    """Delete a single output stem; the job and its other stems remain."""
    safe = os.path.basename(filename)
    path = os.path.join(JOBS_DIR, job_id, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    os.remove(path)
    # Drop the cached waveform peaks too, if any.
    try:
        os.remove(path + ".peaks.json")
    except OSError:
        pass
    job = store.get(job_id)
    if job:
        # Keep the record but mark it deleted, so the UI can show a disabled chip.
        outs = []
        for o in job.to_info().outputs:
            d = o.model_dump()
            if o.filename == safe:
                d["deleted"] = True
            outs.append(d)
        job.update(outputs=outs)
        store.persist(job)
    return {"deleted": safe}


# --- Serve the built web UI (single-process native run) ---------------------
# When web/dist exists, mount it so `uvicorn api.main:app` serves both the API
# and the SPA on one port — no nginx needed. In Docker the api image has no
# web/dist (nginx serves the SPA there), so this is a harmless no-op. Mounted
# last so it never shadows the /api/* routes.

def _web_dist() -> str | None:
    base = getattr(sys, "_MEIPASS", REPO_ROOT)  # PyInstaller bundle, else source
    for cand in (
        os.environ.get("UVR_WEB_DIST"),
        os.path.join(base, "web", "dist"),
        os.path.join(REPO_ROOT, "web", "dist"),
    ):
        if cand and os.path.isdir(cand):
            return cand
    return None


_dist = _web_dist()
if _dist:
    app.mount("/", StaticFiles(directory=_dist, html=True), name="spa")

"""Job store (persisted to disk) + a serial background worker.

Separation on CPU is heavy, so jobs run one-at-a-time on a single worker thread.

Job records are kept in memory for fast access but mirrored to the data volume
(`<persist_dir>/<id>/job.json`) so the history/files list survives API restarts
and rebuilds. On startup the store reloads those records, and also reconstructs
records for any output dirs that predate persistence (so existing files still
show up). For a multi-replica deployment this would move to Redis/DB.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .schemas import JobInfo, JobStatus, OutputFile

# Single worker => separation jobs are serialized. Bump for multi-GPU hosts.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uvr-worker")

_AUDIO_EXTS = (".wav", ".flac", ".mp3")
_STEM_RE = re.compile(r"_\(([^)]+)\)\.[^.]+$")
# Cap the persisted log so a chatty inference run can't bloat job.json. Keep a
# chunk of the HEAD (the settings summary) plus the tail.
_MAX_PERSIST_LOG = 16_000
_PERSIST_LOG_HEAD = 2_000


@dataclass
class Job:
    id: str
    kind: str
    status: str = JobStatus.queued.value
    progress: float = 0.0
    message: str = ""
    log: str = ""
    error: Optional[str] = None
    input_filename: Optional[str] = None
    options: Optional[dict] = None
    outputs: list = field(default_factory=list)  # list[OutputFile]
    device: Optional[str] = None
    input_bytes: int = 0
    audio_seconds: float = 0.0
    started_at: Optional[float] = None
    duration_sec: Optional[float] = None
    peak_mem_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.updated_at = time.time()

    def append_log(self, text: str):
        with self._lock:
            # Mirror the GUI console: \r return-to-line-start is used for
            # progress; collapse it so the stored log stays readable.
            if text.startswith("\r"):
                head, _, _ = self.log.rpartition("\n")
                self.log = (head + "\n" if head else "") + text.lstrip("\r")
            else:
                self.log += text
            self.updated_at = time.time()

    def _outputs_as_models(self) -> list[OutputFile]:
        return [o if isinstance(o, OutputFile) else OutputFile(**o) for o in self.outputs]

    def to_info(self) -> JobInfo:
        # Build manually (not dataclasses.asdict) to avoid deep-copying the Lock.
        with self._lock:
            return JobInfo(
                id=self.id,
                kind=self.kind,
                status=self.status,
                progress=self.progress,
                message=self.message,
                log=self.log,
                error=self.error,
                input_filename=self.input_filename,
                options=self.options,
                outputs=self._outputs_as_models(),
                device=self.device,
                input_bytes=self.input_bytes,
                audio_seconds=self.audio_seconds,
                started_at=self.started_at,
                duration_sec=self.duration_sec,
                peak_mem_bytes=self.peak_mem_bytes,
                created_at=self.created_at,
                updated_at=self.updated_at,
            )

    def to_dict(self) -> dict:
        """JSON-serializable snapshot for persistence (no Lock, bounded log)."""
        with self._lock:
            # Keep the HEAD (settings summary lives there) plus the tail, so the
            # settings survive truncation on long runs.
            if len(self.log) > _MAX_PERSIST_LOG:
                head = self.log[:_PERSIST_LOG_HEAD]
                tail = self.log[-(_MAX_PERSIST_LOG - _PERSIST_LOG_HEAD):]
                log = f"{head}\n…[log truncated]…\n{tail}"
            else:
                log = self.log
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "log": log,
                "error": self.error,
                "input_filename": self.input_filename,
                "options": self.options,
                "outputs": [o.model_dump() for o in self._outputs_as_models()],
                "device": self.device,
                "input_bytes": self.input_bytes,
                "audio_seconds": self.audio_seconds,
                "started_at": self.started_at,
                "duration_sec": self.duration_sec,
                "peak_mem_bytes": self.peak_mem_bytes,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        job = cls(id=d["id"], kind=d.get("kind", "separation"))
        job.status = d.get("status", JobStatus.completed.value)
        job.progress = d.get("progress", 0.0)
        job.message = d.get("message", "")
        job.log = d.get("log", "")
        job.error = d.get("error")
        job.input_filename = d.get("input_filename")
        job.options = d.get("options")
        job.outputs = [OutputFile(**o) for o in d.get("outputs", [])]
        job.device = d.get("device")
        job.input_bytes = d.get("input_bytes", 0)
        job.audio_seconds = d.get("audio_seconds", 0.0)
        job.started_at = d.get("started_at")
        job.duration_sec = d.get("duration_sec")
        job.peak_mem_bytes = d.get("peak_mem_bytes", 0)
        job.created_at = d.get("created_at", time.time())
        job.updated_at = d.get("updated_at", job.created_at)
        return job


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._persist_dir: Optional[str] = None

    def set_persist_dir(self, path: str):
        self._persist_dir = path
        os.makedirs(path, exist_ok=True)

    def persist(self, job: Job):
        """Public wrapper to flush a job to disk (e.g. after a direct cancel)."""
        self._persist(job)

    def _persist(self, job: Job):
        if not self._persist_dir:
            return
        job_dir = os.path.join(self._persist_dir, job.id)
        try:
            os.makedirs(job_dir, exist_ok=True)
            tmp = os.path.join(job_dir, "job.json.tmp")
            with open(tmp, "w") as f:
                json.dump(job.to_dict(), f)
            os.replace(tmp, os.path.join(job_dir, "job.json"))
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    def create(self, kind: str, **fields) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, **fields)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Drop the in-memory record. Caller removes the on-disk files."""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def active_separation(self) -> Optional[Job]:
        """A separation job that is running or queued, if any (one-at-a-time)."""
        active = (JobStatus.running.value, JobStatus.queued.value)
        with self._lock:
            for j in self._jobs.values():
                if j.kind == "separation" and j.status in active:
                    return j
        return None

    def submit(self, job: Job, target: Callable[[Job], None], on_finish: Optional[Callable[[Job], None]] = None):
        """Run ``target(job)`` on the worker pool, tracking status/errors.
        ``on_finish`` runs after the final status is set (any outcome)."""

        def _run():
            from .schemas import JobCancelled

            job.update(status=JobStatus.running.value)
            self._persist(job)
            try:
                target(job)
                if job.status == JobStatus.running.value:
                    job.update(status=JobStatus.completed.value, progress=1.0)
            except JobCancelled:
                job.append_log("\nCancelled by user.\n")
                job.update(status=JobStatus.cancelled.value, message="Cancelled", error=None)
            except Exception as exc:  # noqa: BLE001 - surface any engine error
                import traceback

                job.append_log("\n" + traceback.format_exc())
                job.update(status=JobStatus.failed.value, error=str(exc))
            finally:
                if on_finish:
                    try:
                        on_finish(job)
                    except Exception:  # noqa: BLE001 - metrics are best-effort
                        pass
                self._persist(job)

        _executor.submit(_run)

    # --- startup recovery ---------------------------------------------------
    def load_from_disk(self):
        """Repopulate from <persist_dir>/<id>/. Loads job.json where present,
        otherwise reconstructs a record from the output files in the dir (so
        pre-persistence jobs still appear). Jobs left mid-flight by a restart
        are marked failed."""
        if not self._persist_dir or not os.path.isdir(self._persist_dir):
            return

        loaded: dict[str, Job] = {}
        for entry in os.scandir(self._persist_dir):
            if not entry.is_dir():
                continue
            meta = os.path.join(entry.path, "job.json")
            try:
                if os.path.isfile(meta):
                    with open(meta) as f:
                        job = Job.from_dict(json.load(f))
                    # A job that was running/queued when the process died is not
                    # actually being worked — the worker pool started empty.
                    if job.status in (JobStatus.running.value, JobStatus.queued.value):
                        job.status = JobStatus.failed.value
                        job.error = "Interrupted by API restart"
                else:
                    job = self._reconstruct(entry.name, entry.path)
                    if job is None:
                        continue
                loaded[job.id] = job
            except Exception:  # noqa: BLE001 - skip unreadable job dirs
                continue

        with self._lock:
            for jid, job in loaded.items():
                self._jobs.setdefault(jid, job)

    def _reconstruct(self, job_id: str, job_dir: str) -> Optional[Job]:
        files = sorted(
            f.name for f in os.scandir(job_dir)
            if f.is_file() and f.name.lower().endswith(_AUDIO_EXTS)
        )
        if not files:
            return None

        outputs = []
        for name in files:
            m = _STEM_RE.search(name)
            stem = m.group(1) if m else name
            outputs.append(OutputFile(stem=stem, filename=name,
                                      url=f"/api/jobs/{job_id}/files/{name}"))

        # Best-effort input name: strip the "<jobid>_" prefix and "_(stem).ext".
        first = files[0]
        base = first[len(job_id) + 1:] if first.startswith(job_id + "_") else first
        base = _STEM_RE.sub("", base) or first

        job = Job(id=job_id, kind="separation")
        job.status = JobStatus.completed.value
        job.progress = 1.0
        job.message = "Recovered from disk"
        job.input_filename = base
        job.outputs = outputs
        try:
            job.created_at = job.updated_at = os.path.getmtime(job_dir)
        except OSError:
            pass
        return job


store = JobStore()

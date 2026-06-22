"""Pydantic request/response models for the UVR API."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Arch(str, Enum):
    vr = "vr"
    mdx = "mdx"
    demucs = "demucs"


class OutputFormat(str, Enum):
    wav = "WAV"
    flac = "FLAC"
    mp3 = "MP3"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobCancelled(Exception):
    """Raised when a running separation is cancelled by the user."""


class SeparationOptions(BaseModel):
    """Tunables accepted with a separation request. All optional with sensible
    defaults that mirror the desktop app."""

    model_config = ConfigDict(protected_namespaces=())

    arch: Arch
    model_name: str = Field(..., description="Model basename, e.g. 'UVR-MDX-NET-Inst_HQ_3'")

    primary_stem_only: bool = False
    secondary_stem_only: bool = False
    output_format: OutputFormat = OutputFormat.wav
    normalization: bool = False
    denoise: bool = False
    semitone_shift: float = 0.0
    # None => auto-detect (GPU if available, else CPU); True/False forces it.
    use_gpu: Optional[bool] = None

    # Sample mode: process only a short clip from ~1/3 into the track (quick preview).
    sample_mode: bool = False
    sample_seconds: int = Field(15, ge=5, le=30)

    # VR-specific
    aggression: int = 10
    tta: bool = False
    window_size: int = 512
    post_process: bool = False
    high_end_process: bool = False

    # MDX-specific
    segment_size: int = 256
    overlap: Optional[float] = None  # None => "Default"; also used by Demucs

    # Demucs-specific
    shifts: int = 2
    demucs_segment: Optional[int] = None  # None => "Default"


class ModelInfo(BaseModel):
    arch: Arch
    name: str  # model basename used for separation (the value ModelData resolves)
    download_name: str  # friendly catalog name used by the download endpoint
    filename: str
    installed: bool


class ModelDetail(BaseModel):
    arch: Arch
    name: str
    download_name: str
    filename: str
    installed: bool
    bytes: int = 0
    stems: list[str] = Field(default_factory=list)  # what it outputs
    primary_stem: Optional[str] = None
    secondary_stem: Optional[str] = None
    technical: dict = Field(default_factory=dict)  # model_data internals
    note: Optional[str] = None


class OutputFile(BaseModel):
    stem: str
    filename: str
    url: str
    deleted: bool = False


class JobInfo(BaseModel):
    id: str
    kind: str  # "separation" | "download"
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    log: str = ""
    error: Optional[str] = None
    input_filename: Optional[str] = None
    options: Optional[dict] = None
    outputs: list[OutputFile] = Field(default_factory=list)
    bytes: int = 0  # disk used by this job's input + output files
    device: Optional[str] = None  # cpu / cuda / mps the job ran on
    input_bytes: int = 0  # size of the input audio processed
    audio_seconds: float = 0.0  # duration of audio actually processed (sample-aware)
    started_at: Optional[float] = None  # wall-clock when processing began
    duration_sec: Optional[float] = None  # processing wall-clock time
    peak_mem_bytes: int = 0  # peak RSS of the separation child process
    created_at: float = 0.0
    updated_at: float = 0.0


class StorageInfo(BaseModel):
    total_bytes: int
    uploads_bytes: int
    outputs_bytes: int
    job_count: int


class InputInfo(BaseModel):
    id: str
    filename: str
    bytes: int
    created_at: float


class StatRow(BaseModel):
    """Aggregated processing performance for one (model, device) on this host."""
    model: str
    arch: str
    device: str
    runs: int  # total finished (any outcome)
    completed: int
    failed: int
    cancelled: int
    total_audio_min: float
    total_sec: float
    sec_per_audio_min: float  # avg processing seconds per audio minute (full runs)
    avg_peak_mb: float  # average peak memory across runs that reported it
    max_peak_mb: float  # worst-case peak memory seen
    last_run: float


class StatsInfo(BaseModel):
    host_device: str  # device the host currently resolves to
    rows: list[StatRow]


class DownloadRequest(BaseModel):
    arch: Arch
    name: str = Field(..., description="Display name from the download list")

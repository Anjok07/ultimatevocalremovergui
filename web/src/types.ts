export type Arch = "vr" | "mdx" | "demucs";

export interface SystemInfo {
  cuda: boolean;
  mps: boolean;
  name: string | null;
  torch: string | null;
  device: "cuda" | "mps" | "cpu";
  gpu_available: boolean;
  mode: string; // UVR_USE_GPU: auto | on | off
  // Host spec (no host bind needed — from /proc + cgroup)
  cpu_model?: string | null;
  cpu_cores?: number | null;
  cpu_mhz?: number | null;
  ram_total?: number | null;
  ram_limit?: number | null;
}

export interface ModelInfo {
  arch: Arch;
  name: string; // basename used for separation
  download_name: string; // friendly name used for downloading
  filename: string;
  installed: boolean;
}

export interface ModelDetail {
  arch: Arch;
  name: string;
  download_name: string;
  filename: string;
  installed: boolean;
  bytes: number;
  stems: string[];
  primary_stem?: string | null;
  secondary_stem?: string | null;
  technical: Record<string, unknown>;
  note?: string | null;
}

export type HelpTexts = Record<string, string>;

export interface NowPlaying {
  url: string; // streaming audio URL
  peaksUrl: string; // precomputed waveform peaks JSON
  label: string; // shown in the player (e.g. "song — Vocals")
  filename: string; // for the download
}

export interface OutputFile {
  stem: string;
  filename: string;
  url: string;
  deleted?: boolean;
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobInfo {
  id: string;
  kind: "separation" | "download";
  status: JobStatus;
  progress: number;
  message: string;
  log: string;
  error?: string | null;
  input_filename?: string | null;
  options?: Record<string, unknown> | null;
  outputs: OutputFile[];
  bytes: number;
  device?: string | null;
  input_bytes: number;
  audio_seconds: number;
  started_at?: number | null;
  duration_sec?: number | null;
  peak_mem_bytes: number;
  created_at: number;
  updated_at: number;
}

export interface StatRow {
  model: string;
  arch: string;
  device: string;
  runs: number;
  completed: number;
  failed: number;
  cancelled: number;
  total_audio_min: number;
  total_sec: number;
  sec_per_audio_min: number;
  avg_peak_mb: number;
  max_peak_mb: number;
  last_run: number;
}

export interface StatsInfo {
  host_device: string;
  rows: StatRow[];
}

export interface StorageInfo {
  total_bytes: number;
  uploads_bytes: number;
  outputs_bytes: number;
  job_count: number;
}

export interface InputInfo {
  id: string;
  filename: string;
  bytes: number;
  created_at: number;
}

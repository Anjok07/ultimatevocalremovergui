import type {
  Arch,
  HelpTexts,
  InputInfo,
  JobInfo,
  ModelDetail,
  ModelInfo,
  StatsInfo,
  StorageInfo,
  SystemInfo,
} from "./types";

// Derive the API base from the document's base URL so the app works both at a
// domain root (https://host/ -> https://host/api/) and under a sub-path
// (https://host/uvr/ -> https://host/uvr/api/). Requires the page to be served
// with a trailing slash (Caddy adds it via a redir).
const API_BASE = new URL("api/", document.baseURI).toString();

/** Absolute URL for an API path (e.g. "jobs/abc/files/x.wav"). */
export const apiUrl = (path: string) => API_BASE + path.replace(/^\//, "");

export interface PeaksData {
  peaks: number[];
  duration: number;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getSystem: () => fetch(apiUrl("system")).then(json<SystemInfo>),

  listModels: () => fetch(apiUrl("models")).then(json<ModelInfo[]>),

  getHelp: () => fetch(apiUrl("help")).then(json<HelpTexts>),

  getModelDetail: (arch: Arch, name: string) =>
    fetch(apiUrl(`models/detail?arch=${arch}&name=${encodeURIComponent(name)}`)).then(
      json<ModelDetail>
    ),

  downloadModel: (arch: Arch, name: string) =>
    fetch(apiUrl("models/download"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arch, name }),
    }).then(json<JobInfo>),

  separate: (form: FormData) =>
    fetch(apiUrl("separate"), { method: "POST", body: form }).then(json<JobInfo>),

  listJobs: () => fetch(apiUrl("jobs")).then(json<JobInfo[]>),

  getJob: (id: string) => fetch(apiUrl(`jobs/${id}`)).then(json<JobInfo>),

  deleteJob: (id: string) =>
    fetch(apiUrl(`jobs/${id}`), { method: "DELETE" }).then(json<{ deleted: string }>),

  cancelJob: (id: string) =>
    fetch(apiUrl(`jobs/${id}/cancel`), { method: "POST" }).then(json<{ cancelled: string }>),

  deleteStem: (id: string, filename: string) =>
    fetch(apiUrl(`jobs/${id}/files/${encodeURIComponent(filename)}`), {
      method: "DELETE",
    }).then(json<{ deleted: string }>),

  getStorage: () => fetch(apiUrl("storage")).then(json<StorageInfo>),

  getStats: () => fetch(apiUrl("stats")).then(json<StatsInfo>),

  listInputs: () => fetch(apiUrl("inputs")).then(json<InputInfo[]>),

  uploadInput: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(apiUrl("inputs"), { method: "POST", body: form }).then(json<InputInfo>);
  },

  deleteInput: (id: string) =>
    fetch(apiUrl(`inputs/${id}`), { method: "DELETE" }).then(json<{ deleted: string }>),
};

export function humanBytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
}

# UVR Web Service (API + Web UI)

A headless HTTP API and React web UI wrapped around Ultimate Vocal Remover's
separation engines, plus a Docker Compose deployment.

This reuses the existing separation engines (`separate.py`) and model config
(`ModelData` in `UVR.py`) unchanged — a small "headless root"
(`api/headless.py`) stands in for the Tkinter window so the engines run without
a GUI. See [the architecture notes](#architecture).

## Quick start (Docker Compose)

CPU-only deployment:

```bash
docker compose up --build
```

Then open **http://localhost:8400**.

- The web UI (nginx) is bound to `127.0.0.1:8400` and proxies `/api/*` to the
  API service (which is not published — internal to the compose network).
- Model weights persist in `./models` (mounted into the API container).
- Job inputs/outputs persist in the `uvr-data` named volume.

## GPU acceleration

The app **auto-detects** the compute device at runtime (CUDA → MPS → CPU); the
device is shown as a chip in the UI header and per-job in the job log. There is
**one Dockerfile** — only the build args differ per target, wired by a compose
override. No code changes.

**x86_64 + NVIDIA** (requires NVIDIA driver + the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This builds with `BASE_IMAGE=nvidia/cuda:…cudnn8-runtime`, CUDA torch wheels
(`…/whl/cu121`), and `onnxruntime-gpu`, and reserves the host GPU(s).

**NVIDIA Jetson (arm64 / L4T):**

```bash
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up --build
```

Edit `docker-compose.jetson.yml` first to set `BASE_IMAGE` to the
[l4t-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
tag matching your JetPack/L4T (`cat /etc/nv_tegra_release`). `INSTALL_TORCH=0`
there reuses the CUDA-enabled torch from the L4T base. The MDX (onnxruntime)
path needs a Jetson `onnxruntime-gpu` wheel for GPU; VR/Demucs use torch and
are accelerated by the base torch alone. CPU fallback still works if the GPU
isn't visible.

**Forcing the device:** set `UVR_USE_GPU` (`auto` | `1`/`on` | `0`/`off`) on the
api service, or pass `use_gpu` (bool) per request to `/api/separate`. `GET
/api/system` reports `{cuda, mps, device, gpu_available, torch, name}`.

Non-NVIDIA GPUs (AMD/Intel) aren't supported in this Linux service: DirectML is
Windows-only, and Intel iGPU offload would only cover the MDX path via an
onnxruntime OpenVINO build — not worth it on older integrated graphics. Use CPU
there.

### Serving behind a reverse proxy (subdomain or sub-path)

The SPA is built with a relative base (`base: "./"`) and derives its API base
from `document.baseURI`, so a single build works **both** at a domain root and
under a sub-path. This deployment is served at `https://dabba.princesamuel.me/uvr`
via Caddy:

```caddyfile
redir /uvr /uvr/ 301
handle_path /uvr/* {
    reverse_proxy localhost:8400 {
        flush_interval -1
    }
}
```

`handle_path` strips the `/uvr` prefix so nginx sees `/`, `/assets/*`, `/api/*`.
The `redir` adds the trailing slash so `document.baseURI` ends in `/` (otherwise
the API base would resolve one path level too high). To serve at a subdomain
root instead, just `reverse_proxy` to `localhost:8400` with no prefix stripping —
the same build works unchanged.

Workflow in the UI:
1. Pick an **architecture** (MDX-Net / VR Arch / Demucs) and a **model**.
2. If the model isn't installed (`○`), click the **download** button.
3. Choose an **audio file**, set options, and click **Separate**.
4. Watch progress in the **Jobs** panel and download the resulting stems.

## Run natively (no Docker) — single process, one port

**Easiest:** use the run script — it creates a venv, installs deps, builds the
UI, makes the data folders, sets the env, and launches (opens your browser):

```bash
./run.sh          # Linux / macOS
run.bat           # Windows (double-click or from a terminal)
```

Override any setting via env, e.g. `UVR_PORT=9000 ./run.sh`. Settings live at the
top of the script (host/port, data dir, threads, GPU, open-browser).

Manual steps (what the scripts do): when `web/dist` exists, the FastAPI app
serves both the API **and** the UI, so a native run is one process on one port
(no nginx). `run_web.py` starts the server and opens your browser.

```bash
# 1. build the UI once
cd web && npm ci && npm run build && cd ..

# 2. Python deps (CPU). Use a venv.
python -m venv .venv
# Linux/macOS: source .venv/bin/activate   |   Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
pip install -r requirements.txt -r requirements-api.txt

# 3. launch (opens http://127.0.0.1:8000)
python run_web.py
```

Prerequisites: **ffmpeg** on `PATH` (for MP3/FLAC output) and a C toolchain (the
`diffq` dependency builds a small C extension).

For UI development with hot-reload instead of a build, run the Vite dev server
(`cd web && npm run dev` → `http://localhost:5173`, proxies `/api` → `:8000`)
alongside `uvicorn api.main:app`.

### Windows

The python.org installer **includes tkinter** (which UVR imports), so no extra
system package is needed. Then:

1. Install **ffmpeg** and add it to `PATH` (or drop `ffmpeg.exe` next to the app).
2. Install **Microsoft C++ Build Tools** (Desktop C++ workload) — required to
   build `diffq`.
3. Run the three steps above (`npm run build`, `pip install …`, `python run_web.py`).
   On Windows, `pip install torch==2.2.2 torchvision==0.17.2` (no index URL) is
   the CPU build.

The separation engine isolates each job in a spawned child process; this is
native to Windows and handled by `multiprocessing.freeze_support()` in
`run_web.py`.

### Standalone Windows build (PyInstaller)

`packaging/uvr_web.spec` bundles the launcher, engine, and UI into a
double-clickable app (server + UI that opens in the browser). It uses UVR's
`sys.frozen`/`_MEIPASS` conventions, and `run_web.py` points `UVR_DATA_DIR` next
to the executable so models and job outputs are written to a writable location.

The GitHub Actions workflow `.github/workflows/build-windows.yml` builds it on a
`windows-latest` runner and uploads the artifact (run it via *Actions → Build
Windows → Run workflow*). Building locally:

```powershell
cd web; npm ci; npm run build; cd ..
pip install pyinstaller
pyinstaller packaging/uvr_web.spec      # -> dist/uvr-web/uvr-web.exe
```

Bundling torch/onnxruntime into a frozen app is fiddly; the spec is a working
starting point and may need per-platform hook tweaks.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Liveness check |
| `GET`  | `/api/models` | List catalog models (VR/MDX/Demucs) with `installed` flags |
| `POST` | `/api/models/download` | Download a model by `download_name`; returns a job |
| `POST` | `/api/separate` | Multipart upload + options; returns a separation job |
| `GET`  | `/api/jobs` | List all jobs (newest first) |
| `GET`  | `/api/jobs/{id}` | Job status, progress, log, outputs |
| `GET`  | `/api/jobs/{id}/files/{filename}` | Download an output stem |

Each model has two identifiers: `name` (the basename used for separation) and
`download_name` (the friendly catalog name used to download it). Use `name`
when calling `/api/separate` and `download_name` when calling
`/api/models/download`.

### Example

```bash
# Download a model
curl -X POST localhost:8000/api/models/download \
  -H 'Content-Type: application/json' \
  -d '{"arch":"mdx","name":"UVR-MDX-NET Inst HQ 3"}'

# Separate (after the download job completes)
curl -X POST localhost:8000/api/separate \
  -F file=@song.wav \
  -F arch=mdx \
  -F model_name=UVR-MDX-NET-Inst_HQ_3 \
  -F output_format=WAV

# Poll the job
curl localhost:8000/api/jobs/<job_id>
```

## Architecture

```
Browser ──> web (nginx :8080) ──/api──> api (uvicorn :8000)
                                            │
                                            ├─ api/main.py        FastAPI routes
                                            ├─ api/jobs.py         in-memory job store + 1 worker thread
                                            ├─ api/registry.py     model catalog + download-on-demand
                                            ├─ api/separation.py   builds ModelData + runs Seperate*
                                            └─ api/headless.py     fake Tkinter root (settings provider)
                                                     │
                                            UVR.py ModelData  +  separate.py Seperate{VR,MDX,MDXC,Demucs}
```

Key design points:
- **No fork of the engines.** `api/headless.py` injects a `HeadlessRoot` as
  `UVR.root`, supplying exactly the `*_var.get()` settings and helper methods
  that `ModelData` and the engines read. Defaults mirror the desktop app;
  request fields override them.
- **Jobs are serialized.** Separation is CPU/RAM heavy, so a single worker
  thread processes one job at a time (`ThreadPoolExecutor(max_workers=1)`).
  Progress and the engine's console output are streamed onto the job.
- **Download-on-demand.** Models come from UVR's public repos (catalog in
  `gui_data/model_manual_download.json`); after a download the hash/param
  mappers are refreshed so `ModelData` recognizes the new weights.

## Scope & limitations (first cut)

- **CPU-only**, single-model separation (VR / MDX-Net / Demucs). Ensemble mode
  and the audio tools (time-stretch, pitch, align, match) are not exposed.
- **No authentication** — intended for a trusted/local network or behind a
  separate proxy.
- **In-memory job store** — jobs are lost on API restart (output files persist
  on the volume). A multi-replica setup would need a shared store/queue.

## Resource requirements

CPU separation of a full song peaks around **~2 GB RAM** with the lean defaults
(`api/separation.py` disables onnxruntime's CPU memory arena and caps threads via
`UVR_NUM_THREADS`/`OMP_NUM_THREADS`). The api container is capped at `mem_limit:
4g` in `docker-compose.yml` so a single large job can't trigger a host-wide OOM
that takes down other services — the cgroup contains any kill to this container.

If you hit OOMs (very long tracks, or several models): keep `mdx_segment_size` at
its default `256` for most models (a value matching the model's `dim_t` uses the
light onnxruntime path; a mismatch falls back to a much heavier onnx2pytorch
path), lower `UVR_NUM_THREADS`, or raise `mem_limit` if the host has the RAM.

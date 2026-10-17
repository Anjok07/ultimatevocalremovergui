#!/usr/bin/env python3
"""Launch the UVR web service and open it in the default browser.

A single-process native runner — no Docker, no nginx. The FastAPI app serves
both the API and the built web UI (web/dist) on one port.

    python run_web.py            # http://127.0.0.1:8000 opens automatically

Env:
    UVR_HOST (127.0.0.1)  UVR_PORT (8000)  UVR_OPEN_BROWSER (1)

This is also the PyInstaller entry point. ``multiprocessing.freeze_support()``
is essential: each separation runs in a spawned child process, which in a frozen
build would otherwise re-execute this launcher (spawning servers/browsers).
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import webbrowser


def main() -> None:
    host = os.environ.get("UVR_HOST", "127.0.0.1")
    port = int(os.environ.get("UVR_PORT", "8000"))
    url = f"http://{host}:{port}/"

    # Frozen (PyInstaller) build: keep user data next to the executable rather
    # than inside the read-only bundle, and locate the bundled web UI.
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
        os.environ.setdefault("UVR_DATA_DIR", os.path.join(app_dir, "data"))
        meipass = getattr(sys, "_MEIPASS", app_dir)
        os.environ.setdefault("UVR_WEB_DIST", os.path.join(meipass, "web", "dist"))

    if os.environ.get("UVR_OPEN_BROWSER", "1") != "0":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    import uvicorn
    from api.main import app

    print(f"\n  Ultimate Vocal Remover — web service\n  → {url}   (Ctrl+C to stop)\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Must be the very first thing for spawn/frozen child processes.
    multiprocessing.freeze_support()
    main()

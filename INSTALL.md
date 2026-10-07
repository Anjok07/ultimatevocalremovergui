# Linux Installation & Troubleshooting Guide

> Consolidated from the most common Linux installation failures reported in the issue tracker (error patterns verified against #2107 and #2259).
> If you hit an error that is not covered here, please open an issue and include: your distro, your Python version (`python3 --version`), and the full `pip` output.

## Quick start (Debian / Ubuntu / Mint)

```bash
# 1. Install Python 3.10 (recommended — see "Why Python 3.10?" below)
sudo apt install python3.10 python3.10-venv git

# 2. Get the code
git clone https://github.com/Anjok07/ultimatevocalremovergui.git
cd ultimatevocalremovergui

# 3. Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Upgrade build tooling first (avoids several wheel-build errors)
pip install --upgrade pip wheel

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run
python UVR.py
```

## Why Python 3.10?

Several pins in `requirements.txt` (e.g. `numpy==1.23.5`, `scipy==1.9.3`, `audioread==3.0.0`, `cryptography==3.4.6`) predate Python 3.11+ and have **no wheels** for newer interpreters, so `pip` falls back to building them from source — and those source builds fail on modern toolchains:

| Python | Status | Notes |
|---|---|---|
| 3.8 – 3.10 | ✅ Recommended | Binary wheels exist for all pinned packages |
| 3.11 | ⚠️ Partial | Most packages resolve; occasional manual fixes needed |
| 3.12 | ❌ | `imp` module removed; several pinned packages fail |
| 3.13 / 3.14 | ❌ | e.g. `audioread==3.0.0` is sdist-only; source build fails (`subprocess-exited-with-error`) |

- On **Arch-based** distros, install an older Python side-by-side (e.g. `python310` from the AUR), or use [uv](https://docs.astral.sh/uv/): `uv venv --python 3.10 venv`.
- Always invoke `python3`, not `python` — many distros no longer ship a bare `python` command (PEP 394).

## Common errors and fixes

### `Getting requirements to build wheel ... error: subprocess-exited-with-error` (audioread)
- **Seen with:** Python 3.13/3.14, e.g. Arch (#2259)
- **Cause:** `audioread==3.0.0` is only published as an sdist (pip shows `Using cached audioread-3.0.0.tar.gz`); the source build fails on newer interpreters.
- **Fix:** Use a Python 3.10 venv (see above).

### `ModuleNotFoundError: No module named 'imp'`
- **Seen with:** Python 3.12 (#2107)
- **Cause:** the `imp` module was removed in Python 3.12; older pinned packages still import it.
- **Fix:** Use Python 3.10 or 3.11.

### `error in playsound setup command: use_2to3 is invalid` / playsound wheel-build failure
- **Seen with:** Python 3.11+ (#2107)
- **Cause:** `playsound 1.3.0` uses `use_2to3` in its setup, which setuptools ≥ 58 removed.
- **Fix:** `pip install playsound==1.2.2` (last version without `use_2to3`), or use the maintained drop-in `playsound3` (`pip install playsound3`, imports as `playsound3`).

### `ModuleNotFoundError: No module named 'sklearn'` at runtime / "The 'sklearn' package is deprecated"
- **Cause:** librosa requires `scikit-learn`; installing the deprecated `sklearn` shim package from PyPI instead leads to hard deprecation errors.
- **Fix:** `pip install scikit-learn` (NOT `sklearn`).

### Generic `Preparing metadata ... error` / wheel-build failures
- **Fix:** inside the venv run `pip install --upgrade pip wheel`, then retry the install.

## GPU (CUDA) notes

`requirements.txt` installs the CPU build of PyTorch. For NVIDIA GPU support, install the matching PyTorch build **before** running `pip install -r requirements.txt`, following the official picker: https://pytorch.org/get-started/locally/ — the already-installed torch build then satisfies the `torch` requirement.

## References

The error patterns above were verified against: #2107 (Ubuntu/Mint, Python 3.11/3.12: `imp`, playsound, sklearn) and #2259 (Arch, Python 3.14: audioread sdist build failure). See also the open installation issues on Linux for related reports.
^^

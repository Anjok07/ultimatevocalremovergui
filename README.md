- **DEVELOPMENT UPDATE - October 6th, 2026:**
    - Please see the following issue for the latest UVR development update! - [Issue 2399](https://github.com/Anjok07/ultimatevocalremovergui/issues/2399)

# Ultimate Vocal Remover GUI v5.6
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/gui_data/img/UVR_v5.6.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application uses state-of-the-art source separation models to remove vocals from audio files. UVR's core developers trained all of the models provided in this package (except for the Demucs v3 and v4 4-stem models).

- **Core Developers**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

- **Support the Project**
    - [Donate](https://www.buymeacoffee.com/uvr5)

## Installation

These bundles contain the UVR interface, Python, PyTorch, and other dependencies needed to run the application effectively. No prerequisites are required.

### Windows Installation

- Please Note:
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.
    - You must install UVR to the main C:\ drive. Installing UVR to a secondary drive will cause instability.

- Download the UVR installer for Windows via the link below:
    - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_v5.6.0_setup.exe)
    - [Main Download Link mirror](https://www.mediafire.com/file_premium/jiatpgp0ljou52p/UVR_v5.6.0_setup.exe/file)
- If you use an **AMD Radeon or Intel Arc graphics card**, you can try the DirectML version:
    - [DirectML Version - Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_1_15_25_22_30_BETA_full.exe)
- Update Package instructions for those who have UVR already installed:
    - If you already have UVR installed you can install this package over it or download it straight from the application or [click here for the patch](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_Patch_10_6_23_4_27.exe).

<details id="WindowsManual">
  <summary>Windows Manual Installation</summary>

### Manual Windows Installation

- Download and extract the repository [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
- Download and install Python [here](https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe)
   - Make sure to check "Add python.exe to PATH" during the install
- Run the following commands from the extracted repo directory:

```
python.exe -m pip install -r requirements.txt
```

If you have a compatible Nvidia GPU, run the following command:

```
python.exe -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu117
```

If you do not have FFmpeg or Rubber Band installed and want to avoid going through the process of installing them the long way, follow the instructions below.

**FFmpeg Installation**

- Download the precompiled build [here](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
- From the archive, extract the following file to the UVR application directory:
   - ```ffmpeg-5.1.2-essentials_build/bin/ffmpeg.exe```

**Rubber Band Installation**

In order to use the Time Stretch or Change Pitch tool, you'll need Rubber Band.

- Download the precompiled build [here](https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip)
- From the archive, extract the following files to the UVR application directory:
   - ```rubberband-3.1.2-gpl-executable-windows/rubberband.exe```
   - ```rubberband-3.1.2-gpl-executable-windows/sndfile.dll```

</details>

### MacOS Installation
- Please Note:
    - The MacOS Sonoma mouse clicking issue has been fixed.
    - Apple Silicon GPU acceleration is available through MPS. Some model paths may fall back to CPU when PyTorch or the selected backend cannot run an operation on Apple GPU.
    - This bundle is intended for those running macOS Big Sur and above.
    - Application functionality for systems running macOS Catalina or lower is not guaranteed.
    - Application functionality for older or budget Mac systems is not guaranteed.
    - Once everything is installed, the application may take up to 5-10 minutes to start for the first time (depending on your Macbook).

- Download the UVR dmg for MacOS via one of the links below:
    - Mac M1 (arm64) users:
       - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_arm64.dmg)
       - [Main Download Link mirror](https://www.mediafire.com/file_premium/u3rk54wsqadpy93/Ultimate_Vocal_Remover_v5_6_MacOS_arm64.dmg/file)

    - Mac Intel (x86_64) users:
       - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_x86_64.dmg)
       - [Main Download Link mirror](https://www.mediafire.com/file_premium/2gf1werx5ly5ylz/Ultimate_Vocal_Remover_v5_6_MacOS_x86_64.dmg/file)

<details id="CannotOpen">
  <summary>MacOS Users: Having Trouble Opening UVR?</summary>

> Due to Apples strict application security, you may need to follow these steps to open UVR.
>
> First, run the following command via Terminal.app to allow applications to run from all sources (it's recommended that you re-enable this once UVR opens properly.)
> 
> ```bash
> sudo spctl --master-disable
> ```
> 
> Second, run the following command to bypass Notarization: 
> 
> ```bash
> sudo xattr -rd com.apple.quarantine /Applications/Ultimate\ Vocal\ Remover.app
> ```

</details>

<details id="MacInstall">
  <summary>Manual MacOS Installation</summary>

### Manual MacOS Installation

- Download and save this repository [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
- Download and install Python 3.10 [here](https://www.python.org/ftp/python/3.10.9/python-3.10.9-macos11.pkg)
- From the saved directory run the following - 

```
pip3 install -r requirements.txt
```

- If your Mac is running with an M1, please run the following command next. If not, skip this step. - 

```
cp /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile_arm64.dylib /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile.dylib
```

**FFmpeg Installation**

- Once everything is done installing, download the correct FFmpeg binary for your system [here](http://www.osxexperts.net) and place it into the main application directory.

**Rubber Band Installation**

In order to use the Time Stretch or Change Pitch tool, you'll need Rubber Band.

- Download the precompiled build [here](https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip)
- From the archive, extract the following files to the UVR/lib_v5 application directory:
   - ```rubberband-3.1.2-gpl-executable-macos/rubberband```

This process has been tested on a MacBook Pro 2021 (using M1) and a MacBook Air 2017 and is confirmed to be working on both.

</details>


### Linux Installation (Updated Instructions)

<details id="LinuxInstall">
  <summary>See Linux Installation Instructions</summary>

<br />

**These installation instructions are for Debian & Arch-based Linux systems.**

---

#### **Step 1: Download the Repository**
- Download and save this repository from [GitHub](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip).
- Extract the downloaded file to a directory of your choice.

---

#### **Step 2: Install Dependencies**
Use the following commands based on your system type:

**For Debian-based systems (Ubuntu, Mint, etc.):**
```bash
sudo apt update && sudo apt upgrade
sudo apt-get install -y ffmpeg python3-pip python3-tk
```

**For Arch-based systems (EndeavourOS):**
```bash
sudo pacman -Syu
sudo pacman -S ffmpeg python-pip tk
```

---

#### **Step 3: Set Up a Virtual Environment (Recommended)**
Setting up a virtual environment (venv) ensures that the program's dependencies do not interfere with system-wide Python packages.

1. **Navigate to the extracted repository directory:**
   ```bash
   cd /path/to/ultimatevocalremovergui
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - For **Debian-based and Arch-based systems:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies in the virtual environment:**
   ```bash
   pip install -r requirements.txt
   ```

---

#### **Step 4: Run the Application**
While the virtual environment is activated, start the application:
```bash
python UVR.py
```

---

#### **Important Notes**
1. **Avoid Modifying System Files:**  
   Previous instructions suggested deleting the `/usr/lib/python3.11/EXTERNALLY-MANAGED` file, which is dangerous and can break Python package management. Do **NOT** delete this file.

2. **Why Use Virtual Environments?**  
   Virtual environments isolate the program's dependencies, preventing conflicts with system Python packages. More information is available [here](https://stackoverflow.com/questions/75602063/pip-install-r-requirements-txt-is-failing-this-environment-is-externally-mana/75696359#75696359).

3. **Known Issues and Discussions:**  
   - [Issue #1578](https://github.com/Anjok07/ultimatevocalremovergui/issues/1578)  
   - [Pull Request #1068](https://github.com/Anjok07/ultimatevocalremovergui/pull/1068)

---

If you encounter issues, refer to the [GitHub Issues](https://github.com/Anjok07/ultimatevocalremovergui/issues) page for help. 

</details>

### Other Application Notes
- Nvidia GTX 1060 6GB is the minimum requirement for GPU conversions.
- Nvidia GPUs with at least 8GBs of V-RAM are recommended.
- Apple Silicon Macs can use Apple GPU (MPS) acceleration when running a modern PyTorch build with MPS support.
- AMD Radeon GPU supported is limited at this time.
   - There is currently a working branch for AMD GPU users [here](https://github.com/Anjok07/ultimatevocalremovergui/tree/v5.6-amd-gpu)
- This application is only compatible with 64-bit platforms.
- This application relies on the Rubber Band library for the Time-Stretch and Pitch-Shift options.
- This application relies on FFmpeg to process non-wav audio files.
- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
- These models are computationally intensive. 

### Performance:
- Model load times are faster.
- Importing/exporting audio files is faster.

## Inference Backends

UVR now uses an inference backend selector behind the existing GPU controls. The available backend modes are:

- `Auto`
- `CPU`
- `CUDA`
- `Apple GPU (MPS)`
- `CoreML`

`Auto` prefers CUDA first, then Apple GPU (MPS), then CoreML for ONNX models on macOS, then CPU. On macOS Apple Silicon, the settings UI shows the relevant choices: `Auto`, `Apple GPU (MPS)`, `CoreML`, and `CPU`.

Backend coverage:

- VR and MDXC models run through PyTorch and can use CPU, CUDA, or Apple GPU (MPS).
- MDX ONNX models keep ONNX Runtime CUDA and CPU support. On Apple GPU (MPS), MDX ONNX models are converted through `onnx2pytorch` and run as PyTorch models. `CoreML` is available as an explicit experimental ONNX Runtime backend.
- Demucs v3 and v4 can attempt Apple GPU (MPS). If that path fails, UVR falls back to CPU and records the fallback in the log. Demucs v1 and v2 keep the older CPU/CUDA behavior by default.

The conversion log includes the selected backend, runner, model load time, and fallback status. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by default for compatibility, but the primary MPS paths are expected to work without hiding unsupported device transfers.

## Developer Smoke Tests

The real-model smoke harness lives in `tools/smoke_inference.py`. It intentionally avoids importing `UVR.py` because `UVR.py` starts the tkinter mainloop on import. The harness drives the separator classes directly, writes results to `.research-artifacts/smoke/smoke_results.jsonl`, and keeps generated audio under `.research-artifacts/smoke/`.

Recommended Apple Silicon setup:

```bash
uv python install 3.10
uv venv .venv-smoke --python 3.10
.venv-smoke/bin/python -m ensurepip --upgrade
.venv-smoke/bin/python -m pip install playsound==1.2.2
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True .venv-smoke/bin/python -m pip install -r requirements.txt
.venv-smoke/bin/python -m pip uninstall -y PySoundFile
.venv-smoke/bin/python -m pip install soundfile==0.13.1 onnx2pytorch
```

Run the blocking smoke flow:

```bash
.venv-smoke/bin/python tools/smoke_inference.py prepare
.venv-smoke/bin/python tools/smoke_inference.py doctor --append
.venv-smoke/bin/python tools/smoke_inference.py probe --append
.venv-smoke/bin/python tools/smoke_inference.py run --phase blocking --continue-on-error --append
.venv-smoke/bin/python tools/smoke_inference.py summary --append
```

The blocking phase covers backend probing plus single-model real inference for VR, MDX ONNX, MDX ONNX CoreML, MDXC, and Demucs v4. Additional phases are available for release checks:

- `extended`: secondary model, vocal splitter, ensemble, FLAC output, and MP3 output.
- `backend_modes`: explicit Apple GPU (MPS) cases and Demucs v4 CPU comparison.
- `gpu_disabled`: verifies that disabling GPU forces CPU even when Auto or MPS is selected.
- `demucs_legacy`: verifies Demucs v1/v2 backend policy without requiring legacy weights.
- `format_inputs`: FLAC input, MP3 input, and mono input.
- `long_audio`: 60-second MDX MPS and 180-second VR CPU stability checks.
- `all`: blocking, extended, backend mode, GPU-disabled, Demucs legacy, format-input, and long-audio checks.

Use `summary` after a run to print the latest pass/fail state and CPU-vs-MPS timing pairs.

The `prepare` command downloads the minimal real model set into the app's existing model folders and generates an 8-second stereo WAV input. Do not commit `.venv-smoke`, `.research-artifacts`, downloaded model weights, `__pycache__`, or AppleDouble `._*` files.

### MDX23C Apple GPU Functional Check

A full-length MDX23C functional check was run on Apple Silicon with the packaged app resources:

- Model: `dist/Ultimate Vocal Remover.app/Contents/Resources/models/MDX_Net_Models/MDX23C-8KFFT-InstVoc_HQ.ckpt`
- Input: `example_data/kamiina_ep01.wav`
- Config: `model_2_stem_full_band_8k.yaml`
- Backend mode: `Auto`
- Actual backend: `Apple GPU (MPS)`
- Torch device: `mps`
- Fallback: `false`
- Result: `passed`
- Elapsed time: `2070.068s`

The model hash did not match an entry in the bundled `model_data.json`, so the GUI does not automatically infer its MDX23C config from metadata. For this file, use the 8KFFT config `model_2_stem_full_band_8k.yaml`.

The effective MDX23C parameters for that check were:

- Segment size: `256`
- Segment default: `true`, so `inference.dim_t=256` from the config was used.
- Overlap: `8`
- Batch size: `1`
- `n_fft`: `8192`
- `dim_f`: `4096`
- `hop_length`: `1024`
- Runtime chunk size: `1024 * (256 - 1) = 261120`
- Runtime hop size: `261120 // 8 = 32640`
- Instruments: `Vocals`, `Instrumental`
- Output format: WAV `PCM_16`
- Output normalization: `false`
- Pitch shift: `0`
- Match frequency cutoff: `true`, but it is not used when pitch shift is zero.

The generated verification files live under `.research-artifacts/functional-mdxc-app/` and are intentionally not committed.

## Troubleshooting

### Common Issues

- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.
- Memory allocation errors can usually be resolved by lowering the "Segment" or "Window" sizes.

#### MacOS Sonoma Left-click Bug
There's a known issue on MacOS Sonoma where left-clicks aren't registering correctly within the app. This was impacting all applications built with Tkinter on Sonoma and has since been resolved. Please download the latest version via the following link if you are still experiencing issues - [link](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.6)

This issue was being tracked [here](https://github.com/Anjok07/ultimatevocalremovergui/issues/840).

### Issue Reporting

Please be as detailed as possible when posting a new issue. 

If possible, click the "Settings Button" to the left of the "Start Processing" button and click the "Error Log" button for detailed error information that can be provided to us.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits
- [ZFTurbo](https://github.com/ZFTurbo) - Created & trained the weights for the new MDX23C models. 
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - Designed the official UVR logo, icon, banner, and splash screen.
- [tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original Demucs AI code. 
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. 
- This project is 100% open-source and free for anyone to use and modify as they wish. 
- We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

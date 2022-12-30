# Ultimate Vocal Remover GUI v5.5.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/gui_data/img/UVR_5_5.png?raw=true" />

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
    - You must install UVR to the main C:\ drive if you use the Windows installer. Installing UVR to a secondary drive will cause application instability.
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.

- Download the UVR installer for Windows via the link below:
    - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.5.0/UVR_v5.5.0_setup.exe)
    - [Main Download Link mirror](https://www.mediafire.com/file/ov1a4s91kwrith1/UVR_v5.5.0_setup.exe/file)
- Update Package instructions for those who have UVR already installed:
    - If you already have UVR installed you can install this package over it or download it straight from the application.

### MacOS Installation

- Please Note:
    - Once everything is installed, the application may take up to 5-10 minutes to start for the first time (depending on your Macbook).
    - This .app is intended for those running macOS BigSur and above.
    - Application functionality for systems running macOS Catalina or lower.
    - Application functionality for older or budget Mac systems is not guaranteed.

- Download the UVR dmg for MacOS via one of the links below:
    - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.5.0/Ultimate_Vocal_Remover_v5_5_MacOS.dmg)
    - [Main Download Link mirror](https://www.mediafire.com/file/xai2l5jsxugajgi/Ultimate_Vocal_Remover_v5_5_MacOS.dmg/file)

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
- Download Python 3.10 [here](https://www.python.org/ftp/python/3.10.9/python-3.10.9-macos11.pkg)
- From the saved directory run the following - 

```
pip3 install -r requirements.txt
```

- If your Mac is running with an M1, please run the following command next. If not, skip this step. - 

```
cp /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile_arm64.dylib /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile.dylib
```

- Once everything is done installing, download and unzip the archive containing ffmpeg for MacOS to the UVR directory. Archive [download here](https://www.mediafire.com/file/zl0ylz150ouh366/ffmpeg_mac.zip/file)

This process has been tested on a MacBook Pro 2021 (using M1) and a MacBook Air 2017 and is confirmed to be working on both.

**Please Note:**

- For manual installs, drag-n-drop currently does not work for Macs running M1.
- So far everthing else is working as expected.

</details>

### Linux Installation

<details id="LinuxInstall">
  <summary>See Linux Installation Instructions</summary>

<br />
    
**These install instructions are for Ubuntu 22.10.**

- Download and save this repository [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
- From the saved directory run the following commands in this order- 

```
sudo apt-get update
```
```
sudo apt install python3-pip
```
```
sudo apt-get -y install python3-tk
```
```
pip3 install -r requirements.txt
```

</details>

### Other Application Notes

- Nvidia GPUs with at least 8GBs of V-RAM are recommended.
- This application is only compatible with 64-bit platforms. 
- This application relies on Sox - Sound Exchange for Noise Reduction.
- This application relies on FFmpeg to process non-wav audio files.
- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
- These models are computationally intensive. 

## Change Log

### Patch Version: 

- UVR_Patch_12_29_22_07_20

### Fixes & Changes:

- The progress bar is now fully synced up with every process in the application.
- Drag-n-drop feature should now work every time.
- Users can now drop large batches of files and directories as inputs. When directoriesare dropped, the application will search for any file with an audioextension and add it to the list of inputs.
- Fixed low resolution icon.
- Added the ability to download models manually if the application can't connect to the internet on it's own.
- Various bug fixes for the Download Center.
- Various design changes.

### Performance:

- Model load times are faster.
- Importing/exporting audio files is faster.

### New Options:

- "Select Saved Settings" option - Allows the user to save the current settings of the whole application. You can also load a saved setting or reset them to the default.
- "Right-click" menu - Allows for quick access to important options.
- "Help Hints" option - When enabled, users can hover over options to see pop-up text that describes that option. The right-clicking option also allows copying the "Help Hint" text.
- Secondary Model Mode - This option is an expanded version of the "Demucs Model" option that was only available to MDX-Net. Except now, this option is available in all three AI Networks and for any stem. Any model can now be Secondary, and the user can choose the amount of influence it has on the final result.
- Robust caching for ensemble mode, allowing for much faster processing times.
- Clicking the "Input" field will pop-up a new window that allows the user to go through all of the selected audio inputs. Within this menu, users can:
    - Remove inputs.
    - Verify inputs.
    - Create samples of selected inputs.
- "Sample Mode" option - Allows the user to process only part of a track to sample settings or a model without running a full conversion.
    - The number in the parentheses is the current number of seconds the generated sample will be.
    - You can choose the number of seconds to extract from the track in the "Additional Settings" menu.

### VR Architecture:

- Ability to toggle "High-End Processing."
- Support for the latest VR architecture
    - Crop Size and Batch Size are specifically for models using the latest architecture only.

### MDX-NET:

- "Denoise Output" option - When enabled, this option results in cleaner results, but the processing time will be longer. This option has replaced Noise Reduction.
- "Spectral Inversion" option - This option uses spectral inversion techniques for a cleaner secondary stem result. This option may slow down the audio export process.
- Secondary stem now has the same frequency cut-off as the main stem.

### Demucs:

- Demucs v4 models are now supported, including the 6 stem model.
- Ability to combine remaining stems instead of inverting selected stem with the mixture only when a user does not select "All Stems".
- A "Pre-process" model that allows the user to run an inference through a robust vocal or instrumental model and separate the remaining stems from its generated instrumental mix. This option can significantly reduce vocal bleed in other Demucs-generated non-vocal stems.
  - The Pre-process model is intended for Demucs separations for all stems except vocals and instrumentals.

### Ensemble Mode: 

- Ensemble Mode has been extended to include the following:
    - "Averaging" is a new algorithm that averages the final results.
    - Unlimited models in the ensemble.
    - Ability to save different ensembles.
    - Ability to ensemble outputs for all individual stem types.
    - Ability to choose unique ensemble algorithms.
    - Ability to ensemble all 4 Demucs stems at once.

## Troubleshooting

### Common Issues

- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.
- Memory allocation errors can usually be resolved by lowering the "Chunk Size".

### Issue Reporting

Please be as detailed as possible when posting a new issue. 

If possible, click the "Settings Button" to the left of the "Start Processing" button and click the "Error Log" button for detailed error information that can be provided to us.

## Manual Installation (For Developers)

**PLEASE NOTE:** Manual installs are **not** possible at this time! The new manual install instructions will be updated before the end of the year.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits

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

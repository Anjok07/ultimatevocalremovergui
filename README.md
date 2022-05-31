# Ultimate Vocal Remover GUI v5.2.1
<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/UVRv5.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application uses state-of-the-art source separation models to remove vocals from audio files. UVR's core developers trained all of the models provided in this package (except for the Demucs helper model).

- **Core Developers**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

## Installation

### Windows Installation

This installation bundle contains the UVR interface, Python (stripped to the bare essentials), PyTorch, and other dependencies needed to run the application effectively. No prerequisite installs required.

- Please Note:
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPU systems is not guaranteed.

- Download the UVR installer via one of the following mirrors below:
    - [Main Download Link](https://download.multimedia.workers.dev/UVR_v5.2.1_setup.exe)
    - [Google Drive Mirror](https://drive.google.com/file/d/1kA1dsZGTu7s2R_wuXO290HxtkBpzvfnC/view?usp=drivesdk)

- **Optional**
    - The Model Expansion Pack can be downloaded [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.2.0/v5_model_expansion_pack.zip)
        - Please navigate to the "Updates" tab within the Help Guide provided in the GUI for instructions on installing the Model Expansion pack.
    - This version of the GUI is fully backward compatible with the v4 models.

- **Please Note:** A new patch has been released. 
    - The has been addressed:
        - Fixed an issue with the Demucs model.
        - The application now automatically detects your resolution and sets itself accordingly.
        - Ensemble Customization (Nagivate to the "Advanced" tab in the Help Guide for more info)
           - Be sure to download and extract the newest model extension pack to get the most out of this option.
        - Enhanced error handling.
  - Patch installation instructions:
     1. Download the _*UVR_Patch.zip*_ file [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.2.0/UVR_Patch.zip)
     2. Navigate to the application directory
     3. Close UVR if you have it open.
     4. Delete the "data.pkl" file (you will receive a "Key Error" if you don't remove it.)
     5. Delete or rename the _"UVR.exe"_ file within the application directory
     6. Extract the _"UVR.exe"_ file and lib_v5 directory within the _*UVR_Patch.zip*_ archive to the application directory. 
     7. Open the application to ensure workability.

### Other Platforms

This application can be run on Mac & Linux by performing a manual install (see the **Manual Developer Installation** section below for more information). Some features may not be available on non-Windows platforms.

## Application Manual

**General Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/gen_opt.png?raw=true" />

**VR Architecture Options**

<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/vr_opt.png" />

**MDX-Net Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/mdx_opt.png?raw=true" />

**Ensemble Options**

<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/ense_opt_up.png" />

**User Ensemble**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/user_ens_opt.png?raw=true" />

### Other Application Notes

- Nvidia GPUs with at least 8GBs of V-RAM are recommended.
- This application is only compatible with 64-bit platforms. 
- This application relies on Sox - Sound Exchange for Noise Reduction.
- This application relies on FFmpeg to process non-wav audio files.
- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
- These models are computationally intensive. Please proceed with caution and pay attention to your PC to ensure it doesn't overheat. ***We are not responsible for any hardware damage.***

## Change Log

- **v4 vs. v5**
   - The v5 models significantly outperform the v4 models.
   - The extraction's aggressiveness can be adjusted using the "Aggression Setting." The default value of 10 is optimal for most tracks.
   - All v2 and v4 models have been removed.
   - Ensemble Mode added - This allows the user to get the most robust result from each model.
   - Stacked models have been entirely removed.
     The new aggression setting and model ensembling have replaced the stacked model feature.
   - The NFFT, HOP_SIZE, and SR values are now set internally.
   - The MDX-NET AI engine and models have been added.
     - This is a brand new feature added to the UVR GUI. 
     - 4 MDX-Net models are included in this package.
     - The MDX-Net models provided were trained by the core UVR developers
     - This network is less resource-intensive but incredibly powerful.
     - MDX-Net is a Hybrid Waveform/Spectrogram network.

## Troubleshooting

### Common Issues

- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.
- Memory allocation errors can usually be resolved by lowering the "Chunk Size".

### Issue Reporting

Please be as detailed as possible when posting a new issue. 

If possible, click the "Help Guide" button to the left of the "Start Processing" button and navigate to the "Error Log" tab for detailed error information that can be provided to us.

## Manual Installation (For Developers)

These instructions are for those installing UVR v5.2.0 **manually** only.

1. Download & install Python 3.9 or lower (but no lower than 3.6) [here](https://www.python.org/downloads/)
    - **Note:** Ensure the *"Add Python to PATH"* box is checked
2. Download the Source code [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
3. Download the models.zip [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.2.0/models.zip)
4. Extract the *ultimatevocalremovergui-master* folder within ultimatevocalremovergui-master.zip where ever you wish.
5. Extract the the folders within the models.zip to the *ultimatevocalremovergui-master/models* directory.
6. Download the SoX archive [here](https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download) and extract the contents into the *ultimatevocalremovergui-master/lib_v5/sox* directory.
7. Open the command prompt from the ultimatevocalremovergui-master directory and run the following commands, separately - 

```
pip install --no-cache-dir -r requirements.txt
```
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

From here you should be able to open and run the UVR.py file

- FFmpeg 

    - FFmpeg must be installed and configured for the application to process any track that isn't a *.wav* file. You will need to look up instruction on how to configure it on your operating system.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits

- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - Designed the official UVR logo, icon, banner, splash screen, and interface.
- [tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original Demucs AI code. 
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. This project is 100% open-source and free for anyone to use and modify as they wish. 
- We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

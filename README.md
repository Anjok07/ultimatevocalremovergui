# Ultimate Vocal Remover GUI v5.4.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/UVR_v54.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application uses state-of-the-art source separation models to remove vocals from audio files. UVR's core developers trained all of the models provided in this package (except for the Demucs helper model).

- **Core Developers**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

## Installation

### Windows Installation

This installation bundle contains the UVR interface, Python, PyTorch, and other dependencies needed to run the application effectively. No prerequisites are required.

- Please Note:
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.

- Download the UVR installer via the link below:
    - [Main Download Link](https://uvr.uvr.workers.dev/UVR_v5.4_setup.exe)
- For those who have UVR already installed:
    - [Update Package](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.4.0/UVR_v5.4_Update_Package.exe)
- **Optional**
    - Additional models and application patches can be downloaded via the "Settings" menu within the application.

- **Please Note:** See the latest release page for more recent updates [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.4.0)

### Other Platforms

This application can be run on Mac & Linux by performing a manual install (see the **Manual Developer Installation** section below for more information). Some features may not be available on non-Windows platforms.

## Application Manual

**General Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/gen_opt.png?raw=true" />

**VR Architecture Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/vr_opt.png?raw=true" />

**MDX-Net Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/mdx_opt.png?raw=true" />

**Demucs v3 Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/demucs_opt.png?raw=true" />

**Ensemble Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/ense_opt.png?raw=true" />

**User Ensemble**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/user_ens_opt.png?raw=true" />

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
   - The Demucs v3 AI engine and models have been added.
   - The ability to separate all 4 stems through Demucs v3.

## Troubleshooting

### Common Issues

- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.
- Memory allocation errors can usually be resolved by lowering the "Chunk Size".

### Issue Reporting

Please be as detailed as possible when posting a new issue. 

If possible, click the "Settings Button" to the left of the "Start Processing" button and click the "Error Log" button for detailed error information that can be provided to us.

## Manual Installation (For Developers)

These instructions are for those installing UVR v5.2.0 **manually** only.

1. Download & install Python 3.9 or lower (but no lower than 3.6) [here](https://www.python.org/downloads/)
    - **Note:** Ensure the *"Add Python to PATH"* box is checked
2. Download the Source code [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
3. Download the models via the "Settings" menu within the application.
4. Extract the *ultimatevocalremovergui-master* folder within ultimatevocalremovergui-master.zip where ever you wish.
5. Download the SoX archive [here](https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download) and extract the contents into the *ultimatevocalremovergui-master/lib_v5/sox* directory.
6. Open the command prompt from the ultimatevocalremovergui-master directory and run the following commands, separately - 

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
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - Designed the official UVR logo, icon, banner, and splash screen.
- [tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original Demucs AI code. 
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. 
- This project is 100% open-source and free for anyone to use and modify as they wish. 
- We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

(Almost complete. Full release coming this week!)

# Ultimate Vocal Remover GUI v5.2.0
<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/UVRv5.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application uses state-of-the-art AI and models to remove vocals from tracks. UVR's core developers trained all of the models provided in this package.

- **Core Developers**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

## Installation

UVR v5.2.0 and all of its features are only available on Windows. However, this application can be run on Mac & Linux by performing a manual install (see the **Manual Developer Installation** section below for more information). Some features may not be available on non-Windows platforms.

### Windows Installation

The installer does not require any prerequisites. All of the required libraries are included within the installation.

- Download the UVR installer [here]()
- **Optional**
    - The Model Expansion Pack can be downloaded [here]()
        - Please navigate to the "Updates" tab within the Help Guide provided in the GUI for instructions on installing the Model Expansion pack.
    - This version of the GUI is fully backward compatible with the v4 models.

## Application Manual

**General Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/gen_opt.png?raw=true" />

**VR Architecture Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/vr_opt.png?raw=true" />

**MDX-Net Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/mdx_opt.png?raw=true" />

**Ensemble Options**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/ense_opt.png?raw=true" />

**User Ensemble**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/v5.2.0/img/user_ens_opt.png?raw=true" />

## Other GUI Notes

- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
  - **Note:** This application will *not* be friendly to older or budget hardware. Please proceed with caution! Please pay attention to your PC and make sure it doesn't overheat. ***We are not responsible for any hardware damage.***

## Manual Developer Installation

These instructions are for those installing UVR v5.2.0 **manually** only.

1. Download & install Python 3.9.8 [here](https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe) (Windows link)
    - **Note:** Ensure the *"Add Python 3.9 to PATH"* box is checked
2. Download the Source code zip [here]()
3. Download the models.zip [here]()
4. Extract the *ultimatevocalremovergui-master* folder within ultimatevocalremovergui-master.zip where ever you wish.
5. Extract the *models* folder within models.zip to the *ultimatevocalremovergui-master* directory.
6. Open the command prompt from the ultimatevocalremovergui-master directory and run the following commands, separately - 

```
pip install --no-cache-dir -r requirements.txt
```
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- FFmpeg 

    - FFmpeg must be installed and configured for the application to process any track that isn't a *.wav* file. Instructions for installing FFmpeg is provided in the "More Info" tab within the Help Guide.

- Running the GUI & Models

    - Open the file labeled *'UVR.py'*.
    - It's recommended that you create a shortcut for the file labeled *'UVR.py'* to your desktop for easy access.
        - **Note:** If you are unable to open the *'UVR.py'* file, please go to the **troubleshooting** section below.
    - **Note:** All output audio files will be in the *'.wav'* format.

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
     - 4 MDX-Net models trained by UVR developers are included in this package.
     - The MDX-Net models provided were trained by the UVR core developers
     - This network is less resource-intensive but incredibly powerful.
     - MDX-Net is a Hybrid Waveform/Spectrogram network.

## Troubleshooting

### Common Issues

- This application is only compatible with 64-bit platforms. 
- This application is not compatible with 32-bit versions of Python. Please make sure your version of Python is 64-bit. 
- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.

### Issue Reporting

Please be as detailed as possible when posting a new issue. Navigate to the "Error Log" tab in the Help Guide for detailed error information that can be provided to us.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits

[DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
[tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
[Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
[Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original MDX-Net AI code. 
Bas Curtiz - Designed the official UVR logo, icon, banner, splash screen, and interface.

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. This project is 100% open-source and free for anyone to use and modify as they wish. 
- We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

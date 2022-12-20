# Ultimate Vocal Remover GUI v5.5.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/gui_data/img/UVR_5_5.jpg?raw=true" />

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

### Windows Installation

This installation bundle contains the UVR interface, Python, PyTorch, and other dependencies needed to run the application effectively. No prerequisites are required.

- Please Note:
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.

- Download the UVR installer via the link below:
    - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.5.0/UVR_v5.5.0_setup.exe)
    - [Main Download Link mirror](https://www.mediafire.com/file/59n8g1sa47ji91n/UVR_v5.5.0_setup.exe/file)
- Update Package instructions for those who have UVR already installed:
    - Please download the patch straight through the application.
- **Optional**
    - Additional models and application patches can be downloaded via the "Settings" menu within the application.

- **Please Note:** Please install UVR to the main C:\ drive if you use the Windows installer. Installing UVR to a secondary drive will cause application instability.

### MacOS Installation

- The stand alone version will be coming soon!
- If you wish to run UVR on MacOS at this time, you can manually install the UVR Mac build [here]( https://github.com/Anjok07/ultimatevocalremovergui/tree/uvr_5_5_MacOS)

### Linux Installation

- Linux installs will need to be done manually. See the Manual install section, or try the Mac Build above.

### Other Application Notes

- Nvidia GPUs with at least 8GBs of V-RAM are recommended.
- This application is only compatible with 64-bit platforms. 
- This application relies on Sox - Sound Exchange for Noise Reduction.
- This application relies on FFmpeg to process non-wav audio files.
- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
- These models are computationally intensive. Please proceed with caution and pay attention to your PC to ensure it doesn't overheat. ***We are not responsible for any hardware damage.***

## Change Log

### Patch Version: 

- UVR_Patch_12_18_22_6_41

### Fixes & Changes:

- The progress bar is now fully synced up with every process in the application.
- Drag-n-drop feature should now work every time.
- Users can now drop massive amounts of files and directories as inputs, and the application will add them to the conversion list.
- Various bug fixes for the Download Center.
- Various design changes.

### Performance:

- Model load times are faster.
- Importing/exporting audio files is faster.

### New Options:

- "Select Saved Settings" option - Allows the user to save the current settings of the whole application or reset them to the default.
- "Right-click" menu - Allows for quick access to important options.
- "Help Hints" option - When enabled, users can hover over options to see pop-up text that describes that option. The right-clicking option also allows copying the "Help Hint" text.
- Secondary Model Mode - This option is an expanded version of the "Demucs Model" option that was only available to MDX-Net. Except now, this option is available in all three AI Networks and stems. Any model can now be Secondary, and the user can choose the amount of influence the Secondary model has, unlike before.
- Robust caching for ensemble mode, allowing for much faster processing times.
- You can now drag and drop as many files/folders as inputs. The application willautomatically go through each selected directory for audio files.
- Clicking the "Input" field will pop-up a new window that allows the user to go through all of the selected audio inputs and remove some, if desired.
- "Sample Mode" option - Allows the user to process only part of a track to sample settings or a model without running a full conversion.
    - The number in the parentheses is the current number of seconds the generated sample will be.
    - You can choose the number of seconds to extract from the track in the "Additional Settings" menu.

### VR Architecture:

- Support for the latest VR architecture
    - Crop Size and Batch Size are specifically for models using the latest architecture only.
    - Ability to toggle "High-End Processing."

### MDX-NET:

- "Denoise Output" option - When enabled, this option results in cleaner results, but the processing time will be longer. This option has replaced Noise Reduction.
- "Spectral Inversion" option - This option uses inversion techniques for a cleaner secondary stem result. This option may slow down the audio export process.
- Secondary stem now has the same frequency cut-off as the main stem.

### Demucs:

- Demucs v4 models are now supported, including the 6 stem models.
- Ability to combine remaining stems instead of inverting selected stems with the 
mixture only when a user selects 2 stems.
- A "Pre-process" model that allows the user to run an inference through a robust vocal or instrumental model and separate the remaining stems from its generated instrumental mix. This option can significantly reduce vocal bleed in other Demucs-generated non-vocal stems. 

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

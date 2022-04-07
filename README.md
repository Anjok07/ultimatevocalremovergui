# Ultimate Vocal Remover GUI v5.0.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/UVRV5.png" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application is a GUI version of the vocal remover AI created and posted by GitHub user [tsurumeso](https://github.com/tsurumeso). This version also comes with eight high-performance models trained by me. You can find tsurumeso's original command-line version [here](https://github.com/tsurumeso/vocal-remover). 

- **The Developers**
    - [Anjok07](https://github.com/anjok07)- Model collaborator & UVR developer.
    - [aufr33](https://github.com/aufr33) - Model collaborator & fellow UVR developer. This project wouldn't be what it is without your help. Thank you for your continued support!
    - [DilanBoskan](https://github.com/DilanBoskan) - The main UVR GUI developer. Thank you for helping bring the GUI to life! Your hard work and continued support are greatly appreciated.
    - [tsurumeso](https://github.com/tsurumeso) - The engineer who authored the original AI code. Thank you for the hard work and dedication you put into the AI code UVR is built on!

## Change Log

- **v4 vs. v5**
   - The v5 models significantly outperform the v4 models.
   - The extraction's aggressiveness can be adjusted using the "Aggression Setting". The default value of 10 is optimal for most tracks.
   - All v2 and v4 models have been removed.
   - Stacked models have been entirely removed.
     - Stacked model feature has been replaced by the new aggression setting and model ensembling.
   - The NFFT, HOP_SIZE, and SR values are now set internally.

- **Upcoming v5.1 Update**
   - Model ensembling
   - MDX-NET AI engine and model support

## Installation

The application was made with Tkinter for cross-platform compatibility, so it should work with Windows, Mac, and Linux systems. However, this application has only been tested on Windows 10 & Linux Ubuntu.

### Install Required Applications & Packages

1. Download & install Python 3.7 [here](https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe) (Windows link)
    - **Note:** Ensure the *"Add Python 3.7 to PATH"* box is checked
2. Once Python has been installed, download **Ultimate Vocal Remover GUI Version 5.0.0** [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.0.0)
3. Please add the models included in the latest package to your 'models' folder. 
    - **Note:** At this time, the GUI is hardcoded to run the models included in this package only.

```
pip install --no-cache-dir -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### FFmpeg 

FFmpeg must be installed and configured for the application to process any track that isn't a *.wav* file. Instructions for installing FFmpeg can be found on YouTube, WikiHow, Reddit, GitHub, and many other sources around the web.

- **Note:** If you are experiencing any errors when attempting to process any media files, not in the *.wav* format, please ensure FFmpeg is installed & configured correctly.

### Running the Vocal Remover GUI & Models

- Open the file labeled *'VocalRemover.py'*.
   - It's recommended that you create a shortcut for the file labeled *'VocalRemover.py'* to your desktop for easy access.
     - **Note:** If you are unable to open the *'VocalRemover.py'* file, please go to the [**troubleshooting**](https://github.com/Anjok07/ultimatevocalremovergui/tree/beta#troubleshooting) section below.
- **Note:** All output audio files will be in the *'.wav'* format.

## Option Guide
  
### Model Selections:

- **Choose Main Model** - Here is where you choose the main model to perform a deep vocal removal.
  - Each of the models provided were trained on different parameters, though they can convert tracks of all genres. 
  - Each model differs in the way they process given tracks.  
     - The *'Model Test Mode'* option makes it easier for the user to test different models on given tracks.

### Checkboxes
- **GPU Conversion** - Selecting this option ensures the GPU is used to process conversions. 
  - **Note:** This option will not work if you don't have a Cuda compatible GPU.
    - Nvidia GPU's are most compatible with Cuda.
  - **Note:** CPU conversions are much slower than those processed through the GPU. 
- **Post-process** - This option can potentially identify leftover instrumental artifacts within the vocal outputs. This option may improve the separation of *some* songs. 
  - **Note:** Having this option selected can adversely affect the conversion process, depending on the track. Because of this, it's only recommended as a last resort.
- **TTA** - This option performs Test-Time-Augmentation to improve the separation quality. 
  - **Note:** Having this selected will increase the time it takes to complete a conversion.
- **Output Image** - Selecting this option will include the spectrograms in *.jpg* format for the instrumental & vocal audio outputs.
- **Model Test Mode** - This option makes it easier for users to test the results of different models and model combinations by eliminating the hassle of manually changing the filenames and/or creating new folders when processing the same track through multiple models. This option structures the model testing process.
  - When *'Model Test Mode'* is selected, the application will auto-generate a new folder in the *'Save to'* path you have chosen.
    - The new auto-generated folder will be named after the model(s) selected.
    - The output audio files will be saved to the auto-generated directory.
    - The filenames for the instrumental & vocal outputs will have the selected model(s) name(s) appended. 

### Parameter Values

- **Window Size** - The smaller your window size, the better your conversions will be. However, a smaller window means longer conversion times and heavier resource usage. 
    - Here are the selectable window size values - 
        - **1024** - Low conversion quality, shortest conversion time, low resource usage
        - **512** - Average conversion quality, average conversion time, normal resource usage
        - **320** - Better conversion quality, long conversion time, high resource usage

- **Aggression Setting** - This option allows you to set how strong the vocal removal will be. The range is 0-100. The higher the value, the more the vocal data will be removed. Higher values can result in muddy-sounding instrumentals for instrumental models depending on the track being converted, so this isn't always recommended. The default is 10 for instrumental & vocal models.

- **Default Values:**
  - **Window Size** - 512
  - **Aggression Setting** - 10 (optimal setting for all conversions)

### Other Buttons:

- **Open Export Directory** - This button will open your 'save to' directory.
- **Restart Button** - If the application hangs for any reason, you can hit the circular arrow button to the right of the *'Start Conversion'* button.

## Models Included

All of the models included in the release were trained on large datasets containing diverse sets of music genres.

- **PLEASE NOTE:** Do not change the name of the models provided! The required parameters are specified and appended to the end of the filenames.

- **Model Network Types**
    - **HP2** - The model layers are much larger. However, this makes them resource heavy.
    - **HP** - The model layers are the standard size for UVR v5.

Here's a list of the models included within the package -

- **Main Models**
    - **HP2_3BAND_44100_MSB2.pth** - This is a strong instrumental model trained using more data and new parameters.
    - **HP2_4BAND_44100_1.pth** - This is a strong instrumental model.
    - **HP2_4BAND_44100_2.pth** - This is a fine tuned version of the HP2_4BAND_44100_1.pth model.
    - **HP_4BAND_44100_A.pth** - This is a strong instrumental model.
    - **HP_4BAND_44100_B.pth** - This is a fine tuned version of the HP_4BAND_44100_A.pth model.
    - **HP_KAROKEE_4BAND_44100_SN.pth** - This is a model that removes main vocals while leaving background vocals intact.
    - **HP_Vocal_4BAND_44100.pth** - This model emphasizes vocal extraction. The vocal stem will be clean, but the instrumental might sound muddy.
    - **HP_Vocal_AGG_4BAND_44100.pth** - This model also emphasizes vocal extraction and is a bit more aggressive than the previous model.

## Other GUI Notes

- The application will automatically remember your *'save to'* path upon closing and reopening until it's changed.
  - **Note:** The last directory accessed within the application will also be remembered.
- Multiple conversions are supported.
- The ability to drag & drop audio files to convert has also been added.
- Conversion times will significantly depend on your hardware. 
  - **Note:** This application will *not* be friendly to older or budget hardware. Please proceed with caution! Please pay attention to your PC and make sure it doesn't overheat. ***We are not responsible for any hardware damage.***

## Troubleshooting

### Common Issues

- This application is not compatible with 32-bit versions of Python. Please make sure your version of Python is 64-bit. 
- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.

### Issue Reporting

Please be as detailed as possible when posting a new issue. Make sure to provide any error outputs and/or screenshots/gif's to give us a clearer understanding of the issue you are experiencing.

If the *'VocalRemover.py'* file won't open *under any circumstances* and all other resources have been exhausted, please do the following - 

1. Open the cmd prompt from the UVR-V5GUI directory
2. Run the following command - 
```
python VocalRemover.py
```
3. Copy and paste the error output shown in the cmd prompt to the issues center on the GitHub repository.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

- **PLEASE NOTE:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers Anjok07, aufr33, & tsurumeso.

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. This project is 100% open-source and free for anyone to use and/or modify as they wish. 
- Please note that we do not maintain or directly support any of tsurumesos AI application code. We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

# Ultimate Vocal Remover GUI v4.0.0

![alt text](https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/beta/img/UVRBETA.jpg)

This application is a GUI version of the vocal remover AI created and posted by tsurumeso. This would not have been possible without tsurumeso's hard work and dedication! You can find tsurumeso's original command line version [here](https://github.com/tsurumeso/vocal-remover)

A very special thanks to the main code contributor [DilanBoskan](https://github.com/DilanBoskan)! DilanBoskan, thank you for all of your support in helping bring this project to life!

## Installation

The application was made with Tkinter for cross platform compatibility, so this should work with Windows, Mac, and Linux systems. I've only personally tested this on Windows 10 & Linux Ubuntu.

### Install Required Applications & Packages

1. Download & install Python 3.7 [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe) (Make sure to check the box that says "Add Python 3.7 to PATH" if you're on Windows)
2. Once Python has installed, open the Windows Command Prompt and run the following installs -

```
pip install Pillow
pip install tqdm==4.30.0
pip install librosa==0.6.3
pip install opencv-python
pip install numba==0.48.0
pip install SoundFile
pip install soundstretch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Getting & Running the Vocal Remover GUI & Models

1. Download Ultimate Vocal Remover GUI Version 4.1.0 here
2. Place the UVR-V4GUI folder where ever you wish (I put mine in my documents folder) and open the file labeled "VocalRemover.py" (I recommend you create a shortcut for the file labeled "VocalRemover.py" to your desktop)
3. Open the application and proceed to the next section for more information

## Option Guide

### Choose AI Engine:

- This option allows you to toggle between tsurumeso's v2 & v4 AI engines. 
- Please note - The TTA option and the ability to set the N_FFT value is not possible in the v2 AI engine as those are v4 options.

### Model Selections:

- Choose Main Model - Here is where you choose the main model to convert your tracks with.
- Choose Stacked Model - These models are only meant to process converted tracks. Selecting the "Stack Passes" option will enable you to select a stacked model to run with the main model. If you wish to only run a stacked model on a track, make sure the "Stack Conversion Only" option is checked.
- Keep in mind the dropdown options change between upon choosing a new AI engine!

### Parameter Values:

All models released by me will have the values it was trained on appended to the end of the filename like so "MGM-HIGHEND_sr44100_hl512_w512_nf2048.pth". The "sr44100_hl512_w512_nf2048" portion automatically sets those values in the application, so please do not change the model files names. If there are no value appended to the end of the model, the defaults are set and the value field will be editable. The default values are - 

- SR - 44100
- HOP LENGTH - 1024
- WINDOW SIZE - 512
- N_FFT - 2048

### Checkboxes:
- GPU Conversion - This option ensures the GPU is used for conversions. It will not work if you don't have a Cuda compatible GPU (Nividia GPU's are most compatible with Cuda) 
- Post-process - This option may improve the separation on some songs. I recommend only using it if conversions don't come  out well
- TTA - This option may improve the separation on some songs. Having this selected will run a track twice. Please note,    this option is NOT compatible with the v2 AI engine.
- Output Image - This option will include a spectrogram of the resulting instrumental & vocal tracks.
- Stack Passes - This option allows you to set the number of times you would like a track to run through a stacked model
- Stack Conversion Only - Selecting this will allow you to run a pair through the stacked model only.
- Model Test Mode - This option is meant to make it easier for users to test the results of different models without having to manually create new folders and/or change the filenames. When it's selected, the application will automatically generate a new folder with the name of the selected model in the "Save to" path you have chosen. The completed files will have the selected model name appended to it and be saved to the auto-generated folder.

### Other Buttons:

- Add New Model - This button will automatically take you to the models folder. If you are adding a new model, make sure to add it accordingly based on the AI engine it was trained on! All new models added will automatically be detected without having to restart the application.
- Restart Button - If the application hangs for any reason you can hit the circular arrow button immediately to the right of the "Start Conversion" button.

## Models Included:

Here's a list of the models (PLEASE DO NOT CHANGE THE NAME OF THE FIRST 2 MODELS LISTED AS THE PARAMETERS ARE SPECIFIED IN THE FILENAMES!):

- (Pending)

## Troubleshooting:

- If the VocalRemover.py file won't open under any circumstances and you have exhausted all other resources, please do the following - 

1. Open the cmd prompt from the UVR-V4GUI directory
2. Run the following command - 
```
python VocalRemover.py
```
3. Copy and paste the error in the cmd prompt to the issues center here on my GitHub.

## Other GUI Notes:

- The application will automatically remember your "save to" path upon closing and reopening until you change it
- You can select as many files as you like. Multiple conversions are supported!
- The Stacked Model is meant to clean up vocal residue left over in the form of vocal pinches and static. They are only meant for instrumentals created via converted tracks previously run through one of the main models!
- The "Stack Passes" option should only be used with the Stacked Model. This option allows you to set the amount of times you want a track to run through a model. The amount of times you need to run it through will vary greatly by track. Most tracks won't require any more than 2-5 passes. If you do 5 or more passes on a track you risk quality degradation.
- Conversion times will greatly depend on your hardware. This application will NOT be friendly to older or budget hardware. Please proceed with caution! Pay attention to your PC and make sure it doesn't overheat.

```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference

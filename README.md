# Ultimate Vocal Remover GUI (Develop Version)

***NEW VERSION COMING 11/14/2020!***

![alt text](https://github.com/Anjok07/ultimatevocalremovergui/blob/develop/Images/UVR-app.jpg)

This is a deep-learning-based tool that extracts the instrumental from a track containing vocals. This project is a GUI version of the vocal remover created and posted by tsurumeso. This would not have been possible without tsurumeso's work and dedication! You can find tsurumeso's original command line version [here](https://github.com/tsurumeso/vocal-remover)

## Installation

The application was made with Tkinter for cross platform compatibility, so this should work with Windows, Mac, and Linux systems. I've only personally tested this on Windows 10 & Linux Ubuntu.

### Install Required Applications & Packages

1. Download & install Python 3.7 (Make sure to check the box that says "Add Python 3.7 to PATH" if you're on Windows)
2. Once Python has installed, open the Windows Command Prompt and run the following installs -
- If you plan on doing conversions with your Nvidia GPU, please install the following -
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
- If you don't have a compatible Nvidia GPU and plan on only using the CPU version please do not check the "GPU Conversion" option in the GUI and install the following -

```
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
- The rest need to be installed regardless! -

```
pip install Pillow
pip install tqdm==4.30.0
pip install librosa==0.6.3
pip install opencv-python
pip install numba==0.48.0
pip install SoundFile
pip install soundstretch
```
3. For the ability to convert mp3, mp4, m4a, and flac files, you'll need ffmpeg installed and configured!

### Getting the Vocal Remover GUI & Models
Download the latest version from [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v2.2.0-GUI-Dev).

## Running the Vocal Remover Application GUI
1. Place this folder where ever you wish (I put mine in my documents folder) and open the file labeled "VocalRemover.py" (I reccomend you create a shortcut for the file labeled "VocalRemover.py" to your desktop)
2. Open the application

### Notes Regarding the GUI

 - The application will automatically remember your "save to" path upon closing and reopening until you change it
 - You can select as many files as you like. Multiple conversions are supported!
 - Conversions on wav files should always work with no issue. However, you will need to install and configure ffmpeg in order for conversions on mp3, mp4, m4a, and FLAC formats. If you select non-wav music files without having ffmpeg configured and attempt a conversion it will freeze and you will have to restart the application.
 - Only check the GPU box if you have the Cuda driver installed for your Nvidia GPU. Most Nvidia GPU's released prior to 2015 or with less than 4GB's of V-RAM might not be compatible.
- The dropdown model menu consists of a provisional model that I trained with roughly 200 tracks. I included the option to add your own model as well if you've trained your own. Alternatively, you can also simply add a model to the models directory and restart the application, as it will automatically show there.
- The SR, N FFT, HOP LENGTH, and WINDOW SIZE parameters are set to the defaults. Those were the parameters used in training, so changing them may result in poor conversion performance unless the model is compatible with the changes made. Those are essentially advanced settings, so I recommend you leave them as is unless you know exactly what you're doing.
- The Stacked Model is meant to clean up vocal residue left over in the form of vocal pinches and static. This model is only meant for instrumentals created via converted tracks that ran through one of the main models!
- The "Stack Passes" option should only be used with the Stacked Model. This option allows you to set the amount of times you want a track to run through a model. The amount of times you need to run it through will vary greatly by track. Most tracks won't require any more than 2-5 passes. If you do 5 or more passes on a track you risk quality degradation. When doing stack passes the first and last "vocal" track will give you an idea of how much static was removed.
- Conversion times will greatly depend on your hardware. This application will NOT be friendly to older or budget hardware. Please proceed with caution! Pay attention to your PC and make sure it doesn't overheat.

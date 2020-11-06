# Ultimate Vocal Remover GUI

******A NEW UPDATE IS COMING THE WEEK ENDING 11/14/2020!******

![alt text](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/Images/UVR-App.jpg)

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
Download the latest version from [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v2.2.0-GUI).

## Running the Vocal Remover Application GUI
1. Place this folder where ever you wish (I put mine in my documents folder) and open the file labeled "VocalRemover.py" (I reccomend you create a shortcut for the file labeled "VocalRemover.py" to your desktop)
2. Open the application

### Notes Regarding the GUI

 - The application will automatically remember your "save to" path upon closing and reopening until you change it
 - You can select as many files as you like. Multiple conversions are supported!
 - Conversions on wav files should always work with no issue. However, you will need to install and configure ffmpeg in order for conversions on mp3, mp4, m4a, and FLAC formats. If you select non-wav music files without having ffmpeg configured and attempt a conversion it will freeze and you will have to restart the application.
 - Only check the GPU box if you have the Cuda driver installed for your Nvidia GPU. Most Nvidia GPU's released prior to 2015 or with less than 4GB's of V-RAM might not be compatible.
- The dropdown model menu consists of the Multi-Genre Model I just finished (trained on 700 pairs), a stacked model (a model trained on converted data), & the stock model the AI originally came with (for comparison). I included the option to add your own model as well if you've trained your own. Alternatively, you can also simply add a model to the models directory and restart the application, as it will automatically show there.
- The SR, HOP LENGTH, and WINDOW SIZE parameters are set to the defaults. Those were the parameters used in training, so changing them may result in poor conversion performance unless the model is compatible with the changes made. Those are essentially advanced settings, so I recommend you leave them as is unless you know exactly what you're doing.
- The Post-Process option is a developement option. Keep it unchecked for most conversions, unless you know what you're doing.
- The "Save Mask PNG" option allows you to to save a copy of the spectrogram as a PNG.
- The Stacked Model is meant to clean up vocal residue left over in the form of vocal pinches and static. This model is only meant for instrumentals created via converted tracks that ran through one of the main models!
- The "Stack Passes" option should only be used with the Stacked Model. This option allows you to set the amount of times you want a track to run through a model. The amount of times you need to run it through will vary greatly by track. Most tracks won't require any more than 2-5 passes. If you do 5 or more passes on a track you risk quality degration. When doing stack passes the first and last "vocal" track will give you an idea of how much static was removed.
- Conversion times will greatly depend on your hardware. This application will NOT be friendly to older or budget hardware. Please proceed with caution! Pay attention to your PC and make sure it doesn't overheat.

## Train Your Own Model

### Install SoundStretch
```
sudo apt install soundstretch
```
### Run training script
```
python train.py -i dataset/instrumentals -m dataset/mixtures -M 0.5 -g 0
```

`-i` specifies an instrumental audio directory, and `-m` specifies the corresponding mixture audio directory.

```
dataset/
  +- instrumentals/
  |    +- 01_foo_inst.wav
  |    +- 02_bar_inst.mp3
  |    +- ...
  +- mixtures/
       +- 01_foo_mix.wav
       +- 02_bar_mix.mp3
       +- ...
```

### Notes on Training

- FIRST AND FORMOST! - The notes below are from my own experience. I've learned as I went along. All technical questions regarding the training process should be directed to tsurumesos' Vocal Remover Github [here](https://github.com/tsurumeso/vocal-remover) 
- Training can take very long, I mean from a day to days, to a weeks straight depending on how many pairs you're using and your PC specs.
- The better your PC specifications are, the quicker training will be.
- Training using your GPU is a must! CPU training is possible but prohibitavley slow. The more V-RAM, the bigger you can make your batch size. 4 is the default.
- If you choose to train using your GPU, you will need to do research on your GPU first to see how it can be used to train. For example, if you have a high-performance Nvidia Graphics card, you'll need to install compatible Cuda drivers for it to work properly. 
- The dataset should be comprised of pairs consisting of instrumentals & mixes. These pairs can be made out of multi-tracks/stems, official instrumental & mix versions of tracks, or augumented data.
- The pairs really should be perfectly aligned. Also, the spectrograms must match up evenly, and the pairs should have the same volume. Otherwise, training will not be effective.
- Keep in mind, not every official studio instrumental will align with its official mix counterpart. If it doesn't align, it shouldn't be used in your dataset. What I found was if the timing is slightly different between the 2 tracks, it will render it impossible to align. If you're not familiar with how to align tracks, just know it's the same process people use to extract vocals using an instrumental and mix (vocal extraction isn't necessary for this. Although knowing if they can be aligned using this method will make for a good litmus test that will tell you if the pair is a good candidate for training. There are tons of tutorials on YouTube that show how to perfectly align tracks. Also, you can bypass some of that work by making an instrumental and mix out of multi-track/stems. Using multi-tracks to create pairs is the most concrete way to build a perfectly aligned dataset. However, carefully aligning wave-forms and spectrograms for instrumentals and mixes is a pretty good way too because you can easily expand your dataset.
- From my own experience, I advise against using instrumentals with background vocals and TV tracks as they can undermine the effectiveness of the resulting models.
- I have found that you can use tracks of any quality, as long as the tracks in the pair are the exact same quality (natively) (use Spek or Audacity to confirm this). For example, if you have a lossless wav mix, but the instrumental is only a 128kbps mp3, you'll probably want to convert the lossless mix down to 128kbps so the spectrograms can match up. Don't convert a file up from 128kbps to anything higher like 320kbps, as that won't help at all. If you have an instrumental that's 320kbps and a lossless version of it doesn't exist, I recommend you convert the lossless mix wav file down to 320kbps. With that being said, using high-quality tracks does make training more efficient, but 320kbps and a lossless wave file don't appear to make much of a difference at all. However, I suggest not using ANY pair below 128kbps. You're better off keeping the dataset between 128-320kbps.
- When you start training you'll see it go through what are called "Epochs". 100 epochs are the standard, but more can be set if you feel it's necessary. However, if you train on more than 100 epochs you run the risk of the training session stagnating (basically waste training time). You need to keep a close eye on the "training loss" & "validation loss" numbers. The goal is to get those numbers as low as you can. However, the "training loss" number should always be slightly (and I mean slightly) lower than the "validation loss". If the "validation loss" number is ever significantly higher than the "training loss" after more than 10 full epochs, that means you might overfitting the model. If the "training loss" & "validation loss" number are both high and stay high after 2 or 3 epochs, then you either need to give it more time to train, have a poor dataset, or your dataset is too small (you should be training with a bare minimum of 50-75 pairs).
- A new model spawns after each epoch with the best validation loss, so you can get a good idea as to rather or not it's getting better based on the data you're training it on. However, I recommend you not run conversions during the training process. Load the model to an external device and test it on a separate PC, or via GoogleColab if you must. (Please note: GoogleColab is not compatible with GUI's. However, you can use the command line version of the AI for inferences)
- I recommend dedicating a PC to training. Close all applications and clear as much RAM as you can. Even if you use your GPU. A.I. training like this is a computationally-intensive process, so make sure your PC is properly cooled and check it's temperature every so often to keep it from overheating. Especially if you're not using a commercial grade server, or a high-end cooling system.
- Be Patient! The bigger the dataset, the longer training will take. I've yielded the best results using at least 300 pairs or more. The multi-genre model took a over a week and a half to train.

### Offline data augmentation
```
python augment.py -i dataset/instrumentals -m dataset/mixtures -p -1
python augment.py -i dataset/instrumentals -m dataset/mixtures -p 1
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference

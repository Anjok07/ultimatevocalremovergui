    Vocal Remover Installation Instructions

    **These instructions are for Windows Systems only!**

    The application was made with Tkinter for cross platform compatibility, so this should work with Windows, Mac, and Linux systems. I've only personally tested this on Windows 10 & Linux Ubuntu.

    Prerequisites for Running the Vocal Remover GUI

    1. Download & install Python here *Make sure to check the box that says "Add Python 3.7 to PATH"


    2. Once Python has installed, open the Windows Command Prompt and run the following installs -

    - If you plan on doing conversions with your Nvidia GPU, please install the following -

    pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

    - If you don't have a compatible Nvidia GPU and plan on only using the CPU version please do not check the "GPU Conversion" option in the GUI and install the following -

    pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

    - The rest need to be installed regardless! -

    pip install Pillow
    pip install tqdm==4.30.0
    pip install librosa==0.6.3
    pip install opencv-python
    pip install numba==0.48.0
    pip install SoundFile
    pip install soundstretch


    3. If you have a Nvidia GPU, I strongly advise that you install the following Cuda driver so you can run conversions with it. You can download the driver https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64

    4. For the ability to convert mp3, mp4, m4a, and flac files, you'll need ffmpeg installed and configured! I found a pretty good video on YouTube that explains this process; It's actually a Spleeter tutorial, but I provided a link to the relevant portion of the video that covers the ffmpeg installation process. Check this YouTube video out here - https://www.youtube.com/watch?v=tgnuOSLPwMI&feature=youtu.be&t=307

    Running the Vocal Remover Application GUI

    2. Place this folder where ever you wish (I put mine in my documents folder) and create a shortcut for the file labeled "VocalRemover.py" to your desktop
    3. Open the application

    Notes Regarding the GUI
 

    - The application will automatically remember your "save to" path upon closing and reopening until you change it
    - You can select as many files as you like. Multiple conversions are supported!
    - Conversions on wav files should always work with no issue. However, you will need to install and configure ffmpeg in order for conversions on mp3, mp4, m4a, and FLAC formats. If you select non-wav music files without having ffmpeg configured and attempt a conversion it will freeze and you will have to restart the application.
    - Only check the GPU box if you have the Cuda driver installed for your Nvidia GPU. Most Nvidia GPU's released prior to 2015 or with less than 4GB's of V-RAM might not be compatible.
    - The dropdown model menu consists of the Multi-Genre Model I just finished (trained on 700 pairs), a stacked model (a model trained on converted data), & the stock model the AI originally came with (for comparison). I added the option to add your own model as well if you've trained your own. Alternatively, you can also simply add a model to the models directory and restart the application, as it will automatically show there.
    - The SR, HOP LENGTH, and WINDOW SIZE parameters are set to the defaults. Those were the parameters used in training, so changing them may result in poor conversion performance unless the model is compatible with the changes made. Those are essentially advanced settings, so I recommend you leave them as is unless you know exactly what you're doing.
    - The Post-Process option is a developement option. Keep it unchecked for most conversions, unless you have a model that is compatible with it.
    - The "Save Mask PNG" option allows you to to save a copy of the spectrogram as a PNG.
    - The Stacked Model is meant to clean up vocal residue left over in the form of vocal pinches and static. 
    - The "Stack Passes" option should only be used with the Stacked Model. This option allows you to set the amount of times you want a track to run through the model. The amount of times you need to run it through will vary greatly by track. Most tracks won't require any more than 5 passes. If you do 5 or more passes on a track you risk quality degration. When doing stack passes the first and last "vocal" track will give you an idea of how much static was removed.
    - Conversion times will greatly depend on your hardware. This application will NOT be friendly to older or budget hardware. Please proceed with caution! Pay attention to your PC and make sure it doesn't overheat.

    Other Notes

    - Since there wasn't much demand for training, I went ahead and moved the training instructions to a readme file included within this package. 
    - I will be releasing new models over the course of the next few months. 


    Other Notes

    - Running conversions on this application is a computationally intensive processes! Please note, I do not take any responsibility for damaged hardware as a result of this application! It's highly recommended that you not use this on older or budget hardware. USE AT YOUR OWN RISK! 



	Enjoy!




Copyright@Anjok
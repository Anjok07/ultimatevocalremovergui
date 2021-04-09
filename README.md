# Ultimate Vocal Remover v5 Command Line Beta

## About

This application is a heavily modified version of the vocal remover AI created and posted by GitHub user [tsurumeso](https://github.com/tsurumeso). You can find tsurumeso's original command line version [here](https://github.com/tsurumeso/vocal-remover). The official v5 GUI is still under developement and will be released some time in Q3 2021. New beta models for this version will be released at the end of the week. 

- **Special Thanks**
    - [tsurumeso](https://github.com/tsurumeso) - The engineer who authored the AI code. Thank you for the hard work and dedication you put into the AI application this GUI is built around!
    - [aufr33](https://github.com/aufr33) - Model collaborator and fellow UVR developer. This project wouldn't be what it is without your help, thank you for your continued support!
    - [DilanBoskan](https://github.com/DilanBoskan) - The main GUI code contributor. Thank you for helping bring the GUI to life! Your hard work and continued support is greatly appreciated.

## Installation

### Install Required Applications & Packages

Please run the requirements command even if you have v4 installed!

```
pip install --no-cache-dir -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### FFmpeg 

FFmpeg must be installed and configured in order for the application to be able to process any track that isn't a *.wav* file. Instructions for installing FFmpeg can be found on YouTube, WikiHow, Reddit, GitHub, and many other sources around the web.

- **Note:** If you are experiencing any errors when attempting to process any media files that are not in the *.wav* format, please ensure FFmpeg is installed & configured correctly.

## Running Inferences & Model Details

Each model requires specific parameters to run smoothly. Those parameters are intricately defined within the JSON files provided. Please make sure the correct JSON files are selected when running inferences!

### Option Guide

Please note, this version is based on vocal-remover 4.0.0 of tsurumeso's original code. Significant improvements and changes were made. Those changes include the following - 

- New format of spectrograms. Instead of a single spectrogram with a fixed FFT size, combined spectrograms are now used. This version combines several different types of spectrograms within specific frequency ranges. This approach allowed for a clearer view of the high frequencies and good resolutions at low frequencies, thus allowing for more targeted vocal removals.
- The arguments --sr, --n_fft, --hop_length are removed. JSON files are now used instead.
- The following new features were added
	- **--high_end_process** - This argument restores the high frequencies of the instrumental (but not the vocals). The 3 choices for this argument are:
		- *none* - No processing (default)
		- *bypass* - This copies the missing frequencies from the input.
		- *correlation* - This also copies missing frequencies from the input, however, the magnitude of the copied frequency will depend on the magnitude of the generated instrumental's high frequencies.
	- **--aggressiveness** - This argument allows you to set how strong the vocal removal will be. The range is 0.0-1.0 The higher the value, the more the vocals will be removed. Please note, the highest value can result in muddy sounding instrumentals depending on the track being converted, so this isn't always recommended. The default is 0.1. For the vocal model specifically, the recommended value is 0.5-0.6.

### Models Included

All of the models included in the release were trained on large datasets containing diverse sets of music genres. These are all beta models that may or may not make it into the final release. We are working to have even better models in the final release of v5! You can download the model pack [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.0)

**Please Note:** These models are *not* compatible with the v4 GUI! The GUI for v5 is still under development.

Here's a list of the models included within the v5 beta package -

- **V5 Beta Models**
    - **2band_32000 Models**
        - **MGM-v5-2Band-32000-BETA1.pth** - This model does very well on lower frequencies. Frequency cut-off is 16000 Hz. Must be used with **2band_32000.json** file!
        - **MGM-v5-2Band-32000-BETA2.pth** - This model does very well on lower frequencies. Frequency cut-off is 16000 Hz. Must be used with **2band_32000.json** file!
        - **MGM-v5-KAROKEE-32000-BETA1.pth** - Model by aufr33. This model focuses on removing main vocals only, leaving the BV vocals mostly intact. Frequency cut-off is 16000 kHz. Must be used with **2band_32000.json** file!
        - **MGM-v5-KAROKEE-32000-BETA2-AGR.pth** - Model by aufr33. This model focuses a bit more on removing vocals from lower frequencies.Frequency cut-off is 16000 Hz. Must be used with **2band_32000.json** file!
        - **MGM-v5-Vocal_2Band-32000-BETA1.pth** - This is a model that provides cleaner vocal stems! Frequency cut-off is 16000 Hz. Must be used with **2band_32000.json** file!
        - **MGM-v5-Vocal_2Band-32000-BETA2.pth** - This is a model that provides cleaner vocal stems! Frequency cut-off is 16000 Hz. Must be used with **2band_32000.json** file!
    - **3band_44100 Models**
        - **MGM-v5-3Band-44100-BETA.pth** - This model does well removing vocals within the mid-rang frequencies. Frequency cut-off is 18000 Hz. Must be used with **3band_44100.json** file!
    - **3band_44100_mid Models**
        - **MGM-v5-MIDSIDE-44100-BETA1.pth** - This model does well removing vocals within the mid-range frequencies. Frequency cut-off is 18000 Hz. Must be used with **3band_44100_mid.json** file!
        - **MGM-v5-MIDSIDE-44100-BETA2.pth** - This model does well removing vocals within the mid-range frequencies. Frequency cut-off is 18000 Hz. Must be used with **3band_44100_mid.json** file!
    - **4band_44100**
        - **MGM-v5-4Band-44100-BETA1.pth** - This model does very well on lower-mid range frequencies. Frequency cut-off is 20000 Hz. Must be used with **4band_44100.json** file!
        - **MGM-v5-4Band-44100-BETA2.pth** - This model does very well on lower-mid range frequencies. Frequency cut-off is 20000 Hz. Must be used with **4band_44100.json** file!

### Inference Command Structure

The following example shows how to run a model from the "2band_32000 Models" section above.
```
python inference.py -g 0 -m 2band_32000.json -P models/MGM-v5-2Band-32000-BETA1.pth -i "INPUT"
```

### Windows Batch Files

We included Windows batch files to help automate the inference process for those running Microsoft Windows! The "windows-batch-files" folder contains a separate bat file for each model.

- To use them, please do the following:
	- Move the bat files to the main directory from the "windows-batch-files" folder.
	- Drag an audio file into the bat file named after the model you wish to run it through.
	- The outputs will be in the separated folder.

## Troubleshooting

### Common Issues

- This application is not compatible with 32-bit versions of Python. Please make sure your version of Python is 64-bit. 
- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.

### Issue Reporting

Please be as detailed as possible when posting a new issue. Make sure to provide any error outputs and/or screenshots/gif's to give us a clearer understanding of the issue you are experiencing.

If you are unable to run conversions *under any circumstances* and all other resources have been exhausted, please do the following - 

1. Copy and paste the error output shown in the cmd prompt to the issues center on the GitHub repository.

## License

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover** please send us a pull request and we will review it. This project is 100% open-source and free for anyone to use and/or modify as they wish. 
- We only maintain the development and support for **Ultimate Vocal Remover** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

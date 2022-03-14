# Ultimate Vocal Remover v5 Command Line Beta

## About

This application is a heavily modified version of the vocal remover AI created and posted by GitHub user [tsurumeso](https://github.com/tsurumeso). You can find tsurumeso's original command line version [here](https://github.com/tsurumeso/vocal-remover). The official v5 GUI is still under developement and will be released some time in Q4 2021. 

- **The Developers**
    - [Anjok07](https://github.com/anjok07)- Model collaborator & UVR developer.
    - [aufr33](https://github.com/aufr33) - Model collaborator & fellow UVR developer. This project wouldn't be what it is without your help, thank you for your continued support!
    - [DilanBoskan](https://github.com/DilanBoskan) - The main UVR GUI developer. Thank you for helping bring the GUI to life! Your hard work and continued support is greatly appreciated.
    - [tsurumeso](https://github.com/tsurumeso) - The engineer who authored the original AI code. Thank you for the hard work and dedication you put into the AI code UVR is built on!

## Installation

### Install Required Applications & Packages

Please run the requirements command even if you have v4 installed!

pip:
```
pip install --no-cache-dir -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

conda:
```
. Conda-Setup.sh uvr5b
conda activate uvr5b
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
	- **--high_end_process** - This argument restores the high-end frequencies of the output audio. In order to optimize performance, some models have bandwidths below 20 kHz. This argument allows you to choose one of 4 post-processing options:
		- *mirroring* -  This is the default value. This algorithm uses the high-end frequencies from the input and mirrored output's frequencies.
		- *mirroring2* - This version of mirroring is optimized for better performance.
		- *bypass* - This copies the missing frequencies from the input without any modification.
		- *none* - No processing.
	- **--aggressiveness** - This argument allows you to set how strong the vocal removal will be. The range is 0.0-1.0 The higher the value, the more the vocals will be removed. Please note, the highest value can result in muddy sounding instrumentals depending on the track being converted, so this isn't always recommended. The default is 0.1. For the vocal model specifically, the recommended value is 0.5-0.6.
	- **--deepextraction** - This argument generates an additional instrumental output with deep artifact vocal removal. This option is experimental and is more suited for acoustic or other light types of tracks with stubborn vocals. Many others might sound bad.
	- **--chunks** - Split the input file into chunks to reduce RAM consumption.

### Models Included

All of the models included in the release were trained on large datasets containing diverse sets of music genres. These are all beta models that may or may not make it into the final release. We are working to have even better models in the final release of v5! You can download the latest model pack [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.2)

**Please Note:** These models are *not* compatible with the v4 GUI! The GUI for v5 is still under development.

Here's a list of the models included within the v5 beta package -

- **V5 Beta Models**
    - **Newest 4band_44100 HP Models - Download [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.2)**. Must be used with 4band_44100.json file!
      - **HP_4BAND_3090** - This model was trained with a bigger dataset and optimized training parameters. The model weighs 121MB's due to increased capacity.
      - **HP2-4BAND-3090_4band_1** - This model was trained with a bigger dataset and optimized training parameters. The model weighs 524MB's due to increased capacity. Conversions will take longer with this capacity type.
      - **HP2-4BAND-3090_4band_2** - A lightly fine-tuned version of "HP2-4BAND-3090_4band_1". The model weighs 524MB's due to increased capacity. Conversions will take longer with this capacity type.
      - **Vocal_HP_4BAND_3090** - This is a new HP vocal model. The model weighs 121MB's due to increased capacity.
      - **Vocal_HP_4BAND_3090_AGG** - This is a more aggressive version of the "Vocal_HP_4BAND_3090" vocal model. The model weighs 121MB's due to increased capacity.
    - **A link and descriptions of the older beta models can be found [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.1)**

### Inference Command Structure

The following example shows how to run the HP_4BAND_3090 model -
```
python inference.py -g 0 -m modelparams/4band_44100.json -P models/HP_4BAND_3090.pth -i "INPUT"
```

The following examples show how to run the ensemble model scripts -

```
python ensemble_inference.py -g 0 -i "INPUT"
```

Or if you wish to save all individual outputs generated in addition to the final ensembled outputs, please run the following - 

```
python ensemble_inference.py -g 0 -s -i "INPUT"
```

By default, 14 models will be ensembled. However, you can also specify the models you wish to ensemble. For the ensemble_inference script specifically, do not input the ".pth" extension within the command, only the name. Here is an example - 

```
python ensemble_inference.py -g 0 -P "MODELNAME1" "MODELNAME2" "MODELNAME3" -i "INPUT"
```

- **Please Note the Following:** 
	- Do not specify the model parameters or architectures for the ensemble inference script. Those details are already fixed. 
	- When ensembling models with low and high bandwidth conversions, '--bypass' is highly recommended.
	- The ensembled outputs generated through the ensemble scripts can be found in the "ensembled" folder.

### Ensembler

The stand alone ensembler has the ability to take 2 or more instrumental or vocal outputs generated by different models and combine the best results from all of them! Here is how to use it manually - 

- For instrumental outputs, run the following command: 

```
python lib/spec_utils.py -a min_mag -m modelparams/1band_sr44100_hl512.json "INPUT1" "INPUT2" -o "CHOOSEFILENAME"
```

- For vocal outputs, run the following command: 

```
python lib/spec_utils.py -a max_mag -m modelparams/1band_sr44100_hl512.json "INPUT1" "INPUT2" -o "CHOOSEFILENAME"
```

To automate the ensembling process, please use the 'Ensemble-Outputs.bat' script reference below.

### Windows Batch Files

We included the following Windows batch files to help automate commands:

- Select-Models-HP-Only.bat
 	- This batch script is for the models provided [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.2) only!
	- Simply drag the audio file you wish to convert into the 'Select-Models-HP-Only.bat.bat' file provided. 
	- From there you will be asked if you want TTA enabled, then prompted to type the letter associated with the model you wish to run and hit "enter". 

- Drag-n-Drop-CHOOSE-YOUR-MODEL.bat
 	- This batch script is for the models provided [here](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/5.0.1) only!
	- Simply drag the audio file you wish to convert into the 'Drag-n-Drop-CHOOSE-YOUR-MODEL.bat' file provided. 
	- From there you will be asked if you want TTA enabled, then prompted to type the letter associated with the model you wish to run and hit "enter". 
	- Once you hit enter, you will be asked fs you want an additional "deep extraction" instrumental output in addition to the 2 to be provided.

- Ensemble-Outputs.bat
	- Simply drag the instrumental or vocal outputs you wish to ensemble into the batch script. 
	- From there you will be asked if the outputs are instrumental or not.
	- Once you hit enter, the script will take a moment to ensemble the outputs.
	- The ensembled outputs can be found in the "ensembled" folder.

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

- **PLEASE NOTE:** For all third party application developers who wish to use our models, please honor the MIT-license by providing credit to UVR and it's developers Anjok07, aufr33, & tsurumeso.

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover** please send us a pull request and we will review it. This project is 100% open-source and free for anyone to use and/or modify as they wish. 
- We only maintain the development and support for **Ultimate Vocal Remover** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf

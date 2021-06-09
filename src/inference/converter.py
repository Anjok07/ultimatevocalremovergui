# pylint: disable=no-name-in-module, import-error
# -GUI (Threads)-
from PySide2 import QtCore  # (QRunnable, QThread, QObject, Signal, Slot)
from PySide2 import QtWidgets
# Multithreading
import threading
from multiprocessing import Pool
# -Required for conversion-
import cv2
import librosa
import audioread
import numpy as np
import soundfile as sf
import torch
# -Root imports-
from .lib import dataset
from .lib import nets
from .lib import spec_utils
from ..resources.resources_manager import (ResourcePaths, Logger)
# -Other-
import traceback
# Loading Bar
from tqdm import tqdm
# Timer
import datetime as dt
import time
import os
# Annotating
from typing import (Dict, Tuple, Optional, Callable)

default_data = {
    # Paths
    'input_paths': [],  # List of paths
    'export_path': '',  # Export path
    # Processing Options
    'gpuConversion': False,
    'postProcess': True,
    'tta': True,
    'outputImage': False,
    # Models
    'instrumentalModel': '',  # Path to instrumental (not needed if not used)
    'vocalModel': '',  # Path to vocal model (not needed if not used)
    'isVocal': False,
    # Model Folder
    'modelFolder': False,  # Model Test Mode
    # Constants
    'window_size': 320,
    'deepExtraction': True,
    'aggressiveness': 0.02,
    # Allows to process multiple music files at once
    'multithreading': False,
    # What to save
    'save_instrumentals': True,
    'save_vocals': True,
}

class VocalRemover:
    def __init__(self, seperation_data: dict, logger: Optional[Logger] = None):
        # -Universal Data (Same for each file)-
        self.seperation_data = seperation_data
        self.general_data = {
            'total_loops': None,
            'folder_path': None,
            'file_add_on': None,
        }
        self.logger = logger
        self.models = {}
        self.devices = {}
        # Threads
        self.all_threads = []
        # -File Specific Data (Different for each file)-
        # Updated on every conversion or loop
        self.loop_data = {
            # File specific
            'file_base_name': None,
            'file_num': 0,
            # Loop specific
            'command_base_text': None,
            'loop_num': 0,
            'progress_step': 0.0,
            'music_file': None,
            'model_device': {
                'model': None,
                'device': None,
                'model_name': None,
            },
            'constants': {
                'sr': None,
                'hop_length': None,
                'window_size': None,
                'n_fft': None,
            },
            'X': None,
            'X_mag': None,
            'X_phase': None,
            'prediction': None,
            'sampling_rate': None,
            'wav_vocals': None,
            'wav_instrument': None,
            'y_spec': None,
            'v_spec': None,
            # Spectogram from last seperation
            'temp_spectogramm': None,
        }
        # Needed for embedded audio player (GUI)
        self.latest_instrumental_path: str
        self.latest_vocal_path: str

    def seperate_files(self):
        """
        Seperate all files
        """
        # Track time
        stime = time.perf_counter()
        self._check_for_valid_inputs()
        self._fill_general_data()
        self.all_threads = []

        for file_num, file_path in enumerate(self.seperation_data['input_paths'], start=1):
            if self.seperation_data['multithreading']:
                thread = threading.Thread(target=self._seperate, args=(file_path, file_num),
                                          daemon=True)
                thread.start()
                self.all_threads.append(thread)
            else:
                self._seperate(file_path,
                               file_num)

        for thread in self.all_threads:
            thread.join()
        # Free RAM
        torch.cuda.empty_cache()

        self.logger.info('Conversion(s) Completed and Saving all Files!')
        self.logger.info(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')

    def write_to_gui(self, text: Optional[str] = None, include_base_text: bool = True, progress_step: Optional[float] = None):
        """
        Update progress and/or write text to the command line

        A new line '\\n' will be automatically appended to the text
        """
        self.logger.info(text)
        print(text)

    def _fill_general_data(self):
        """
        Fill the data implemented in general_data
        """
        def get_folderPath_fileAddOn() -> Tuple[str, str]:
            """
            Get export path and text, whic hwill be appended on the music files name
            """
            file_add_on = ''
            if self.seperation_data['modelFolder']:
                # Model Test Mode selected
                # -Instrumental-
                if os.path.isfile(self.seperation_data['instrumentalModel']):
                    file_add_on += os.path.splitext(os.path.basename(self.seperation_data['instrumentalModel']))[0]
                # -Vocal-
                elif os.path.isfile(self.seperation_data['vocalModel']):
                    file_add_on += os.path.splitext(os.path.basename(self.seperation_data['vocalModel']))[0]
                # -Stack-
                if os.path.isfile(self.seperation_data['stackModel']):
                    file_add_on += '-' + os.path.splitext(os.path.basename(self.seperation_data['stackModel']))[0]

                # Generate paths
                folder_path = os.path.join(self.seperation_data['export_path'], file_add_on)
                file_add_on = f'_{file_add_on}'

                if not os.path.isdir(folder_path):
                    # Folder does not exist
                    os.mkdir(folder_path)
            else:
                # Not Model Test Mode selected
                folder_path = self.seperation_data['export_path']

            return folder_path, file_add_on

        def get_models_devices() -> list:
            """
            Return models and devices found
            """
            models = {}
            devices = {}

            # -Instrumental-
            if os.path.isfile(self.seperation_data['instrumentalModel']):
                device = torch.device('cpu')
                model = nets.CascadedASPPNet(self.seperation_data['n_fft'])
                model.load_state_dict(torch.load(self.seperation_data['instrumentalModel'],
                                                 map_location=device))
                if torch.cuda.is_available() and self.seperation_data['gpuConversion']:
                    device = torch.device('cuda:0')
                    model.to(device)

                models['instrumental'] = model
                devices['instrumental'] = device
            # -Vocal-
            elif os.path.isfile(self.seperation_data['vocalModel']):
                device = torch.device('cpu')
                model = nets.CascadedASPPNet(self.seperation_data['n_fft'])
                model.load_state_dict(torch.load(self.seperation_data['vocalModel'],
                                                 map_location=device))
                if torch.cuda.is_available() and self.seperation_data['gpuConversion']:
                    device = torch.device('cuda:0')
                    model.to(device)

                models['vocal'] = model
                devices['vocal'] = device
            # -Stack-
            if os.path.isfile(self.seperation_data['stackModel']):
                device = torch.device('cpu')
                model = nets.CascadedASPPNet(self.seperation_data['n_fft'])
                model.load_state_dict(torch.load(self.seperation_data['stackModel'],
                                                 map_location=device))
                if torch.cuda.is_available() and self.seperation_data['gpuConversion']:
                    device = torch.device('cuda:0')
                    model.to(device)

                models['stack'] = model
                devices['stack'] = device

            return models, devices

        def get_total_loops() -> int:
            """
            Determine how many loops the program will
            have to prepare for
            """
            if self.seperation_data['stackOnly']:
                # Stack Conversion Only
                total_loops = self.seperation_data['stackPasses']
            else:
                # 1 for the instrumental/vocal
                total_loops = 1
                # Add number of stack pass loops
                total_loops += self.seperation_data['stackPasses']

            return total_loops

        # -Get data-
        total_loops = get_total_loops()
        folder_path, file_add_on = get_folderPath_fileAddOn()
        self.logger.info('Loading models...')
        models, devices = get_models_devices()

        # -Set data-
        self.general_data['total_files'] = len(self.seperation_data['input_paths'])
        self.general_data['total_loops'] = total_loops
        self.general_data['folder_path'] = folder_path
        self.general_data['file_add_on'] = file_add_on
        self.models = models
        self.devices = devices

    def _check_for_valid_inputs(self):
        """
        Check if all inputs have been entered correctly.

        If errors are found, an exception is raised
        """
        # Check input paths
        if not len(self.seperation_data['input_paths']):
            # No music file specified
            raise Exception('No music file to seperate defined!')
        if (not isinstance(self.seperation_data['input_paths'], tuple) and
                not isinstance(self.seperation_data['input_paths'], list)):
            # Music file not specified in a list or tuple
            raise Exception('Please specify your music file path/s in a list or tuple!')
        for input_path in self.seperation_data['input_paths']:
            # Go through each music file
            if not os.path.isfile(input_path):
                # Invalid path
                raise Exception(f'Invalid music file! Please make sure that the file still exists or that the path is valid!\nPath: "{input_path}"')  # nopep8
        # Output path
        if (not os.path.isdir(self.seperation_data['export_path']) and
                not self.seperation_data['export_path'] == ''):
            # Export path either invalid or not specified
            raise Exception(f'Invalid export directory! Please make sure that the directory still exists or that the path is valid!\nPath: "{self.seperation_data["export_path"]}"')  # nopep8

        # Check models
        if not self.seperation_data['useModel'] in ['vocal', 'instrumental']:
            # Invalid 'useModel'
            raise Exception("Parameter 'useModel' has to be either 'vocal' or 'instrumental'")
        if not os.path.isfile(self.seperation_data[f"{self.seperation_data['useModel']}Model"]):
            # No or invalid instrumental/vocal model given
            # but model is needed
            raise Exception(f"Not specified or invalid model path for {self.seperation_data['useModel']} model!")
        if (self.seperation_data['stackOnly'] or
                self.seperation_data['stackPasses'] > 0):
            # First check if stacked model is needed
            if not os.path.isfile(self.seperation_data['stackModel']):
                # No or invalid stack model given
                # but model is needed
                raise Exception(f"Not specified or invalid model path for stacked model!")

    def _seperate(self, file_path: str, file_num: int):
        """
        Seperate given music file,
        file_num is used to determine progress
        """

        # -Update file specific variables-
        self.loop_data['file_num'] = file_num
        self.loop_data['music_file'] = file_path
        self.loop_data['file_base_name'] = self._get_file_base_name(file_path)

        for loop_num in range(self.general_data['total_loops']):
            self.loop_data['loop_num'] = loop_num
            # -Get loop specific variables-
            command_base_text = self._get_base_text()
            model_device = self._get_model_device_file()
            constants = self._get_constants(model_device['model_name'])
            # -Update loop specific variables
            self.loop_data['constants'] = constants
            self.loop_data['command_base_text'] = command_base_text
            self.loop_data['model_device'] = model_device

            # -Seperation-
            if not self.loop_data['loop_num']:
                # First loop
                self._load_wave_source()
            self._wave_to_spectogram()
            if self.seperation_data['postProcess']:
                # Postprocess
                self._post_process()
            self._inverse_stft_of_instrumentals_and_vocals()
            self._save_files()
        else:
            # End of seperation
            if self.seperation_data['outputImage']:
                self._save_mask()

        self.write_to_gui(text='Completed Seperation!\n',
                          progress_step=1)

    # -Data Getter Methods-
    def _get_base_text(self) -> str:
        """
        Determine the prefix text of the console
        """
        loop_add_on = ''
        if self.general_data['total_loops'] > 1:
            # More than one loop for conversion
            loop_add_on = f" ({self.loop_data['loop_num']+1}/{self.general_data['total_loops']})"

        return 'File {file_num}/{total_files}:{loop} '.format(file_num=self.loop_data['file_num'],
                                                              total_files=self.general_data['total_files'],
                                                              loop=loop_add_on)

    def _get_constants(self, model_name: str) -> dict:
        """
        Get the sr, hop_length, window_size, n_fft
        """
        if self.loop_data['loop_num'] == 0:
            # Instrumental/Vocal Model
            seperation_params = {
                'sr': self.seperation_data['sr_stacked'],
                'hop_length': self.seperation_data['hop_length_stacked'],
                'window_size': self.seperation_data['window_size_stacked'],
                'n_fft': self.seperation_data['n_fft_stacked'],
            }
        else:
            # Stacked model
            seperation_params = {
                'sr': self.seperation_data['sr'],
                'hop_length': self.seperation_data['hop_length'],
                'window_size': self.seperation_data['window_size'],
                'n_fft': self.seperation_data['n_fft'],
            }
        if self.seperation_data['customParameters']:
            # Typed constants are fixed
            return seperation_params

        # -Decode Model Name-
        text = model_name.replace('.pth', '')
        text_parts = text.split('_')[1:]
        for text_part in text_parts:
            if 'sr' in text_part:
                text_part = text_part.replace('sr', '')
                if text_part.isdecimal():
                    try:
                        seperation_params['sr'] = int(text_part)
                        continue
                    except ValueError:
                        # Cannot convert string to int
                        pass
            if 'hl' in text_part:
                text_part = text_part.replace('hl', '')
                if text_part.isdecimal():
                    try:
                        seperation_params['hop_length'] = int(text_part)
                        continue
                    except ValueError:
                        # Cannot convert string to int
                        pass
            if 'w' in text_part:
                text_part = text_part.replace('w', '')
                if text_part.isdecimal():
                    try:
                        seperation_params['window_size'] = int(text_part)
                        continue
                    except ValueError:
                        # Cannot convert string to int
                        pass
            if 'nf' in text_part:
                text_part = text_part.replace('nf', '')
                if text_part.isdecimal():
                    try:
                        seperation_params['n_fft'] = int(text_part)
                        continue
                    except ValueError:
                        # Cannot convert string to int
                        pass

        return seperation_params

    def _get_model_device_file(self) -> dict:
        """
        Get the used models and devices for this loop
        Also extract the model name and the music file
        which will be used
        """
        model_device = {
            'model': None,
            'device': None,
            'model_name': None,
        }
        if self.seperation_data['stackOnly']:
            # Stack Only Conversion
            if os.path.isfile(self.seperation_data['stackModel']):
                model_device['model'] = self.models['stack']
                model_device['device'] = self.devices['stack']
                model_device['model_name'] = os.path.basename(self.seperation_data['stackModel'])
            else:
                raise ValueError(f'Selected stack only model, however, stack model path file cannot be found\nPath: "{self.seperation_data["stackModel"]}"')  # nopep8
        elif not self.loop_data['loop_num']:
            # First Loop
            model_device['model'] = self.models[self.seperation_data['useModel']]
            model_device['device'] = self.devices[self.seperation_data['useModel']]
            model_device['model_name'] = os.path.basename(
                self.seperation_data[f'{self.seperation_data["useModel"]}Model'])
        else:
            # Every other iteration
            model_device['model'] = self.models['stack']
            model_device['device'] = self.devices['stack']
            model_device['model_name'] = os.path.basename(self.seperation_data['stackModel'])

        return model_device

    def _get_file_base_name(self, file_path: str) -> str:
        """
        Get the path infos for the given music file
        """
        return f"{self.loop_data['file_num']}_{os.path.splitext(os.path.basename(file_path))[0]}"

    # -Seperation Methods-
    def _load_wave_source(self):
        """
        Load the wave source
        """
        self.write_to_gui(text='Loading wave source...',
                          progress_step=0)
        try:
            X, sampling_rate = librosa.load(path=self.loop_data['music_file'],
                                            sr=self.loop_data['constants']['sr'],
                                            mono=False, dtype=np.float32,
                                            res_type=self.seperation_data['resType'])
        except audioread.NoBackendError:
            raise Exception(
                f'Invalid music file provided! Please check its validity.\nFile: "{self.loop_data["music_file"]}"')

        if X.ndim == 1:
            X = np.asarray([X, X])

        self.loop_data['X'] = X
        self.loop_data['sampling_rate'] = sampling_rate

    def _wave_to_spectogram(self):
        """
        Wave to spectogram
        """
        def preprocess(X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)

            return X_mag, X_phase

        def execute(X_mag_pad, roi_size, n_window, device, model, progrs_info: str = ''):
            model.eval()
            with torch.no_grad():
                preds = []
                bar_format = '{desc}    |{bar}{r_bar}'
                pbar = tqdm(range(n_window), bar_format=bar_format)

                for progrs, i in enumerate(pbar):
                    # Progress management
                    if progrs_info == '1/2':
                        progres_step = 0.1 + 0.35 * (progrs / n_window)
                    elif progrs_info == '2/2':
                        progres_step = 0.45 + 0.35 * (progrs / n_window)
                    else:
                        progres_step = 0.1 + 0.7 * (progrs / n_window)
                    self.write_to_gui(progress_step=progres_step)

                    progress = self._get_progress(progres_step)
                    text = f'{int(progress)} %'
                    if progress < 10:
                        text += ' '
                    pbar.set_description_str(text)

                    start = i * roi_size
                    X_mag_window = X_mag_pad[None, :, :,
                                             start:start + self.seperation_data['window_size']]
                    X_mag_window = torch.from_numpy(X_mag_window).to(device)

                    pred = model.predict(X_mag_window)

                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])

                pred = np.concatenate(preds, axis=2)

            return pred

        def inference(X_spec, device, model):
            X_mag, X_phase = preprocess(X_spec)

            coef = X_mag.max()
            X_mag_pre = X_mag / coef

            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame,
                                                          self.seperation_data['window_size'], model.offset)
            n_window = int(np.ceil(n_frame / roi_size))

            X_mag_pad = np.pad(
                X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred = execute(X_mag_pad, roi_size, n_window,
                           device, model)
            pred = pred[:, :, :n_frame]

            return pred * coef, X_mag, np.exp(1.j * X_phase)

        def inference_tta(X_spec, device, model):
            X_mag, X_phase = preprocess(X_spec)

            coef = X_mag.max()
            X_mag_pre = X_mag / coef

            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame,
                                                          self.seperation_data['window_size'], model.offset)
            n_window = int(np.ceil(n_frame / roi_size))

            X_mag_pad = np.pad(
                X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred = execute(X_mag_pad, roi_size, n_window,
                           device, model, progrs_info='1/2')
            pred = pred[:, :, :n_frame]

            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            X_mag_pad = np.pad(
                X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred_tta = execute(X_mag_pad, roi_size, n_window,
                               device, model, progrs_info='2/2')
            pred_tta = pred_tta[:, :, roi_size // 2:]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)

        self.write_to_gui(text='Stft of wave source...',
                          progress_step=0.1)

        if not self.loop_data['loop_num']:
            X = spec_utils.wave_to_spectrogram(wave=self.loop_data['X'],
                                               hop_length=self.seperation_data['hop_length'],
                                               n_fft=self.seperation_data['n_fft'])
        else:
            X = self.loop_data['temp_spectogramm']

        if self.seperation_data['tta']:
            prediction, X_mag, X_phase = inference_tta(X_spec=X,
                                                       device=self.loop_data['model_device']['device'],
                                                       model=self.loop_data['model_device']['model'])
        else:
            prediction, X_mag, X_phase = inference(X_spec=X,
                                                   device=self.loop_data['model_device']['device'],
                                                   model=self.loop_data['model_device']['model'])

        self.loop_data['prediction'] = prediction
        self.loop_data['X'] = X
        self.loop_data['X_mag'] = X_mag
        self.loop_data['X_phase'] = X_phase

    def _post_process(self):
        """
        Post process
        """
        self.write_to_gui(text='Post processing...',
                          progress_step=0.8)

        pred_inv = np.clip(self.loop_data['X_mag'] - self.loop_data['prediction'], 0, np.inf)
        prediction = spec_utils.mask_silence(self.loop_data['prediction'], pred_inv)

        self.loop_data['prediction'] = prediction

    def _inverse_stft_of_instrumentals_and_vocals(self):
        """
        Inverse stft of instrumentals and vocals
        """
        self.write_to_gui(text='Inverse stft of instruments and vocals...',
                          progress_step=0.85)

        y_spec = self.loop_data['prediction'] * self.loop_data['X_phase']
        v_spec = np.clip(self.loop_data['X_mag'] - self.loop_data['prediction'],
                         0, np.inf) * self.loop_data['X_phase']

        if self.loop_data['loop_num'] == (self.general_data['total_loops'] - 1):
            # Only compute wave on last loop
            wav_instrument = spec_utils.spectrogram_to_wave(y_spec,
                                                            hop_length=self.seperation_data['hop_length'])
            self.loop_data['wav_instrument'] = wav_instrument
        wav_vocals = spec_utils.spectrogram_to_wave(v_spec,
                                                    hop_length=self.seperation_data['hop_length'])
        self.loop_data['wav_vocals'] = wav_vocals

        # Needed for mask creation
        self.loop_data['y_spec'] = y_spec
        self.loop_data['v_spec'] = v_spec

        self.loop_data['temp_spectogramm'] = y_spec

    def _save_files(self):
        """
        Save the files
        """
        def get_vocal_instrumental_name() -> Tuple[str, str, str]:
            """
            Get vocal and instrumental file names and update the
            folder_path temporarily if needed
            """
            loop_num = self.loop_data['loop_num']
            total_loops = self.general_data['total_loops']
            file_base_name = self.loop_data['file_base_name']
            vocal_name = None
            instrumental_name = None
            folder_path = self.general_data['folder_path']

            # Get the Suffix Name
            if (not loop_num or
                    loop_num == (total_loops - 1)):  # First or Last Loop
                if self.seperation_data['stackOnly']:
                    if loop_num == (total_loops - 1):  # Last Loop
                        if not (total_loops - 1):  # Only 1 Loop
                            vocal_name = '(Vocals)'
                            instrumental_name = '(Instrumental)'
                        else:
                            vocal_name = '(Vocal_Final_Stacked_Output)'
                            instrumental_name = '(Instrumental_Final_Stacked_Output)'
                elif self.seperation_data['useModel'] == 'instrumental':
                    if not loop_num:  # First Loop
                        vocal_name = '(Vocals)'
                    if loop_num == (total_loops - 1):  # Last Loop
                        if not (total_loops - 1):  # Only 1 Loop
                            instrumental_name = '(Instrumental)'
                        else:
                            instrumental_name = '(Instrumental_Final_Stacked_Output)'
                elif self.seperation_data['useModel'] == 'vocal':
                    if not loop_num:  # First Loop
                        instrumental_name = '(Instrumental)'
                    if loop_num == (total_loops - 1):  # Last Loop
                        if not (total_loops - 1):  # Only 1 Loop
                            vocal_name = '(Vocals)'
                        else:
                            vocal_name = '(Vocals_Final_Stacked_Output)'
                if self.seperation_data['useModel'] == 'vocal':
                    # Reverse names
                    vocal_name, instrumental_name = instrumental_name, vocal_name
            elif self.seperation_data['saveAllStacked']:
                stacked_folder_name = file_base_name + ' Stacked Outputs'  # nopep8
                folder_path = os.path.join(folder_path, stacked_folder_name)

                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)

                if self.seperation_data['stackOnly']:
                    vocal_name = f'(Vocal_{loop_num}_Stacked_Output)'
                    instrumental_name = f'(Instrumental_{loop_num}_Stacked_Output)'
                elif (self.seperation_data['useModel'] == 'vocal' or
                        self.seperation_data['useModel'] == 'instrumental'):
                    vocal_name = f'(Vocals_{loop_num}_Stacked_Output)'
                    instrumental_name = f'(Instrumental_{loop_num}_Stacked_Output)'

                if self.seperation_data['useModel'] == 'vocal':
                    # Reverse names
                    vocal_name, instrumental_name = instrumental_name, vocal_name
            return vocal_name, instrumental_name, folder_path

        self.write_to_gui(text='Saving Files...',
                          progress_step=0.9)

        vocal_name, instrumental_name, folder_path = get_vocal_instrumental_name()

        # -Save files-
        instrumental_file_name = f"{self.loop_data['file_base_name']}_{instrumental_name}{self.general_data['file_add_on']}.wav"
        vocal_file_name = f"{self.loop_data['file_base_name']}_{vocal_name}{self.general_data['file_add_on']}.wav"
        self.latest_instrumental_path = os.path.join(folder_path,
                                                        instrumental_file_name)
        self.latest_vocal_path = os.path.join(folder_path,
                                                vocal_file_name)
        # Instrumental
        if (instrumental_name is not None and
                self.seperation_data['save_instrumentals']):

            sf.write(self.latest_instrumental_path,
                     self.loop_data['wav_instrument'].T, self.loop_data['sampling_rate'])
        # Vocal
        if (vocal_name is not None and
                self.seperation_data['save_vocals']):
            sf.write(self.latest_vocal_path,
                     self.loop_data['wav_vocals'].T, self.loop_data['sampling_rate'])

    def _save_mask(self):
        """
        Save output image
        """
        mask_path = os.path.join(self.general_data['folder_path'], self.loop_data['file_base_name'])
        with open('{}_Instruments.jpg'.format(mask_path), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(self.loop_data['y_spec'])
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_Vocals.jpg'.format(mask_path), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(self.loop_data['v_spec'])
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    # -Other Methods-
    def _get_progress(self, progress_step: Optional[float] = None) -> float:
        """
        Get current conversion progress in percent
        """
        if progress_step is not None:
            self.loop_data['progress_step'] = progress_step
        try:
            base = (100 / self.general_data['total_files'])
            progress = base * (self.loop_data['file_num'] - 1)
            progress += (base / self.general_data['total_loops']) * \
                (self.loop_data['loop_num'] + self.loop_data['progress_step'])
        except TypeError:
            # One data point not specified yet
            progress = 0

        return progress


class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        str: Time elapsed
    message
        str: Message to write to GUI
    progress
        int (0-100): Progress update
    error
        Tuple[str, str]:
            Index 0: Error Message
            Index 1: Detailed Message

    '''
    start = QtCore.Signal()
    finished = QtCore.Signal(str, tuple)
    message = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    error = QtCore.Signal(tuple)


class VocalRemoverWorker(VocalRemover, QtCore.QRunnable):
    '''
    Threaded Vocal Remover

    Only use in conjunction with GUI
    '''

    def __init__(self, logger, seperation_data: dict = {}):
        super(VocalRemoverWorker, self).__init__(seperation_data, logger=logger)
        super(VocalRemover, self).__init__(seperation_data, logger=logger)
        super(QtCore.QRunnable, self).__init__()
        self.signals = WorkerSignals()
        self.logger = logger
        self.seperation_data = seperation_data
        self.setAutoDelete(False)

    @ QtCore.Slot()
    def run(self):
        """
        Seperate files
        """
        stime = time.perf_counter()

        try:
            self.signals.start.emit()
            self.logger.info(msg='----- The seperation has started! -----')
            try:
                self.seperate_files()
            except RuntimeError:
                # Application was forcefully closed
                print('Application forcefully closed')
                return
        except Exception as e:
            self.logger.exception(msg='An Exception has occurred!')
            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
            message = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\nIf the issue is not clear, please contact the creator and attach a screenshot of the detailed message with the file and settings that caused it!'
            print(traceback_text)
            print(type(e).__name__, e)
            self.signals.error.emit([str(e), message])
            return
        elapsed_seconds = int(time.perf_counter() - stime)
        elapsed_time = str(dt.timedelta(seconds=elapsed_seconds))
        self.signals.finished.emit(elapsed_time, [self.latest_instrumental_path, self.latest_vocal_path])

    def write_to_gui(self, text: Optional[str] = None, include_base_text: bool = True, progress_step: Optional[float] = None):
        if text is not None:
            if include_base_text:
                # Include base text
                text = f"{self.loop_data['command_base_text']} {text}"
            self.signals.message.emit(text)

        if progress_step is not None:
            self.signals.progress.emit(self._get_progress(progress_step))

    def _save_files(self):
        """
        Also save files in temp location for in GUI audio playback
        """
        super()._save_files()
        if self.loop_data['loop_num'] == (self.general_data['total_loops'] - 1):  # Last loop
            sf.write(os.path.join(ResourcePaths.tempDir, self.latest_instrumental_path),
                     self.loop_data['wav_instrument'].T, self.loop_data['sampling_rate'])
            sf.write(os.path.join(ResourcePaths.tempDir, self.latest_vocal_path),
                     self.loop_data['wav_vocals'].T, self.loop_data['sampling_rate'])

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
from .lib import spec_utils
from .lib.model_param_init import ModelParameters
from .lib import nets
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
    'model': '',  # Path to instrumental (not needed if not used)
    'modelDataPath': '',  # Path to model parameters
    'isVocal': False,
    # Model Folder
    'modelFolder': False,  # Model Test Mode
    # Constants
    'window_size': 320,
    'deepExtraction': True,
    'aggressiveness': 0.02,
    'highEndProcess': "Mirroring",  # Model Test Mode
    # Allows to process multiple music files at once
    'multithreading': False,
    # What to save
    'save_instrumentals': True,
    'save_vocals': True,
}
valid_high_end_process = ['none', 'bypass', 'correlation', 'mirroring', 'mirroring2']


class VocalRemover:
    def __init__(self, seperation_data: dict, logger: Optional[Logger] = None):
        # -Universal Data (Same for each file)-
        self.seperation_data = seperation_data
        self.general_data = {
            'total_files': None,
            'folder_path': None,
            'file_add_on': None,
            'model': None,
            'device': None,
            'model_parameters': None,
        }
        self.logger = logger
        # Threads
        self.all_threads = []
        # -File Specific Data (Different for each file)-
        # Updated on every conversion or loop
        self.file_data = {
            # File specific
            'file_base_name': None,
            'file_path': None,
            'file_num': 0,
            'command_base_text': None,
            'progress_step': 0.0,  # From 0 to 1
            # Seperation
            'X_spec_m': None,
            'input_high_end': None,
            'input_high_end_h': None,
            'prediction': None,
            'X_mag': None,
            'X_phase': None,
            'y_spec_m': None,
            'v_spec_m': None,
            'wave_instrumentals': None,
            'wave_vocals': None,
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
            self._seperate(file_path,
                           file_num)
            # if self.seperation_data['multithreading']:
            #     thread = threading.Thread(target=self._seperate, args=(file_path, file_num),
            #                               daemon=True)
            #     thread.start()
            #     self.all_threads.append(thread)

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
            folder_path = self.seperation_data['export_path']
            file_add_on = ''

            if self.seperation_data['modelFolder']:
                # Model Test Mode selected
                model_name = os.path.splitext(os.path.basename(self.seperation_data['model']))[0]
                # Generate paths
                folder_path = os.path.join(self.seperation_data['export_path'], model_name)
                file_add_on = f'_{model_name}'
                print(folder_path)
                if not os.path.isdir(folder_path):
                    # Folder does not exist
                    os.mkdir(folder_path)

            return folder_path, file_add_on

        def get_model_device() -> list:
            """
            Return models and devices found
            """
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(self.general_data['model_parameters']['bins'] * 2)
            model.load_state_dict(torch.load(self.seperation_data['model'], map_location=device))
            if torch.cuda.is_available() and self.seperation_data['gpuConversion']:
                device = torch.device('cuda:0')
                model.to(device)

            return model, device

        # Need this for getting models so setting early
        self.general_data['model_parameters'] = ModelParameters(self.seperation_data['modelDataPath']).param

        # -Get data-
        folder_path, file_add_on = get_folderPath_fileAddOn()
        self.logger.info('Loading models...')
        model, device = get_model_device()

        # -Set data-
        self.general_data['total_files'] = len(self.seperation_data['input_paths'])
        self.general_data['folder_path'] = folder_path
        self.general_data['file_add_on'] = file_add_on
        self.general_data['model'] = model
        self.general_data['device'] = device

    def _check_for_valid_inputs(self):
        """
        Check if all inputs have been entered correctly.

        If errors are found, an exception is raised
        """
        self.seperation_data['highEndProcess'] = self.seperation_data['highEndProcess'].replace(" ", "").lower()
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
        try:
            if not os.path.isfile(self.seperation_data["model"]):
                raise Exception()
        except (Exception):
            model_type = "vocal" if self.seperation_data['isVocal'] else "instrumental"
            raise Exception(f"Please choose an {model_type} model!")
        # Check High End Process
        if self.seperation_data['highEndProcess'] not in valid_high_end_process:
            # No or invalid high end process
            raise Exception(
                f"Invalid high end process mode!\nValid modes: {valid_high_end_process}\nChosen mode: {self.seperation_data['highEndProcess']}")

    def _seperate(self, file_path: str, file_num: int):
        """
        Seperate given music file,
        file_num is used to determine progress
        """

        # -Update file specific variables-
        self.file_data['file_num'] = file_num
        self.file_data['file_path'] = file_path
        self.file_data['file_base_name'] = self._get_file_base_name(file_path)
        self.file_data['command_base_text'] = self._get_base_text()

        # -Seperation-
        self._load_wave_source()
        self._wave_to_spectogram()
        if self.seperation_data['postProcess']:
            # Postprocess
            self._post_process()
        self._inverse_stft_of_instrumentals_and_vocals()
        self._save_files()

        # End of seperation
        if self.seperation_data['outputImage']:
            self._save_mask()

        self.write_to_gui(text='Completed Seperation!\n',
                          progress_step=1)

    # -Data Getter Methods-
    def _get_file_base_name(self, file_path: str) -> str:
        """
        Get the path infos for the given music file
        """
        prefix = ""
        if self.general_data['total_files'] > 1:
            prefix = f"{self.file_data['file_num']}_"

        return f"{prefix}{os.path.splitext(os.path.basename(file_path))[0]}"

    def _get_base_text(self) -> str:
        """
        Determine the prefix text of the console
        """
        return 'File {file_num}/{total_files}: '.format(file_num=self.file_data['file_num'],
                                                        total_files=self.general_data['total_files'])

    # -Seperation Methods-
    def _load_wave_source(self):
        """
        Load the wave source
        """
        self.write_to_gui(text='Loading wave source...',
                          progress_step=0)
        try:
            X_wave, X_spec_s = {}, {}

            bands_n = len(self.general_data['model_parameters']['band'])

            for d in range(bands_n, 0, -1):
                bp = self.general_data['model_parameters']['band'][d]

                if d == bands_n:  # high-end band
                    X_wave[d], _ = librosa.load(self.file_data['file_path'], bp['sr'], False,
                                                dtype=np.float32, res_type=bp['res_type'])

                    if X_wave[d].ndim == 1:
                        X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
                else:  # lower bands
                    X_wave[d] = librosa.resample(X_wave[d+1], self.general_data['model_parameters']['band'][d+1]['sr'],
                                                 bp['sr'], res_type=bp['res_type'])

                X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                    X_wave[d], bp['hl'], bp['n_fft'], self.general_data['model_parameters']['mid_side'], self.general_data['model_parameters']['reverse'])

                if d == bands_n and self.seperation_data['highEndProcess'] != 'none':
                    input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + \
                        (self.general_data['model_parameters']['pre_filter_stop'] -
                         self.general_data['model_parameters']['pre_filter_start'])
                    input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

            X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.general_data['model_parameters'])

            del X_wave, X_spec_s
        except audioread.NoBackendError:
            raise Exception(
                f'Invalid music file provided! Please check its validity.\nAlso consider installing ffmpeg to fix this issue.\nFile: "{self.file_data["file_path"]}"')

        self.file_data['X_spec_m'] = X_spec_m
        self.file_data['input_high_end'] = input_high_end
        self.file_data['input_high_end_h'] = input_high_end_h

    def _wave_to_spectogram(self):
        """
        Wave to spectogram
        """
        def preprocess(X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)

            return X_mag, X_phase

        def execute(X_mag_pad, roi_size, n_window, progrs_info: str = ''):
            self.general_data['model'].eval()
            with torch.no_grad():
                preds = []

                for progrs, i in enumerate(range(n_window)):
                    # Progress management
                    if progrs_info == '1/2':
                        progres_step = 0.1 + 0.35 * (progrs / n_window)
                    elif progrs_info == '2/2':
                        progres_step = 0.45 + 0.35 * (progrs / n_window)
                    else:
                        progres_step = 0.1 + 0.7 * (progrs / n_window)
                    self.write_to_gui(progress_step=progres_step)

                    start = i * roi_size
                    X_mag_window = X_mag_pad[None, :, :, start:start + self.seperation_data['window_size']]
                    X_mag_window = torch.from_numpy(X_mag_window).to(self.general_data['device'])

                    aggressiveness_info = {'value': self.seperation_data['aggressiveness'],
                                           'split_bin': self.general_data['model_parameters']['band'][1]['crop_stop']}
                    pred = self.general_data['model'].predict(X_mag_window, aggressiveness_info)

                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])

                pred = np.concatenate(preds, axis=2)

            return pred

        def inference():
            X_mag, X_phase = preprocess(self.file_data['X_spec_m'])

            coef = X_mag.max()
            X_mag_pre = X_mag / coef

            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.seperation_data['window_size'],
                                                          self.general_data['model'].offset)
            n_window = int(np.ceil(n_frame / roi_size))

            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred = execute(X_mag_pad, roi_size, n_window)
            pred = pred[:, :, :n_frame]

            return pred * coef, X_mag, np.exp(1.j * X_phase)

        def inference_tta():
            X_mag, X_phase = preprocess(self.file_data['X_spec_m'])

            coef = X_mag.max()
            X_mag_pre = X_mag / coef

            n_frame = X_mag_pre.shape[2]
            pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.seperation_data['window_size'],
                                                          self.general_data['model'].offset)
            n_window = int(np.ceil(n_frame / roi_size))

            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred = execute(X_mag_pad, roi_size, n_window, '1/2')
            pred = pred[:, :, :n_frame]

            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

            pred_tta = execute(X_mag_pad, roi_size, n_window, '2/2')
            pred_tta = pred_tta[:, :, roi_size // 2:]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)

        self.write_to_gui(text='Stft of wave source...',
                          progress_step=0.1)

        if self.seperation_data['tta']:
            prediction, X_mag, X_phase = inference_tta()
        else:
            prediction, X_mag, X_phase = inference()

        self.file_data['prediction'] = prediction
        self.file_data['X_mag'] = X_mag
        self.file_data['X_phase'] = X_phase

    def _post_process(self):
        """
        Post process
        """
        self.write_to_gui(text='Post processing...',
                          progress_step=0.8)
        pred_inv = np.clip(self.file_data['X_mag'] - self.file_data['prediction'], 0, np.inf)
        prediction = spec_utils.mask_silence(self.file_data['prediction'], pred_inv)

        self.file_data['prediction'] = prediction

    def _inverse_stft_of_instrumentals_and_vocals(self):
        """
        Inverse stft of instrumentals and vocals
        """
        self.write_to_gui(text='Inverse stft of instruments and vocals...',
                          progress_step=0.85)

        y_spec_m = self.file_data['prediction'] * self.file_data['X_phase']
        v_spec_m = self.file_data['X_spec_m'] - y_spec_m

        # -Instrumental-
        if self.seperation_data['highEndProcess'] == 'bypass':
            wave_instrumentals = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.general_data['model_parameters'],
                                                                    self.file_data['input_high_end_h'], self.file_data['input_high_end'])
        elif self.seperation_data['highEndProcess'] == 'correlation':
            print('Deprecated: correlation will be removed in the final release. Please use the mirroring instead.')

            for i in range(self.file_data['input_high_end'].shape[2]):
                for c in range(2):
                    X_mag_max = np.amax(self.file_data['input_high_end'][c, :, i])
                    b1 = self.general_data['model_parameters']['pre_filter_start']-self.file_data['input_high_end_h']//2
                    b2 = self.general_data['model_parameters']['pre_filter_start']-1
                    if X_mag_max > 0 and np.sum(np.abs(v_spec_m[c, b1:b2, i])) / (b2 - b1) > 0.07:
                        y_mag = np.median(y_spec_m[c, b1:b2, i])
                        self.file_data['input_high_end'][c, :, i] = np.true_divide(self.file_data['input_high_end'][c, :, i], abs(
                            X_mag_max) / min(abs(y_mag * 4), abs(X_mag_max)))

            wave_instrumentals = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.general_data['model_parameters'],
                                                                    self.file_data['input_high_end_h'], self.file_data['input_high_end'])
        elif self.seperation_data['highEndProcess'].startswith('mirroring'):
            self.file_data['input_high_end'] = spec_utils.mirroring(self.seperation_data['highEndProcess'], y_spec_m,
                                                                    self.file_data['input_high_end'], self.general_data['model_parameters'])

            wave_instrumentals = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.general_data['model_parameters'],
                                                                    self.file_data['input_high_end_h'], self.file_data['input_high_end'])
        else:
            wave_instrumentals = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.general_data['model_parameters'])
        # -Vocal-
        if self.seperation_data['highEndProcess'].startswith('mirroring'):
            self.file_data['input_high_end'] = spec_utils.mirroring(self.seperation_data['highEndProcess'], v_spec_m,
                                                                    self.file_data['input_high_end'], self.general_data['model_parameters'])

            wave_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.general_data['model_parameters'],
                                                             self.file_data['input_high_end_h'], self.file_data['input_high_end'])
        else:
            wave_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.general_data['model_parameters'])

        self.file_data['y_spec_m'] = y_spec_m
        self.file_data['v_spec_m'] = v_spec_m

        if (self.seperation_data['isVocal']):
            wave_instrumentals, wave_vocals = wave_vocals, wave_instrumentals
        self.file_data['wave_instrumentals'] = wave_instrumentals
        self.file_data['wave_vocals'] = wave_vocals

    def _save_files(self):
        """
        Save the files
        """

        self.write_to_gui(text='Saving Files...',
                          progress_step=0.9)

        instrumental_file_name = f"{self.file_data['file_base_name']}_(Instrumentals){self.general_data['file_add_on']}.wav"
        vocal_file_name = f"{self.file_data['file_base_name']}_(Vocals){self.general_data['file_add_on']}.wav"
        # -Save files-
        self.latest_instrumental_path = os.path.join(self.general_data['folder_path'],
                                                     instrumental_file_name)
        self.latest_vocal_path = os.path.join(self.general_data['folder_path'],
                                              vocal_file_name)
        # Instrumental
        if self.seperation_data['save_instrumentals']:
            sf.write(self.latest_instrumental_path,
                     self.file_data['wave_instrumentals'], self.general_data['model_parameters']['sr'])
        # Vocal
        if self.seperation_data['save_vocals']:
            sf.write(self.latest_vocal_path,
                     self.file_data['wave_vocals'], self.general_data['model_parameters']['sr'])

    def _save_mask(self):
        """
        Save output image
        """
        mask_path = os.path.join(self.general_data['folder_path'], self.file_data['file_base_name'])
        with open('{}_Instrumentals.jpg'.format(mask_path), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(self.file_data['y_spec_m'])
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)
        with open('{}_Vocals.jpg'.format(mask_path), mode='wb') as f:
            image = spec_utils.spectrogram_to_image(self.file_data['v_spec_m'])
            _, bin_image = cv2.imencode('.jpg', image)
            bin_image.tofile(f)

    # -Other Methods-
    def _get_progress(self, progress_step: Optional[float] = None) -> float:
        """
        Get current conversion progress in percent
        """
        if progress_step is not None:
            self.file_data['progress_step'] = progress_step
        try:
            base = (100 / self.general_data['total_files'])
            progress = base * (self.file_data['file_num'] - 1) + (self.file_data['progress_step'] * 100)
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
            except RuntimeError as e:
                # Application was forcefully closed
                print('Application forcefully closed', e.args)
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
                text = f"{self.file_data['command_base_text']} {text}"
            self.signals.message.emit(text)

        if progress_step is not None:
            self.signals.progress.emit(self._get_progress(progress_step))

    def _save_files(self):
        """
        Also save files in temp location for in GUI audio playback
        """
        super()._save_files()
        sf.write(os.path.join(ResourcePaths.tempDir, self.latest_instrumental_path),
                 self.file_data['wave_instrumentals'], self.general_data['model_parameters']['sr'])
        sf.write(os.path.join(ResourcePaths.tempDir, self.latest_vocal_path),
                 self.file_data['wave_vocals'], self.general_data['model_parameters']['sr'])

import pprint
import argparse
import os

import cv2
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib_v4 import dataset
from lib_v4 import nets
from lib_v4 import spec_utils
import torch

# Command line text parsing and widget manipulation
from collections import defaultdict
import tkinter as tk
import traceback  # Error Message Recent Calls
import time  # Timer


class VocalRemover(object):

    def __init__(self, data, text_widget: tk.Text):
        self.data = data
        self.text_widget = text_widget
        self.models = defaultdict(lambda: None)
        self.devices = defaultdict(lambda: None)
        self._load_models()
        # self.offset = model.offset

    def _load_models(self):
        self.text_widget.write('Loading models...\n')  # nopep8 Write Command Text

        # -Instrumental-
        if os.path.isfile(data['instrumentalModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(self.data['n_fft'])
            model.load_state_dict(torch.load(self.data['instrumentalModel'],
                                             map_location=device))
            if torch.cuda.is_available() and self.data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(self.data['gpu']))
                model.to(device)

            self.models['instrumental'] = model
            self.devices['instrumental'] = device
        # -Vocal-
        elif os.path.isfile(data['vocalModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(self.data['n_fft'])
            model.load_state_dict(torch.load(self.data['vocalModel'],
                                             map_location=device))
            if torch.cuda.is_available() and self.data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(self.data['gpu']))
                model.to(device)

            self.models['vocal'] = model
            self.devices['vocal'] = device
        # -Stack-
        if os.path.isfile(self.data['stackModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(self.data['n_fft'])
            model.load_state_dict(torch.load(self.data['stackModel'],
                                             map_location=device))
            if torch.cuda.is_available() and self.data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(self.data['gpu']))
                model.to(device)

            self.models['stack'] = model
            self.devices['stack'] = device

        self.text_widget.write('Done!\n')

    def _execute(self, X_mag_pad, roi_size, n_window, device, model):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :,
                                         start:start + self.data['window_size']]
                X_mag_window = torch.from_numpy(X_mag_window).to(device)

                pred = model.predict(X_mag_window)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)

        return pred

    def preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def inference(self, X_spec, device, model):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame,
                                                      self.data['window_size'], model.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window,
                             device, model)
        pred = pred[:, :, :n_frame]

        return pred * coef, X_mag, np.exp(1.j * X_phase)

    def inference_tta(self, X_spec, device, model):
        X_mag, X_phase = self.preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame,
                                                      self.data['window_size'], model.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred = self._execute(X_mag_pad, roi_size, n_window,
                             device, model)
        pred = pred[:, :, :n_frame]

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = self._execute(X_mag_pad, roi_size, n_window,
                                 device, model)
        pred_tta = pred_tta[:, :, roi_size // 2:]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)


data = {
    # Paths
    'input_paths': None,
    'export_path': None,
    # Processing Options
    'gpu': -1,
    'postprocess': True,
    'tta': True,
    'output_image': True,
    # Models
    'instrumentalModel': None,
    'vocalModel': None,
    'stackModel': None,
    'useModel': None,
    # Stack Options
    'stackPasses': 0,
    'stackOnly': False,
    'saveAllStacked': False,
    # Model Folder
    'modelFolder': False,
    # Constants
    'sr': 44_100,
    'hop_length': 1_024,
    'window_size': 320,
    'n_fft': 2_048,
    # Resolution Type
    'resType': 'kaiser_fast',
    # Parsed constants should be fixed
    'manType': False,
}
default_sr = data['sr']
default_hop_length = data['hop_length']
default_window_size = data['window_size']
default_n_fft = data['n_fft']


def update_progress(progress_var, total_files, total_loops, file_num, loop_num, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    base = (100 / total_files)
    progress = base * (file_num - 1)
    progress += (base / total_loops) * (loop_num + step)

    progress_var.set(progress)


def get_baseText(total_files, total_loops, file_num, loop_num):
    """Create the base text for the command widget"""
    text = 'File {file_num}/{total_files}:{loop} '.format(file_num=file_num,
                                                          total_files=total_files,
                                                          loop='' if total_loops <= 1 else f' ({loop_num+1}/{total_loops})')
    return text


def update_constants(model_name):
    """
    Decode the conversion settings from the model's name
    """
    global data
    text = model_name.replace('.pth', '')
    text_parts = text.split('_')[1:]

    data['sr'] = default_sr
    data['hop_length'] = default_hop_length
    data['window_size'] = default_window_size
    data['n_fft'] = default_n_fft

    if data['manType']:
        # Default constants should be fixed
        return

    for text_part in text_parts:
        if 'sr' in text_part:
            text_part = text_part.replace('sr', '')
            if text_part.isdecimal():
                try:
                    data['sr'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'hl' in text_part:
            text_part = text_part.replace('hl', '')
            if text_part.isdecimal():
                try:
                    data['hop_length'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'w' in text_part:
            text_part = text_part.replace('w', '')
            if text_part.isdecimal():
                try:
                    data['window_size'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'nf' in text_part:
            text_part = text_part.replace('nf', '')
            if text_part.isdecimal():
                try:
                    data['n_fft'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass


def determineExportPath():
    """
    Determine the path, where the music file is stored
    """
    folder_path = data["export_path"]

    if data['modelFolder']:
        # Model Test Mode selected
        folder_name = ''
        # -Instrumental-
        if os.path.isfile(data['instrumentalModel']):
            folder_name += os.path.splitext(os.path.basename(data['instrumentalModel']))[0]
        # -Vocal-
        elif os.path.isfile(data['vocalModel']):
            folder_name += os.path.splitext(os.path.basename(data['vocalModel']))[0]
        # -Stack-
        if os.path.isfile(data['stackModel']):
            folder_name += '-' + os.path.splitext(os.path.basename(data['stackModel']))[0]

        # Add generated folder name to export Path
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            # Folder does not exist
            os.mkdir(folder_path)

    return folder_path


def main(window: tk.Wm, text_widget: tk.Text, button_widget: tk.Button, progress_var: tk.Variable,
         **kwargs: dict):
    def save_files(wav_instrument, wav_vocals):
        """Save output music files"""
        vocal_name = None
        instrumental_name = None
        save_path = os.path.dirname(base_name)

        # Get the Suffix Name
        if (not loop_num or
                loop_num == (total_loops - 1)):  # First or Last Loop
            if data['stackOnly']:
                if loop_num == (total_loops - 1):  # Last Loop
                    if not (total_loops - 1):  # Only 1 Loop
                        vocal_name = '(Vocals)'
                        instrumental_name = '(Instrumental)'
                    else:
                        vocal_name = '(Vocal_Final_Stacked_Output)'
                        instrumental_name = '(Instrumental_Final_Stacked_Output)'
            elif data['useModel'] == 'instrumental':
                if not loop_num:  # First Loop
                    vocal_name = '(Vocals)'
                if loop_num == (total_loops - 1):  # Last Loop
                    if not (total_loops - 1):  # Only 1 Loop
                        instrumental_name = '(Instrumental)'
                    else:
                        instrumental_name = '(Instrumental_Final_Stacked_Output)'
            elif data['useModel'] == 'vocal':
                if not loop_num:  # First Loop
                    instrumental_name = '(Instrumental)'
                if loop_num == (total_loops - 1):  # Last Loop
                    if not (total_loops - 1):  # Only 1 Loop
                        vocal_name = '(Vocals)'
                    else:
                        vocal_name = '(Vocals_Final_Stacked_Output)'
            if data['useModel'] == 'vocal':
                # Reverse names
                vocal_name, instrumental_name = instrumental_name, vocal_name
        elif data['saveAllStacked']:
            folder_name = os.path.basename(base_name) + ' Stacked Outputs'  # nopep8
            save_path = os.path.join(save_path, folder_name)

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            if data['stackOnly']:
                vocal_name = f'(Vocal_{loop_num}_Stacked_Output)'
                instrumental_name = f'(Instrumental_{loop_num}_Stacked_Output)'
            elif (data['useModel'] == 'vocal' or
                  data['useModel'] == 'instrumental'):
                vocal_name = f'(Vocals_{loop_num}_Stacked_Output)'
                instrumental_name = f'(Instrumental_{loop_num}_Stacked_Output)'

            if data['useModel'] == 'vocal':
                # Reverse names
                vocal_name, instrumental_name = instrumental_name, vocal_name

        # Save Temp File
        # For instrumental the instrumental is the temp file
        # and for vocal the instrumental is the temp file due
        # to reversement
        sf.write(f'temp.wav',
                 wav_instrument.T, sr)

        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            instrumental_path = os.path.join(save_path,
                                             f'{os.path.basename(base_name)}_{instrumental_name}_{modelFolderName}.wav')

            sf.write(instrumental_path,
                     wav_instrument.T, sr)
        # Vocal
        if vocal_name is not None:
            vocal_path = os.path.join(save_path,
                                      f'{os.path.basename(base_name)}_{vocal_name}_{modelFolderName}.wav')
            sf.write(vocal_path,
                     wav_vocals.T, sr)

    data.update(kwargs)

    # --Setup--
    # Update default settings
    global default_sr
    global default_hop_length
    global default_window_size
    global default_n_fft
    default_sr = data['sr']
    default_hop_length = data['hop_length']
    default_window_size = data['window_size']
    default_n_fft = data['n_fft']

    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button

    vocal_remover = VocalRemover(data, text_widget)
    folder_path = determineExportPath()

    # Determine Loops
    total_loops = data['stackPasses']
    if not data['stackOnly']:
        total_loops += 1
    for file_num, music_file in enumerate(data['input_paths'], start=1):
        # Determine File Name
        base_name = os.path.join(folder_path,
                                 f'{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}')
        try:
            # --Seperate Music Files--
            for loop_num in range(total_loops):
                # -Determine which model will be used-
                if not loop_num:
                    # First Iteration
                    if data['stackOnly']:
                        if os.path.isfile(data['stackModel']):
                            model_name = os.path.basename(data['stackModel'])
                            model = vocal_remover.models['stack']
                            device = vocal_remover.devices['stack']
                        else:
                            raise ValueError(f'Selected stack only model, however, stack model path file cannot be found\nPath: "{data["stackModel"]}"')  # nopep8
                    else:
                        model_name = os.path.basename(data[f'{data["useModel"]}Model'])
                        model = vocal_remover.models[data['useModel']]
                        device = vocal_remover.devices[data['useModel']]
                else:
                    model_name = os.path.basename(data['stackModel'])
                    # Every other iteration
                    model = vocal_remover.models['stack']
                    device = vocal_remover.devices['stack']
                    # Reference new music file
                    music_file = 'temp.wav'

                # -Get text and update progress-
                base_text = get_baseText(total_files=len(data['input_paths']),
                                         total_loops=total_loops,
                                         file_num=file_num,
                                         loop_num=loop_num)
                progress_kwargs = {'progress_var': progress_var,
                                   'total_files': len(data['input_paths']),
                                   'total_loops': total_loops,
                                   'file_num': file_num,
                                   'loop_num': loop_num}
                update_progress(**progress_kwargs,
                                step=0)
                update_constants(model_name)

                # -Go through the different steps of seperation-
                # Wave source
                text_widget.write(base_text + 'Loading wave source...\n')
                X, sr = librosa.load(music_file, data['sr'], False,
                                     dtype=np.float32, res_type=data['resType'])
                if X.ndim == 1:
                    X = np.asarray([X, X])
                text_widget.write(base_text + 'Done!\n')

                update_progress(**progress_kwargs,
                                step=0.1)
                # Stft of wave source
                text_widget.write(base_text + 'Stft of wave source...\n')
                X = spec_utils.wave_to_spectrogram(X,
                                                   data['hop_length'], data['n_fft'])
                if data['tta']:
                    pred, X_mag, X_phase = vocal_remover.inference_tta(X,
                                                                       device=device,
                                                                       model=model)
                else:
                    pred, X_mag, X_phase = vocal_remover.inference(X,
                                                                   device=device,
                                                                   model=model)
                text_widget.write(base_text + 'Done!\n')

                update_progress(**progress_kwargs,
                                step=0.6)
                # Postprocess
                if data['postprocess']:
                    text_widget.write(base_text + 'Post processing...\n')
                    pred_inv = np.clip(X_mag - pred, 0, np.inf)
                    pred = spec_utils.mask_silence(pred, pred_inv)
                    text_widget.write(base_text + 'Done!\n')

                    update_progress(**progress_kwargs,
                                    step=0.65)

                # Inverse stft
                text_widget.write(base_text + 'Inverse stft of instruments and vocals...\n')  # nopep8
                y_spec = pred * X_phase
                wav_instrument = spec_utils.spectrogram_to_wave(y_spec,
                                                                hop_length=data['hop_length'])
                v_spec = np.clip(X_mag - pred, 0, np.inf) * X_phase
                wav_vocals = spec_utils.spectrogram_to_wave(v_spec,
                                                            hop_length=data['hop_length'])
                text_widget.write(base_text + 'Done!\n')

                update_progress(**progress_kwargs,
                                step=0.7)
                # Save output music files
                text_widget.write(base_text + 'Saving Files...\n')
                save_files(wav_instrument, wav_vocals)
                text_widget.write(base_text + 'Done!\n')

                update_progress(**progress_kwargs,
                                step=0.8)
            else:
                # Save output image
                if data['output_image']:
                    with open('{}_Instruments.jpg'.format(base_name), mode='wb') as f:
                        image = spec_utils.spectrogram_to_image(y_spec)
                        _, bin_image = cv2.imencode('.jpg', image)
                        bin_image.tofile(f)
                    with open('{}_Vocals.jpg'.format(base_name), mode='wb') as f:
                        image = spec_utils.spectrogram_to_image(v_spec)
                        _, bin_image = cv2.imencode('.jpg', image)
                        bin_image.tofile(f)

            text_widget.write(base_text + 'Completed Seperation!\n\n')
        except Exception as e:
            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
            message = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\nFile: {music_file}\nLoop: {loop_num}\nPlease contact the creator and attach a screenshot of this error with the file and settings that caused it!'
            tk.messagebox.showerror(master=window,
                                    title='Untracked Error',
                                    message=message)
            print(traceback_text)
            print(type(e).__name__, e)
            print(message)
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return

        os.remove('temp.wav')
    progress_var.set(0)
    text_widget.write(f'Conversion(s) Completed and Saving all Files!\n')
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache()
    button_widget.configure(state=tk.NORMAL)  # Enable Button

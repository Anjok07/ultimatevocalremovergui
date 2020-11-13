import argparse
import os

import cv2
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib_v2 import dataset
from lib_v2 import nets
from lib_v2 import spec_utils

import torch
# Variable manipulation and command line text parsing
from collections import defaultdict
import tkinter as tk
import time  # Timer
import traceback  # Error Message Recent Calls


class Namespace:
    """
    Replaces ArgumentParser
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
    'window_size': 512,
    'n_fft': 2_048,
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

    # First set everything to default ->
    # If file name is not decodeable (invalid or no text_parts), constants stay at default
    data['sr'] = default_sr
    data['hop_length'] = default_hop_length
    data['window_size'] = default_window_size
    data['n_fft'] = default_n_fft

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


def determineModelFolderName():
    """
    Determine the name that is used for the folder and appended
    to the back of the music files
    """
    modelFolderName = ''
    if not data['modelFolder']:
        # Model Test Mode not selected
        return modelFolderName

    # -Instrumental-
    if os.path.isfile(data['instrumentalModel']):
        modelFolderName += os.path.splitext(os.path.basename(data['instrumentalModel']))[0] + '-'
    # -Vocal-
    elif os.path.isfile(data['vocalModel']):
        modelFolderName += os.path.splitext(os.path.basename(data['vocalModel']))[0] + '-'
    # -Stack-
    if os.path.isfile(data['stackModel']):
        modelFolderName += os.path.splitext(os.path.basename(data['stackModel']))[0]
    else:
        modelFolderName = modelFolderName[:-1]

    if modelFolderName:
        modelFolderName = '/' + modelFolderName

    return modelFolderName


def main(window: tk.Wm, text_widget: tk.Text, button_widget: tk.Button, progress_var: tk.Variable,
         **kwargs: dict):
    def load_models():
        text_widget.write('Loading models...\n')  # nopep8 Write Command Text
        models = defaultdict(lambda: None)
        devices = defaultdict(lambda: None)

        # -Instrumental-
        if os.path.isfile(data['instrumentalModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet()
            model.load_state_dict(torch.load(data['instrumentalModel'],
                                             map_location=device))
            if torch.cuda.is_available() and data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(data['gpu']))
                model.to(device)

            models['instrumental'] = model
            devices['instrumental'] = device
        # -Vocal-
        elif os.path.isfile(data['vocalModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet()
            model.load_state_dict(torch.load(data['vocalModel'],
                                             map_location=device))
            if torch.cuda.is_available() and data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(data['gpu']))
                model.to(device)

            models['vocal'] = model
            devices['vocal'] = device
        # -Stack-
        if os.path.isfile(data['stackModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet()
            model.load_state_dict(torch.load(data['stackModel'],
                                             map_location=device))
            if torch.cuda.is_available() and data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(data['gpu']))
                model.to(device)

            models['stack'] = model
            devices['stack'] = device

        text_widget.write('Done!\n')
        return models, devices

    def load_wave_source():
        X, sr = librosa.load(music_file,
                             data['sr'],
                             False,
                             dtype=np.float32,
                             res_type='kaiser_fast')

        return X, sr

    def stft_wave_source(X, model, device):
        X = spec_utils.calc_spec(X, data['hop_length'])
        X, phase = np.abs(X), np.exp(1.j * np.angle(X))
        coeff = X.max()
        X /= coeff

        offset = model.offset
        l, r, roi_size = dataset.make_padding(
            X.shape[2], data['window_size'], offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        X_roll = np.roll(X_pad, roi_size // 2, axis=2)

        model.eval()
        with torch.no_grad():
            masks = []
            masks_roll = []
            length = int(np.ceil(X.shape[2] / roi_size))
            for i in tqdm(range(length)):
                update_progress(**progress_kwargs,
                                step=0.1 + 0.5*(i/(length - 1)))
                start = i * roi_size
                X_window = torch.from_numpy(np.asarray([
                    X_pad[:, :, start:start + data['window_size']],
                    X_roll[:, :, start:start + data['window_size']]
                ])).to(device)
                pred = model.predict(X_window)
                pred = pred.detach().cpu().numpy()
                masks.append(pred[0])
                masks_roll.append(pred[1])

            mask = np.concatenate(masks, axis=2)[:, :, :X.shape[2]]
            mask_roll = np.concatenate(masks_roll, axis=2)[
                :, :, :X.shape[2]]
            mask = (mask + np.roll(mask_roll, -roi_size // 2, axis=2)) / 2

        if data['postprocess']:
            vocal = X * (1 - mask) * coeff
            mask = spec_utils.mask_uninformative(mask, vocal)

        inst = X * mask * coeff
        vocal = X * (1 - mask) * coeff

        return inst, vocal, phase, mask

    def invert_instrum_vocal(inst, vocal, phase):
        wav_instrument = spec_utils.spec_to_wav(inst, phase, data['hop_length'])  # nopep8
        wav_vocals = spec_utils.spec_to_wav(vocal, phase, data['hop_length'])  # nopep8

        return wav_instrument, wav_vocals

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

        appendModelFolderName = modelFolderName.replace('/', '_')
        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{instrumental_name}{appendModelFolderName}',
            )
            sf.write(instrumental_path,
                     wav_instrument.T, sr)
        # Vocal
        if vocal_name is not None:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{vocal_name}{appendModelFolderName}',
            )
            sf.write(vocal_path,
                     wav_vocals.T, sr)

    def output_image():
        norm_mask = np.uint8((1 - mask) * 255).transpose(1, 2, 0)
        norm_mask = np.concatenate([
            np.max(norm_mask, axis=2, keepdims=True),
            norm_mask], axis=2)[::-1]
        _, bin_mask = cv2.imencode('.png', norm_mask)
        text_widget.write(base_text + 'Saving Mask...\n')  # nopep8 Write Command Text
        with open(f'{base_name}_(Mask).png', mode='wb') as f:
            bin_mask.tofile(f)

    data.update(kwargs)

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

    models, devices = load_models()
    modelFolderName = determineModelFolderName()
    if modelFolderName:
        folder_path = f'{data["export_path"]}{modelFolderName}'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

    # Determine Loops
    total_loops = data['stackPasses']
    if not data['stackOnly']:
        total_loops += 1

    for file_num, music_file in enumerate(data['input_paths'], start=1):
        try:
            # Determine File Name
            base_name = f'{data["export_path"]}{modelFolderName}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'

            for loop_num in range(total_loops):
                # -Determine which model will be used-
                if not loop_num:
                    # First Iteration
                    if data['stackOnly']:
                        if os.path.isfile(data['stackModel']):
                            model_name = os.path.basename(data['stackModel'])
                            model = models['stack']
                            device = devices['stack']
                        else:
                            raise ValueError(f'Selected stack only model, however, stack model path file cannot be found\nPath: "{data["stackModel"]}"')  # nopep8
                    else:
                        model_name = os.path.basename(data[f'{data["useModel"]}Model'])
                        model = models[data['useModel']]
                        device = devices[data['useModel']]
                else:
                    model_name = os.path.basename(data['stackModel'])
                    # Every other iteration
                    model = models['stack']
                    device = devices['stack']
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
                text_widget.write(base_text + 'Loading wave source...\n')  # nopep8 Write Command Text
                X, sr = load_wave_source()
                text_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

                update_progress(**progress_kwargs,
                                step=0.1)
                # Stft of wave source
                text_widget.write(base_text + 'Stft of wave source...\n')  # nopep8 Write Command Text
                inst, vocal, phase, mask = stft_wave_source(X, model, device)
                text_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

                update_progress(**progress_kwargs,
                                step=0.6)
                # Inverse stft
                text_widget.write(base_text + 'Inverse stft of instruments and vocals...\n')  # nopep8 Write Command Text
                wav_instrument, wav_vocals = invert_instrum_vocal(inst, vocal, phase)  # nopep8
                text_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

                update_progress(**progress_kwargs,
                                step=0.7)
                # Save Files
                text_widget.write(base_text + 'Saving Files...\n')  # nopep8 Write Command Text
                save_files(wav_instrument, wav_vocals)
                text_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

                update_progress(**progress_kwargs,
                                step=0.8)

            else:
                # Save Output Image (Mask)
                if data['output_image']:
                    text_widget.write(base_text + 'Creating Mask...\n')  # nopep8 Write Command Text
                    output_image()
                    text_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

            text_widget.write(base_text + 'Completed Seperation!\n\n')  # nopep8 Write Command Text
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
    progress_var.set(0)  # Update Progress
    text_widget.write(f'Conversion(s) Completed and Saving all Files!\n')  # nopep8 Write Command Text
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    button_widget.configure(state=tk.NORMAL)  # Enable Button
import pprint
import argparse
import os
import importlib

import cv2
import librosa
import math
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib_v5 import dataset
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
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

        nn_arch_sizes = [
            31191, # default
            33966, 123821, 123812, 537238 # custom
        ]
        
        global args
        global model_params_d
                
        p = argparse.ArgumentParser()
        p.add_argument('--paramone', type=str, default='lib_v5/modelparams/4band_44100.json')
        p.add_argument('--paramtwo', type=str, default='lib_v5/modelparams/4band_v2.json')
        p.add_argument('--paramthree', type=str, default='lib_v5/modelparams/3band_44100_msb2.json')
        p.add_argument('--paramfour', type=str, default='lib_v5/modelparams/4band_v2_sn.json')
        p.add_argument('--aggressiveness',type=float, default=data['agg']/100)
        p.add_argument('--nn_architecture', type=str, choices= ['auto'] + list('{}KB'.format(s) for s in nn_arch_sizes), default='auto')
        p.add_argument('--high_end_process', type=str, default='mirroring')
        args = p.parse_args()  
        
        if 'auto' == args.nn_architecture:
            model_size = math.ceil(os.stat(data['instrumentalModel']).st_size / 1024)
            args.nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
        
        nets = importlib.import_module('lib_v5.nets' + f'_{args.nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
        
        ModelName=(data['instrumentalModel'])

        ModelParam1="4BAND_44100"
        ModelParam2="4BAND_44100_B"
        ModelParam3="MSB2"
        ModelParam4="4BAND_44100_SN"

        if ModelParam1 in ModelName:  
            model_params_d=args.paramone
        if ModelParam2 in ModelName:  
            model_params_d=args.paramtwo
        if ModelParam3 in ModelName:  
            model_params_d=args.paramthree
        if ModelParam4 in ModelName:  
            model_params_d=args.paramfour
            
        print(model_params_d)
        
        mp = ModelParameters(model_params_d)
        
        # -Instrumental-
        if os.path.isfile(data['instrumentalModel']):
            device = torch.device('cpu')
            model = nets.CascadedASPPNet(mp.param['bins'] * 2)
            model.load_state_dict(torch.load(self.data['instrumentalModel'],
                                             map_location=device))
            if torch.cuda.is_available() and self.data['gpu'] >= 0:
                device = torch.device('cuda:{}'.format(self.data['gpu']))
                model.to(device)

            self.models['instrumental'] = model
            self.devices['instrumental'] = device

        self.text_widget.write('Done!\n')

    def _execute(self, X_mag_pad, roi_size, n_window, device, model, aggressiveness):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start:start + self.data['window_size']]
                X_mag_window = torch.from_numpy(X_mag_window).to(device)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)

        return pred

    def preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def inference(self, X_spec, device, model, aggressiveness):
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
                             device, model, aggressiveness)
        pred = pred[:, :, :n_frame]

        return pred * coef, X_mag, np.exp(1.j * X_phase)

    def inference_tta(self, X_spec, device, model, aggressiveness):
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
                             device, model, aggressiveness)
        pred = pred[:, :, :n_frame]

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(
            X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        pred_tta = self._execute(X_mag_pad, roi_size, n_window,
                                 device, model, aggressiveness)
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
    'useModel': None,
    # Constants
    'window_size': 512,
    'agg': 10 
}

default_window_size = data['window_size']
default_agg = data['agg']

def update_progress(progress_var, total_files, file_num, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    base = (100 / total_files)
    progress = base * (file_num - 1)
    progress += step

    progress_var.set(progress)


def get_baseText(total_files, file_num):
    """Create the base text for the command widget"""
    text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                total_files=total_files)
    return text


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
        modelFolderName += os.path.splitext(os.path.basename(data['instrumentalModel']))[0]

    if modelFolderName:
        modelFolderName = '/' + modelFolderName

    return modelFolderName


def main(window: tk.Wm, text_widget: tk.Text, button_widget: tk.Button, progress_var: tk.Variable,
         **kwargs: dict):
    def save_files(wav_instrument, wav_vocals):
        """Save output music files"""
        vocal_name = '(Vocals)'
        instrumental_name = '(Instrumental)'
        save_path = os.path.dirname(base_name)

        # Swap names if vocal model

        VModel="Vocal"

        if VModel in model_name:  
            # Reverse names
            vocal_name, instrumental_name = instrumental_name, vocal_name

        # Save Temp File
        # For instrumental the instrumental is the temp file
        # and for vocal the instrumental is the temp file due
        # to reversement
        sf.write(f'temp.wav',
                 wav_instrument, mp.param['sr'])

        appendModelFolderName = modelFolderName.replace('/', '_')
        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{instrumental_name}{appendModelFolderName}',
            )
            
            sf.write(instrumental_path,
                     wav_instrument, mp.param['sr'])
        # Vocal
        if vocal_name is not None:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{vocal_name}{appendModelFolderName}',
            )
            sf.write(vocal_path,
                     wav_vocals, mp.param['sr'])

    data.update(kwargs)

    # Update default settings
    global default_window_size
    global default_agg
    default_window_size = data['window_size']
    default_agg = data['agg']

    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button

    vocal_remover = VocalRemover(data, text_widget)
    modelFolderName = determineModelFolderName()
    if modelFolderName:
        folder_path = f'{data["export_path"]}{modelFolderName}'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

    # Separation Preperation
    try:
            for file_num, music_file in enumerate(data['input_paths'], start=1):
                    # Determine File Name
                    base_name = f'{data["export_path"]}{modelFolderName}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                    # Start Separation
                    model_name = os.path.basename(data[f'{data["useModel"]}Model'])
                    model = vocal_remover.models[data['useModel']]
                    device = vocal_remover.devices[data['useModel']]

                    # -Get text and update progress-
                    base_text = get_baseText(total_files=len(data['input_paths']),
                                                file_num=file_num)
                    progress_kwargs = {'progress_var': progress_var,
                                        'total_files': len(data['input_paths']),
                                        'file_num': file_num}
                    update_progress(**progress_kwargs,
                                    step=0)
                    
                    mp = ModelParameters(model_params_d)
                        
                    # -Go through the different steps of seperation-
                    # Wave source
                    text_widget.write(base_text + 'Loading wave source...\n')
                    
                    X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
                    
                    bands_n = len(mp.param['band'])
                    
                    for d in range(bands_n, 0, -1):        
                        bp = mp.param['band'][d]
                    
                        if d == bands_n: # high-end band
                            X_wave[d], _ = librosa.load(
                                music_file, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                                
                            if X_wave[d].ndim == 1:
                                X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
                        else: # lower bands
                            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
                            
                        # Stft of wave source
                        
                        X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], 
                                                                        mp.param['mid_side_b2'], mp.param['reverse'])
                        
                        if d == bands_n and args.high_end_process != 'none':
                            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
                            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
                    
                    text_widget.write(base_text + 'Done!\n')

                    update_progress(**progress_kwargs,
                                    step=0.1)

                    text_widget.write(base_text + 'Stft of wave source...\n')
                    X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
                    
                    del X_wave, X_spec_s
                    
                    if data['tta']:
                        pred, X_mag, X_phase = vocal_remover.inference_tta(X_spec_m,
                                                                            device,
                                                                            model, {'value': args.aggressiveness,'split_bin': mp.param['band'][1]['crop_stop']})
                    else:
                        pred, X_mag, X_phase = vocal_remover.inference(X_spec_m,
                                                                        device,
                                                                        model, {'value': args.aggressiveness,'split_bin': mp.param['band'][1]['crop_stop']})

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
                    y_spec_m = pred * X_phase
                    v_spec_m = X_spec_m - y_spec_m
                    
                    if args.high_end_process.startswith('mirroring'):        
                        input_high_end_ = spec_utils.mirroring(args.high_end_process, y_spec_m, input_high_end, mp)
        
                        wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)       
                    else:
                        wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)

                    if args.high_end_process.startswith('mirroring'):        
                        input_high_end_ = spec_utils.mirroring(args.high_end_process, v_spec_m, input_high_end, mp)

                        wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp, input_high_end_h, input_high_end_)       
                    else:        
                        wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
                    
                    text_widget.write(base_text + 'Done!\n')

                    update_progress(**progress_kwargs,
                                    step=0.7)
                    # Save output music files
                    text_widget.write(base_text + 'Saving Files...\n')
                    save_files(wav_instrument, wav_vocals)
                    text_widget.write(base_text + 'Done!\n')

                    update_progress(**progress_kwargs,
                                    step=0.8)

                    # Save output image
                    if data['output_image']:
                        with open('{}_Instruments.jpg'.format(base_name), mode='wb') as f:
                            image = spec_utils.spectrogram_to_image(y_spec_m)
                            _, bin_image = cv2.imencode('.jpg', image)
                            bin_image.tofile(f)
                        with open('{}_Vocals.jpg'.format(base_name), mode='wb') as f:
                            image = spec_utils.spectrogram_to_image(v_spec_m)
                            _, bin_image = cv2.imencode('.jpg', image)
                            bin_image.tofile(f)

                    text_widget.write(base_text + 'Completed Seperation!\n\n')
    except Exception as e:
        traceback_text = ''.join(traceback.format_tb(e.__traceback__))
        message = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\nFile: {music_file}\nPlease contact the creator and attach a screenshot of this error with the file and settings that caused it!'
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
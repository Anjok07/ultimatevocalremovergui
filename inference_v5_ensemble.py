from functools import total_ordering
import pprint
import argparse
import os
from statistics import mode

import cv2
import librosa
import numpy as np
import soundfile as sf
import shutil
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
        # self.offset = model.offset

data = {
    # Paths
    'input_paths': None,
    'export_path': None,
    # Processing Options
    'gpu': -1,
    'postprocess': True,
    'tta': True,
    'save': True,
    'output_image': True,
    # Models
    'instrumentalModel': None,
    'useModel': None,
    # Constants
    'window_size': 512,
    'agg': 10,
    'ensChoose': 'HP1 Models'
}

default_window_size = data['window_size']
default_agg = data['agg']

def update_progress(progress_var, total_files, file_num, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    base = (100 / total_files)
    progress = base * (file_num - 1)
    progress += base * step

    progress_var.set(progress)

def get_baseText(total_files, file_num):
    """Create the base text for the command widget"""
    text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                total_files=total_files)
    return text

def main(window: tk.Wm, text_widget: tk.Text, button_widget: tk.Button, progress_var: tk.Variable,
         **kwargs: dict):
    
    global args
    global nn_arch_sizes

    nn_arch_sizes = [
        31191, # default
        33966, 123821, 123812, 537238 # custom
    ]
    
    p = argparse.ArgumentParser()
    p.add_argument('--aggressiveness',type=float, default=data['agg']/100)
    p.add_argument('--high_end_process', type=str, default='mirroring')
    args = p.parse_args()  
    
                    
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
        
        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_1}_{instrumental_name}',
            )
            
            sf.write(instrumental_path,
                     wav_instrument, mp.param['sr'])
        # Vocal
        if vocal_name is not None:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{ModelName_1}_{vocal_name}',
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

    # Separation Preperation
    try:    #Ensemble Dictionary
            HP1_Models = [
                {
                    'model_name':'HP_4BAND_44100_A',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP_4BAND_44100_A.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 1/2'
                },
                {
                    'model_name':'HP_4BAND_44100_B',
                    'model_params':'lib_v5/modelparams/4band_v2.json',
                    'model_location':'models/Main Models/HP_4BAND_44100_B.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 2/2'
                }
            ]
            
            HP2_Models = [
                {
                    'model_name':'HP2_4BAND_44100_1',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP2_4BAND_44100_1.pth',
                    'using_archtecture': '537238KB',
                    'loop_name': 'Ensemble Mode - Model 1/3'
                },
                {
                    'model_name':'HP2_4BAND_44100_2',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP2_4BAND_44100_2.pth',
                    'using_archtecture': '537238KB',
                    'loop_name': 'Ensemble Mode - Model 2/3'
                },
                {
                    'model_name':'HP2_3BAND_44100_MSB2',
                    'model_params':'lib_v5/modelparams/3band_44100_msb2.json',
                    'model_location':'models/Main Models/HP2_3BAND_44100_MSB2.pth',
                    'using_archtecture': '537227KB',
                    'loop_name': 'Ensemble Mode - Model 3/3'
                }
            ]
        
            All_HP_Models = [
                {
                    'model_name':'HP_4BAND_44100_A',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP_4BAND_44100_A.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 1/5'
                },
                {
                    'model_name':'HP_4BAND_44100_B',
                    'model_params':'lib_v5/modelparams/4band_v2.json',
                    'model_location':'models/Main Models/HP_4BAND_44100_B.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 2/5'
                },
                {
                    'model_name':'HP2_4BAND_44100_1',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP2_4BAND_44100_1.pth',
                    'using_archtecture': '537238KB',
                    'loop_name': 'Ensemble Mode - Model 3/5'
                    
                },
                {
                    'model_name':'HP2_4BAND_44100_2',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP2_4BAND_44100_2.pth',
                    'using_archtecture': '537238KB',
                    'loop_name': 'Ensemble Mode - Model 4/5'
                    
                },
                {
                    'model_name':'HP2_3BAND_44100_MSB2',
                    'model_params':'lib_v5/modelparams/3band_44100_msb2.json',
                    'model_location':'models/Main Models/HP2_3BAND_44100_MSB2.pth',
                    'using_archtecture': '537227KB',
                    'loop_name': 'Ensemble Mode - Model 5/5'
                }
            ]
            
            Vocal_Models = [
                {
                    'model_name':'HP_Vocal_4BAND_44100',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP_Vocal_4BAND_44100.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 1/2'
                },
                {
                    'model_name':'HP_Vocal_AGG_4BAND_44100',
                    'model_params':'lib_v5/modelparams/4band_44100.json',
                    'model_location':'models/Main Models/HP_Vocal_AGG_4BAND_44100.pth',
                    'using_archtecture': '123821KB',
                    'loop_name': 'Ensemble Mode - Model 2/2'
                }
            ]

            if data['ensChoose'] == 'HP1 Models':
                loops = HP1_Models
                ensefolder = 'HP_Models_Saved_Outputs'
                ensemode = 'HP_Models'
            if data['ensChoose'] == 'HP2 Models':
                loops = HP2_Models
                ensefolder = 'HP2_Models_Saved_Outputs'
                ensemode = 'HP2_Models'
            if data['ensChoose'] == 'All HP Models':
                loops = All_HP_Models
                ensefolder = 'All_HP_Models_Saved_Outputs'
                ensemode = 'All_HP_Models'
            if data['ensChoose'] == 'Vocal Models':           
                loops = Vocal_Models
                ensefolder = 'Vocal_Models_Saved_Outputs'
                ensemode = 'Vocal_Models'

            #Prepare Audiofile(s)
            for file_num, music_file in enumerate(data['input_paths'], start=1):
                # -Get text and update progress-
                base_text = get_baseText(total_files=len(data['input_paths']),
                                            file_num=file_num)
                progress_kwargs = {'progress_var': progress_var,
                                    'total_files': len(data['input_paths']),
                                    'file_num': file_num}
                update_progress(**progress_kwargs,
                                step=0)        

                #Prepare to loop models
                for i, c in tqdm(enumerate(loops), disable=True, desc='Iterations..'):
                    
                        text_widget.write(c['loop_name'] + '\n\n')
                        
                        text_widget.write(base_text + 'Loading ' + c['model_name'] + '... ')

                        arch_now = c['using_archtecture']

                        if arch_now == '123821KB':
                            from lib_v5 import nets_123821KB as nets
                        elif arch_now == '537238KB':
                            from lib_v5 import nets_537238KB as nets
                        elif arch_now == '537227KB':
                            from lib_v5 import nets_537227KB as nets
                        
                        def determineenseFolderName():
                            """
                            Determine the name that is used for the folder and appended
                            to the back of the music files
                            """
                            enseFolderName = ''

                            if str(ensefolder):
                                enseFolderName += os.path.splitext(os.path.basename(ensefolder))[0]

                            if enseFolderName:
                                enseFolderName = '/' + enseFolderName

                            return enseFolderName
                        
                        enseFolderName = determineenseFolderName()
                        if enseFolderName:
                            folder_path = f'{data["export_path"]}{enseFolderName}'
                            if not os.path.isdir(folder_path):
                                os.mkdir(folder_path)
                                       
                        # Determine File Name
                        base_name = f'{data["export_path"]}{enseFolderName}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                        enseExport = f'{data["export_path"]}{enseFolderName}/'
                        trackname = f'{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                        
                        ModelName_1=(c['model_name'])

                        print('Model Parameters:', c['model_params'])
                        
                        mp = ModelParameters(c['model_params'])
                        
                        #Load model
                        if os.path.isfile(c['model_location']):
                            device = torch.device('cpu')
                            model = nets.CascadedASPPNet(mp.param['bins'] * 2)
                            model.load_state_dict(torch.load(c['model_location'],
                                                            map_location=device))
                            if torch.cuda.is_available() and data['gpu'] >= 0:
                                device = torch.device('cuda:{}'.format(data['gpu']))
                                model.to(device)

                        text_widget.write('Done!\n')
                        
                        model_name = os.path.basename(c["model_name"])

                        # -Go through the different steps of seperation-
                        # Wave source
                        text_widget.write(base_text + 'Loading wave source... ')
                        
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
                        
                        text_widget.write('Done!\n')

                        update_progress(**progress_kwargs,
                                        step=0.1)

                        text_widget.write(base_text + 'Stft of wave source... ')
                        text_widget.write('Done!\n')
                        text_widget.write(base_text + "Please Wait...\n")

                        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
                        
                        del X_wave, X_spec_s
                        
                        def inference(X_spec, device, model, aggressiveness):
                            
                            def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness):
                                model.eval()
                                    
                                with torch.no_grad():
                                    preds = []
                                    
                                    iterations = [n_window]

                                    total_iterations = sum(iterations)
                                
                                    text_widget.write(base_text + "Processing "f"{total_iterations} Slices... ")
                                    
                                    for i in tqdm(range(n_window)): 
                                        update_progress(**progress_kwargs,
                                            step=(0.1 + (0.8/n_window * i)))
                                        start = i * roi_size
                                        X_mag_window = X_mag_pad[None, :, :, start:start + data['window_size']]
                                        X_mag_window = torch.from_numpy(X_mag_window).to(device)

                                        pred = model.predict(X_mag_window, aggressiveness)

                                        pred = pred.detach().cpu().numpy()
                                        preds.append(pred[0])
                                        
                                    pred = np.concatenate(preds, axis=2)
                                
                                    text_widget.write('Done!\n')
                                return pred
                            
                            def preprocess(X_spec):
                                X_mag = np.abs(X_spec)
                                X_phase = np.angle(X_spec)

                                return X_mag, X_phase
                            
                            X_mag, X_phase = preprocess(X_spec)

                            coef = X_mag.max()
                            X_mag_pre = X_mag / coef

                            n_frame = X_mag_pre.shape[2]
                            pad_l, pad_r, roi_size = dataset.make_padding(n_frame,
                                                                        data['window_size'], model.offset)
                            n_window = int(np.ceil(n_frame / roi_size))

                            X_mag_pad = np.pad(
                                X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
                            
                            pred = _execute(X_mag_pad, roi_size, n_window,
                                                device, model, aggressiveness)
                            pred = pred[:, :, :n_frame]
                            
                            if data['tta']:
                                pad_l += roi_size // 2
                                pad_r += roi_size // 2
                                n_window += 1

                                X_mag_pad = np.pad(
                                    X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

                                pred_tta = _execute(X_mag_pad, roi_size, n_window,
                                                        device, model, aggressiveness)
                                pred_tta = pred_tta[:, :, roi_size // 2:]
                                pred_tta = pred_tta[:, :, :n_frame]

                                return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)
                            else:
                                return pred * coef, X_mag, np.exp(1.j * X_phase)
                        
                        aggressiveness = {'value': args.aggressiveness, 'split_bin': mp.param['band'][1]['crop_stop']}
                        
                        if data['tta']:
                            text_widget.write(base_text + "Running Inferences (TTA)... \n")
                        else:
                            text_widget.write(base_text + "Running Inference... \n")
                        
                        pred, X_mag, X_phase = inference(X_spec_m,
                                                                device,
                                                                model, aggressiveness)
                        
                        update_progress(**progress_kwargs,
                                        step=0.85)
                        
                        # Postprocess
                        if data['postprocess']:
                            text_widget.write(base_text + 'Post processing... ')
                            pred_inv = np.clip(X_mag - pred, 0, np.inf)
                            pred = spec_utils.mask_silence(pred, pred_inv)
                            text_widget.write('Done!\n')

                            update_progress(**progress_kwargs,
                                            step=0.85)

                        # Inverse stft
                        text_widget.write(base_text + 'Inverse stft of instruments and vocals... ')  # nopep8 
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
                        
                        text_widget.write('Done!\n')

                        update_progress(**progress_kwargs,
                                        step=0.9)
                        
                        # Save output music files
                        text_widget.write(base_text + 'Saving Files... ')
                        save_files(wav_instrument, wav_vocals)
                        text_widget.write('Done!\n')

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
                        
                # Emsembling Outputs
                def get_files(folder="", prefix="", suffix=""):
                    return [f"{folder}{i}" for i in os.listdir(folder) if i.startswith(prefix) if i.endswith(suffix)]
                
                ensambles = [
                    {
                        'algorithm':'min_mag',
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                        'output':'{}_Ensembled_{}_(Instrumental)'.format(trackname, ensemode),
                        'type': 'Instrumentals'
                    },
                    {
                        'algorithm':'max_mag',
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                        'output': '{}_Ensembled_{}_(Vocals)'.format(trackname, ensemode),
                        'type': 'Vocals'
                    }
                ]

                for i, e in tqdm(enumerate(ensambles), desc="Ensembling..."):
                    
                    text_widget.write(base_text + "Ensembling " + e['type'] + "... ") 
                
                    wave, specs = {}, {}
                            
                    mp = ModelParameters(e['model_params'])
                    
                    for i in range(len(e['files'])):    
                        spec = {}
                        
                        for d in range(len(mp.param['band']), 0, -1):          
                            bp = mp.param['band'][d]            
                            
                            if d == len(mp.param['band']): # high-end band                
                                wave[d], _ = librosa.load(
                                    e['files'][i], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                                
                                if len(wave[d].shape) == 1: # mono to stereo
                                    wave[d] = np.array([wave[d], wave[d]])
                            else: # lower bands
                                wave[d] = librosa.resample(wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
                                    
                            spec[d] = spec_utils.wave_to_spectrogram(wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse'])
                            
                        specs[i] = spec_utils.combine_spectrograms(spec, mp)
                    
                    del wave    

                    sf.write(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])), 
                             spec_utils.cmb_spectrogram_to_wave(spec_utils.ensembling(e['algorithm'], 
                                                                            specs), mp), mp.param['sr'])
                    
                    if not data['save']: # Deletes all outputs if Save All Outputs: is checked
                            files = e['files']
                            for file in files:
                                os.remove(file)

                    text_widget.write("Done!\n")

                    update_progress(**progress_kwargs,
                    step=0.95)
                text_widget.write("\n")
           
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
        button_widget.configure(state=tk.NORMAL)  #Enable Button
        return
            
    if len(os.listdir(enseExport)) == 0: #Check if the folder is empty
        shutil.rmtree(folder_path) #Delete folder if empty
        
    update_progress(**progress_kwargs,
    step=1) 
        
    print('Done!')
        
    os.remove('temp.wav')

    progress_var.set(0)
    text_widget.write(f'Conversions Completed!\n')
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache()
    button_widget.configure(state=tk.NORMAL)  #Enable Button

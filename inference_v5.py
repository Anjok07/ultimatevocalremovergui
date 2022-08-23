from collections import defaultdict
from datetime import datetime
from demucs.apply import BagOfModels, apply_model
from demucs.hdemucs import HDemucs
from demucs.pretrained import get_model as _gm
from lib_v5 import dataset
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
from models import stft, istft
from pathlib import Path
from random import randrange
from tqdm import tqdm
from tkinter import filedialog
import lib_v5.filelist
import cv2
import hashlib
import importlib
import librosa
import math
import numpy as np
import os
import pydub
import shutil
import soundfile as sf
import time  # Timer
import tkinter as tk
import torch
import traceback  # Error Message Recent Calls

class VocalRemover(object):
    
    def __init__(self, data, text_widget: tk.Text):
        self.data = data
        self.text_widget = text_widget
        self.models = defaultdict(lambda: None)
        self.devices = defaultdict(lambda: None)
        # self.offset = model.offset
        
data = {
    'agg': 10,
    'demucsmodel_sel_VR': 'UVR_Demucs_Model_1',
    'demucsmodelVR': True,
    'export_path': None,
    'gpu': -1,
    'high_end_process': 'mirroring',
    'input_paths': None,
    'inst_only': False,
    'instrumentalModel': None,
    'ModelParams': 'Auto',
    'mp3bit': '320k',
    'normalize': False,
    'output_image': True,
    'overlap': 0.5,
    'postprocess': True,
    'saveFormat': 'wav',
    'segment': 'None',
    'settest': False,
    'shifts': 0,
    'split_mode': False,
    'tta': True,
    'useModel': None,
    'voc_only': False,
    'wavtype': 'PCM_16',
    'window_size': 512,
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
    
    global gui_progress_bar
    global nn_arch_sizes
    global nn_architecture
    
    global overlap_set
    global shift_set
    global split_mode
    global demucs_model_set
    global wav_type_set
    
    global flac_type_set
    global mp3_bit_set
    global space
    
    wav_type_set = data['wavtype']
    gui_progress_bar = progress_var
    #Error Handling
    
    runtimeerr = "CUDNN error executing cudnnSetTensorNdDescriptor"
    systemmemerr = "DefaultCPUAllocator: not enough memory"
    cuda_err = "CUDA out of memory"
    mod_err = "ModuleNotFoundError"
    file_err = "FileNotFoundError"
    ffmp_err = """audioread\__init__.py", line 116, in audio_open"""
    sf_write_err = "sf.write"
    demucs_model_missing_err = "is neither a single pre-trained model or a bag of models."
    
    try:
        with open('errorlog.txt', 'w') as f:
            f.write(f'No errors to report at this time.' + f'\n\nLast Process Method Used: VR Architecture' +
                    f'\nLast Conversion Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
    except:
        pass
    
    nn_arch_sizes = [
        31191, # default
        33966, 123821, 123812, 129605, 537238 # custom
    ]
    
    nn_architecture = list('{}KB'.format(s) for s in nn_arch_sizes)

       
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
        if data['demucsmodelVR']:
            samplerate = 44100
        else:
            samplerate = mp.param['sr']
            
            
        sf.write(f'temp.wav',
                 normalization_set(wav_instrument).T, samplerate, subtype=wav_type_set)

        appendModelFolderName = modelFolderName.replace('/', '_')
        
        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            if data['modelFolder']:
                instrumental_path = '{save_path}/{file_name}.wav'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{instrumental_name}',)
                instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{instrumental_name}',)
                instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{instrumental_name}',)
            else:
                instrumental_path = '{save_path}/{file_name}.wav'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{instrumental_name}',)
                instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{instrumental_name}',)
                instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{instrumental_name}',)
             
        if os.path.isfile(instrumental_path):
            file_exists_i = 'there'
        else:
            file_exists_i = 'not_there'
             
        if VModel in model_name and data['voc_only']:
                sf.write(instrumental_path,
                        normalization_set(wav_instrument).T, samplerate, subtype=wav_type_set)
        elif VModel in model_name and data['inst_only']:
            pass
        elif data['voc_only']:
            pass
        else:
                sf.write(instrumental_path,
                        normalization_set(wav_instrument).T, samplerate, subtype=wav_type_set)
                
        # Vocal
        if vocal_name is not None:
            if data['modelFolder']:
                vocal_path = '{save_path}/{file_name}.wav'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{vocal_name}',)
                vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{vocal_name}',)
                vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}{appendModelFolderName}_{vocal_name}',)
            else:
                vocal_path = '{save_path}/{file_name}.wav'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{vocal_name}',)
                vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{vocal_name}',)
                vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                    save_path=save_path,
                    file_name=f'{os.path.basename(base_name)}_{vocal_name}',)
            
            if os.path.isfile(vocal_path):
                file_exists_v = 'there'
            else:
                file_exists_v = 'not_there'

            if VModel in model_name and data['inst_only']:
                sf.write(vocal_path,
                            normalization_set(wav_vocals).T, samplerate, subtype=wav_type_set)
            elif VModel in model_name and data['voc_only']:
                pass
            elif data['inst_only']:
                pass
            else:
                sf.write(vocal_path,
                            normalization_set(wav_vocals).T, samplerate, subtype=wav_type_set)
        
            if data['saveFormat'] == 'Mp3':
                try:
                    if data['inst_only'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(vocal_path)
                        musfile.export(vocal_path_mp3, format="mp3", bitrate=mp3_bit_set)
                        if file_exists_v == 'there':
                            pass
                        else:
                            try:
                                os.remove(vocal_path)
                            except:
                                pass
                    if data['voc_only'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(instrumental_path)
                        musfile.export(instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set)
                        if file_exists_i == 'there':
                            pass
                        else:
                            try:
                                os.remove(instrumental_path)
                            except:
                                pass   
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        text_widget.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                        text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        text_widget.write(base_text + 'Moving on...\n')
                    else:
                        text_widget.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                        text_widget.write(base_text + 'Please check error log.\n')
                        text_widget.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as mp3 "{os.path.basename(music_file)}":\n' + 
                                    f'Process Method: VR Architecture\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass
                
            if data['saveFormat'] == 'Flac':
                try:
                    if VModel in model_name:
                        if data['inst_only'] == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(instrumental_path)
                            musfile.export(instrumental_path_flac, format="flac") 
                            if file_exists_v == 'there':
                                pass
                            else:
                                try:
                                    os.remove(instrumental_path)
                                except:
                                    pass
                        if data['voc_only'] == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(vocal_path)
                            musfile.export(vocal_path_flac, format="flac")  
                            if file_exists_i == 'there':
                                pass
                            else:
                                try:
                                    os.remove(vocal_path)
                                except:
                                    pass  
                    else:
                        if data['inst_only'] == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(vocal_path)
                            musfile.export(vocal_path_flac, format="flac") 
                            if file_exists_v == 'there':
                                pass
                            else:
                                try:
                                    os.remove(vocal_path)
                                except:
                                    pass
                        if data['voc_only'] == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(instrumental_path)
                            musfile.export(instrumental_path_flac, format="flac")  
                            if file_exists_i == 'there':
                                pass
                            else:
                                try:
                                    os.remove(instrumental_path)
                                except:
                                    pass        
                except Exception as e:
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                    if "ffmpeg" in errmessage:
                        text_widget.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                        text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                        text_widget.write(base_text + 'Moving on...\n')
                    else:
                        text_widget.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                        text_widget.write(base_text + 'Please check error log.\n')
                        text_widget.write(base_text + 'Moving on...\n')
                    try:
                        with open('errorlog.txt', 'w') as f:
                            f.write(f'Last Error Received:\n\n' +
                                    f'Error Received while attempting to save file as flac "{os.path.basename(music_file)}":\n' + 
                                    f'Process Method: VR Architecture\n\n' +
                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                    f'If this error persists, please contact the developers.\n\n' + 
                                    f'Raw error details:\n\n' +
                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                    except:
                        pass
           
           
    data.update(kwargs)

    # Update default settings
    global default_window_size
    global default_agg
    global normalization_set
    global update_prog
    
    update_prog = update_progress
    default_window_size = data['window_size']
    default_agg = data['agg']
    space = ' '*90

    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button

    overlap_set = float(data['overlap'])
    shift_set = int(data['shifts'])
    demucs_model_set = data['demucsmodel_sel_VR']
    split_mode = data['split_mode']
    
    if data['wavtype'] == '32-bit Float':
        wav_type_set = 'FLOAT'
    elif data['wavtype'] == '64-bit Float':
        wav_type_set = 'DOUBLE'
    else:
        wav_type_set = data['wavtype']
        
    flac_type_set = data['flactype']
    mp3_bit_set = data['mp3bit']
    
    if data['normalize'] == True:
        normalization_set = spec_utils.normalize
        print('normalization on')
    else:
        normalization_set = spec_utils.nonormalize
        print('normalization off')

    vocal_remover = VocalRemover(data, text_widget)
    modelFolderName = determineModelFolderName()

    timestampnum = round(datetime.utcnow().timestamp())
    randomnum = randrange(100000, 1000000)

    # Separation Preperation
    try:        #Load File(s)
                for file_num, music_file in enumerate(data['input_paths'], start=1):
                        # Determine File Name
                        m=music_file
                        
                        if data['settest']:
                            try:
                                base_name = f'{data["export_path"]}/{str(timestampnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                            except:
                                base_name = f'{data["export_path"]}/{str(randomnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                        else:
                            base_name = f'{data["export_path"]}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                        
                        global inference_type
                        
                        inference_type = 'inference_vr'
                        model_name = os.path.basename(data[f'{data["useModel"]}Model'])
                        model = vocal_remover.models[data['useModel']]
                        device = vocal_remover.devices[data['useModel']]
                        # -Get text and update progress-
                        base_text = get_baseText(total_files=len(data['input_paths']),
                                                    file_num=file_num)
                        progress_kwargs = {'progress_var': progress_var,
                                        'total_files': len(data['input_paths']),
                                        'file_num': file_num}
                        progress_demucs_kwargs = {'total_files': len(data['input_paths']),
                                        'file_num': file_num, 'inference_type': inference_type}
                        update_progress(**progress_kwargs,
                                        step=0)        
                
                        try:
                            total, used, free = shutil.disk_usage("/") 
                                
                            total_space = int(total/1.074e+9)
                            used_space = int(used/1.074e+9)
                            free_space = int(free/1.074e+9)
                                
                            if int(free/1.074e+9) <= int(2):
                                text_widget.write('Error: Not enough storage on main drive to continue. Your main drive must have \nat least 3 GB\'s of storage in order for this application function properly. \n\nPlease ensure your main drive has at least 3 GB\'s of storage and try again.\n\n')
                                text_widget.write('Detected Total Space: ' + str(total_space) + ' GB' + '\n')
                                text_widget.write('Detected Used Space: ' + str(used_space) + ' GB' + '\n')
                                text_widget.write('Detected Free Space: ' + str(free_space) + ' GB' + '\n')
                                progress_var.set(0)
                                button_widget.configure(state=tk.NORMAL)  # Enable Button
                                return 
                            
                            if int(free/1.074e+9) in [3, 4, 5, 6, 7, 8]:
                                text_widget.write('Warning: Your main drive is running low on storage. Your main drive must have \nat least 3 GB\'s of storage in order for this application function properly.\n\n')
                                text_widget.write('Detected Total Space: ' + str(total_space) + ' GB' + '\n')
                                text_widget.write('Detected Used Space: ' + str(used_space) + ' GB' + '\n')
                                text_widget.write('Detected Free Space: ' + str(free_space) + ' GB' + '\n\n')
                        except:
                            pass
                        
                        if data['wavtype'] == '64-bit Float':
                            if data['saveFormat'] == 'Flac':
                                text_widget.write('Please select \"WAV\" as your save format to use 64-bit Float.\n')
                                text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                                progress_var.set(0)
                                button_widget.configure(state=tk.NORMAL)  # Enable Button
                                return 
                            
                        if data['wavtype'] == '64-bit Float':
                            if data['saveFormat'] == 'Mp3':
                                text_widget.write('Please select \"WAV\" as your save format to use 64-bit Float.\n')
                                text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                                progress_var.set(0)
                                button_widget.configure(state=tk.NORMAL)  # Enable Button
                                return 
                
                        #Load Model      
                        text_widget.write(base_text + 'Loading model...')
                        
                        model_size = math.ceil(os.stat(data['instrumentalModel']).st_size / 1024)
                        nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
                        
                        nets = importlib.import_module('lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
                        
                        aggresive_set = float(data['agg']/100)
                        
                        ModelName=(data['instrumentalModel'])

                        #Package Models
                        text_widget.write('Done!\n')
                        
                        if data['ModelParams'] == 'Auto':
                            model_hash = hashlib.md5(open(ModelName,'rb').read()).hexdigest()
                            model_params = []   
                            model_params = lib_v5.filelist.provide_model_param_hash(model_hash)
                            #print(model_params)
                            if model_params[0] == 'Not Found Using Hash':
                                model_params = []   
                                model_params = lib_v5.filelist.provide_model_param_name(ModelName)
                            if model_params[0] == 'Not Found Using Name':
                                text_widget.write(base_text + f'Unable to set model parameters automatically with the selected model.\n')
                                confirm = tk.messagebox.askyesno(title='Unrecognized Model Detected',
                                        message=f'\nThe application could not automatically set the model param for the selected model.\n\n' + 
                                        f'Would you like to select the model param file for this model?\n\n')
                                
                                if confirm:
                                    model_param_selection = filedialog.askopenfilename(initialdir='lib_v5/modelparams', 
                                                                            title=f'Select Model Param', 
                                                                            filetypes=[("Model Param", "*.json")])
                                    
                                    model_param_file_path = str(model_param_selection)
                                    model_param_file = os.path.splitext(os.path.basename(model_param_file_path))[0] + '.json'
                                    model_params = [model_param_file_path, model_param_file]
                                    
                                    with open(f"lib_v5/filelists/model_cache/vr_param_cache/{model_hash}.txt", 'w') as f:
                                        f.write(model_param_file)
                                        
                                    
                                    if model_params[0] == '':
                                        text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
                                        text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
                                        text_widget.write(f'\nError Received:\n\n')
                                        text_widget.write(f'Model parameters are missing.\n\n')
                                        text_widget.write(f'Please check the following:\n')
                                        text_widget.write(f'1. Make sure the model is still present.\n')
                                        text_widget.write(f'2. If you are running a model that was not originally included in this package, \nplease append the modelparam name to the model name.\n')
                                        text_widget.write(f'  - Example if using \"4band_v2.json\" modelparam: \"model_4band_v2.pth\"\n\n')
                                        text_widget.write(f'Please address this and try again.\n\n')
                                        text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                                        torch.cuda.empty_cache()
                                        progress_var.set(0)
                                        button_widget.configure(state=tk.NORMAL)  # Enable Button
                                        return
                                        
                                    else:
                                        pass
                                else:
                                    text_widget.write(base_text + f'Model param not selected.\n')
                                    text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
                                    text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
                                    text_widget.write(f'\nError Received:\n\n')
                                    text_widget.write(f'Model parameters are missing.\n\n')
                                    text_widget.write(f'Please check the following:\n')
                                    text_widget.write(f'1. Make sure the model is still present.\n')
                                    text_widget.write(f'2. If you are running a model that was not originally included in this package, \nplease append the modelparam name to the model name.\n')
                                    text_widget.write(f'  - Example if using \"4band_v2.json\" modelparam: \"model_4band_v2.pth\"\n\n')
                                    text_widget.write(f'Please address this and try again.\n\n')
                                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                                    torch.cuda.empty_cache()
                                    progress_var.set(0)
                                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                                    return

                        else:
                            param = data['ModelParams']
                            model_param_file_path = f'lib_v5/modelparams/{param}'
                            model_params = [model_param_file_path, param]
                        
                        text_widget.write(base_text + 'Loading assigned model parameters ' + '\"' + model_params[1] + '\"... ')
                        mp = ModelParameters(model_params[0])
                        text_widget.write('Done!\n')
                        # -Instrumental-
                        if os.path.isfile(data['instrumentalModel']):
                            device = torch.device('cpu')
                            model = nets.CascadedASPPNet(mp.param['bins'] * 2)
                            model.load_state_dict(torch.load(data['instrumentalModel'],
                                                            map_location=device))
                            if torch.cuda.is_available() and data['gpu'] >= 0:
                                device = torch.device('cuda:{}'.format(data['gpu']))
                                model.to(device)
                                
                            vocal_remover.models['instrumental'] = model
                            vocal_remover.devices['instrumental'] = device

                        
                        model_name = os.path.basename(data[f'{data["useModel"]}Model'])
                            
                        # -Go through the different steps of Separation-
                        # Wave source
                        text_widget.write(base_text + 'Loading audio source...')
                        
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
                            
                            if d == bands_n and data['high_end_process'] != 'none':
                                input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
                                input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
                        
                        text_widget.write('Done!\n')

                        update_progress(**progress_kwargs,
                                        step=0.1)

                        text_widget.write(base_text + 'Loading the stft of audio source...')
                        
                        text_widget.write(' Done!\n')

                        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, mp)
                        
                        del X_wave, X_spec_s
                        
                        def inference(X_spec, device, model, aggressiveness):
                            
                            def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, tta=False):
                                model.eval()
                                    
                                global active_iterations
                                global progress_value
                                
                                with torch.no_grad():
                                    preds = []
                                    
                                    iterations = [n_window]

                                    if data['tta']:
                                        total_iterations = sum(iterations)
                                        total_iterations = total_iterations*2
                                    else:
                                        total_iterations = sum(iterations)
                                        
                                    if tta:
                                        active_iterations = sum(iterations)
                                        active_iterations = active_iterations - 2
                                        total_iterations = total_iterations - 2
                                    else:
                                        active_iterations = 0
                                    
                                    progress_bar = 0
                                    for i in range(n_window): 
                                        active_iterations += 1
                                        if data['demucsmodelVR']:
                                            update_progress(**progress_kwargs,
                                                step=(0.1 + (0.5/total_iterations * active_iterations)))
                                        else:
                                            update_progress(**progress_kwargs,
                                                step=(0.1 + (0.8/total_iterations * active_iterations)))
                                        start = i * roi_size
                                        progress_bar += 100
                                        progress_value = progress_bar
                                        active_iterations_step = active_iterations*100
                                        step = (active_iterations_step / total_iterations)
                                        
                                        percent_prog = f"{base_text}Inference Progress: {active_iterations}/{total_iterations} | {round(step)}%"
                                        text_widget.percentage(percent_prog)
                                        X_mag_window = X_mag_pad[None, :, :, start:start + data['window_size']]
                                        X_mag_window = torch.from_numpy(X_mag_window).to(device)

                                        pred = model.predict(X_mag_window, aggressiveness)

                                        pred = pred.detach().cpu().numpy()
                                        preds.append(pred[0])
                                        
                                    pred = np.concatenate(preds, axis=2)
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
                                                        device, model, aggressiveness, tta=True)
                                pred_tta = pred_tta[:, :, roi_size // 2:]
                                pred_tta = pred_tta[:, :, :n_frame]

                                return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.j * X_phase)
                            else:
                                return pred * coef, X_mag, np.exp(1.j * X_phase)
                                    
                        aggressiveness = {'value': aggresive_set, 'split_bin': mp.param['band'][1]['crop_stop']}
                        
                        if data['tta']:
                            text_widget.write(base_text + f"Running Inferences (TTA)... {space}\n")
                        else:
                            text_widget.write(base_text + f"Running Inference... {space}\n")
                        
                        pred, X_mag, X_phase = inference(X_spec_m,
                                                                device,
                                                                model, aggressiveness)

                        text_widget.write('\n')
                        
                        if data['postprocess']:
                            try:
                                text_widget.write(base_text + 'Post processing...')
                                pred_inv = np.clip(X_mag - pred, 0, np.inf)
                                pred = spec_utils.mask_silence(pred, pred_inv)
                                text_widget.write(' Done!\n')
                            except Exception as e:
                                text_widget.write('\n' + base_text + 'Post process failed, check error log.\n')
                                text_widget.write(base_text + 'Moving on...\n')
                                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                                errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                                try:
                                    with open('errorlog.txt', 'w') as f:
                                        f.write(f'Last Error Received:\n\n' +
                                                f'Error Received while attempting to run Post Processing on "{os.path.basename(music_file)}":\n' + 
                                                f'Process Method: VR Architecture\n\n' +
                                                f'If this error persists, please contact the developers.\n\n' + 
                                                f'Raw error details:\n\n' +
                                                errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                                except:
                                    pass

                            update_progress(**progress_kwargs,
                                            step=0.95)

                        # Inverse stft
                        y_spec_m = pred * X_phase
                        v_spec_m = X_spec_m - y_spec_m
                        
                        def demix_demucs(mix):

                            print(' Running Demucs Inference...')
                            
                            if split_mode:
                                text_widget.write(base_text + f'Running Demucs Inference... {space}')
                            else:
                                text_widget.write(base_text + f'Running Demucs Inference... ')
                            
                            mix = torch.tensor(mix, dtype=torch.float32)
                            ref = mix.mean(0)        
                            mix = (mix - ref.mean()) / ref.std()
                            widget_text = text_widget
                            with torch.no_grad():
                                sources = apply_model(demucs, 
                                                      mix[None], 
                                                      gui_progress_bar,
                                                      widget_text,
                                                      update_prog,
                                                      split=split_mode, 
                                                      device=device, 
                                                      overlap=overlap_set, 
                                                      shifts=shift_set, 
                                                      progress=False,
                                                      segmen=True,
                                                      **progress_demucs_kwargs)[0]
                            
                            if split_mode:
                                text_widget.write('\n')
                            else:
                                update_progress(**progress_kwargs,
                                                step=0.9)
                                text_widget.write('Done!\n')
                                
                            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
                            sources[[0,1]] = sources[[1,0]]
                            
                            return sources
                        
                        def demucs_prediction(m):
                            global demucs_sources
                            mix, samplerate = librosa.load(m, mono=False, sr=44100)
                            if mix.ndim == 1:
                                mix = np.asfortranarray([mix,mix])
                            
                            mix = mix.T
                            
                            demucs_sources = demix_demucs(mix.T)
                        
                        if data['demucsmodelVR']:
                            demucs = HDemucs(sources=["other", "vocals"])
                            path_d = Path('models/Demucs_Models/v3_repo')
                            #print('What Demucs model was chosen? ', demucs_model_set)
                            demucs = _gm(name=demucs_model_set, repo=path_d)
                            
                            if data['segment'] == 'None':
                                segment = None
                                if isinstance(demucs, BagOfModels):
                                    if segment is not None:
                                        for sub in demucs.models:
                                            sub.segment = segment
                                else:
                                    if segment is not None:
                                        sub.segment = segment
                            else:
                                try:
                                    segment = int(data['segment'])
                                    if isinstance(demucs, BagOfModels):
                                        if segment is not None:
                                            for sub in demucs.models:
                                                sub.segment = segment
                                    else:
                                        if segment is not None:
                                            sub.segment = segment
                                    #text_widget.write(base_text + "Segments set to "f"{segment}.\n")
                                except:
                                    segment = None
                                    if isinstance(demucs, BagOfModels):
                                        if segment is not None:
                                            for sub in demucs.models:
                                                sub.segment = segment
                                    else:
                                        if segment is not None:
                                            sub.segment = segment
                            
                            demucs.cpu()
                            demucs.eval()
                            
                            demucs_prediction(m)
                        
                        if data['voc_only'] and not data['inst_only']:
                            pass
                        else:
                            text_widget.write(base_text + 'Saving Instrumental... ')
                        
                        if data['high_end_process'].startswith('mirroring'):        
                            input_high_end_ = spec_utils.mirroring(data['high_end_process'], y_spec_m, input_high_end, mp)
                            if data['demucsmodelVR']:
                                wav_instrument = spec_utils.cmb_spectrogram_to_wave_d(y_spec_m, mp, input_high_end_h, input_high_end_, demucs=True) 
                                demucs_inst = demucs_sources[0]
                                sources = [wav_instrument,demucs_inst]
                                spec = [stft(sources[0],2048,1024),stft(sources[1],2048,1024)]
                                ln = min([spec[0].shape[2], spec[1].shape[2]])
                                spec[0] = spec[0][:,:,:ln]
                                spec[1] = spec[1][:,:,:ln]
                                v_spec_c = np.where(np.abs(spec[1]) <= np.abs(spec[0]), spec[1], spec[0])
                                wav_instrument = istft(v_spec_c,1024)
                            else:
                                wav_instrument = spec_utils.cmb_spectrogram_to_wave_d(y_spec_m, mp, input_high_end_h, input_high_end_, demucs=False)
                             
                            if data['voc_only'] and not data['inst_only']:
                                pass
                            else:
                                text_widget.write('Done!\n')   
                        else:
                            wav_instrument = spec_utils.cmb_spectrogram_to_wave_d(y_spec_m, mp)
                            if data['voc_only'] and not data['inst_only']:
                                pass
                            else:
                                text_widget.write('Done!\n')    

                        if data['inst_only'] and not data['voc_only']:
                            pass
                        else:
                            text_widget.write(base_text + 'Saving Vocals... ')
                        
                        if data['high_end_process'].startswith('mirroring'):        
                            input_high_end_ = spec_utils.mirroring(data['high_end_process'], v_spec_m, input_high_end, mp)
                            if data['demucsmodelVR']:
                                wav_vocals = spec_utils.cmb_spectrogram_to_wave_d(v_spec_m, mp, input_high_end_h, input_high_end_, demucs=True)
                                demucs_voc = demucs_sources[1]
                                sources = [wav_vocals,demucs_voc]
                                spec = [stft(sources[0],2048,1024),stft(sources[1],2048,1024)]
                                ln = min([spec[0].shape[2], spec[1].shape[2]])
                                spec[0] = spec[0][:,:,:ln]
                                spec[1] = spec[1][:,:,:ln]
                                v_spec_c = np.where(np.abs(spec[1]) >= np.abs(spec[0]), spec[1], spec[0])
                                wav_vocals = istft(v_spec_c,1024)
                            else:
                                wav_vocals = spec_utils.cmb_spectrogram_to_wave_d(v_spec_m, mp, input_high_end_h, input_high_end_, demucs=False)
                            
                            if data['inst_only'] and not data['voc_only']:
                                    pass
                            else:
                                text_widget.write('Done!\n')     
                        else:
                            wav_vocals = spec_utils.cmb_spectrogram_to_wave_d(v_spec_m, mp, demucs=False)
                            if data['inst_only'] and not data['voc_only']:
                                    pass
                            else:
                                text_widget.write('Done!\n')     

                        update_progress(**progress_kwargs,
                                        step=1)

                        # Save output music files
                        save_files(wav_instrument, wav_vocals)

                        update_progress(**progress_kwargs,
                                        step=1)

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
           

                        text_widget.write(base_text + 'Completed Separation!\n\n')
    except Exception as e:
        traceback_text = ''.join(traceback.format_tb(e.__traceback__))
        message = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
        if runtimeerr in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Your PC cannot process this audio file with the chunk size selected.\nPlease lower the chunk size and try again.\n\n')
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'Your PC cannot process this audio file with the chunk size selected.\nPlease lower the chunk size and try again.\n\n' +
                            f'If this error persists, please contact the developers.\n\n' + 
                            f'Raw error details:\n\n' +
                            message + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if cuda_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please close any GPU intensive applications and try again.\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please close any GPU intensive applications and try again.\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            f'Raw error details:\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return
        
        if mod_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Application files(s) are missing.\n')
            text_widget.write("\n" + f'{type(e).__name__} - "{e}"' + "\n\n")
            text_widget.write(f'Please check for missing files/scripts in the app directory and try again.\n')
            text_widget.write(f'If the error persists, please reinstall application or contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'Application files(s) are missing.\n' + 
                            f'Please check for missing files/scripts in the app directory and try again.\n' + 
                            f'If the error persists, please reinstall application or contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if file_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Missing file error raised.\n')
            text_widget.write("\n" + f'{type(e).__name__} - "{e}"' + "\n\n")
            text_widget.write("\n" + f'Please address the error and try again.' + "\n")
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'Missing file error raised.\n' + 
                            "\n" + f'Please address the error and try again.' + "\n" +
                            f'If this error persists, please contact the developers.\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if ffmp_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The input file type is not supported or FFmpeg is missing.\n')
            text_widget.write(f'Please select a file type supported by FFmpeg and try again.\n\n')
            text_widget.write(f'If FFmpeg is missing or not installed, you will only be able to process \".wav\" files \nuntil it is available on this system.\n\n')
            text_widget.write(f'See the \"More Info\" tab in the Help Guide.\n\n')
            text_widget.write(f'If this error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'The input file type is not supported or FFmpeg is missing.\nPlease select a file type supported by FFmpeg and try again.\n\n' + 
                            f'If FFmpeg is missing or not installed, you will only be able to process \".wav\" files until it is available on this system.\n\n' + 
                            f'See the \"More Info\" tab in the Help Guide.\n\n' + 
                            f'If this error persists, please contact the developers.\n\n' +
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if sf_write_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'Could not write audio file.\n')
            text_widget.write(f'This could be due to low storage on target device or a system permissions issue.\n')
            text_widget.write(f"\nGo to the Settings Menu and click \"Open Error Log\" for raw error details.\n")
            text_widget.write(f'\nIf the error persists, please contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'Could not write audio file.\n' + 
                            f'This could be due to low storage on target device or a system permissions issue.\n' + 
                            f'If the error persists, please contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if systemmemerr in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough system memory to use this \nmodel.\n\n')
            text_widget.write(f'Please do the following:\n\n1. Restart this application.\n2. Ensure any CPU intensive applications are closed.\n3. Then try again.\n\n')
            text_widget.write(f'Please Note: Intel Pentium and Intel Celeron processors do not work well with \nthis application.\n\n')
            text_widget.write(f'If the error persists, the system may not have enough RAM, or your CPU might \nnot be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'The application was unable to allocate enough system memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Restart this application.\n2. Ensure any CPU intensive applications are closed.\n3. Then try again.\n\n' + 
                            f'Please Note: Intel Pentium and Intel Celeron processors do not work well with this application.\n\n' +
                            f'If the error persists, the system may not have enough RAM, or your CPU might \nnot be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if demucs_model_missing_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The selected Demucs model is missing.\n\n')
            text_widget.write(f'Please download the model or make sure it is in the correct directory.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: VR Architecture\n\n' +
                            f'The selected Demucs model is missing.\n\n' + 
                            f'Please download the model or make sure it is in the correct directory.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        print(traceback_text)
        print(type(e).__name__, e)
        print(message)
        
        try:
            with open('errorlog.txt', 'w') as f:
                f.write(f'Last Error Received:\n\n' +
                        f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                        f'Process Method: VR Architecture\n\n' +
                        f'If this error persists, please contact the developers with the error details.\n\n' +
                        message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
        except:
            tk.messagebox.showerror(master=window,
                            title='Error Details',
                            message=message)
        progress_var.set(0)
        text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
        text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
        text_widget.write(f'\nError Received:\n')
        text_widget.write("\Go to the Settings Menu and click \"Open Error Log\" for raw error details.\n")
        text_widget.write("\n" + f'Please address the error and try again.' + "\n")
        text_widget.write(f'If this error persists, please contact the developers with the error details.\n\n')
        text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
        torch.cuda.empty_cache()
        button_widget.configure(state=tk.NORMAL)  # Enable Button
        return

    try:
        os.remove('temp.wav')
    except:
        pass

    progress_var.set(0)
    text_widget.write(f'Conversion(s) Completed!\n')
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache()
    button_widget.configure(state=tk.NORMAL)  # Enable Button
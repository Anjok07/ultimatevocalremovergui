import os
import importlib
import pydub
import shutil
import hashlib

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
from datetime import datetime

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
        # self.offset = model.offset
        
data = {
    # Paths
    'input_paths': None,
    'export_path': None,
    'saveFormat': 'wav',
    # Processing Options
    'gpu': -1,
    'postprocess': True,
    'tta': True,
    'output_image': True,
    'voc_only': False,
    'inst_only': False,
    # Models
    'instrumentalModel': None,
    'useModel': None,
    # Constants
    'window_size': 512,
    'agg': 10,
    'high_end_process': 'mirroring',
    'ModelParams': 'Auto'
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
    
    global model_params_d
    global nn_arch_sizes
    global nn_architecture
    
    #Error Handling
    
    runtimeerr = "CUDNN error executing cudnnSetTensorNdDescriptor"
    systemmemerr = "DefaultCPUAllocator: not enough memory"
    cuda_err = "CUDA out of memory"
    mod_err = "ModuleNotFoundError"
    file_err = "FileNotFoundError"
    ffmp_err = """audioread\__init__.py", line 116, in audio_open"""
    sf_write_err = "sf.write"
    
    try:
        with open('errorlog.txt', 'w') as f:
            f.write(f'No errors to report at this time.' + f'\n\nLast Process Method Used: VR Architecture' +
                    f'\nLast Conversion Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
    except:
        pass
    
    nn_arch_sizes = [
        31191, # default
        33966, 123821, 123812, 537238 # custom
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
        sf.write(f'temp.wav',
                 wav_instrument, mp.param['sr'])

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
                        wav_instrument, mp.param['sr'])
        elif VModel in model_name and data['inst_only']:
            pass
        elif data['voc_only']:
            pass
        else:
                sf.write(instrumental_path,
                        wav_instrument, mp.param['sr'])
                
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
                            wav_vocals, mp.param['sr'])
            elif VModel in model_name and data['voc_only']:
                pass
            elif data['inst_only']:
                pass
            else:
                sf.write(vocal_path,
                            wav_vocals, mp.param['sr'])
        
            if data['saveFormat'] == 'Mp3':
                try:
                    if data['inst_only'] == True:
                        pass
                    else:
                        musfile = pydub.AudioSegment.from_wav(vocal_path)
                        musfile.export(vocal_path_mp3, format="mp3", bitrate="320k")
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
                        musfile.export(instrumental_path_mp3, format="mp3", bitrate="320k")
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
    default_window_size = data['window_size']
    default_agg = data['agg']

    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button

    vocal_remover = VocalRemover(data, text_widget)
    modelFolderName = determineModelFolderName()

    # Separation Preperation
    try:        #Load File(s)
                for file_num, music_file in enumerate(data['input_paths'], start=1):
                        # Determine File Name
                        base_name = f'{data["export_path"]}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                        
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
                
                        #Load Model      
                        text_widget.write(base_text + 'Loading models...')
                        
                        model_size = math.ceil(os.stat(data['instrumentalModel']).st_size / 1024)
                        nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
                        
                        nets = importlib.import_module('lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
                        
                        aggresive_set = float(data['agg']/100)
                        
                        ModelName=(data['instrumentalModel'])

                        #Package Models
                        
                        model_hash = hashlib.md5(open(ModelName,'rb').read()).hexdigest()
                        print(model_hash)
                        
                        #v5 Models
                        
                        if model_hash == '47939caf0cfe52a0e81442b85b971dfd':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if model_hash == '4e4ecb9764c50a8c414fee6e10395bbe':  
                            model_params_auto=str('lib_v5/modelparams/4band_v2.json')
                            param_name_auto=str('4band_v2')
                        if model_hash == 'e60a1e84803ce4efc0a6551206cc4b71':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if model_hash == 'a82f14e75892e55e994376edbf0c8435':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if model_hash == '6dd9eaa6f0420af9f1d403aaafa4cc06':   
                            model_params_auto=str('lib_v5/modelparams/4band_v2_sn.json')
                            param_name_auto=str('4band_v2_sn')
                        if model_hash == '5c7bbca45a187e81abbbd351606164e5':    
                            model_params_auto=str('lib_v5/modelparams/3band_44100_msb2.json')
                            param_name_auto=str('3band_44100_msb2')
                        if model_hash == 'd6b2cb685a058a091e5e7098192d3233':    
                            model_params_auto=str('lib_v5/modelparams/3band_44100_msb2.json')
                            param_name_auto=str('3band_44100_msb2')
                        if model_hash == 'c1b9f38170a7c90e96f027992eb7c62b': 
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if model_hash == 'c3448ec923fa0edf3d03a19e633faa53':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if model_hash == '68aa2c8093d0080704b200d140f59e54':  
                            model_params_auto=str('lib_v5/modelparams/3band_44100.json')
                            param_name_auto=str('3band_44100.json')
                        if model_hash == 'fdc83be5b798e4bd29fe00fe6600e147':  
                            model_params_auto=str('lib_v5/modelparams/3band_44100_mid.json')
                            param_name_auto=str('3band_44100_mid.json')
                        if model_hash == '2ce34bc92fd57f55db16b7a4def3d745':  
                            model_params_auto=str('lib_v5/modelparams/3band_44100_mid.json')
                            param_name_auto=str('3band_44100_mid.json')
                        if model_hash == '52fdca89576f06cf4340b74a4730ee5f':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100.json')
                        if model_hash == '41191165b05d38fc77f072fa9e8e8a30':  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100.json')
                        if model_hash == '89e83b511ad474592689e562d5b1f80e':  
                            model_params_auto=str('lib_v5/modelparams/2band_32000.json')
                            param_name_auto=str('2band_32000.json')
                        if model_hash == '0b954da81d453b716b114d6d7c95177f':  
                            model_params_auto=str('lib_v5/modelparams/2band_32000.json')
                            param_name_auto=str('2band_32000.json')
                            
                        #v4 Models
                            
                        if model_hash == '6a00461c51c2920fd68937d4609ed6c8':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr16000_hl512.json')
                            param_name_auto=str('1band_sr16000_hl512')
                        if model_hash == '0ab504864d20f1bd378fe9c81ef37140':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                            param_name_auto=str('1band_sr32000_hl512')
                        if model_hash == '7dd21065bf91c10f7fccb57d7d83b07f':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                            param_name_auto=str('1band_sr32000_hl512')
                        if model_hash == '80ab74d65e515caa3622728d2de07d23':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                            param_name_auto=str('1band_sr32000_hl512')
                        if model_hash == 'edc115e7fc523245062200c00caa847f':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr33075_hl384.json')
                            param_name_auto=str('1band_sr33075_hl384')
                        if model_hash == '28063e9f6ab5b341c5f6d3c67f2045b7':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr33075_hl384.json')
                            param_name_auto=str('1band_sr33075_hl384')
                        if model_hash == 'b58090534c52cbc3e9b5104bad666ef2':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl512.json')
                            param_name_auto=str('1band_sr44100_hl512')
                        if model_hash == '0cdab9947f1b0928705f518f3c78ea8f':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl512.json')
                            param_name_auto=str('1band_sr44100_hl512')
                        if model_hash == 'ae702fed0238afb5346db8356fe25f13':  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl1024.json')
                            param_name_auto=str('1band_sr44100_hl1024')
                        
                        #User Models
  
                        #1 Band
                        if '1band_sr16000_hl512' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr16000_hl512.json')
                            param_name_auto=str('1band_sr16000_hl512')
                        if '1band_sr32000_hl512' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                            param_name_auto=str('1band_sr32000_hl512')
                        if '1band_sr33075_hl384' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr33075_hl384.json')
                            param_name_auto=str('1band_sr33075_hl384')
                        if '1band_sr44100_hl256' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl256.json')
                            param_name_auto=str('1band_sr44100_hl256')
                        if '1band_sr44100_hl512' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl512.json')
                            param_name_auto=str('1band_sr44100_hl512')
                        if '1band_sr44100_hl1024' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/1band_sr44100_hl1024.json')
                            param_name_auto=str('1band_sr44100_hl1024')
                            
                        #2 Band
                        if '2band_44100_lofi' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/2band_44100_lofi.json')
                            param_name_auto=str('2band_44100_lofi')
                        if '2band_32000' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/2band_32000.json')
                            param_name_auto=str('2band_32000')
                        if '2band_48000' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/2band_48000.json')
                            param_name_auto=str('2band_48000')
                            
                        #3 Band   
                        if '3band_44100' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/3band_44100.json')
                            param_name_auto=str('3band_44100')
                        if '3band_44100_mid' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/3band_44100_mid.json')
                            param_name_auto=str('3band_44100_mid')
                        if '3band_44100_msb2' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/3band_44100_msb2.json')
                            param_name_auto=str('3band_44100_msb2')
                            
                        #4 Band    
                        if '4band_44100' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100.json')
                            param_name_auto=str('4band_44100')
                        if '4band_44100_mid' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100_mid.json')
                            param_name_auto=str('4band_44100_mid')
                        if '4band_44100_msb' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100_msb.json')
                            param_name_auto=str('4band_44100_msb')
                        if '4band_44100_msb2' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100_msb2.json')
                            param_name_auto=str('4band_44100_msb2')
                        if '4band_44100_reverse' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100_reverse.json')
                            param_name_auto=str('4band_44100_reverse')
                        if '4band_44100_sw' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_44100_sw.json') 
                            param_name_auto=str('4band_44100_sw')
                        if '4band_v2' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_v2.json')
                            param_name_auto=str('4band_v2')
                        if '4band_v2_sn' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/4band_v2_sn.json')
                            param_name_auto=str('4band_v2_sn')
                        if 'tmodelparam' in ModelName:  
                            model_params_auto=str('lib_v5/modelparams/tmodelparam.json')
                            param_name_auto=str('User Model Param Set')
  
                        text_widget.write(' Done!\n')
                        
                        
                        if data['ModelParams'] == 'Auto':
                            param_name = param_name_auto
                            model_params_d = model_params_auto
                        else:
                            param_name = str(data['ModelParams'])
                            model_params_d = str('lib_v5/modelparams/' + data['ModelParams'])
                        
                        try:
                            print('Model Parameters:', model_params_d)
                            text_widget.write(base_text + 'Loading assigned model parameters ' + '\"' + param_name + '\"... ')
                        except Exception as e:
                            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                            errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
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
                            try:
                                with open('errorlog.txt', 'w') as f:
                                    f.write(f'Last Error Received:\n\n' +
                                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                                            f'Process Method: VR Architecture\n\n' +
                                            f'Model parameters are missing.\n\n' + 
                                            f'Please check the following:\n' + 
                                            f'1. Make sure the model is still present.\n' +
                                            f'2. If you are running a model that was not originally included in this package, please append the modelparam name to the model name.\n' + 
                                            f'  - Example if using \"4band_v2.json\" modelparam: \"model_4band_v2.pth\"\n\n' +
                                            f'Please address this and try again.\n\n' +
                                            f'Raw error details:\n\n' +
                                            errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                            except:
                                pass
                            torch.cuda.empty_cache()
                            progress_var.set(0)
                            button_widget.configure(state=tk.NORMAL)  # Enable Button
                            return
                        
                        
                        mp = ModelParameters(model_params_d)
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

                        mp = ModelParameters(model_params_d)
                            
                        # -Go through the different steps of seperation-
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
                                    
                        aggressiveness = {'value': aggresive_set, 'split_bin': mp.param['band'][1]['crop_stop']}
                        
                        if data['tta']:
                            text_widget.write(base_text + "Running Inferences (TTA)...\n")
                        else:
                            text_widget.write(base_text + "Running Inference...\n")
                        
                        pred, X_mag, X_phase = inference(X_spec_m,
                                                                device,
                                                                model, aggressiveness)

                        update_progress(**progress_kwargs,
                                        step=0.9)
                        # Postprocess
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
                        
                        if data['voc_only'] and not data['inst_only']:
                            pass
                        else:
                            text_widget.write(base_text + 'Saving Instrumental... ')
                        
                        if data['high_end_process'].startswith('mirroring'):        
                            input_high_end_ = spec_utils.mirroring(data['high_end_process'], y_spec_m, input_high_end, mp)
                            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)    
                            if data['voc_only'] and not data['inst_only']:
                                pass
                            else:
                                text_widget.write('Done!\n')   
                        else:
                            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
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

                            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp, input_high_end_h, input_high_end_)  
                            if data['inst_only'] and not data['voc_only']:
                                    pass
                            else:
                                text_widget.write('Done!\n')     
                        else:        
                            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
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
           

                        text_widget.write(base_text + 'Completed Seperation!\n\n')
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
            text_widget.write(f"\nFor raw error details, go to the Error Log tab in the Help Guide.\n")
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
        text_widget.write("\nFor raw error details, go to the Error Log tab in the Help Guide.\n")
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
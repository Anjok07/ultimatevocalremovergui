from functools import total_ordering
import importlib
import os
from statistics import mode
from pathlib import Path
import pydub
import hashlib
from random import randrange

import subprocess
import soundfile as sf
import torch
import numpy as np
from demucs.model import Demucs
from demucs.utils import apply_model
from models import get_models, spec_effects
import onnxruntime as ort
import time
import os
from tqdm import tqdm
import warnings
import sys
import librosa
import psutil

import cv2
import math
import librosa
import numpy as np
import soundfile as sf
import shutil
from tqdm import tqdm
from datetime import datetime

from lib_v5 import dataset
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
import torch

# Command line text parsing and widget manipulation
from collections import defaultdict
import tkinter as tk
import traceback  # Error Message Recent Calls
import time  # Timer

class Predictor():        
    def __init__(self):
        pass
    
    def prediction_setup(self, demucs_name,
                               channels=64):
        
        global device

        print('Print the gpu setting: ', data['gpu'])

        if data['gpu'] >= 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if data['gpu'] == -1:
            device = torch.device('cpu')
        
        if data['demucsmodel']:
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
            widget_text.write(base_text + 'Loading Demucs model... ')
            update_progress(**progress_kwargs,
            step=0.05)   
            self.demucs.to(device)
            self.demucs.load_state_dict(torch.load(demucs_name))
            widget_text.write('Done!\n')
            self.demucs.eval()
        self.onnx_models = {}
        c = 0
        
        self.models = get_models('tdf_extra', load=False, device=cpu, stems='vocals')
        widget_text.write(base_text + 'Loading ONNX model... ')
        update_progress(**progress_kwargs,
        step=0.1)
        c+=1
        
        if data['gpu'] >= 0:
            if torch.cuda.is_available():
                run_type = ['CUDAExecutionProvider']
            else:
                data['gpu'] = -1
                widget_text.write("\n" + base_text + "No NVIDIA GPU detected. Switching to CPU... ")    
                run_type = ['CPUExecutionProvider']     
        elif data['gpu'] == -1:
            run_type = ['CPUExecutionProvider']
            
        print(run_type)
        print(str(device))

        self.onnx_models[c] = ort.InferenceSession(os.path.join('models/MDX_Net_Models', model_set), providers=run_type)
        widget_text.write('Done!\n')
        
    def prediction(self, m):  
        #mix, rate = sf.read(m)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix,mix])
        mix = mix.T
        sources = self.demix(mix.T)
        widget_text.write(base_text + 'Inferences complete!\n')
        c = -1
    
        #Main Save Path
        save_path = os.path.dirname(base_name)
        
        #Vocal Path
        vocal_name = '(Vocals)'
        if data['modelFolder']:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}',)
        else:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}',)
        
        #Instrumental Path
        Instrumental_name = '(Instrumental)'
        if data['modelFolder']:
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{Instrumental_name}',)
        else: 
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{Instrumental_name}',)
            
        #Non-Reduced Vocal Path
        vocal_name = '(Vocals)'
        if data['modelFolder']:
            non_reduced_vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}_No_Reduction',)
        else:
            non_reduced_vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}_No_Reduction',)
            
        if os.path.isfile(non_reduced_vocal_path):
            file_exists_n = 'there'
        else:
            file_exists_n = 'not_there'

        if os.path.isfile(vocal_path):
            file_exists = 'there'
        else:
            file_exists = 'not_there'

        if not data['noisereduc_s'] == 'None':
            c += 1
            if not data['demucsmodel']:
                if data['inst_only'] and not data['voc_only']:
                    widget_text.write(base_text + 'Preparing to save Instrumental...')
                else:
                    widget_text.write(base_text + 'Saving vocals... ')
                sf.write(non_reduced_vocal_path, sources[c].T, rate)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')        
                widget_text.write(base_text + 'Performing Noise Reduction... ')
                reduction_sen = float(int(data['noisereduc_s'])/10)
                subprocess.call("lib_v5\\sox\\sox.exe" + ' "' + 
                            f"{str(non_reduced_vocal_path)}"  + '" "' + f"{str(vocal_path)}" + '" ' + 
                            "noisered lib_v5\\sox\\mdxnetnoisereduc.prof " + f"{reduction_sen}", 
                            shell=True, stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                widget_text.write('Done!\n')        
                update_progress(**progress_kwargs,
                step=(0.95))
            else:
                if data['inst_only'] and not data['voc_only']:
                    widget_text.write(base_text + 'Preparing Instrumental...')
                else:
                    widget_text.write(base_text + 'Saving Vocals... ')
                sf.write(non_reduced_vocal_path, sources[3].T, rate)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
                widget_text.write(base_text + 'Performing Noise Reduction... ')
                reduction_sen = float(int(data['noisereduc_s'])/10)
                subprocess.call("lib_v5\\sox\\sox.exe" + ' "' + 
                            f"{str(non_reduced_vocal_path)}"  + '" "' + f"{str(vocal_path)}" + '" ' + 
                            "noisered lib_v5\\sox\\mdxnetnoisereduc.prof " + f"{reduction_sen}", 
                            shell=True, stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                update_progress(**progress_kwargs,
                step=(0.95))
                widget_text.write('Done!\n')   
        else:
            c += 1
            if not data['demucsmodel']:
                widget_text.write(base_text + 'Saving Vocals..')
                sf.write(vocal_path, sources[c].T, rate)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
            else:
                widget_text.write(base_text + 'Saving Vocals... ')
                sf.write(vocal_path, sources[3].T, rate)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
        
        if data['voc_only'] and not data['inst_only']:
            pass
        else:
            finalfiles = [
                {
                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                    'files':[str(music_file), vocal_path],
                }
            ]         
            widget_text.write(base_text + 'Saving Instrumental... ')      
            for i, e in tqdm(enumerate(finalfiles)):

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
                
                ln = min([specs[0].shape[2], specs[1].shape[2]])
                specs[0] = specs[0][:,:,:ln]
                specs[1] = specs[1][:,:,:ln]
                X_mag = np.abs(specs[0])
                y_mag = np.abs(specs[1])            
                max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
                v_spec = specs[1] - max_mag * np.exp(1.j * np.angle(specs[0]))
                update_progress(**progress_kwargs,
                step=(0.95))
                sf.write(Instrumental_path, spec_utils.cmb_spectrogram_to_wave(-v_spec, mp), mp.param['sr'])
                if data['inst_only']:
                    if file_exists == 'there':
                        pass
                    else:
                        try:
                            os.remove(vocal_path)
                        except:
                            pass

                widget_text.write('Done!\n')
          
        if data['noisereduc_s'] == 'None':
            pass
        elif data['inst_only']:
            if file_exists_n == 'there':
                pass
            else:
                try:
                    os.remove(non_reduced_vocal_path)
                except:
                    pass
        else:
            try:
                os.remove(non_reduced_vocal_path)
            except:
                pass
        
        widget_text.write(base_text + 'Completed Seperation!\n\n')

    def demix(self, mix):
        # 1 = demucs only
        # 0 = onnx only
        if data['chunks'] == 'Full':
            chunk_set = 0
        else: 
            chunk_set = data['chunks']

        if data['chunks'] == 'Auto':
            if data['gpu'] == 0:
                try:
                    gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                except:
                    widget_text.write(base_text + 'NVIDIA GPU Required for conversion!\n')
                if int(gpu_mem) <= int(5):
                    chunk_set = int(5)
                    widget_text.write(base_text + 'Chunk size auto-set to 5... \n')
                if gpu_mem in [6, 7]:
                    chunk_set = int(30)
                    widget_text.write(base_text + 'Chunk size auto-set to 30... \n')
                if gpu_mem in [8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(40)
                    widget_text.write(base_text + 'Chunk size auto-set to 40... \n')
                if int(gpu_mem) >= int(16):
                    chunk_set = int(60)
                    widget_text.write(base_text + 'Chunk size auto-set to 60... \n')
            if data['gpu'] == -1:
                sys_mem = psutil.virtual_memory().total >> 30
                if int(sys_mem) <= int(4):
                    chunk_set = int(1)
                    widget_text.write(base_text + 'Chunk size auto-set to 1... \n')
                if sys_mem in [5, 6, 7, 8]:
                    chunk_set = int(10)
                    widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                    chunk_set = int(25)
                    widget_text.write(base_text + 'Chunk size auto-set to 25... \n')
                if int(sys_mem) >= int(17):
                    chunk_set = int(60)
                    widget_text.write(base_text + 'Chunk size auto-set to 60... \n')
        elif data['chunks'] == 'Full':
            chunk_set = 0
            widget_text.write(base_text + "Chunk size set to full... \n")
        else:
            chunk_set = int(data['chunks'])
            widget_text.write(base_text + "Chunk size user-set to "f"{chunk_set}... \n")
            
        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set*44100
        assert not margin == 0, 'margin cannot be zero!'
        if margin > chunk_size:
            margin = chunk_size

        b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])
        segmented_mix = {}
        
        if chunk_set == 0 or samples < chunk_size:
            chunk_size = samples
        
        counter = -1
        for skip in range(0, samples, chunk_size):
            counter+=1
    
            s_margin = 0 if counter == 0 else margin
            end = min(skip+chunk_size+margin, samples)

            start = skip-s_margin

            segmented_mix[skip] = mix[:,start:end].copy()
            if end == samples:
                break
        
        if not data['demucsmodel']:
            sources = self.demix_base(segmented_mix, margin_size=margin)

        else: # both, apply spec effects
            base_out = self.demix_base(segmented_mix, margin_size=margin)
            demucs_out = self.demix_demucs(segmented_mix, margin_size=margin)
            nan_count = np.count_nonzero(np.isnan(demucs_out)) + np.count_nonzero(np.isnan(base_out))
            if nan_count > 0:
                print('Warning: there are {} nan values in the array(s).'.format(nan_count))
                demucs_out, base_out = np.nan_to_num(demucs_out), np.nan_to_num(base_out)
            sources = {}

            sources[3] = (spec_effects(wave=[demucs_out[3],base_out[0]],
                                        algorithm='default',
                                        value=b[3])*1.03597672895) # compensation
        return sources
    
    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        onnxitera = len(mixes)
        onnxitera_calc = onnxitera * 2
        gui_progress_bar_onnx = 0
        widget_text.write(base_text + "Running ONNX Inference...\n")
        widget_text.write(base_text + "Processing "f"{onnxitera} slices... ")
        print(' Running ONNX Inference...')
        for mix in mixes:
            gui_progress_bar_onnx += 1
            if data['demucsmodel']:
                update_progress(**progress_kwargs,
                    step=(0.1 + (0.5/onnxitera_calc * gui_progress_bar_onnx)))
            else:
                update_progress(**progress_kwargs,
                    step=(0.1 + (0.9/onnxitera * gui_progress_bar_onnx)))
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            
            mod = 0 
            for model in self.models:
                mod += 1
                trim = model.n_fft//2
                gen_size = model.chunk_size-2*trim
                pad = gen_size - n_sample%gen_size
                mix_p = np.concatenate((np.zeros((2,trim)), cmix, np.zeros((2,pad)), np.zeros((2,trim))), 1)
                mix_waves = []
                i = 0
                while i < n_sample + pad:
                    waves = np.array(mix_p[:, i:i+model.chunk_size])
                    mix_waves.append(waves)
                    i += gen_size
                mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(cpu)
                with torch.no_grad():
                    _ort = self.onnx_models[mod]
                    spek = model.stft(mix_waves)
                    
                    tar_waves = model.istft(torch.tensor(_ort.run(None, {'input': spek.cpu().numpy()})[0]))#.cpu()

                    tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]

                    start = 0 if mix == 0 else margin_size
                    end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                    if margin_size == 0:
                        end = None
                    sources.append(tar_signal[:,start:end])

        
            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        del self.onnx_models
        widget_text.write('Done!\n')
        return _sources
    
    def demix_demucs(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Running Demucs Inference...\n")
        widget_text.write(base_text + "Processing "f"{len(mix)} slices... ")
        print(' Running Demucs Inference...')
        for nmix in mix:
            gui_progress_bar_demucs += 1
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            shift_set = 0
            with torch.no_grad():
                sources = apply_model(self.demucs, cmix.to(device), split=True, overlap=overlap_set, shifts=shift_set)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
        widget_text.write('Done!\n')
        return sources
        

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

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hide_opt():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class VocalRemover(object):
    
    def __init__(self, data, text_widget: tk.Text):
        self.data = data
        self.text_widget = text_widget
        self.models = defaultdict(lambda: None)
        self.devices = defaultdict(lambda: None)
        # self.offset = model.offset
        


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

class VocalRemover(object):
    
    def __init__(self, data, text_widget: tk.Text):
        self.data = data
        self.text_widget = text_widget
        # self.offset = model.offset

data = {
    # Paths
    'input_paths': None,
    'export_path': None,
    'saveFormat': 'wav',
    'vr_ensem': '2_HP-UVR',
    'vr_ensem_a': '1_HP-UVR',
    'vr_ensem_b': '2_HP-UVR',
    'vr_ensem_c': 'No Model',
    'vr_ensem_d': 'No Model',
    'vr_ensem_e': 'No Model',
    'vr_ensem_mdx_a': 'No Model',
    'vr_ensem_mdx_b': 'No Model',
    'vr_ensem_mdx_c': 'No Model',
    'mdx_ensem': 'UVR-MDX-NET 1',
    'mdx_ensem_b': 'No Model',
    # Processing Options
    'gpu': -1,
    'postprocess': True,
    'tta': True,
    'output_image': True,
    'voc_only': False,
    'inst_only': False,
    'demucsmodel': True,
    'chunks': 'auto',
    'non_red': False,
    'noisereduc_s': 3,
    'mixing': 'default',
    'ensChoose': 'Basic Ensemble',
    'algo': 'Instrumentals (Min Spec)',
    #Advanced Options
    'appendensem': True,
    # Models
    'instrumentalModel': None,
    'useModel': None,
    # Constants
    'window_size': 512,
    'agg': 10,
    'high_end_process': 'mirroring'
}

default_window_size = data['window_size']
default_agg = data['agg']
default_chunks = data['chunks']
default_noisereduc_s = data['noisereduc_s']


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
    
    global widget_text
    global gui_progress_bar
    global music_file
    global channel_set
    global margin_set
    global overlap_set
    global default_chunks
    global default_noisereduc_s
    global base_name
    global progress_kwargs
    global base_text
    global model_set
    global model_set_name
    global ModelName_2

    model_set = 'UVR_MDXNET_9703.onnx'
    model_set_name = 'UVR_MDXNET_9703'
    
    # Update default settings
    default_chunks = data['chunks']
    default_noisereduc_s = data['noisereduc_s']

    channel_set = int(64)
    margin_set = int(44100)
    overlap_set = float(0.5)
    
    widget_text = text_widget
    gui_progress_bar = progress_var
    
    #Error Handling

    onnxmissing = "[ONNXRuntimeError] : 3 : NO_SUCHFILE"
    onnxmemerror = "onnxruntime::CudaCall CUDA failure 2: out of memory"
    onnxmemerror2 = "onnxruntime::BFCArena::AllocateRawInternal"
    systemmemerr = "DefaultCPUAllocator: not enough memory"
    runtimeerr = "CUDNN error executing cudnnSetTensorNdDescriptor"
    cuda_err = "CUDA out of memory"
    enex_err = "local variable \'enseExport\' referenced before assignment"
    mod_err = "ModuleNotFoundError"
    file_err = "FileNotFoundError"
    ffmp_err = """audioread\__init__.py", line 116, in audio_open"""
    sf_write_err = "sf.write"
    
    try:
        with open('errorlog.txt', 'w') as f:
            f.write(f'No errors to report at this time.' + f'\n\nLast Process Method Used: Ensemble Mode' +
                    f'\nLast Conversion Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
    except:
        pass

    global nn_arch_sizes
    global nn_architecture
    
    nn_arch_sizes = [
        31191, # default
        33966, 123821, 123812, 537238, 537227 # custom
    ]
              
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
             
        if VModel in ModelName_1 and data['voc_only']:
                sf.write(instrumental_path,
                        wav_instrument, mp.param['sr'])
        elif VModel in ModelName_1 and data['inst_only']:
            pass
        elif data['voc_only']:
            pass
        else:
                sf.write(instrumental_path,
                        wav_instrument, mp.param['sr'])
                
        # Vocal
        if vocal_name is not None:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{ModelName_1}_{vocal_name}',
            )
            
            if VModel in ModelName_1 and data['inst_only']:
                sf.write(vocal_path,
                            wav_vocals, mp.param['sr'])
            elif VModel in ModelName_1 and data['voc_only']:
                pass
            elif data['inst_only']:
                pass
            else:
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

    if os.path.exists('models/Main_Models/7_HP2-UVR.pth') \
    or os.path.exists('models/Main_Models/8_HP2-UVR.pth') \
    or os.path.exists('models/Main_Models/9_HP2-UVR.pth'):
        hp2_ens = 'on'
    else:
        hp2_ens = 'off'

    timestampnum = round(datetime.utcnow().timestamp())
    randomnum = randrange(100000, 1000000)

    print('Do all of the HP models exist? ' + hp2_ens)

    # Separation Preperation
    try:    #Ensemble Dictionary

        if not data['ensChoose'] == 'User Ensemble':
            
                #1st Model
              
                if data['vr_ensem_a'] == 'MGM_MAIN_v4':
                    vr_ensem_a = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_a_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_a'] == 'MGM_HIGHEND_v4':
                    vr_ensem_a = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_a_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_a'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_a = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_a_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_a'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_a = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_a_name = 'MGM_LOWEND_B_v4'
                else:
                    vr_ensem_a_name = data['vr_ensem_a']
                    vr_ensem_a = f'models/Main_Models/{vr_ensem_a_name}.pth'
                    
                #2nd Model
                       
                if data['vr_ensem_b'] == 'MGM_MAIN_v4':
                    vr_ensem_b = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_b_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_b'] == 'MGM_HIGHEND_v4':
                    vr_ensem_b = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_b_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_b'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_b = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_b_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_b'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_b = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_b_name = 'MGM_LOWEND_B_v4'
                else:
                    vr_ensem_b_name = data['vr_ensem_b']
                    vr_ensem_b = f'models/Main_Models/{vr_ensem_b_name}.pth'
                    
                #3rd Model
                    
                if data['vr_ensem_c'] == 'MGM_MAIN_v4':
                    vr_ensem_c = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_c_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_c'] == 'MGM_HIGHEND_v4':
                    vr_ensem_c = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_c_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_c'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_c = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_c_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_c'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_c = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_c_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem_c'] == 'No Model':
                    vr_ensem_c = 'pass'
                    vr_ensem_c_name = 'pass'
                else:
                    vr_ensem_c_name = data['vr_ensem_c']
                    vr_ensem_c = f'models/Main_Models/{vr_ensem_c_name}.pth'
                     
                #4th Model
                  
                if data['vr_ensem_d'] == 'MGM_MAIN_v4':
                    vr_ensem_d = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_d_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_d'] == 'MGM_HIGHEND_v4':
                    vr_ensem_d = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_d_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_d'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_d = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_d_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_d'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_d = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_d_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem_d'] == 'No Model':
                    vr_ensem_d = 'pass' 
                    vr_ensem_d_name = 'pass'
                else:
                    vr_ensem_d_name = data['vr_ensem_d']
                    vr_ensem_d = f'models/Main_Models/{vr_ensem_d_name}.pth'
                    
                # 5th Model
                 
                if data['vr_ensem_e'] == 'MGM_MAIN_v4':
                    vr_ensem_e = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_e_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_e'] == 'MGM_HIGHEND_v4':
                    vr_ensem_e = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_e_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_e'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_e = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_e_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_e'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_e = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_e_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem_e'] == 'No Model':
                    vr_ensem_e = 'pass' 
                    vr_ensem_e_name = 'pass'                    
                else:
                    vr_ensem_e_name = data['vr_ensem_e']
                    vr_ensem_e = f'models/Main_Models/{vr_ensem_e_name}.pth'
                    
                if data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/2'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/2'
                        }
                    ] 
                elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 3/3'
                        }  
                    ] 
                elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 3/3' 
                        }  
                    ] 
                elif data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/3'
                        }  
                    ] 
                elif data['vr_ensem_d'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ]
                    
                elif data['vr_ensem_c'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ] 
                elif data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ] 
                else:
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/5'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/5'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/5'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 4/5' 
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 5/5'
                        }  
                    ] 
                
                HP2_Models = [
                    {
                        'model_name':'7_HP2-UVR',
                        'model_name_c':'1st HP2 Model',
                        'model_location':'models/Main_Models/7_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 1/3'
                    },
                    {
                        'model_name':'8_HP2-UVR',
                        'model_name_c':'2nd HP2 Model',
                        'model_location':'models/Main_Models/8_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 2/3'
                    },
                    {
                        'model_name':'9_HP2-UVR',
                        'model_name_c':'3rd HP2 Model',
                        'model_location':'models/Main_Models/9_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 3/3'
                    }
                ]
            
                All_HP_Models = [
                    {
                        'model_name':'7_HP2-UVR',
                        'model_name_c':'1st HP2 Model',
                        'model_location':'models/Main_Models/7_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 1/5'
                        
                    },
                    {
                        'model_name':'8_HP2-UVR',
                        'model_name_c':'2nd HP2 Model',
                        'model_location':'models/Main_Models/8_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 2/5'
                        
                    },
                    {
                        'model_name':'9_HP2-UVR',
                        'model_name_c':'3rd HP2 Model',
                        'model_location':'models/Main_Models/9_HP2-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 3/5'
                    },
                    {
                        'model_name':'1_HP-UVR',
                        'model_name_c':'1st HP Model',
                        'model_location':'models/Main_Models/1_HP-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 4/5'
                    },
                    {
                        'model_name':'2_HP-UVR',
                        'model_name_c':'2nd HP Model',
                        'model_location':'models/Main_Models/2_HP-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 5/5'
                    }
                ]
                
                Vocal_Models = [
                    {
                        'model_name':'3_HP-Vocal-UVR',
                        'model_name_c':'1st Vocal Model',
                        'model_location':'models/Main_Models/3_HP-Vocal-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 1/2'
                    },
                    {
                        'model_name':'4_HP-Vocal-UVR',
                        'model_name_c':'2nd Vocal Model',
                        'model_location':'models/Main_Models/4_HP-Vocal-UVR.pth',
                        'loop_name': 'Ensemble Mode - Model 2/2'
                    }
                ]

                #VR Model 1
                
                if data['vr_ensem'] == 'MGM_MAIN_v4':
                    vr_ensem = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_name = 'MGM_MAIN_v4'
                elif data['vr_ensem'] == 'MGM_HIGHEND_v4':
                    vr_ensem = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem'] == 'MGM_LOWEND_A_v4':
                    vr_ensem = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem'] == 'MGM_LOWEND_B_v4':
                    vr_ensem = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem'] == 'No Model':
                    vr_ensem = 'pass' 
                    vr_ensem_name = 'pass' 
                else:
                    vr_ensem_name = data['vr_ensem']
                    vr_ensem = f'models/Main_Models/{vr_ensem_name}.pth'
                
                #VR Model 2
                
                if data['vr_ensem_mdx_a'] == 'MGM_MAIN_v4':
                    vr_ensem_mdx_a = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_mdx_a_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_mdx_a'] == 'MGM_HIGHEND_v4':
                    vr_ensem_mdx_a = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_mdx_a_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_mdx_a'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_mdx_a = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_mdx_a_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_mdx_a'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_mdx_a = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_mdx_a_name = 'MGM_LOWEND_B_v4' 
                elif data['vr_ensem_mdx_a'] == 'No Model':
                    vr_ensem_mdx_a = 'pass' 
                    vr_ensem_mdx_a_name = 'pass' 
                else:
                    vr_ensem_mdx_a_name = data['vr_ensem_mdx_a']
                    vr_ensem_mdx_a = f'models/Main_Models/{vr_ensem_mdx_a_name}.pth'

                #VR Model 3
                
                if data['vr_ensem_mdx_b'] == 'MGM_MAIN_v4':
                    vr_ensem_mdx_b = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_mdx_b_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_mdx_b'] == 'MGM_HIGHEND_v4':
                    vr_ensem_mdx_b = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_mdx_b_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_mdx_b'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_mdx_b = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_mdx_b_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_mdx_b'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_mdx_b = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_mdx_b_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem_mdx_b'] == 'No Model':
                    vr_ensem_mdx_b = 'pass' 
                    vr_ensem_mdx_b_name = 'pass' 
                else:
                    vr_ensem_mdx_b_name = data['vr_ensem_mdx_b']
                    vr_ensem_mdx_b = f'models/Main_Models/{vr_ensem_mdx_b_name}.pth'
                
                #VR Model 4
                
                if data['vr_ensem_mdx_c'] == 'MGM_MAIN_v4':
                    vr_ensem_mdx_c = 'models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth'
                    vr_ensem_mdx_c_name = 'MGM_MAIN_v4'
                elif data['vr_ensem_mdx_c'] == 'MGM_HIGHEND_v4':
                    vr_ensem_mdx_c = 'models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
                    vr_ensem_mdx_c_name = 'MGM_HIGHEND_v4'
                elif data['vr_ensem_mdx_c'] == 'MGM_LOWEND_A_v4':
                    vr_ensem_mdx_c = 'models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth' 
                    vr_ensem_mdx_c_name = 'MGM_LOWEND_A_v4'
                elif data['vr_ensem_mdx_c'] == 'MGM_LOWEND_B_v4':
                    vr_ensem_mdx_c = 'models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth' 
                    vr_ensem_mdx_c_name = 'MGM_LOWEND_B_v4'
                elif data['vr_ensem_mdx_c'] == 'No Model':
                    vr_ensem_mdx_c = 'pass' 
                    vr_ensem_mdx_c_name = 'pass' 
                else:
                    vr_ensem_mdx_c_name = data['vr_ensem_mdx_c']
                    vr_ensem_mdx_c = f'models/Main_Models/{vr_ensem_mdx_c_name}.pth'
                                       
                #MDX-Net Model
                
                if data['mdx_ensem'] == 'UVR-MDX-NET 1':
                    mdx_ensem = 'UVR_MDXNET_9703'
                if data['mdx_ensem'] == 'UVR-MDX-NET 2':
                    mdx_ensem = 'UVR_MDXNET_9682'
                if data['mdx_ensem'] == 'UVR-MDX-NET 3':
                    mdx_ensem = 'UVR_MDXNET_9662'
                if data['mdx_ensem'] == 'UVR-MDX-NET Karaoke':
                    mdx_ensem = 'UVR_MDXNET_Karaoke'
                    
                #MDX-Net Model 2
                    
                if data['mdx_ensem_b'] == 'UVR-MDX-NET 1':
                    mdx_ensem_b = 'UVR_MDXNET_9703'
                if data['mdx_ensem_b'] == 'UVR-MDX-NET 2':
                    mdx_ensem_b = 'UVR_MDXNET_9682'
                if data['mdx_ensem_b'] == 'UVR-MDX-NET 3':
                    mdx_ensem_b = 'UVR_MDXNET_9662'
                if data['mdx_ensem_b'] == 'UVR-MDX-NET Karaoke':
                    mdx_ensem_b = 'UVR_MDXNET_Karaoke'
                if data['mdx_ensem_b'] == 'No Model':
                    mdx_ensem_b = 'pass'
                
                
                
                if data['vr_ensem'] == 'No Model' and data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_ensem}',
                        },
                        {
                            'model_name': 'pass',
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': 'pass',
                            'model_location':'pass',
                            'loop_name': f'Ensemble Mode - Last Model - {mdx_ensem_b}',
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}',
                        },
                        {
                            'model_name': 'pass',
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': 'pass',
                            'model_location':'pass',
                            'loop_name': 'Ensemble Mode - Last Model',
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                    ]

                elif data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_b'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        }
                    ]
                else:
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': 'pass',
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem_b,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                    
                if data['ensChoose'] == 'Basic Ensemble':
                    loops = Basic_Ensem
                    ensefolder = 'Basic_Ensemble_Outputs'
                    if data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name
                    elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_e_name                        
                    elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_d_name
                    elif data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_c_name
                    elif data['vr_ensem_c'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_d_name + '_' + vr_ensem_e_name
                    elif data['vr_ensem_d'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_c_name + '_' + vr_ensem_e_name
                    elif data['vr_ensem_e'] == 'No Model':
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_c_name + '_' + vr_ensem_d_name
                    else:
                        ensemode = 'Basic_Ensemble' + '_' + vr_ensem_a_name + '_' + vr_ensem_b_name + '_' + vr_ensem_c_name + '_' + vr_ensem_d_name + '_' + vr_ensem_e_name
                if data['ensChoose'] == 'HP2 Models':
                    loops = HP2_Models
                    ensefolder = 'HP2_Models_Ensemble_Outputs'
                    ensemode = 'HP2_Models'
                if data['ensChoose'] == 'All HP/HP2 Models':
                    loops = All_HP_Models
                    ensefolder = 'All_HP_HP2_Models_Ensemble_Outputs'
                    ensemode = 'All_HP_HP2_Models'
                if data['ensChoose'] == 'Vocal Models':           
                    loops = Vocal_Models
                    ensefolder = 'Vocal_Models_Ensemble_Outputs'
                    ensemode = 'Vocal_Models'
                if data['ensChoose'] == 'MDX-Net/VR Ensemble':           
                    loops = mdx_vr
                    ensefolder = 'MDX_VR_Ensemble_Outputs'
                    if data['vr_ensem'] == 'No Model' and data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                        ensemode = 'MDX-Net_Models'
                    elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name
                    elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_c_name
                    elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_b_name
                    elif data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_a_name
                    elif data['vr_ensem_mdx_a'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_b_name + '_' + vr_ensem_mdx_c_name
                    elif data['vr_ensem_mdx_b'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_a_name + '_' + vr_ensem_mdx_c_name
                    elif data['vr_ensem_mdx_c'] == 'No Model':
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_a_name + '_' + vr_ensem_mdx_b_name
                    else:
                        ensemode = 'MDX-Net_' + vr_ensem_name + '_' + vr_ensem_mdx_a_name + '_' + vr_ensem_mdx_b_name + '_' + vr_ensem_mdx_c_name

                #Prepare Audiofile(s)
                for file_num, music_file in enumerate(data['input_paths'], start=1):
                    print(data['input_paths'])
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
                      
                    #Prepare to loop models
                    for i, c in tqdm(enumerate(loops), disable=True, desc='Iterations..'):
                        
                            try:
                                ModelName_2=(c['mdx_model_name'])
                            except:
                                pass
                        
                        
                            if hp2_ens == 'off' and loops == HP2_Models:
                                    text_widget.write(base_text + 'You must install the UVR expansion pack in order to use this ensemble.\n')
                                    text_widget.write(base_text + 'Please install the expansion pack or choose another ensemble.\n')
                                    text_widget.write(base_text + 'See the \"Updates\" tab in the Help Guide for installation instructions.\n')
                                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
                                    torch.cuda.empty_cache()
                                    button_widget.configure(state=tk.NORMAL)
                                    return
                            elif hp2_ens == 'off' and loops == All_HP_Models:
                                    text_widget.write(base_text + 'You must install the UVR expansion pack in order to use this ensemble.\n')
                                    text_widget.write(base_text + 'Please install the expansion pack or choose another ensemble.\n')
                                    text_widget.write(base_text + 'See the \"Updates\" tab in the Help Guide for installation instructions.\n')
                                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
                                    torch.cuda.empty_cache()
                                    button_widget.configure(state=tk.NORMAL)
                                    return


                            def determineenseFolderName():
                                """
                                Determine the name that is used for the folder and appended
                                to the back of the music files
                                """
                                enseFolderName = ''

                                if str(ensefolder):
                                    enseFolderName += os.path.splitext(os.path.basename(ensefolder))[0]

                                if enseFolderName:
                                    try:
                                        enseFolderName = '/' + enseFolderName + '_' + str(timestampnum)
                                    except:
                                        enseFolderName = '/' + enseFolderName + '_' + str(randomnum)

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
                            
                            
                            if c['model_location'] == 'pass':
                                pass
                            else:
                                presentmodel = Path(c['model_location'])
                                
                                if presentmodel.is_file():
                                    print(f'The file {presentmodel} exist')
                                else: 
                                    if data['ensChoose'] == 'MDX-Net/VR Ensemble':
                                        text_widget.write(base_text + 'Model "' + c['model_name'] + '.pth" is missing.\n')
                                        text_widget.write(base_text + 'Installation of v5 Model Expansion Pack required to use this model.\n')
                                        text_widget.write(base_text + f'If the error persists, please verify all models are present.\n\n')
                                        text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                                        torch.cuda.empty_cache()
                                        progress_var.set(0)
                                        button_widget.configure(state=tk.NORMAL)  # Enable Button
                                        return 
                                    else:
                                        text_widget.write(base_text + 'Model "' + c['model_name'] + '.pth" is missing.\n')
                                        text_widget.write(base_text + 'Installation of v5 Model Expansion Pack required to use this model.\n\n')
                                        continue
                            
                                text_widget.write(c['loop_name'] + '\n\n')
                                
                                text_widget.write(base_text + 'Loading ' + c['model_name_c'] + '... ')
                                    
                                aggresive_set = float(data['agg']/100)
                                
                        
                                model_size = math.ceil(os.stat(c['model_location']).st_size / 1024)
                                nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
                                
                                nets = importlib.import_module('lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
                                
                                text_widget.write('Done!\n')
                                
                                ModelName=(c['model_location'])

                                #Package Models
                                
                                model_hash = hashlib.md5(open(ModelName,'rb').read()).hexdigest()
                                print(model_hash)
                                
                                #v5 Models
                                
                                if model_hash == '47939caf0cfe52a0e81442b85b971dfd':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100')
                                if model_hash == '4e4ecb9764c50a8c414fee6e10395bbe':  
                                    model_params_d=str('lib_v5/modelparams/4band_v2.json')
                                    param_name=str('4band_v2')
                                if model_hash == 'e60a1e84803ce4efc0a6551206cc4b71':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100')
                                if model_hash == 'a82f14e75892e55e994376edbf0c8435':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100')
                                if model_hash == '6dd9eaa6f0420af9f1d403aaafa4cc06':   
                                    model_params_d=str('lib_v5/modelparams/4band_v2_sn.json')
                                    param_name=str('4band_v2_sn')
                                if model_hash == '5c7bbca45a187e81abbbd351606164e5':    
                                    model_params_d=str('lib_v5/modelparams/3band_44100_msb2.json')
                                    param_name=str('3band_44100_msb2')
                                if model_hash == 'd6b2cb685a058a091e5e7098192d3233':    
                                    model_params_d=str('lib_v5/modelparams/3band_44100_msb2.json')
                                    param_name=str('3band_44100_msb2')
                                if model_hash == 'c1b9f38170a7c90e96f027992eb7c62b': 
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100')
                                if model_hash == 'c3448ec923fa0edf3d03a19e633faa53':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100')
                                if model_hash == '68aa2c8093d0080704b200d140f59e54':  
                                    model_params_d=str('lib_v5/modelparams/3band_44100.json')
                                    param_name=str('3band_44100.json')
                                if model_hash == 'fdc83be5b798e4bd29fe00fe6600e147':  
                                    model_params_d=str('lib_v5/modelparams/3band_44100_mid.json')
                                    param_name=str('3band_44100_mid.json')
                                if model_hash == '2ce34bc92fd57f55db16b7a4def3d745':  
                                    model_params_d=str('lib_v5/modelparams/3band_44100_mid.json')
                                    param_name=str('3band_44100_mid.json')
                                if model_hash == '52fdca89576f06cf4340b74a4730ee5f':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100.json')
                                if model_hash == '41191165b05d38fc77f072fa9e8e8a30':  
                                    model_params_d=str('lib_v5/modelparams/4band_44100.json')
                                    param_name=str('4band_44100.json')
                                if model_hash == '89e83b511ad474592689e562d5b1f80e':  
                                    model_params_d=str('lib_v5/modelparams/2band_32000.json')
                                    param_name=str('2band_32000.json')
                                if model_hash == '0b954da81d453b716b114d6d7c95177f':  
                                    model_params_d=str('lib_v5/modelparams/2band_32000.json')
                                    param_name=str('2band_32000.json')
                                    
                                #v4 Models
                                    
                                if model_hash == '6a00461c51c2920fd68937d4609ed6c8':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr16000_hl512.json')
                                    param_name=str('1band_sr16000_hl512')
                                if model_hash == '0ab504864d20f1bd378fe9c81ef37140':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                                    param_name=str('1band_sr32000_hl512')
                                if model_hash == '7dd21065bf91c10f7fccb57d7d83b07f':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                                    param_name=str('1band_sr32000_hl512')
                                if model_hash == '80ab74d65e515caa3622728d2de07d23':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr32000_hl512.json')
                                    param_name=str('1band_sr32000_hl512')
                                if model_hash == 'edc115e7fc523245062200c00caa847f':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr33075_hl384.json')
                                    param_name=str('1band_sr33075_hl384')
                                if model_hash == '28063e9f6ab5b341c5f6d3c67f2045b7':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr33075_hl384.json')
                                    param_name=str('1band_sr33075_hl384')
                                if model_hash == 'b58090534c52cbc3e9b5104bad666ef2':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr44100_hl512.json')
                                    param_name=str('1band_sr44100_hl512')
                                if model_hash == '0cdab9947f1b0928705f518f3c78ea8f':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr44100_hl512.json')
                                    param_name=str('1band_sr44100_hl512')
                                if model_hash == 'ae702fed0238afb5346db8356fe25f13':  
                                    model_params_d=str('lib_v5/modelparams/1band_sr44100_hl1024.json')
                                    param_name=str('1band_sr44100_hl1024')
                                    

                                ModelName_1=(c['model_name'])

                                print('Model Parameters:', model_params_d)
                                text_widget.write(base_text + 'Loading assigned model parameters ' + '\"' + param_name + '\"... ')
                                
                                mp = ModelParameters(model_params_d)
                                
                                text_widget.write('Done!\n')
                                
                                #Load model
                                if os.path.isfile(c['model_location']):
                                    device = torch.device('cpu')
                                    model = nets.CascadedASPPNet(mp.param['bins'] * 2)
                                    model.load_state_dict(torch.load(c['model_location'],
                                                                    map_location=device))
                                    if torch.cuda.is_available() and data['gpu'] >= 0:
                                        device = torch.device('cuda:{}'.format(data['gpu']))
                                        model.to(device)
                                
                                model_name = os.path.basename(c["model_name"])

                                # -Go through the different steps of seperation-
                                # Wave source
                                text_widget.write(base_text + 'Loading audio source... ')
                                
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

                                text_widget.write(base_text + 'Loading the stft of audio source... ')
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
                                
                                aggressiveness = {'value': aggresive_set, 'split_bin': mp.param['band'][1]['crop_stop']}
                                
                                if data['tta']:
                                    text_widget.write(base_text + "Running Inferences (TTA)... \n")
                                else:
                                    text_widget.write(base_text + "Running Inference... \n")
                                
                                pred, X_mag, X_phase = inference(X_spec_m,
                                                                        device,
                                                                        model, aggressiveness)
                                
                                # update_progress(**progress_kwargs,
                                #                 step=0.8)
                                
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
                                                        f'Process Method: Ensemble Mode\n\n' +
                                                        f'If this error persists, please contact the developers.\n\n' + 
                                                        f'Raw error details:\n\n' +
                                                        errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                                        except:
                                            pass

                                # Inverse stft
                                # nopep8 
                                y_spec_m = pred * X_phase
                                v_spec_m = X_spec_m - y_spec_m

                                if data['voc_only']:
                                    pass
                                else:
                                    text_widget.write(base_text + 'Saving Instrumental... ')
                                
                                if data['high_end_process'].startswith('mirroring'):        
                                    input_high_end_ = spec_utils.mirroring(data['high_end_process'], y_spec_m, input_high_end, mp)
                                    wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp, input_high_end_h, input_high_end_)    
                                    if data['voc_only']:
                                        pass
                                    else:
                                        text_widget.write('Done!\n')   
                                else:
                                    wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, mp)
                                    if data['voc_only']:
                                        pass
                                    else:
                                        text_widget.write('Done!\n')    

                                if data['inst_only']:
                                    pass
                                else:
                                    text_widget.write(base_text + 'Saving Vocals... ')
                                
                                if data['high_end_process'].startswith('mirroring'):        
                                    input_high_end_ = spec_utils.mirroring(data['high_end_process'], v_spec_m, input_high_end, mp)
                                    wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp, input_high_end_h, input_high_end_)   
                                    if data['inst_only']:
                                            pass
                                    else:
                                        text_widget.write('Done!\n')     
                                else:        
                                    wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, mp)
                                    if data['inst_only']:
                                            pass
                                    else:
                                        text_widget.write('Done!\n')  

                            
                                update_progress(**progress_kwargs,
                                                step=1)
                                
                                # Save output music files
                                save_files(wav_instrument, wav_vocals)

                                # Save output image
                                if data['output_image']:
                                    with open('{}_{}_Instruments.jpg'.format(base_name, c['model_name']), mode='wb') as f:
                                        image = spec_utils.spectrogram_to_image(y_spec_m)
                                        _, bin_image = cv2.imencode('.jpg', image)
                                        bin_image.tofile(f)
                                    with open('{}_{}_Vocals.jpg'.format(base_name, c['model_name']), mode='wb') as f:
                                        image = spec_utils.spectrogram_to_image(v_spec_m)
                                        _, bin_image = cv2.imencode('.jpg', image)
                                        bin_image.tofile(f)
                                        
                                text_widget.write(base_text + 'Completed Seperation!\n\n')  
                    

                            if data['ensChoose'] == 'MDX-Net/VR Ensemble':
                                
                                mdx_name = c['mdx_model_name']
                                
                                if c['mdx_model_name'] == 'pass':
                                    pass
                                else:
                                    text_widget.write('Ensemble Mode - Running Model - ' + mdx_name + '\n\n')


                                    update_progress(**progress_kwargs,
                                                    step=0)    
                                    
                                    if data['noisereduc_s'] == 'None':
                                        pass
                                    else:
                                        if not os.path.isfile("lib_v5\sox\sox.exe"):
                                            data['noisereduc_s'] = 'None'
                                            data['non_red'] = False
                                            widget_text.write(base_text + 'SoX is missing and required for noise reduction.\n')
                                            widget_text.write(base_text + 'See the \"More Info\" tab in the Help Guide.\n')
                                            widget_text.write(base_text + 'Noise Reduction will be disabled until SoX is available.\n\n')
                                    
                                    e = os.path.join(data["export_path"])
                                    
                                    demucsmodel = 'models/Demucs_Model/demucs_extra-3646af93_org.th'

                                    pred = Predictor()
                                    pred.prediction_setup(demucs_name=demucsmodel,
                                                        channels=channel_set)
                                    
                                    # split
                                    pred.prediction(
                                        m=music_file,
                                    )
                            else:
                                pass

                    # Emsembling Outputs
                    def get_files(folder="", prefix="", suffix=""):
                        return [f"{folder}{i}" for i in os.listdir(folder) if i.startswith(prefix) if i.endswith(suffix)]
                
                    if data['appendensem'] == False:
                        voc_inst = [
                            {
                                'algorithm':'min_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                'output':'{}_(Instrumental)'.format(trackname),
                                'type': 'Instrumentals'
                            },
                            {
                                'algorithm':'max_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                'output': '{}_(Vocals)'.format(trackname),
                                'type': 'Vocals'
                            }
                        ]

                        inst = [
                            {
                                'algorithm':'min_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                'output':'{}_(Instrumental)'.format(trackname),
                                'type': 'Instrumentals'
                            }
                        ]

                        vocal = [
                            {
                                'algorithm':'max_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                'output': '{}_(Vocals)'.format(trackname),
                                'type': 'Vocals'
                            }
                        ]
                    else:
                        voc_inst = [
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

                        inst = [
                            {
                                'algorithm':'min_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                'output':'{}_Ensembled_{}_(Instrumental)'.format(trackname, ensemode),
                                'type': 'Instrumentals'
                            }
                        ]

                        vocal = [
                            {
                                'algorithm':'max_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                'output': '{}_Ensembled_{}_(Vocals)'.format(trackname, ensemode),
                                'type': 'Vocals'
                            }
                        ] 

                    if data['voc_only']:
                        ensembles = vocal
                    elif data['inst_only']:
                        ensembles = inst
                    else:
                        ensembles = voc_inst
                        
                    try:
                        for i, e in tqdm(enumerate(ensembles), desc="Ensembling..."):
                            
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
                            
                            if data['saveFormat'] == 'Mp3':
                                try:
                                    musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                                    musfile.export((os.path.join('{}'.format(data['export_path']),'{}.mp3'.format(e['output']))), format="mp3", bitrate="320k")
                                    os.remove((os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output']))))
                                except Exception as e:
                                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                                    if "ffmpeg" in errmessage:
                                        text_widget.write('\n' + base_text + 'Failed to save output(s) as Mp3(s).\n')
                                        text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                                        text_widget.write(base_text + 'Moving on... ')
                                    else:
                                        text_widget.write('\n' + base_text + 'Failed to save output(s) as Mp3(s).\n')
                                        text_widget.write(base_text + 'Please check error log.\n')
                                        text_widget.write(base_text + 'Moving on... ')
                                    try:
                                        with open('errorlog.txt', 'w') as f:
                                            f.write(f'Last Error Received:\n\n' +
                                                    f'Error Received while attempting to save file as mp3 "{os.path.basename(music_file)}".\n\n' + 
                                                    f'Process Method: Ensemble Mode\n\n' +
                                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                                    f'If this error persists, please contact the developers.\n\n' + 
                                                    f'Raw error details:\n\n' +
                                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                                    except:
                                        pass
                            
                            if data['saveFormat'] == 'Flac':
                                try:
                                    musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                                    musfile.export((os.path.join('{}'.format(data['export_path']),'{}.flac'.format(e['output']))), format="flac")
                                    os.remove((os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output']))))
                                except Exception as e:
                                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                                    errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                                    if "ffmpeg" in errmessage:
                                        text_widget.write('\n' + base_text + 'Failed to save output(s) as Flac(s).\n')
                                        text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                                        text_widget.write(base_text + 'Moving on... ')
                                    else:
                                        text_widget.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                                        text_widget.write(base_text + 'Please check error log.\n')
                                        text_widget.write(base_text + 'Moving on... ')
                                    try:
                                        with open('errorlog.txt', 'w') as f:
                                            f.write(f'Last Error Received:\n\n' +
                                                    f'Error Received while attempting to save file as flac "{os.path.basename(music_file)}".\n' + 
                                                    f'Process Method: Ensemble Mode\n\n' +
                                                    f'FFmpeg might be missing or corrupted.\n\n' +
                                                    f'If this error persists, please contact the developers.\n\n' + 
                                                    f'Raw error details:\n\n' +
                                                    errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n')
                                    except: 
                                        pass
                            
                            text_widget.write("Done!\n")                                                        
                    except:
                        text_widget.write('\n' + base_text + 'Not enough files to ensemble.')
                        pass
                    
                        update_progress(**progress_kwargs,
                        step=0.95)
                    text_widget.write("\n")

                    try:
                        if not data['save']: # Deletes all outputs if Save All Outputs isn't checked
                            files = get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav")
                            for file in files:
                                os.remove(file)
                        if not data['save']:
                            files = get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav")
                            for file in files:
                                os.remove(file)
                    except:
                        pass
                    
                    if data['save'] and data['saveFormat'] == 'Mp3':
                        try:
                            text_widget.write(base_text + 'Saving all ensemble outputs in Mp3... ')
                            path = enseExport
                            #Change working directory
                            os.chdir(path)
                            audio_files = os.listdir()
                            for file in audio_files:
                                #spliting the file into the name and the extension
                                name, ext = os.path.splitext(file)
                                if ext == ".wav":
                                    if trackname in file:
                                        musfile = pydub.AudioSegment.from_wav(file)
                                        #rename them using the old name + ".wav"
                                        musfile.export("{0}.mp3".format(name), format="mp3", bitrate="320k")    
                            try:
                                files = get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav")
                                for file in files:
                                    os.remove(file) 
                            except:
                                pass
                            try:  
                                files = get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav")
                                for file in files:
                                    os.remove(file) 
                            except:
                                pass
                                    
                            text_widget.write('Done!\n\n')
                            base_path = os.path.dirname(os.path.abspath(__file__))
                            os.chdir(base_path)

                        except Exception as e:
                            base_path = os.path.dirname(os.path.abspath(__file__))
                            os.chdir(base_path)
                            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                            errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                            if "ffmpeg" in errmessage:
                                text_widget.write('\n' + base_text + 'Failed to save output(s) as Mp3(s).\n')
                                text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                                text_widget.write(base_text + 'Moving on...\n')
                            else:
                                text_widget.write('\n' + base_text + 'Failed to save output(s) as Mp3(s).\n')
                                text_widget.write(base_text + 'Please check error log.\n')
                                text_widget.write(base_text + 'Moving on...\n')
                            try:
                                with open('errorlog.txt', 'w') as f:
                                    f.write(f'Last Error Received:\n\n' +
                                            f'\nError Received while attempting to save ensembled outputs as mp3s.\n' + 
                                            f'Process Method: Ensemble Mode\n\n' +
                                            f'FFmpeg might be missing or corrupted.\n\n' +
                                            f'If this error persists, please contact the developers.\n\n' + 
                                            f'Raw error details:\n\n' +
                                            errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                            except:
                                pass
                        
                    if data['save'] and data['saveFormat'] == 'Flac':
                        try:
                            text_widget.write(base_text + 'Saving all ensemble outputs in Flac... ')
                            path = enseExport
                            #Change working directory
                            os.chdir(path)
                            audio_files = os.listdir()
                            for file in audio_files:
                                #spliting the file into the name and the extension
                                name, ext = os.path.splitext(file)
                                if ext == ".wav":
                                    if trackname in file:
                                        musfile = pydub.AudioSegment.from_wav(file)
                                        #rename them using the old name + ".wav"
                                        musfile.export("{0}.flac".format(name), format="flac")    
                            try:
                                files = get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav")
                                for file in files:
                                    os.remove(file) 
                            except:
                                pass
                            try:  
                                files = get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav")
                                for file in files:
                                    os.remove(file) 
                            except:
                                pass
                            
                            text_widget.write('Done!\n\n')
                            base_path = os.path.dirname(os.path.abspath(__file__))
                            os.chdir(base_path)

                        except Exception as e:
                            base_path = os.path.dirname(os.path.abspath(__file__))
                            os.chdir(base_path)
                            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                            errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                            if "ffmpeg" in errmessage:
                                text_widget.write('\n' + base_text + 'Failed to save output(s) as Flac(s).\n')
                                text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                                text_widget.write(base_text + 'Moving on...\n')
                            else:
                                text_widget.write('\n' + base_text + 'Failed to save output(s) as Flac(s).\n')
                                text_widget.write(base_text + 'Please check error log.\n')
                                text_widget.write(base_text + 'Moving on...\n')
                            try:
                                with open('errorlog.txt', 'w') as f:
                                    f.write(f'Last Error Received:\n\n' +
                                            f'\nError Received while attempting to ensembled outputs as Flacs.\n' + 
                                            f'Process Method: Ensemble Mode\n\n' +
                                            f'FFmpeg might be missing or corrupted.\n\n' +
                                            f'If this error persists, please contact the developers.\n\n' + 
                                            f'Raw error details:\n\n' +
                                            errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                            except:
                                pass          
                  
                  
                try:
                    os.remove('temp.wav')
                except:
                    pass
        
                if len(os.listdir(enseExport)) == 0: #Check if the folder is empty
                    shutil.rmtree(folder_path) #Delete folder if empty
                                   
        else:
            progress_kwargs = {'progress_var': progress_var,
                                'total_files': len(data['input_paths']),
                                'file_num': len(data['input_paths'])}
            base_text = get_baseText(total_files=len(data['input_paths']),
                            file_num=len(data['input_paths']))

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

            music_file = data['input_paths']
            if len(data['input_paths']) <= 1:
                text_widget.write(base_text + "Not enough files to process.\n")
                pass
            else:   
                update_progress(**progress_kwargs,
                step=0.2) 
                
                savefilename = (data['input_paths'][0])
                trackname1 = f'{os.path.splitext(os.path.basename(savefilename))[0]}'
                
                insts = [
                    {
                        'algorithm':'min_mag',
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'output':'{}_User_Ensembled_(Min Spec)'.format(trackname1),
                        'type': 'Instrumentals'
                    }
                ]

                vocals = [
                    {
                        'algorithm':'max_mag',
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'output': '{}_User_Ensembled_(Max Spec)'.format(trackname1),
                        'type': 'Vocals'
                    }
                ]
                
                invert_spec = [
                    {
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'output': '{}_diff_si'.format(trackname1),
                        'type': 'Spectral Inversion'
                    }
                ]
                
                invert_nor = [
                    {
                        'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                        'output': '{}_diff_ni'.format(trackname1),
                        'type': 'Normal Inversion'
                    }
                ]
                
                if data['algo'] == 'Instrumentals (Min Spec)':
                    ensem = insts
                if data['algo'] == 'Vocals (Max Spec)':
                    ensem = vocals
                if data['algo'] == 'Invert (Spectral)':
                    ensem = invert_spec
                if data['algo'] == 'Invert (Normal)':
                    ensem = invert_nor

                #Prepare to loop models
                if data['algo'] == 'Instrumentals (Min Spec)' or data['algo'] == 'Vocals (Max Spec)':
                    for i, e in tqdm(enumerate(ensem), desc="Ensembling..."):
                        text_widget.write(base_text + "Ensembling " + e['type'] + "... ") 
                    
                        wave, specs = {}, {}
                                
                        mp = ModelParameters(e['model_params'])
                        
                        for i in range(len(data['input_paths'])):    
                            spec = {}
                            
                            for d in range(len(mp.param['band']), 0, -1):          
                                bp = mp.param['band'][d]            
                                
                                if d == len(mp.param['band']): # high-end band                
                                    wave[d], _ = librosa.load(
                                        data['input_paths'][i], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                                    
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
                        
                    if data['saveFormat'] == 'Mp3':
                        try:
                            musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                            musfile.export((os.path.join('{}'.format(data['export_path']),'{}.mp3'.format(e['output']))), format="mp3", bitrate="320k")
                            os.remove((os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output']))))
                        except Exception as e:
                            text_widget.write('\n' + base_text + 'Failed to save output(s) as Mp3.')
                            text_widget.write('\n' + base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                            text_widget.write(base_text + 'Moving on...\n')
                            text_widget.write(base_text + f'Complete!\n')
                            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                            errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                            try:
                                with open('errorlog.txt', 'w') as f:
                                    f.write(f'Last Error Received:\n\n' +
                                            f'Error Received while attempting to run user ensemble:\n' + 
                                            f'Process Method: Ensemble Mode\n\n' +
                                            f'FFmpeg might be missing or corrupted.\n\n' +
                                            f'If this error persists, please contact the developers.\n\n' + 
                                            f'Raw error details:\n\n' +
                                            errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                            except:
                                pass
                            progress_var.set(0)
                            button_widget.configure(state=tk.NORMAL)
                                
                            return
                    
                    if data['saveFormat'] == 'Flac':
                        try:
                            musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                            musfile.export((os.path.join('{}'.format(data['export_path']),'{}.flac'.format(e['output']))), format="flac")
                            os.remove((os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output']))))
                        except Exception as e:
                            text_widget.write('\n' + base_text + 'Failed to save output as Flac.\n')
                            text_widget.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                            text_widget.write(base_text + 'Moving on...\n')
                            text_widget.write(base_text + f'Complete!\n')
                            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                            errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                            try:
                                with open('errorlog.txt', 'w') as f:
                                    f.write(f'Last Error Received:\n\n' +
                                            f'Error Received while attempting to run user ensemble:\n' + 
                                            f'Process Method: Ensemble Mode\n\n' +
                                            f'FFmpeg might be missing or corrupted.\n\n' +
                                            f'If this error persists, please contact the developers.\n\n' + 
                                            f'Raw error details:\n\n' +
                                            errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                            except:
                                pass
                            progress_var.set(0)
                            button_widget.configure(state=tk.NORMAL)
                            return
                        
                    text_widget.write("Done!\n")
                if data['algo'] == 'Invert (Spectral)' and data['algo'] == 'Invert (Normal)':
                    if len(data['input_paths']) != 2:
                        text_widget.write(base_text + "Invalid file count.\n")
                        pass
                    else:
                        for i, e in tqdm(enumerate(ensem), desc="Inverting..."):
                             
                            wave, specs = {}, {}
                                    
                            mp = ModelParameters(e['model_params'])
                            
                            for i in range(len(data['input_paths'])):    
                                spec = {}
                                
                                for d in range(len(mp.param['band']), 0, -1):          
                                    bp = mp.param['band'][d]            
                                    
                                    if d == len(mp.param['band']): # high-end band                
                                        wave[d], _ = librosa.load(
                                            data['input_paths'][i], bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                                        
                                        if len(wave[d].shape) == 1: # mono to stereo
                                            wave[d] = np.array([wave[d], wave[d]])
                                    else: # lower bands
                                        wave[d] = librosa.resample(wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
                                            
                                    spec[d] = spec_utils.wave_to_spectrogram(wave[d], bp['hl'], bp['n_fft'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse'])
                                    
                                specs[i] = spec_utils.combine_spectrograms(spec, mp)
                            
                            del wave   
                            
                            ln = min([specs[0].shape[2], specs[1].shape[2]])
                            specs[0] = specs[0][:,:,:ln]
                            specs[1] = specs[1][:,:,:ln]
                            if data['algo'] == 'Invert (Spectral)':
                                text_widget.write(base_text + "Performing " + e['type'] + "... ")
                                X_mag = np.abs(specs[0])
                                y_mag = np.abs(specs[1])            
                                max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
                                v_spec = specs[1] - max_mag * np.exp(1.j * np.angle(specs[0]))
                                sf.write(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])), 
                                        spec_utils.cmb_spectrogram_to_wave(-v_spec, mp), mp.param['sr'])
                            if data['algo'] == 'Invert (Normal)':
                                v_spec = specs[0] - specs[1]
                                sf.write(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])), 
                                        spec_utils.cmb_spectrogram_to_wave(v_spec, mp), mp.param['sr'])
                            text_widget.write("Done!\n")
                            


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
                            f'Process Method: Ensemble Mode\n\n' +
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
                            f'Process Method: Ensemble Mode\n\n' +
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
                            f'Process Method: Ensemble Mode\n\n' +
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
                            f'Process Method: Ensemble Mode\n\n' +
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
                            f'Process Method: Ensemble Mode\n\n' +
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
        
        if onnxmissing in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application could not detect this MDX-Net model on your system.\n')
            text_widget.write(f'Please make sure all the models are present in the correct directory.\n')
            text_widget.write(f'If the error persists, please reinstall application or contact the developers.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Ensemble Mode\n\n' +
                            f'The application could not detect this MDX-Net model on your system.\n' + 
                            f'Please make sure all the models are present in the correct directory.\n' + 
                            f'If the error persists, please reinstall application or contact the developers.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if onnxmemerror in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Ensemble Mode\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
            progress_var.set(0)
            button_widget.configure(state=tk.NORMAL)  # Enable Button
            return 
        
        if onnxmemerror2 in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to allocate enough GPU memory to use this model.\n')
            text_widget.write(f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n')
            text_widget.write(f'If the error persists, your GPU might not be supported.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Ensemble Mode\n\n' +
                            f'The application was unable to allocate enough GPU memory to use this model.\n' + 
                            f'Please do the following:\n\n1. Close any GPU intensive applications.\n2. Lower the set chunk size.\n3. Then try again.\n\n' + 
                            f'If the error persists, your GPU might not be supported.\n\n' + 
                            message + f'\nError Time Stamp [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
            except:
                pass
            torch.cuda.empty_cache()
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
                            f'Process Method: Ensemble Mode\n\n' +
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
                            f'Process Method: Ensemble Mode\n\n' +
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
        
        if enex_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The application was unable to locate a model you selected for this ensemble.\n')
            text_widget.write(f'\nPlease do the following to use all compatible models:\n\n1. Navigate to the \"Updates\" tab in the Help Guide.\n2. Download and install the v5 Model Expansion Pack.\n3. Then try again.\n\n')
            text_widget.write(f'If the error persists, please verify all models are present.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: Ensemble Mode\n\n' +
                            f'The application was unable to locate a model you selected for this ensemble.\n' + 
                            f'\nPlease do the following to use all compatible models:\n\n1. Navigate to the \"Updates\" tab in the Help Guide.\n2. Download and install the model expansion pack.\n3. Then try again.\n\n' + 
                            f'If the error persists, please verify all models are present.\n\n' + 
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
                        f'Process Method: Ensemble Mode\n\n' +
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
 
    update_progress(**progress_kwargs,
    step=1) 
    

    print('Done!')
    
    progress_var.set(0)
    if not data['ensChoose'] == 'User Ensemble':
        text_widget.write(base_text + f'Conversions Completed!\n')
    elif data['algo'] == 'Instrumentals (Min Spec)' and len(data['input_paths']) <= 1 or data['algo'] == 'Vocals (Max Spec)' and len(data['input_paths']) <= 1:
        text_widget.write(base_text + f'Please select 2 or more files to use this feature and try again.\n')
    elif data['algo'] == 'Instrumentals (Min Spec)' or data['algo'] == 'Vocals (Max Spec)':
        text_widget.write(base_text + f'Ensemble Complete!\n')
    elif len(data['input_paths']) != 2 and data['algo'] == 'Invert (Spectral)' or len(data['input_paths']) != 2 and data['algo'] == 'Invert (Normal)':
        text_widget.write(base_text + f'Please select exactly 2 files to extract difference.\n')
    elif data['algo'] == 'Invert (Spectral)' or data['algo'] == 'Invert (Normal)':
        text_widget.write(base_text + f'Complete!\n')
        
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache()
    button_widget.configure(state=tk.NORMAL)  #Enable Button

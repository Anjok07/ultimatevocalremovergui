from collections import defaultdict
from datetime import datetime
from demucs.apply import BagOfModels, apply_model
from demucs.hdemucs import HDemucs
from demucs.model_v2 import Demucs
from demucs.pretrained import get_model as _gm
from demucs.tasnet_v2 import ConvTasNet
from demucs.utils import apply_model_v1
from demucs.utils import apply_model_v2
from functools import total_ordering
from lib_v5 import dataset
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
from models import get_models, spec_effects
from pathlib import Path
from random import randrange
from statistics import mode
from tqdm import tqdm
from tqdm import tqdm
from tkinter import filedialog
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.font
import tkinter as tk
from tkinter import *
from tkinter.tix import *
import lib_v5.filelist
import cv2
import gzip
import hashlib
import importlib
import librosa
import json
import math
import numpy as np
import numpy as np
import onnxruntime as ort
import os
import pathlib
import psutil
import pydub
import re
import shutil
import soundfile as sf
import soundfile as sf
import subprocess
import sys
import time
import time  # Timer
import tkinter as tk
import torch
import torch
import traceback  # Error Message Recent Calls
import warnings

class Predictor():        
    def __init__(self):
        pass
    
    def mdx_options(self):
        """
        Open Advanced MDX Options
        """
        self.okVar = tk.IntVar()
        self.n_fft_scale_set_var = tk.StringVar(value='6144')
        self.dim_f_set_var = tk.StringVar(value='2048')
        self.mdxnetModeltype_var = tk.StringVar(value='Vocals')
        self.noise_pro_select_set_var = tk.StringVar(value='MDX-NET_Noise_Profile_14_kHz')
        self.compensate_v_var = tk.StringVar(value=1.03597672895)
        
        mdx_model_set = Toplevel()

        mdx_model_set.geometry("490x515")
        window_height = 490
        window_width = 515
        
        mdx_model_set.title("Specify Parameters")
        
        mdx_model_set.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = mdx_model_set.winfo_screenwidth()
        screen_height = mdx_model_set.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        mdx_model_set.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        mdx_model_set.iconbitmap('img\\UVR-Icon-v2.ico')
        
        mdx_model_set_window = ttk.Notebook(mdx_model_set)
        
        mdx_model_set_window.pack(expand = 1, fill ="both")
        
        mdx_model_set_window.grid_rowconfigure(0, weight=1)
        mdx_model_set_window.grid_columnconfigure(0, weight=1)
        
        frame0=Frame(mdx_model_set_window,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        frame0.tkraise(frame0)
        
        space_small = '  '*20
        space_small_1 = '  '*10
        
        l0=tk.Label(frame0, text=f'{space_small}Stem Type{space_small}', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=5)
        
        l0=ttk.OptionMenu(frame0, self.mdxnetModeltype_var, None, 'Vocals', 'Instrumental')
        l0.grid(row=4,column=0,padx=0,pady=5)

        l0=tk.Label(frame0, text='N_FFT Scale', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text=f'{space_small_1}(Manual Set){space_small_1}', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=1,padx=0,pady=5)
        
        self.options_n_fft_scale_Opt = l0=ttk.OptionMenu(frame0, self.n_fft_scale_set_var, None, '4096', '6144', '7680', '8192', '16384')
        
        self.options_n_fft_scale_Opt
        l0.grid(row=6,column=0,padx=0,pady=5)
        
        self.options_n_fft_scale_Entry = l0=ttk.Entry(frame0, textvariable=self.n_fft_scale_set_var, justify='center')
        
        self.options_n_fft_scale_Entry
        l0.grid(row=6,column=1,padx=0,pady=5)

        l0=tk.Label(frame0, text='Dim_f', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='(Manual Set)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=1,padx=0,pady=5)
        
        self.options_dim_f_Opt = l0=ttk.OptionMenu(frame0, self.dim_f_set_var, None, '2048', '3072', '4096')
        
        self.options_dim_f_Opt
        l0.grid(row=8,column=0,padx=0,pady=5)
        
        self.options_dim_f_Entry = l0=ttk.Entry(frame0, textvariable=self.dim_f_set_var, justify='center')
        
        self.options_dim_f_Entry
        l0.grid(row=8,column=1,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Noise Profile', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=0,padx=0,pady=5)
        
        l0=ttk.OptionMenu(frame0, self.noise_pro_select_set_var, None, 'MDX-NET_Noise_Profile_14_kHz', 'MDX-NET_Noise_Profile_17_kHz', 'MDX-NET_Noise_Profile_Full_Band')
        l0.grid(row=10,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Volume Compensation', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=11,column=0,padx=0,pady=10)
        
        self.options_compensate = l0=ttk.Entry(frame0, textvariable=self.compensate_v_var, justify='center')
        
        self.options_compensate
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text="Continue & Set These Parameters", command=lambda: self.okVar.set(1))
        l0.grid(row=13,column=0,padx=0,pady=30)
        
        def stop():
            widget_text.write(f'Please configure the ONNX model settings accordingly and try again.\n\n')
            widget_text.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            gui_progress_bar.set(0)
            widget_button.configure(state=tk.NORMAL)  # Enable Button
            self.okVar.set(1)
            stop_button()
            mdx_model_set.destroy()
            return
        
        l0=ttk.Button(frame0,text="Stop Process", command=stop)
        l0.grid(row=13,column=1,padx=0,pady=30)
        
        #print('print from mdx_model_set ', model_hash)
        
        #source_val = 0
        
        mdx_model_set.protocol("WM_DELETE_WINDOW", stop)
        
        frame0.wait_variable(self.okVar)
        
        global n_fft_scale_set
        global dim_f_set
        global modeltype
        global stemset_n
        global source_val
        global noise_pro_set
        global compensate
        global demucs_model_set
        
        stemtype = self.mdxnetModeltype_var.get()
        
        if stemtype == 'Vocals':
            modeltype = 'v'
            stemset_n = '(Vocals)'
            source_val = 3
        if stemtype == 'Instrumental':
            modeltype = 'v'
            stemset_n = '(Instrumental)'
            source_val = 2
        if stemtype == 'Other':
            modeltype = 'o'
            stemset_n = '(Other)'
            source_val = 2
        if stemtype == 'Drums':
            modeltype = 'd'
            stemset_n = '(Drums)'
            source_val = 1
        if stemtype == 'Bass':
            modeltype = 'b'
            stemset_n = '(Bass)'
            source_val = 0
            
        compensate = self.compensate_v_var.get()
        n_fft_scale_set = int(self.n_fft_scale_set_var.get())
        dim_f_set = int(self.dim_f_set_var.get())
        noise_pro_set = self.noise_pro_select_set_var.get()
        
        mdx_model_params = {
                'modeltype' : modeltype,
                'stemset_n' : stemset_n,
                'source_val' : source_val,
                'compensate' : compensate,
                'n_fft_scale_set' : n_fft_scale_set,
                'dim_f_set' : dim_f_set,
                'noise_pro' : noise_pro_set,
                }
        
        mdx_model_params_r = json.dumps(mdx_model_params, indent=4)
        
        with open(f"lib_v5/filelists/model_cache/mdx_model_cache/{model_hash}.json", "w") as outfile:
            outfile.write(mdx_model_params_r)

        if stemset_n == '(Instrumental)':
            if not 'UVR' in demucs_model_set:
                if demucs_switch == 'on':
                    widget_text.write(base_text + 'The selected Demucs model cannot be used with this model.\n')
                    widget_text.write(base_text + 'Only 2 stem Demucs models are compatible with this model.\n')
                    widget_text.write(base_text + 'Setting Demucs model to \"UVR_Demucs_Model_1\".\n\n')
                    demucs_model_set = 'UVR_Demucs_Model_1'
        
        mdx_model_set.destroy()
    
    def prediction_setup(self):
        
        global device

        if data['gpu'] >= 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
        if data['gpu'] == -1:
            device = torch.device('cpu')
        
        if demucs_switch == 'on':
            
            #print('check model here: ', demucs_model_set)
            
            #'demucs.th.gz', 'demucs_extra.th.gz', 'light.th.gz', 'light_extra.th.gz'
            
            if 'tasnet.th' in demucs_model_set or 'tasnet_extra.th' in demucs_model_set or \
            'demucs.th' in demucs_model_set or \
            'demucs_extra.th' in demucs_model_set or 'light.th' in demucs_model_set or \
            'light_extra.th' in demucs_model_set or 'v1' in demucs_model_set or '.gz' in demucs_model_set:
                load_from = "models/Demucs_Models/"f"{demucs_model_set}"
                if str(load_from).endswith(".gz"):
                    load_from = gzip.open(load_from, "rb")
                klass, args, kwargs, state = torch.load(load_from)
                self.demucs = klass(*args, **kwargs)
                widget_text.write(base_text + 'Loading Demucs v1 model... ')
                update_progress(**progress_kwargs,
                step=0.05)   
                self.demucs.to(device) 
                self.demucs.load_state_dict(state)
                widget_text.write('Done!\n')
                if not data['segment'] == 'Default':
                    widget_text.write(base_text + 'Note: Segments only available for Demucs v3\n')
                else:
                    pass
            
            elif 'tasnet-beb46fac.th' in demucs_model_set or 'tasnet_extra-df3777b2.th' in demucs_model_set or \
            'demucs48_hq-28a1282c.th' in demucs_model_set or'demucs-e07c671f.th' in demucs_model_set or \
            'demucs_extra-3646af93.th' in demucs_model_set or 'demucs_unittest-09ebc15f.th' in demucs_model_set or \
            'v2' in demucs_model_set:
                if '48' in demucs_model_set:
                    channels=48
                elif 'unittest' in demucs_model_set:
                    channels=4
                else:
                    channels=64
                    
                if 'tasnet' in demucs_model_set:
                    self.demucs = ConvTasNet(sources=["drums", "bass", "other", "vocals"], X=10)
                else:
                    self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=channels)
                widget_text.write(base_text + 'Loading Demucs v2 model... ')
                update_progress(**progress_kwargs,
                step=0.05)   
                self.demucs.to(device) 
                self.demucs.load_state_dict(torch.load("models/Demucs_Models/"f"{demucs_model_set}"))
                widget_text.write('Done!\n')
                if not data['segment'] == 'Default':
                    widget_text.write(base_text + 'Note: Segments only available for Demucs v3\n')
                else:
                    pass
                self.demucs.eval()
                
            else:  
                if 'UVR' in demucs_model_set:
                    self.demucs = HDemucs(sources=["other", "vocals"])
                else:
                    self.demucs = HDemucs(sources=["drums", "bass", "other", "vocals"])
                widget_text.write(base_text + 'Loading Demucs model... ')
                update_progress(**progress_kwargs,
                step=0.05)   
                path_d = Path('models/Demucs_Models/v3_repo')
                #print('What Demucs model was chosen? ', demucs_model_set)
                self.demucs = _gm(name=demucs_model_set, repo=path_d)
                self.demucs.to(device)
                self.demucs.eval()
                widget_text.write('Done!\n')
                if isinstance(self.demucs, BagOfModels):
                    widget_text.write(base_text + f"Selected Demucs model is a bag of {len(self.demucs.models)} model(s).\n")
                    
                if data['segment'] == 'Default':
                    segment = None
                    if isinstance(self.demucs, BagOfModels):
                        if segment is not None:
                            for sub in self.demucs.models:
                                sub.segment = segment
                    else:
                        if segment is not None:
                            sub.segment = segment
                else:
                    try:
                        segment = int(data['segment'])
                        if isinstance(self.demucs, BagOfModels):
                            if segment is not None:
                                for sub in self.demucs.models:
                                    sub.segment = segment
                        else:
                            if segment is not None:
                                sub.segment = segment
                        if split_mode:
                            widget_text.write(base_text + "Segments set to "f"{segment}.\n")
                    except:
                        segment = None
                        if isinstance(self.demucs, BagOfModels):
                            if segment is not None:
                                for sub in self.demucs.models:
                                    sub.segment = segment
                        else:
                            if segment is not None:
                                sub.segment = segment

        self.onnx_models = {}
        c = 0
        
        if demucs_only == 'on':
            pass
        else:
            self.models = get_models('tdf_extra', load=False, device=cpu, stems=modeltype, n_fft_scale=n_fft_scale_set, dim_f=dim_f_set)
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

        if demucs_only == 'off':
            self.onnx_models[c] = ort.InferenceSession(os.path.join('models/MDX_Net_Models', model_set), providers=run_type)
            #print(demucs_model_set)
            widget_text.write('Done!\n')
        elif demucs_only == 'on':
            #print(demucs_model_set)
            pass
          
    def prediction(self, m):  
        
        mix, samplerate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix,mix])
        samplerate = samplerate
        
        mix = mix.T
        sources = self.demix(mix.T)
        widget_text.write(base_text + 'Inferences complete!\n')
        
        c = -1
    
        inst_only = data['inst_only']
        voc_only = data['voc_only']
    
        if stemset_n == '(Instrumental)':
            if data['inst_only'] == True:
                voc_only = True
                inst_only = False
            if data['voc_only'] == True:
                inst_only = True
                voc_only = False
    
        #Main Save Path
        save_path = os.path.dirname(base_name)
        
        #Write name
        
        if stemset_n == '(Vocals)':
            stem_text_a = 'Vocals'
            stem_text_b = 'Instrumental'
        elif stemset_n == '(Instrumental)':
            stem_text_a = 'Instrumental'
            stem_text_b = 'Vocals'
        
        #Vocal Path

        if stemset_n == '(Vocals)':
            vocal_name = '(Vocals)'
        elif stemset_n == '(Instrumental)':
            vocal_name = '(Instrumental)'

        if data['modelFolder']:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}',)
        else:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{vocal_name}',)
        
        #Instrumental Path
        if stemset_n == '(Vocals)':
            Instrumental_name = '(Instrumental)'
        elif stemset_n == '(Instrumental)':
            Instrumental_name = '(Vocals)'
            
        if data['modelFolder']:
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{Instrumental_name}',)
        else: 
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_2}_{Instrumental_name}',)
            
        #Non-Reduced Vocal Path
        if stemset_n == '(Vocals)':
            vocal_name = '(Vocals)'
        elif stemset_n == '(Instrumental)':
            vocal_name = '(Instrumental)'
    
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

        if demucs_only == 'on':
            data['noisereduc_s'] == 'None'

        if not data['noisereduc_s'] == 'None':
            c += 1
            if demucs_switch == 'off':
                if inst_only and not voc_only:
                    widget_text.write(base_text + f'Preparing to save {stem_text_b}...')
                else:
                    widget_text.write(base_text + f'Saving {stem_text_a}... ')
                sf.write(non_reduced_vocal_path, sources[c].T, samplerate, subtype=wav_type_set)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')        
                widget_text.write(base_text + 'Performing Noise Reduction... ')
                reduction_sen = float(int(data['noisereduc_s'])/10)
                subprocess.call("lib_v5\\sox\\sox.exe" + ' "' + 
                            f"{str(non_reduced_vocal_path)}"  + '" "' + f"{str(vocal_path)}" + '" ' + 
                            "noisered lib_v5\\sox\\" + noise_pro_set + ".prof " + f"{reduction_sen}", 
                            shell=True, stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                widget_text.write('Done!\n')        
                update_progress(**progress_kwargs,
                step=(0.95))
            else:
                if inst_only and not voc_only:
                    widget_text.write(base_text + f'Preparing to save {stem_text_b}...')
                else:
                    widget_text.write(base_text + f'Saving {stem_text_a}... ')
                if demucs_only == 'on':
                    if 'UVR' in model_set_name:
                        sf.write(vocal_path, sources[1].T, samplerate, subtype=wav_type_set)
                        update_progress(**progress_kwargs,
                        step=(0.95))
                        widget_text.write('Done!\n') 
                    if 'extra' in model_set_name:
                        sf.write(vocal_path, sources[3].T, samplerate, subtype=wav_type_set)    
                        update_progress(**progress_kwargs,
                        step=(0.95))
                        widget_text.write('Done!\n') 
                else:
                    sf.write(non_reduced_vocal_path, sources[3].T, samplerate, subtype=wav_type_set)
                    update_progress(**progress_kwargs,
                    step=(0.9))
                    widget_text.write('Done!\n')
                    widget_text.write(base_text + 'Performing Noise Reduction... ')
                    reduction_sen = float(data['noisereduc_s'])/10
                    subprocess.call("lib_v5\\sox\\sox.exe" + ' "' + 
                                f"{str(non_reduced_vocal_path)}"  + '" "' + f"{str(vocal_path)}" + '" ' + 
                                "noisered lib_v5\\sox\\" + noise_pro_set + ".prof " + f"{reduction_sen}", 
                                shell=True, stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    update_progress(**progress_kwargs,
                    step=(0.95))
                    widget_text.write('Done!\n')   
        else:
            c += 1
            if demucs_switch == 'off':
                widget_text.write(base_text + f'Saving {stem_text_a}... ')
                sf.write(vocal_path, sources[c].T, samplerate, subtype=wav_type_set)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
            else:
                widget_text.write(base_text + f'Saving {stem_text_a}... ')
                if demucs_only == 'on':
                    if 'UVR' in model_set_name:
                        sf.write(vocal_path, sources[1].T, samplerate, subtype=wav_type_set)
                    if 'extra' in model_set_name:
                        sf.write(vocal_path, sources[3].T, samplerate, subtype=wav_type_set)
                else:
                    sf.write(vocal_path, sources[3].T, samplerate, subtype=wav_type_set)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
        
        if voc_only and not inst_only:
            pass
        else:
            finalfiles = [
                {
                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                    'files':[str(music_file), vocal_path],
                }
            ]         
            widget_text.write(base_text + f'Saving {stem_text_b}... ')      
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
                sf.write(Instrumental_path, normalization_set(spec_utils.cmb_spectrogram_to_wave(-v_spec, mp)), mp.param['sr'], subtype=wav_type_set)
                if inst_only:
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
        elif inst_only:
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
        
        widget_text.write(base_text + 'Completed Separation!\n\n')

    def demix(self, mix):
        
        if data['chunks'] == 'Full':
            chunk_set = 0
            widget_text.write(base_text + "Chunk size user-set to \"Full\"... \n")
        elif data['chunks'] == 'Auto':
            if data['gpu'] == 0:
                try:
                    gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                except:
                    widget_text.write(base_text + 'NVIDIA GPU Required for conversion!\n')
                    data['gpu'] = -1
                    pass
                if int(gpu_mem) <= int(6):
                    chunk_set = int(5)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 5... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 5... \n')
                if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(10)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                if int(gpu_mem) >= int(16):
                    chunk_set = int(40)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 40... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 40... \n')
            if data['gpu'] == -1:
                sys_mem = psutil.virtual_memory().total >> 30
                if int(sys_mem) <= int(4):
                    chunk_set = int(1)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 1... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 1... \n')
                if sys_mem in [5, 6, 7, 8]:
                    chunk_set = int(10)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                    chunk_set = int(25)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 25... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 25... \n')
                    
                if int(sys_mem) >= int(17):
                    chunk_set = int(60)
                    if demucs_only == 'on':
                        if no_chunk_demucs:
                            widget_text.write(base_text + 'Chunk size auto-set to 60... \n')
                    else:
                        widget_text.write(base_text + 'Chunk size auto-set to 60... \n')
        elif data['chunks'] == '0':
            chunk_set = 0
            if demucs_only == 'on':
                if no_chunk_demucs:
                    widget_text.write(base_text + "Chunk size user-set to \"Full\"... \n")
            else:
                widget_text.write(base_text + "Chunk size user-set to \"Full\"... \n")
        else:
            chunk_set = int(data['chunks'])
            if demucs_only == 'on':
                if no_chunk_demucs:
                    widget_text.write(base_text + "Chunk size user-set to "f"{chunk_set}... \n")
            else:
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
  
        
        if demucs_switch == 'off':
            sources = self.demix_base(segmented_mix, margin_size=margin)
        elif demucs_only == 'on':
            
            if 'tasnet.th' in demucs_model_set or 'tasnet_extra.th' in demucs_model_set or \
            'demucs.th' in demucs_model_set or \
            'demucs_extra.th' in demucs_model_set or 'light.th' in demucs_model_set or \
            'light_extra.th' in demucs_model_set or 'v1' in demucs_model_set or '.gz' in demucs_model_set:
                if no_chunk_demucs == False:
                    sources = self.demix_demucs_v1_split(mix)
                if no_chunk_demucs == True:
                    sources = self.demix_demucs_v1(segmented_mix, margin_size=margin)
            elif 'tasnet-beb46fac.th' in demucs_model_set or 'tasnet_extra-df3777b2.th' in demucs_model_set or \
            'demucs48_hq-28a1282c.th' in demucs_model_set or'demucs-e07c671f.th' in demucs_model_set or \
            'demucs_extra-3646af93.th' in demucs_model_set or 'demucs_unittest-09ebc15f.th' in demucs_model_set or \
            'v2' in demucs_model_set:
                if no_chunk_demucs == False:
                    sources = self.demix_demucs_v2_split(mix)
                if no_chunk_demucs == True:
                    sources = self.demix_demucs_v2(segmented_mix, margin_size=margin)
            else:
                if no_chunk_demucs == False:
                    sources = self.demix_demucs_split(mix)
                if no_chunk_demucs == True:
                    sources = self.demix_demucs(segmented_mix, margin_size=margin)
        else: # both, apply spec effects
            base_out = self.demix_base(segmented_mix, margin_size=margin)
            if 'tasnet.th' in demucs_model_set or 'tasnet_extra.th' in demucs_model_set or \
            'demucs.th' in demucs_model_set or \
            'demucs_extra.th' in demucs_model_set or 'light.th' in demucs_model_set or \
            'light_extra.th' in demucs_model_set or 'v1' in demucs_model_set or '.gz' in demucs_model_set:
                if no_chunk_demucs == False:
                    demucs_out = self.demix_demucs_v1_split(mix)
                if no_chunk_demucs == True:
                    demucs_out = self.demix_demucs_v1(segmented_mix, margin_size=margin)
            elif 'tasnet-beb46fac.th' in demucs_model_set or 'tasnet_extra-df3777b2.th' in demucs_model_set or \
            'demucs48_hq-28a1282c.th' in demucs_model_set or'demucs-e07c671f.th' in demucs_model_set or \
            'demucs_extra-3646af93.th' in demucs_model_set or 'demucs_unittest-09ebc15f.th' in demucs_model_set or \
            'v2' in demucs_model_set:
                if no_chunk_demucs == False:
                    demucs_out = self.demix_demucs_v2_split(mix)
                if no_chunk_demucs == True:
                    demucs_out = self.demix_demucs_v2(segmented_mix, margin_size=margin)
            else:
                if no_chunk_demucs == False:
                    demucs_out = self.demix_demucs_split(mix)
                if no_chunk_demucs == True:
                    demucs_out = self.demix_demucs(segmented_mix, margin_size=margin)
            nan_count = np.count_nonzero(np.isnan(demucs_out)) + np.count_nonzero(np.isnan(base_out))
            if nan_count > 0:
                print('Warning: there are {} nan values in the array(s).'.format(nan_count))
                demucs_out, base_out = np.nan_to_num(demucs_out), np.nan_to_num(base_out)
            sources = {}

            if 'UVR' in demucs_model_set:
                if stemset_n == '(Instrumental)':
                    sources[3] = (spec_effects(wave=[demucs_out[0],base_out[0]],
                                                algorithm=data['mixing'],
                                                value=b[3])*float(compensate)) # compensation
                else:
                    sources[3] = (spec_effects(wave=[demucs_out[1],base_out[0]],
                                                algorithm=data['mixing'],
                                                value=b[3])*float(compensate)) # compensation
            else:
                sources[3] = (spec_effects(wave=[demucs_out[3],base_out[0]],
                                            algorithm=data['mixing'],
                                            value=b[3])*float(compensate)) # compensation
                
        if demucs_switch == 'off':    
            return sources*float(compensate)
        else:
            return sources
    
    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        onnxitera = len(mixes)
        onnxitera_calc = onnxitera * 2
        gui_progress_bar_onnx = 0
        progress_bar = 0
        
        print(' Running ONNX Inference...')

        if onnxitera == 1:
            widget_text.write(base_text + f"Running ONNX Inference... ")
        else:
            widget_text.write(base_text + f"Running ONNX Inference...{space}\n")

        for mix in mixes:
            gui_progress_bar_onnx += 1
            if data['demucsmodel']:
                update_progress(**progress_kwargs,
                    step=(0.1 + (0.5/onnxitera_calc * gui_progress_bar_onnx)))
            else:
                update_progress(**progress_kwargs,
                    step=(0.1 + (0.8/onnxitera * gui_progress_bar_onnx)))
 
            progress_bar += 100
            step = (progress_bar / onnxitera)
            
            if onnxitera == 1:
                pass
            else:
                percent_prog = f"{base_text}MDX-Net Inference Progress: {gui_progress_bar_onnx}/{onnxitera} | {round(step)}%"
                widget_text.percentage(percent_prog)
 
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
        
        if onnxitera == 1:
            widget_text.write('Done!\n')
        else:
            widget_text.write('\n')
        
        return _sources
    
    def demix_demucs(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        progress_bar = 0
        if demucsitera == 1:
            widget_text.write(base_text + f"Running Demucs Inference... ")
        else:
            widget_text.write(base_text + f"Running Demucs Inference...{space}\n")
        
        print(' Running Demucs Inference...')
        for nmix in mix:
            gui_progress_bar_demucs += 1
            progress_bar += 100
            step = (progress_bar / demucsitera)
            if demucsitera == 1:
                pass
            else:
                percent_prog = f"{base_text}Demucs Inference Progress: {gui_progress_bar_demucs}/{demucsitera} | {round(step)}%"
                widget_text.percentage(percent_prog)
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            with torch.no_grad():
                sources = apply_model(self.demucs, cmix[None], 
                                      gui_progress_bar, 
                                      widget_text,
                                      update_prog,
                                      split=split_mode, 
                                      device=device, 
                                      overlap=overlap_set, 
                                      shifts=shift_set, 
                                      progress=False, 
                                      segmen=False,
                                      **progress_demucs_kwargs)[0]
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)
        
        if demucsitera == 1:
            widget_text.write('Done!\n')
        else:
            widget_text.write('\n')
        #print('the demucs model is done running')

        return sources
    
    def demix_demucs_split(self, mix):
        
        if split_mode:
            widget_text.write(base_text + f"Running Demucs Inference...{space}\n")
        else:
            widget_text.write(base_text + f"Running Demucs Inference... ")
        print(' Running Demucs Inference...')
          
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        
        with torch.no_grad():
            sources = apply_model(self.demucs, 
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
            widget_text.write('\n')
        else:
            widget_text.write('Done!\n')
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        
        return sources
    
    def demix_demucs_v1(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        progress_bar = 0
        print(' Running Demucs Inference...')
        if demucsitera == 1:
            widget_text.write(base_text + f"Running Demucs v1 Inference... ")
        else:
            widget_text.write(base_text + f"Running Demucs v1 Inference...{space}\n")
        for nmix in mix:
            gui_progress_bar_demucs += 1
            progress_bar += 100
            step = (progress_bar / demucsitera)
            if demucsitera == 1:
                pass
            else:
                percent_prog = f"{base_text}Demucs v1 Inference Progress: {gui_progress_bar_demucs}/{demucsitera} | {round(step)}%"
                widget_text.percentage(percent_prog)
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            with torch.no_grad():
                sources = apply_model_v1(self.demucs, 
                                         cmix.to(device), 
                                         gui_progress_bar, 
                                         widget_text,
                                         update_prog,
                                         split=split_mode, 
                                         segmen=False,
                                         shifts=shift_set,
                                         **progress_demucs_kwargs)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)

        if demucsitera == 1:
            widget_text.write('Done!\n')
        else:
            widget_text.write('\n')

        return sources
    
    def demix_demucs_v1_split(self, mix):

        print(' Running Demucs Inference...')
        if split_mode:
            widget_text.write(base_text + f"Running Demucs v1 Inference...{space}\n")
        else:
            widget_text.write(base_text + f"Running Demucs v1 Inference... ")
        
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model_v1(self.demucs, 
                                        mix.to(device), 
                                        gui_progress_bar, 
                                        widget_text,
                                        update_prog,
                                        split=split_mode, 
                                        segmen=True,
                                        shifts=shift_set,
                                        **progress_demucs_kwargs)
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]

        if split_mode:
            widget_text.write('\n')
        else:
            widget_text.write('Done!\n')
            
        return sources
    
    def demix_demucs_v2(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        progress_bar = 0
        if demucsitera == 1:
            widget_text.write(base_text + f"Running Demucs v2 Inference... ")
        else:
            widget_text.write(base_text + f"Running Demucs v2 Inference...{space}\n")
            
        for nmix in mix:
            gui_progress_bar_demucs += 1
            progress_bar += 100
            step = (progress_bar / demucsitera)
            if demucsitera == 1:
                pass
            else:
                percent_prog = f"{base_text}Demucs v2 Inference Progress: {gui_progress_bar_demucs}/{demucsitera} | {round(step)}%"
                widget_text.percentage(percent_prog)
            
            update_progress(**progress_kwargs,
                step=(0.35 + (1.05/demucsitera_calc * gui_progress_bar_demucs)))
            cmix = mix[nmix]
            cmix = torch.tensor(cmix, dtype=torch.float32)
            ref = cmix.mean(0)        
            cmix = (cmix - ref.mean()) / ref.std()
            with torch.no_grad():
                sources = apply_model_v2(self.demucs, 
                                         cmix.to(device), 
                                         gui_progress_bar, 
                                         widget_text,
                                         update_prog,
                                         split=split_mode, 
                                         segmen=False,
                                         overlap=overlap_set, 
                                         shifts=shift_set,
                                         **progress_demucs_kwargs)
            sources = (sources * ref.std() + ref.mean()).cpu().numpy()
            sources[[0,1]] = sources[[1,0]]

            start = 0 if nmix == 0 else margin_size
            end = None if nmix == list(mix.keys())[::-1][0] else -margin_size
            if margin_size == 0:
                end = None
            processed[nmix] = sources[:,:,start:end].copy()

        sources = list(processed.values())
        sources = np.concatenate(sources, axis=-1)

        if demucsitera == 1:
            widget_text.write('Done!\n')
        else:
            widget_text.write('\n')

        return sources
    
    def demix_demucs_v2_split(self, mix):
        print(' Running Demucs Inference...')
        
        if split_mode:
            widget_text.write(base_text + f"Running Demucs v2 Inference...{space}\n")
        else:
            widget_text.write(base_text + f"Running Demucs v2 Inference... ")
            
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        with torch.no_grad():
            sources = apply_model_v2(self.demucs, 
                                        mix.to(device), 
                                        gui_progress_bar, 
                                        widget_text,
                                        update_prog,
                                        split=split_mode, 
                                        segmen=True,
                                        overlap=overlap_set, 
                                        shifts=shift_set,
                                        **progress_demucs_kwargs)
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]

        if split_mode:
            widget_text.write('\n')
        else:
            widget_text.write('Done!\n')
            
        return sources

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
    'agg': 10,
    'algo': 'Instrumentals (Min Spec)',
    'appendensem': False,
    'autocompensate': True,
    'chunks': 'auto',
    'compensate': 1.03597672895,
    'demucs_only': False,
    'demucsmodel': False,
    'DemucsModel_MDX': 'UVR_Demucs_Model_1',
    'ensChoose': 'Basic VR Ensemble',
    'export_path': None,
    'gpu': -1,
    'high_end_process': 'mirroring',
    'input_paths': None,
    'inst_only': False,
    'instrumentalModel': None,
    'margin': 44100,
    'mdx_ensem': 'MDX-Net: UVR-MDX-NET 1',
    'mdx_ensem_b': 'No Model',
    'mdx_only_ensem_a': 'MDX-Net: UVR-MDX-NET Main',
    'mdx_only_ensem_b': 'MDX-Net: UVR-MDX-NET 1',
    'mdx_only_ensem_c': 'No Model',
    'mdx_only_ensem_d': 'No Model',
    'mdx_only_ensem_e': 'No Model',
    'mixing': 'Default',
    'mp3bit': '320k',
    'no_chunk': False,
    'noise_pro_select': 'Auto Select',
    'noisereduc_s': 3,
    'non_red': False,
    'normalize': False,
    'output_image': True,
    'overlap': 0.5,
    'postprocess': True,
    'saveFormat': 'wav',
    'segment': 'Default',
    'shifts': 0,
    'split_mode': False,
    'tta': True,
    'useModel': None,
    'voc_only': False,
    'vr_ensem': '2_HP-UVR',
    'vr_ensem_a': '1_HP-UVR',
    'vr_ensem_b': '2_HP-UVR',
    'vr_ensem_c': 'No Model',
    'vr_ensem_d': 'No Model',
    'vr_ensem_e': 'No Model',
    'vr_ensem_mdx_a': 'No Model',
    'vr_ensem_mdx_b': 'No Model',
    'vr_ensem_mdx_c': 'No Model',
    'vr_multi_USER_model_param_1': 'Auto',
    'vr_multi_USER_model_param_2': 'Auto',
    'vr_multi_USER_model_param_3': 'Auto',
    'vr_multi_USER_model_param_4': 'Auto',
    'vr_basic_USER_model_param_1': 'Auto',
    'vr_basic_USER_model_param_2': 'Auto',
    'vr_basic_USER_model_param_3': 'Auto',
    'vr_basic_USER_model_param_4': 'Auto',
    'vr_basic_USER_model_param_5': 'Auto',
    'wavtype': 'PCM_16',
    'window_size': 512
}

default_window_size = data['window_size']
default_agg = data['agg']
default_chunks = data['chunks']
default_noisereduc_s = data['noisereduc_s']


def update_progress(progress_var, total_files, file_num, step: float = 1):
    """Calculate the progress for the progress widget in the GUI"""
    
    total_count = model_count * total_files
    base = (100 / total_count)
    progress = base * current_model_bar - base
    progress += base * step

    progress_var.set(progress)

def get_baseText(total_files, file_num):
    """Create the base text for the command widget"""
    text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                total_files=total_files)
    
    return text

def main(window: tk.Wm, 
         text_widget: tk.Text, 
         button_widget: tk.Button, 
         progress_var: tk.Variable,
         stop_thread,
         **kwargs: dict):
    
    global widget_text
    global gui_progress_bar
    global music_file
    global default_chunks
    global default_noisereduc_s
    global gui_progress_bar
    global base_name
    global progress_kwargs
    global base_text
    global modeltype
    global model_set
    global model_set_name
    global ModelName_2
    global compensate
    global autocompensate
    global demucs_model_set
    global progress_demucs_kwargs
    global channel_set
    global margin_set
    global overlap_set
    global shift_set
    global noise_pro_set
    global n_fft_scale_set
    global dim_f_set
    global split_mode
    global demucs_switch
    global demucs_only
    global no_chunk_demucs
    global wav_type_set
    global flac_type_set
    global mp3_bit_set
    global model_hash
    global space
    global stime
    global stemset_n
    global source_val
    global widget_button
    global stop_button
    
    wav_type_set = data['wavtype']
    
    # Update default settings
    default_chunks = data['chunks']
    default_noisereduc_s = data['noisereduc_s']
    autocompensate = data['autocompensate']
    
    stop_button = stop_thread
    widget_text = text_widget
    gui_progress_bar = progress_var
    widget_button = button_widget
    space = ' '*90
    
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
    demucs_model_missing_err = "is neither a single pre-trained model or a bag of models."
    
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
        33966, 123821, 123812, 129605, 537238, 537227 # custom
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
                 normalization_set(wav_instrument), mp.param['sr'], subtype=wav_type_set)
        
        # -Save files-
        # Instrumental
        if instrumental_name is not None:
            instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(base_name)}_{ModelName_1}_{instrumental_name}',
            )
             
        if VModel in ModelName_1 and data['voc_only']:
                sf.write(instrumental_path,
                        normalization_set(wav_instrument), mp.param['sr'], subtype=wav_type_set)
        elif VModel in ModelName_1 and data['inst_only']:
            pass
        elif data['voc_only']:
            pass
        else:
                sf.write(instrumental_path,
                        normalization_set(wav_instrument), mp.param['sr'], subtype=wav_type_set)
                
        # Vocal
        if vocal_name is not None:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name=f'{os.path.basename(base_name)}_{ModelName_1}_{vocal_name}',
            )
            
            if VModel in ModelName_1 and data['inst_only']:
                sf.write(vocal_path,
                            normalization_set(wav_vocals), mp.param['sr'], subtype=wav_type_set)
            elif VModel in ModelName_1 and data['voc_only']:
                pass
            elif data['inst_only']:
                pass
            else:
                sf.write(vocal_path,
                            normalization_set(wav_vocals), mp.param['sr'], subtype=wav_type_set)

    data.update(kwargs)
    
    global update_prog
    
    update_prog = update_progress
    no_chunk_demucs = data['no_chunk']
    space = ' '*90

    if data['DemucsModel_MDX'] == "Tasnet v1":
        demucs_model_set_name = 'tasnet.th'
    elif data['DemucsModel_MDX'] == "Tasnet_extra v1":
        demucs_model_set_name = 'tasnet_extra.th'
    elif data['DemucsModel_MDX'] == "Demucs v1":
        demucs_model_set_name = 'demucs.th'
    elif data['DemucsModel_MDX'] == "Demucs v1.gz":
        demucs_model_set_name = 'demucs.th.gz'
    elif data['DemucsModel_MDX'] == "Demucs_extra v1":
        demucs_model_set_name = 'demucs_extra.th'
    elif data['DemucsModel_MDX'] == "Demucs_extra v1.gz":
        demucs_model_set_name = 'demucs_extra.th.gz'
    elif data['DemucsModel_MDX'] == "Light v1":
        demucs_model_set_name = 'light.th'
    elif data['DemucsModel_MDX'] == "Light v1.gz":
        demucs_model_set_name = 'light.th.gz'
    elif data['DemucsModel_MDX'] == "Light_extra v1":
        demucs_model_set_name = 'light_extra.th'
    elif data['DemucsModel_MDX'] == "Light_extra v1.gz":
        demucs_model_set_name = 'light_extra.th.gz'
    elif data['DemucsModel_MDX'] == "Tasnet v2":
        demucs_model_set_name = 'tasnet-beb46fac.th'
    elif data['DemucsModel_MDX'] == "Tasnet_extra v2":
        demucs_model_set_name = 'tasnet_extra-df3777b2.th'
    elif data['DemucsModel_MDX'] == "Demucs48_hq v2":
        demucs_model_set_name = 'demucs48_hq-28a1282c.th'
    elif data['DemucsModel_MDX'] == "Demucs v2":
        demucs_model_set_name = 'demucs-e07c671f.th'
    elif data['DemucsModel_MDX'] == "Demucs_extra v2":
        demucs_model_set_name = 'demucs_extra-3646af93.th'
    elif data['DemucsModel_MDX'] == "Demucs_unittest v2":
        demucs_model_set_name = 'demucs_unittest-09ebc15f.th'
    elif '.ckpt' in data['DemucsModel_MDX'] and 'v2' in data['DemucsModel_MDX']:
        demucs_model_set_name = data['DemucsModel_MDX']
    elif '.ckpt' in data['DemucsModel_MDX'] and 'v1' in data['DemucsModel_MDX']:
        demucs_model_set_name = data['DemucsModel_MDX']
    else:
        demucs_model_set_name = data['DemucsModel_MDX']
        
    if data['mdx_ensem'] == "Demucs: Tasnet v1":
        demucs_model_set_name_muilti_a = 'tasnet.th'
    elif data['mdx_ensem'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_muilti_a = 'tasnet_extra.th'
    elif data['mdx_ensem'] == "Demucs: Demucs v1":
        demucs_model_set_name_muilti_a = 'demucs.th'
    elif data['mdx_ensem'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_muilti_a = 'demucs_extra.th'
    elif data['mdx_ensem'] == "Demucs: Light v1":
        demucs_model_set_name_muilti_a = 'light.th'
    elif data['mdx_ensem'] == "Demucs: Light_extra v1":
        demucs_model_set_name_muilti_a = 'light_extra.th'
    elif data['mdx_ensem'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_muilti_a = 'demucs.th.gz'
    elif data['mdx_ensem'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_muilti_a = 'demucs_extra.th.gz'
    elif data['mdx_ensem'] == "Demucs: Light v1.gz":
        demucs_model_set_name_muilti_a = 'light.th.gz'
    elif data['mdx_ensem'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_muilti_a = 'light_extra.th.gz'
    elif data['mdx_ensem'] == "Demucs: Tasnet v2":
        demucs_model_set_name_muilti_a = 'tasnet-beb46fac.th'
    elif data['mdx_ensem'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_muilti_a = 'tasnet_extra-df3777b2.th'
    elif data['mdx_ensem'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_muilti_a = 'demucs48_hq-28a1282c.th'
    elif data['mdx_ensem'] == "Demucs: Demucs v2":
        demucs_model_set_name_muilti_a = 'demucs-e07c671f.th'
    elif data['mdx_ensem'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_muilti_a = 'demucs_extra-3646af93.th'
    elif data['mdx_ensem'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_muilti_a = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_ensem'] == "Demucs: mdx_extra":
        demucs_model_set_name_muilti_a = 'mdx_extra'
    elif data['mdx_ensem'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_muilti_a = 'mdx_extra_q'
    elif data['mdx_ensem'] == "Demucs: mdx":
        demucs_model_set_name_muilti_a = 'mdx'
    elif data['mdx_ensem'] == "Demucs: mdx_q":
        demucs_model_set_name_muilti_a = 'mdx_q'
    elif data['mdx_ensem'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_muilti_a = 'UVR_Demucs_Model_1'
    elif data['mdx_ensem'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_muilti_a = 'UVR_Demucs_Model_2'
    elif data['mdx_ensem'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_muilti_a = 'UVR_Demucs_Model_Bag'
        
    else:
        demucs_model_set_name_muilti_a = data['mdx_ensem']
        
    if data['mdx_ensem_b'] == "Demucs: Tasnet v1":
        demucs_model_set_name_muilti_b = 'tasnet.th'
    elif data['mdx_ensem_b'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_muilti_b = 'tasnet_extra.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs v1":
        demucs_model_set_name_muilti_b = 'demucs.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_muilti_b = 'demucs_extra.th'
    elif data['mdx_ensem_b'] == "Demucs: Light v1":
        demucs_model_set_name_muilti_b = 'light.th'
    elif data['mdx_ensem_b'] == "Demucs: Light_extra v1":
        demucs_model_set_name_muilti_b = 'light_extra.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_muilti_b = 'demucs.th.gz'
    elif data['mdx_ensem_b'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_muilti_b = 'demucs_extra.th.gz'
    elif data['mdx_ensem_b'] == "Demucs: Light v1.gz":
        demucs_model_set_name_muilti_b = 'light.th.gz'
    elif data['mdx_ensem_b'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_muilti_b = 'light_extra.th.gz'
    elif data['mdx_ensem_b'] == "Demucs: Tasnet v2":
        demucs_model_set_name_muilti_b = 'tasnet-beb46fac.th'
    elif data['mdx_ensem_b'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_muilti_b = 'tasnet_extra-df3777b2.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_muilti_b = 'demucs48_hq-28a1282c.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs v2":
        demucs_model_set_name_muilti_b = 'demucs-e07c671f.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_muilti_b = 'demucs_extra-3646af93.th'
    elif data['mdx_ensem_b'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_muilti_b = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_ensem_b'] == "Demucs: mdx_extra":
        demucs_model_set_name_muilti_b = 'mdx_extra'
    elif data['mdx_ensem_b'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_muilti_b = 'mdx_extra_q'
    elif data['mdx_ensem_b'] == "Demucs: mdx":
        demucs_model_set_name_muilti_b = 'mdx'
    elif data['mdx_ensem_b'] == "Demucs: mdx_q":
        demucs_model_set_name_muilti_b = 'mdx_q'
    elif data['mdx_ensem_b'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_muilti_b = 'UVR_Demucs_Model_1'
    elif data['mdx_ensem_b'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_muilti_b = 'UVR_Demucs_Model_2'
    elif data['mdx_ensem_b'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_muilti_b = 'UVR_Demucs_Model_Bag' 
    else:
        demucs_model_set_name_muilti_b = data['mdx_ensem_b']
        
    if data['mdx_only_ensem_a'] == "Demucs: Tasnet v1":
        demucs_model_set_name_a = 'tasnet.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_a = 'tasnet_extra.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs v1":
        demucs_model_set_name_a = 'demucs.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_a = 'demucs_extra.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Light v1":
        demucs_model_set_name_a = 'light.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Light_extra v1":
        demucs_model_set_name_a = 'light_extra.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_a = 'demucs.th.gz'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_a = 'demucs_extra.th.gz'
    elif data['mdx_only_ensem_a'] == "Demucs: Light v1.gz":
        demucs_model_set_name_a = 'light.th.gz'
    elif data['mdx_only_ensem_a'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_a = 'light_extra.th.gz'
    elif data['mdx_only_ensem_a'] == "Demucs: Tasnet v2":
        demucs_model_set_name_a = 'tasnet-beb46fac.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_a = 'tasnet_extra-df3777b2.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_a = 'demucs48_hq-28a1282c.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs v2":
        demucs_model_set_name_a = 'demucs-e07c671f.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_a = 'demucs_extra-3646af93.th'
    elif data['mdx_only_ensem_a'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_a = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_only_ensem_a'] == "Demucs: mdx_extra":
        demucs_model_set_name_a = 'mdx_extra'
    elif data['mdx_only_ensem_a'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_a = 'mdx_extra_q'
    elif data['mdx_only_ensem_a'] == "Demucs: mdx":
        demucs_model_set_name_a = 'mdx'
    elif data['mdx_only_ensem_a'] == "Demucs: mdx_q":
        demucs_model_set_name_a = 'mdx_q'
    elif data['mdx_only_ensem_a'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_a = 'UVR_Demucs_Model_1'
    elif data['mdx_only_ensem_a'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_a = 'UVR_Demucs_Model_2'
    elif data['mdx_only_ensem_a'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_a = 'UVR_Demucs_Model_Bag'
        
    else:
        demucs_model_set_name_a = data['mdx_only_ensem_a']
        
        
    if data['mdx_only_ensem_b'] == "Demucs: Tasnet v1":
        demucs_model_set_name_b = 'tasnet.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_b = 'tasnet_extra.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs v1":
        demucs_model_set_name_b = 'demucs.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_b = 'demucs_extra.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Light v1":
        demucs_model_set_name_b = 'light.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Light_extra v1":
        demucs_model_set_name_b = 'light_extra.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_b = 'demucs.th.gz'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_b = 'demucs_extra.th.gz'
    elif data['mdx_only_ensem_b'] == "Demucs: Light v1.gz":
        demucs_model_set_name_b = 'light.th.gz'
    elif data['mdx_only_ensem_b'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_b = 'light_extra.th.gz'
    elif data['mdx_only_ensem_b'] == "Demucs: Tasnet v2":
        demucs_model_set_name_b = 'tasnet-beb46fac.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_b = 'tasnet_extra-df3777b2.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_b = 'demucs48_hq-28a1282c.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs v2":
        demucs_model_set_name_b = 'demucs-e07c671f.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_b = 'demucs_extra-3646af93.th'
    elif data['mdx_only_ensem_b'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_b = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_only_ensem_b'] == "Demucs: mdx_extra":
        demucs_model_set_name_b = 'mdx_extra'
    elif data['mdx_only_ensem_b'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_b = 'mdx_extra_q'
    elif data['mdx_only_ensem_b'] == "Demucs: mdx":
        demucs_model_set_name_b = 'mdx'
    elif data['mdx_only_ensem_b'] == "Demucs: mdx_q":
        demucs_model_set_name_b = 'mdx_q'
    elif data['mdx_only_ensem_b'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_b = 'UVR_Demucs_Model_1'
    elif data['mdx_only_ensem_b'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_b = 'UVR_Demucs_Model_2'
    elif data['mdx_only_ensem_b'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_b = 'UVR_Demucs_Model_Bag'
        
    else:
        demucs_model_set_name_b = data['mdx_only_ensem_b']
        
    if data['mdx_only_ensem_c'] == "Demucs: Tasnet v1":
        demucs_model_set_name_c = 'tasnet.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_c = 'tasnet_extra.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs v1":
        demucs_model_set_name_c = 'demucs.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_c = 'demucs_extra.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Light v1":
        demucs_model_set_name_c = 'light.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Light_extra v1":
        demucs_model_set_name_c = 'light_extra.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_c = 'demucs.th.gz'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_c = 'demucs_extra.th.gz'
    elif data['mdx_only_ensem_c'] == "Demucs: Light v1.gz":
        demucs_model_set_name_c = 'light.th.gz'
    elif data['mdx_only_ensem_c'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_c = 'light_extra.th.gz'
    elif data['mdx_only_ensem_c'] == "Demucs: Tasnet v2":
        demucs_model_set_name_c = 'tasnet-beb46fac.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_c = 'tasnet_extra-df3777b2.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_c = 'demucs48_hq-28a1282c.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs v2":
        demucs_model_set_name_c = 'demucs-e07c671f.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_c = 'demucs_extra-3646af93.th'
    elif data['mdx_only_ensem_c'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_c = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_only_ensem_c'] == "Demucs: mdx_extra":
        demucs_model_set_name_c = 'mdx_extra'
    elif data['mdx_only_ensem_c'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_c = 'mdx_extra_q'
    elif data['mdx_only_ensem_c'] == "Demucs: mdx":
        demucs_model_set_name_c = 'mdx'
    elif data['mdx_only_ensem_c'] == "Demucs: mdx_q":
        demucs_model_set_name_c = 'mdx_q'
    elif data['mdx_only_ensem_c'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_c = 'UVR_Demucs_Model_1'
    elif data['mdx_only_ensem_c'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_c = 'UVR_Demucs_Model_2'
    elif data['mdx_only_ensem_c'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_c = 'UVR_Demucs_Model_Bag'
        
    else:
        demucs_model_set_name_c = data['mdx_only_ensem_c']
        
    if data['mdx_only_ensem_d'] == "Demucs: Tasnet v1":
        demucs_model_set_name_d = 'tasnet.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_d = 'tasnet_extra.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs v1":
        demucs_model_set_name_d = 'demucs.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_d = 'demucs_extra.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Light v1":
        demucs_model_set_name_d = 'light.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Light_extra v1":
        demucs_model_set_name_d = 'light_extra.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_d = 'demucs.th.gz'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_d = 'demucs_extra.th.gz'
    elif data['mdx_only_ensem_d'] == "Demucs: Light v1.gz":
        demucs_model_set_name_d = 'light.th.gz'
    elif data['mdx_only_ensem_d'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_d = 'light_extra.th.gz'
    elif data['mdx_only_ensem_d'] == "Demucs: Tasnet v2":
        demucs_model_set_name_d = 'tasnet-beb46fac.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_d = 'tasnet_extra-df3777b2.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_d = 'demucs48_hq-28a1282c.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs v2":
        demucs_model_set_name_d = 'demucs-e07c671f.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_d = 'demucs_extra-3646af93.th'
    elif data['mdx_only_ensem_d'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_d = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_only_ensem_d'] == "Demucs: mdx_extra":
        demucs_model_set_name_d = 'mdx_extra'
    elif data['mdx_only_ensem_d'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_d = 'mdx_extra_q'
    elif data['mdx_only_ensem_d'] == "Demucs: mdx":
        demucs_model_set_name_d = 'mdx'
    elif data['mdx_only_ensem_d'] == "Demucs: mdx_q":
        demucs_model_set_name_d = 'mdx_q'
    elif data['mdx_only_ensem_d'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_d = 'UVR_Demucs_Model_1'
    elif data['mdx_only_ensem_d'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_d = 'UVR_Demucs_Model_2'
    elif data['mdx_only_ensem_d'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_d = 'UVR_Demucs_Model_Bag' 
    else:
        demucs_model_set_name_d = data['mdx_only_ensem_d']
        
    if data['mdx_only_ensem_e'] == "Demucs: Tasnet v1":
        demucs_model_set_name_e = 'tasnet.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Tasnet_extra v1":
        demucs_model_set_name_e = 'tasnet_extra.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs v1":
        demucs_model_set_name_e = 'demucs.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs_extra v1":
        demucs_model_set_name_e = 'demucs_extra.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Light v1":
        demucs_model_set_name_e = 'light.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Light_extra v1":
        demucs_model_set_name_e = 'light_extra.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs v1.gz":
        demucs_model_set_name_e = 'demucs.th.gz'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs_extra v1.gz":
        demucs_model_set_name_e = 'demucs_extra.th.gz'
    elif data['mdx_only_ensem_e'] == "Demucs: Light v1.gz":
        demucs_model_set_name_e = 'light.th.gz'
    elif data['mdx_only_ensem_e'] == "Demucs: Light_extra v1.gz":
        demucs_model_set_name_e = 'light_extra.th.gz'
    elif data['mdx_only_ensem_e'] == "Demucs: Tasnet v2":
        demucs_model_set_name_e = 'tasnet-beb46fac.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Tasnet_extra v2":
        demucs_model_set_name_e = 'tasnet_extra-df3777b2.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs48_hq v2":
        demucs_model_set_name_e = 'demucs48_hq-28a1282c.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs v2":
        demucs_model_set_name_e = 'demucs-e07c671f.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs_extra v2":
        demucs_model_set_name_e = 'demucs_extra-3646af93.th'
    elif data['mdx_only_ensem_e'] == "Demucs: Demucs_unittest v2":
        demucs_model_set_name_e = 'demucs_unittest-09ebc15f.th'
    elif data['mdx_only_ensem_e'] == "Demucs: mdx_extra":
        demucs_model_set_name_e = 'mdx_extra'
    elif data['mdx_only_ensem_e'] == "Demucs: mdx_extra_q":
        demucs_model_set_name_e = 'mdx_extra_q'
    elif data['mdx_only_ensem_e'] == "Demucs: mdx":
        demucs_model_set_name_e = 'mdx'
    elif data['mdx_only_ensem_e'] == "Demucs: mdx_q":
        demucs_model_set_name_e = 'mdx_q'
    elif data['mdx_only_ensem_e'] == "Demucs: UVR_Demucs_Model_1":
        demucs_model_set_name_e = 'UVR_Demucs_Model_1'
    elif data['mdx_only_ensem_e'] == "Demucs: UVR_Demucs_Model_2":
        demucs_model_set_name_e = 'UVR_Demucs_Model_2'
    elif data['mdx_only_ensem_e'] == "Demucs: UVR_Demucs_Model_Bag":
        demucs_model_set_name_e = 'UVR_Demucs_Model_Bag'
        
    else:
        demucs_model_set_name_e = data['mdx_only_ensem_e']


    # Update default settings
    global default_window_size
    global default_agg
    global normalization_set
    
    default_window_size = data['window_size']
    default_agg = data['agg']

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

    #print('Do all of the HP models exist? ' + hp2_ens)

    # Separation Preperation
    try:    #Ensemble Dictionary

        overlap_set = float(data['overlap'])
        channel_set = int(data['channel'])
        margin_set = int(data['margin'])
        shift_set = int(data['shifts'])
        demucs_model_set = demucs_model_set_name
        split_mode = data['split_mode']
        
        
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

        if not data['ensChoose'] == 'Manual Ensemble':
            
                ##### Basic VR Ensemble #####
            
                #1st Model
              
                vr_ensem_a_name = data['vr_ensem_a']
                vr_ensem_a = f'models/Main_Models/{vr_ensem_a_name}.pth'
                vr_param_ens_a = data['vr_basic_USER_model_param_1']
                    
                #2nd Model
                       
                vr_ensem_b_name = data['vr_ensem_b']
                vr_ensem_b = f'models/Main_Models/{vr_ensem_b_name}.pth'
                vr_param_ens_b = data['vr_basic_USER_model_param_2']
                    
                #3rd Model
                    
                vr_ensem_c_name = data['vr_ensem_c']
                vr_ensem_c = f'models/Main_Models/{vr_ensem_c_name}.pth'
                vr_param_ens_c = data['vr_basic_USER_model_param_3']
                     
                #4th Model
                  
                vr_ensem_d_name = data['vr_ensem_d']
                vr_ensem_d = f'models/Main_Models/{vr_ensem_d_name}.pth'
                vr_param_ens_d = data['vr_basic_USER_model_param_4']
                    
                # 5th Model
                 
                vr_ensem_e_name = data['vr_ensem_e']
                vr_ensem_e = f'models/Main_Models/{vr_ensem_e_name}.pth'
                vr_param_ens_e = data['vr_basic_USER_model_param_5']
                
                basic_vr_ensemble_list = [vr_ensem_a_name, vr_ensem_b_name, vr_ensem_c_name, vr_ensem_d_name, vr_ensem_e_name]
                no_models = basic_vr_ensemble_list.count('No Model')
                vr_ensem_count = 5 - no_models
 
                if data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/2'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/2'
                        }
                    ] 
                elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_d'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_param': vr_param_ens_e,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 3/3'
                        }  
                    ] 
                elif data['vr_ensem_c'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_param': vr_param_ens_d,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 3/3' 
                        }  
                    ] 
                elif data['vr_ensem_d'] == 'No Model' and data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/3'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/3'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_param': vr_param_ens_c,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/3'
                        }  
                    ] 
                elif data['vr_ensem_d'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_param': vr_param_ens_c,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_param': vr_param_ens_e,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ]
                    
                elif data['vr_ensem_c'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_param': vr_param_ens_d,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_param': vr_param_ens_e,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ] 
                elif data['vr_ensem_e'] == 'No Model':
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/4'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/4'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_param': vr_param_ens_c,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/4'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_param': vr_param_ens_d,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 4/4'
                        }
                    ] 
                else:
                    Basic_Ensem = [
                        {
                            'model_name': vr_ensem_a_name,
                            'model_name_c':vr_ensem_a_name,
                            'model_param': vr_param_ens_a,
                            'model_location': vr_ensem_a,
                            'loop_name': 'Ensemble Mode - Model 1/5'
                        },
                        {
                            'model_name': vr_ensem_b_name,
                            'model_name_c':vr_ensem_b_name,
                            'model_param': vr_param_ens_b,
                            'model_location': vr_ensem_b,
                            'loop_name': 'Ensemble Mode - Model 2/5'
                        },
                        {
                            'model_name': vr_ensem_c_name,
                            'model_name_c':vr_ensem_c_name,
                            'model_param': vr_param_ens_c,
                            'model_location': vr_ensem_c,
                            'loop_name': 'Ensemble Mode - Model 3/5'
                        },
                        {
                            'model_name': vr_ensem_d_name,
                            'model_name_c':vr_ensem_d_name,
                            'model_param': vr_param_ens_d,
                            'model_location': vr_ensem_d,
                            'loop_name': 'Ensemble Mode - Model 4/5' 
                        },
                        {
                            'model_name': vr_ensem_e_name,
                            'model_name_c':vr_ensem_e_name,
                            'model_param': vr_param_ens_e,
                            'model_location': vr_ensem_e,
                            'loop_name': 'Ensemble Mode - Model 5/5'
                        }  
                    ] 
                
                ##### Multi-AI Ensemble #####

                #VR Model 1
                
                vr_ensem_name = data['vr_ensem']
                vr_ensem = f'models/Main_Models/{vr_ensem_name}.pth'
                vr_param_ens_multi = data['vr_multi_USER_model_param_1']
                
                #VR Model 2
                
                vr_ensem_mdx_a_name = data['vr_ensem_mdx_a']
                vr_ensem_mdx_a = f'models/Main_Models/{vr_ensem_mdx_a_name}.pth'
                vr_param_ens_multi_a = data['vr_multi_USER_model_param_2']

                #VR Model 3
                
                vr_ensem_mdx_b_name = data['vr_ensem_mdx_b']
                vr_ensem_mdx_b = f'models/Main_Models/{vr_ensem_mdx_b_name}.pth'
                vr_param_ens_multi_b = data['vr_multi_USER_model_param_3']
                
                #VR Model 4
                
                vr_ensem_mdx_c_name = data['vr_ensem_mdx_c']
                vr_ensem_mdx_c = f'models/Main_Models/{vr_ensem_mdx_c_name}.pth'
                vr_param_ens_multi_c = data['vr_multi_USER_model_param_4']    
                                                          
                #MDX-Net/Demucs Model 1
                
                if 'MDX-Net:' in data['mdx_ensem']:
                    mdx_model_run_mul_a = 'yes'
                    mdx_net_model_name = data['mdx_ensem']
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_ensem = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_ensem = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_ensem = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_ensem = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_ensem = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_ensem = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_ensem = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_ensem = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_ensem = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_ensem = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_ensem = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_ensem = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_ensem']:
                    mdx_model_run_mul_a = 'no'
                    mdx_ensem = demucs_model_set_name_muilti_a
                    
                if data['mdx_ensem'] == 'No Model':
                    mdx_ensem = 'pass'
                    mdx_model_run_mul_a = 'pass'

                #MDX-Net/Demucs Model 2
                
                if 'MDX-Net:' in data['mdx_ensem_b']:
                    mdx_model_run_mul_b = 'yes'
                    mdx_net_model_name = data['mdx_ensem_b']
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_ensem_b = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_ensem_b = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_ensem_b = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_ensem_b = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_ensem_b = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_ensem_b = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_ensem_b = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_ensem_b = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_ensem_b = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_ensem_b = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_ensem_b = 'UVR_MDXNET_Inst_2'
                        
                    else:
                        mdx_ensem_b = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_ensem_b']:
                    mdx_model_run_mul_b = 'no'
                    mdx_ensem_b = demucs_model_set_name_muilti_b
                    
                if data['mdx_ensem_b'] == 'No Model':
                    mdx_ensem_b = 'pass'
                    mdx_model_run_mul_b = 'pass'
                    
                multi_ai_ensemble_list = [vr_ensem_name, vr_ensem_mdx_a_name, vr_ensem_mdx_b_name, vr_ensem_mdx_c_name, data['mdx_ensem'], data['mdx_ensem_b']]
                no_multi_models = multi_ai_ensemble_list.count('No Model')
                multi_ensem_count = 6 - no_multi_models
                
                if data['vr_ensem'] == 'No Model' and data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_ensem}',
                        },
                        {
                            'model_name': 'pass',
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': 'pass',
                            'model_param': 'pass',
                            'model_location':'pass',
                            'loop_name': f'Ensemble Mode - Last Model - {mdx_ensem_b}',
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}',
                        },
                        {
                            'model_name': 'pass',
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': 'pass',
                            'model_param': 'pass',
                            'model_location':'pass',
                            'loop_name': 'Ensemble Mode - Last Model',
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_b'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_param': vr_param_ens_multi_c,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_param': vr_param_ens_multi_b,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                    ]

                elif data['vr_ensem_mdx_b'] == 'No Model' and data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_param': vr_param_ens_multi_a,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_a'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'mdx_model_run': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_param': vr_param_ens_multi_b,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_param': vr_param_ens_multi_c,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_b'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'mdx_model_run': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_param': vr_param_ens_multi_a,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_param': vr_param_ens_multi_c,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                elif data['vr_ensem_mdx_c'] == 'No Model':
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'mdx_model_run': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_param': vr_param_ens_multi_a,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_param': vr_param_ens_multi_b,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        }
                    ]
                else:
                    mdx_vr = [
                        {
                            'model_name': vr_ensem_name,
                            'mdx_model_name': 'pass',
                            'mdx_model_run': 'pass',
                            'model_name_c': vr_ensem_name,
                            'model_param': vr_param_ens_multi,
                            'model_location':vr_ensem,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_a_name,
                            'mdx_model_name': 'pass',
                            'mdx_model_run': 'pass',
                            'model_name_c': vr_ensem_mdx_a_name,
                            'model_param': vr_param_ens_multi_a,
                            'model_location':vr_ensem_mdx_a,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_a_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_b_name,
                            'mdx_model_name': mdx_ensem_b,
                            'mdx_model_run': mdx_model_run_mul_b,
                            'model_name_c': vr_ensem_mdx_b_name,
                            'model_param': vr_param_ens_multi_b,
                            'model_location':vr_ensem_mdx_b,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_b_name}'
                        },
                        {
                            'model_name': vr_ensem_mdx_c_name,
                            'mdx_model_name': mdx_ensem,
                            'mdx_model_run': mdx_model_run_mul_a,
                            'model_name_c': vr_ensem_mdx_c_name,
                            'model_param': vr_param_ens_multi_c,
                            'model_location':vr_ensem_mdx_c,
                            'loop_name': f'Ensemble Mode - Running Model - {vr_ensem_mdx_c_name}'
                        }
                    ]
                    
                ##### Basic MD Ensemble #####
                    
                #MDX-Net/Demucs Model 1

                if 'MDX-Net:' in data['mdx_only_ensem_a']:
                    mdx_model_run_a = 'yes'
                    mdx_net_model_name = str(data['mdx_only_ensem_a'])
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #print('mdx_net_model_name ', mdx_net_model_name)
                    #mdx_only_ensem_a = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_only_ensem_a = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_only_ensem_a = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_only_ensem_a = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_only_ensem_a = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_only_ensem_a = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_only_ensem_a = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_only_ensem_a = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_only_ensem_a = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_only_ensem_a = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_only_ensem_a = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_only_ensem_a = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_only_ensem_a']:
                    mdx_model_run_a = 'no'
                    mdx_only_ensem_a = demucs_model_set_name_a
                    
                if data['mdx_only_ensem_a'] == 'No Model':
                    mdx_model_run_a = 'no'
                    mdx_only_ensem_a = 'pass'

                #MDX-Net/Demucs Model 2
                
                if 'MDX-Net:' in data['mdx_only_ensem_b']:
                    mdx_model_run_b = 'yes'
                    mdx_net_model_name = str(data['mdx_only_ensem_b'])
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_only_ensem_b = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_only_ensem_b = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_only_ensem_b = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_only_ensem_b = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_only_ensem_b = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_only_ensem_b = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_only_ensem_b = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_only_ensem_b = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_only_ensem_b = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_only_ensem_b = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_only_ensem_b = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_only_ensem_b = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_only_ensem_b']:
                    mdx_model_run_b = 'no'
                    mdx_only_ensem_b = demucs_model_set_name_b
                    
                if data['mdx_only_ensem_b'] == 'No Model':
                    mdx_model_run_b = 'no'
                    mdx_only_ensem_b = 'pass'
                    
                #MDX-Net/Demucs Model 3
                
                if 'MDX-Net:' in data['mdx_only_ensem_c']:
                    mdx_model_run_c = 'yes'
                    mdx_net_model_name = data['mdx_only_ensem_c']
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_only_ensem_c = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_only_ensem_c = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_only_ensem_c = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_only_ensem_c = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_only_ensem_c = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_only_ensem_c = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_only_ensem_c = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_only_ensem_c = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_only_ensem_c = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_only_ensem_c = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_only_ensem_c = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_only_ensem_c = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_only_ensem_c']:
                    mdx_model_run_c = 'no'
                    mdx_only_ensem_c = demucs_model_set_name_c
                    
                if data['mdx_only_ensem_c'] == 'No Model':
                    mdx_model_run_c = 'no'
                    mdx_only_ensem_c = 'pass'
                    
                #MDX-Net/Demucs Model 4
                    
                if 'MDX-Net:' in data['mdx_only_ensem_d']:
                    mdx_model_run_d = 'yes'
                    mdx_net_model_name = data['mdx_only_ensem_d']
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_only_ensem_d = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_only_ensem_d = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_only_ensem_d = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_only_ensem_d = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_only_ensem_d = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_only_ensem_d = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_only_ensem_d = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_only_ensem_d = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_only_ensem_d = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_only_ensem_d = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_only_ensem_d = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_only_ensem_d = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_only_ensem_d']:
                    mdx_model_run_d = 'no'
                    mdx_only_ensem_d = demucs_model_set_name_d
                    
                if data['mdx_only_ensem_d'] == 'No Model':
                    mdx_model_run_d = 'no'
                    mdx_only_ensem_d = 'pass'
                    
                #MDX-Net/Demucs Model 5
                
                if 'MDX-Net:' in data['mdx_only_ensem_e']:
                    mdx_model_run_e = 'yes'
                    mdx_net_model_name = data['mdx_only_ensem_e']
                    head, sep, tail = mdx_net_model_name.partition('MDX-Net: ')
                    mdx_net_model_name = tail
                    #mdx_only_ensem_e = mdx_net_model_name
                    if mdx_net_model_name == 'UVR-MDX-NET 1':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx'):
                            mdx_only_ensem_e = 'UVR_MDXNET_1_9703'
                        else:
                            mdx_only_ensem_e = 'UVR_MDXNET_9703'
                    elif mdx_net_model_name == 'UVR-MDX-NET 2':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx'):
                            mdx_only_ensem_e = 'UVR_MDXNET_2_9682'
                        else:
                            mdx_only_ensem_e = 'UVR_MDXNET_9682'
                    elif mdx_net_model_name == 'UVR-MDX-NET 3':
                        if os.path.isfile('models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx'):
                            mdx_only_ensem_e = 'UVR_MDXNET_3_9662'
                        else:
                            mdx_only_ensem_e = 'UVR_MDXNET_9662'
                    elif mdx_net_model_name == 'UVR-MDX-NET Karaoke':
                        mdx_only_ensem_e = 'UVR_MDXNET_KARA'
                    elif mdx_net_model_name == 'UVR-MDX-NET Main':
                        mdx_only_ensem_e = 'UVR_MDXNET_Main'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 1':
                        mdx_only_ensem_e = 'UVR_MDXNET_Inst_1'
                    elif mdx_net_model_name == 'UVR-MDX-NET Inst 2':
                        mdx_only_ensem_e = 'UVR_MDXNET_Inst_2'
                    else:
                        mdx_only_ensem_e = mdx_net_model_name
                        
                if 'Demucs:' in data['mdx_only_ensem_e']:
                    mdx_model_run_e = 'no'
                    mdx_only_ensem_e = demucs_model_set_name_e
                    
                if data['mdx_only_ensem_e'] == 'No Model':
                    mdx_model_run_e = 'no'
                    mdx_only_ensem_e = 'pass'
                      
                    
                if data['mdx_only_ensem_c'] == 'No Model' and data['mdx_only_ensem_d'] == 'No Model' and data['mdx_only_ensem_e'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        }
                    ] 
                elif data['mdx_only_ensem_c'] == 'No Model' and data['mdx_only_ensem_d'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_e,
                            'mdx_model_run': mdx_model_run_e,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_e}'
                        }  
                    ] 
                elif data['mdx_only_ensem_c'] == 'No Model' and data['mdx_only_ensem_e'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_d,
                            'mdx_model_run': mdx_model_run_d,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_d}'
                        }  
                    ] 
                elif data['mdx_only_ensem_d'] == 'No Model' and data['mdx_only_ensem_e'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_c,
                            'mdx_model_run': mdx_model_run_c,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_c}'
                        }  
                    ] 
                elif data['mdx_only_ensem_d'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_c,
                            'mdx_model_run': mdx_model_run_c,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_c}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_e,
                            'mdx_model_run': mdx_model_run_e,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_e}'
                        }
                    ]
                    
                elif data['mdx_only_ensem_c'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_d,
                            'mdx_model_run': mdx_model_run_d,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_d}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_e,
                            'mdx_model_run': mdx_model_run_e,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_e}'
                        }
                    ] 
                elif data['mdx_only_ensem_e'] == 'No Model':
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_c,
                            'mdx_model_run': mdx_model_run_c,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_c}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_d,
                            'mdx_model_run': mdx_model_run_d,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_d}'
                        }
                    ] 
                else:
                    mdx_demuc_only = [
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_a,
                            'mdx_model_run': mdx_model_run_a,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_a}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_b,
                            'mdx_model_run': mdx_model_run_b,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_b}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_c,
                            'mdx_model_run': mdx_model_run_c,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_c}'
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_d,
                            'mdx_model_run': mdx_model_run_d,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_d}' 
                        },
                        {
                            'model_name': 'pass',
                            'model_name_c':'pass',
                            'mdx_model_name': mdx_only_ensem_e,
                            'mdx_model_run': mdx_model_run_e,
                            'model_location': 'pass',
                            'loop_name': f'Ensemble Mode - Running Model - {mdx_only_ensem_e}'
                        }  
                    ] 
                    
                basic_md_ensemble_list = [data['mdx_only_ensem_a'], data['mdx_only_ensem_b'], data['mdx_only_ensem_c'], data['mdx_only_ensem_d'], data['mdx_only_ensem_e']]
                no_basic_md_models = basic_md_ensemble_list.count('No Model')
                basic_md_ensem_count = 5 - no_basic_md_models
                    
                global model_count
                    
                if data['ensChoose'] == 'Multi-AI Ensemble':           
                    loops = mdx_vr
                    ensefolder = 'Multi_AI_Ensemble_Outputs'
                    ensemode = 'Multi_AI_Ensemble'
                    model_count = multi_ensem_count
 
                if data['ensChoose'] == 'Basic VR Ensemble':
                    loops = Basic_Ensem
                    ensefolder = 'Basic_VR_Outputs'
                    ensemode = 'Multi_VR_Ensemble'
                    model_count = vr_ensem_count

                if data['ensChoose'] == 'Basic MD Ensemble':
                    loops = mdx_demuc_only
                    ensefolder = 'Basic_MDX_Net_Demucs_Ensemble'
                    ensemode = 'Basic_MDX_Net_Demucs_Ensemble'
                    model_count = basic_md_ensem_count

                global current_model_bar
                
                current_model_bar = 0

                #Prepare Audiofile(s)
                for file_num, music_file in enumerate(data['input_paths'], start=1):
                    # -Get text and update progress-
                    
                    current_model = 1
                    
                    
                    base_text = get_baseText(total_files=len(data['input_paths']),
                                                file_num=file_num)
                    progress_kwargs = {'progress_var': progress_var,
                                        'total_files': len(data['input_paths']),
                                        'file_num': file_num}   
                    
                    try:
                        
                        if float(data['noisereduc_s']) >= 11:
                            text_widget.write('Error: Noise Reduction only supports values between 0-10.\nPlease set a value between 0-10 (with or without decimals) and try again.')
                            progress_var.set(0)
                            button_widget.configure(state=tk.NORMAL)  # Enable Button
                            return
                        
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
                                if c['mdx_model_name'] == 'tasnet.th':
                                    ModelName_2 = "Tasnet_v1"
                                elif c['mdx_model_name'] == 'tasnet_extra.th':
                                    ModelName_2 = "Tasnet_extra_v1"
                                elif c['mdx_model_name'] == 'demucs.th':
                                    ModelName_2 = "Demucs_v1"
                                elif c['mdx_model_name'] == 'demucs_extra.th':
                                    ModelName_2 = "Demucs_extra_v1"
                                elif c['mdx_model_name'] == 'light_extra.th':
                                    ModelName_2 = "Light_v1"
                                elif c['mdx_model_name'] == 'light_extra.th':
                                    ModelName_2 = "Light_extra_v1"
                                elif c['mdx_model_name'] == 'tasnet-beb46fac.th':
                                    ModelName_2 = "Tasnet_v2"
                                elif c['mdx_model_name'] == 'tasnet_extra-df3777b2.th':
                                    ModelName_2 = "Tasnet_extra_v2"
                                elif c['mdx_model_name'] == 'demucs48_hq-28a1282c.th':
                                    ModelName_2 = "Demucs48_hq_v2"
                                elif c['mdx_model_name'] == 'demucs-e07c671f.th':
                                    ModelName_2 = "Demucs_v2"
                                elif c['mdx_model_name'] == 'demucs_extra-3646af93.th':
                                    ModelName_2 = "Demucs_extra_v2"
                                elif c['mdx_model_name'] == 'demucs_unittest-09ebc15f.th':
                                    ModelName_2 = "Demucs_unittest_v2"
                                else:
                                    ModelName_2 = c['mdx_model_name']
                            except:
                                pass


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
                            
                            def get_numbers_from_filename(filename):
                                return re.search(r'\d+', filename).group(0)
                            
                            foldernum = get_numbers_from_filename(enseFolderName)
                            
                            
                            if c['model_location'] == 'pass':
                                pass
                            else:
                                model_name = c['model_name']
                                text_widget.write(f'Ensemble Mode - {model_name} - Model {current_model}/{model_count}\n\n')
                                current_model += 1
                                current_model_bar += 1
                                update_progress(**progress_kwargs,
                                    step=0)   
                                presentmodel = Path(c['model_location'])
                                
                                if presentmodel.is_file():
                                    pass
                                else: 
                                    if data['ensChoose'] == 'Multi-AI Ensemble':
                                        text_widget.write(base_text + 'Model "' + c['model_name'] + '.pth" is missing.\n')
                                        text_widget.write(base_text + 'This model can be downloaded straight from the \"Settings\" options.\n')
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
                                
                                text_widget.write(base_text + 'Loading VR model... ')
                                    
                                aggresive_set = float(data['agg']/100)
                                
                                model_size = math.ceil(os.stat(c['model_location']).st_size / 1024)
                                nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x:abs(x-model_size)))
                                
                                nets = importlib.import_module('lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
                                
                                text_widget.write('Done!\n')
                                
                                ModelName=(c['model_location'])
                                ModelParamSettings=(c['model_param'])
                                #Package Models
                                
                                if ModelParamSettings == 'Auto':
                                    model_hash = hashlib.md5(open(ModelName,'rb').read()).hexdigest()
                                    model_params = []   
                                    model_params = lib_v5.filelist.provide_model_param_hash(model_hash)
                                    #print(model_params)
                                    if model_params[0] == 'Not Found Using Hash':
                                        model_params = []   
                                        model_params = lib_v5.filelist.provide_model_param_name(ModelName)
                                    if model_params[0] == 'Not Found Using Name':
                                        text_widget.write(base_text + f'Unable to set model parameters automatically with the selected model. Continue?\n')
                                        confirm = tk.messagebox.askyesno(title='Unrecognized Model Detected',
                                                message=f'\nThe application could not automatically set the model param for the selected model.\n\n' + 
                                                f'Would you like to select the Model Param file for this model?\n\n' + 
                                                f'This model will be skipped if no Model Param is selected.')
                                        
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
                                                text_widget.write(base_text + f'Model param not selected.\n')
                                                text_widget.write(base_text + f'Moving on to next model...\n\n')

                                                continue
                                            else:
                                                pass
                                        else:
                                            text_widget.write(base_text + f'Model param not selected.\n')
                                            text_widget.write(base_text + f'Moving on to next model...\n\n')

                                            continue
                                            
                                            
                                else:
                                    model_param_file_path = f'lib_v5/modelparams/{ModelParamSettings}'
                                    model_params = [model_param_file_path, ModelParamSettings]
                                    
                                ModelName_1=(c['model_name'])

                                #print('Model Parameters:', model_params[0])
                                text_widget.write(base_text + 'Loading assigned model parameters ' + '\"' + model_params[1] + '\"... ')
                                
                                mp = ModelParameters(model_params[0])
                                
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

                                # -Go through the different steps of Separation-
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
                                        
                                text_widget.write(base_text + 'Completed Separation!\n\n')  
                    
###################################
                            if data['ensChoose'] == 'Multi-AI Ensemble' or data['ensChoose'] == 'Basic MD Ensemble':
                                
                                
                                
                                if data['demucsmodel']:
                                    demucs_switch = 'on'
                                else:
                                    demucs_switch = 'off'
                                
                                if data['demucs_only']:
                                    demucs_only = 'on'
                                else:
                                    demucs_only = 'off'
                                
                                if c['mdx_model_name'] == 'tasnet.th':
                                    post_mdx_name = "Tasnet v1"
                                elif c['mdx_model_name'] == 'tasnet_extra.th':
                                    post_mdx_name = "Tasnet_extra v1"
                                elif c['mdx_model_name'] == 'demucs.th':
                                    post_mdx_name = "Demucs v1"
                                elif c['mdx_model_name'] == 'demucs_extra.th':
                                    post_mdx_name = "Demucs_extra v1"
                                elif c['mdx_model_name'] == 'light_extra.th':
                                    post_mdx_name = "Light v1"
                                elif c['mdx_model_name'] == 'light_extra.th':
                                    post_mdx_name = "Light_extra v1"
                                elif c['mdx_model_name'] == 'tasnet-beb46fac.th':
                                    post_mdx_name = "Tasnet v2"
                                elif c['mdx_model_name'] == 'tasnet_extra-df3777b2.th':
                                    post_mdx_name = "Tasnet_extra v2"
                                elif c['mdx_model_name'] == 'demucs48_hq-28a1282c.th':
                                    post_mdx_name = "Demucs48_hq v2"
                                elif c['mdx_model_name'] == 'demucs-e07c671f.th':
                                    post_mdx_name = "Demucs v2"
                                elif c['mdx_model_name'] == 'demucs_extra-3646af93.th':
                                    post_mdx_name = "Demucs_extra v2"
                                elif c['mdx_model_name'] == 'demucs_unittest-09ebc15f.th':
                                    post_mdx_name = "Demucs_unittest v2"
                                else:
                                    post_mdx_name = c['mdx_model_name']
                                
                                mdx_name = c['mdx_model_name']
                                
                                
                                if c['mdx_model_name'] == 'pass':
                                    pass
                                else:
                                    text_widget.write(f'Ensemble Mode - {post_mdx_name} - Model {current_model}/{model_count}\n\n')
                                    #text_widget.write('Ensemble Mode - Running Model - ' + post_mdx_name + '\n\n')
                                        
                                    if c['mdx_model_run'] == 'no':
                                        if 'UVR' in mdx_name:
                                            demucs_only = 'on'
                                            demucs_switch = 'on'
                                            demucs_model_set = mdx_name
                                            model_set = ''
                                            model_set_name = 'UVR'
                                            modeltype = 'v'
                                            noise_pro = 'MDX-NET_Noise_Profile_14_kHz'
                                            stemset_n = '(Vocals)'
                                        else:
                                            demucs_only = 'on'
                                            demucs_switch = 'on'
                                            demucs_model_set = mdx_name
                                            model_set = ''
                                            model_set_name = 'extra'
                                            modeltype = 'v'
                                            noise_pro = 'MDX-NET_Noise_Profile_14_kHz'
                                            stemset_n = '(Vocals)'
                                    if c['mdx_model_run'] == 'yes':
                                        demucs_only = 'off'
                                        model_set = f"{mdx_name}.onnx"
                                        model_set_name = mdx_name
                                        demucs_model_set = demucs_model_set_name
                                        mdx_model_path = f'models/MDX_Net_Models/{mdx_name}.onnx'
                                        
                                        model_hash = hashlib.md5(open(mdx_model_path,'rb').read()).hexdigest()
                                        model_params_mdx = []   
                                        model_params_mdx = lib_v5.filelist.provide_mdx_model_param_name(model_hash)
                                        
                                        modeltype = model_params_mdx[0]
                                        noise_pro = model_params_mdx[1]
                                        stemset_n = model_params_mdx[2]
                                        if autocompensate:
                                            compensate = model_params_mdx[3]
                                        else:
                                            compensate = data['compensate']
                                        source_val = model_params_mdx[4]
                                        n_fft_scale_set = model_params_mdx[5]
                                        dim_f_set = model_params_mdx[6]

                                        #print(model_params_mdx)
                                    
                            
                                    #print('demucs_only? ', demucs_only)
                                    
                                    if demucs_only == 'on':
                                        inference_type = 'demucs_only'
                                    else:
                                        inference_type = 'inference_mdx'
                                        
                                    progress_demucs_kwargs = {'total_files': len(data['input_paths']),
                                                    'file_num': file_num, 'inference_type': inference_type}
                                                            
                                    if data['noise_pro_select'] == 'Auto Select':
                                        noise_pro_set = noise_pro
                                    else:
                                        noise_pro_set = data['noise_pro_select']
                                    
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

                                    current_model += 1
                                    current_model_bar += 1
                                    
                                    update_progress(**progress_kwargs,
                                                    step=0)   

                                    pred = Predictor()
                                    
                                    if c['mdx_model_run'] == 'yes':
                                        if stemset_n == '(Bass)' or stemset_n == '(Drums)' or stemset_n == '(Other)':
                                            widget_text.write(base_text + 'Only vocal and instrumental MDX-Net models are supported in \nensemble mode.\n')
                                            widget_text.write(base_text + 'Moving on to next model...\n\n')
                                            continue
                                        if stemset_n == '(Instrumental)':
                                            if not 'UVR' in demucs_model_set:
                                                if data['demucsmodel']:
                                                    widget_text.write(base_text + 'The selected Demucs model cannot be used with this model.\n')
                                                    widget_text.write(base_text + 'Only 2 stem Demucs models are compatible with this model.\n')
                                                    widget_text.write(base_text + 'Setting Demucs model to \"UVR_Demucs_Model_1\".\n\n')
                                                    demucs_model_set = 'UVR_Demucs_Model_1'
                                        if modeltype == 'Not Set' or \
                                        noise_pro == 'Not Set' or \
                                        stemset_n == 'Not Set' or \
                                        compensate == 'Not Set' or \
                                        source_val == 'Not Set' or \
                                        n_fft_scale_set == 'Not Set' or \
                                        dim_f_set == 'Not Set':
                                            confirm = tk.messagebox.askyesno(title='Unrecognized Model Detected',
                                                    message=f'\nWould you like to set the correct model parameters for this model before continuing?\n')
                                            
                                            if confirm:
                                                pred.mdx_options()
                                            else:
                                                text_widget.write(base_text + 'An unrecognized model has been detected.\n')
                                                text_widget.write(base_text + 'Please configure the ONNX model settings accordingly and try again.\n')
                                                text_widget.write(base_text + 'Moving on to next model...\n\n')
                                                continue
                                    
                                    pred.prediction_setup()
                                    
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
                        if data['settest']:
                            voc_inst = [
                                {
                                    'algorithm':'min_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                    'output':'{}_{}_(Instrumental)'.format(foldernum, trackname),
                                    'type': 'Instrumentals'
                                },
                                {
                                    'algorithm':'max_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                    'output': '{}_{}_(Vocals)'.format(foldernum, trackname),
                                    'type': 'Vocals'
                                }
                            ]

                            inst = [
                                {
                                    'algorithm':'min_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                    'output':'{}_{}_(Instrumental)'.format(foldernum, trackname),
                                    'type': 'Instrumentals'
                                }
                            ]

                            vocal = [
                                {
                                    'algorithm':'max_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                    'output': '{}_{}_(Vocals)'.format(foldernum, trackname),
                                    'type': 'Vocals'
                                }
                            ]
                        else:
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
                        if data['settest']:
                            voc_inst = [
                                {
                                    'algorithm':'min_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                    'output':'{}_{}_Ensembled_{}_(Instrumental)'.format(foldernum, trackname, ensemode),
                                    'type': 'Instrumentals'
                                },
                                {
                                    'algorithm':'max_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                    'output': '{}_{}_Ensembled_{}_(Vocals)'.format(foldernum, trackname, ensemode),
                                    'type': 'Vocals'
                                }
                            ]

                            inst = [
                                {
                                    'algorithm':'min_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Instrumental).wav"),
                                    'output':'{}_{}_Ensembled_{}_(Instrumental)'.format(foldernum, trackname, ensemode),
                                    'type': 'Instrumentals'
                                }
                            ]

                            vocal = [
                                {
                                    'algorithm':'max_mag',
                                    'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                    'files':get_files(folder=enseExport, prefix=trackname, suffix="_(Vocals).wav"),
                                    'output': '{}_{}_Ensembled_{}_(Vocals)'.format(foldernum, trackname, ensemode),
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
                                    normalization_set(spec_utils.cmb_spectrogram_to_wave(spec_utils.ensembling(e['algorithm'], 
                                                                                    specs), mp)), mp.param['sr'], subtype=wav_type_set)
                            
                            if data['saveFormat'] == 'Mp3':
                                try:
                                    musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                                    musfile.export((os.path.join('{}'.format(data['export_path']),'{}.mp3'.format(e['output']))), format="mp3", bitrate=mp3_bit_set)
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
                                        text_widget.write('\n' + base_text + 'Failed to save output(s) as Flac(s).\n')
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
                                        musfile.export("{0}.mp3".format(name), format="mp3", bitrate=mp3_bit_set)    
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
                
                timestampnum = round(datetime.utcnow().timestamp())
                randomnum = randrange(100000, 1000000)
                
                if data['settest']:
                    try:
                        insts = [
                            {
                                'algorithm':'min_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output':'{}_{}_Manual_Ensemble_(Min Spec)'.format(timestampnum, trackname1),
                                'type': 'Instrumentals'
                            }
                        ]

                        vocals = [
                            {
                                'algorithm':'max_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_Manual_Ensemble_(Max Spec)'.format(timestampnum, trackname1),
                                'type': 'Vocals'
                            }
                        ]
                        
                        invert_spec = [
                            {
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_diff_si'.format(timestampnum, trackname1),
                                'type': 'Spectral Inversion'
                            }
                        ]
                        
                        invert_nor = [
                            {
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_diff_ni'.format(timestampnum, trackname1),
                                'type': 'Normal Inversion'
                            }
                        ]
                    except:
                        insts = [
                            {
                                'algorithm':'min_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output':'{}_{}_Manual_Ensemble_(Min Spec)'.format(randomnum, trackname1),
                                'type': 'Instrumentals'
                            }
                        ]

                        vocals = [
                            {
                                'algorithm':'max_mag',
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_Manual_Ensemble_(Max Spec)'.format(randomnum, trackname1),
                                'type': 'Vocals'
                            }
                        ]
                        
                        invert_spec = [
                            {
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_diff_si'.format(randomnum, trackname1),
                                'type': 'Spectral Inversion'
                            }
                        ]
                        
                        invert_nor = [
                            {
                                'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                                'output': '{}_{}_diff_ni'.format(randomnum, trackname1),
                                'type': 'Normal Inversion'
                            }
                        ]
                else:
                    insts = [
                        {
                            'algorithm':'min_mag',
                            'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                            'output':'{}_Manual_Ensemble_(Min Spec)'.format(trackname1),
                            'type': 'Instrumentals'
                        }
                    ]

                    vocals = [
                        {
                            'algorithm':'max_mag',
                            'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                            'output': '{}_Manual_Ensemble_(Max Spec)'.format(trackname1),
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
                                normalization_set(spec_utils.cmb_spectrogram_to_wave(spec_utils.ensembling(e['algorithm'], 
                                                                                specs), mp)), mp.param['sr'], subtype=wav_type_set)
                        
                    if data['saveFormat'] == 'Mp3':
                        try:
                            musfile = pydub.AudioSegment.from_wav(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])))
                            musfile.export((os.path.join('{}'.format(data['export_path']),'{}.mp3'.format(e['output']))), format="mp3", bitrate=mp3_bit_set)
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
                                            f'Error Received while attempting to run Manual Ensemble:\n' + 
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
                                            f'Error Received while attempting to run Manual Ensemble:\n' + 
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
                                        spec_utils.cmb_spectrogram_to_wave(-v_spec, mp), mp.param['sr'], subtype=wav_type_set)
                            if data['algo'] == 'Invert (Normal)':
                                v_spec = specs[0] - specs[1]
                                sf.write(os.path.join('{}'.format(data['export_path']),'{}.wav'.format(e['output'])), 
                                        spec_utils.cmb_spectrogram_to_wave(v_spec, mp), mp.param['sr'], subtype=wav_type_set)
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
            text_widget.write(f"\nGo to the Settings Menu and click \"Open Error Log\" for raw error details.\n")
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
                            f'Process Method: Ensemble Mode\n\n' +
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
        text_widget.write("\nGo to the Settings Menu and click \"Open Error Log\" for raw error details.\n")
        text_widget.write("\n" + f'Please address the error and try again.' + "\n")
        text_widget.write(f'If this error persists, please contact the developers with the error details.\n\n')
        text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
        torch.cuda.empty_cache()
        button_widget.configure(state=tk.NORMAL)  # Enable Button
        return
 
    update_progress(**progress_kwargs,
    step=1) 
    
    #print('Done!')
    
    progress_var.set(0)
    if not data['ensChoose'] == 'Manual Ensemble':
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

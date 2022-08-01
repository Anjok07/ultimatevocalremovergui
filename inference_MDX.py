from datetime import datetime
from demucs.apply import BagOfModels, apply_model
from demucs.hdemucs import HDemucs
from demucs.model_v2 import Demucs
from demucs.pretrained import get_model as _gm
from demucs.tasnet_v2 import ConvTasNet
from demucs.utils import apply_model_v1
from demucs.utils import apply_model_v2
from lib_v5 import spec_utils
from lib_v5.model_param_init import ModelParameters
from models import get_models, spec_effects
from pathlib import Path
from random import randrange
from tqdm import tqdm
from unittest import skip
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.font
import tkinter as tk
from tkinter import *
from tkinter.tix import *
import json
import gzip
import hashlib
import librosa
import numpy as np
import onnxruntime as ort
import os
import os.path
import pathlib
import psutil
import pydub
import shutil
import soundfile as sf
import subprocess
import sys
import time
import time  # Timer
import tkinter as tk
import torch
import traceback  # Error Message Recent Calls
import warnings
import lib_v5.filelist

#from typing import Literal
    
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
        
        top= Toplevel()

        top.geometry("740x550")
        window_height = 740
        window_width = 550
        
        top.title("Specify Parameters")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        top.attributes("-topmost", True)
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        tabControl = ttk.Notebook(top)
        
        tabControl.pack(expand = 1, fill ="both")
        
        tabControl.grid_rowconfigure(0, weight=1)
        tabControl.grid_columnconfigure(0, weight=1)
        
        frame0=Frame(tabControl,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        frame0.tkraise(frame0)
        
        space_small = '  '*20
        space_small_1 = '  '*10
        
        l0=tk.Label(frame0, text=f'{space_small}Stem Type{space_small}', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=5)
        
        l0=ttk.OptionMenu(frame0, self.mdxnetModeltype_var, None, 'Vocals', 'Instrumental', 'Other', 'Bass', 'Drums')
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
        
        l0=ttk.Button(frame0,text="Continue", command=lambda: self.okVar.set(1))
        l0.grid(row=13,column=0,padx=0,pady=30)
        
        def stop():
            widget_text.write(f'Please configure the ONNX model settings accordingly and try again.\n\n')
            widget_text.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            torch.cuda.empty_cache()
            gui_progress_bar.set(0)
            widget_button.configure(state=tk.NORMAL)  # Enable Button
            top.destroy()
            return
        
        l0=ttk.Button(frame0,text="Stop Process", command=stop)
        l0.grid(row=13,column=1,padx=0,pady=30)
        
        def change_event():
            self.okVar.set(1)
            #top.destroy()
            pass
        
        top.protocol("WM_DELETE_WINDOW", change_event)
        
        frame0.wait_variable(self.okVar)
        
        global n_fft_scale_set
        global dim_f_set
        global modeltype
        global stemset_n
        global stem_text_a
        global stem_text_b
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
            source_val = 0
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
            
        if stemset_n == '(Vocals)':
            stem_text_a = 'Vocals'
            stem_text_b = 'Instrumental'
        elif stemset_n == '(Instrumental)':
            stem_text_a = 'Instrumental'
            stem_text_b = 'Vocals'
        elif stemset_n == '(Other)':
            stem_text_a = 'Other'
            stem_text_b = 'the no \"Other\" track'
        elif stemset_n == '(Drums)':
            stem_text_a = 'Drums'
            stem_text_b = 'no \"Drums\" track'
        elif stemset_n == '(Bass)':
            stem_text_a = 'Bass'
            stem_text_b = 'No \"Bass\" track'
        else: 
            stem_text_a = 'Vocals'
            stem_text_b = 'Instrumental'
            
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
            
        if 'UVR' in demucs_model_set:
            if stemset_n == '(Bass)' or stemset_n == '(Drums)' or stemset_n == '(Other)':
                widget_text.write(base_text + 'The selected Demucs model can only be used with vocal or instrumental stems.\n')
                widget_text.write(base_text + 'Please select a 4 stem Demucs model next time.\n')
                widget_text.write(base_text + 'Setting Demucs Model to \"mdx_extra\"\n')
                demucs_model_set = 'mdx_extra'

            
        if stemset_n == '(Instrumental)':
            if not 'UVR' in demucs_model_set:
                widget_text.write(base_text + 'The selected Demucs model cannot be used with this model.\n')
                widget_text.write(base_text + 'Only 2 stem Demucs models are compatible with this model.\n')
                widget_text.write(base_text + 'Setting Demucs model to \"UVR_Demucs_Model_1\".\n\n')
                demucs_model_set = 'UVR_Demucs_Model_1'
        
        top.destroy()

    def prediction_setup(self):
        
        global device

        if data['gpu'] >= 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
        if data['gpu'] == -1:
            device = torch.device('cpu')

        if data['demucsmodel']:
            if demucs_model_version == 'v1':
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
                    
            if demucs_model_version == 'v2':
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
                self.demucs.eval()
                
            if demucs_model_version == 'v3':
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

        self.onnx_models = {}
        c = 0
        
        self.models = get_models('tdf_extra', load=False, device=cpu, stems=modeltype, n_fft_scale=int(n_fft_scale_set), dim_f=int(dim_f_set))
        if not data['demucs_only']:
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
            
        print('Selected Model: ', mdx_model_path)
        self.onnx_models[c] = ort.InferenceSession(os.path.join(mdx_model_path), providers=run_type)
        
        if not data['demucs_only']:
            widget_text.write('Done!\n')
        
    def prediction(self, m):  
        mix, samplerate = librosa.load(m, mono=False, sr=44100)
        #print('print mix: ', mix)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix,mix])  
        samplerate = samplerate
        mix = mix.T
        sources = self.demix(mix.T)
        widget_text.write(base_text + 'Inferences complete!\n')
        c = -1
    
        #Main Save Path
        save_path = os.path.dirname(_basename)
        
        inst_only = data['inst_only']
        voc_only = data['voc_only']
        
        #print('stemset_n: ', stemset_n)
        
        if stemset_n == '(Instrumental)':
            if data['inst_only'] == True:
                voc_only = True
                inst_only = False
            if data['voc_only'] == True:
                inst_only = True
                voc_only = False
        
        #Vocal Path
        if stemset_n == '(Vocals)':
            vocal_name = '(Vocals)'
        elif stemset_n == '(Instrumental)':
            vocal_name = '(Instrumental)'
        elif stemset_n == '(Other)':
            vocal_name = '(Other)'
        elif stemset_n == '(Drums)':
            vocal_name = '(Drums)'
        elif stemset_n == '(Bass)':
            vocal_name = '(Bass)'
            
        if data['modelFolder']:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}',)
            vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}',)
            vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}',)
        else:
            vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
            vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
            vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}',)
        
        #Instrumental Path

        if stemset_n == '(Vocals)':
            Instrumental_name = '(Instrumental)'
        elif stemset_n == '(Instrumental)':
            Instrumental_name = '(Vocals)'
        elif stemset_n == '(Other)':
            Instrumental_name = '(No_Other)'
        elif stemset_n == '(Drums)':
            Instrumental_name = '(No_Drums)'
        elif stemset_n == '(Bass)':
            Instrumental_name = '(No_Bass)'

        if data['modelFolder']:
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}',)
            Instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}',)
            Instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}',)            
        else: 
            Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)
            Instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)
            Instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}',)   
                     
        #Non-Reduced Vocal Path
        if stemset_n == '(Vocals)':
            vocal_name = '(Vocals)'
        elif stemset_n == '(Other)':
            vocal_name = '(Other)'
        elif stemset_n == '(Drums)':
            vocal_name = '(Drums)'
        elif stemset_n == '(Bass)':
            vocal_name = '(Bass)'
        elif stemset_n == '(Instrumental)':
            vocal_name = '(Instrumental)'

        if data['modelFolder']:
            non_reduced_vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}_No_Reduction',)
            non_reduced_vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}_No_Reduction',)
            non_reduced_vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_{model_set_name}_No_Reduction',)
        else:
            non_reduced_vocal_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_No_Reduction',)
            non_reduced_vocal_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_No_Reduction',)
            non_reduced_vocal_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{vocal_name}_No_Reduction',)
            
        if data['modelFolder']:
            non_reduced_Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}_No_Reduction',)
            non_reduced_Instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}_No_Reduction',)
            non_reduced_Instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_{model_set_name}_No_Reduction',)            
        else: 
            non_reduced_Instrumental_path = '{save_path}/{file_name}.wav'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_No_Reduction',)
            non_reduced_Instrumental_path_mp3 = '{save_path}/{file_name}.mp3'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_No_Reduction',)
            non_reduced_Instrumental_path_flac = '{save_path}/{file_name}.flac'.format(
                save_path=save_path,
                file_name = f'{os.path.basename(_basename)}_{Instrumental_name}_No_Reduction',)   
            
            
        if os.path.isfile(non_reduced_vocal_path):
            file_exists_n = 'there'
        else:
            file_exists_n = 'not_there'

        if os.path.isfile(vocal_path):
            file_exists_v = 'there'
        else:
            file_exists_v = 'not_there'
            
        if os.path.isfile(Instrumental_path):
            file_exists_i = 'there'
        else:
            file_exists_i = 'not_there'

        #print('Is there already a voc file there? ', file_exists_v)

        if not data['noisereduc_s'] == 'None':
            c += 1

            if not data['demucsmodel']:
                if inst_only:
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
                if inst_only:
                    widget_text.write(base_text + f'Preparing {stem_text_b}...')
                else:
                    widget_text.write(base_text + f'Saving {stem_text_a}... ')

                if data['demucs_only']:
                    if 'UVR' in demucs_model_set:
                        if stemset_n == '(Instrumental)':
                            sf.write(non_reduced_vocal_path, sources[0].T, samplerate, subtype=wav_type_set)
                        else:
                            sf.write(non_reduced_vocal_path, sources[1].T, samplerate, subtype=wav_type_set)
                else:
                    sf.write(non_reduced_vocal_path, sources[source_val].T, samplerate, subtype=wav_type_set)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
                widget_text.write(base_text + 'Performing Noise Reduction... ')
                reduction_sen = float(data['noisereduc_s'])/10
                #print(noise_pro_set)
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

            if not data['demucsmodel']:
                if inst_only:
                    widget_text.write(base_text + f'Preparing {stem_text_b}...')
                else:
                    widget_text.write(base_text + f'Saving {stem_text_a}... ')
                sf.write(vocal_path, sources[c].T, samplerate, subtype=wav_type_set)
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
            else:
                if inst_only:
                    widget_text.write(base_text + f'Preparing {stem_text_b}...')
                else:
                    widget_text.write(base_text + f'Saving {stem_text_a}... ')
                    
                if data['demucs_only']:
                    if 'UVR' in demucs_model_set:
                        if stemset_n == '(Instrumental)':
                            sf.write(vocal_path, sources[0].T, samplerate, subtype=wav_type_set)
                        else:
                            sf.write(vocal_path, sources[1].T, samplerate, subtype=wav_type_set)
                    else:
                        sf.write(vocal_path, sources[source_val].T, samplerate, subtype=wav_type_set)
                else:
                    sf.write(vocal_path, sources[source_val].T, samplerate, subtype=wav_type_set)
                    
                update_progress(**progress_kwargs,
                step=(0.9))
                widget_text.write('Done!\n')
        
        if voc_only and not inst_only:
            pass
        else:
            if not data['noisereduc_s'] == 'None':
                if data['nophaseinst']:
                    finalfiles = [
                        {
                            'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                            'files':[str(music_file), non_reduced_vocal_path],
                        }
                    ]   
                else:
                    finalfiles = [
                        {
                            'model_params':'lib_v5/modelparams/1band_sr44100_hl512.json',
                            'files':[str(music_file), vocal_path],
                        }
                    ]  
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
                step=(1))
                
                if not data['noisereduc_s'] == 'None':
                    if data['nophaseinst']:
                        sf.write(non_reduced_Instrumental_path, normalization_set(spec_utils.cmb_spectrogram_to_wave(-v_spec, mp)), mp.param['sr'], subtype=wav_type_set)
                    
                        reduction_sen = float(data['noisereduc_s'])/10
                        #print(noise_pro_set)
                        
                        subprocess.call("lib_v5\\sox\\sox.exe" + ' "' + 
                                    f"{str(non_reduced_Instrumental_path)}"  + '" "' + f"{str(Instrumental_path)}" + '" ' + 
                                    "noisered lib_v5\\sox\\" + noise_pro_set + ".prof " + f"{reduction_sen}", 
                                    shell=True, stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        sf.write(Instrumental_path, normalization_set(spec_utils.cmb_spectrogram_to_wave(-v_spec, mp)), mp.param['sr'], subtype=wav_type_set)
                else:
                    sf.write(Instrumental_path, normalization_set(spec_utils.cmb_spectrogram_to_wave(-v_spec, mp)), mp.param['sr'], subtype=wav_type_set)
                
                if inst_only:
                    if file_exists_v == 'there':
                        pass
                    else:
                        try:
                            os.remove(vocal_path)
                        except:
                            pass
     
                widget_text.write('Done!\n')
          
        if data['saveFormat'] == 'Mp3':
            try:
                
                if inst_only == True:
                    if data['non_red'] == True:
                        if not data['nophaseinst']:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(non_reduced_Instrumental_path)
                            musfile.export(non_reduced_Instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set) 
                            try:
                                os.remove(non_reduced_Instrumental_path)
                            except:
                                pass
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
                    if data['non_red'] == True:
                        if not data['nophaseinst']:
                            pass
                        else:
                            if voc_only == True:
                                pass
                            else:
                                musfile = pydub.AudioSegment.from_wav(non_reduced_Instrumental_path)
                                musfile.export(non_reduced_Instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set) 
                                if file_exists_n == 'there':
                                    pass
                                else:
                                    try:
                                        os.remove(non_reduced_Instrumental_path)
                                    except:
                                        pass
                if voc_only == True:
                    if data['non_red'] == True:
                        musfile = pydub.AudioSegment.from_wav(non_reduced_vocal_path)
                        musfile.export(non_reduced_vocal_path_mp3, format="mp3", bitrate=mp3_bit_set) 
                        try:
                            os.remove(non_reduced_vocal_path)
                        except:
                            pass
                    pass
                else:
                    musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                    musfile.export(Instrumental_path_mp3, format="mp3", bitrate=mp3_bit_set)  
                    if file_exists_i == 'there':
                        pass
                    else:
                        try:
                            os.remove(Instrumental_path)
                        except:
                            pass  
                    if data['non_red'] == True:
                        if inst_only == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(non_reduced_vocal_path)
                            musfile.export(non_reduced_vocal_path_mp3, format="mp3", bitrate=mp3_bit_set) 
                        if file_exists_n == 'there':
                            pass
                        else:
                            try:
                                os.remove(non_reduced_vocal_path)
                            except:
                                pass
                        
            except Exception as e:
                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                if "ffmpeg" in errmessage:
                    widget_text.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                    widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                    widget_text.write(base_text + 'Moving on...\n')
                else:
                    widget_text.write(base_text + 'Failed to save output(s) as Mp3(s).\n')
                    widget_text.write(base_text + 'Please check error log.\n')
                    widget_text.write(base_text + 'Moving on...\n')
                try:
                    with open('errorlog.txt', 'w') as f:
                        f.write(f'Last Error Received:\n\n' +
                                f'Error Received while attempting to save file as mp3 "{os.path.basename(music_file)}":\n\n' + 
                                f'Process Method: MDX-Net\n\n' +
                                f'FFmpeg might be missing or corrupted.\n\n' +
                                f'If this error persists, please contact the developers.\n\n' + 
                                f'Raw error details:\n\n' +
                                errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                except:
                    pass
            
        if data['saveFormat'] == 'Flac':
            try:
                if inst_only == True:
                    if data['non_red'] == True:
                        if not data['nophaseinst']:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(non_reduced_Instrumental_path)
                            musfile.export(non_reduced_Instrumental_path_flac, format="flac") 
                            try:
                                os.remove(non_reduced_Instrumental_path)
                            except:
                                pass
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
                    if data['non_red'] == True:
                        if not data['nophaseinst']:
                            pass
                        else:
                            if voc_only == True:
                                pass
                            else:
                                musfile = pydub.AudioSegment.from_wav(non_reduced_Instrumental_path)
                                musfile.export(non_reduced_Instrumental_path_flac, format="flac") 
                                if file_exists_n == 'there':
                                    pass
                                else:
                                    try:
                                        os.remove(non_reduced_Instrumental_path)
                                    except:
                                        pass
                if voc_only == True:
                    if data['non_red'] == True:
                        musfile = pydub.AudioSegment.from_wav(non_reduced_vocal_path)
                        musfile.export(non_reduced_vocal_path_flac, format="flac") 
                        try:
                            os.remove(non_reduced_vocal_path)
                        except:
                            pass
                    pass
                else:
                    musfile = pydub.AudioSegment.from_wav(Instrumental_path)
                    musfile.export(Instrumental_path_flac, format="flac")  
                    if file_exists_i == 'there':
                        pass
                    else:
                        try:
                            os.remove(Instrumental_path)
                        except:
                            pass  
                    if data['non_red'] == True:
                        if inst_only == True:
                            pass
                        else:
                            musfile = pydub.AudioSegment.from_wav(non_reduced_vocal_path)
                            musfile.export(non_reduced_vocal_path_flac, format="flac") 
                        if file_exists_n == 'there':
                            pass
                        else:
                            try:
                                os.remove(non_reduced_vocal_path)
                            except:
                                pass
                        
            except Exception as e:
                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                errmessage = f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\n'
                if "ffmpeg" in errmessage:
                    widget_text.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                    widget_text.write(base_text + 'FFmpeg might be missing or corrupted, please check error log.\n')
                    widget_text.write(base_text + 'Moving on...\n')
                else:
                    widget_text.write(base_text + 'Failed to save output(s) as Flac(s).\n')
                    widget_text.write(base_text + 'Please check error log.\n')
                    widget_text.write(base_text + 'Moving on...\n')
                try:
                    with open('errorlog.txt', 'w') as f:
                        f.write(f'Last Error Received:\n\n' +
                                f'Error Received while attempting to save file as flac "{os.path.basename(music_file)}":\n\n' + 
                                f'Process Method: MDX-Net\n\n' +
                                f'FFmpeg might be missing or corrupted.\n\n' +
                                f'If this error persists, please contact the developers.\n\n' + 
                                f'Raw error details:\n\n' +
                                errmessage + f'\nError Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
                except:
                    pass

        if data['noisereduc_s'] == 'None':
            pass
        elif data['non_red'] == True:
            if inst_only:
                if file_exists_n == 'there':
                    pass
                else:
                    try:
                        os.remove(non_reduced_vocal_path)
                    except:
                        pass
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
                os.remove(non_reduced_Instrumental_path)
            except:
                pass
        
        widget_text.write(base_text + 'Completed Separation!\n')

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
                if int(gpu_mem) <= int(6):
                    chunk_set = int(5)
                    widget_text.write(base_text + 'Chunk size auto-set to 5... \n')
                if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(10)
                    widget_text.write(base_text + 'Chunk size auto-set to 10... \n')
                if int(gpu_mem) >= int(16):
                    chunk_set = int(40)
                    widget_text.write(base_text + 'Chunk size auto-set to 40... \n')
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
        elif data['demucs_only']:
            if split_mode == True:
                sources = self.demix_demucs_split(mix)
            if split_mode == False:
                sources = self.demix_demucs(segmented_mix, margin_size=margin)
        else: # both, apply spec effects
            base_out = self.demix_base(segmented_mix, margin_size=margin)
            #print(split_mode)
            
            
            if demucs_model_version == 'v1': 
                demucs_out = self.demix_demucs_v1(segmented_mix, margin_size=margin)
            if demucs_model_version == 'v2': 
                demucs_out = self.demix_demucs_v2(segmented_mix, margin_size=margin)
            if demucs_model_version == 'v3':
                if split_mode == True:
                    demucs_out = self.demix_demucs_split(mix)
                if split_mode == False:
                    demucs_out = self.demix_demucs(segmented_mix, margin_size=margin)
            nan_count = np.count_nonzero(np.isnan(demucs_out)) + np.count_nonzero(np.isnan(base_out))
            if nan_count > 0:
                print('Warning: there are {} nan values in the array(s).'.format(nan_count))
                demucs_out, base_out = np.nan_to_num(demucs_out), np.nan_to_num(base_out)
            sources = {}
            #print(data['mixing'])
            
            if 'UVR' in demucs_model_set:
                if stemset_n == '(Instrumental)':
                    sources[source_val] = (spec_effects(wave=[demucs_out[0],base_out[0]],
                                                algorithm=data['mixing'],
                                                value=b[source_val])*float(compensate)) # compensation
                else:
                    sources[source_val] = (spec_effects(wave=[demucs_out[1],base_out[0]],
                                                algorithm=data['mixing'],
                                                value=b[source_val])*float(compensate)) # compensation
            else:
                sources[source_val] = (spec_effects(wave=[demucs_out[source_val],base_out[0]],
                                            algorithm=data['mixing'],
                                            value=b[source_val])*float(compensate)) # compensation
                
        if not data['demucsmodel']:  
            return sources*float(compensate)
        else:
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
        #print('shift_set ', shift_set)
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Split Mode is off. (Chunks enabled for Demucs Model)\n")
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
            with torch.no_grad():
                #print(split_mode)
                sources = apply_model(self.demucs, cmix[None], split=split_mode, device=device, overlap=overlap_set, shifts=shift_set, progress=False)[0]
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
        #print('the demucs model is done running')

        return sources
    
    def demix_demucs_split(self, mix):

        #print('shift_set ', shift_set)
        widget_text.write(base_text + "Split Mode is on. (Chunks disabled for Demucs Model)\n")
        widget_text.write(base_text + "Running Demucs Inference...\n")
        widget_text.write(base_text + "Processing "f"{len(mix)} slices... ")
        print(' Running Demucs Inference...')
          
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        
        with torch.no_grad():
            sources = apply_model(self.demucs, mix[None], split=split_mode, device=device, overlap=overlap_set, shifts=shift_set, progress=False)[0]
            
        widget_text.write('Done!\n')
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
    
        #print('the demucs model is done running')
        
        return sources
    
    def demix_demucs_v1(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Running Demucs v1 Inference...\n")
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
            with torch.no_grad():
                sources = apply_model_v1(self.demucs, cmix.to(device), split=split_mode, shifts=shift_set)
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
    
    def demix_demucs_v2(self, mix, margin_size):
        processed = {}
        demucsitera = len(mix)
        demucsitera_calc = demucsitera * 2
        gui_progress_bar_demucs = 0
        widget_text.write(base_text + "Running Demucs v2 Inference...\n")
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
            with torch.no_grad():
                sources = apply_model_v2(self.demucs, cmix.to(device), split=split_mode, overlap=overlap_set, shifts=shift_set)
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
    

        
data = {
    'autocompensate': True,
    'aud_mdx': True,
    'bit': '',
    'chunks': 10,
    'compensate': 1.03597672895,
    'demucs_only': False,
    'demucsmodel': False,
    'DemucsModel_MDX': 'UVR_Demucs_Model_1',
    'dim_f': 2048,
    'export_path': None,
    'flactype': 'PCM_16',
    'gpu': -1,
    'input_paths': None,
    'inst_only': False,
    'margin': 44100,
    'mdxnetModel': 'UVR-MDX-NET Main',
    'mdxnetModeltype': 'Vocals (Custom)',
    'mixing': 'Default',
    'modelFolder': False,
    'mp3bit': '320k',
    'n_fft_scale': 6144,
    'noise_pro_select': 'Auto Select',
    'noisereduc_s': 3,
    'non_red': False,
    'nophaseinst': True,
    'normalize': False,
    'overlap': 0.5,
    'saveFormat': 'Wav',
    'shifts': 0,
    'split_mode': False,
    'voc_only': False,
    'wavtype': 'PCM_16',
}

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

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

def hide_opt():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def main(window: tk.Wm, 
         text_widget: tk.Text, 
         button_widget: tk.Button, 
         progress_var: tk.Variable,
         **kwargs: dict):

    global widget_text
    global gui_progress_bar
    global music_file
    global default_chunks
    global default_noisereduc_s
    global _basename
    global _mixture
    global modeltype
    global n_fft_scale_set
    global dim_f_set
    global progress_kwargs
    global base_text
    global model_set_name
    global stemset_n
    global stem_text_a
    global stem_text_b
    global noise_pro_set
    global demucs_model_set
    global autocompensate
    global compensate
    global channel_set
    global margin_set
    global overlap_set
    global shift_set
    global source_val
    global split_mode
    global demucs_model_set
    global wav_type_set
    global flac_type_set
    global mp3_bit_set
    global normalization_set
    global demucs_model_version
    global mdx_model_path
    global widget_button
    global stime
    global model_hash
    global demucs_switch
    global inst_only
    global voc_only
    
    
    # Update default settings
    default_chunks = data['chunks']
    default_noisereduc_s = data['noisereduc_s']
    
    widget_text = text_widget
    gui_progress_bar = progress_var
    widget_button = button_widget
    
    #Error Handling
    
    onnxmissing = "[ONNXRuntimeError] : 3 : NO_SUCHFILE"
    onnxmemerror = "onnxruntime::CudaCall CUDA failure 2: out of memory"
    onnxmemerror2 = "onnxruntime::BFCArena::AllocateRawInternal"
    systemmemerr = "DefaultCPUAllocator: not enough memory"
    runtimeerr = "CUDNN error executing cudnnSetTensorNdDescriptor"
    cuda_err = "CUDA out of memory"
    mod_err = "ModuleNotFoundError"
    file_err = "FileNotFoundError"
    ffmp_err = """audioread\__init__.py", line 116, in audio_open"""
    sf_write_err = "sf.write"
    model_adv_set_err = "Got invalid dimensions for input"
    demucs_model_missing_err = "is neither a single pre-trained model or a bag of models."
    
    try:
        with open('errorlog.txt', 'w') as f:
            f.write(f'No errors to report at this time.' + f'\n\nLast Process Method Used: MDX-Net' +
                    f'\nLast Conversion Time Stamp: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n') 
    except:
        pass
    
    data.update(kwargs)
    
    if data['DemucsModel_MDX'] == "Tasnet v1":
        demucs_model_set_name = 'tasnet.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Tasnet_extra v1":
        demucs_model_set_name = 'tasnet_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Demucs v1":
        demucs_model_set_name = 'demucs.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Demucs v1.gz":
        demucs_model_set_name = 'demucs.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Demucs_extra v1":
        demucs_model_set_name = 'demucs_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Demucs_extra v1.gz":
        demucs_model_set_name = 'demucs_extra.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Light v1":
        demucs_model_set_name = 'light.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Light v1.gz":
        demucs_model_set_name = 'light.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Light_extra v1":
        demucs_model_set_name = 'light_extra.th'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Light_extra v1.gz":
        demucs_model_set_name = 'light_extra.th.gz'
        demucs_model_version = 'v1'
    elif data['DemucsModel_MDX'] == "Tasnet v2":
        demucs_model_set_name = 'tasnet-beb46fac.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel_MDX'] == "Tasnet_extra v2":
        demucs_model_set_name = 'tasnet_extra-df3777b2.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel_MDX'] == "Demucs48_hq v2":
        demucs_model_set_name = 'demucs48_hq-28a1282c.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel_MDX'] == "Demucs v2":
        demucs_model_set_name = 'demucs-e07c671f.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel_MDX'] == "Demucs_extra v2":
        demucs_model_set_name = 'demucs_extra-3646af93.th'
        demucs_model_version = 'v2'
    elif data['DemucsModel_MDX'] == "Demucs_unittest v2":
        demucs_model_set_name = 'demucs_unittest-09ebc15f.th'
        demucs_model_version = 'v2'
    elif '.ckpt' in data['DemucsModel_MDX'] and 'v2' in data['DemucsModel_MDX']:
        demucs_model_set_name = data['DemucsModel_MDX']
        demucs_model_version = 'v2'
    elif '.ckpt' in data['DemucsModel_MDX'] and 'v1' in data['DemucsModel_MDX']:
        demucs_model_set_name = data['DemucsModel_MDX']
        demucs_model_version = 'v1'
    elif '.gz' in data['DemucsModel_MDX']:
        demucs_model_set_name = data['DemucsModel_MDX']
        demucs_model_version = 'v1'
    else:
        demucs_model_set_name = data['DemucsModel_MDX']
        demucs_model_version = 'v3'
    
    autocompensate = data['autocompensate']
        
    model_set_name = data['mdxnetModel']
    
    if model_set_name == 'UVR-MDX-NET 1':
        mdx_model_name = 'UVR_MDXNET_1_9703'
    elif model_set_name == 'UVR-MDX-NET 2':
        mdx_model_name = 'UVR_MDXNET_2_9682'
    elif model_set_name == 'UVR-MDX-NET 3':
        mdx_model_name = 'UVR_MDXNET_3_9662'
    elif model_set_name == 'UVR-MDX-NET Karaoke':
        mdx_model_name = 'UVR_MDXNET_KARA'
    elif model_set_name == 'UVR-MDX-NET Main':
        mdx_model_name = 'UVR_MDXNET_Main'
    else:
        mdx_model_name = data['mdxnetModel']
    
    
    mdx_model_path = f'models/MDX_Net_Models/{mdx_model_name}.onnx'
    
    model_hash = hashlib.md5(open(mdx_model_path,'rb').read()).hexdigest()
    model_params = []   
    model_params = lib_v5.filelist.provide_mdx_model_param_name(model_hash)
    
    modeltype = model_params[0]
    noise_pro = model_params[1]
    stemset_n = model_params[2]
    compensate_set = model_params[3]
    source_val = model_params[4]
    n_fft_scale_set = model_params[5]
    dim_f_set = model_params[6]
    
    if not data['aud_mdx']:
        if data['mdxnetModeltype'] == 'Vocals (Custom)':
            modeltype = 'v'
            source_val = 3
            stemset_n = '(Vocals)'
            n_fft_scale_set = data['n_fft_scale']
            dim_f_set = data['dim_f']
        if data['mdxnetModeltype'] == 'Instrumental (Custom)':
            modeltype = 'v'
            source_val = 0
            stemset_n = '(Instrumental)'
            n_fft_scale_set = data['n_fft_scale']
            dim_f_set = data['dim_f']
        if data['mdxnetModeltype'] == 'Other (Custom)':
            modeltype = 'v'
            source_val = 2
            stemset_n = '(Other)'
            n_fft_scale_set = data['n_fft_scale']
            dim_f_set = data['dim_f']
        if data['mdxnetModeltype'] == 'Drums (Custom)':
            modeltype = 'v'
            source_val = 1
            stemset_n = '(Drums)'
            n_fft_scale_set = data['n_fft_scale']
            dim_f_set = data['dim_f']
        if data['mdxnetModeltype'] == 'Bass (Custom)':
            modeltype = 'v'
            source_val = 0
            stemset_n = '(Bass)'
            n_fft_scale_set = data['n_fft_scale']
            dim_f_set = data['dim_f']
            
    if stemset_n == '(Vocals)':
        stem_text_a = 'Vocals'
        stem_text_b = 'Instrumental'
    elif stemset_n == '(Instrumental)':
        stem_text_a = 'Instrumental'
        stem_text_b = 'Vocals'
    elif stemset_n == '(Other)':
        stem_text_a = 'Other'
        stem_text_b = 'the no \"Other\" track'
    elif stemset_n == '(Drums)':
        stem_text_a = 'Drums'
        stem_text_b = 'the no \"Drums\" track'
    elif stemset_n == '(Bass)':
        stem_text_a = 'Bass'
        stem_text_b = 'the no \"Bass\" track'
    else: 
        stem_text_a = 'Vocals'
        stem_text_b = 'Instrumental'
      
    if autocompensate:
        compensate = compensate_set
    else:
        compensate = data['compensate']
        
    if data['noise_pro_select'] == 'Auto Select':
        noise_pro_set = noise_pro
    else:
        noise_pro_set = data['noise_pro_select']
       
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
        #print('normalization on')
    else:
        normalization_set = spec_utils.nonormalize
        #print('normalization off')

    #print(n_fft_scale_set)
    #print(dim_f_set)
    #print(demucs_model_set_name)
    
    inst_only = data['inst_only']
    voc_only = data['voc_only']

    stime = time.perf_counter()
    progress_var.set(0)
    text_widget.clear()
    button_widget.configure(state=tk.DISABLED)  # Disable Button

    try:    #Load File(s)
        for file_num, music_file in tqdm(enumerate(data['input_paths'], start=1)):

            overlap_set = float(data['overlap'])
            channel_set = int(data['channel'])
            margin_set = int(data['margin'])
            shift_set = int(data['shifts'])
            demucs_model_set = demucs_model_set_name
            split_mode = data['split_mode']
            demucs_switch = data['demucsmodel']
            
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

            _mixture = f'{data["input_paths"]}'
            
            timestampnum = round(datetime.utcnow().timestamp())
            randomnum = randrange(100000, 1000000)
        
            if data['settest']:
                try:
                    _basename = f'{data["export_path"]}/{str(timestampnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
                except:
                    _basename = f'{data["export_path"]}/{str(randomnum)}_{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
            else:
                _basename = f'{data["export_path"]}/{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
            # -Get text and update progress-
            base_text = get_baseText(total_files=len(data['input_paths']),
                                        file_num=file_num)
            progress_kwargs = {'progress_var': progress_var,
                            'total_files': len(data['input_paths']),
                            'file_num': file_num}
            
            
            if 'UVR' in demucs_model_set:
                if stemset_n == '(Bass)' or stemset_n == '(Drums)' or stemset_n == '(Other)':
                    widget_text.write('The selected Demucs model can only be used with vocal or instrumental stems.\n')
                    widget_text.write('Please select a 4 stem Demucs model and try again.\n\n')
                    widget_text.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    gui_progress_bar.set(0)
                    widget_button.configure(state=tk.NORMAL)  # Enable Button
                    return

                
            if stemset_n == '(Instrumental)':
                if not 'UVR' in demucs_model_set:
                    widget_text.write(base_text + 'The selected Demucs model cannot be used with this model.\n')
                    widget_text.write(base_text + 'Only 2 stem Demucs models are compatible with this model.\n')
                    widget_text.write(base_text + 'Setting Demucs model to \"UVR_Demucs_Model_1\".\n\n')
                    demucs_model_set = 'UVR_Demucs_Model_1'
            
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
            
            if data['noisereduc_s'] == 'None':
                pass
            else:
                if not os.path.isfile("lib_v5\sox\sox.exe"):
                    data['noisereduc_s'] = 'None'
                    data['non_red'] = False
                    widget_text.write(base_text + 'SoX is missing and required for noise reduction.\n')
                    widget_text.write(base_text + 'See the \"More Info\" tab in the Help Guide.\n')
                    widget_text.write(base_text + 'Noise Reduction will be disabled until SoX is available.\n\n')
        
            update_progress(**progress_kwargs,
                            step=0)       
            
            e = os.path.join(data["export_path"])
            
            demucsmodel = 'models/Demucs_Models/' + str(data['DemucsModel_MDX'])

            pred = Predictor()


            print('\n\nmodeltype: ', modeltype)
            print('noise_pro: ', noise_pro)
            print('stemset_n: ', stemset_n)
            print('compensate_set: ', compensate_set)
            print('source_val: ', source_val)
            print('n_fft_scale_set: ', n_fft_scale_set)
            print('dim_f_set: ', dim_f_set, '\n')

            if modeltype == 'Not Set' or \
            noise_pro == 'Not Set' or \
            stemset_n == 'Not Set' or \
            compensate_set == 'Not Set' or \
            source_val == 'Not Set' or \
            n_fft_scale_set == 'Not Set' or \
            dim_f_set == 'Not Set':
                confirm = tk.messagebox.askyesno(title='Unrecognized Model Detected',
                        message=f'\nWould you like to set the correct model parameters for this model before continuing?\n')
                
                if confirm:
                    pred.mdx_options()
                else:
                    text_widget.write(f'An unrecognized model has been detected.\n\n')
                    text_widget.write(f'Please configure the ONNX model settings accordingly and try again.\n\n')
                    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
                    torch.cuda.empty_cache()
                    progress_var.set(0)
                    button_widget.configure(state=tk.NORMAL)  # Enable Button
                    return
                
            pred.prediction_setup()
            
            #print(demucsmodel)
            
            # split
            pred.prediction(
                m=music_file,
            )
            
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
                            f'Process Method: MDX-Net\n\n' +
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
        
        if model_adv_set_err in message:
            text_widget.write("\n" + base_text + f'Separation failed for the following audio file:\n')
            text_widget.write(base_text + f'"{os.path.basename(music_file)}"\n')
            text_widget.write(f'\nError Received:\n\n')
            text_widget.write(f'The current ONNX model settings are not compatible with the selected \nmodel.\n\n')
            text_widget.write(f'Please re-configure the advanced ONNX model settings accordingly and try \nagain.\n\n')
            text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')
            try:
                with open('errorlog.txt', 'w') as f:
                    f.write(f'Last Error Received:\n\n' +
                            f'Error Received while processing "{os.path.basename(music_file)}":\n' + 
                            f'Process Method: MDX-Net\n\n' +
                            f'The current ONNX model settings are not compatible with the selected model.\n\n' + 
                            f'Please re-configure the advanced ONNX model settings accordingly and try again.\n\n' + 
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
                            f'Process Method: MDX-Net\n\n' +
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
                        f'Process Method: MDX-Net\n\n' +
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
        try:
            torch.cuda.empty_cache()
        except:
            pass
        button_widget.configure(state=tk.NORMAL)  # Enable Button
        return
    
    progress_var.set(0)

    text_widget.write(f'\nConversion(s) Completed!\n')
        
    text_widget.write(f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}')  # nopep8
    torch.cuda.empty_cache() 
    button_widget.configure(state=tk.NORMAL)  # Enable Button
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Successfully completed music demixing.");print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))


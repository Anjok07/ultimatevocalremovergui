# GUI modules
import audioread
import base64
import gui_data.sv_ttk
import hashlib
import json
import librosa
import logging
import math
import natsort
import onnxruntime as onnx
import os
import pickle  # Save Data
import psutil
import pyglet
import pyperclip
import base64
import queue
import re
import shutil
import string
import subprocess
import sys
import soundfile as sf
import time
#import timeit
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
import tkinter.ttk as ttk
import torch
import urllib.request
import webbrowser
import wget
import traceback
#import multiprocessing as KThread
from __version__ import VERSION, PATCH
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
from gui_data.app_size_values import ImagePath, AdjustedValues as av
from gui_data.constants import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
#from gui_data.tkinterdnd2 import TkinterDnD, DND_FILES  # Enable Drag & Drop MACOS_COMMENT
from lib_v5.vr_network.model_param_init import ModelParameters
from kthread import KThread
from lib_v5 import spec_utils
from pathlib  import Path
from separate import SeperateDemucs, SeperateMDX, SeperateVR, save_format
from playsound import playsound
from tkinter import *
from tkinter.tix import *
import re
from typing import List

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('UVR BEGIN')

is_macos = True

right_click_button = '<Button-2>'

try:
    with open(os.path.join(os.getcwd(), 'tmp', 'splash.txt'), 'w') as f:
        f.write('1')
except:
    pass

def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Paramters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    with open('data.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)

def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    try:
        with open('data.pkl', 'rb') as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it

        save_data(data=DEFAULT_DATA)

        return load_data()

def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''
    
    with open(dictionary) as d:
        data = d.read()

    return json.loads(data)

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')

#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-icon.ico')
MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-icon.png')
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.TTF')#ensemble_temps
MENU_COMBOBOX_WIDTH = 18

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')
SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}
data = load_data()

def drop(event, accept_mode: str = 'files'):
    """Drag & Drop verification process"""
    
    path = event.data

    if accept_mode == 'folder':
        path = path.replace('{', '').replace('}', '')
        if not os.path.isdir(path):
            tk.messagebox.showerror(title='Invalid Folder',
                                    message='Your given export path is not a valid folder!')
            return
        # Set Variables
        root.export_path_var.set(path)
    elif accept_mode == 'files':
        # Clean path text and set path to the list of paths
        path = path.replace("{", "").replace("}", "")
        for drive_letter in list(string.ascii_lowercase.upper()):
            path = path.replace(f" {drive_letter}:", f";{drive_letter}:")
        path = path.split(';')
        path[-1] = path[-1].replace(';', '')
        # Set Variables
        root.inputPaths = tuple(path)
        root.process_input_selections()
        root.update_inputPaths()

    else:
        # Invalid accept mode
        return
    
class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False):

        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1
        self.is_normalization = root.is_normalization_var.get()
        self.is_primary_stem_only = root.is_primary_stem_only_var.get()
        self.is_secondary_stem_only = root.is_secondary_stem_only_var.get()
        self.is_denoise = root.is_denoise_var.get()
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        self.is_invert_spec = root.is_invert_spec_var.get()
        self.demucs_stems = root.demucs_stems_var.get()
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.is_demucs_pre_proc_model_inst_mix = False
        
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None

        if selected_process_method == ENSEMBLE_MODE:
            partitioned_name = model_name.partition(ENSEMBLE_PARTITION)
            self.process_method = partitioned_name[0]
            self.model_name = partitioned_name[2]
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            self.is_ensemble_mode = True if not is_secondary_model and not is_pre_proc_model else False
            self.is_4_stem_ensemble = True if root.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE and self.is_ensemble_mode else False
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not self.ensemble_primary_stem == VOCAL_STEM else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var.get() if not self.is_secondary_model else False
            self.aggression_setting = float(int(root.aggression_setting_var.get())/100)
            self.is_tta = root.is_tta_var.get()
            self.is_post_process = root.is_post_process_var.get()
            self.window_size = int(root.window_size_var.get())
            self.batch_size = int(root.batch_size_var.get())
            self.crop_size = int(root.crop_size_var.get())
            self.is_high_end_process = 'mirroring' if root.is_high_end_process_var.get() else 'None'
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER)
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.margin = int(root.margin_var.get())
            self.chunks = root.determine_auto_chunks(root.chunks_var.get(), self.is_gpu_conversion)
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    self.compensate = self.model_data["compensate"] if root.compensate_var.get() == AUTO_SELECT else float(root.compensate_var.get())
                    self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                    self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                    self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var.get() if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not root.demucs_stems_var.get() in [VOCAL_STEM, INST_STEM] else False
            self.overlap = float(root.overlap_var.get())
            self.margin_demucs = int(root.margin_demucs_var.get())
            self.chunks_demucs = root.determine_auto_chunks(root.chunks_demucs_var.get(), self.is_gpu_conversion)
            self.shifts = int(root.shifts_var.get())
            self.is_split_mode = root.is_split_mode_var.get()
            self.segment = root.segment_var.get()
            self.is_chunk_demucs = root.is_chunk_demucs_var.get()
            self.is_demucs_combine_stems = root.is_demucs_combine_stems_var.get()
            self.is_primary_stem_only = root.is_primary_stem_only_var.get() if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var.get() 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var.get() if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var.get()
            self.get_demucs_model_path()
            self.get_demucs_model_data()

        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_status else None
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        if self.is_secondary_model_activated and self.model_status:

            if (not self.is_ensemble_mode and root.demucs_stems_var.get() == ALL_STEMS and self.process_method == DEMUCS_ARCH_TYPE) or self.is_4_stem_ensemble:
                
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                    
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = False if all(i is None for i in self.secondary_model_4_stem) else True
                self.demucs_4_stem_added_count = self.demucs_4_stem_added_count - 1 if self.is_secondary_model_activated else self.demucs_4_stem_added_count
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [None if i is None else i.model_basename for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True 
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and self.process_method == DEMUCS_ARCH_TYPE else self.primary_stem
                self.secondary_model_data(primary_stem)
                
        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model_activated = True
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var.get() if self.pre_proc_model else False

    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
              
    def get_mdx_model_path(self):
        
        for file_name, chosen_mdx_model in MDX_NAME_SELECT.items():
            if self.model_name in chosen_mdx_model:
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}.onnx")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}.onnx")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = [True for x in DEMUCS_NEWER_TAGS if x in self.model_name]
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in DEMUCS_NAME_SELECT.items():
            if self.model_name in chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        self.demucs_source_list = DEMUCS_2_SOURCE if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE
        self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE_MAPPER
        self.demucs_stem_count = 2 if DEMUCS_UVR_MODEL in self.model_name else 4
        
        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]

    def get_model_data(self, model_hash_dir, hash_mapper):

        model_settings_json = os.path.join(model_hash_dir, "{}.json".format(self.model_hash))

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings
            else:
                return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        
        if not self.is_dry_check:
            confirm = tk.messagebox.askyesno(title=UNRECOGNIZED_MODEL[0],
                                            message=f"\"{self.model_name}\"{UNRECOGNIZED_MODEL[1]}",
                                            parent=root)
                                            
            if confirm:
                if self.process_method == VR_ARCH_TYPE:
                    root.pop_up_vr_param(self.model_hash)
                    return root.vr_model_params
                if self.process_method == MDX_ARCH_TYPE:
                    root.pop_up_mdx_model(self.model_hash, self.model_path)
                    return root.mdx_model_params
            else:
                return None
        else:
            return None

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                with open(self.model_path, 'rb') as f:
                    f.seek(- 10000 * 1024, 2)
                    self.model_hash = hashlib.md5(f.read()).hexdigest()
                
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)

class Ensembler():
    def __init__(self, is_manual_ensemble=False):
        self.is_save_all_outputs_ensemble = root.is_save_all_outputs_ensemble_var.get()
        chosen_ensemble_name = '{}'.format(root.chosen_ensemble_var.get().replace(" ", "_")) if not root.chosen_ensemble_var.get() == CHOOSE_ENSEMBLE_OPTION else 'Ensembled'
        ensemble_algorithm = root.ensemble_type_var.get().partition("/")
        ensemble_main_stem_pair = root.ensemble_main_stem_var.get().partition("/")
        time_stamp = round(time.time())
        self.audio_tool = MANUAL_ENSEMBLE
        self.main_export_path = Path(root.export_path_var.get())
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if root.is_append_ensemble_name_var.get() else ''
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else ENSEMBLE_TEMP_PATH
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var.get() else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = root.is_normalization_var.get()
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        if not is_manual_ensemble:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = root.ensemble_type_var.get()
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}.wav'.format(audio_file_output))
        if stem_outputs:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set)
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        algorithm = root.choose_algorithm_var.get()
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}_({}).wav'.format(self.is_testing_audio, audio_file_base, algorithm))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembles"""
        
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]

class AudioTools():
    def __init__(self, audio_tool):
        time_stamp = round(time.time())
        self.audio_tool = audio_tool
        self.main_export_path = Path(root.export_path_var.get())
        self.wav_type_set = root.wav_type_set
        self.is_normalization = root.is_normalization_var.get()
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var.get() else ''
        self.save_format = lambda save_path:save_format(save_path, root.save_format_var.get(), root.mp3_bit_set_var.get())
        
    def align_inputs(self, audio_inputs, audio_file_base, audio_file_2_base, command_Text):
        audio_file_base = f"{self.is_testing_audio}{audio_file_base}"
        audio_file_2_base = f"{self.is_testing_audio}{audio_file_2_base}"
        
        aligned_path = os.path.join('{}'.format(self.main_export_path),'{}_aligned.wav'.format(audio_file_2_base))
        inverted_path = os.path.join('{}'.format(self.main_export_path),'{}_inverted.wav'.format(audio_file_base))

        spec_utils.align_audio(audio_inputs[0], audio_inputs[1], aligned_path, inverted_path, self.wav_type_set, self.is_normalization, command_Text, root.progress_bar_main_var, self.save_format)
        
    def pitch_or_time_shift(self, audio_file, audio_file_base):
        
        rate = float(root.time_stretch_rate_var.get()) if self.audio_tool == TIME_STRETCH else float(root.pitch_rate_var.get())
        is_pitch = False if self.audio_tool == TIME_STRETCH else True
        file_text = TIME_TEXT if self.audio_tool == TIME_STRETCH else PITCH_TEXT
        save_path = os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}{file_text}.wav")
        spec_utils.augment_audio(save_path, audio_file, rate, self.is_normalization, self.wav_type_set, self.save_format, is_pitch=is_pitch)
        
class ToolTip(object):

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                      background="#ffffe0", foreground="black", relief=SOLID, borderwidth=1,
                      font=("tahoma", f"{FONT_SIZE_1}", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ThreadSafeConsole(tk.Text):
    """
    Text Widget which is thread safe for tkinter
    """
    
    def __init__(self, master, **options):
        tk.Text.__init__(self, master, **options)
        self.queue = queue.Queue()
        self.update_me()

    def write(self, line):
        self.queue.put(line)

    def clear(self):
        self.queue.put(None)

    def update_me(self):
        self.configure(state=tk.NORMAL)
        try:
            while 1:
                line = self.queue.get_nowait()
                if line is None:
                    self.delete(1.0, tk.END)
                else:
                    self.insert(tk.END, str(line))
                self.see(tk.END)
                self.update_idletasks()
        except queue.Empty:
            pass
        self.configure(state=tk.DISABLED)
        self.after(100, self.update_me)
        
    def copy_text(self):
        hightlighted_text = self.selection_get()
        self.clipboard_clear()
        self.clipboard_append(hightlighted_text)
        
    def select_all_text(self):
        self.tag_add('sel', '1.0', 'end')

class MainWindow(tk.Tk): #MACOS_COMMENT
    # --Constants--
    # Layout
    
    IMAGE_HEIGHT = av.IMAGE_HEIGHT
    FILEPATHS_HEIGHT = av.FILEPATHS_HEIGHT
    OPTIONS_HEIGHT = av.OPTIONS_HEIGHT
    CONVERSIONBUTTON_HEIGHT = av.CONVERSIONBUTTON_HEIGHT
    COMMAND_HEIGHT = av.COMMAND_HEIGHT
    PROGRESS_HEIGHT = av.PROGRESS_HEIGHT
    PADDING = av.PADDING
    COL1_ROWS = 11
    COL2_ROWS = 11
    
    def __init__(self):
        #Run the __init__ method on the tk.Tk class
        super().__init__()
        
        gui_data.sv_ttk.set_theme("dark")
        gui_data.sv_ttk.use_dark_theme()  # Set dark theme

        # Calculate window height
        height = self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT
        height += self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PROGRESS_HEIGHT
        height += self.PADDING * 5  # Padding
        width = 680
        self.main_window_width = width
        self.main_window_height = height

        # --Window Settings--

        self.title('Ultimate Vocal Remover')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=self.main_window_width,
            height=height,
            xpad=int(self.winfo_screenwidth()/2 - width/2),
            ypad=int(self.winfo_screenheight()/2 - height/2 - 30)))
 
        self.tk.call('wm', 'iconphoto', self._w, tk.PhotoImage(file=MAIN_ICON_IMG_PATH))
        self.configure(bg='#0e0e0f')  # Set background color to #0c0c0d
        self.protocol("WM_DELETE_WINDOW", self.save_values)
        self.resizable(False, False)
        #self.withdraw()
        self.update()

        #Load Images
        img = ImagePath(BASE_PATH)
        self.logo_img = img.open_image(path=img.banner_path, size=(self.winfo_width(), 9999))
        self.efile_img = img.efile_img
        self.stop_img = img.stop_img
        self.help_img = img.help_img
        self.download_img = img.download_img
        self.donate_img = img.donate_img
        self.key_img = img.key_img
        self.credits_img = img.credits_img
        
        #Placeholders
        self.error_log_var = tk.StringVar(value='')
        self.vr_secondary_model_names = []
        self.mdx_secondary_model_names = []
        self.demucs_secondary_model_names = []
        self.vr_primary_model_names = []
        self.mdx_primary_model_names = []
        self.demucs_primary_model_names = []
        
        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}
        
        # -Tkinter Value Holders-
        
        try:
            self.load_saved_vars(data)
        except Exception as e:
            self.error_log_var.set(error_text('Loading Saved Variables', e))
            self.load_saved_vars(DEFAULT_DATA)
            
        self.cached_sources_clear()
        
        self.method_mapper = {
            VR_ARCH_PM: self.vr_model_var,
            MDX_ARCH_TYPE: self.mdx_net_model_var,
            DEMUCS_ARCH_TYPE: self.demucs_model_var}

        self.vr_secondary_model_vars = {'voc_inst_secondary_model': self.vr_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.vr_other_secondary_model_var,
                                        'bass_secondary_model': self.vr_bass_secondary_model_var,
                                        'drums_secondary_model': self.vr_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.vr_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.vr_voc_inst_secondary_model_scale_var,
							            'other_secondary_model_scale': self.vr_other_secondary_model_scale_var,
							            'bass_secondary_model_scale': self.vr_bass_secondary_model_scale_var,
							            'drums_secondary_model_scale': self.vr_drums_secondary_model_scale_var}
        
        self.demucs_secondary_model_vars = {'voc_inst_secondary_model': self.demucs_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.demucs_other_secondary_model_var,
                                        'bass_secondary_model': self.demucs_bass_secondary_model_var,
                                        'drums_secondary_model': self.demucs_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.demucs_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.demucs_voc_inst_secondary_model_scale_var,
							            'other_secondary_model_scale': self.demucs_other_secondary_model_scale_var,
							            'bass_secondary_model_scale': self.demucs_bass_secondary_model_scale_var,
							            'drums_secondary_model_scale': self.demucs_drums_secondary_model_scale_var}
        
        self.mdx_secondary_model_vars = {'voc_inst_secondary_model': self.mdx_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.mdx_other_secondary_model_var,
                                        'bass_secondary_model': self.mdx_bass_secondary_model_var,
                                        'drums_secondary_model': self.mdx_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.mdx_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.mdx_voc_inst_secondary_model_scale_var,
							            'other_secondary_model_scale': self.mdx_other_secondary_model_scale_var,
							            'bass_secondary_model_scale': self.mdx_bass_secondary_model_scale_var,
							            'drums_secondary_model_scale': self.mdx_drums_secondary_model_scale_var}

        #Main Application Vars
        self.progress_bar_main_var = tk.IntVar(value=0)
        self.inputPathsEntry_var = tk.StringVar(value='')
        self.conversion_Button_Text_var = tk.StringVar(value=START_PROCESSING)
        self.chosen_ensemble_var = tk.StringVar(value=CHOOSE_ENSEMBLE_OPTION)
        self.ensemble_main_stem_var = tk.StringVar(value=CHOOSE_STEM_PAIR)
        self.ensemble_type_var = tk.StringVar(value=MAX_MIN)
        self.save_current_settings_var = tk.StringVar(value=SELECT_SAVED_SET)
        self.demucs_stems_var = tk.StringVar(value=ALL_STEMS)
        self.is_primary_stem_only_Text_var = tk.StringVar(value='')
        self.is_secondary_stem_only_Text_var = tk.StringVar(value='')
        self.is_primary_stem_only_Demucs_Text_var = tk.StringVar(value='')
        self.is_secondary_stem_only_Demucs_Text_var = tk.StringVar(value='')
        self.scaling_var = tk.DoubleVar(value=1.0)
        self.active_processing_thread = None
        self.verification_thread = None
        self.is_menu_settings_open = False
        
        self.is_open_menu_advanced_vr_options = tk.BooleanVar(value=False)
        self.is_open_menu_advanced_demucs_options = tk.BooleanVar(value=False)
        self.is_open_menu_advanced_mdx_options = tk.BooleanVar(value=False)
        self.is_open_menu_advanced_ensemble_options = tk.BooleanVar(value=False)
        self.is_open_menu_view_inputs = tk.BooleanVar(value=False)
        self.is_open_menu_help = tk.BooleanVar(value=False)
        self.is_open_menu_error_log = tk.BooleanVar(value=False)

        self.mdx_model_params = None
        self.vr_model_params = None
        self.current_text_box = None
        self.wav_type_set = None
        self.is_online_model_menu = None
        self.progress_bar_var = tk.IntVar(value=0)
        self.is_confirm_error_var = tk.BooleanVar(value=False)
        self.clear_cache_torch = False
        self.vr_hash_MAPPER = load_model_hash_data(VR_HASH_JSON)
        self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
        self.is_gpu_available = torch.cuda.is_available()
        self.is_process_stopped = False
        self.inputs_from_dir = []
        self.iteration = 0
        self.vr_primary_source = None
        self.vr_secondary_source = None
        self.mdx_primary_source = None
        self.mdx_secondary_source = None
        self.demucs_primary_source = None
        self.demucs_secondary_source = None

        #Download Center Vars
        self.online_data = {}
        self.is_online = False
        self.lastest_version = ''
        self.model_download_demucs_var = tk.StringVar(value='')
        self.model_download_mdx_var = tk.StringVar(value='')
        self.model_download_vr_var = tk.StringVar(value='')
        self.selected_download_var = tk.StringVar(value=NO_MODEL)
        self.select_download_var = tk.StringVar(value='')
        self.download_progress_info_var = tk.StringVar(value='')
        self.download_progress_percent_var = tk.StringVar(value='')
        self.download_progress_bar_var = tk.IntVar(value=0)
        self.download_stop_var = tk.StringVar(value='') 
        self.app_update_status_Text_var = tk.StringVar(value='')
        self.app_update_button_Text_var = tk.StringVar(value='')
        self.user_code_validation_var = tk.StringVar(value='')
        self.download_link_path_var = tk.StringVar(value='') 
        self.download_save_path_var = tk.StringVar(value='')
        self.download_update_link_var = tk.StringVar(value='') 
        self.download_update_path_var = tk.StringVar(value='') 
        self.download_demucs_models_list = []
        self.download_demucs_newer_models = []
        self.refresh_list_Button = None
        self.stop_download_Button_DISABLE = None
        self.enable_tabs = None
        self.is_download_thread_active = False
        self.is_process_thread_active = False
        self.is_active_processing_thread = False
        self.active_download_thread = None

        # Font
        pyglet.font.add_file(FONT_PATH)
        self.font = tk.font.Font(family='Century Gothic', size=13)
        self.fontRadio = tk.font.Font(family='Century Gothic', size=12) 
        
        #Model Update
        self.last_found_ensembles = ENSEMBLE_OPTIONS
        self.last_found_settings = ENSEMBLE_OPTIONS
        self.last_found_models = ()
        self.model_data_table = ()
        self.ensemble_model_list = ()
                
        # --Widgets--
        self.fill_main_frame()
        self.bind_widgets()
        self.online_data_refresh(user_refresh=False)

        # --Update Widgets--
        self.update_available_models()
        self.update_main_widget_states()
        self.update_loop()
        self.update_button_states()
        self.delete_temps()
        self.download_validate_code()
        self.ensemble_listbox_Option.configure(state=tk.DISABLED)

        self.command_Text.write(f'Ultimate Vocal Remover {VERSION} [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
        self.new_update_notify = lambda latest_version:self.command_Text.write(f"\n\nNew Update Found: {latest_version}\n\nClick the update button in the \"Settings\" menu to download and install!")
        self.update_checkbox_text = lambda:self.selection_action_process_method(self.chosen_process_method_var.get())
  
    # Menu Functions
    def main_window_LABEL_SET(self, master, text):return ttk.Label(master=master, text=text, background='#0e0e0f', font=self.font, foreground='#13a4c9', anchor=tk.CENTER)
    def menu_title_LABEL_SET(self, frame, text, width=35):return ttk.Label(master=frame, text=text, font=("Century Gothic",  f"{FONT_SIZE_5}", "underline"), justify="center", foreground="#13a4c9", width=width, anchor=tk.CENTER)
    def menu_sub_LABEL_SET(self, frame, text, font_size=FONT_SIZE_2):return ttk.Label(master=frame, text=text, font=("Century Gothic", f"{font_size}"), foreground='#13a4c9', anchor=tk.CENTER)
    def menu_FRAME_SET(self, frame):return Frame(frame, highlightbackground='#0e0e0f', highlightcolor='#0e0e0f', highlightthicknes=20)
    def check_is_menu_settings_open(self):self.menu_settings() if not self.is_menu_settings_open else None
    
    def check_is_open_menu_advanced_vr_options(self):
        if not self.is_open_menu_advanced_vr_options.get():
            self.menu_advanced_vr_options() 
        else:
            self.menu_advanced_vr_options_close_window()
            self.menu_advanced_vr_options()
        
    def check_is_open_menu_advanced_demucs_options(self): 
        if not self.is_open_menu_advanced_demucs_options.get():
            self.menu_advanced_demucs_options() 
        else:
            self.menu_advanced_demucs_options_close_window()
            self.menu_advanced_demucs_options()
  
    def check_is_open_menu_advanced_mdx_options(self): 
        if not self.is_open_menu_advanced_mdx_options.get():
            self.menu_advanced_mdx_options() 
        else:
            self.menu_advanced_mdx_options_close_window()
            self.menu_advanced_mdx_options()

    def check_is_open_menu_advanced_ensemble_options(self): 
        if not self.is_open_menu_advanced_ensemble_options.get():
            self.menu_advanced_ensemble_options() 
        else:
            self.menu_advanced_ensemble_options_close_window()
            self.menu_advanced_ensemble_options()

    def check_is_open_menu_help(self): 
        if not self.is_open_menu_help.get():
            self.menu_help() 
        else:
            self.menu_help_close_window()
            self.menu_help()

    def check_is_open_menu_error_log(self): 
        if not self.is_open_menu_error_log.get():
            self.menu_error_log() 
        else:
            self.menu_error_log_close_window()
            self.menu_error_log()
            
    def check_is_open_menu_view_inputs(self): 
        if not self.is_open_menu_view_inputs.get():
            self.menu_view_inputs() 
        else:
            self.menu_view_inputs_close_window()
            self.menu_view_inputs()
            
    #Ensemble Listbox Functions
    def ensemble_listbox_get_all_selected_models(self):return [self.ensemble_listbox_Option.get(i) for i in self.ensemble_listbox_Option.curselection()]
    def ensemble_listbox_select_from_indexs(self, indexes):return [self.ensemble_listbox_Option.selection_set(i) for i in indexes]
    def ensemble_listbox_clear_and_insert_new(self, model_ensemble_updated):return (self.ensemble_listbox_Option.delete(0, 'end'), [self.ensemble_listbox_Option.insert(tk.END, models) for models in model_ensemble_updated])
    def ensemble_listbox_get_indexes_for_files(self, updated, selected):return [updated.index(model) for model in selected if model in updated]
    
    def process_iteration(self):
        self.iteration = self.iteration + 1
    
    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False):
        
        if arch_type == ENSEMBLE_STEM_CHECK:
            
            model_data = self.model_data_table
            missing_models = [model.model_status for model in model_data if not model.model_status]
            
            if missing_models or not model_data:
                model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.ensemble_model_list]
                self.model_data_table = model_data

        if arch_type == ENSEMBLE_MODE:
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.ensemble_listbox_get_all_selected_models()]
        if arch_type == ENSEMBLE_CHECK:
            model_data: List[ModelData] = [ModelData(model)]
        if arch_type == VR_ARCH_TYPE or arch_type == VR_ARCH_PM:
            model_data: List[ModelData] = [ModelData(model, VR_ARCH_TYPE)]
        if arch_type == MDX_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]
        if arch_type == DEMUCS_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]

        return model_data
        
    def clear_cache(self, network):
        
        if network == VR_ARCH_TYPE:
            dir = VR_HASH_DIR
        if network == MDX_ARCH_TYPE:
            dir = MDX_HASH_DIR     
        
        [os.remove(os.path.join(dir, x)) for x in os.listdir(dir) if x not in 'model_data.json']
        self.vr_model_var.set(CHOOSE_MODEL)
        self.mdx_net_model_var.set(CHOOSE_MODEL)
        self.model_data_table.clear()
        self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION)
        self.ensemble_main_stem_var.set(CHOOSE_STEM_PAIR)
        self.ensemble_listbox_Option.configure(state=tk.DISABLED)
        self.update_checkbox_text()
    
    def thread_check(self, thread_to_check):
        '''Checks if thread is alive'''
        
        is_running = False
        
        if type(thread_to_check) is KThread:
            if thread_to_check.is_alive():
                is_running = True
                
        return is_running
                       
    # -Widget Methods--
    
    def fill_main_frame(self):
        """Creates root window widgets"""
        
        self.title_Label = tk.Label(master=self, image=self.logo_img, compound=tk.TOP)
        self.title_Label.place(x=-2, y=-2)

        button_y = self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT - 8 + self.PADDING*2

        self.fill_filePaths_Frame()
        self.fill_options_Frame()
        
        self.conversion_Button = ttk.Button(master=self, textvariable=self.conversion_Button_Text_var, command=self.process_initialize)
        self.conversion_Button.place(x=50, y=button_y, width=-100, height=35,
                                    relx=0, rely=0, relwidth=1, relheight=0)
        self.conversion_Button_enable = lambda:(self.conversion_Button_Text_var.set(START_PROCESSING), self.conversion_Button.configure(state=tk.NORMAL))
        self.conversion_Button_disable = lambda message:(self.conversion_Button_Text_var.set(message), self.conversion_Button.configure(state=tk.DISABLED))
        
        self.stop_Button = ttk.Button(master=self, image=self.stop_img, command=self.confirm_stop_process)
        self.stop_Button.place(x=-10 - 35, y=button_y, width=35, height=35,
                                relx=1, rely=0, relwidth=0, relheight=0)
        self.help_hints(self.stop_Button, text=STOP_HELP)
        
        self.settings_Button = ttk.Button(master=self, image=self.help_img, command=self.check_is_menu_settings_open)
        self.settings_Button.place(x=-670, y=button_y, width=35, height=35,
                                relx=1, rely=0, relwidth=0, relheight=0)
        self.help_hints(self.settings_Button, text=SETTINGS_HELP)
    
        self.progressbar = ttk.Progressbar(master=self, variable=self.progress_bar_main_var)
        self.progressbar.place(x=25, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PADDING*4, width=-50, height=self.PROGRESS_HEIGHT,
                            relx=0, rely=0, relwidth=1, relheight=0)

         # Select Music Files Option
        self.console_Frame = Frame(master=self, highlightbackground='#101012', highlightcolor='#101012', highlightthicknes=2)
        self.console_Frame.place(x=15, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.PADDING + 5 *3, width=-30, height=self.COMMAND_HEIGHT+7,
                                relx=0, rely=0, relwidth=1, relheight=0)

        self.command_Text = ThreadSafeConsole(master=self.console_Frame, background='#0c0c0d',fg='#898b8e', font=('Century Gothic', FONT_SIZE_4), borderwidth=0)
        self.command_Text.pack(fill=BOTH, expand=1)
        self.command_Text.bind(right_click_button, lambda e:self.right_click_console(e))
            
    def fill_filePaths_Frame(self):
        """Fill Frame with neccessary widgets"""

         # Select Music Files Option
        self.filePaths_Frame = ttk.Frame(master=self)
        self.filePaths_Frame.place(x=10, y=155, width=-20, height=self.FILEPATHS_HEIGHT, relx=0, rely=0, relwidth=1, relheight=0)

        self.filePaths_musicFile_Button = ttk.Button(master=self.filePaths_Frame, text='Select Input', command=self.input_select_filedialog)
        self.filePaths_musicFile_Button.place(x=0, y=5, width=0, height=-5, relx=0, rely=0, relwidth=0.3, relheight=0.5) 
        self.filePaths_musicFile_Entry = ttk.Entry(master=self.filePaths_Frame, textvariable=self.inputPathsEntry_var, font=self.fontRadio, state=tk.DISABLED)
        self.filePaths_musicFile_Entry.place(x=7.5, y=5, width=-50, height=-5, relx=0.3, rely=0, relwidth=0.7, relheight=0.5)                                   
        self.filePaths_musicFile_Open = ttk.Button(master=self, image=self.efile_img, command=lambda:subprocess.Popen(["open", os.path.dirname(self.inputPaths[0]) if self.inputPaths and os.path.isdir(os.path.dirname(self.inputPaths[0])) else self.error_dialoge(INVALID_INPUT)]))  #MACOS_COMMENT
        self.filePaths_musicFile_Open.place(x=-45, y=160, width=35, height=33, relx=1, rely=0, relwidth=0, relheight=0)
        self.filePaths_musicFile_Entry.configure(cursor="hand2")
        self.help_hints(self.filePaths_musicFile_Button, text=INPUT_FOLDER_ENTRY_HELP) 
        self.help_hints(self.filePaths_musicFile_Open, text=INPUT_FOLDER_BUTTON_HELP)     

        # Save To Option
        self.filePaths_saveTo_Button = ttk.Button(master=self.filePaths_Frame, text='Select Output', command=self.export_select_filedialog)
        self.filePaths_saveTo_Button.place(x=0, y=5, width=0, height=-5, relx=0, rely=0.5, relwidth=0.3, relheight=0.5)
        self.filePaths_saveTo_Entry = ttk.Entry(master=self.filePaths_Frame, textvariable=self.export_path_var, font=self.fontRadio, state=tk.DISABLED)
        self.filePaths_saveTo_Entry.place(x=7.5, y=5, width=-50, height=-5, relx=0.3, rely=0.5, relwidth=0.7, relheight=0.5)
        self.filePaths_saveTo_Open = ttk.Button(master=self, image=self.efile_img, command=lambda:subprocess.Popen(["open", self.export_path_var.get() if os.path.isdir(self.export_path_var.get()) else self.error_dialoge(INVALID_EXPORT)]))  #MACOS_COMMENT
        self.filePaths_saveTo_Open.place(x=-45, y=197.5, width=35, height=33, relx=1, rely=0, relwidth=0, relheight=0)
        self.help_hints(self.filePaths_saveTo_Button, text=OUTPUT_FOLDER_ENTRY_HELP) 
        self.help_hints(self.filePaths_saveTo_Entry, text=OUTPUT_FOLDER_ENTRY_HELP) 
        self.help_hints(self.filePaths_saveTo_Open, text=OUTPUT_FOLDER_BUTTON_HELP)    
        
    def fill_options_Frame(self):
        """Fill Frame with neccessary widgets"""
        
        self.options_Frame = ttk.Frame(master=self)
        self.options_Frame.place(x=10, y=250, width=-20, height=self.OPTIONS_HEIGHT, relx=0, rely=0, relwidth=1, relheight=0)

        # -Create Widgets-

        ## Save Format
        self.wav_button = ttk.Radiobutton(master=self.options_Frame, text=WAV, variable=self.save_format_var, value=WAV)
        self.wav_button.place(x=457, y=-5, width=0, height=6, relx=0, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.help_hints(self.wav_button, text=f'{FORMAT_SETTING_HELP}{WAV}')
        self.flac_button = ttk.Radiobutton(master=self.options_Frame, text=FLAC, variable=self.save_format_var, value=FLAC)
        self.flac_button.place(x=300, y=-5, width=0, height=6, relx=1/3, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.help_hints(self.flac_button, text=f'{FORMAT_SETTING_HELP}{FLAC}')
        self.mp3_button = ttk.Radiobutton(master=self.options_Frame, text=MP3, variable=self.save_format_var, value=MP3)
        self.mp3_button.place(x=143, y=-5, width=0, height=6, relx=2/3, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.help_hints(self.mp3_button, text=f'{FORMAT_SETTING_HELP}{MP3}')

        # Choose Conversion Method
        self.chosen_process_method_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose Process Method')#tk.Button(master=self.options_Frame, text='Choose Process Method', anchor=tk.CENTER, background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=lambda:self.pop_up_vr_param('ihbuhb'))
        self.chosen_process_method_Label.place(x=0, y=MAIN_ROW_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.chosen_process_method_Option = ttk.OptionMenu(self.options_Frame, self.chosen_process_method_var, None, *PROCESS_METHODS, command=lambda s:self.selection_action_process_method(s, from_widget=True))
        self.chosen_process_method_Option.place(x=0, y=MAIN_ROW_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.chosen_process_method_var.trace_add('write', lambda *args: self.update_main_widget_states())
        self.help_hints(self.chosen_process_method_Label, text=CHOSEN_PROCESS_METHOD_HELP)
        
        #  Choose Settings Option
        self.save_current_settings_Label = self.main_window_LABEL_SET(self.options_Frame, 'Select Saved Settings')
        self.save_current_settings_Label_place = lambda:self.save_current_settings_Label.place(x=MAIN_ROW_2_X[0], y=LOW_MENU_Y[0], width=0, height=LABEL_HEIGHT, relx=2/3, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.save_current_settings_Option = ttk.OptionMenu(self.options_Frame, self.save_current_settings_var)
        self.save_current_settings_Option_place = lambda:self.save_current_settings_Option.place(x=MAIN_ROW_2_X[1], y=LOW_MENU_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=2/3, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.save_current_settings_Label, text=SAVE_CURRENT_SETTINGS_HELP)
        
        ### MDX-NET ###

        #  Choose MDX-Net Model
        self.mdx_net_model_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose MDX-Net Model')
        self.mdx_net_model_Label_place = lambda:self.mdx_net_model_Label.place(x=0, y=LOW_MENU_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.mdx_net_model_Option = ttk.OptionMenu(self.options_Frame, self.mdx_net_model_var)
        self.mdx_net_model_Option_place = lambda:self.mdx_net_model_Option.place(x=0, y=LOW_MENU_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.mdx_net_model_Label, text=CHOOSE_MODEL_HELP)

        # MDX-chunks
        self.chunks_Label = self.main_window_LABEL_SET(self.options_Frame, 'Chunks')
        self.chunks_Label_place = lambda:self.chunks_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.chunks_Option = ttk.Combobox(self.options_Frame, value=CHUNKS, textvariable=self.chunks_var)
        self.chunks_Option_place = lambda:self.chunks_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.chunks_Option, self.chunks_var, REG_CHUNKS, CHUNKS)
        self.help_hints(self.chunks_Label, text=CHUNKS_HELP)

        # MDX-Margin
        self.margin_Label = self.main_window_LABEL_SET(self.options_Frame, 'Margin Size')
        self.margin_Label_place = lambda:self.margin_Label.place(x=MAIN_ROW_2_X[0], y=MAIN_ROW_2_Y[0], width=0, height=LABEL_HEIGHT, relx=2/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.margin_Option = ttk.Combobox(self.options_Frame, value=MARGIN_SIZE, textvariable=self.margin_var)
        self.margin_Option_place = lambda:self.margin_Option.place(x=MAIN_ROW_2_X[1], y=MAIN_ROW_2_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=2/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.margin_Option, self.margin_var, REG_MARGIN, MARGIN_SIZE)
        self.help_hints(self.margin_Label, text=MARGIN_HELP)

        ### VR ARCH ###
        
        #  Choose VR Model
        self.vr_model_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose VR Model')
        self.vr_model_Label_place = lambda:self.vr_model_Label.place(x=0, y=LOW_MENU_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.vr_model_Option = ttk.OptionMenu(self.options_Frame, self.vr_model_var)
        self.vr_model_Option_place = lambda:self.vr_model_Option.place(x=0, y=LOW_MENU_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.vr_model_Label, text=CHOOSE_MODEL_HELP)
        
        # Aggression Setting
        self.aggression_setting_Label = self.main_window_LABEL_SET(self.options_Frame, 'Aggression Setting')
        self.aggression_setting_Label_place = lambda:self.aggression_setting_Label.place(x=MAIN_ROW_2_X[0], y=MAIN_ROW_2_Y[0], width=0, height=LABEL_HEIGHT, relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.aggression_setting_Option = ttk.Combobox(self.options_Frame, value=VR_AGGRESSION, textvariable=self.aggression_setting_var)
        self.aggression_setting_Option_place = lambda:self.aggression_setting_Option.place(x=MAIN_ROW_2_X[1], y=MAIN_ROW_2_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.aggression_setting_Option, self.aggression_setting_var, REG_AGGRESSION, ['10'])
        self.help_hints(self.aggression_setting_Label, text=AGGRESSION_SETTING_HELP)
        
        # Window Size
        self.window_size_Label = self.main_window_LABEL_SET(self.options_Frame, 'Window Size')#anchor=tk.CENTER
        self.window_size_Label_place = lambda:self.window_size_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.window_size_Option = ttk.Combobox(self.options_Frame, value=VR_WINDOW, textvariable=self.window_size_var)
        self.window_size_Option_place = lambda:self.window_size_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.window_size_Option, self.window_size_var, REG_WINDOW, VR_WINDOW)
        self.help_hints(self.window_size_Label, text=WINDOW_SIZE_HELP)
        
        ### DEMUCS ###
        
        #  Choose Demucs Model
        self.demucs_model_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose Demucs Model')
        self.demucs_model_Label_place = lambda:self.demucs_model_Label.place(x=0, y=LOW_MENU_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.demucs_model_Option = ttk.OptionMenu(self.options_Frame, self.demucs_model_var)
        self.demucs_model_Option_place = lambda:self.demucs_model_Option.place(x=0, y=LOW_MENU_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.demucs_model_Label, text=CHOOSE_MODEL_HELP)

        # Choose Demucs Stems
        self.demucs_stems_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose Stem(s)')
        self.demucs_stems_Label_place = lambda:self.demucs_stems_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.demucs_stems_Option = ttk.OptionMenu(self.options_Frame, self.demucs_stems_var, None)
        self.demucs_stems_Option_place = lambda:self.demucs_stems_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.help_hints(self.demucs_stems_Label, text=DEMUCS_STEMS_HELP)

        # Demucs-Segment
        self.segment_Label = self.main_window_LABEL_SET(self.options_Frame, 'Segment')
        self.segment_Label_place = lambda:self.segment_Label.place(x=MAIN_ROW_2_X[0], y=MAIN_ROW_2_Y[0], width=0, height=LABEL_HEIGHT, relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.segment_Option = ttk.Combobox(self.options_Frame, value=DEMUCS_SEGMENTS, textvariable=self.segment_var)
        self.segment_Option_place = lambda:self.segment_Option.place(x=MAIN_ROW_2_X[1], y=MAIN_ROW_2_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.segment_Option, self.segment_var, REG_SEGMENTS, DEMUCS_SEGMENTS)
        self.help_hints(self.segment_Label, text=SEGMENT_HELP)
     
        # Stem A
        self.is_primary_stem_only_Demucs_Option = ttk.Checkbutton(master=self.options_Frame, textvariable=self.is_primary_stem_only_Demucs_Text_var, variable=self.is_primary_stem_only_Demucs_var, command=lambda:self.is_primary_stem_only_Demucs_Option_toggle())
        self.is_primary_stem_only_Demucs_Option_place = lambda:self.is_primary_stem_only_Demucs_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.is_primary_stem_only_Demucs_Option_toggle = lambda:self.is_secondary_stem_only_Demucs_var.set(False) if self.is_primary_stem_only_Demucs_var.get() else self.is_secondary_stem_only_Demucs_Option.configure(state=tk.NORMAL)
        self.help_hints(self.is_primary_stem_only_Demucs_Option, text=SAVE_STEM_ONLY_HELP)
        
        # Stem B
        self.is_secondary_stem_only_Demucs_Option = ttk.Checkbutton(master=self.options_Frame, textvariable=self.is_secondary_stem_only_Demucs_Text_var, variable=self.is_secondary_stem_only_Demucs_var, command=lambda:self.is_secondary_stem_only_Demucs_Option_toggle())
        self.is_secondary_stem_only_Demucs_Option_place = lambda:self.is_secondary_stem_only_Demucs_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.is_secondary_stem_only_Demucs_Option_toggle = lambda:self.is_primary_stem_only_Demucs_var.set(False) if self.is_secondary_stem_only_Demucs_var.get() else self.is_primary_stem_only_Demucs_Option.configure(state=tk.NORMAL)
        self.is_stem_only_Demucs_Options_Enable = lambda:(self.is_primary_stem_only_Demucs_Option.configure(state=tk.NORMAL), self.is_secondary_stem_only_Demucs_Option.configure(state=tk.NORMAL))
        self.help_hints(self.is_secondary_stem_only_Demucs_Option, text=SAVE_STEM_ONLY_HELP)

        ### ENSEMBLE MODE ###

        # Ensemble Mode
        self.chosen_ensemble_Label = self.main_window_LABEL_SET(self.options_Frame, 'Ensemble Options')
        self.chosen_ensemble_Label_place = lambda:self.chosen_ensemble_Label.place(x=0, y=LOW_MENU_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.chosen_ensemble_Option = ttk.OptionMenu(self.options_Frame, self.chosen_ensemble_var)
        self.chosen_ensemble_Option_place = lambda:self.chosen_ensemble_Option.place(x=0, y=LOW_MENU_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.chosen_ensemble_Label, text=CHOSEN_ENSEMBLE_HELP)
                        
        # Ensemble Main Stems
        self.ensemble_main_stem_Label = self.main_window_LABEL_SET(self.options_Frame, 'Main Stem Pair')
        self.ensemble_main_stem_Label_place = lambda:self.ensemble_main_stem_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.ensemble_main_stem_Option = ttk.OptionMenu(self.options_Frame, self.ensemble_main_stem_var, None, *ENSEMBLE_MAIN_STEM, command=self.selection_action_ensemble_stems)
        self.ensemble_main_stem_Option_place = lambda:self.ensemble_main_stem_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.ensemble_main_stem_Label, text=ENSEMBLE_MAIN_STEM_HELP)

        # Ensemble Algorithm
        self.ensemble_type_Label = self.main_window_LABEL_SET(self.options_Frame, 'Ensemble Algorithm')
        self.ensemble_type_Label_place = lambda:self.ensemble_type_Label.place(x=MAIN_ROW_2_X[0], y=MAIN_ROW_2_Y[0], width=0, height=LABEL_HEIGHT, relx=2/3, rely=2/11, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.ensemble_type_Option = ttk.OptionMenu(self.options_Frame, self.ensemble_type_var, None, *ENSEMBLE_TYPE)
        self.ensemble_type_Option_place = lambda:self.ensemble_type_Option.place(x=MAIN_ROW_2_X[1], y=MAIN_ROW_2_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT,relx=2/3, rely=3/11, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.help_hints(self.ensemble_type_Label, text=ENSEMBLE_TYPE_HELP)
        
         # Select Music Files Option
    
        # Ensemble Save Ensemble Outputs
        
        self.ensemble_listbox_Label = self.main_window_LABEL_SET(self.options_Frame, 'Available Models')
        self.ensemble_listbox_Label_place = lambda:self.ensemble_listbox_Label.place(x=MAIN_ROW_2_X[0], y=MAIN_ROW_2_Y[1], width=0, height=LABEL_HEIGHT, relx=2/3, rely=5/11, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.ensemble_listbox_Frame = Frame(self.options_Frame, highlightbackground='#04332c', highlightcolor='#04332c', highlightthicknes=1)
        self.ensemble_listbox_Option = tk.Listbox(self.ensemble_listbox_Frame, selectmode=tk.MULTIPLE, activestyle='dotbox', font=("Century Gothic", f"{FONT_SIZE_1}"), background='#070708', exportselection=0, relief=SOLID, borderwidth=0)
        self.ensemble_listbox_scroll = ttk.Scrollbar(self.options_Frame, orient=VERTICAL)
        self.ensemble_listbox_Option.config(yscrollcommand=self.ensemble_listbox_scroll.set)
        self.ensemble_listbox_scroll.configure(command=self.ensemble_listbox_Option.yview)
        self.ensemble_listbox_Option_place = lambda:(self.ensemble_listbox_Frame.place(x=-25, y=-20, width=0, height=67, relx=2/3, rely=6/11, relwidth=1/3, relheight=1/self.COL1_ROWS), 
                                                     self.ensemble_listbox_scroll.place(x=195, y=-20, width=-48, height=69, relx=2/3, rely=6/11, relwidth=1/10, relheight=1/self.COL1_ROWS))
        self.ensemble_listbox_Option_pack = lambda:self.ensemble_listbox_Option.pack(fill=BOTH, expand=1)
        self.help_hints(self.ensemble_listbox_Label, text=ENSEMBLE_LISTBOX_HELP)
        
        ### AUDIO TOOLS ###

        # Chosen Audio Tool
        self.chosen_audio_tool_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose Audio Tool')
        self.chosen_audio_tool_Label_place = lambda:self.chosen_audio_tool_Label.place(x=0, y=LOW_MENU_Y[0], width=LEFT_ROW_WIDTH, height=LABEL_HEIGHT, relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.chosen_audio_tool_Option = ttk.OptionMenu(self.options_Frame, self.chosen_audio_tool_var, None, *AUDIO_TOOL_OPTIONS)
        self.chosen_audio_tool_Option_place = lambda:self.chosen_audio_tool_Option.place(x=0, y=LOW_MENU_Y[1], width=LEFT_ROW_WIDTH, height=OPTION_HEIGHT, relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.chosen_audio_tool_var.trace_add('write', lambda *args: self.update_main_widget_states())
        self.help_hints(self.chosen_audio_tool_Label, text=AUDIO_TOOLS_HELP)
        
        # Choose Agorithim
        self.choose_algorithm_Label = self.main_window_LABEL_SET(self.options_Frame, 'Choose Algorithm')
        self.choose_algorithm_Label_place = lambda:self.choose_algorithm_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.choose_algorithm_Option = ttk.OptionMenu(self.options_Frame,  self.choose_algorithm_var, None, *MANUAL_ENSEMBLE_OPTIONS)
        self.choose_algorithm_Option_place = lambda:self.choose_algorithm_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # Time Stretch
        self.time_stretch_rate_Label = self.main_window_LABEL_SET(self.options_Frame, 'Rate')
        self.time_stretch_rate_Label_place = lambda:self.time_stretch_rate_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.time_stretch_rate_Option = ttk.Combobox(self.options_Frame, value=TIME_PITCH, textvariable=self.time_stretch_rate_var)
        self.time_stretch_rate_Option_place = lambda:self.time_stretch_rate_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.time_stretch_rate_Option, self.time_stretch_rate_var, REG_TIME_PITCH, TIME_PITCH)

        # Pitch Rate
        self.pitch_rate_Label = self.main_window_LABEL_SET(self.options_Frame, 'Semitones')
        self.pitch_rate_Label_place = lambda:self.pitch_rate_Label.place(x=MAIN_ROW_X[0], y=MAIN_ROW_Y[0], width=0, height=LABEL_HEIGHT, relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.pitch_rate_Option = ttk.Combobox(self.options_Frame, value=TIME_PITCH, textvariable=self.pitch_rate_var)
        self.pitch_rate_Option_place = lambda:self.pitch_rate_Option.place(x=MAIN_ROW_X[1], y=MAIN_ROW_Y[1], width=MAIN_ROW_WIDTH, height=OPTION_HEIGHT, relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.combobox_entry_validation(self.pitch_rate_Option, self.pitch_rate_var, REG_TIME_PITCH, TIME_PITCH)

        ### SHARED SETTINGS ###
        
        # GPU Selection
        self.is_gpu_conversion_Option = ttk.Checkbutton(master=self.options_Frame, text='GPU Conversion', variable=self.is_gpu_conversion_var)
        self.is_gpu_conversion_Option_place = lambda:self.is_gpu_conversion_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.is_gpu_conversion_Disable = lambda:(self.is_gpu_conversion_Option.configure(state=tk.DISABLED), self.is_gpu_conversion_var.set(False))
        self.help_hints(self.is_gpu_conversion_Option, text=IS_GPU_CONVERSION_HELP)

        # Vocal Only
        self.is_primary_stem_only_Option = ttk.Checkbutton(master=self.options_Frame, textvariable=self.is_primary_stem_only_Text_var, variable=self.is_primary_stem_only_var, command=lambda:self.is_primary_stem_only_Option_toggle())
        self.is_primary_stem_only_Option_place = lambda:self.is_primary_stem_only_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.is_primary_stem_only_Option_toggle = lambda:self.is_secondary_stem_only_var.set(False) if self.is_primary_stem_only_var.get() else self.is_secondary_stem_only_Option.configure(state=tk.NORMAL)
        self.help_hints(self.is_primary_stem_only_Option, text=SAVE_STEM_ONLY_HELP)
        
        # Instrumental Only 
        self.is_secondary_stem_only_Option = ttk.Checkbutton(master=self.options_Frame, textvariable=self.is_secondary_stem_only_Text_var, variable=self.is_secondary_stem_only_var, command=lambda:self.is_secondary_stem_only_Option_toggle())
        self.is_secondary_stem_only_Option_place = lambda:self.is_secondary_stem_only_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.is_secondary_stem_only_Option_toggle = lambda:self.is_primary_stem_only_var.set(False) if self.is_secondary_stem_only_var.get() else self.is_primary_stem_only_Option.configure(state=tk.NORMAL)
        self.is_stem_only_Options_Enable = lambda:(self.is_primary_stem_only_Option.configure(state=tk.NORMAL), self.is_secondary_stem_only_Option.configure(state=tk.NORMAL))
        self.help_hints(self.is_secondary_stem_only_Option, text=SAVE_STEM_ONLY_HELP)
        
        # Sample Mode
        self.model_sample_mode_Option = ttk.Checkbutton(master=self.options_Frame, textvariable=self.model_sample_mode_duration_checkbox_var, variable=self.model_sample_mode_var)#f'Sample ({self.model_sample_mode_duration_var.get()} Seconds)'
        self.model_sample_mode_Option_place = lambda rely=8:self.model_sample_mode_Option.place(x=CHECK_BOX_X, y=CHECK_BOX_Y, width=CHECK_BOX_WIDTH, height=CHECK_BOX_HEIGHT, relx=1/3, rely=rely/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.help_hints(self.model_sample_mode_Option, text=MODEL_SAMPLE_MODE_HELP)
        
        self.GUI_LIST = (self.vr_model_Label,
        self.vr_model_Option,
        self.aggression_setting_Label,
        self.aggression_setting_Option,
        self.window_size_Label,
        self.window_size_Option,
        self.demucs_model_Label,
        self.demucs_model_Option,
        self.demucs_stems_Label,
        self.demucs_stems_Option,
        self.segment_Label,
        self.segment_Option,
        self.mdx_net_model_Label,
        self.mdx_net_model_Option,
        self.chunks_Label,
        self.chunks_Option,
        self.margin_Label,
        self.margin_Option,
        self.chosen_ensemble_Label,
        self.chosen_ensemble_Option,
        self.save_current_settings_Label,
        self.save_current_settings_Option,
        self.ensemble_main_stem_Label,
        self.ensemble_main_stem_Option,
        self.ensemble_type_Label,
        self.ensemble_type_Option,
        self.ensemble_listbox_Label,
        self.ensemble_listbox_Frame,
        self.ensemble_listbox_Option,
        self.ensemble_listbox_scroll,
        self.chosen_audio_tool_Label,
        self.chosen_audio_tool_Option,
        self.choose_algorithm_Label,
        self.choose_algorithm_Option,
        self.time_stretch_rate_Label,
        self.time_stretch_rate_Option,
        self.pitch_rate_Label,
        self.pitch_rate_Option,
        self.is_gpu_conversion_Option,
        self.is_primary_stem_only_Option,
        self.is_secondary_stem_only_Option,
        self.is_primary_stem_only_Demucs_Option,
        self.is_secondary_stem_only_Demucs_Option,
        self.model_sample_mode_Option)
        
        REFRESH_VARS = (self.mdx_net_model_var,
                        self.vr_model_var,
                        self.demucs_model_var,
                        self.demucs_stems_var,
                        self.is_chunk_demucs_var,
                        self.is_primary_stem_only_Demucs_var,
                        self.is_secondary_stem_only_Demucs_var,
                        self.is_primary_stem_only_var,
                        self.is_secondary_stem_only_var,
                        self.model_download_demucs_var,
                        self.model_download_mdx_var,
                        self.model_download_vr_var,
                        self.select_download_var,
                        self.is_primary_stem_only_Demucs_Text_var,
                        self.is_secondary_stem_only_Demucs_Text_var,
                        self.chosen_process_method_var,
                        self.ensemble_main_stem_var)
        
        # Change States
        for var in REFRESH_VARS:
            var.trace_add('write', lambda *args: self.update_button_states())
    
    def combobox_entry_validation(self, combobox: ttk.Combobox, var: tk.StringVar, pattern, default):
        """Verifies valid input for comboboxes"""
        
        validation = lambda value:False if re.fullmatch(pattern, value) is None else True
        invalid = lambda:(var.set(default[0]))
        combobox.config(validate='focus', validatecommand=(self.register(validation), '%P'), invalidcommand=(self.register(invalid),))

    def bind_widgets(self):
        """Bind widgets to the drag & drop mechanic"""
        
        self.chosen_audio_tool_align = tk.BooleanVar(value=True)
        add_align = lambda e:(self.chosen_audio_tool_Option['menu'].add_radiobutton(label=ALIGN_INPUTS, command=tk._setit(self.chosen_audio_tool_var, ALIGN_INPUTS)), self.chosen_audio_tool_align.set(False)) if self.chosen_audio_tool_align else None
        
        # self.filePaths_saveTo_Button.drop_target_register(DND_FILES) MACOS_COMMENT
        # self.filePaths_saveTo_Entry.drop_target_register(DND_FILES) MACOS_COMMENT
        # self.drop_target_register(DND_FILES) MACOS_COMMENT
        
        #self.dnd_bind('<<Drop>>', lambda e: drop(e, accept_mode='files')) MACOS_COMMENT
        self.bind("<a> <s> <\>", add_align)
        # self.filePaths_saveTo_Button.dnd_bind('<<Drop>>', lambda e: drop(e, accept_mode='folder')) MACOS_COMMENT
        # self.filePaths_saveTo_Entry.dnd_bind('<<Drop>>', lambda e: drop(e, accept_mode='folder')) MACOS_COMMENT
        self.ensemble_listbox_Option.bind('<<ListboxSelect>>', lambda e: self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION))
        
        self.options_Frame.bind(right_click_button, lambda e:self.right_click_menu_popup(e, main_menu=True))
        self.filePaths_musicFile_Entry.bind(right_click_button, lambda e:self.input_right_click_menu(e))
        self.filePaths_musicFile_Entry.bind('<Button-1>', lambda e:self.check_is_open_menu_view_inputs())
        
    #--Input/Export Methods--
    
    def input_select_filedialog(self):
        """Make user select music files"""

        if self.lastDir is not None:
            if not os.path.isdir(self.lastDir):
                self.lastDir = None

        paths = tk.filedialog.askopenfilenames(
            parent=self,
            title=f'Select Music Files',
            initialfile='',
            initialdir=self.lastDir)
        
        if paths:  # Path selected
            self.inputPaths = paths
            
            self.process_input_selections()
            self.update_inputPaths()

    def export_select_filedialog(self):
        """Make user select a folder to export the converted files in"""

        export_path = None
        
        path = tk.filedialog.askdirectory(
            parent=self,
            title=f'Select Folder',)

        if path:  # Path selected
            self.export_path_var.set(path)
            export_path = self.export_path_var.get()
            
        return export_path
     
    def update_inputPaths(self):
        """Update the music file entry"""
        
        if self.inputPaths:
            if len(self.inputPaths) == 1:
                text = self.inputPaths[0]
            else:
                count = len(self.inputPaths) - 1
                file_text = 'file' if len(self.inputPaths) == 2 else 'files'
                text = f"{self.inputPaths[0]}, +{count} {file_text}"
        else:
            # Empty Selection
            text = ''
            
        self.inputPathsEntry_var.set(text)

    #--Utility Methods--

    def restart(self):
        """Restart the application after asking for confirmation"""
        
        confirm = tk.messagebox.askyesno(title='Restart Confirmation',
                message='This will restart the application and halt any running processes. Your current settings will be saved. \n\n Are you sure you wish to continue?')
        
        if confirm:
            #self.save_values()
            try:
                subprocess.Popen(f'/Applications/Ultimate\ Vocal\ Remover.app') # MAC_COMMENT
            except Exception:
                logging.exception("Restart")
                subprocess.Popen(f'python "{__file__}"', shell=True)
                
            self.destroy()
        
    def delete_temps(self):  
        """Deletes temp files"""
        
        DIRECTORIES = (BASE_PATH, VR_MODELS_DIR, MDX_MODELS_DIR, DEMUCS_MODELS_DIR, DEMUCS_NEWER_REPO_DIR)
        EXTENSIONS = (('.aes', '.txt', '.tmp'))
        
        try:
            if os.path.isfile(f"{PATCH}.dmg"): # MAC_COMMENT
                os.remove(f"{PATCH}.dmg") # MAC_COMMENT
            
            if os.path.isfile(SPLASH_DOC):
                os.remove(SPLASH_DOC)
            
            for dir in DIRECTORIES:
                for temp_file in os.listdir(dir):
                    if temp_file.endswith(EXTENSIONS):
                        if os.path.isfile(os.path.join(dir, temp_file)):
                            os.remove(os.path.join(dir, temp_file))
        except Exception as e:
            self.error_log_var.set(error_text('Temp File Deletion', e))
        
    def get_files_from_dir(self, directory, ext):
        """Gets files from specified directory that ends with specified extention"""

        return tuple(os.path.splitext(x)[0] for x in os.listdir(directory) if x.endswith(ext))
        
    def determine_auto_chunks(self, chunks, gpu):
        """Determines appropriate chunk size based on user computer specs"""
        
        if chunks == 'Full':
            chunk_set = 0
        elif chunks == 'Auto':
            if gpu == 0:
                gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                if gpu_mem <= int(6):
                    chunk_set = int(5)
                if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(10)
                if gpu_mem >= int(16):
                    chunk_set = int(40)
            if gpu == -1:
                sys_mem = psutil.virtual_memory().total >> 30
                if sys_mem <= int(4):
                    chunk_set = int(1)
                if sys_mem in [5, 6, 7, 8]:
                    chunk_set = int(10)
                if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                    chunk_set = int(25)
                if sys_mem >= int(17):
                    chunk_set = int(60) 
        elif chunks == '0':
            chunk_set = 0
        else:
            chunk_set = int(chunks)
                 
        return chunk_set

    def return_ensemble_stems(self, is_primary=False): 
        """Grabs and returns the chosen ensemble stems."""
        
        ensemble_stem = self.ensemble_main_stem_var.get().partition("/")
        
        if is_primary:
            return ensemble_stem[0]
        else:
            return ensemble_stem[0], ensemble_stem[2]

    def message_box(self, message):
        """Template for confirmation box"""
        
        confirm = tk.messagebox.askyesno(title=message[0],
                                         message=message[1],
                                         parent=root)
        
        return confirm

    def error_dialoge(self, message):
        """Template for messagebox that informs user of error"""

        tk.messagebox.showerror(master=self,
                                  title=message[0],
                                  message=message[1],
                                  parent=root) 
      
    def model_list(self, primary_stem: str, secondary_stem: str, is_4_stem_check=False, is_dry_check=False, is_no_demucs=False):
        stem_check = self.assemble_model_data(arch_type=ENSEMBLE_STEM_CHECK, is_dry_check=is_dry_check)
        
        if is_no_demucs:
            return [model.model_and_process_tag for model in stem_check if model.primary_stem == primary_stem or model.primary_stem == secondary_stem]
        else:
            if is_4_stem_check:
                return [model.model_and_process_tag for model in stem_check if model.demucs_stem_count == 4]
            else:
                return [model.model_and_process_tag for model in stem_check if model.primary_stem == primary_stem or model.primary_stem == secondary_stem or primary_stem.lower() in model.demucs_source_list]
      
    def help_hints(self, widget, text):
        toolTip = ToolTip(widget)
        def enter(event):
            if self.help_hints_var.get():
                toolTip.showtip(text)
        def leave(event):
            toolTip.hidetip()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)
        widget.bind(right_click_button, lambda e:copy_help_hint(e))
        
        def copy_help_hint(event):
            if self.help_hints_var.get():
                right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
                right_click_menu.add_command(label='Copy Help Hint Text', command=right_click_menu_copy_hint)
                
                try:
                    right_click_menu.tk_popup(event.x_root,event.y_root)
                finally:
                    right_click_menu.grab_release()
            else:
                self.right_click_menu_popup(event, main_menu=True)

        def right_click_menu_copy_hint():
            pyperclip.copy(text)
      
    def input_right_click_menu(self, event):

        right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
        right_click_menu.add_command(label='See All Inputs', command=self.check_is_open_menu_view_inputs)
        
        try:
            right_click_menu.tk_popup(event.x_root,event.y_root)
        finally:
            right_click_menu.grab_release()

    def cached_sources_clear(self):

        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}
      
    def cached_model_source_holder(self, process_method, sources, model_name=None):
        
        if process_method == VR_ARCH_TYPE:
            self.vr_cache_source_mapper = {**self.vr_cache_source_mapper, **{model_name: sources}}
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}
        if process_method == DEMUCS_ARCH_TYPE:
            self.demucs_cache_source_mapper = {**self.demucs_cache_source_mapper, **{model_name: sources}}
                             
    def cached_source_callback(self, process_method, model_name=None):
        
        model, sources = None, None
        
        if process_method == VR_ARCH_TYPE:
            mapper = self.vr_cache_source_mapper
        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper
        if process_method == DEMUCS_ARCH_TYPE:
            mapper = self.demucs_cache_source_mapper
        
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value
        
        return model, sources
  
    def cached_source_model_list_check(self, model_list: list[ModelData]):

        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.vr_primary_model_names = primary_model_names(VR_ARCH_TYPE)
        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
        self.vr_secondary_model_names = secondary_model_names(VR_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
        self.demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
        self.demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
        
        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)
        
        print('self.demucs_pre_proc_model_name: ', self.demucs_pre_proc_model_name)
        
        self.all_models = self.vr_primary_model_names + self.mdx_primary_model_names + self.demucs_primary_model_names + self.vr_secondary_model_names + self.mdx_secondary_model_names + self.demucs_secondary_model_names + self.demucs_pre_proc_model_name
      
    def verify_audio(self, audio_file, is_process=True, sample_path=None):
        is_good = False
        error_data = ''
        
        if os.path.isfile(audio_file):
            try:
                librosa.load(audio_file, duration=3, mono=False, sr=44100) if not type(sample_path) is str else self.create_sample(audio_file, sample_path)
                is_good = True
            except Exception as e:
                error_name = f'{type(e).__name__}'
                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                message = f'{error_name}: "{e}"\n{traceback_text}"'
                if is_process:
                    audio_base_name = os.path.basename(audio_file)
                    self.error_log_var.set(f'Error Loading the Following File:\n\n\"{audio_base_name}\"\n\nRaw Error Details:\n\n{message}')
                else:
                    error_data = AUDIO_VERIFICATION_CHECK(audio_file, message)

        if is_process:
            return is_good
        else:
            return is_good, error_data
      
    def create_sample(self, audio_file, sample_path=SAMPLE_CLIP_PATH):
        try:
            with audioread.audio_open(audio_file) as f:
                track_length = int(f.duration)
        except Exception as e:
            print('Audioread failed to get duration. Trying Librosa...')
            y, sr = librosa.load(audio_file, mono=False, sr=44100)
            track_length = int(librosa.get_duration(y=y, sr=sr))
        
        clip_duration = int(self.model_sample_mode_duration_var.get())
        
        if track_length >= clip_duration:
            offset_cut = track_length//3
            off_cut = offset_cut + track_length
            if not off_cut >= clip_duration:
                offset_cut = 0
            name_apped = f'{clip_duration}_second_'
        else:
            offset_cut, clip_duration = 0, track_length
            name_apped = ''

        sample = librosa.load(audio_file, offset=offset_cut, duration=clip_duration, mono=False, sr=44100)[0].T
        audio_sample = os.path.join(sample_path, f'{os.path.splitext(os.path.basename(audio_file))[0]}_{name_apped}sample.wav')
        sf.write(audio_sample, sample, 44100)
        
        return audio_sample

    #--Right Click Menu Pop-Ups--

    def right_click_select_settings_sub(self, parent_menu, process_method):
        saved_settings_sub_menu = Menu(parent_menu, font=('Century Gothic', FONT_SIZE_1), tearoff=False)
        settings_options = self.last_found_settings + SAVE_SET_OPTIONS
        
        for settings_options in settings_options:
            settings_options = settings_options.replace("_", " ")
            saved_settings_sub_menu.add_command(label=settings_options, command=lambda o=settings_options:self.selection_action_saved_settings(o, process_method=process_method))

        saved_settings_sub_menu.insert_separator(len(self.last_found_settings))
        
        return saved_settings_sub_menu
    
    def right_click_menu_popup(self, event, text_box=False, main_menu=False):
        
        right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
        
        PM_RIGHT_CLICK_MAPPER = {
                        ENSEMBLE_MODE:self.check_is_open_menu_advanced_ensemble_options,
                        VR_ARCH_PM:self.check_is_open_menu_advanced_vr_options,
                        MDX_ARCH_TYPE:self.check_is_open_menu_advanced_mdx_options,
                        DEMUCS_ARCH_TYPE:self.check_is_open_menu_advanced_demucs_options}
        
        PM_RIGHT_CLICK_VAR_MAPPER = {
                        ENSEMBLE_MODE:True,
                        VR_ARCH_PM:self.vr_is_secondary_model_activate_var.get(),
                        MDX_ARCH_TYPE:self.mdx_is_secondary_model_activate_var.get(),
                        DEMUCS_ARCH_TYPE:self.demucs_is_secondary_model_activate_var.get()}
        
        saved_settings_sub_load_for_menu = Menu(right_click_menu, font=('Century Gothic', FONT_SIZE_1), tearoff=False)
        saved_settings_sub_load_for_menu.add_cascade(label=VR_ARCH_SETTING_LOAD, menu=self.right_click_select_settings_sub(saved_settings_sub_load_for_menu, VR_ARCH_PM))
        saved_settings_sub_load_for_menu.add_cascade(label=MDX_SETTING_LOAD, menu=self.right_click_select_settings_sub(saved_settings_sub_load_for_menu, MDX_ARCH_TYPE))
        saved_settings_sub_load_for_menu.add_cascade(label=DEMUCS_SETTING_LOAD, menu=self.right_click_select_settings_sub(saved_settings_sub_load_for_menu, DEMUCS_ARCH_TYPE))
        saved_settings_sub_load_for_menu.add_cascade(label=ALL_ARCH_SETTING_LOAD, menu=self.right_click_select_settings_sub(saved_settings_sub_load_for_menu, None))

        if not main_menu:
            right_click_menu.add_command(label='Copy', command=self.right_click_menu_copy)
            right_click_menu.add_command(label='Paste', command=lambda:self.right_click_menu_paste(text_box=text_box))
            right_click_menu.add_command(label='Delete', command=lambda:self.right_click_menu_delete(text_box=text_box))
        else:
            for method_type, option in PM_RIGHT_CLICK_MAPPER.items():
                if method_type == self.chosen_process_method_var.get():
                    if PM_RIGHT_CLICK_VAR_MAPPER[method_type] or (method_type == DEMUCS_ARCH_TYPE and self.is_demucs_pre_proc_model_activate_var.get()):
                        right_click_menu.add_cascade(label='Select Saved Settings', menu=saved_settings_sub_load_for_menu)
                        right_click_menu.add_separator()
                        for method_type_sub, option_sub in PM_RIGHT_CLICK_MAPPER.items():
                            if method_type_sub == ENSEMBLE_MODE and not self.chosen_process_method_var.get() == ENSEMBLE_MODE:
                                pass
                            else:
                                right_click_menu.add_command(label=f'Advanced {method_type_sub} Settings', command=option_sub)
                    else:
                        right_click_menu.add_command(label=f'Advanced {method_type} Settings', command=option)
                        break
                    
            if not self.is_menu_settings_open:
                right_click_menu.add_command(label='Additional Settings', command=lambda:self.menu_settings(select_tab_2=True))
                    
            help_hints_label = 'Enable' if self.help_hints_var.get() == False else 'Disable'
            help_hints_bool = True if self.help_hints_var.get() == False else False
            right_click_menu.add_command(label=f'{help_hints_label} Help Hints', command=lambda:self.help_hints_var.set(help_hints_bool))
            
            if self.error_log_var.get():
                right_click_menu.add_command(label='Error Log', command=self.check_is_open_menu_error_log)
            
        try:
            right_click_menu.tk_popup(event.x_root,event.y_root)
        finally:
            right_click_menu.grab_release()

    def right_click_menu_copy(self):
        hightlighted_text = self.current_text_box.selection_get()
        self.clipboard_clear()
        self.clipboard_append(hightlighted_text)

    def right_click_menu_paste(self, text_box=False):
        clipboard = self.clipboard_get()
        self.right_click_menu_delete(text_box=True) if text_box else self.right_click_menu_delete()
        self.current_text_box.insert(self.current_text_box.index(tk.INSERT), clipboard)

    def right_click_menu_delete(self, text_box=False):
        if text_box:
            try:
                s0 = self.current_text_box.index("sel.first")
                s1 = self.current_text_box.index("sel.last")
                self.current_text_box.tag_configure('highlight')
                self.current_text_box.tag_add("highlight", s0, s1)
                start_indexes = self.current_text_box.tag_ranges("highlight")[0::2]
                end_indexes = self.current_text_box.tag_ranges("highlight")[1::2]

                for start, end in zip(start_indexes, end_indexes):
                    self.current_text_box.tag_remove("highlight", start, end)

                for start, end in zip(start_indexes, end_indexes):
                    self.current_text_box.delete(start, end)
            except Exception as e:
                print('RIGHT-CLICK-DELETE ERROR: \n', e)
        else:
            self.current_text_box.delete(0, END)
    
    def right_click_console(self, event):
        right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
        right_click_menu.add_command(label='Copy', command=self.command_Text.copy_text)
        right_click_menu.add_command(label='Select All', command=self.command_Text.select_all_text)
        
        try:
            right_click_menu.tk_popup(event.x_root,event.y_root)
        finally:
            right_click_menu.grab_release()

    #--Secondary Window Methods--

    def menu_placement(self, window: Toplevel, title, pop_up=False, is_help_hints=False, close_function=None):
        """Prepares and centers each secondary window relative to the main window"""
        
        window.wm_attributes('-alpha', 0.0) if is_macos else None
        window.geometry("+%d+%d" %(8000, 5000))
        window.resizable(False, False)
        window.wm_transient(root)
        window.title(title)
        window.iconbitmap(None) #MACOS_COMMENT
        window.update()
        window.deiconify()

        root_location_x = root.winfo_x()
        root_location_y = root.winfo_y()
        
        root_x = root.winfo_width() 
        root_y = root.winfo_height()

        sub_menu_x = window.winfo_width() 
        sub_menu_y = window.winfo_height()

        menu_offset_x = (root_x - sub_menu_x) // 2
        menu_offset_y = (root_y - sub_menu_y) // 2
        window.geometry("+%d+%d" %(root_location_x+menu_offset_x, root_location_y+menu_offset_y))
        window.wm_attributes('-alpha', 1.0) if is_macos else None

        def right_click_menu(event):
            help_hints_label = 'Enable' if self.help_hints_var.get() == False else 'Disable'
            help_hints_bool = True if self.help_hints_var.get() == False else False
            right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
            if is_help_hints:
                right_click_menu.add_command(label=f'{help_hints_label} Help Hints', command=lambda:self.help_hints_var.set(help_hints_bool))
            right_click_menu.add_command(label='Exit Window', command=close_function)
            
            try:
                right_click_menu.tk_popup(event.x_root,event.y_root)
            finally:
                right_click_menu.grab_release()
        
        if close_function:
            window.bind(right_click_button, lambda e:right_click_menu(e))
            
        if pop_up:
            window.grab_set()
            root.wait_window(window)
        
    def menu_tab_control(self, toplevel, ai_network_vars, is_demucs=False):
        """Prepares the tabs setup for some windows"""
        
        tabControl = ttk.Notebook(toplevel)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Settings Guide')
        tabControl.add(tab2, text ='Secondary Model')
        
        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        
        self.menu_secondary_model(tab2, ai_network_vars)
        
        if is_demucs:
            tab3 = ttk.Frame(tabControl)
            tabControl.add(tab3, text ='Pre-process Model')
            tab3.grid_rowconfigure(0, weight=1)
            tab3.grid_columnconfigure(0, weight=1)
            
            return tab1, tab3
        else:
            return tab1
        
    def menu_view_inputs(self):
                     
        menu_view_inputs_top = Toplevel(root)
    
        self.is_open_menu_view_inputs.set(True)
        self.menu_view_inputs_close_window = lambda:close_window()
        menu_view_inputs_top.protocol("WM_DELETE_WINDOW", self.menu_view_inputs_close_window)
    
        input_length_var = tk.StringVar(value='')   
        input_info_text_var = tk.StringVar(value='')  
        is_widen_box_var = tk.BooleanVar(value=False) 
        is_play_file_var = tk.BooleanVar(value=False) 
        varification_text_var = tk.StringVar(value='Verify Inputs')

        reset_list = lambda:(input_files_listbox_Option.delete(0, 'end'), [input_files_listbox_Option.insert(tk.END, inputs) for inputs in self.inputPaths])
        audio_input_total = lambda:input_length_var.set(f'Audio Input Total: {len(self.inputPaths)}')
        audio_input_total()

        def list_diff(list1, list2): return list(set(list1).symmetric_difference(set(list2)))

        def list_to_string(list1): return '\n'.join(''.join(sub) for sub in list1)

        def close_window():
            self.verification_thread.kill() if self.thread_check(self.verification_thread) else None
            self.is_open_menu_view_inputs.set(False)
            menu_view_inputs_top.destroy()

        def drag_n_drop(e):
            input_info_text_var.set('')
            drop(e, accept_mode='files')
            reset_list()
            audio_input_total()
            
        def selected_files(is_remove=False):
            if not self.thread_check(self.active_processing_thread):
                items_list = [input_files_listbox_Option.get(i) for i in input_files_listbox_Option.curselection()]
                inputPaths = list(self.inputPaths)# if is_remove else items_list
                if is_remove:
                    [inputPaths.remove(i) for i in items_list if items_list]
                else:
                    [inputPaths.remove(i) for i in self.inputPaths if i not in items_list]
                removed_files = list_diff(self.inputPaths, inputPaths)
                [input_files_listbox_Option.delete(input_files_listbox_Option.get(0, tk.END).index(i)) for i in removed_files]
                starting_len = len(self.inputPaths)
                self.inputPaths = tuple(inputPaths)
                self.update_inputPaths()
                audio_input_total()
                input_info_text_var.set(f'{starting_len - len(self.inputPaths)} input(s) removed.')
            else:
                input_info_text_var.set('You cannot remove inputs during an active process.')
            
        def box_size():
            input_info_text_var.set('')
            input_files_listbox_Option.config(width=230, height=25) if is_widen_box_var.get() else input_files_listbox_Option.config(width=110, height=17)
            self.menu_placement(menu_view_inputs_top, 'Selected Inputs', pop_up=True)

        def input_options(is_select_inputs=True):
            input_info_text_var.set('')
            if is_select_inputs:
                self.input_select_filedialog()
            else:
                self.inputPaths = ()
            reset_list()
            self.update_inputPaths()
            audio_input_total()

        def pop_open_file_path(is_play_file=False):
            if self.inputPaths:
                track_selected = self.inputPaths[input_files_listbox_Option.index(tk.ACTIVE)]
                if os.path.isfile(track_selected):
                    subprocess.Popen(["open", track_selected if is_play_file else os.path.dirname(track_selected)]) #MACOS_COMMENT
        
        def get_export_dir():
            if os.path.isdir(self.export_path_var.get()):
                export_dir = self.export_path_var.get()
            else:
                export_dir = self.export_select_filedialog()

            return export_dir
        
        def verify_audio(is_create_samples=False):
            inputPaths = list(self.inputPaths)
            iterated_list = self.inputPaths if not is_create_samples else [input_files_listbox_Option.get(i) for i in input_files_listbox_Option.curselection()]
            removed_files = []
            export_dir = None
            total_audio_count, current_file = len(iterated_list), 0
            if iterated_list:
                for i in iterated_list:
                    current_file += 1
                    input_info_text_var.set(f'{SAMPLE_BEGIN if is_create_samples else VERIFY_BEGIN}{current_file}/{total_audio_count}')
                    if is_create_samples:
                        export_dir = get_export_dir()
                        if not export_dir:
                            input_info_text_var.set(f'No export directory selected.')
                            return
                    is_good, error_data = self.verify_audio(i, is_process=False, sample_path=export_dir)
                    if not is_good:
                        inputPaths.remove(i)
                        removed_files.append(error_data)#sample = self.create_sample(i)
                        
                varification_text_var.set('Verify Inputs')
                input_files_listbox_Option.configure(state=tk.NORMAL)
                
                if removed_files:
                    input_info_text_var.set(f'{len(removed_files)} Broken or Incompatible File(s) Removed. Check Error Log for details.')
                    error_text = ''
                    for i in removed_files:
                        error_text += i
                    removed_files = list_diff(self.inputPaths, inputPaths)
                    [input_files_listbox_Option.delete(input_files_listbox_Option.get(0, tk.END).index(i)) for i in removed_files]
                    self.error_log_var.set(REMOVED_FILES(list_to_string(removed_files), error_text))
                    self.inputPaths = tuple(inputPaths)
                    self.update_inputPaths()
                else:
                    input_info_text_var.set(f'No errors found!')
                    
                audio_input_total()
            else:
                input_info_text_var.set(f'No Files {SELECTED_VER if is_create_samples else DETECTED_VER}')
                varification_text_var.set('Verify Inputs')
                input_files_listbox_Option.configure(state=tk.NORMAL)
                return
            
            #print(list_to_string(self.inputPaths))
            audio_input_total()
            
        def verify_audio_start_thread(is_create_samples=False):
            
            if not self.thread_check(self.active_processing_thread):
                if not self.thread_check(self.verification_thread):
                    varification_text_var.set('Stop Progress')
                    input_files_listbox_Option.configure(state=tk.DISABLED)
                    self.verification_thread = KThread(target=lambda:verify_audio(is_create_samples=is_create_samples))
                    self.verification_thread.start()
                else:
                    input_files_listbox_Option.configure(state=tk.NORMAL)
                    varification_text_var.set('Verify Inputs')
                    input_info_text_var.set('Process Stopped')
                    self.verification_thread.kill()
            else:
                input_info_text_var.set('You cannot verify inputs during an active process.')

        def right_click_menu(event):
                right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
                right_click_menu.add_command(label='Remove Selected Items Only', command=lambda:selected_files(is_remove=True))
                right_click_menu.add_command(label='Keep Selected Items Only', command=lambda:selected_files(is_remove=False))
                right_click_menu.add_command(label='Clear All Input(s)', command=lambda:input_options(is_select_inputs=False))
                right_click_menu.add_separator()
                right_click_menu_sub = Menu(right_click_menu, font=('Century Gothic', FONT_SIZE_1), tearoff=False)
                right_click_menu.add_command(label='Verify and Create Samples of Selected Inputs', command=lambda:verify_audio_start_thread(is_create_samples=True))
                right_click_menu.add_cascade(label='Preferred Double Click Action', menu=right_click_menu_sub)
                if is_play_file_var.get():
                    right_click_menu_sub.add_command(label='Enable: Open Audio File Directory', command=lambda:(input_files_listbox_Option.bind('<Double-Button>', lambda e:pop_open_file_path()), is_play_file_var.set(False)))
                else:
                    right_click_menu_sub.add_command(label='Enable: Open Audio File', command=lambda:(input_files_listbox_Option.bind('<Double-Button>', lambda e:pop_open_file_path(is_play_file=True)), is_play_file_var.set(True)))

                try:
                    right_click_menu.tk_popup(event.x_root,event.y_root)
                finally:
                    right_click_menu.grab_release()

        menu_view_inputs_Frame = self.menu_FRAME_SET(menu_view_inputs_top)
        menu_view_inputs_Frame.grid(row=0,column=0,padx=0,pady=0)  

        self.main_window_LABEL_SET(menu_view_inputs_Frame, 'Selected Inputs').grid(row=0,column=0,padx=0,pady=5)
        tk.Label(menu_view_inputs_Frame, textvariable=input_length_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#13a4c9').grid(row=1, column=0, padx=0, pady=5)
        ttk.Button(menu_view_inputs_Frame, text='Select Input(s)', command=lambda:input_options()).grid(row=2,column=0,padx=0,pady=10)
        
        input_files_listbox_Option = tk.Listbox(menu_view_inputs_Frame, selectmode=tk.EXTENDED, activestyle='dotbox', font=("Century Gothic", f"{FONT_SIZE_1}"), background='#101414', exportselection=0, width=110, height=17, relief=SOLID, borderwidth=0)
        input_files_listbox_vertical_scroll = ttk.Scrollbar(menu_view_inputs_Frame, orient=VERTICAL)
        input_files_listbox_Option.config(yscrollcommand=input_files_listbox_vertical_scroll.set)
        input_files_listbox_vertical_scroll.configure(command=input_files_listbox_Option.yview)
        input_files_listbox_Option.grid(row=4, sticky=W)
        input_files_listbox_vertical_scroll.grid(row=4, column=1, sticky=NS)

        tk.Label(menu_view_inputs_Frame, textvariable=input_info_text_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#13a4c9').grid(row=5, column=0, padx=0, pady=0)
        ttk.Checkbutton(menu_view_inputs_Frame, text='Widen Box', variable=is_widen_box_var, command=lambda:box_size()).grid(row=6,column=0,padx=0,pady=0)
        verify_audio_Button = ttk.Button(menu_view_inputs_Frame, textvariable=varification_text_var, command=lambda:verify_audio_start_thread())
        verify_audio_Button.grid(row=7,column=0,padx=0,pady=5)
        ttk.Button(menu_view_inputs_Frame, text='Close Window', command=lambda:menu_view_inputs_top.destroy()).grid(row=8,column=0,padx=0,pady=5)

        #menu_view_inputs_top.drop_target_register(DND_FILES) MACOS_COMMENT
        # menu_view_inputs_top.dnd_bind('<<Drop>>', lambda e: drag_n_drop(e)) MACOS_COMMENT
        input_files_listbox_Option.bind(right_click_button, lambda e:right_click_menu(e))
        input_files_listbox_Option.bind('<Double-Button>', lambda e:pop_open_file_path())
        input_files_listbox_Option.bind('<Delete>', lambda e:selected_files(is_remove=True))
        input_files_listbox_Option.bind('<BackSpace>', lambda e:selected_files(is_remove=False))

        reset_list()
        
        self.menu_placement(menu_view_inputs_top, 'Selected Inputs', pop_up=True)

    def menu_settings(self, select_tab_2=False, select_tab_3=False):
        """Open Settings and Download Center"""

        settings_menu = Toplevel()
        
        option_var = tk.StringVar(value=SELECT_SAVED_SETTING)
        self.is_menu_settings_open = True
        
        tabControl = ttk.Notebook(settings_menu)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Settings Guide')
        tabControl.add(tab2, text ='Additional Settings')
        tabControl.add(tab3, text ='Download Center')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        
        tab3.grid_rowconfigure(0, weight=1)
        tab3.grid_columnconfigure(0, weight=1)

        self.disable_tabs = lambda:(tabControl.tab(0, state="disabled"), tabControl.tab(1, state="disabled"))
        self.enable_tabs = lambda:(tabControl.tab(0, state="normal"), tabControl.tab(1, state="normal"))        
        self.main_menu_var = tk.StringVar(value='Choose Option') 
        model_sample_mode_duration_label_var = tk.StringVar(value=f'{self.model_sample_mode_duration_var.get()} Seconds') 
        
        self.download_progress_bar_var.set(0)
        self.download_progress_info_var.set('')
        self.download_progress_percent_var.set('')
        
        OPTION_LIST = {
            ENSEMBLE_OPTION:self.check_is_open_menu_advanced_ensemble_options,
            MDX_OPTION:self.check_is_open_menu_advanced_mdx_options,
            DEMUCS_OPTION:self.check_is_open_menu_advanced_demucs_options,
            VR_OPTION:self.check_is_open_menu_advanced_vr_options,
            HELP_OPTION:self.check_is_open_menu_help,
            ERROR_OPTION:self.check_is_open_menu_error_log}
        
        def set_vars_for_sample_mode(event):
            value = int(float(event))
            value = round(value / 5) * 5
            self.model_sample_mode_duration_var.set(value)
            self.model_sample_mode_duration_checkbox_var.set(SAMPLE_MODE_CHECKBOX(value))
            model_sample_mode_duration_label_var.set(f'{value} Seconds')
            
        #Settings Tab 1
        settings_menu_main_Frame = self.menu_FRAME_SET(tab1)
        settings_menu_main_Frame.grid(row=0,column=0,padx=0,pady=0)  
        settings_title_Label = self.menu_title_LABEL_SET(settings_menu_main_Frame, "General Menu")
        settings_title_Label.grid(row=0,column=0,padx=0,pady=15)
        
        select_Label = self.menu_sub_LABEL_SET(settings_menu_main_Frame, 'Additional Menus & Information')
        select_Label.grid(row=1,column=0,padx=0,pady=5)
        
        select_Option = ttk.OptionMenu(settings_menu_main_Frame, self.main_menu_var, None, *ADVANCED_SETTINGS, command=lambda selection:(OPTION_LIST[selection](), close_window()))
        select_Option.grid(row=2,column=0,padx=0,pady=5)
        
        help_hints_Option = ttk.Checkbutton(settings_menu_main_Frame, text='Enable Help Hints', variable=self.help_hints_var, width=13) 
        help_hints_Option.grid(row=3,column=0,padx=0,pady=5)
        
        open_app_dir_Button = ttk.Button(settings_menu_main_Frame, text='Open Application Directory', command=lambda:subprocess.Popen(["open", BASE_PATH])) #MACOS_COMMENT
        open_app_dir_Button.grid(row=6,column=0,padx=0,pady=5)
        
        reset_all_app_settings_Button = ttk.Button(settings_menu_main_Frame, text='Reset All Settings to Default', command=lambda:self.load_to_default_confirm())
        reset_all_app_settings_Button.grid(row=7,column=0,padx=0,pady=5)
        
        # restart_app_Button = ttk.Button(settings_menu_main_Frame, text='Restart Application', command=lambda:self.restart())
        # restart_app_Button.grid(row=8,column=0,padx=0,pady=5)
        
        close_settings_win_Button = ttk.Button(settings_menu_main_Frame, text='Close Window', command=lambda:close_window())
        close_settings_win_Button.grid(row=9,column=0,padx=0,pady=5)

        app_update_Label = self.menu_title_LABEL_SET(settings_menu_main_Frame, "Application Updates")
        app_update_Label.grid(row=10,column=0,padx=0,pady=15)
        
        self.app_update_button = ttk.Button(settings_menu_main_Frame, textvariable=self.app_update_button_Text_var, command=lambda:self.pop_up_update_confirmation())
        self.app_update_button.grid(row=11,column=0,padx=0,pady=5)
        
        self.app_update_status_Label = tk.Label(settings_menu_main_Frame, textvariable=self.app_update_status_Text_var, font=("Century Gothic",  f"{FONT_SIZE_5}"), width=35, justify="center", relief="ridge", fg="#13a4c9")
        self.app_update_status_Label.grid(row=12,column=0,padx=0,pady=20)
        
        donate_Button = ttk.Button(settings_menu_main_Frame, image=self.donate_img, command=lambda:webbrowser.open_new_tab(DONATE_LINK_BMAC))
        donate_Button.grid(row=13,column=0,padx=0,pady=5)
        self.help_hints(donate_Button, text=DONATE_HELP)
        
        #Settings Tab 2
        settings_menu_format_Frame = self.menu_FRAME_SET(tab2)
        settings_menu_format_Frame.grid(row=0,column=0,padx=0,pady=0)  
        
        audio_format_title_Label = self.menu_title_LABEL_SET(settings_menu_format_Frame, "Audio Format Settings", width=20)
        audio_format_title_Label.grid(row=0,column=0,padx=0,pady=10)
        
        self.wav_type_set_Label = self.menu_sub_LABEL_SET(settings_menu_format_Frame, 'Wav Type')
        self.wav_type_set_Label.grid(row=1,column=0,padx=0,pady=5)
        
        self.wav_type_set_Option = ttk.OptionMenu(settings_menu_format_Frame, self.wav_type_set_var, None, *WAV_TYPE)
        self.wav_type_set_Option.grid(row=2,column=0,padx=20,pady=5)
        
        self.mp3_bit_set_Label = self.menu_sub_LABEL_SET(settings_menu_format_Frame, 'Mp3 Bitrate')
        self.mp3_bit_set_Label.grid(row=3,column=0,padx=0,pady=5)
        
        self.mp3_bit_set_Option = ttk.OptionMenu(settings_menu_format_Frame, self.mp3_bit_set_var, None, *MP3_BIT_RATES)
        self.mp3_bit_set_Option.grid(row=4,column=0,padx=20,pady=5)
        
        audio_format_title_Label = self.menu_title_LABEL_SET(settings_menu_format_Frame, "General Process Settings")
        audio_format_title_Label.grid(row=5,column=0,padx=0,pady=10)
        
        self.is_testing_audio_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Settings Test Mode', width=GEN_SETTINGS_WIDTH, variable=self.is_testing_audio_var) 
        self.is_testing_audio_Option.grid(row=7,column=0,padx=0,pady=0)
        self.help_hints(self.is_testing_audio_Option, text=IS_TESTING_AUDIO_HELP)
        
        self.is_add_model_name_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Model Test Mode', width=GEN_SETTINGS_WIDTH, variable=self.is_add_model_name_var) 
        self.is_add_model_name_Option.grid(row=8,column=0,padx=0,pady=0)
        self.help_hints(self.is_add_model_name_Option, text=IS_MODEL_TESTING_AUDIO_HELP)
        
        self.is_create_model_folder_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Generate Model Folders', width=GEN_SETTINGS_WIDTH, variable=self.is_create_model_folder_var) 
        self.is_create_model_folder_Option.grid(row=9,column=0,padx=0,pady=0)
        self.help_hints(self.is_create_model_folder_Option, text=IS_CREATE_MODEL_FOLDER_HELP)
        
        self.is_accept_any_input_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Accept Any Input', width=GEN_SETTINGS_WIDTH, variable=self.is_accept_any_input_var) 
        self.is_accept_any_input_Option.grid(row=10,column=0,padx=0,pady=0)
        self.help_hints(self.is_accept_any_input_Option, text=IS_ACCEPT_ANY_INPUT_HELP)
        
        self.is_task_complete_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Notification Chimes', width=GEN_SETTINGS_WIDTH, variable=self.is_task_complete_var) 
        self.is_task_complete_Option.grid(row=11,column=0,padx=0,pady=0)
        self.help_hints(self.is_task_complete_Option, text=IS_TASK_COMPLETE_HELP)
        
        is_normalization_Option = ttk.Checkbutton(settings_menu_format_Frame, text='Normalize Output', width=GEN_SETTINGS_WIDTH, variable=self.is_normalization_var) 
        is_normalization_Option.grid(row=12,column=0,padx=0,pady=0)
        self.help_hints(is_normalization_Option, text=IS_NORMALIZATION_HELP)
        
        model_sample_mode_Label = self.menu_title_LABEL_SET(settings_menu_format_Frame, "Model Sample Mode Settings")
        model_sample_mode_Label.grid(row=13,column=0,padx=0,pady=10)
        
        self.model_sample_mode_duration_Label = self.menu_sub_LABEL_SET(settings_menu_format_Frame, 'Sample Clip Duration')
        self.model_sample_mode_duration_Label.grid(row=14,column=0,padx=0,pady=5)
        
        tk.Label(settings_menu_format_Frame, textvariable=model_sample_mode_duration_label_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#13a4c9').grid(row=15,column=0,padx=0,pady=2)
        model_sample_mode_duration_Option = ttk.Scale(settings_menu_format_Frame, variable=self.model_sample_mode_duration_var, from_=5, to=120, command=set_vars_for_sample_mode, orient='horizontal')
        model_sample_mode_duration_Option.grid(row=16,column=0,padx=0,pady=2)
        
        delete_your_settings_Label = self.menu_title_LABEL_SET(settings_menu_format_Frame, "Delete User Saved Setting")
        delete_your_settings_Label.grid(row=17,column=0,padx=0,pady=10)
        self.help_hints(delete_your_settings_Label, text=DELETE_YOUR_SETTINGS_HELP)
        
        delete_your_settings_Option = ttk.OptionMenu(settings_menu_format_Frame, option_var)
        delete_your_settings_Option.grid(row=18,column=0,padx=20,pady=5)
        self.deletion_list_fill(delete_your_settings_Option, option_var, self.last_found_settings, SETTINGS_CACHE_DIR, SELECT_SAVED_SETTING)
        
        #Settings Tab 3
        settings_menu_download_center_Frame = self.menu_FRAME_SET(tab3)
        settings_menu_download_center_Frame.grid(row=0,column=0,padx=0,pady=0)  
        
        download_center_title_Label = self.menu_title_LABEL_SET(settings_menu_download_center_Frame, "Application Download Center")
        download_center_title_Label.grid(row=0,column=0,padx=20,pady=10)

        select_download_Label = self.menu_sub_LABEL_SET(settings_menu_download_center_Frame, "Select Download")
        select_download_Label.grid(row=1,column=0,padx=0,pady=10)
        
        self.model_download_vr_Button = ttk.Radiobutton(settings_menu_download_center_Frame, text='VR Arch', width=8, variable=self.select_download_var, value='VR Arc', command=lambda:self.download_list_state())
        self.model_download_vr_Button.grid(row=3,column=0,padx=0,pady=5)
        self.model_download_vr_Option = ttk.OptionMenu(settings_menu_download_center_Frame, self.model_download_vr_var)
        self.model_download_vr_Option.grid(row=4,column=0,padx=0,pady=5)
        
        self.model_download_mdx_Button = ttk.Radiobutton(settings_menu_download_center_Frame, text='MDX-Net', width=8, variable=self.select_download_var, value='MDX-Net', command=lambda:self.download_list_state())
        self.model_download_mdx_Button.grid(row=5,column=0,padx=0,pady=5)
        self.model_download_mdx_Option = ttk.OptionMenu(settings_menu_download_center_Frame, self.model_download_mdx_var)
        self.model_download_mdx_Option.grid(row=6,column=0,padx=0,pady=5)

        self.model_download_demucs_Button = ttk.Radiobutton(settings_menu_download_center_Frame, text='Demucs', width=8, variable=self.select_download_var, value='Demucs', command=lambda:self.download_list_state())
        self.model_download_demucs_Button.grid(row=7,column=0,padx=0,pady=5)
        self.model_download_demucs_Option = ttk.OptionMenu(settings_menu_download_center_Frame, self.model_download_demucs_var)
        self.model_download_demucs_Option.grid(row=8,column=0,padx=0,pady=5)
        
        self.download_Button = ttk.Button(settings_menu_download_center_Frame, image=self.download_img, command=lambda:self.download_item())#, command=download_model)
        self.download_Button.grid(row=9,column=0,padx=0,pady=5)
        
        self.download_progress_info_Label = tk.Label(settings_menu_download_center_Frame, textvariable=self.download_progress_info_var, font=("Century Gothic", f"{FONT_SIZE_2}"), foreground='#13a4c9', borderwidth=0)
        self.download_progress_info_Label.grid(row=10,column=0,padx=0,pady=5)
        
        self.download_progress_percent_Label = tk.Label(settings_menu_download_center_Frame, textvariable=self.download_progress_percent_var, font=("Century Gothic", f"{FONT_SIZE_2}"), wraplength=350, foreground='#13a4c9')
        self.download_progress_percent_Label.grid(row=11,column=0,padx=0,pady=5)
        
        self.download_progress_bar_Progressbar = ttk.Progressbar(settings_menu_download_center_Frame, variable=self.download_progress_bar_var)
        self.download_progress_bar_Progressbar.grid(row=12,column=0,padx=0,pady=5)
        
        self.stop_download_Button = ttk.Button(settings_menu_download_center_Frame, textvariable=self.download_stop_var, width=15, command=lambda:self.download_post_action(DOWNLOAD_STOPPED))
        self.stop_download_Button.grid(row=13,column=0,padx=0,pady=5)
        self.stop_download_Button_DISABLE = lambda:(self.download_stop_var.set(""), self.stop_download_Button.configure(state=tk.DISABLED))
        self.stop_download_Button_ENABLE = lambda:(self.download_stop_var.set("Stop Download"), self.stop_download_Button.configure(state=tk.NORMAL))

        self.refresh_list_Button = ttk.Button(settings_menu_download_center_Frame, text='Refresh List', command=lambda:(self.online_data_refresh(refresh_list_Button=True), self.download_list_state()))#, command=refresh_list)
        self.refresh_list_Button.grid(row=14,column=0,padx=0,pady=5)
        
        self.download_key_Button = ttk.Button(settings_menu_download_center_Frame, image=self.key_img, command=lambda:self.pop_up_user_code_input())
        self.download_key_Button.grid(row=15,column=0,padx=0,pady=5)
                            
        self.download_center_Buttons = (self.model_download_vr_Button,
                                        self.model_download_mdx_Button,
                                        self.model_download_demucs_Button,
                                        self.download_Button,
                                        self.download_key_Button)
        
        self.download_lists = (self.model_download_vr_Option,
                               self.model_download_mdx_Option,
                               self.model_download_demucs_Option)
        
        self.download_list_vars = (self.model_download_vr_var,
                              self.model_download_mdx_var,
                              self.model_download_demucs_var)
        
        self.online_data_refresh()
        self.download_list_state()
        
        if self.is_online:
            self.download_list_fill()

        self.menu_placement(settings_menu, "Settings Guide", is_help_hints=True, close_function=lambda:close_window())

        if select_tab_2:
            tabControl.select(tab2)
            
        if select_tab_3:
            tabControl.select(tab3)

        def close_window():
            self.active_download_thread.terminate() if self.thread_check(self.active_download_thread) else None
            self.is_menu_settings_open = False
            settings_menu.destroy()

        settings_menu.protocol("WM_DELETE_WINDOW", close_window)

    def menu_advanced_vr_options(self):
        """Open Advanced VR Options"""     

        vr_opt = Toplevel()
        
        tab1 = self.menu_tab_control(vr_opt, self.vr_secondary_model_vars)

        self.is_open_menu_advanced_vr_options.set(True)
        self.menu_advanced_vr_options_close_window = lambda:(self.is_open_menu_advanced_vr_options.set(False), vr_opt.destroy())
        vr_opt.protocol("WM_DELETE_WINDOW", self.menu_advanced_vr_options_close_window)
        
        vr_opt_frame = self.menu_FRAME_SET(tab1)
        vr_opt_frame.grid(row=0,column=0,padx=0,pady=0)  
        
        vr_title = self.menu_title_LABEL_SET(vr_opt_frame, "Advanced VR Options")
        vr_title.grid(row=0,column=0,padx=0,pady=10)
  
        if not self.chosen_process_method_var.get() == VR_ARCH_PM:
            window_size_Label = self.menu_sub_LABEL_SET(vr_opt_frame, 'Window Size')
            window_size_Label.grid(row=1,column=0,padx=0,pady=5)
            window_size_Option = ttk.Combobox(vr_opt_frame, value=VR_WINDOW, width=MENU_COMBOBOX_WIDTH, textvariable=self.window_size_var)
            window_size_Option.grid(row=2,column=0,padx=0,pady=5)
            self.combobox_entry_validation(window_size_Option, self.window_size_var, REG_WINDOW, VR_WINDOW)
            self.help_hints(window_size_Label, text=WINDOW_SIZE_HELP)
            
            aggression_setting_Label = self.menu_sub_LABEL_SET(vr_opt_frame, 'Aggression Setting')
            aggression_setting_Label.grid(row=3,column=0,padx=0,pady=5)
            aggression_setting_Option = ttk.Combobox(vr_opt_frame, value=VR_BATCH, width=MENU_COMBOBOX_WIDTH, textvariable=self.aggression_setting_var)
            aggression_setting_Option.grid(row=4,column=0,padx=0,pady=5)
            self.combobox_entry_validation(aggression_setting_Option, self.aggression_setting_var, REG_WINDOW, VR_BATCH)
            self.help_hints(aggression_setting_Label, text=AGGRESSION_SETTING_HELP)
        
        self.crop_size_Label = self.menu_sub_LABEL_SET(vr_opt_frame, 'Crop Size')
        self.crop_size_Label.grid(row=5,column=0,padx=0,pady=5)
        self.crop_size_sub_Label = self.menu_sub_LABEL_SET(vr_opt_frame, '(Works with select models only)', font_size=FONT_SIZE_1)
        self.crop_size_sub_Label.grid(row=6,column=0,padx=0,pady=0)
        self.crop_size_Option = ttk.Combobox(vr_opt_frame, value=VR_CROP, width=MENU_COMBOBOX_WIDTH, textvariable=self.crop_size_var)
        self.crop_size_Option.grid(row=7,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.crop_size_Option, self.crop_size_var, REG_WINDOW, VR_CROP)
        self.help_hints(self.crop_size_Label, text=CROP_SIZE_HELP)
        
        self.batch_size_Label = self.menu_sub_LABEL_SET(vr_opt_frame, 'Batch Size')
        self.batch_size_Label.grid(row=8,column=0,padx=0,pady=5)
        self.batch_size_sub_Label = self.menu_sub_LABEL_SET(vr_opt_frame, '(Works with select models only)', font_size=FONT_SIZE_1)
        self.batch_size_sub_Label.grid(row=9,column=0,padx=0,pady=0)
        self.batch_size_Option = ttk.Combobox(vr_opt_frame, value=VR_BATCH, width=MENU_COMBOBOX_WIDTH, textvariable=self.batch_size_var)
        self.batch_size_Option.grid(row=10,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.batch_size_Option, self.batch_size_var, REG_WINDOW, VR_BATCH)
        self.help_hints(self.batch_size_Label, text=BATCH_SIZE_HELP)
        
        self.is_tta_Option = ttk.Checkbutton(vr_opt_frame, text='Enable TTA', width=16, variable=self.is_tta_var) 
        self.is_tta_Option.grid(row=11,column=0,padx=0,pady=0)
        self.help_hints(self.is_tta_Option, text=IS_TTA_HELP)
        
        self.is_post_process_Option = ttk.Checkbutton(vr_opt_frame, text='Post-Process', width=16, variable=self.is_post_process_var) 
        self.is_post_process_Option.grid(row=12,column=0,padx=0,pady=0)
        self.help_hints(self.is_post_process_Option, text=IS_POST_PROCESS_HELP)
        
        self.is_high_end_process_Option = ttk.Checkbutton(vr_opt_frame, text='High-End Process', width=16, variable=self.is_high_end_process_var) 
        self.is_high_end_process_Option.grid(row=13,column=0,padx=0,pady=0)
        self.help_hints(self.is_high_end_process_Option, text=IS_HIGH_END_PROCESS_HELP)
        
        self.vr_clear_cache_Button = ttk.Button(vr_opt_frame, text='Clear Auto-Set Cache', command=lambda:self.clear_cache(VR_ARCH_TYPE))
        self.vr_clear_cache_Button.grid(row=14,column=0,padx=0,pady=5)
        self.help_hints(self.vr_clear_cache_Button, text=CLEAR_CACHE_HELP)
        
        self.open_vr_model_dir_Button = ttk.Button(vr_opt_frame, text='Open VR Models Folder', command=lambda:subprocess.Popen(["open", VR_MODELS_DIR])) #MACOS_COMMENT
        self.open_vr_model_dir_Button.grid(row=15,column=0,padx=0,pady=5)
        
        self.vr_return_Button=ttk.Button(vr_opt_frame, text=BACK_TO_MAIN_MENU, command=lambda:(self.menu_advanced_vr_options_close_window(), self.check_is_menu_settings_open()))
        self.vr_return_Button.grid(row=16,column=0,padx=0,pady=5)
        
        self.vr_close_Button = ttk.Button(vr_opt_frame, text='Close Window', command=lambda:self.menu_advanced_vr_options_close_window())
        self.vr_close_Button.grid(row=17,column=0,padx=0,pady=5)
        
        self.menu_placement(vr_opt, "Advanced VR Options", is_help_hints=True, close_function=self.menu_advanced_vr_options_close_window)

    def menu_advanced_demucs_options(self):
        """Open Advanced Demucs Options"""
        
        demuc_opt = Toplevel()

        self.is_open_menu_advanced_demucs_options.set(True)
        self.menu_advanced_demucs_options_close_window = lambda:(self.is_open_menu_advanced_demucs_options.set(False), demuc_opt.destroy())
        demuc_opt.protocol("WM_DELETE_WINDOW", self.menu_advanced_demucs_options_close_window)
        pre_proc_list = self.model_list(VOCAL_STEM, INST_STEM, is_dry_check=True, is_no_demucs=True)

        tab1, tab3 = self.menu_tab_control(demuc_opt, self.demucs_secondary_model_vars, is_demucs=True)
        
        demucs_frame = self.menu_FRAME_SET(tab1)
        demucs_frame.grid(row=0,column=0,padx=0,pady=0)  
        
        demucs_pre_model_frame = self.menu_FRAME_SET(tab3)
        demucs_pre_model_frame.grid(row=0,column=0,padx=0,pady=0)  
        
        demucs_title_Label = self.menu_title_LABEL_SET(demucs_frame, "Advanced Demucs Options")
        demucs_title_Label.grid(row=0,column=0,padx=0,pady=10)
        
        enable_chunks = lambda:(self.margin_demucs_Option.configure(state=tk.NORMAL), self.chunks_demucs_Option.configure(state=tk.NORMAL))
        disable_chunks = lambda:(self.margin_demucs_Option.configure(state=tk.DISABLED), self.chunks_demucs_Option.configure(state=tk.DISABLED))
        chunks_toggle = lambda:enable_chunks() if self.is_chunk_demucs_var.get() else disable_chunks()
        enable_pre_proc_model = lambda:(is_demucs_pre_proc_model_inst_mix_Option.configure(state=tk.NORMAL), demucs_pre_proc_model_Option.configure(state=tk.NORMAL))
        disable_pre_proc_model = lambda:(is_demucs_pre_proc_model_inst_mix_Option.configure(state=tk.DISABLED), demucs_pre_proc_model_Option.configure(state=tk.DISABLED), self.is_demucs_pre_proc_model_inst_mix_var.set(False))
        pre_proc_model_toggle = lambda:enable_pre_proc_model() if self.is_demucs_pre_proc_model_activate_var.get() else disable_pre_proc_model()
        
        if not self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
            segment_Label = self.menu_sub_LABEL_SET(demucs_frame, 'Segments')
            segment_Label.grid(row=1,column=0,padx=0,pady=10)
            segment_Option = ttk.Combobox(demucs_frame, value=DEMUCS_SEGMENTS, width=MENU_COMBOBOX_WIDTH, textvariable=self.segment_var)
            segment_Option.grid(row=2,column=0,padx=0,pady=0)
            self.combobox_entry_validation(segment_Option, self.segment_var, REG_SEGMENTS, DEMUCS_SEGMENTS)
            self.help_hints(segment_Label, text=SEGMENT_HELP)
        
        self.shifts_Label = self.menu_sub_LABEL_SET(demucs_frame, 'Shifts')
        self.shifts_Label.grid(row=3,column=0,padx=0,pady=5)
        self.shifts_Option = ttk.Combobox(demucs_frame, value=DEMUCS_SHIFTS, width=MENU_COMBOBOX_WIDTH, textvariable=self.shifts_var)
        self.shifts_Option.grid(row=4,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.shifts_Option, self.shifts_var, REG_SHIFTS, DEMUCS_SHIFTS)
        self.help_hints(self.shifts_Label, text=SHIFTS_HELP)

        self.overlap_Label = self.menu_sub_LABEL_SET(demucs_frame, 'Overlap')
        self.overlap_Label.grid(row=5,column=0,padx=0,pady=5)
        self.overlap_Option = ttk.Combobox(demucs_frame, value=DEMUCS_OVERLAP, width=MENU_COMBOBOX_WIDTH, textvariable=self.overlap_var)
        self.overlap_Option.grid(row=6,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.overlap_Option, self.overlap_var, REG_OVERLAP, DEMUCS_OVERLAP)
        self.help_hints(self.overlap_Label, text=OVERLAP_HELP)

        self.chunks_demucs_Label = self.menu_sub_LABEL_SET(demucs_frame, 'Chunks')
        self.chunks_demucs_Label.grid(row=7,column=0,padx=0,pady=5)
        self.chunks_demucs_Option = ttk.Combobox(demucs_frame, value=CHUNKS, width=MENU_COMBOBOX_WIDTH, textvariable=self.chunks_demucs_var)
        self.chunks_demucs_Option.grid(row=8,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.chunks_demucs_Option, self.chunks_demucs_var, REG_CHUNKS, CHUNKS)
        self.help_hints(self.chunks_demucs_Label, text=CHUNKS_HELP)
        
        self.margin_demucs_Label = self.menu_sub_LABEL_SET(demucs_frame, 'Chunk Margin')
        self.margin_demucs_Label.grid(row=9,column=0,padx=0,pady=5)
        self.margin_demucs_Option = ttk.Combobox(demucs_frame, value=MARGIN_SIZE, width=MENU_COMBOBOX_WIDTH, textvariable=self.margin_demucs_var)
        self.margin_demucs_Option.grid(row=10,column=0,padx=0,pady=5)
        self.combobox_entry_validation(self.margin_Option, self.margin_demucs_var, REG_MARGIN, MARGIN_SIZE)
        self.help_hints(self.margin_demucs_Label, text=MARGIN_HELP)
        
        self.is_chunk_demucs_Option = ttk.Checkbutton(demucs_frame, text='Enable Chunks', width=18, variable=self.is_chunk_demucs_var, command=chunks_toggle) 
        self.is_chunk_demucs_Option.grid(row=11,column=0,padx=0,pady=0)
        self.help_hints(self.is_chunk_demucs_Option, text=IS_CHUNK_DEMUCS_HELP)
        
        self.is_split_mode_Option = ttk.Checkbutton(demucs_frame, text='Split Mode', width=18, variable=self.is_split_mode_var) 
        self.is_split_mode_Option.grid(row=12,column=0,padx=0,pady=0)
        self.help_hints(self.is_split_mode_Option, text=IS_SPLIT_MODE_HELP)
        
        self.is_demucs_combine_stems_Option = ttk.Checkbutton(demucs_frame, text='Combine Stems', width=18, variable=self.is_demucs_combine_stems_var) 
        self.is_demucs_combine_stems_Option.grid(row=13,column=0,padx=0,pady=0)
        self.help_hints(self.is_demucs_combine_stems_Option, text=IS_DEMUCS_COMBINE_STEMS_HELP)
        
        is_invert_spec_Option = ttk.Checkbutton(demucs_frame, text='Spectral Inversion', width=18, variable=self.is_invert_spec_var) 
        is_invert_spec_Option.grid(row=14,column=0,padx=0,pady=0)
        self.help_hints(is_invert_spec_Option, text=IS_INVERT_SPEC_HELP)
        
        self.open_demucs_model_dir_Button = ttk.Button(demucs_frame, text='Open Demucs Model Folder', command=lambda:subprocess.Popen(["open", DEMUCS_MODELS_DIR])) #MACOS_COMMENT
        self.open_demucs_model_dir_Button.grid(row=15,column=0,padx=0,pady=5)
        
        self.demucs_return_Button = ttk.Button(demucs_frame, text=BACK_TO_MAIN_MENU, command=lambda:(self.menu_advanced_demucs_options_close_window(), self.check_is_menu_settings_open()))
        self.demucs_return_Button.grid(row=16,column=0,padx=0,pady=5)
        
        self.demucs_close_Button = ttk.Button(demucs_frame, text='Close Window', command=lambda:self.menu_advanced_demucs_options_close_window())
        self.demucs_close_Button.grid(row=17,column=0,padx=0,pady=5)
        
        demucs_pre_proc_model_title_Label = self.menu_title_LABEL_SET(demucs_pre_model_frame, "Pre-process Model")
        demucs_pre_proc_model_title_Label.grid(row=0,column=0,padx=0,pady=15)
        
        demucs_pre_proc_model_Label = self.menu_sub_LABEL_SET(demucs_pre_model_frame, 'Select Model', font_size=FONT_SIZE_3)
        demucs_pre_proc_model_Label.grid(row=1,column=0,padx=0,pady=0)
        demucs_pre_proc_model_Option = ttk.OptionMenu(demucs_pre_model_frame, self.demucs_pre_proc_model_var, None, NO_MODEL, *pre_proc_list)
        demucs_pre_proc_model_Option.configure(width=33)
        demucs_pre_proc_model_Option.grid(row=2,column=0,padx=0,pady=10)
        
        is_demucs_pre_proc_model_inst_mix_Option = ttk.Checkbutton(demucs_pre_model_frame, text='Save Instrumental Mixture', width=27, variable=self.is_demucs_pre_proc_model_inst_mix_var) 
        is_demucs_pre_proc_model_inst_mix_Option.grid(row=3,column=0,padx=0,pady=0)
        self.help_hints(is_demucs_pre_proc_model_inst_mix_Option, text=PRE_PROC_MODEL_INST_MIX_HELP)
        
        is_demucs_pre_proc_model_activate_Option = ttk.Checkbutton(demucs_pre_model_frame, text='Activate Pre-process Model', width=27, variable=self.is_demucs_pre_proc_model_activate_var, command=pre_proc_model_toggle) 
        is_demucs_pre_proc_model_activate_Option.grid(row=4,column=0,padx=0,pady=0)
        self.help_hints(is_demucs_pre_proc_model_activate_Option, text=PRE_PROC_MODEL_ACTIVATE_HELP)
                
        chunks_toggle()
        pre_proc_model_toggle()
        
        self.menu_placement(demuc_opt, "Advanced Demucs Options", is_help_hints=True, close_function=self.menu_advanced_demucs_options_close_window)
        
    def menu_advanced_mdx_options(self):
        """Open Advanced MDX Options"""

        mdx_net_opt = Toplevel()

        self.is_open_menu_advanced_mdx_options.set(True)
        self.menu_advanced_mdx_options_close_window = lambda:(self.is_open_menu_advanced_mdx_options.set(False), mdx_net_opt.destroy())
        mdx_net_opt.protocol("WM_DELETE_WINDOW", self.menu_advanced_mdx_options_close_window)

        tab1 = self.menu_tab_control(mdx_net_opt, self.mdx_secondary_model_vars)
        
        mdx_net_frame = self.menu_FRAME_SET(tab1)
        mdx_net_frame.grid(row=0,column=0,padx=0,pady=0)  

        mdx_opt_title = self.menu_title_LABEL_SET(mdx_net_frame, "Advanced MDX-Net Options")
        mdx_opt_title.grid(row=0,column=0,padx=0,pady=10)
        
        if not self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
            chunks_Label = self.menu_sub_LABEL_SET(mdx_net_frame, 'Chunks')
            chunks_Label.grid(row=1,column=0,padx=0,pady=5)
            chunks_Option = ttk.Combobox(mdx_net_frame, value=CHUNKS, width=MENU_COMBOBOX_WIDTH, textvariable=self.chunks_var)
            chunks_Option.grid(row=2,column=0,padx=0,pady=5)
            self.combobox_entry_validation(chunks_Option, self.chunks_var, REG_CHUNKS, CHUNKS)
            self.help_hints(chunks_Label, text=CHUNKS_HELP)
        
            margin_Label = self.menu_sub_LABEL_SET(mdx_net_frame, 'Chunk Margin')
            margin_Label.grid(row=3,column=0,padx=0,pady=5)
            margin_Option = ttk.Combobox(mdx_net_frame, value=MARGIN_SIZE, width=MENU_COMBOBOX_WIDTH, textvariable=self.margin_var)
            margin_Option.grid(row=4,column=0,padx=0,pady=5)
            self.combobox_entry_validation(margin_Option, self.margin_var, REG_MARGIN, MARGIN_SIZE)
            self.help_hints(margin_Label, text=MARGIN_HELP)
        
        compensate_Label = self.menu_sub_LABEL_SET(mdx_net_frame, 'Volume Compensation')
        compensate_Label.grid(row=5,column=0,padx=0,pady=5)
        compensate_Option = ttk.Combobox(mdx_net_frame, value=VOL_COMPENSATION, width=MENU_COMBOBOX_WIDTH, textvariable=self.compensate_var)
        compensate_Option.grid(row=6,column=0,padx=0,pady=5)
        self.combobox_entry_validation(compensate_Option, self.compensate_var, REG_COMPENSATION, VOL_COMPENSATION)
        self.help_hints(compensate_Label, text=COMPENSATE_HELP)
        
        is_denoise_Option = ttk.Checkbutton(mdx_net_frame, text='Denoise Output', width=14, variable=self.is_denoise_var) 
        is_denoise_Option.grid(row=8,column=0,padx=0,pady=0)
        self.help_hints(is_denoise_Option, text=IS_DENOISE_HELP)

        is_invert_spec_Option = ttk.Checkbutton(mdx_net_frame, text='Spectral Inversion', width=14, variable=self.is_invert_spec_var) 
        is_invert_spec_Option.grid(row=9,column=0,padx=0,pady=0)
        self.help_hints(is_invert_spec_Option, text=IS_INVERT_SPEC_HELP)

        clear_mdx_cache_Button = ttk.Button(mdx_net_frame, text='Clear Auto-Set Cache', command=lambda:self.clear_cache(MDX_ARCH_TYPE))
        clear_mdx_cache_Button.grid(row=10,column=0,padx=0,pady=5)
        self.help_hints(clear_mdx_cache_Button, text=CLEAR_CACHE_HELP)
        
        open_mdx_model_dir_Button = ttk.Button(mdx_net_frame, text='Open MDX-Net Models Folder', command=lambda:subprocess.Popen(["open", MDX_MODELS_DIR])) #MACOS_COMMENT
        open_mdx_model_dir_Button.grid(row=11,column=0,padx=0,pady=5)
        
        mdx_return_Button = ttk.Button(mdx_net_frame, text=BACK_TO_MAIN_MENU, command=lambda:(self.menu_advanced_mdx_options_close_window(), self.check_is_menu_settings_open()))
        mdx_return_Button.grid(row=12,column=0,padx=0,pady=5)

        mdx_close_Button = ttk.Button(mdx_net_frame, text='Close Window', command=lambda:self.menu_advanced_mdx_options_close_window())
        mdx_close_Button.grid(row=13,column=0,padx=0,pady=5)
        
        self.menu_placement(mdx_net_opt, "Advanced MDX-Net Options", is_help_hints=True, close_function=self.menu_advanced_mdx_options_close_window)

    def menu_advanced_ensemble_options(self):
        """Open Ensemble Custom"""
        
        custom_ens_opt = Toplevel()
        
        self.is_open_menu_advanced_ensemble_options.set(True)
        self.menu_advanced_ensemble_options_close_window = lambda:(self.is_open_menu_advanced_ensemble_options.set(False), custom_ens_opt.destroy())
        custom_ens_opt.protocol("WM_DELETE_WINDOW", self.menu_advanced_ensemble_options_close_window)

        option_var = tk.StringVar(value=SELECT_SAVED_ENSEMBLE)

        custom_ens_opt_frame = self.menu_FRAME_SET(custom_ens_opt)
        custom_ens_opt_frame.grid(row=0,column=0,padx=0,pady=0)  
        
        settings_title_Label = self.menu_title_LABEL_SET(custom_ens_opt_frame, "Advanced Option Menu")
        settings_title_Label.grid(row=1,column=0,padx=0,pady=10)
        
        delete_entry_Label = self.menu_sub_LABEL_SET(custom_ens_opt_frame, 'Remove Saved Ensemble')
        delete_entry_Label.grid(row=2,column=0,padx=0,pady=5)
        delete_entry_Option = ttk.OptionMenu(custom_ens_opt_frame, option_var)
        delete_entry_Option.grid(row=3,column=0,padx=20,pady=5)
        
        is_save_all_outputs_ensemble_Option = ttk.Checkbutton(custom_ens_opt_frame, text='Save All Outputs', width=25, variable=self.is_save_all_outputs_ensemble_var)
        is_save_all_outputs_ensemble_Option.grid(row=4,column=0,padx=0,pady=0)
        self.help_hints(is_save_all_outputs_ensemble_Option, text=IS_SAVE_ALL_OUTPUTS_ENSEMBLE_HELP)

        is_append_ensemble_name_Option = ttk.Checkbutton(custom_ens_opt_frame, text='Append Ensemble Name', width=25, variable=self.is_append_ensemble_name_var) 
        is_append_ensemble_name_Option.grid(row=5,column=0,padx=0,pady=0)
        self.help_hints(is_append_ensemble_name_Option, text=IS_APPEND_ENSEMBLE_NAME_HELP)

        ensemble_return_Button = ttk.Button(custom_ens_opt_frame, text="Back to Main Menu", command=lambda:(self.menu_advanced_ensemble_options_close_window(), self.check_is_menu_settings_open()))
        ensemble_return_Button.grid(row=10,column=0,padx=0,pady=5)
        
        ensemble_close_Button = ttk.Button(custom_ens_opt_frame, text='Close Window', command=lambda:self.menu_advanced_ensemble_options_close_window())
        ensemble_close_Button.grid(row=11,column=0,padx=0,pady=5)
        
        self.deletion_list_fill(delete_entry_Option, option_var, self.last_found_ensembles, ENSEMBLE_CACHE_DIR, SELECT_SAVED_ENSEMBLE)
        
        self.menu_placement(custom_ens_opt, "Advanced Ensemble Options", is_help_hints=True, close_function=self.menu_advanced_ensemble_options_close_window)
        
    def menu_help(self):
        """Open Help Guide"""
        
        help_guide_opt = Toplevel()

        self.is_open_menu_help.set(True)
        self.menu_help_close_window = lambda:(self.is_open_menu_help.set(False), help_guide_opt.destroy())
        help_guide_opt.protocol("WM_DELETE_WINDOW", self.menu_help_close_window)
        
        tabControl = ttk.Notebook(help_guide_opt)

        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Credits')
        tabControl.add(tab2, text ='Resources')
        tabControl.add(tab3, text ='Application License & Version Information')
        tabControl.add(tab4, text ='Application Change Log')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        
        tab3.grid_rowconfigure(0, weight=1)
        tab3.grid_columnconfigure(0, weight=1)
        
        tab4.grid_rowconfigure(0, weight=1)
        tab4.grid_columnconfigure(0, weight=1)
        
        section_title_Label = lambda place, frame, text, font_size=FONT_SIZE_4: tk.Label(master=frame, text=text,font=("Century Gothic", f"{font_size}", "bold"), justify="center", fg="#F4F4F4").grid(row=place,column=0,padx=0,pady=3)
        description_Label = lambda place, frame, text, font=FONT_SIZE_2: tk.Label(master=frame, text=text, font=("Century Gothic", f"{font}"), justify="center", fg="#F6F6F7").grid(row=place,column=0,padx=0,pady=3)

        def credit_label(place, frame, text, link=None, message=None, is_link=False, is_top=False):
            if is_top:
                thank = tk.Label(master=frame, text=text, font=("Century Gothic", f"{FONT_SIZE_3}", "bold"), justify="center", fg="#13a4c9")
            else:
                thank = tk.Label(master=frame, text=text, font=("Century Gothic", f"{FONT_SIZE_3}", "underline" if is_link else "normal"), justify="center", fg="#13a4c9")
            thank.configure(cursor="hand2") if is_link else None
            thank.grid(row=place,column=0,padx=0,pady=1)
            if link:
                thank.bind("<Button-1>", lambda e:webbrowser.open_new_tab(link))
            if message:
                description_Label(place+1, frame, message)
        
        def Link(place, frame, text, link, description, font=FONT_SIZE_2): 
            link_label = tk.Label(master=frame, text=text, font=("Century Gothic", f"{FONT_SIZE_4}", "underline"), foreground='#15bfeb', justify="center", cursor="hand2")
            link_label.grid(row=place,column=0,padx=0,pady=5)
            link_label.bind("<Button-1>", lambda e:webbrowser.open_new_tab(link))
            description_Label(place+1, frame, description, font=font)

        def right_click_menu(event):
                right_click_menu = Menu(self, font=('Century Gothic', FONT_SIZE_1), tearoff=0)
                right_click_menu.add_command(label='Return to Settings Menu', command=lambda:(self.menu_help_close_window(), self.check_is_menu_settings_open()))
                right_click_menu.add_command(label='Exit Window', command=lambda:self.menu_help_close_window())
                
                try:
                    right_click_menu.tk_popup(event.x_root,event.y_root)
                finally:
                    right_click_menu.grab_release()

        help_guide_opt.bind(right_click_button, lambda e:right_click_menu(e))
        credits_Frame = Frame(tab1, highlightthicknes=50)
        credits_Frame.grid(row=0, column=0, padx=0, pady=0)
        tk.Label(credits_Frame, image=self.credits_img).grid(row=1,column=0,padx=0,pady=5)

        section_title_Label(place=0,
                            frame=credits_Frame,
                            text="Core UVR Developers")
        
        credit_label(place=2,
                     frame=credits_Frame,
                     text="Anjok07\nAufr33",
                     is_top=True)
        
        section_title_Label(place=3,
                            frame=credits_Frame,
                            text="Special Thanks")
        
        credit_label(place=6,
                     frame=credits_Frame,
                     text="Tsurumeso",
                     message="Developed the original VR Architecture AI code.",
                     link="https://github.com/tsurumeso/vocal-remover",
                     is_link=True)
        
        credit_label(place=8,
                     frame=credits_Frame,
                     text="Kuielab & Woosung Choi",
                     message="Developed the original MDX-Net AI code.",
                     link="https://github.com/kuielab",
                     is_link=True)
        
        credit_label(place=10,
                     frame=credits_Frame,
                     text="Adefossez & Demucs",
                     message="Core developer of Facebook's Demucs Music Source Separation.",
                     link="https://github.com/facebookresearch/demucs",
                     is_link=True)
        
        credit_label(place=12,
                     frame=credits_Frame,
                     text="Bas Curtiz",
                     message="Designed the official UVR logo, icon, banner, splash screen.")
        
        credit_label(place=14,
                     frame=credits_Frame,
                     text="DilanBoskan",
                     message="Your contributions at the start of this project were essential to the success of UVR. Thank you!")
        
        credit_label(place=16,
                     frame=credits_Frame,
                     text="Audio Separation and CC Karokee & Friends Discord Communities",
                     message="Thank you for the support!")

        more_info_tab_Frame = Frame(tab2, highlightthicknes=30)
        more_info_tab_Frame.grid(row=0,column=0,padx=0,pady=0)

        section_title_Label(place=3, 
                            frame=more_info_tab_Frame, 
                            text="Resources")

        Link(place=4, 
             frame=more_info_tab_Frame, 
             text="Ultimate Vocal Remover (Official GitHub)", 
             link="https://github.com/Anjok07/ultimatevocalremovergui", 
             description="You can find updates, report issues, and give us a shout via our official GitHub.",
             font=FONT_SIZE_3)
        
        Link(place=8, 
             frame=more_info_tab_Frame, 
             text="X-Minus AI", 
             link="https://x-minus.pro/ai", 
             description="Many of the models provided are also on X-Minus.\n" + \
                         "X-Minus benefits users without the computing resources to run the GUI or models locally.",
             font=FONT_SIZE_3)
        
        Link(place=12, 
             frame=more_info_tab_Frame, 
             text="FFmpeg", 
             link="https://www.wikihow.com/Install-FFmpeg-on-Windows", 
             description="UVR relies on FFmpeg for processing non-wav audio files.\n" + \
                         "If you are missing FFmpeg, please see the installation guide via the link provided.",
             font=FONT_SIZE_3)
        
        Link(place=18, 
             frame=more_info_tab_Frame, 
             text="Rubber Band Library", 
             link="https://breakfastquay.com/rubberband/",
             description="UVR uses the Rubber Band library for the sound stretch and pitch shift tool.\n" + \
                         "You can get more information on it via the link provided.",
             font=FONT_SIZE_3)
        
        Link(place=22, 
             frame=more_info_tab_Frame, 
             text="Official UVR Patreon", 
             link=DONATE_LINK_PATREON, 
             description="If you wish to support and donate to this project, click the link above and become a Patreon!",
             font=FONT_SIZE_3)
        
        
        appplication_license_tab_Frame = Frame(tab3)
        appplication_license_tab_Frame.grid(row=0,column=0,padx=0,pady=0)
        
        appplication_license_Label = tk.Label(appplication_license_tab_Frame, text='UVR License Information', font=("Century Gothic", f"{FONT_SIZE_6}", "bold"), justify="center", fg="#f4f4f4")
        appplication_license_Label.grid(row=0,column=0,padx=0,pady=25)
        
        appplication_license_Text = tk.Text(appplication_license_tab_Frame, font=("Century Gothic", f"{FONT_SIZE_4}"), fg="white", bg="black", width=80, wrap=WORD, borderwidth=0)
        appplication_license_Text.grid(row=1,column=0,padx=0,pady=0)
        appplication_license_Text_scroll = ttk.Scrollbar(appplication_license_tab_Frame, orient=VERTICAL)
        appplication_license_Text.config(yscrollcommand=appplication_license_Text_scroll.set)
        appplication_license_Text_scroll.configure(command=appplication_license_Text.yview)
        appplication_license_Text.grid(row=4,sticky=W)
        appplication_license_Text_scroll.grid(row=4, column=1, sticky=NS)
        appplication_license_Text.insert("insert", LICENSE_TEXT(VERSION, PATCH))
        appplication_license_Text.configure(state=tk.DISABLED)
        
        application_change_log_tab_Frame = Frame(tab4)
        application_change_log_tab_Frame.grid(row=0,column=0,padx=0,pady=0)
        
        if os.path.isfile(CHANGE_LOG):
            with open(CHANGE_LOG, 'r') as file :
                change_log_text = file.read()
        else:
            change_log_text = 'Change log unavailable.'
        
        application_change_log_Label = tk.Label(application_change_log_tab_Frame, text='UVR Change Log', font=("Century Gothic", f"{FONT_SIZE_6}", "bold"), justify="center", fg="#f4f4f4")
        application_change_log_Label.grid(row=0,column=0,padx=0,pady=25)
        
        application_change_log_Text = tk.Text(application_change_log_tab_Frame, font=("Century Gothic", f"{FONT_SIZE_4}"), fg="white", bg="black", width=80, wrap=WORD, borderwidth=0)
        application_change_log_Text.grid(row=1,column=0,padx=0,pady=0)
        application_change_log_Text_scroll = ttk.Scrollbar(application_change_log_tab_Frame, orient=VERTICAL)
        application_change_log_Text.config(yscrollcommand=application_change_log_Text_scroll.set)
        application_change_log_Text_scroll.configure(command=application_change_log_Text.yview)
        application_change_log_Text.grid(row=4,sticky=W)
        application_change_log_Text_scroll.grid(row=4, column=1, sticky=NS)
        application_change_log_Text.insert("insert", change_log_text)
        application_change_log_Text.configure(state=tk.DISABLED)

        self.menu_placement(help_guide_opt, "Information Guide")

    def menu_error_log(self):
        """Open Error Log"""

        self.is_confirm_error_var.set(False)
        
        copied_var = tk.StringVar(value='')
        error_log_screen = Toplevel()
        
        self.is_open_menu_error_log.set(True)
        self.menu_error_log_close_window = lambda:(self.is_open_menu_error_log.set(False), error_log_screen.destroy())
        error_log_screen.protocol("WM_DELETE_WINDOW", self.menu_error_log_close_window)
        
        error_log_frame = self.menu_FRAME_SET(error_log_screen)
        error_log_frame.grid(row=0,column=0,padx=0,pady=0)  
        
        error_consol_title_Label = self.menu_title_LABEL_SET(error_log_frame, "Error Console")
        error_consol_title_Label.grid(row=1,column=0,padx=20,pady=10)
        
        # error_details_Text = tk.Text(error_log_frame, font=("Century Gothic", f"{FONT_SIZE_1}"), fg="#D37B7B", bg="black", width=110, relief="sunken")
        # error_details_Text.grid(row=4,column=0,padx=0,pady=0)
        # error_details_Text.insert("insert", self.error_log_var.get())
        # error_details_Text.bind(right_click_button, lambda e:self.right_click_menu_popup(e, text_box=True))

        error_details_Text = tk.Text(error_log_frame, font=("Century Gothic", f"{FONT_SIZE_1}"), fg="#D37B7B", bg="black", width=110, wrap=WORD, borderwidth=0)
        error_details_Text.grid(row=4,column=0,padx=0,pady=0)
        error_details_Text.insert("insert", self.error_log_var.get())
        error_details_Text.bind(right_click_button, lambda e:self.right_click_menu_popup(e, text_box=True))
        self.current_text_box = error_details_Text
        error_details_Text_scroll = ttk.Scrollbar(error_log_frame, orient=VERTICAL)
        error_details_Text.config(yscrollcommand=error_details_Text_scroll.set)
        error_details_Text_scroll.configure(command=error_details_Text.yview)
        error_details_Text.grid(row=4,sticky=W)
        error_details_Text_scroll.grid(row=4, column=1, sticky=NS)

        copy_text_Label = tk.Label(error_log_frame, textvariable=copied_var, font=("Century Gothic",  f"{FONT_SIZE_0}"), justify="center", fg="#f4f4f4")
        copy_text_Label.grid(row=5,column=0,padx=20,pady=0)
        
        copy_text_Button = ttk.Button(error_log_frame, text="Copy All Text", command=lambda:(pyperclip.copy(error_details_Text.get(1.0, tk.END+"-1c")), copied_var.set('Copied!')))
        copy_text_Button.grid(row=6,column=0,padx=20,pady=5)
        
        report_issue_Button = ttk.Button(error_log_frame, text="Report Issue", command=lambda:webbrowser.open_new_tab(ISSUE_LINK))
        report_issue_Button.grid(row=7,column=0,padx=20,pady=5)

        error_log_return_Button = ttk.Button(error_log_frame, text="Back to Main Menu", command=lambda:(self.menu_error_log_close_window(), self.menu_settings()))
        error_log_return_Button.grid(row=8,column=0,padx=20,pady=5)
        
        error_log_close_Button = ttk.Button(error_log_frame, text='Close Window', command=lambda:self.menu_error_log_close_window())
        error_log_close_Button.grid(row=9,column=0,padx=20,pady=5)
        
        self.menu_placement(error_log_screen, "UVR Error Log")

    def menu_secondary_model(self, tab, ai_network_vars: dict):
        
        #Settings Tab 1
        secondary_model_Frame = self.menu_FRAME_SET(tab)
        secondary_model_Frame.grid(row=0,column=0,padx=0,pady=0)  
        
        settings_title_Label = self.menu_title_LABEL_SET(secondary_model_Frame, "Secondary Model")
        settings_title_Label.grid(row=0,column=0,padx=0,pady=15)
        
        voc_inst_list = self.model_list(VOCAL_STEM, INST_STEM, is_dry_check=True)
        other_list = self.model_list(OTHER_STEM, NO_OTHER_STEM, is_dry_check=True)
        bass_list = self.model_list(BASS_STEM, NO_BASS_STEM, is_dry_check=True)
        drum_list = self.model_list(DRUM_STEM, NO_DRUM_STEM, is_dry_check=True)
        
        voc_inst_secondary_model_var = ai_network_vars["voc_inst_secondary_model"]
        other_secondary_model_var = ai_network_vars["other_secondary_model"]
        bass_secondary_model_var = ai_network_vars["bass_secondary_model"]
        drums_secondary_model_var = ai_network_vars["drums_secondary_model"]
        voc_inst_secondary_model_scale_var = ai_network_vars['voc_inst_secondary_model_scale']
        other_secondary_model_scale_var = ai_network_vars['other_secondary_model_scale']
        bass_secondary_model_scale_var = ai_network_vars['bass_secondary_model_scale']
        drums_secondary_model_scale_var = ai_network_vars['drums_secondary_model_scale']
        is_secondary_model_activate_var = ai_network_vars["is_secondary_model_activate"]
        
        change_state_lambda = lambda:change_state(NORMAL if is_secondary_model_activate_var.get() else DISABLED)
        init_convert_to_percentage = lambda raw_value:f"{int(float(raw_value)*100)}%"
        
        voc_inst_secondary_model_scale_LABEL_var = tk.StringVar(value=init_convert_to_percentage(voc_inst_secondary_model_scale_var.get()))
        other_secondary_model_scale_LABEL_var = tk.StringVar(value=init_convert_to_percentage(other_secondary_model_scale_var.get()))
        bass_secondary_model_scale_LABEL_var = tk.StringVar(value=init_convert_to_percentage(bass_secondary_model_scale_var.get()))
        drums_secondary_model_scale_LABEL_var = tk.StringVar(value=init_convert_to_percentage(drums_secondary_model_scale_var.get()))

        def change_state(change_state):
            for child_widget in secondary_model_Frame.winfo_children():
                if type(child_widget) is ttk.OptionMenu or type(child_widget) is ttk.Scale:
                    child_widget.configure(state=change_state)

        def convert_to_percentage(raw_value, scale_var: tk.StringVar, label_var: tk.StringVar):
            raw_value = '%0.2f' % float(raw_value)
            scale_var.set(raw_value)
            label_var.set(f"{int(float(raw_value)*100)}%")

        def build_widgets(stem_pair: str, model_list: list, option_var: tk.StringVar, label_var: tk.StringVar, scale_var: tk.DoubleVar, placement: tuple):
            secondary_model_Label = self.menu_sub_LABEL_SET(secondary_model_Frame, f'{stem_pair}', font_size=FONT_SIZE_3)
            secondary_model_Label.grid(row=placement[0],column=0,padx=0,pady=5)
            secondary_model_Option = ttk.OptionMenu(secondary_model_Frame, option_var, None, NO_MODEL, *model_list)
            secondary_model_Option.configure(width=33)
            secondary_model_Option.grid(row=placement[1],column=0,padx=0,pady=5)
            secondary_scale_info_Label = tk.Label(secondary_model_Frame, textvariable=label_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#13a4c9')
            secondary_scale_info_Label.grid(row=placement[2],column=0,padx=0,pady=0)   
            secondary_model_scale_Option = ttk.Scale(secondary_model_Frame, variable=scale_var, from_=0.01, to=0.99, command=lambda s:convert_to_percentage(s, scale_var, label_var), orient='horizontal')
            secondary_model_scale_Option.grid(row=placement[3],column=0,padx=0,pady=2)
            self.help_hints(secondary_model_Label, text=SECONDARY_MODEL_HELP)
            self.help_hints(secondary_scale_info_Label, text=SECONDARY_MODEL_SCALE_HELP)

        build_widgets(stem_pair=VOCAL_PAIR,
                      model_list=voc_inst_list,
                      option_var=voc_inst_secondary_model_var,
                      label_var=voc_inst_secondary_model_scale_LABEL_var,
                      scale_var=voc_inst_secondary_model_scale_var,
                      placement=VOCAL_PAIR_PLACEMENT)
        
        build_widgets(stem_pair=OTHER_PAIR,
                      model_list=other_list,
                      option_var=other_secondary_model_var,
                      label_var=other_secondary_model_scale_LABEL_var,
                      scale_var=other_secondary_model_scale_var,
                      placement=OTHER_PAIR_PLACEMENT)
        
        build_widgets(stem_pair=BASS_PAIR,
                      model_list=bass_list,
                      option_var=bass_secondary_model_var,
                      label_var=bass_secondary_model_scale_LABEL_var,
                      scale_var=bass_secondary_model_scale_var,
                      placement=BASS_PAIR_PLACEMENT)
        
        build_widgets(stem_pair=DRUM_PAIR,
                      model_list=drum_list,
                      option_var=drums_secondary_model_var,
                      label_var=drums_secondary_model_scale_LABEL_var,
                      scale_var=drums_secondary_model_scale_var,
                      placement=DRUMS_PAIR_PLACEMENT)
     
        is_secondary_model_activate_Option = ttk.Checkbutton(secondary_model_Frame, text='Activate Secondary Model', variable=is_secondary_model_activate_var, command=change_state_lambda) 
        is_secondary_model_activate_Option.grid(row=21,column=0,padx=0,pady=5)
        self.help_hints(is_secondary_model_activate_Option, text=SECONDARY_MODEL_ACTIVATE_HELP)
        
        change_state_lambda()
        
    def pop_up_save_current_settings(self):
        """Save current application settings as..."""
        
        settings_save = Toplevel(root)
        
        settings_save_var = tk.StringVar(value='')
        entry_validation_header_var = tk.StringVar(value='Input Notes')

        settings_save_Frame = self.menu_FRAME_SET(settings_save)
        settings_save_Frame.grid(row=1,column=0,padx=0,pady=0)  
        
        validation = lambda value:False if re.fullmatch(REG_SAVE_INPUT, value) is None and settings_save_var.get() else True
        invalid = lambda:(entry_validation_header_var.set(INVALID_ENTRY))
        save_func = lambda:(self.pop_up_save_current_settings_sub_json_dump(settings_save_var.get()), settings_save.destroy())

        settings_save_title = self.menu_title_LABEL_SET(settings_save_Frame, "Save Current Settings")
        settings_save_title.grid(row=2,column=0,padx=0,pady=0)
        
        settings_save_name_Label = self.menu_sub_LABEL_SET(settings_save_Frame, 'Name Settings')
        settings_save_name_Label.grid(row=3,column=0,padx=0,pady=5)
        settings_save_name_Entry = ttk.Entry(settings_save_Frame, textvariable=settings_save_var, justify='center', width=25)
        settings_save_name_Entry.grid(row=4,column=0,padx=0,pady=5)
        settings_save_name_Entry.config(validate='focus', validatecommand=(self.register(validation), '%P'), invalidcommand=(self.register(invalid),))
        settings_save_name_Entry.bind(right_click_button, self.right_click_menu_popup)
        self.current_text_box = settings_save_name_Entry
        
        entry_validation_header_Label = tk.Label(settings_save_Frame, textvariable=entry_validation_header_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#868687', justify="left")
        entry_validation_header_Label.grid(row=5,column=0,padx=0,pady=0)
        
        entry_rules_Label = tk.Label(settings_save_Frame, text=ENSEMBLE_INPUT_RULE, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#868687', justify="left")
        entry_rules_Label.grid(row=6,column=0,padx=0,pady=0)     
        
        settings_save_Button = ttk.Button(settings_save_Frame, text="Save", command=lambda:save_func() if validation(settings_save_var.get()) else None)
        settings_save_Button.grid(row=7,column=0,padx=0,pady=5)
        
        stop_process_Button = ttk.Button(settings_save_Frame, text="Cancel", command=lambda:settings_save.destroy())
        stop_process_Button.grid(row=8,column=0,padx=0,pady=5)
            
        self.menu_placement(settings_save, "Save Current Settings", pop_up=True)

    def pop_up_save_current_settings_sub_json_dump(self, settings_save_name: str):
        """Dumps current application settings to a json named after user input"""
        
        if settings_save_name:
            self.save_current_settings_var.set(settings_save_name)
            settings_save_name = settings_save_name.replace(" ", "_")
            current_settings = self.save_values(app_close=False)
            
            saved_data_dump = json.dumps(current_settings, indent=4)
            with open(os.path.join(SETTINGS_CACHE_DIR, f'{settings_save_name}.json'), "w") as outfile:
                outfile.write(saved_data_dump)

    def pop_up_update_confirmation(self):
        """Ask user is they want to update"""
        
        is_new_update = self.online_data_refresh(confirmation_box=True)
        
        if is_new_update:
            
            update_confirmation_win = Toplevel()
            
            update_confirmation_Frame = self.menu_FRAME_SET(update_confirmation_win)
            update_confirmation_Frame.grid(row=0,column=0,padx=0,pady=0)  
            
            update_found_label = self.menu_title_LABEL_SET(update_confirmation_Frame, 'Update Found', width=15)
            update_found_label.grid(row=0,column=0,padx=0,pady=10)
            
            confirm_update_label = self.menu_sub_LABEL_SET(update_confirmation_Frame, 'Are you sure you want to continue?\n\nThe application will need to be restarted.\n', font_size=FONT_SIZE_3)
            confirm_update_label.grid(row=1,column=0,padx=0,pady=5)
                    
            yes_button = ttk.Button(update_confirmation_Frame, text='Yes', command=lambda:(self.download_item(is_update_app=True), update_confirmation_win.destroy()))
            yes_button.grid(row=2,column=0,padx=0,pady=5)
            
            no_button = ttk.Button(update_confirmation_Frame, text='No', command=lambda:(update_confirmation_win.destroy()))
            no_button.grid(row=3,column=0,padx=0,pady=5)
            
            self.menu_placement(update_confirmation_win, "Confirm Update", pop_up=True)

    def pop_up_user_code_input(self):
        """Input VIP Code"""
        
        self.user_code_validation_var.set('')
        
        self.user_code = Toplevel()
        
        user_code_Frame = self.menu_FRAME_SET(self.user_code)
        user_code_Frame.grid(row=0,column=0,padx=0,pady=0)  
                
        user_code_title_Label = self.menu_title_LABEL_SET(user_code_Frame, 'User Download Codes', width=20)
        user_code_title_Label.grid(row=0,column=0,padx=0,pady=5)    
        
        user_code_Label = self.menu_sub_LABEL_SET(user_code_Frame, 'Download Code')
        user_code_Label.grid(row=1,column=0,padx=0,pady=5)       
                
        self.user_code_Entry = ttk.Entry(user_code_Frame, textvariable=self.user_code_var, justify='center')
        self.user_code_Entry.grid(row=2,column=0,padx=0,pady=5)
        self.user_code_Entry.bind(right_click_button, self.right_click_menu_popup)
        self.current_text_box = self.user_code_Entry
        
        validation_Label = tk.Label(user_code_Frame, textvariable=self.user_code_validation_var, font=("Century Gothic",  f"{FONT_SIZE_0}"), foreground='#868687')
        validation_Label.grid(row=3,column=0,padx=0,pady=0)     

        user_code_confrim_Button = ttk.Button(user_code_Frame, text='Confirm', command=lambda:self.download_validate_code(confirm=True))
        user_code_confrim_Button.grid(row=4,column=0,padx=0,pady=5)
        
        user_code_cancel_Button = ttk.Button(user_code_Frame, text='Cancel', command=lambda:self.user_code.destroy())
        user_code_cancel_Button.grid(row=5,column=0,padx=0,pady=5)
        
        support_title_Label = self.menu_title_LABEL_SET(user_code_Frame, text='Support UVR', width=20)
        support_title_Label.grid(row=6,column=0,padx=0,pady=5)    
        
        support_sub_Label = tk.Label(user_code_Frame, text="Obtain codes by visiting the following\n \"Buy Me a Coffee\" " +\
                                                            "or \"Patreon\".\nClick one of the buttons below to\n donate, " +\
                                                            "pledge or just obatain the code!\n (Donations are not required to obtain VIP code).", 
                                                            font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#13a4c9')
        support_sub_Label.grid(row=7,column=0,padx=0,pady=5)
        
        uvr_patreon_Button = ttk.Button(user_code_Frame, text='UVR Patreon Link', command=lambda:webbrowser.open_new_tab(DONATE_LINK_PATREON))
        uvr_patreon_Button.grid(row=8,column=0,padx=0,pady=5)
        
        bmac_patreon_Button=ttk.Button(user_code_Frame, text='UVR \"Buy Me a Coffee\" Link', command=lambda:webbrowser.open_new_tab(DONATE_LINK_BMAC))
        bmac_patreon_Button.grid(row=9,column=0,padx=0,pady=5)
        
        self.menu_placement(self.user_code, "Input Code", pop_up=True)

    def pop_up_mdx_model(self, mdx_model_hash, model_path):
        """Opens MDX-Net model settings"""

        model = onnx.load(model_path)
        model_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input][0]
        dim_f = model_shapes[2]
        dim_t = int(math.log(model_shapes[3], 2))

        mdx_model_set = Toplevel(root)

        mdx_n_fft_scale_set_var = tk.StringVar(value='6144')
        mdx_dim_f_set_var = tk.StringVar(value=dim_f)
        mdx_dim_t_set_var = tk.StringVar(value=dim_t)
        primary_stem_var = tk.StringVar(value='Vocals')
        mdx_compensate_var = tk.StringVar(value=1.035)
             
        mdx_model_set_Frame = self.menu_FRAME_SET(mdx_model_set)
        mdx_model_set_Frame.grid(row=2,column=0,padx=0,pady=0)  
        
        mdx_model_set_title = self.menu_title_LABEL_SET(mdx_model_set_Frame, "Specify MDX-Net Model Parameters")
        mdx_model_set_title.grid(row=0,column=0,padx=0,pady=15)
                
        set_stem_name_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, 'Primary Stem')
        set_stem_name_Label.grid(row=3,column=0,padx=0,pady=5)
        set_stem_name_Option = ttk.OptionMenu(mdx_model_set_Frame, primary_stem_var, None, *STEM_SET_MENU)
        set_stem_name_Option.configure(width=12)
        set_stem_name_Option.grid(row=4,column=0,padx=0,pady=5)
        self.help_hints(set_stem_name_Label, text=SET_STEM_NAME_HELP)

        mdx_dim_t_set_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, 'Dim_t')
        mdx_dim_t_set_Label.grid(row=5,column=0,padx=0,pady=5)
        mdx_dim_f_set_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, '(Leave this setting as is if you are unsure.)')
        mdx_dim_f_set_Label.grid(row=6,column=0,padx=0,pady=5)
        mdx_dim_t_set_Option = ttk.Combobox(mdx_model_set_Frame, value=('7', '8'), textvariable=mdx_dim_t_set_var)
        mdx_dim_t_set_Option.configure(width=12)
        mdx_dim_t_set_Option.grid(row=7,column=0,padx=0,pady=5)
        self.help_hints(mdx_dim_t_set_Label, text=MDX_DIM_T_SET_HELP)
        
        mdx_dim_f_set_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, 'Dim_f')
        mdx_dim_f_set_Label.grid(row=8,column=0,padx=0,pady=5)
        mdx_dim_f_set_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, '(Leave this setting as is if you are unsure.)')
        mdx_dim_f_set_Label.grid(row=9,column=0,padx=0,pady=5)
        mdx_dim_f_set_Option = ttk.Combobox(mdx_model_set_Frame, value=(MDX_POP_DIMF), textvariable=mdx_dim_f_set_var)
        mdx_dim_f_set_Option.configure(width=12)
        mdx_dim_f_set_Option.grid(row=10,column=0,padx=0,pady=5)
        self.help_hints(mdx_dim_f_set_Label, text=MDX_DIM_F_SET_HELP)

        mdx_n_fft_scale_set_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, 'N_FFT Scale')
        mdx_n_fft_scale_set_Label.grid(row=11,column=0,padx=0,pady=5)
        mdx_n_fft_scale_set_Option = ttk.Combobox(mdx_model_set_Frame, values=(MDX_POP_NFFT), textvariable=mdx_n_fft_scale_set_var)
        mdx_n_fft_scale_set_Option.configure(width=12)
        mdx_n_fft_scale_set_Option.grid(row=12,column=0,padx=0,pady=5)
        self.help_hints(mdx_n_fft_scale_set_Label, text=MDX_N_FFT_SCALE_SET_HELP)
        
        mdx_compensate_Label = self.menu_sub_LABEL_SET(mdx_model_set_Frame, 'Volume Compensation')
        mdx_compensate_Label.grid(row=13,column=0,padx=0,pady=5)
        mdx_compensate_Entry = ttk.Combobox(mdx_model_set_Frame, value=('1.035', '1.08'), textvariable=mdx_compensate_var)
        mdx_compensate_Entry.configure(width=14)
        mdx_compensate_Entry.grid(row=15,column=0,padx=0,pady=5)
        self.help_hints(mdx_compensate_Label, text=POPUP_COMPENSATE_HELP)

        mdx_param_set_Button = ttk.Button(mdx_model_set_Frame, text="Confirm", command=lambda:pull_data())
        mdx_param_set_Button.grid(row=16,column=0,padx=0,pady=10)
        
        stop_process_Button = ttk.Button(mdx_model_set_Frame, text="Cancel", command=lambda:cancel())
        stop_process_Button.grid(row=17,column=0,padx=0,pady=0)
        
        def pull_data():
            mdx_model_params = {
                'compensate': float(mdx_compensate_var.get()),
                'mdx_dim_f_set': int(mdx_dim_f_set_var.get()),
                'mdx_dim_t_set': int(mdx_dim_t_set_var.get()),
                'mdx_n_fft_scale_set': int(mdx_n_fft_scale_set_var.get()),
                'primary_stem': primary_stem_var.get()
                }
            
            self.pop_up_mdx_model_sub_json_dump(mdx_model_params, mdx_model_hash)
            mdx_model_set.destroy()

        def cancel():
            mdx_model_set.destroy()
            
        mdx_model_set.protocol("WM_DELETE_WINDOW", cancel)
            
        self.menu_placement(mdx_model_set, "Specify Parameters", pop_up=True)
                        
    def pop_up_mdx_model_sub_json_dump(self, mdx_model_params, mdx_model_hash):
        """Dumps current selected MDX-Net model settings to a json named after model hash"""
        
        self.mdx_model_params = mdx_model_params

        mdx_model_params_dump = json.dumps(mdx_model_params, indent=4)
        with open(os.path.join(MDX_HASH_DIR, f'{mdx_model_hash}.json'), "w") as outfile:
            outfile.write(mdx_model_params_dump)
        
    def pop_up_vr_param(self, vr_model_hash):
        """Opens VR param settings"""

        vr_param_menu = Toplevel()
        
        get_vr_params = lambda dir, ext:tuple(os.path.splitext(x)[0] for x in os.listdir(dir) if x.endswith(ext))
        new_vr_params = get_vr_params(VR_PARAM_DIR, '.json')
        vr_model_param_var = tk.StringVar(value='None Selected')
        vr_model_stem_var = tk.StringVar(value='Vocals')
        
        def pull_data():
            vr_model_params = {
                'vr_model_param': vr_model_param_var.get(),
                'primary_stem': vr_model_stem_var.get()}
            
            if not vr_model_param_var.get() == 'None Selected':
                self.pop_up_vr_param_sub_json_dump(vr_model_params, vr_model_hash)
                vr_param_menu.destroy()
            else:
                self.vr_model_params = None
        
        def cancel():
            self.vr_model_params = None
            vr_param_menu.destroy()
        
        vr_param_Frame = self.menu_FRAME_SET(vr_param_menu)
        vr_param_Frame.grid(row=0,column=0,padx=0,pady=0)  
            
        vr_param_title_title = self.menu_title_LABEL_SET(vr_param_Frame, "Specify VR Model Parameters", width=25)
        vr_param_title_title.grid(row=0,column=0,padx=0,pady=0)
                
        vr_model_stem_Label = self.menu_sub_LABEL_SET(vr_param_Frame, 'Primary Stem')
        vr_model_stem_Label.grid(row=1,column=0,padx=0,pady=5)    
        vr_model_stem_Option = ttk.OptionMenu(vr_param_Frame, vr_model_stem_var, None, *STEM_SET_MENU)
        vr_model_stem_Option.grid(row=2,column=0,padx=20,pady=5)
        self.help_hints(vr_model_stem_Label, text=SET_STEM_NAME_HELP)
                
        vr_model_param_Label = self.menu_sub_LABEL_SET(vr_param_Frame, 'Select Model Param')
        vr_model_param_Label.grid(row=3,column=0,padx=0,pady=5)
        vr_model_param_Option = ttk.OptionMenu(vr_param_Frame, vr_model_param_var)
        vr_model_param_Option.configure(width=30)
        vr_model_param_Option.grid(row=4,column=0,padx=20,pady=5)
        self.help_hints(vr_model_param_Label, text=VR_MODEL_PARAM_HELP)

        vr_param_confrim_Button = ttk.Button(vr_param_Frame, text='Confirm', command=lambda:pull_data())
        vr_param_confrim_Button.grid(row=5,column=0,padx=0,pady=5)
        
        vr_param_cancel_Button = ttk.Button(vr_param_Frame, text='Cancel', command=cancel)
        vr_param_cancel_Button.grid(row=6,column=0,padx=0,pady=5)
        
        for option_name in new_vr_params:
            vr_model_param_Option['menu'].add_radiobutton(label=option_name, command=tk._setit(vr_model_param_var, option_name))
        
        vr_param_menu.protocol("WM_DELETE_WINDOW", cancel)
        
        self.menu_placement(vr_param_menu, "Choose Model Param", pop_up=True)

    def pop_up_vr_param_sub_json_dump(self, vr_model_params, vr_model_hash):
        """Dumps current selected VR model settings to a json named after model hash"""
        
        self.vr_model_params = vr_model_params

        vr_model_params_dump = json.dumps(vr_model_params, indent=4)
        
        with open(os.path.join(VR_HASH_DIR, f'{vr_model_hash}.json'), "w") as outfile:
            outfile.write(vr_model_params_dump)

    def pop_up_save_ensemble(self):
        """
        Save Ensemble as...
        """
        
        ensemble_save = Toplevel(root)
        
        ensemble_save_var = tk.StringVar(value='')
        entry_validation_header_var = tk.StringVar(value='Input Notes')

        ensemble_save_Frame = self.menu_FRAME_SET(ensemble_save)
        ensemble_save_Frame.grid(row=1,column=0,padx=0,pady=0)  
        
        validation = lambda value:False if re.fullmatch(REG_SAVE_INPUT, value) is None and ensemble_save_var.get() else True
        invalid = lambda:(entry_validation_header_var.set(INVALID_ENTRY))
        save_func = lambda:(self.pop_up_save_ensemble_sub_json_dump(self.ensemble_listbox_get_all_selected_models(), ensemble_save_var.get()), ensemble_save.destroy())

        if len(self.ensemble_listbox_get_all_selected_models()) <= 1:
            ensemble_save_title = self.menu_title_LABEL_SET(ensemble_save_Frame, "Not Enough Models", width=20)
            ensemble_save_title.grid(row=1,column=0,padx=0,pady=0)
            
            ensemble_save_title = self.menu_sub_LABEL_SET(ensemble_save_Frame, "You must select 2 or more models to save an ensemble.")
            ensemble_save_title.grid(row=2,column=0,padx=0,pady=5)
            
            stop_process_Button = ttk.Button(ensemble_save_Frame, text="OK", command=lambda:ensemble_save.destroy())
            stop_process_Button.grid(row=3,column=0,padx=0,pady=10)
        else:
            ensemble_save_title = self.menu_title_LABEL_SET(ensemble_save_Frame, "Save Current Ensemble")
            ensemble_save_title.grid(row=2,column=0,padx=0,pady=0)
            
            ensemble_name_Label = self.menu_sub_LABEL_SET(ensemble_save_Frame, 'Ensemble Name')
            ensemble_name_Label.grid(row=3,column=0,padx=0,pady=5)
            ensemble_name_Entry = ttk.Entry(ensemble_save_Frame, textvariable=ensemble_save_var, justify='center', width=25)
            ensemble_name_Entry.grid(row=4,column=0,padx=0,pady=5)
            ensemble_name_Entry.config(validate='focus', validatecommand=(self.register(validation), '%P'), invalidcommand=(self.register(invalid),))
            
            entry_validation_header_Label = tk.Label(ensemble_save_Frame, textvariable=entry_validation_header_var, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#868687', justify="left")
            entry_validation_header_Label.grid(row=5,column=0,padx=0,pady=0)
            
            entry_rules_Label = tk.Label(ensemble_save_Frame, text=ENSEMBLE_INPUT_RULE, font=("Century Gothic", f"{FONT_SIZE_1}"), foreground='#868687', justify="left")
            entry_rules_Label.grid(row=6,column=0,padx=0,pady=0)     
            
            mdx_param_set_Button = ttk.Button(ensemble_save_Frame, text="Save", command=lambda:save_func() if validation(ensemble_save_var.get()) else None)
            mdx_param_set_Button.grid(row=7,column=0,padx=0,pady=5)
            
            stop_process_Button = ttk.Button(ensemble_save_Frame, text="Cancel", command=lambda:ensemble_save.destroy())
            stop_process_Button.grid(row=8,column=0,padx=0,pady=5)
            
        self.menu_placement(ensemble_save, "Save Current Ensemble", pop_up=True)
        
    def pop_up_save_ensemble_sub_json_dump(self, selected_ensemble_model, ensemble_save_name: str):
        """Dumps current ensemble settings to a json named after user input"""
        
        if ensemble_save_name:
            self.chosen_ensemble_var.set(ensemble_save_name)
            ensemble_save_name = ensemble_save_name.replace(" ", "_")
            saved_data = {
                'ensemble_main_stem': self.ensemble_main_stem_var.get(),
                'ensemble_type': self.ensemble_type_var.get(),
                'selected_models': selected_ensemble_model,
                }
            
            saved_data_dump = json.dumps(saved_data, indent=4)
            with open(os.path.join(ENSEMBLE_CACHE_DIR, f'{ensemble_save_name}.json'), "w") as outfile:
                outfile.write(saved_data_dump)

    def deletion_list_fill(self, option_menu: ttk.OptionMenu, selection_var: tk.StringVar, selection_list, selection_dir, var_set):
        """Fills the saved settings menu located in tab 2 of the main settings window"""
        
        option_menu['menu'].delete(0, 'end')
        for selection in selection_list:
            selection = selection.replace("_", " ")
            option_menu['menu'].add_radiobutton(label=selection, 
                                                command=tk._setit(selection_var, 
                                                                    selection, 
                                                                    lambda s:(self.deletion_entry(s, option_menu, selection_dir), 
                                                                    selection_var.set(var_set))))
        
    def deletion_entry(self, selection: str, option_menu, path):
        """Deletes selected user saved application settings"""
        
        if selection not in [SELECT_SAVED_SET, SELECT_SAVED_ENSEMBLE]:
            saved_path = os.path.join(path, f'{selection.replace(" ", "_")}.json')
            confirm = self.message_box(DELETE_ENS_ENTRY)
            
            if confirm:
                if os.path.isfile(saved_path):
                    os.remove(saved_path)
                    r_index=option_menu['menu'].index(selection) # index of selected option. 
                    option_menu['menu'].delete(r_index)    # deleted the option

    #--Download Center Methods--    

    def online_data_refresh(self, user_refresh=True, confirmation_box=False, refresh_list_Button=False):
        """Checks for application updates"""
        
        def online_check():

            self.app_update_status_Text_var.set('Loading version information...')
            self.app_update_button_Text_var.set('Check for Updates')
            is_new_update = False
            
            try:
                self.online_data = json.load(urllib.request.urlopen(DOWNLOAD_CHECKS))
                self.is_online = True
                self.lastest_version = self.online_data["current_version"]
                
                if self.lastest_version == PATCH:
                    self.app_update_status_Text_var.set('UVR Version Current')
                else:
                    is_new_update = True
                    self.app_update_status_Text_var.set(f"Update Found: {self.lastest_version}")
                    self.app_update_button_Text_var.set('Click Here to Update')
                    self.download_update_link_var.set('{}{}.dmg'.format(UPDATE_REPO, self.lastest_version))
                    self.download_update_path_var.set(os.path.join(BASE_PATH, f'{self.lastest_version}.dmg'))
                    
                    if not user_refresh:
                        self.new_update_notify(self.lastest_version)

                if user_refresh:
                    self.download_list_state()
                    self.download_list_fill()
                    for widget in self.download_center_Buttons:widget.configure(state=tk.NORMAL)
                    
                if refresh_list_Button:
                    self.download_progress_info_var.set('Download List Refreshed!')
                    
                self.download_model_settings()
                    
            except Exception as e:
                self.error_log_var.set(error_text('Online Data Refresh', e))
                self.offline_state_set()
                is_new_update = False
                
                if user_refresh:
                    self.download_list_state(disable_only=True)
                    for widget in self.download_center_Buttons:widget.configure(state=tk.DISABLED)

            return is_new_update
            
        if confirmation_box:
            return online_check()
        else:
            self.current_thread = KThread(target=online_check)
            self.current_thread.start()
                
    def offline_state_set(self):
        """Changes relevent settings and "Download Center" buttons if no internet connection is available"""
        
        self.app_update_status_Text_var.set(f'Version Status: {NO_CONNECTION}')
        self.download_progress_info_var.set(NO_CONNECTION) 
        self.app_update_button_Text_var.set('Refresh')
        self.refresh_list_Button.configure(state=tk.NORMAL) if self.refresh_list_Button else None
        self.stop_download_Button_DISABLE() if self.stop_download_Button_DISABLE else None
        self.enable_tabs() if self.enable_tabs else None
        self.is_online = False

    def download_validate_code(self, confirm=False):
        """Verifies the VIP download code"""
        
        self.decoded_vip_link = vip_downloads(self.user_code_var.get())
        
        if confirm:
            if not self.decoded_vip_link == NO_CODE:
                self.download_progress_info_var.set('VIP Models Added!') 
                self.user_code.destroy()
            else:
                self.download_progress_info_var.set('Incorrect Code') 
                self.user_code_validation_var.set('Code Incorrect')
                
            self.download_list_fill()

    def download_list_fill(self):
        """Fills the download lists with the data retrieved from the update check."""
        
        self.download_demucs_models_list.clear()

        for list_option in self.download_lists:
            list_option['menu'].delete(0, 'end')

        self.vr_download_list = self.online_data["vr_download_list"]
        self.mdx_download_list = self.online_data["mdx_download_list"]
        self.demucs_download_list = self.online_data["demucs_download_list"]
        
        if not self.decoded_vip_link is NO_CODE:
            self.vr_download_list.update(self.online_data["vr_download_vip_list"])
            self.mdx_download_list.update(self.online_data["mdx_download_vip_list"])
                                
        for (selectable, model) in self.vr_download_list.items():
            if not os.path.isfile(os.path.join(VR_MODELS_DIR, model)):
                self.model_download_vr_Option['menu'].add_radiobutton(label=selectable, command=tk._setit(self.model_download_vr_var, selectable, lambda s:self.download_model_select(s, VR_ARCH_TYPE)))
            
        for (selectable, model) in self.mdx_download_list.items():
            if not os.path.isfile(os.path.join(MDX_MODELS_DIR, model)):
                self.model_download_mdx_Option['menu'].add_radiobutton(label=selectable, command=tk._setit(self.model_download_mdx_var, selectable, lambda s:self.download_model_select(s, MDX_ARCH_TYPE)))
   
        for (selectable, model) in self.demucs_download_list.items():  
            for name in model.items():
                if [True for x in DEMUCS_NEWER_ARCH_TYPES if x in selectable]:
                    if not os.path.isfile(os.path.join(DEMUCS_NEWER_REPO_DIR, name[0])):
                        self.download_demucs_models_list.append(selectable)
                else:
                    if not os.path.isfile(os.path.join(DEMUCS_MODELS_DIR, name[0])):
                        self.download_demucs_models_list.append(selectable)
                    
        self.download_demucs_models_list = list(dict.fromkeys(self.download_demucs_models_list)) 
        
        for option_name in self.download_demucs_models_list:    
            self.model_download_demucs_Option['menu'].add_radiobutton(label=option_name, command=tk._setit(self.model_download_demucs_var, option_name, lambda s:self.download_model_select(s, DEMUCS_ARCH_TYPE)))
            
        if self.model_download_vr_Option['menu'].index("end") is None:
            self.model_download_vr_Option['menu'].add_radiobutton(label=NO_NEW_MODELS, command=tk._setit(self.model_download_vr_var, NO_MODEL, lambda s:self.download_model_select(s, MDX_ARCH_TYPE)))
            
        if self.model_download_mdx_Option['menu'].index("end") is None:
            self.model_download_mdx_Option['menu'].add_radiobutton(label=NO_NEW_MODELS, command=tk._setit(self.model_download_mdx_var, NO_MODEL, lambda s:self.download_model_select(s, MDX_ARCH_TYPE)))
            
        if self.model_download_demucs_Option['menu'].index("end") is None:
            self.model_download_demucs_Option['menu'].add_radiobutton(label=NO_NEW_MODELS, command=tk._setit(self.model_download_demucs_var, NO_MODEL, lambda s:self.download_model_select(s, DEMUCS_ARCH_TYPE)))

    def download_model_settings(self):
        '''Update the newest model settings'''
        
        self.vr_hash_MAPPER = json.load(urllib.request.urlopen(VR_MODEL_DATA_LINK))
        self.mdx_hash_MAPPER = json.load(urllib.request.urlopen(MDX_MODEL_DATA_LINK))
        
        try:
            vr_hash_MAPPER_dump = json.dumps(self.vr_hash_MAPPER, indent=4)
            with open(VR_HASH_JSON, "w") as outfile:
                outfile.write(vr_hash_MAPPER_dump)
                
            mdx_hash_MAPPER_dump = json.dumps(self.mdx_hash_MAPPER, indent=4)
            with open(MDX_HASH_JSON, "w") as outfile:
                outfile.write(mdx_hash_MAPPER_dump)
        except Exception as e:
            self.error_log_var.set(e)
            print(e)

    def download_list_state(self, reset=True, disable_only=False):
        """Makes sure only the models from the chosen AI network are selectable."""
        
        for widget in self.download_lists:widget.configure(state=tk.DISABLED)
        
        if reset:
            for download_list_var in self.download_list_vars:
                if self.is_online:
                    download_list_var.set(NO_MODEL)
                    self.download_Button.configure(state=tk.NORMAL)
                else:
                    download_list_var.set(NO_CONNECTION)
                    self.download_Button.configure(state=tk.DISABLED)
            
        if not disable_only:
            
            self.download_Button.configure(state=tk.NORMAL)
            if self.select_download_var.get() == VR_ARCH_TYPE:
                self.model_download_vr_Option.configure(state=tk.NORMAL)
                self.selected_download_var = self.model_download_vr_var
            if self.select_download_var.get() == MDX_ARCH_TYPE:
                self.model_download_mdx_Option.configure(state=tk.NORMAL)
                self.selected_download_var = self.model_download_mdx_var
            if self.select_download_var.get() == DEMUCS_ARCH_TYPE:
                self.model_download_demucs_Option.configure(state=tk.NORMAL)
                self.selected_download_var = self.model_download_demucs_var
                
            self.stop_download_Button_DISABLE()

    def download_model_select(self, selection, type):
        """Prepares the data needed to download selected model."""
        
        self.download_demucs_newer_models.clear()

        model_repo = self.decoded_vip_link if VIP_SELECTION in selection else NORMAL_REPO
        is_demucs_newer = [True for x in DEMUCS_NEWER_ARCH_TYPES if x in selection]

        if type == VR_ARCH_TYPE:
            for selected_model in self.vr_download_list.items():
                if selection in selected_model:
                    self.download_link_path_var.set("{}{}".format(model_repo, selected_model[1]))
                    self.download_save_path_var.set(os.path.join(VR_MODELS_DIR, selected_model[1]))
                    break
                
        if type == MDX_ARCH_TYPE:
            for selected_model in self.mdx_download_list.items():
                if selection in selected_model:
                    self.download_link_path_var.set("{}{}".format(model_repo, selected_model[1]))
                    self.download_save_path_var.set(os.path.join(MDX_MODELS_DIR, selected_model[1]))
                    break
                
        if type == DEMUCS_ARCH_TYPE:
            for selected_model, model_data in self.demucs_download_list.items():
                if selection == selected_model:
                    for key, value in model_data.items():
                        if is_demucs_newer:
                            self.download_demucs_newer_models.append([os.path.join(DEMUCS_NEWER_REPO_DIR, key), value])
                        else:
                            self.download_save_path_var.set(os.path.join(DEMUCS_MODELS_DIR, key))
                            self.download_link_path_var.set(value)

    def download_item(self, is_update_app=False):
        """Downloads the model selected."""
        
        if not is_update_app:
            if self.selected_download_var.get() == NO_MODEL:
                self.download_progress_info_var.set(NO_MODEL)
                return        

        for widget in self.download_center_Buttons:widget.configure(state=tk.DISABLED)
        self.refresh_list_Button.configure(state=tk.DISABLED)
        
        is_demucs_newer = [True for x in DEMUCS_NEWER_ARCH_TYPES if x in self.selected_download_var.get()]

        self.download_list_state(reset=False, disable_only=True)
        self.stop_download_Button_ENABLE()
        self.disable_tabs()
        
        def download_progress_bar(current, total, model=80):
            progress = ('%s' % (100 * current // total))
            self.download_progress_bar_var.set(int(progress))
            self.download_progress_percent_var.set(progress + ' %')
            
        def push_download():
            self.is_download_thread_active = True
            try:
                if is_update_app:
                    self.download_progress_info_var.set(DOWNLOADING_UPDATE)
                    if os.path.isfile(self.download_update_path_var.get()):
                        self.download_progress_info_var.set(FILE_EXISTS)
                    else:
                        wget.download(self.download_update_link_var.get(), self.download_update_path_var.get(), bar=download_progress_bar)
                        
                    self.download_post_action(DOWNLOAD_UPDATE_COMPLETE)
                else:
                    if self.select_download_var.get() == DEMUCS_ARCH_TYPE and is_demucs_newer:
                        for model_num, model_data in enumerate(self.download_demucs_newer_models, start=1):
                            self.download_progress_info_var.set('{} {}/{}...'.format(DOWNLOADING_ITEM, model_num, len(self.download_demucs_newer_models)))
                            if os.path.isfile(model_data[0]):
                                continue
                            else:
                                wget.download(model_data[1], model_data[0], bar=download_progress_bar)
                    else:
                        self.download_progress_info_var.set(SINGLE_DOWNLOAD)
                        if os.path.isfile(self.download_save_path_var.get()):
                            self.download_progress_info_var.set(FILE_EXISTS)
                        else:
                            wget.download(self.download_link_path_var.get(), self.download_save_path_var.get(), bar=download_progress_bar)
                            
                    self.download_post_action(DOWNLOAD_COMPLETE)
                
            except Exception as e:
                self.error_log_var.set(error_text(DOWNLOADING_ITEM, e))
                self.download_progress_info_var.set(DOWNLOAD_FAILED)
                
                if type(e).__name__ == 'URLError':
                    self.offline_state_set()
                else:
                    self.download_progress_percent_var.set(f"{type(e).__name__}")
                    self.download_post_action(DOWNLOAD_FAILED)
                          
        self.active_download_thread = KThread(target=push_download)
        self.active_download_thread.start()

    def download_post_action(self, action):
        """Resets the widget variables in the "Download Center" based on the state of the download."""
        
        for widget in self.download_center_Buttons:widget.configure(state=tk.NORMAL)
        self.refresh_list_Button.configure(state=tk.NORMAL)
        
        self.enable_tabs()
        self.stop_download_Button_DISABLE()
        
        if action == DOWNLOAD_FAILED:
            try:
                self.active_download_thread.terminate()
            finally:
                self.download_progress_info_var.set(DOWNLOAD_FAILED)
                self.download_list_state(reset=False)
        if action == DOWNLOAD_STOPPED:
            try:
                self.active_download_thread.terminate()
            finally:
                self.download_progress_info_var.set(DOWNLOAD_STOPPED)
                self.download_list_state(reset=False)
        if action == DOWNLOAD_COMPLETE:
            self.online_data_refresh()
            self.download_progress_info_var.set(DOWNLOAD_COMPLETE)
            self.download_list_state()
        if action == DOWNLOAD_UPDATE_COMPLETE:
            self.download_progress_info_var.set(DOWNLOAD_UPDATE_COMPLETE)
            if os.path.isfile(self.download_update_path_var.get()):
                subprocess.Popen(self.download_update_path_var.get())
            self.download_list_state()
        
        self.is_download_thread_active = False
        
        self.delete_temps()
   
    #--Refresh/Loop Methods--    
        
    def update_loop(self):
        """Update the model dropdown menus"""
            
        if self.clear_cache_torch:
            torch.cuda.empty_cache()
            self.clear_cache_torch = False
            
        if self.is_process_stopped:
            if self.thread_check(self.active_processing_thread):
                self.conversion_Button_Text_var.set(STOP_PROCESSING)
                self.conversion_Button.configure(state=tk.DISABLED)
                self.stop_Button.configure(state=tk.DISABLED)
            else:
                self.stop_Button.configure(state=tk.NORMAL)
                self.conversion_Button_Text_var.set(START_PROCESSING)
                self.conversion_Button.configure(state=tk.NORMAL)
                self.progress_bar_main_var.set(0)
                torch.cuda.empty_cache()
                self.is_process_stopped = False

        if self.is_confirm_error_var.get():
            self.check_is_open_menu_error_log()
            self.is_confirm_error_var.set(False)

        self.update_available_models()
        self.after(600, self.update_loop)
        
    def update_available_models(self):
        """
        Loops through all models in each model directory and adds them to the appropriate model menu.
        Also updates ensemble listbox and user saved settings list.
        """
        
        def fix_names(file, name_mapper: dict):return tuple(new_name for (old_name, new_name) in name_mapper.items() if file in old_name)
        
        new_vr_models = self.get_files_from_dir(VR_MODELS_DIR, '.pth')
        new_mdx_models = self.get_files_from_dir(MDX_MODELS_DIR, '.onnx')
        new_demucs_models = self.get_files_from_dir(DEMUCS_MODELS_DIR, ('.ckpt', '.gz', '.th')) + self.get_files_from_dir(DEMUCS_NEWER_REPO_DIR, '.yaml')
        new_ensembles_found = self.get_files_from_dir(ENSEMBLE_CACHE_DIR, '.json')
        new_settings_found = self.get_files_from_dir(SETTINGS_CACHE_DIR, '.json')
        new_models_found = new_vr_models + new_mdx_models + new_demucs_models
        is_online = self.is_online_model_menu
        
        def loop_directories(option_menu, option_var, model_list, model_type, name_mapper):
            
            option_list = model_list
            option_menu['menu'].delete(0, 'end')

            if name_mapper:
                option_list = []
                for file_name in model_list:
                    if fix_names(file_name, name_mapper): 
                        file = fix_names(file_name, name_mapper)[0]
                    else: 
                        file = file_name
                    option_list.append(file)
                
                option_list = tuple(option_list)

            for option_name in natsort.natsorted(option_list):
                option_menu['menu'].add_radiobutton(label=option_name, command=tk._setit(option_var, option_name, self.selection_action_models))
            
            if self.is_online:
                option_menu['menu'].insert_separator(len(model_list))
                option_menu['menu'].add_radiobutton(label=DOWNLOAD_MORE, command=tk._setit(option_var, DOWNLOAD_MORE, self.selection_action_models))
                               
            return tuple(f"{model_type}{ENSEMBLE_PARTITION}{model_name}" for model_name in natsort.natsorted(option_list))
        
        if new_models_found != self.last_found_models or is_online != self.is_online:
            self.model_data_table = []
            
            vr_model_list = loop_directories(self.vr_model_Option, self.vr_model_var, new_vr_models, VR_ARCH_TYPE, name_mapper=None)
            mdx_model_list = loop_directories(self.mdx_net_model_Option, self.mdx_net_model_var, new_mdx_models, MDX_ARCH_TYPE, name_mapper=MDX_NAME_SELECT)
            demucs_model_list = loop_directories(self.demucs_model_Option, self.demucs_model_var, new_demucs_models, DEMUCS_ARCH_TYPE, name_mapper=DEMUCS_NAME_SELECT)
            
            self.ensemble_model_list = vr_model_list + mdx_model_list + demucs_model_list
            self.last_found_models = new_models_found
            self.is_online_model_menu = self.is_online
            
            if not self.chosen_ensemble_var.get() == CHOOSE_ENSEMBLE_OPTION:
                self.selection_action_chosen_ensemble(self.chosen_ensemble_var.get())
            else:
                if not self.ensemble_main_stem_var.get() == CHOOSE_STEM_PAIR:
                    self.selection_action_ensemble_stems(self.ensemble_main_stem_var.get(), auto_update=self.ensemble_listbox_get_all_selected_models())
                else:
                    self.ensemble_listbox_clear_and_insert_new(self.ensemble_model_list)
                                    
        if new_ensembles_found != self.last_found_ensembles:
            ensemble_options = new_ensembles_found + ENSEMBLE_OPTIONS
            self.chosen_ensemble_Option['menu'].delete(0, 'end')
            
            for saved_ensemble in ensemble_options:
                saved_ensemble = saved_ensemble.replace("_", " ")
                self.chosen_ensemble_Option['menu'].add_radiobutton(label=saved_ensemble, 
                                                                    command=tk._setit(self.chosen_ensemble_var, saved_ensemble, self.selection_action_chosen_ensemble))
                
            self.chosen_ensemble_Option['menu'].insert_separator(len(new_ensembles_found))
            self.last_found_ensembles = new_ensembles_found
            
        if new_settings_found != self.last_found_settings:
            settings_options = new_settings_found + SAVE_SET_OPTIONS
            self.save_current_settings_Option['menu'].delete(0, 'end')
            
            for settings_options in settings_options:
                settings_options = settings_options.replace("_", " ")
                self.save_current_settings_Option['menu'].add_radiobutton(label=settings_options, 
                                                                          command=tk._setit(self.save_current_settings_var, settings_options, self.selection_action_saved_settings))
                
            self.save_current_settings_Option['menu'].insert_separator(len(new_settings_found))
            self.last_found_settings = new_settings_found
          
    def update_main_widget_states(self):
        """Updates main widget states based on chosen process method"""
        
        for widget in self.GUI_LIST:
            widget.place_forget()

        general_shared_Buttons_place = lambda:(self.is_gpu_conversion_Option_place(), self.model_sample_mode_Option_place())
        stem_save_Options_place = lambda:(self.is_primary_stem_only_Option_place(), self.is_secondary_stem_only_Option_place())
        stem_save_demucs_Options_place = lambda:(self.is_primary_stem_only_Demucs_Option_place(), self.is_secondary_stem_only_Demucs_Option_place())
        no_ensemble_shared = lambda:(self.save_current_settings_Label_place(), self.save_current_settings_Option_place())

        if self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
            self.mdx_net_model_Label_place()
            self.mdx_net_model_Option_place()
            self.chunks_Label_place()
            self.chunks_Option_place()
            self.margin_Label_place()
            self.margin_Option_place()
            general_shared_Buttons_place()
            stem_save_Options_place()
            no_ensemble_shared()
        elif self.chosen_process_method_var.get() == VR_ARCH_PM:
            self.vr_model_Label_place()
            self.vr_model_Option_place()
            self.aggression_setting_Label_place()
            self.aggression_setting_Option_place()
            self.window_size_Label_place()
            self.window_size_Option_place()
            general_shared_Buttons_place()
            stem_save_Options_place()
            no_ensemble_shared()
        elif self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
            self.demucs_model_Label_place()
            self.demucs_model_Option_place()
            self.demucs_stems_Label_place()
            self.demucs_stems_Option_place()
            self.segment_Label_place()
            self.segment_Option_place()
            general_shared_Buttons_place()
            stem_save_demucs_Options_place()
            no_ensemble_shared()
        elif self.chosen_process_method_var.get() == AUDIO_TOOLS:
            self.chosen_audio_tool_Label_place()
            self.chosen_audio_tool_Option_place()
            if self.chosen_audio_tool_var.get() == MANUAL_ENSEMBLE:
                self.choose_algorithm_Label_place()
                self.choose_algorithm_Option_place()
            elif self.chosen_audio_tool_var.get() == TIME_STRETCH:
                self.model_sample_mode_Option_place(rely=5)
                self.time_stretch_rate_Label_place()
                self.time_stretch_rate_Option_place()
            elif self.chosen_audio_tool_var.get() == CHANGE_PITCH:
                self.model_sample_mode_Option_place(rely=5)
                self.pitch_rate_Label_place()
                self.pitch_rate_Option_place()
        elif self.chosen_process_method_var.get() == ENSEMBLE_MODE:
            self.chosen_ensemble_Label_place()
            self.chosen_ensemble_Option_place()
            self.ensemble_main_stem_Label_place()
            self.ensemble_main_stem_Option_place()
            self.ensemble_type_Label_place()
            self.ensemble_type_Option_place()
            self.ensemble_listbox_Label_place()
            self.ensemble_listbox_Option_place()
            self.ensemble_listbox_Option_pack()
            general_shared_Buttons_place()
            stem_save_Options_place()

        self.is_gpu_conversion_Disable() if not self.is_gpu_available else None

        self.update_inputPaths()

    def update_button_states(self):
        """Updates the available stems for selected Demucs model"""
        
        if self.demucs_stems_var.get() == ALL_STEMS:
            self.update_stem_checkbox_labels(PRIMARY_STEM, demucs=True)
        elif self.demucs_stems_var.get() == VOCAL_STEM:
            self.update_stem_checkbox_labels(VOCAL_STEM, demucs=True, is_disable_demucs_boxes=False)
            self.is_stem_only_Demucs_Options_Enable()
        else:
            self.is_stem_only_Demucs_Options_Enable()

        self.demucs_stems_Option['menu'].delete(0,'end')
        
        if not self.demucs_model_var.get() == CHOOSE_MODEL:
            if DEMUCS_UVR_MODEL in self.demucs_model_var.get():
                stems = DEMUCS_2_STEM_OPTIONS
            elif DEMUCS_6_STEM_MODEL in self.demucs_model_var.get():
                stems = DEMUCS_6_STEM_OPTIONS
            else:
                stems = DEMUCS_4_STEM_OPTIONS

            for stem in stems:
                self.demucs_stems_Option['menu'].add_radiobutton(label=stem, 
                                                                 command=tk._setit(self.demucs_stems_var, stem, lambda s:self.update_stem_checkbox_labels(s, demucs=True)))

    def update_stem_checkbox_labels(self, selection, demucs=False, disable_boxes=False, is_disable_demucs_boxes=True):
        """Updates the "save only" checkboxes based on the model selected"""
        
        stem_text = self.is_primary_stem_only_Text_var, self.is_secondary_stem_only_Text_var
        
        if disable_boxes:
            self.is_primary_stem_only_Option.configure(state=tk.DISABLED)
            self.is_secondary_stem_only_Option.configure(state=tk.DISABLED)
            self.is_primary_stem_only_var.set(False)
            self.is_secondary_stem_only_var.set(False)
        
        if demucs:
            stem_text = self.is_primary_stem_only_Demucs_Text_var, self.is_secondary_stem_only_Demucs_Text_var
            if is_disable_demucs_boxes:
                self.is_primary_stem_only_Demucs_Option.configure(state=tk.DISABLED)
                self.is_secondary_stem_only_Demucs_Option.configure(state=tk.DISABLED)
                self.is_primary_stem_only_Demucs_var.set(False)
                self.is_secondary_stem_only_Demucs_var.set(False)

        for primary_stem, secondary_stem in STEM_PAIR_MAPPER.items():
            if selection == primary_stem:
                stem_text[0].set(f"{primary_stem} Only")
                stem_text[1].set(f"{secondary_stem} Only")
              
    def update_ensemble_algorithm_menu(self, is_4_stem=False):
        
        self.ensemble_type_Option['menu'].delete(0, 'end')
        options = ENSEMBLE_TYPE_4_STEM if is_4_stem else ENSEMBLE_TYPE

        if not "/" in self.ensemble_type_var.get() or is_4_stem: 
            self.ensemble_type_var.set(options[0])

        for choice in options:
            self.ensemble_type_Option['menu'].add_command(label=choice, command=tk._setit(self.ensemble_type_var, choice))
                                     
    def selection_action_models(self, selection):
        """Accepts model names and verifies their state"""

        if selection in DOWNLOAD_MORE:
            self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
            self.menu_settings(select_tab_3=True) if not self.is_menu_settings_open else None
            for method_type, model_var in self.method_mapper.items():
                if method_type == self.chosen_process_method_var.get():
                    model_var.set(CHOOSE_ENSEMBLE_OPTION) if method_type in ENSEMBLE_MODE else model_var.set(CHOOSE_MODEL)
        elif selection in CHOOSE_MODEL:
            self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
        else:
            self.is_stem_only_Options_Enable()

        for method_type, model_var in self.method_mapper.items():
            if method_type == self.chosen_process_method_var.get():
                self.selection_action_models_sub(selection, method_type, model_var)
                               
        if self.chosen_process_method_var.get() == ENSEMBLE_MODE:
            model_data = self.assemble_model_data(selection, ENSEMBLE_CHECK)[0]
            if not model_data.model_status:
                return self.model_stems_list.index(selection)
            else: 
                return False

    def selection_action_models_sub(self, selection, ai_type, var: tk.StringVar):
        """Takes input directly from the selection_action_models parent function"""
        
        model_data = self.assemble_model_data(selection, ai_type)[0]

        if not model_data.model_status:
            var.set(CHOOSE_MODEL)
            self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
        else:
            if ai_type == DEMUCS_ARCH_TYPE:
                if not self.demucs_stems_var.get().lower() in model_data.demucs_source_list:
                    self.demucs_stems_var.set(ALL_STEMS if model_data.demucs_stem_count == 4 else VOCAL_STEM)
            else:
                stem = model_data.primary_stem
                self.update_stem_checkbox_labels(stem)

    def selection_action_process_method(self, selection, from_widget=False):
        """Checks model and variable status when toggling between process methods"""
        
        if from_widget:
            self.save_current_settings_var.set(CHOOSE_ENSEMBLE_OPTION)
        
        if selection == ENSEMBLE_MODE:
            if self.ensemble_main_stem_var.get() in [CHOOSE_STEM_PAIR, FOUR_STEM_ENSEMBLE]:
                self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
            else:
                self.update_stem_checkbox_labels(self.return_ensemble_stems(is_primary=True))
                self.is_stem_only_Options_Enable()
        else:
            for method_type, model_var in self.method_mapper.items():
                if method_type in selection:
                    self.selection_action_models(model_var.get())
            
    def selection_action_chosen_ensemble(self, selection):
        """Activates specific actions depending on selected ensemble option"""
        
        if selection not in ENSEMBLE_OPTIONS:
            self.selection_action_chosen_ensemble_load_saved(selection)
        if selection == SAVE_ENSEMBLE: 
            self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION)
            self.pop_up_save_ensemble()
        if selection == MENU_SEPARATOR:
            self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION)
        if selection == CLEAR_ENSEMBLE:
            self.ensemble_listbox_Option.selection_clear(0, 'end') 
            self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION)
           
    def selection_action_chosen_ensemble_load_saved(self, saved_ensemble):
        """Loads the data from selected saved ensemble"""
        
        saved_data = None
        saved_ensemble = saved_ensemble.replace(" ", "_")
        saved_ensemble_path = os.path.join(ENSEMBLE_CACHE_DIR, f'{saved_ensemble}.json')

        if os.path.isfile(saved_ensemble_path):
            saved_data = json.load(open(saved_ensemble_path))
            
        if saved_data:
            self.selection_action_ensemble_stems(saved_data['ensemble_main_stem'], from_menu=False)
            self.ensemble_main_stem_var.set(saved_data['ensemble_main_stem'])
            self.ensemble_type_var.set(saved_data['ensemble_type'])
            self.saved_model_list = saved_data['selected_models']
        
            for saved_model in self.saved_model_list:         
                status = self.assemble_model_data(saved_model, ENSEMBLE_CHECK)[0].model_status
                if not status:
                    self.saved_model_list.remove(saved_model)
            
            indexes = self.ensemble_listbox_get_indexes_for_files(self.model_stems_list, self.saved_model_list)
            
            for i in indexes:
                self.ensemble_listbox_Option.selection_set(i)
            
    def selection_action_ensemble_stems(self, selection: str, from_menu=True, auto_update=None):
        """Filters out all models from ensemble listbox that are incompatible with selected ensemble stem"""
        
        if not selection == CHOOSE_STEM_PAIR:

            if selection == FOUR_STEM_ENSEMBLE:
                self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
                self.update_ensemble_algorithm_menu(is_4_stem=True)
                self.ensemble_primary_stem = PRIMARY_STEM
                self.ensemble_secondary_stem = SECONDARY_STEM
                is_4_stem_check = True
            else:
                self.update_ensemble_algorithm_menu()
                self.is_stem_only_Options_Enable()
                stems = selection.partition("/")
                self.update_stem_checkbox_labels(stems[0])
                self.ensemble_primary_stem = stems[0]
                self.ensemble_secondary_stem = stems[2]
                is_4_stem_check = False
            
            self.model_stems_list = self.model_list(self.ensemble_primary_stem, self.ensemble_secondary_stem, is_4_stem_check=is_4_stem_check)
            self.ensemble_listbox_Option.configure(state=tk.NORMAL)
            self.ensemble_listbox_clear_and_insert_new(self.model_stems_list)

            if auto_update:
                indexes = self.ensemble_listbox_get_indexes_for_files(self.model_stems_list, auto_update)
                self.ensemble_listbox_select_from_indexs(indexes)
        else:
            self.ensemble_listbox_Option.configure(state=tk.DISABLED)
            self.update_stem_checkbox_labels(PRIMARY_STEM, disable_boxes=True)
            self.model_stems_list = ()
                
        if from_menu:
            self.chosen_ensemble_var.set(CHOOSE_ENSEMBLE_OPTION)

    def selection_action_saved_settings(self, selection, process_method=None):
        """Activates specific action based on the selected settings from the saved settings selections"""
        
        if self.thread_check(self.active_processing_thread):
            self.error_dialoge(SET_TO_ANY_PROCESS_ERROR)
        else:
            saved_data = None
            chosen_process_method = self.chosen_process_method_var.get() if not process_method else process_method

            if selection not in SAVE_SET_OPTIONS:
                selection = selection.replace(" ", "_")
                saved_ensemble_path = os.path.join(SETTINGS_CACHE_DIR, f'{selection}.json')

                if os.path.isfile(saved_ensemble_path):
                    saved_data = json.load(open(saved_ensemble_path))
                    
                if saved_data:
                    self.load_saved_settings(saved_data, chosen_process_method)
                    
            if selection == SAVE_SETTINGS: 
                self.save_current_settings_var.set(SELECT_SAVED_SET)
                self.pop_up_save_current_settings()
                
            if selection == RESET_TO_DEFAULT:
                self.save_current_settings_var.set(SELECT_SAVED_SET)
                self.load_saved_settings(DEFAULT_DATA, chosen_process_method)
                
            self.update_checkbox_text()

    #--Processing Methods-- 

    def process_input_selections(self):
        """Grabbing all audio files from selected directories."""
        
        input_list = []

        ext = FFMPEG_EXT if not self.is_accept_any_input_var.get() else ANY_EXT

        for i in self.inputPaths:
            if os.path.isfile(i):
                if i.endswith(ext):
                    input_list.append(i)
            for root, dirs, files in os.walk(i):
                for file in files:
                    if file.endswith(ext):
                        file = os.path.join(root, file)
                        if os.path.isfile(file):
                            input_list.append(file)
                                         
        self.inputPaths = tuple(input_list)

    def process_preliminary_checks(self):
        """Verifies a valid model is chosen"""
        
        if self.wav_type_set_var.get() == '32-bit Float':
            self.wav_type_set = 'FLOAT'
        elif self.wav_type_set_var.get() == '64-bit Float':
            self.wav_type_set = 'FLOAT' if not self.save_format_var.get() == WAV else 'DOUBLE'
        else:
            self.wav_type_set = self.wav_type_set_var.get()
        
        if self.chosen_process_method_var.get() == ENSEMBLE_MODE:
            continue_process = lambda:False if len(self.ensemble_listbox_get_all_selected_models()) <= 1 else True
        if self.chosen_process_method_var.get() == VR_ARCH_PM:
            continue_process = lambda:False if self.vr_model_var.get() == CHOOSE_MODEL else True
        if self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
            continue_process = lambda:False if self.mdx_net_model_var.get() == CHOOSE_MODEL else True
        if self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
            continue_process = lambda:False if self.demucs_model_var.get() == CHOOSE_MODEL else True

        return continue_process()

    def process_storage_check(self):
        """Verifies storage requirments"""
        
        total, used, free = shutil.disk_usage("/") 
        
        space_details = f'Detected Total Space: {int(total/1.074e+9)} GB\'s\n' +\
                        f'Detected Used Space: {int(used/1.074e+9)} GB\'s\n' +\
                        f'Detected Free Space: {int(free/1.074e+9)} GB\'s\n'
            
        appropriate_storage = True
            
        if int(free/1.074e+9) <= int(2):
            self.error_dialoge([STORAGE_ERROR[0], f'{STORAGE_ERROR[1]}{space_details}'])
            appropriate_storage = False
        
        if int(free/1.074e+9) in [3, 4, 5, 6, 7, 8]:
            appropriate_storage = self.message_box([STORAGE_WARNING[0], f'{STORAGE_WARNING[1]}{space_details}{CONFIRM_WARNING}'])
                        
        return appropriate_storage

    def process_initialize(self):
        """Verifies the input/output directories are valid and prepares to thread the main process."""
                                
        if self.inputPaths:
            if not os.path.isfile(self.inputPaths[0]):
                self.error_dialoge(INVALID_INPUT)
                return
        else:
            self.error_dialoge(INVALID_INPUT)
            return
            
        if not os.path.isdir(self.export_path_var.get()):
            self.error_dialoge(INVALID_EXPORT)
            return
        
        if not self.process_storage_check():
            return
        
        if not self.chosen_process_method_var.get() == AUDIO_TOOLS:
            if not self.process_preliminary_checks():
                self.error_dialoge(INVALID_ENSEMBLE if self.chosen_process_method_var.get() == ENSEMBLE_MODE else INVALID_MODEL)
                return
            
            self.active_processing_thread = KThread(target=self.process_start)
            self.active_processing_thread.start()
        else:
            self.active_processing_thread = KThread(target=self.process_tool_start)
            self.active_processing_thread.start()

    def process_button_init(self):
        self.command_Text.clear()
        self.conversion_Button_Text_var.set(WAIT_PROCESSING)
        self.conversion_Button.configure(state=tk.DISABLED)

    def process_get_baseText(self, total_files, file_num):
        """Create the base text for the command widget"""
        
        text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                    total_files=total_files)
        
        return text

    def process_update_progress(self, model_count, total_files, step: float = 1):
        """Calculate the progress for the progress widget in the GUI"""
        
        total_count = model_count * total_files
        base = (100 / total_count)
        progress = base * self.iteration - base
        progress += base * step

        self.progress_bar_main_var.set(progress)
        
        self.conversion_Button_Text_var.set(f'Process Progress: {int(progress)}%')

    def confirm_stop_process(self):
        """Asks for confirmation before halting active process"""
        
        if self.thread_check(self.active_processing_thread):
            confirm = tk.messagebox.askyesno(title=STOP_PROCESS_CONFIRM[0],
                                             message=STOP_PROCESS_CONFIRM[1])

            if confirm:
                try:
                    self.active_processing_thread.terminate()
                finally:
                    self.is_process_stopped = True
                    self.command_Text.write('\n\nProcess stopped by user.')
        else:
            self.clear_cache_torch = True

    def process_end(self, error=None):
        """End of process actions"""
        
        self.cached_sources_clear()
        self.clear_cache_torch = True
        self.conversion_Button_Text_var.set(START_PROCESSING)
        self.conversion_Button.configure(state=tk.NORMAL)
        self.progress_bar_main_var.set(0)
        
        if error:
            error_message_box_text = f'{error_dialouge(error)}{ERROR_OCCURED[1]}'
            confirm = tk.messagebox.askyesno(master=self,
                                             title=ERROR_OCCURED[0],
                                             message=error_message_box_text)
            
            if confirm:
                self.is_confirm_error_var.set(True)
                self.clear_cache_torch = True

            self.clear_cache_torch = True
            
            if MODEL_MISSING_CHECK in error_message_box_text: 
                self.update_checkbox_text()

    def process_tool_start(self):
        """Start the conversion for all the given mp3 and wav files"""

        multiple_files = False
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
        self.process_button_init()
        inputPaths = self.inputPaths
        is_verified_audio = True
        is_model_sample_mode = self.model_sample_mode_var.get()

        try:
            total_files = len(inputPaths)

            if self.chosen_audio_tool_var.get() == TIME_STRETCH:
                audio_tool = AudioTools(TIME_STRETCH)
                self.progress_bar_main_var.set(2)
            if self.chosen_audio_tool_var.get() == CHANGE_PITCH:
                audio_tool = AudioTools(CHANGE_PITCH)
                self.progress_bar_main_var.set(2)
            if self.chosen_audio_tool_var.get() == MANUAL_ENSEMBLE:
                audio_tool = Ensembler(is_manual_ensemble=True)
                multiple_files = True
                if total_files <= 1:
                    self.command_Text.write("Not enough files to process.\n")
                    self.process_end()
                    return
            if self.chosen_audio_tool_var.get() == ALIGN_INPUTS:
                multiple_files = True
                audio_tool = AudioTools(ALIGN_INPUTS)
                if not total_files == 2:
                    self.command_Text.write("You must select exactly 2 inputs!\n")
                    self.process_end()
                    return
            
            for file_num, audio_file in enumerate(inputPaths, start=1):

                base = (100 / total_files)            
                
                if audio_tool.audio_tool in [MANUAL_ENSEMBLE, ALIGN_INPUTS]:
                    audio_file_base = f'{os.path.splitext(os.path.basename(inputPaths[0]))[0]}'
                    
                else:
                    audio_file_base = f'{os.path.splitext(os.path.basename(audio_file))[0]}'
                
                self.base_text = self.process_get_baseText(total_files=total_files, file_num=total_files if multiple_files else file_num)
                command_Text = lambda text:self.command_Text.write(self.base_text + text)
                
                if self.verify_audio(audio_file):
                    if not audio_tool.audio_tool in [MANUAL_ENSEMBLE, ALIGN_INPUTS]:
                        audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
                        self.command_Text.write(f'{NEW_LINE if not file_num ==1 else NO_LINE}{self.base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                    elif audio_tool.audio_tool == ALIGN_INPUTS:
                        self.command_Text.write('File 1 "{}"{}'.format(os.path.basename(inputPaths[0]), NEW_LINE))
                        self.command_Text.write('File 2 "{}"{}'.format(os.path.basename(inputPaths[1]), NEW_LINES))
                    elif audio_tool.audio_tool == MANUAL_ENSEMBLE:
                        for n, i in enumerate(inputPaths):
                            self.command_Text.write('File {} "{}"{}'.format(n+1, os.path.basename(i), NEW_LINE))
                        self.command_Text.write(NEW_LINE)
                        
                    is_verified_audio = True
                else:
                    error_text_console = f'{self.base_text}"{os.path.basename(audio_file)}\" is missing or currupted.\n'
                    self.command_Text.write(f'\n{error_text_console}') if total_files >= 2 else None
                    is_verified_audio = False
                    continue
                
                command_Text('Process starting... ') if not audio_tool.audio_tool == ALIGN_INPUTS else None
                
                if audio_tool.audio_tool == MANUAL_ENSEMBLE:
                    self.progress_bar_main_var.set(50)
                    audio_tool.ensemble_manual(inputPaths, audio_file_base)
                    self.progress_bar_main_var.set(100)
                    self.command_Text.write(DONE)
                    break
                if audio_tool.audio_tool == ALIGN_INPUTS:
                    command_Text('Process starting... \n')
                    audio_file_2_base = f'{os.path.splitext(os.path.basename(inputPaths[1]))[0]}'
                    audio_tool.align_inputs(inputPaths, audio_file_base, audio_file_2_base, command_Text)
                    self.command_Text.write(DONE)
                    break
                if audio_tool.audio_tool in [TIME_STRETCH, CHANGE_PITCH]:
                    audio_tool.pitch_or_time_shift(audio_file, audio_file_base)
                    self.progress_bar_main_var.set(base*file_num)
                    self.command_Text.write(DONE)
                
            if total_files == 1 and not is_verified_audio:
                self.command_Text.write(f'{error_text_console}\n{PROCESS_FAILED}')
                self.command_Text.write(time_elapsed())
                playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            else:
                self.command_Text.write('\nProcess complete\n{}'.format(time_elapsed()))
                playsound(COMPLETE_CHIME) if self.is_task_complete_var.get() else None

            self.process_end()

        except Exception as e:
            self.error_log_var.set(error_text(self.chosen_audio_tool_var.get(), e))
            self.command_Text.write(f'\n\n{PROCESS_FAILED}')
            self.command_Text.write(time_elapsed())
            playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            self.process_end(error=e)
        
    def process_determine_secondary_model(self, process_method, main_model_primary_stem, is_primary_stem_only=False, is_secondary_stem_only=False):
        """Obtains the correct secondary model data for conversion."""
        
        secondary_model_scale = None
        secondary_model = StringVar(value=NO_MODEL)
        
        if process_method == VR_ARCH_TYPE:
            secondary_model_vars = self.vr_secondary_model_vars
        if process_method == MDX_ARCH_TYPE:
            secondary_model_vars = self.mdx_secondary_model_vars
        if process_method == DEMUCS_ARCH_TYPE:
            secondary_model_vars = self.demucs_secondary_model_vars

        if main_model_primary_stem in [VOCAL_STEM, INST_STEM]:
            secondary_model = secondary_model_vars["voc_inst_secondary_model"]
            secondary_model_scale = secondary_model_vars["voc_inst_secondary_model_scale"].get()
        if main_model_primary_stem in [OTHER_STEM, NO_OTHER_STEM]:
            secondary_model = secondary_model_vars["other_secondary_model"]
            secondary_model_scale = secondary_model_vars["other_secondary_model_scale"].get()
        if main_model_primary_stem in [DRUM_STEM, NO_DRUM_STEM]:
            secondary_model = secondary_model_vars["drums_secondary_model"]
            secondary_model_scale = secondary_model_vars["drums_secondary_model_scale"].get()
        if main_model_primary_stem in [BASS_STEM, NO_BASS_STEM]:
            secondary_model = secondary_model_vars["bass_secondary_model"]
            secondary_model_scale = secondary_model_vars["bass_secondary_model_scale"].get()

        if secondary_model_scale:
           secondary_model_scale = float(secondary_model_scale)

        if not secondary_model.get() == NO_MODEL:
            secondary_model = ModelData(secondary_model.get(), 
                                        is_secondary_model=True, 
                                        primary_model_primary_stem=main_model_primary_stem, 
                                        is_primary_model_primary_stem_only=is_primary_stem_only, 
                                        is_primary_model_secondary_stem_only=is_secondary_stem_only)
            if not secondary_model.model_status:
                secondary_model = None
        else:
            secondary_model = None
            
        return secondary_model, secondary_model_scale
        
    def process_determine_demucs_pre_proc_model(self, primary_stem=None):
        """Obtains the correct secondary model data for conversion."""

        pre_proc_model = None

        if not self.demucs_pre_proc_model_var.get() == NO_MODEL and self.is_demucs_pre_proc_model_activate_var.get():
            pre_proc_model = ModelData(self.demucs_pre_proc_model_var.get(), 
                                        primary_model_primary_stem=primary_stem, 
                                        is_pre_proc_model=True)
            if not pre_proc_model.model_status:
                pre_proc_model = None
        else:
            pre_proc_model = None
            
        return pre_proc_model
        
    def process_start(self):
        """Start the conversion for all the given mp3 and wav files"""
        
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
        export_path = self.export_path_var.get()
        is_ensemble = False
        true_model_count = 0
        self.iteration = 0
        is_verified_audio = True
        self.process_button_init()
        inputPaths = self.inputPaths
        inputPath_total_len = len(inputPaths)
        is_model_sample_mode = self.model_sample_mode_var.get()
        
        try:
            if self.chosen_process_method_var.get() == ENSEMBLE_MODE:
                model, ensemble = self.assemble_model_data(), Ensembler()
                export_path, is_ensemble = ensemble.ensemble_folder_name, True
            if self.chosen_process_method_var.get() == VR_ARCH_PM:
                model = self.assemble_model_data(self.vr_model_var.get(), VR_ARCH_TYPE)
            if self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
                model = self.assemble_model_data(self.mdx_net_model_var.get(), MDX_ARCH_TYPE)
            if self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
                model = self.assemble_model_data(self.demucs_model_var.get(), DEMUCS_ARCH_TYPE)

            self.cached_source_model_list_check(model)
            
            true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
            true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
            true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count

            for file_num, audio_file in enumerate(inputPaths, start=1):
                self.cached_sources_clear()
                base_text = self.process_get_baseText(total_files=inputPath_total_len, file_num=file_num)

                if self.verify_audio(audio_file):
                    audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
                    self.command_Text.write(f'{NEW_LINE if not file_num ==1 else NO_LINE}{base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                    is_verified_audio = True
                else:
                    error_text_console = f'{base_text}"{os.path.basename(audio_file)}\" is missing or currupted.\n'
                    self.command_Text.write(f'\n{error_text_console}') if inputPath_total_len >= 2 else None
                    self.iteration += true_model_count
                    is_verified_audio = False
                    continue

                for current_model_num, current_model in enumerate(model, start=1):
                    self.iteration += 1
           
                    if is_ensemble:
                        self.command_Text.write(f'Ensemble Mode - {current_model.model_basename} - Model {current_model_num}/{len(model)}{NEW_LINES}')

                    model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
                    self.command_Text.write(base_text + f'Loading model {model_name_text}...')
                    
                    progress_kwargs = {'model_count': true_model_count,
                                       'total_files': inputPath_total_len}

                    set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(**progress_kwargs, step=(step + (inference_iterations)))
                    write_to_console = lambda progress_text, base_text=base_text:self.command_Text.write(base_text + progress_text)
                    
                    audio_file_base = f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}"
                    audio_file_base = audio_file_base if not self.is_testing_audio_var.get() or is_ensemble else f"{round(time.time())}_{audio_file_base}"
                    audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
                    audio_file_base = audio_file_base if not self.is_add_model_name_var.get() else f"{audio_file_base}_{current_model.model_basename}"

                    if self.is_create_model_folder_var.get() and not is_ensemble:
                        export_path = os.path.join(Path(self.export_path_var.get()), current_model.model_basename, os.path.splitext(os.path.basename(audio_file))[0])
                        if not os.path.isdir(export_path):os.makedirs(export_path) 

                    process_data = {
                                    'model_data': current_model, 
                                    'export_path': export_path,
                                    'audio_file_base': audio_file_base,
                                    'audio_file': audio_file,
                                    'set_progress_bar': set_progress_bar,
                                    'write_to_console': write_to_console,
                                    'process_iteration': self.process_iteration,
                                    'cached_source_callback': self.cached_source_callback,
                                    'cached_model_source_holder': self.cached_model_source_holder,
                                    'list_all_models': self.all_models,
                                    'is_ensemble_master': is_ensemble,
                                    'is_4_stem_ensemble': True if self.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE and is_ensemble else False}

                    if current_model.process_method == VR_ARCH_TYPE:
                        seperator = SeperateVR(current_model, process_data)
                    if current_model.process_method == MDX_ARCH_TYPE:
                        seperator = SeperateMDX(current_model, process_data)
                    if current_model.process_method == DEMUCS_ARCH_TYPE:
                        seperator = SeperateDemucs(current_model, process_data)

                    seperator.seperate()
                    
                    if is_ensemble:
                        self.command_Text.write('\n')

                if is_ensemble:
                    
                    audio_file_base = audio_file_base.replace(f"_{current_model.model_basename}","")
                    self.command_Text.write(base_text + ENSEMBLING_OUTPUTS)
                    
                    if self.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE:
                        for output_stem in DEMUCS_4_SOURCE_LIST:
                            ensemble.ensemble_outputs(audio_file_base, export_path, output_stem, is_4_stem=True)
                    else:
                        if not self.is_secondary_stem_only_var.get():
                            ensemble.ensemble_outputs(audio_file_base, export_path, PRIMARY_STEM)
                        if not self.is_primary_stem_only_var.get():
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM)
                            ensemble.ensemble_outputs(audio_file_base, export_path, SECONDARY_STEM, is_inst_mix=True)

                    self.command_Text.write(DONE)
                    
                if is_model_sample_mode:
                    if os.path.isfile(audio_file):
                        os.remove(audio_file)
                    
                torch.cuda.empty_cache()
                
            shutil.rmtree(export_path) if is_ensemble and len(os.listdir(export_path)) == 0 else None

            if inputPath_total_len == 1 and not is_verified_audio:
                self.command_Text.write(f'{error_text_console}\n{PROCESS_FAILED}')
                self.command_Text.write(time_elapsed())
                playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            else:
                set_progress_bar(1.0)
                self.command_Text.write('\nProcess Complete\n')
                self.command_Text.write(time_elapsed())
                playsound(COMPLETE_CHIME) if self.is_task_complete_var.get() else None
                
            self.process_end()
                
        except Exception as e:
            self.error_log_var.set("{}{}".format(error_text(self.chosen_process_method_var.get(), e), self.get_settings_list()))
            self.command_Text.write(f'\n\n{PROCESS_FAILED}')
            self.command_Text.write(time_elapsed())
            playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            self.process_end(error=e)

    #--Varible Methods--

    def load_to_default_confirm(self):
        """Reset settings confirmation after asking for confirmation"""
        
        if self.thread_check(self.active_processing_thread):
            self.error_dialoge(SET_TO_DEFAULT_PROCESS_ERROR)
        else:
            confirm = tk.messagebox.askyesno(title=RESET_ALL_TO_DEFAULT_WARNING[0],
                                             message=RESET_ALL_TO_DEFAULT_WARNING[1])
            
            if confirm:
                self.load_saved_settings(DEFAULT_DATA)
  
    def load_saved_vars(self, data):
        """Initializes primary Tkinter vars"""
        
        ## ADD_BUTTON
        self.chosen_process_method_var = tk.StringVar(value=data['chosen_process_method'])
        
        #VR Architecture Vars
        self.vr_model_var = tk.StringVar(value=data['vr_model'])
        self.aggression_setting_var = tk.StringVar(value=data['aggression_setting'])
        self.window_size_var = tk.StringVar(value=data['window_size'])
        self.batch_size_var = tk.StringVar(value=data['batch_size'])
        self.crop_size_var = tk.StringVar(value=data['crop_size'])
        self.is_tta_var = tk.BooleanVar(value=data['is_tta'])
        self.is_output_image_var = tk.BooleanVar(value=data['is_output_image'])
        self.is_post_process_var = tk.BooleanVar(value=data['is_post_process'])
        self.is_high_end_process_var = tk.BooleanVar(value=data['is_high_end_process'])
        self.vr_voc_inst_secondary_model_var = tk.StringVar(value=data['vr_voc_inst_secondary_model'])
        self.vr_other_secondary_model_var = tk.StringVar(value=data['vr_other_secondary_model'])
        self.vr_bass_secondary_model_var = tk.StringVar(value=data['vr_bass_secondary_model'])
        self.vr_drums_secondary_model_var = tk.StringVar(value=data['vr_drums_secondary_model'])
        self.vr_is_secondary_model_activate_var = tk.BooleanVar(value=data['vr_is_secondary_model_activate'])
        self.vr_voc_inst_secondary_model_scale_var = tk.StringVar(value=data['vr_voc_inst_secondary_model_scale'])
        self.vr_other_secondary_model_scale_var = tk.StringVar(value=data['vr_other_secondary_model_scale'])
        self.vr_bass_secondary_model_scale_var = tk.StringVar(value=data['vr_bass_secondary_model_scale'])
        self.vr_drums_secondary_model_scale_var = tk.StringVar(value=data['vr_drums_secondary_model_scale'])

        #Demucs Vars
        self.demucs_model_var = tk.StringVar(value=data['demucs_model'])
        self.segment_var = tk.StringVar(value=data['segment'])
        self.overlap_var = tk.StringVar(value=data['overlap'])
        self.shifts_var = tk.StringVar(value=data['shifts'])
        self.chunks_demucs_var = tk.StringVar(value=data['chunks_demucs'])
        self.margin_demucs_var = tk.StringVar(value=data['margin_demucs'])
        self.is_chunk_demucs_var = tk.BooleanVar(value=data['is_chunk_demucs'])
        self.is_primary_stem_only_Demucs_var = tk.BooleanVar(value=data['is_primary_stem_only_Demucs'])
        self.is_secondary_stem_only_Demucs_var = tk.BooleanVar(value=data['is_secondary_stem_only_Demucs'])
        self.is_split_mode_var = tk.BooleanVar(value=data['is_split_mode'])
        self.is_demucs_combine_stems_var = tk.BooleanVar(value=data['is_demucs_combine_stems'])
        self.demucs_voc_inst_secondary_model_var = tk.StringVar(value=data['demucs_voc_inst_secondary_model'])
        self.demucs_other_secondary_model_var = tk.StringVar(value=data['demucs_other_secondary_model'])
        self.demucs_bass_secondary_model_var = tk.StringVar(value=data['demucs_bass_secondary_model'])
        self.demucs_drums_secondary_model_var = tk.StringVar(value=data['demucs_drums_secondary_model'])
        self.demucs_is_secondary_model_activate_var = tk.BooleanVar(value=data['demucs_is_secondary_model_activate'])
        self.demucs_voc_inst_secondary_model_scale_var = tk.StringVar(value=data['demucs_voc_inst_secondary_model_scale'])
        self.demucs_other_secondary_model_scale_var = tk.StringVar(value=data['demucs_other_secondary_model_scale'])
        self.demucs_bass_secondary_model_scale_var = tk.StringVar(value=data['demucs_bass_secondary_model_scale'])
        self.demucs_drums_secondary_model_scale_var = tk.StringVar(value=data['demucs_drums_secondary_model_scale'])
        self.demucs_pre_proc_model_var = tk.StringVar(value=data['demucs_pre_proc_model'])
        self.is_demucs_pre_proc_model_activate_var = tk.BooleanVar(value=data['is_demucs_pre_proc_model_activate'])
        self.is_demucs_pre_proc_model_inst_mix_var = tk.BooleanVar(value=data['is_demucs_pre_proc_model_inst_mix'])
        
        #MDX-Net Vars
        self.mdx_net_model_var = tk.StringVar(value=data['mdx_net_model'])
        self.chunks_var = tk.StringVar(value=data['chunks'])
        self.margin_var = tk.StringVar(value=data['margin'])
        self.compensate_var = tk.StringVar(value=data['compensate'])
        self.is_denoise_var = tk.BooleanVar(value=data['is_denoise'])
        self.is_invert_spec_var = tk.BooleanVar(value=data['is_invert_spec'])
        self.mdx_voc_inst_secondary_model_var = tk.StringVar(value=data['mdx_voc_inst_secondary_model'])
        self.mdx_other_secondary_model_var = tk.StringVar(value=data['mdx_other_secondary_model'])
        self.mdx_bass_secondary_model_var = tk.StringVar(value=data['mdx_bass_secondary_model'])
        self.mdx_drums_secondary_model_var = tk.StringVar(value=data['mdx_drums_secondary_model'])
        self.mdx_is_secondary_model_activate_var = tk.BooleanVar(value=data['mdx_is_secondary_model_activate'])
        self.mdx_voc_inst_secondary_model_scale_var = tk.StringVar(value=data['mdx_voc_inst_secondary_model_scale'])
        self.mdx_other_secondary_model_scale_var = tk.StringVar(value=data['mdx_other_secondary_model_scale'])
        self.mdx_bass_secondary_model_scale_var = tk.StringVar(value=data['mdx_bass_secondary_model_scale'])
        self.mdx_drums_secondary_model_scale_var = tk.StringVar(value=data['mdx_drums_secondary_model_scale'])
    
        #Ensemble Vars
        self.is_save_all_outputs_ensemble_var = tk.BooleanVar(value=data['is_save_all_outputs_ensemble'])
        self.is_append_ensemble_name_var = tk.BooleanVar(value=data['is_append_ensemble_name'])

        #Audio Tool Vars
        self.chosen_audio_tool_var = tk.StringVar(value=data['chosen_audio_tool'])
        self.choose_algorithm_var = tk.StringVar(value=data['choose_algorithm'])
        self.time_stretch_rate_var = tk.StringVar(value=data['time_stretch_rate'])
        self.pitch_rate_var = tk.StringVar(value=data['pitch_rate'])

        #Shared Vars
        self.mp3_bit_set_var = tk.StringVar(value=data['mp3_bit_set'])
        self.save_format_var = tk.StringVar(value=data['save_format'])
        self.wav_type_set_var = tk.StringVar(value=data['wav_type_set'])
        self.user_code_var = tk.StringVar(value=data['user_code']) 
        self.is_gpu_conversion_var = tk.BooleanVar(value=data['is_gpu_conversion'])
        self.is_primary_stem_only_var = tk.BooleanVar(value=data['is_primary_stem_only'])
        self.is_secondary_stem_only_var = tk.BooleanVar(value=data['is_secondary_stem_only'])
        self.is_testing_audio_var = tk.BooleanVar(value=data['is_testing_audio'])
        self.is_add_model_name_var = tk.BooleanVar(value=data['is_add_model_name'])
        self.is_accept_any_input_var = tk.BooleanVar(value=data['is_accept_any_input'])
        self.is_task_complete_var = tk.BooleanVar(value=data['is_task_complete'])
        self.is_normalization_var = tk.BooleanVar(value=data['is_normalization'])
        self.is_create_model_folder_var = tk.BooleanVar(value=data['is_create_model_folder'])
        self.help_hints_var = tk.BooleanVar(value=data['help_hints_var'])
        self.model_sample_mode_var = tk.BooleanVar(value=data['model_sample_mode'])
        self.model_sample_mode_duration_var = tk.StringVar(value=data['model_sample_mode_duration'])
        self.model_sample_mode_duration_checkbox_var = tk.StringVar(value=SAMPLE_MODE_CHECKBOX(self.model_sample_mode_duration_var.get()))
        
        #Path Vars
        self.export_path_var = tk.StringVar(value=data['export_path'])
        self.inputPaths = data['input_paths']
        self.lastDir = data['lastDir']
   
    def load_saved_settings(self, loaded_setting: dict, process_method=None):
        """Loads user saved application settings or resets to default"""
        
        if not process_method or process_method == VR_ARCH_PM:
            self.vr_model_var.set(loaded_setting['vr_model'])
            self.aggression_setting_var.set(loaded_setting['aggression_setting'])
            self.window_size_var.set(loaded_setting['window_size'])
            self.batch_size_var.set(loaded_setting['batch_size'])
            self.crop_size_var.set(loaded_setting['crop_size'])
            self.is_tta_var.set(loaded_setting['is_tta'])
            self.is_output_image_var.set(loaded_setting['is_output_image'])
            self.is_post_process_var.set(loaded_setting['is_post_process'])
            self.is_high_end_process_var.set(loaded_setting['is_high_end_process'])
            self.vr_voc_inst_secondary_model_var.set(loaded_setting['vr_voc_inst_secondary_model'])
            self.vr_other_secondary_model_var.set(loaded_setting['vr_other_secondary_model'])
            self.vr_bass_secondary_model_var.set(loaded_setting['vr_bass_secondary_model'])
            self.vr_drums_secondary_model_var.set(loaded_setting['vr_drums_secondary_model'])
            self.vr_is_secondary_model_activate_var.set(loaded_setting['vr_is_secondary_model_activate'])
            self.vr_voc_inst_secondary_model_scale_var.set(loaded_setting['vr_voc_inst_secondary_model_scale'])
            self.vr_other_secondary_model_scale_var.set(loaded_setting['vr_other_secondary_model_scale'])
            self.vr_bass_secondary_model_scale_var.set(loaded_setting['vr_bass_secondary_model_scale'])
            self.vr_drums_secondary_model_scale_var.set(loaded_setting['vr_drums_secondary_model_scale'])
        
        if not process_method or process_method == DEMUCS_ARCH_TYPE:
            self.demucs_model_var.set(loaded_setting['demucs_model'])
            self.segment_var.set(loaded_setting['segment'])
            self.overlap_var.set(loaded_setting['overlap'])
            self.shifts_var.set(loaded_setting['shifts'])
            self.chunks_demucs_var.set(loaded_setting['chunks_demucs'])
            self.margin_demucs_var.set(loaded_setting['margin_demucs'])
            self.is_chunk_demucs_var.set(loaded_setting['is_chunk_demucs'])
            self.is_primary_stem_only_Demucs_var.set(loaded_setting['is_primary_stem_only_Demucs'])
            self.is_secondary_stem_only_Demucs_var.set(loaded_setting['is_secondary_stem_only_Demucs'])
            self.is_split_mode_var.set(loaded_setting['is_split_mode'])
            self.is_demucs_combine_stems_var.set(loaded_setting['is_demucs_combine_stems'])
            self.demucs_voc_inst_secondary_model_var.set(loaded_setting['demucs_voc_inst_secondary_model'])
            self.demucs_other_secondary_model_var.set(loaded_setting['demucs_other_secondary_model'])
            self.demucs_bass_secondary_model_var.set(loaded_setting['demucs_bass_secondary_model'])
            self.demucs_drums_secondary_model_var.set(loaded_setting['demucs_drums_secondary_model'])
            self.demucs_is_secondary_model_activate_var.set(loaded_setting['demucs_is_secondary_model_activate'])
            self.demucs_voc_inst_secondary_model_scale_var.set(loaded_setting['demucs_voc_inst_secondary_model_scale'])
            self.demucs_other_secondary_model_scale_var.set(loaded_setting['demucs_other_secondary_model_scale'])
            self.demucs_bass_secondary_model_scale_var.set(loaded_setting['demucs_bass_secondary_model_scale'])
            self.demucs_drums_secondary_model_scale_var.set(loaded_setting['demucs_drums_secondary_model_scale'])
            self.demucs_stems_var.set(loaded_setting['demucs_stems'])
            self.update_stem_checkbox_labels(self.demucs_stems_var.get(), demucs=True)
            self.demucs_pre_proc_model_var.set(data['demucs_pre_proc_model'])
            self.is_demucs_pre_proc_model_activate_var.set(data['is_demucs_pre_proc_model_activate'])
            self.is_demucs_pre_proc_model_inst_mix_var.set(data['is_demucs_pre_proc_model_inst_mix'])
        
        if not process_method or process_method == MDX_ARCH_TYPE:
            self.mdx_net_model_var.set(loaded_setting['mdx_net_model'])
            self.chunks_var.set(loaded_setting['chunks'])
            self.margin_var.set(loaded_setting['margin'])
            self.compensate_var.set(loaded_setting['compensate'])
            self.is_denoise_var.set(loaded_setting['is_denoise'])
            self.is_invert_spec_var.set(loaded_setting['is_invert_spec'])
            self.mdx_voc_inst_secondary_model_var.set(loaded_setting['mdx_voc_inst_secondary_model'])
            self.mdx_other_secondary_model_var.set(loaded_setting['mdx_other_secondary_model'])
            self.mdx_bass_secondary_model_var.set(loaded_setting['mdx_bass_secondary_model'])
            self.mdx_drums_secondary_model_var.set(loaded_setting['mdx_drums_secondary_model'])
            self.mdx_is_secondary_model_activate_var.set(loaded_setting['mdx_is_secondary_model_activate'])
            self.mdx_voc_inst_secondary_model_scale_var.set(loaded_setting['mdx_voc_inst_secondary_model_scale'])
            self.mdx_other_secondary_model_scale_var.set(loaded_setting['mdx_other_secondary_model_scale'])
            self.mdx_bass_secondary_model_scale_var.set(loaded_setting['mdx_bass_secondary_model_scale'])
            self.mdx_drums_secondary_model_scale_var.set(loaded_setting['mdx_drums_secondary_model_scale'])
        
        if not process_method:
            self.is_save_all_outputs_ensemble_var.set(loaded_setting['is_save_all_outputs_ensemble'])
            self.is_append_ensemble_name_var.set(loaded_setting['is_append_ensemble_name'])
            self.chosen_audio_tool_var.set(loaded_setting['chosen_audio_tool'])
            self.choose_algorithm_var.set(loaded_setting['choose_algorithm'])
            self.time_stretch_rate_var.set(loaded_setting['time_stretch_rate'])
            self.pitch_rate_var.set(loaded_setting['pitch_rate'])
            self.is_primary_stem_only_var.set(loaded_setting['is_primary_stem_only'])
            self.is_secondary_stem_only_var.set(loaded_setting['is_secondary_stem_only'])
            self.is_testing_audio_var.set(loaded_setting['is_testing_audio'])
            self.is_add_model_name_var.set(loaded_setting['is_add_model_name'])
            self.is_accept_any_input_var.set(loaded_setting["is_accept_any_input"])
            self.is_task_complete_var.set(loaded_setting['is_task_complete'])
            self.is_create_model_folder_var.set(loaded_setting['is_create_model_folder'])
            self.mp3_bit_set_var.set(loaded_setting['mp3_bit_set'])
            self.save_format_var.set(loaded_setting['save_format'])
            self.wav_type_set_var.set(loaded_setting['wav_type_set'])
            self.user_code_var.set(loaded_setting['user_code'])
            
        self.is_gpu_conversion_var.set(loaded_setting['is_gpu_conversion'])
        self.is_normalization_var.set(loaded_setting['is_normalization'])
        self.help_hints_var.set(loaded_setting['help_hints_var'])
        
        self.model_sample_mode_var.set(loaded_setting['model_sample_mode'])
        self.model_sample_mode_duration_var.set(loaded_setting['model_sample_mode_duration'])
        self.model_sample_mode_duration_checkbox_var.set(SAMPLE_MODE_CHECKBOX(self.model_sample_mode_duration_var.get()))
        
    def save_values(self, app_close=True):
        """Saves application data"""

        # -Save Data-
        main_settings={
            'vr_model': self.vr_model_var.get(),
            'aggression_setting': self.aggression_setting_var.get(),
            'window_size': self.window_size_var.get(),
            'batch_size': self.batch_size_var.get(),
            'crop_size': self.crop_size_var.get(),
            'is_tta': self.is_tta_var.get(),
            'is_output_image': self.is_output_image_var.get(),
            'is_post_process': self.is_post_process_var.get(),
            'is_high_end_process': self.is_high_end_process_var.get(),
            'vr_voc_inst_secondary_model': self.vr_voc_inst_secondary_model_var.get(),
            'vr_other_secondary_model': self.vr_other_secondary_model_var.get(),
            'vr_bass_secondary_model': self.vr_bass_secondary_model_var.get(),
            'vr_drums_secondary_model': self.vr_drums_secondary_model_var.get(),
            'vr_is_secondary_model_activate': self.vr_is_secondary_model_activate_var.get(),
            'vr_voc_inst_secondary_model_scale': self.vr_voc_inst_secondary_model_scale_var.get(),
            'vr_other_secondary_model_scale': self.vr_other_secondary_model_scale_var.get(),
            'vr_bass_secondary_model_scale': self.vr_bass_secondary_model_scale_var.get(),
            'vr_drums_secondary_model_scale': self.vr_drums_secondary_model_scale_var.get(),
            'demucs_model': self.demucs_model_var.get(),
            'segment': self.segment_var.get(),
            'overlap': self.overlap_var.get(),
            'shifts': self.shifts_var.get(),
            'chunks_demucs': self.chunks_demucs_var.get(),
            'margin_demucs': self.margin_demucs_var.get(),
            'is_chunk_demucs': self.is_chunk_demucs_var.get(),
            'is_primary_stem_only_Demucs': self.is_primary_stem_only_Demucs_var.get(),
            'is_secondary_stem_only_Demucs': self.is_secondary_stem_only_Demucs_var.get(),
            'is_split_mode': self.is_split_mode_var.get(),
            'is_demucs_combine_stems': self.is_demucs_combine_stems_var.get(),
            'demucs_voc_inst_secondary_model': self.demucs_voc_inst_secondary_model_var.get(),
            'demucs_other_secondary_model': self.demucs_other_secondary_model_var.get(),
            'demucs_bass_secondary_model': self.demucs_bass_secondary_model_var.get(),
            'demucs_drums_secondary_model': self.demucs_drums_secondary_model_var.get(),
            'demucs_is_secondary_model_activate': self.demucs_is_secondary_model_activate_var.get(),
            'demucs_voc_inst_secondary_model_scale': self.demucs_voc_inst_secondary_model_scale_var.get(),
            'demucs_other_secondary_model_scale': self.demucs_other_secondary_model_scale_var.get(),
            'demucs_bass_secondary_model_scale': self.demucs_bass_secondary_model_scale_var.get(),
            'demucs_drums_secondary_model_scale': self.demucs_drums_secondary_model_scale_var.get(),
            'demucs_pre_proc_model': self.demucs_pre_proc_model_var.get(),
            'is_demucs_pre_proc_model_activate': self.is_demucs_pre_proc_model_activate_var.get(),
            'is_demucs_pre_proc_model_inst_mix': self.is_demucs_pre_proc_model_inst_mix_var.get(),
            'mdx_net_model': self.mdx_net_model_var.get(),
            'chunks': self.chunks_var.get(),
            'margin': self.margin_var.get(),
            'compensate': self.compensate_var.get(),
            'is_denoise': self.is_denoise_var.get(),
            'is_invert_spec': self.is_invert_spec_var.get(), 
            'mdx_voc_inst_secondary_model': self.mdx_voc_inst_secondary_model_var.get(),
            'mdx_other_secondary_model': self.mdx_other_secondary_model_var.get(),
            'mdx_bass_secondary_model': self.mdx_bass_secondary_model_var.get(),
            'mdx_drums_secondary_model': self.mdx_drums_secondary_model_var.get(),
            'mdx_is_secondary_model_activate': self.mdx_is_secondary_model_activate_var.get(),
            'mdx_voc_inst_secondary_model_scale': self.mdx_voc_inst_secondary_model_scale_var.get(),
            'mdx_other_secondary_model_scale': self.mdx_other_secondary_model_scale_var.get(),
            'mdx_bass_secondary_model_scale': self.mdx_bass_secondary_model_scale_var.get(),
            'mdx_drums_secondary_model_scale': self.mdx_drums_secondary_model_scale_var.get(),
            'is_save_all_outputs_ensemble': self.is_save_all_outputs_ensemble_var.get(),
            'is_append_ensemble_name': self.is_append_ensemble_name_var.get(),
            'chosen_audio_tool': self.chosen_audio_tool_var.get(),
            'choose_algorithm': self.choose_algorithm_var.get(),
            'time_stretch_rate': self.time_stretch_rate_var.get(),
            'pitch_rate': self.pitch_rate_var.get(),
            'is_gpu_conversion': self.is_gpu_conversion_var.get(),
            'is_primary_stem_only': self.is_primary_stem_only_var.get(),
            'is_secondary_stem_only': self.is_secondary_stem_only_var.get(),
            'is_testing_audio': self.is_testing_audio_var.get(),
            'is_add_model_name': self.is_add_model_name_var.get(),
            'is_accept_any_input': self.is_add_model_name_var.get(),
            'is_task_complete': self.is_task_complete_var.get(),
            'is_normalization': self.is_normalization_var.get(),
            'is_create_model_folder': self.is_create_model_folder_var.get(),
            'mp3_bit_set': self.mp3_bit_set_var.get(),
            'save_format': self.save_format_var.get(),
            'wav_type_set': self.wav_type_set_var.get(),
            'user_code': self.user_code_var.get(),
            'help_hints_var': self.help_hints_var.get(),
            'model_sample_mode': self.model_sample_mode_var.get(),
            'model_sample_mode_duration': self.model_sample_mode_duration_var.get()
            }

        other_data = {
            'chosen_process_method': self.chosen_process_method_var.get(),
            'input_paths': self.inputPaths,
            'lastDir': self.lastDir,
            'export_path': self.export_path_var.get(),
            'model_hash_table': model_hash_table,
        }

        user_saved_extras = {
            'demucs_stems': self.demucs_stems_var.get()}

        if app_close:
            save_data(data={**main_settings, **other_data})
            
            if self.thread_check(self.active_download_thread):
                self.error_dialoge(EXIT_DOWNLOAD_ERROR)
                return

            if self.thread_check(self.active_processing_thread):
                if self.is_process_stopped: 
                    self.error_dialoge(EXIT_HALTED_PROCESS_ERROR)
                else:
                    self.error_dialoge(EXIT_PROCESS_ERROR)
                return
            
            remove_temps(ENSEMBLE_TEMP_PATH)
            remove_temps(SAMPLE_CLIP_PATH)
            self.delete_temps()
            self.destroy()
            
        else:
            return {**main_settings, **user_saved_extras}

    def get_settings_list(self):
        
        settings_dict = self.save_values(app_close=False)
        settings_list = '\n'.join(''.join(f"{key}: {value}") for key, value in settings_dict.items() if not key == 'user_code')

        return f"\nFull Application Settings:\n\n{settings_list}"
        
def secondary_stem(stem):
    """Determines secondary stem"""
    
    for key, value in STEM_PAIR_MAPPER.items():
        if stem in key:
            secondary_stem = value
    
    return secondary_stem

def vip_downloads(password, link_type=VIP_REPO):
    """Attempts to decrypt VIP model link with given input code"""
    
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=link_type[0],
            iterations=390000,)

        key = base64.urlsafe_b64encode(kdf.derive(bytes(password, 'utf-8')))
        f = Fernet(key)

        return str(f.decrypt(link_type[1]), 'UTF-8')
    except Exception:
        return NO_CODE

if __name__ == "__main__":

    try:
        from ctypes import windll, wintypes
        windll.user32.SetThreadDpiAwarenessContext(wintypes.HANDLE(-1))
    except Exception as e:
        print(e)

    root = MainWindow()
        
    root.update_checkbox_text()

    root.mainloop()

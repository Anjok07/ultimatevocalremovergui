# GUI modules
import os
import glob
from re import T
import pyperclip
import natsort
from gc import freeze
import tkinter as tk
from tkinter import *
from tkinter.tix import *
import webbrowser
from tracemalloc import stop
import lib_v5.sv_ttk
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.font
from tkinterdnd2 import TkinterDnD, DND_FILES  # Enable Drag & Drop
import pyglet,tkinter
from datetime import datetime
# Images
from PIL import Image
from PIL import ImageTk
import pickle  # Save Data
from pathlib import Path
import hashlib
import wget
import time
import ctypes
import trace
import zipfile
import traceback
import torch
# Pathfinding
import pathlib
import sys
import subprocess
from collections import defaultdict
# Used for live text displaying
import queue
import threading  # Run the algorithm inside a thread
from subprocess import call
from pathlib import Path
import ctypes as ct
import subprocess  # Run python file
import inference_MDX
import inference_v5
import inference_v5_ensemble
import inference_demucs
import lib_v5.filelist
import shutil
import importlib
import urllib.request
import pyAesCrypt
from __version__ import VERSION
from win32api import GetSystemMetrics

try:
    with open(os.path.join(os.getcwd(), 'tmp', 'splash.txt'), 'w') as f:
        f.write('1')
except:
    pass

class KThread(threading.Thread):
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False
  def start(self):
    self.__run_backup = self.run
    self.run = self.__run     
    threading.Thread.start(self)
  def __run(self):
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup
  def globaltrace(self, frame, why, arg):
    if why == 'call':
      return self.localtrace
    else:
      return None
  def localtrace(self, frame, why, arg):
    if self.killed:
      if why == 'line':
        raise SystemExit()
    return self.localtrace
  def kill(self):
    self.killed = True

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(base_path)  # Change the current working directory to the base path

demucs_v3_repo_folder_path = 'models/Demucs_Models'
demucs_v3_repo_folder_path_b = 'models/Demucs_Models/v3_repo'

if not os.path.isdir(demucs_v3_repo_folder_path):
    os.mkdir(demucs_v3_repo_folder_path)
    
if not os.path.isdir(demucs_v3_repo_folder_path_b):
    os.mkdir(demucs_v3_repo_folder_path_b)

try:
    shutil.move("models/Demucs_Models/5d2d6c55-db83574e.th", "models/Demucs_Models/v3_repo/5d2d6c55-db83574e.th")
    shutil.move("models/Demucs_Models/7fd6ef75-a905dd85.th", "models/Demucs_Models/v3_repo/7fd6ef75-a905dd85.th")
    shutil.move("models/Demucs_Models/14fc6a69-a89dd0ee.th", "models/Demucs_Models/v3_repo/14fc6a69-a89dd0ee.th")
    shutil.move("models/Demucs_Models/83fc094f-4a16d450.th", "models/Demucs_Models/v3_repo/83fc094f-4a16d450.th")
    shutil.move("models/Demucs_Models/464b36d7-e5a9386e.th", "models/Demucs_Models/v3_repo/464b36d7-e5a9386e.th")
    shutil.move("models/Demucs_Models/a1d90b5c-ae9d2452.th", "models/Demucs_Models/v3_repo/a1d90b5c-ae9d2452.th")
    shutil.move("models/Demucs_Models/cfa93e08-61801ae1.th", "models/Demucs_Models/v3_repo/cfa93e08-61801ae1.th")
    shutil.move("models/Demucs_Models/e51eebcc-c1b80bdd.th", "models/Demucs_Models/v3_repo/e51eebcc-c1b80bdd.th")
    shutil.move("models/Demucs_Models/ebf34a2d.th", "models/Demucs_Models/v3_repo/ebf34a2d.th")
    shutil.move("models/Demucs_Models/ebf34a2db.th", "models/Demucs_Models/v3_repo/ebf34a2db.th")
    print('Demucs v3 models have been moved to the correct directory.')
except:
    pass

try:
    srcdir = "models/Demucs_Models/v3_repo"
    dstdir = "models/Demucs_Models"

    for basename in os.listdir(srcdir):
        if basename.endswith('.yaml'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                shutil.copy2(pathname, dstdir)
except:
    pass

try:
    srcdir = "models/Demucs_Models"
    dstdir = "models/Demucs_Models/v3_repo"

    for basename in os.listdir(srcdir):
        if basename.endswith('.yaml'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                shutil.copy2(pathname, dstdir)
except:
    pass     

try:
    srcdir = "models/Demucs_Models"

    for basename in os.listdir(srcdir):
        if basename.endswith('.tmp'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                os.remove(pathname)
except:
    pass   

try:
    srcdir = "models/Main_Models"

    for basename in os.listdir(srcdir):
        if basename.endswith('.tmp'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                os.remove(pathname)
except:
    pass  

try:
    srcdir = "models/MDX_Net_Models"

    for basename in os.listdir(srcdir):
        if basename.endswith('.tmp'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                os.remove(pathname)
except:
    pass  

try:
    with open('lib_v5/filelists/ensemble_list/mdx_demuc_en_list.txt', 'w') as f:
        f.write('No Model\nNo Model\n')
except:
    pass

try:
    srcdir = "models/Demucs_Models/v3_repo"

    for basename in os.listdir(srcdir):
        if basename.endswith('.tmp'):
            pathname = os.path.join(srcdir, basename)
            if os.path.isfile(pathname):
                os.remove(pathname)
except:
    pass   

try:
    os.rename("models/Main_Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth", "models/Main_Models/MGM_HIGHEND_v4.pth")
    os.rename("models/Main_Models/MGM_LOWEND_A_v4_sr32000_hl512_nf2048.pth", "models/Main_Models/MGM_LOWEND_A_v4.pth")
    os.rename("models/Main_Models/MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth", "models/Main_Models/MGM_LOWEND_B_v4.pth")
    os.rename("models/Main_Models/MGM_MAIN_v4_sr44100_hl512_nf2048.pth", "models/Main_Models/MGM_MAIN_v4.pth")

except:
    pass

try:
    os.rename("models/MDX_Net_Models/UVR_MDXNET_9703.onnx", "models/MDX_Net_Models/UVR_MDXNET_1_9703.onnx")
    os.rename("models/MDX_Net_Models/UVR_MDXNET_9682.onnx", "models/MDX_Net_Models/UVR_MDXNET_2_9682.onnx")
    os.rename("models/MDX_Net_Models/UVR_MDXNET_9662.onnx", "models/MDX_Net_Models/UVR_MDXNET_3_9662.onnx")
except:
    pass

#Images
banner_path = os.path.join(base_path, 'img', 'UVR-banner.png')
credits_path = os.path.join(base_path, 'img', 'credits.png')
demucs_opt_path = os.path.join(base_path, 'img', 'demucs_opt.png')
donate_path = os.path.join(base_path, 'img', 'donate.png')
download_path = os.path.join(base_path, 'img', 'download.png')
efile_path = os.path.join(base_path, 'img', 'file.png')
ense_opt_path = os.path.join(base_path, 'img', 'ense_opt.png')
gen_opt_path = os.path.join(base_path, 'img', 'gen_opt.png')
help_path = os.path.join(base_path, 'img', 'help.png')
instrumentalModels_dir = os.path.join(base_path, 'models')
key_path = os.path.join(base_path, 'img', 'key.png')
mdx_opt_path = os.path.join(base_path, 'img', 'mdx_opt.png')
stop_path = os.path.join(base_path, 'img', 'stop.png')
user_ens_opt_path = os.path.join(base_path, 'img', 'user_ens_opt.png')
vr_opt_path = os.path.join(base_path, 'img', 'vr_opt.png')

try:
    with open('uvr_patch_version.txt', 'r') as file :
        patch_version = file.read()
    os.remove(f'{patch_version}.exe')
except:
    pass

DEFAULT_DATA = {
    'agg': 10,
    'aiModel': 'MDX-Net',
    'algo': 'Instrumentals (Min Spec)',
    'appendensem': False,
    'audfile': True,
    'aud_mdx': True,
    'autocompensate': True,
    'break': False,
    'channel': 64,
    'chunks': 'Auto',
    'chunks_d': 'Auto',
    'compensate': 1.03597672895,
    'demucs_only': False,
    'demucs_stems': 'All Stems',
    'DemucsModel': 'mdx_extra',
    'demucsmodel': False,
    'DemucsModel_MDX': 'UVR_Demucs_Model_1',
    'demucsmodel_sel_VR': 'UVR_Demucs_Model_1',
    'demucsmodelVR': False,
    'dim_f': 2048,
    'ensChoose': 'Multi-AI Ensemble',
    'exportPath': '',
    'flactype': 'PCM_16',
    'gpu': False,
    'inputPaths': [],
    'inst_only': False,
    'inst_only_b': False,
    'lastDir': None,
    'margin': 44100,
    'margin_d': 44100,
    'mdx_ensem': 'MDX-Net: UVR-MDX-NET Main',
    'mdx_ensem_b': 'No Model',
    'mdx_only_ensem_a': 'MDX-Net: UVR-MDX-NET Main',
    'mdx_only_ensem_b': 'MDX-Net: UVR-MDX-NET 1',
    'mdx_only_ensem_c': 'No Model',
    'mdx_only_ensem_d': 'No Model',
    'mdx_only_ensem_e': 'No Model',
    'mdxnetModel': 'UVR-MDX-NET Main',
    'mdxnetModeltype': 'Vocals (Custom)',
    'mixing': 'Default',
    'modeldownload': 'No Model Selected',
    'modeldownload_mdx': 'No Model Selected',
    'modeldownload_demucs': 'No Model Selected',
    'modeldownload_type': 'VR Arc',
    'modelFolder': False,
    'modelInstrumentalLabel': '',
    'ModelParams': 'Auto',
    'mp3bit': '320k',
    'n_fft_scale': 6144,
    'no_chunk': False,
    'no_chunk_d': False,
    'noise_pro_select': 'Auto Select',
    'noise_reduc': True,
    'noisereduc_s': '3',
    'non_red': False,
    'nophaseinst': False,
    'normalize': False,
    'output_image': False,
    'overlap': 0.25,
    'overlap_b': 0.25,
    'postprocess': False,
    'save': True,
    'saveFormat': 'Wav',
    'selectdownload': 'VR Arc',
    'segment': 'Default',
    'settest': False,
    'shifts': 2,
    'shifts_b': 2,
    'split_mode': True,
    'tta': False,
    'useModel': 'instrumental',
    'voc_only': False,
    'voc_only_b': False,
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
    'window_size': '512',
}

def open_image(path: str, size: tuple = None, keep_aspect: bool = True, rotate: int = 0) -> ImageTk.PhotoImage:
    """
    Open the image on the path and apply given settings\n
    Paramaters:
        path(str):
            Absolute path of the image
        size(tuple):
            first value - width
            second value - height
        keep_aspect(bool):
            keep aspect ratio of image and resize
            to maximum possible width and height
            (maxima are given by size)
        rotate(int):
            clockwise rotation of image
    Returns(ImageTk.PhotoImage):
        Image of path
    """
    img = Image.open(path).convert(mode='RGBA')
    ratio = img.height/img.width
    img = img.rotate(angle=-rotate)
    if size is not None:
        size = (int(size[0]), int(size[1]))
        if keep_aspect:
            img = img.resize((size[0], int(size[0] * ratio)), Image.ANTIALIAS)
        else:
            img = img.resize(size, Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)

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

def load_data_alt():
    
    save_data(data=DEFAULT_DATA)

    return load_data()
    

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

def drop(event, accept_mode: str = 'files'):
    """
    Drag & Drop verification process
    """
    global dnd
    global dnddir
    
    path = event.data

    if accept_mode == 'folder':
        path = path.replace('{', '').replace('}', '')
        if not os.path.isdir(path):
            tk.messagebox.showerror(title='Invalid Folder',
                                    message='Your given export path is not a valid folder!')
            return
        # Set Variables
        root.exportPath_var.set(path)
    elif accept_mode == 'files':
        # Clean path text and set path to the list of paths
        path = path.replace('{', '')
        path = path.split('} ')
        path[-1] = path[-1].replace('}', '')
        # Set Variables
        dnd = 'yes'
        root.inputPaths = path
        root.update_inputPaths()
        dnddir = os.path.dirname(path[0])
    else:
        # Invalid accept mode
        return

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
        
    def percentage(self, line):
        line = f"percentage_value_{line}"
        self.queue.put(line)
        
    def remove(self, line):
        line = f"remove_line_{line}"
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
                    if "percentage_value_" in str(line):
                        line = str(line)
                        line = line.replace("percentage_value_", "")
                        string_len = len(str(line))
                        self.delete(f"end-{string_len + 1}c", tk.END)
                        self.insert(tk.END, f"\n{line}")
                    elif "remove_line_" in str(line):
                        line = str(line)
                        line = line.replace("remove_line_", "")
                        string_len = len(str(line))
                        self.delete(f"end-{string_len}c", tk.END)
                    else:
                        self.insert(tk.END, str(line))

                self.see(tk.END)
                self.update_idletasks()
        except queue.Empty:
            pass
        self.configure(state=tk.DISABLED)
        self.after(100, self.update_me)

class MainWindow(TkinterDnD.Tk):
    # --Constants--
    # Layout
    
    if GetSystemMetrics(1) >= 900:
        IMAGE_HEIGHT = 140
        FILEPATHS_HEIGHT = 85
        OPTIONS_HEIGHT = 275
        CONVERSIONBUTTON_HEIGHT = 35
        COMMAND_HEIGHT = 200
        PROGRESS_HEIGHT = 30
        PADDING = 10
    elif GetSystemMetrics(1) <= 720:
        IMAGE_HEIGHT = 135
        FILEPATHS_HEIGHT = 85
        OPTIONS_HEIGHT = 274
        CONVERSIONBUTTON_HEIGHT = 35
        COMMAND_HEIGHT = 80
        PROGRESS_HEIGHT = 6
        PADDING = 5
    else:
        IMAGE_HEIGHT = 135
        FILEPATHS_HEIGHT = 85
        OPTIONS_HEIGHT = 274
        CONVERSIONBUTTON_HEIGHT = 35
        COMMAND_HEIGHT = 115
        PROGRESS_HEIGHT = 6
        PADDING = 7
    
    COL1_ROWS = 11
    COL2_ROWS = 11

    def __init__(self):
        # Run the __init__ method on the tk.Tk class
        super().__init__()
        
        # Calculate window height
        height = self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT
        height += self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PROGRESS_HEIGHT
        height += self.PADDING * 5  # Padding

        # --Window Settings--
        self.title('Ultimate Vocal Remover')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=620,
            height=height,
            xpad=int(self.winfo_screenwidth()/2 - 635/2),
            ypad=int(self.winfo_screenheight()/2 - height/2 - 30)))
        if GetSystemMetrics(1) >= 900:
            pass
        else:
            self.tk.call('tk', 'scaling', 1.1)
        self.configure(bg='#0e0e0f')  # Set background color to #0c0c0d
        self.protocol("WM_DELETE_WINDOW", self.save_values)
        self.resizable(False, False)
        self.update()

        # --Variables--
        self.logo_img = open_image(path=banner_path,
                                   size=(self.winfo_width(), 9999))
        self.efile_img = open_image(path=efile_path,
                                      size=(20, 20))
        self.stop_img = open_image(path=stop_path,
                                      size=(20, 20))
        self.help_img = open_image(path=help_path,
                                      size=(20, 20))
        self.download_img = open_image(path=download_path,
                                size=(30, 30))       
        self.donate_img = open_image(path=donate_path,
                                size=(30, 30))    
        self.key_img = open_image(path=key_path,
                                size=(30, 30))     
        if GetSystemMetrics(1) >= 900:
            self.gen_opt_img = open_image(path=gen_opt_path,
                                        size=(900, 826))
            self.mdx_opt_img = open_image(path=mdx_opt_path,
                                        size=(900, 826))
            self.vr_opt_img = open_image(path=vr_opt_path,
                                        size=(900, 826))
            self.demucs_opt_img = open_image(path=demucs_opt_path,
                                        size=(900, 826))
            self.ense_opt_img = open_image(path=ense_opt_path,
                                        size=(900, 826))
            self.user_ens_opt_img = open_image(path=user_ens_opt_path,
                                        size=(900, 826))
            self.credits_img = open_image(path=credits_path,
                                        size=(100, 100))
        elif GetSystemMetrics(1) <= 720:
            self.gen_opt_img = open_image(path=gen_opt_path,
                                        size=(740, 826))
            self.mdx_opt_img = open_image(path=mdx_opt_path,
                                        size=(740, 826))
            self.vr_opt_img = open_image(path=vr_opt_path,
                                        size=(695, 826))
            self.demucs_opt_img = open_image(path=demucs_opt_path,
                                        size=(695, 826))
            self.ense_opt_img = open_image(path=ense_opt_path,
                                        size=(740, 826))
            self.user_ens_opt_img = open_image(path=user_ens_opt_path,
                                        size=(740, 826))
            self.credits_img = open_image(path=credits_path,
                                        size=(50, 50))
        else:
            self.gen_opt_img = open_image(path=gen_opt_path,
                                        size=(740, 826))
            self.mdx_opt_img = open_image(path=mdx_opt_path,
                                        size=(740, 826))
            self.vr_opt_img = open_image(path=vr_opt_path,
                                        size=(730, 826))
            self.demucs_opt_img = open_image(path=vr_opt_path,
                                        size=(730, 826))
            self.ense_opt_img = open_image(path=demucs_opt_path,
                                        size=(740, 826))
            self.user_ens_opt_img = open_image(path=user_ens_opt_path,
                                        size=(740, 826))
            self.credits_img = open_image(path=credits_path,
                                        size=(50, 50))
        
        self.instrumentalLabel_to_path = defaultdict(lambda: '')
        self.lastInstrumentalModels = []
        self.lastInstrumentalModels_ensem = []
        self.lastmdx_demuc_ensem = []
        self.MDXLabel_to_path = defaultdict(lambda: '')
        self.lastMDXModels = []
        self.ModelParamsLabel_to_path = defaultdict(lambda: '')
        self.lastModelParams = []
        self.DemucsLabel_to_path = defaultdict(lambda: '')
        self.lastDemucsModels = []
        self.ModelParamsLabel_ens_to_path = defaultdict(lambda: '')
        self.lastModelParams_ens = []

        # -Tkinter Value Holders-
        data = load_data()
        data_alt = load_data_alt()

        try:
            self.agg_var = tk.StringVar(value=data['agg'])
        except:
            self.agg_var = tk.StringVar(value=data_alt['agg'])
        try:
            self.aiModel_var = tk.StringVar(value=data['aiModel'])
        except:
            self.aiModel_var = tk.StringVar(value=data_alt['aiModel'])
        try:
            self.algo_var = tk.StringVar(value=data['algo'])
        except:
            self.algo_var = tk.StringVar(value=data_alt['algo'])
        try:
            self.appendensem_var = tk.BooleanVar(value=data['appendensem'])
        except:
            self.appendensem_var = tk.BooleanVar(value=data_alt['appendensem'])
        try:
            self.audfile_var = tk.BooleanVar(value=data['audfile'])
        except:
            self.audfile_var = tk.BooleanVar(value=data_alt['audfile'])
        try:
            self.aud_mdx_var = tk.BooleanVar(value=data['aud_mdx'])
        except:
            self.aud_mdx_var = tk.BooleanVar(value=data_alt['aud_mdx'])
        try:
            self.autocompensate_var = tk.BooleanVar(value=data['autocompensate'])
        except:
            self.autocompensate_var = tk.BooleanVar(value=data_alt['autocompensate'])
        try:
            self.channel_var = tk.StringVar(value=data['channel'])
        except:
            self.channel_var = tk.StringVar(value=data_alt['channel'])
        try:
            self.chunks_d_var = tk.StringVar(value=data['chunks_d'])
        except:
            self.chunks_d_var = tk.StringVar(value=data_alt['chunks_d'])
        try:
            self.chunks_var = tk.StringVar(value=data['chunks'])
        except:
            self.chunks_var = tk.StringVar(value=data_alt['chunks'])
        try:
            self.compensate_var = tk.StringVar(value=data['compensate'])
        except:
            self.compensate_var = tk.StringVar(value=data_alt['compensate'])
        try:
            self.demucs_only_var = tk.BooleanVar(value=data['demucs_only'])
        except:
            self.demucs_only_var = tk.BooleanVar(value=data_alt['demucs_only'])
        try:
            self.demucs_stems_var = tk.StringVar(value=data['demucs_stems'])
        except:
            self.demucs_stems_var = tk.StringVar(value=data_alt['demucs_stems'])
        try:
            self.DemucsModel_MDX_var = tk.StringVar(value=data['DemucsModel_MDX'])
        except:
            self.DemucsModel_MDX_var = tk.StringVar(value=data_alt['DemucsModel_MDX'])
        try:
            self.demucsmodel_sel_VR_var = tk.StringVar(value=data['demucsmodel_sel_VR'])
        except:
            self.demucsmodel_sel_VR_var = tk.StringVar(value=data_alt['demucsmodel_sel_VR'])
        try:
            self.demucsmodel_var = tk.BooleanVar(value=data['demucsmodel'])
        except:
            self.demucsmodel_var = tk.BooleanVar(value=data_alt['demucsmodel'])
        try:
            self.DemucsModel_var = tk.StringVar(value=data['DemucsModel'])
        except:
            self.DemucsModel_var = tk.StringVar(value=data_alt['DemucsModel'])
        try:
            self.demucsmodelVR_var = tk.BooleanVar(value=data['demucsmodelVR'])
        except:
            self.demucsmodelVR_var = tk.BooleanVar(value=data_alt['demucsmodelVR'])
        try:
            self.dim_f_var = tk.StringVar(value=data['dim_f'])    
        except:
            self.dim_f_var = tk.StringVar(value=data_alt['dim_f'])    
        try:
            self.ensChoose_var = tk.StringVar(value=data['ensChoose'])
        except:
            self.ensChoose_var = tk.StringVar(value=data_alt['ensChoose'])
        try:
            self.exportPath_var = tk.StringVar(value=data['exportPath'])
        except:
            self.exportPath_var = tk.StringVar(value=data_alt['exportPath'])
        try:
            self.flactype_var = tk.StringVar(value=data['flactype'])
        except:
            self.flactype_var = tk.StringVar(value=data_alt['flactype'])
        try:
            self.gpuConversion_var = tk.BooleanVar(value=data['gpu'])
        except:
            self.gpuConversion_var = tk.BooleanVar(value=data_alt['gpu'])
        try:
            self.inputPathop_var = tk.StringVar(value=data['inputPaths'])
        except:
            self.inputPathop_var = tk.StringVar(value=data_alt['inputPaths'])
        try:
            self.inputPaths = data['inputPaths']
        except:
            self.inputPaths = data_alt['inputPaths']
        try:
            self.inst_only_b_var = tk.BooleanVar(value=data['inst_only_b'])
        except:
            self.inst_only_b_var = tk.BooleanVar(value=data_alt['inst_only_b'])
        try:
            self.inst_only_var = tk.BooleanVar(value=data['inst_only'])
        except:
            self.inst_only_var = tk.BooleanVar(value=data_alt['inst_only'])
        try:
            self.instrumentalModel_var = tk.StringVar(value=data['modelInstrumentalLabel'])
        except:
            self.instrumentalModel_var = tk.StringVar(value=data_alt['modelInstrumentalLabel'])
        try:
            self.margin_var = tk.StringVar(value=data['margin'])
        except:
            self.margin_var = tk.StringVar(value=data_alt['margin'])
        try:
            self.margin_d_var = tk.StringVar(value=data['margin_d'])
        except:
            self.margin_d_var = tk.StringVar(value=data_alt['margin_d'])
        try:
            self.mdx_only_ensem_a_var = tk.StringVar(value=data['mdx_only_ensem_a'])
        except:
            self.mdx_only_ensem_a_var = tk.StringVar(value=data_alt['mdx_only_ensem_a'])
        try:
            self.mdx_only_ensem_b_var = tk.StringVar(value=data['mdx_only_ensem_b'])
        except:
            self.mdx_only_ensem_b_var = tk.StringVar(value=data_alt['mdx_only_ensem_b'])
        try:
            self.mdx_only_ensem_c_var = tk.StringVar(value=data['mdx_only_ensem_c'])
        except:
            self.mdx_only_ensem_c_var = tk.StringVar(value=data_alt['mdx_only_ensem_c'])
        try:
            self.mdx_only_ensem_d_var = tk.StringVar(value=data['mdx_only_ensem_d'])
        except:
            self.mdx_only_ensem_d_var = tk.StringVar(value=data_alt['mdx_only_ensem_d'])
        try:
            self.mdx_only_ensem_e_var = tk.StringVar(value=data['mdx_only_ensem_e'])
        except:
            self.mdx_only_ensem_e_var = tk.StringVar(value=data_alt['mdx_only_ensem_e'])
        try:
            self.mdxensemchoose_b_var = tk.StringVar(value=data['mdx_ensem_b'])
        except:
            self.mdxensemchoose_b_var = tk.StringVar(value=data_alt['mdx_ensem_b'])
        try:
            self.mdxensemchoose_var = tk.StringVar(value=data['mdx_ensem'])
        except:
            self.mdxensemchoose_var = tk.StringVar(value=data_alt['mdx_ensem'])
        try:
            self.mdxnetModel_var = tk.StringVar(value=data['mdxnetModel'])
        except:
            self.mdxnetModel_var = tk.StringVar(value=data_alt['mdxnetModel'])
        try:
            self.mdxnetModeltype_var = tk.StringVar(value=data['mdxnetModeltype'])
        except:
            self.mdxnetModeltype_var = tk.StringVar(value=data_alt['mdxnetModeltype'])
        try:
            self.mixing_var = tk.StringVar(value=data['mixing'])
        except:
            self.mixing_var = tk.StringVar(value=data_alt['mixing'])
        try:
            self.modeldownload_type_var = tk.StringVar(value=data['modeldownload_type'])
        except:
            self.modeldownload_type_var = tk.StringVar(value=data_alt['modeldownload_type'])
        try:
            self.modeldownload_var = tk.StringVar(value=data['modeldownload'])
        except:
            self.modeldownload_var = tk.StringVar(value=data_alt['modeldownload'])
        try:
            self.modeldownload_mdx_var = tk.StringVar(value=data['modeldownload_mdx'])
        except:
            self.modeldownload_mdx_var = tk.StringVar(value=data_alt['modeldownload_mdx'])
        try:
            self.modeldownload_demucs_var = tk.StringVar(value=data['modeldownload_demucs'])
        except:
            self.modeldownload_demucs_var = tk.StringVar(value=data_alt['modeldownload_demucs'])
        try:
            self.modelFolder_var = tk.BooleanVar(value=data['modelFolder'])
        except:
            self.modelFolder_var = tk.BooleanVar(value=data_alt['modelFolder'])
        try:
            self.ModelParams_var = tk.StringVar(value=data['ModelParams'])
        except:
            self.ModelParams_var = tk.StringVar(value=data_alt['ModelParams'])
        try:
            self.mp3bit_var = tk.StringVar(value=data['mp3bit'])
        except:
            self.mp3bit_var = tk.StringVar(value=data_alt['mp3bit'])
        try:
            self.n_fft_scale_var = tk.StringVar(value=data['n_fft_scale'])
        except:
            self.n_fft_scale_var = tk.StringVar(value=data_alt['n_fft_scale'])
        try:
            self.no_chunk_var = tk.BooleanVar(value=data['no_chunk'])
        except:
            self.no_chunk_var = tk.BooleanVar(value=data_alt['no_chunk'])
        try:
            self.no_chunk_d_var = tk.BooleanVar(value=data['no_chunk_d'])
        except:
            self.no_chunk_d_var = tk.BooleanVar(value=data_alt['no_chunk_d'])
        try:
            self.noise_pro_select_var = tk.StringVar(value=data['noise_pro_select'])
        except:
            self.noise_pro_select_var = tk.StringVar(value=data_alt['noise_pro_select'])
        try:
            self.noisereduc_s_var = tk.StringVar(value=data['noisereduc_s'])
        except:
            self.noisereduc_s_var = tk.StringVar(value=data_alt['noisereduc_s'])
        try:
            self.noisereduc_var = tk.BooleanVar(value=data['noise_reduc'])
        except:
            self.noisereduc_var = tk.BooleanVar(value=data_alt['noise_reduc'])
        try:
            self.non_red_var = tk.BooleanVar(value=data['non_red'])
        except:
            self.non_red_var = tk.BooleanVar(value=data_alt['non_red'])
        try:
            self.nophaseinst_var = tk.BooleanVar(value=data['nophaseinst'])
        except:
            self.nophaseinst_var = tk.BooleanVar(value=data_alt['nophaseinst'])
        try:
            self.normalize_var = tk.BooleanVar(value=data['normalize'])
        except:
            self.normalize_var = tk.BooleanVar(value=data_alt['normalize'])
        try:
            self.outputImage_var = tk.BooleanVar(value=data['output_image'])
        except:
            self.outputImage_var = tk.BooleanVar(value=data_alt['output_image'])
        try:
            self.overlap_b_var = tk.StringVar(value=data['overlap_b'])
        except:
            self.overlap_b_var = tk.StringVar(value=data_alt['overlap_b'])
        try:
            self.overlap_var = tk.StringVar(value=data['overlap'])
        except:
            self.overlap_var = tk.StringVar(value=data_alt['overlap'])
        try:
            self.postprocessing_var = tk.BooleanVar(value=data['postprocess'])
        except:
            self.postprocessing_var = tk.BooleanVar(value=data_alt['postprocess'])
        try:
            self.save_var = tk.BooleanVar(value=data['save'])
        except:
            self.save_var = tk.BooleanVar(value=data_alt['save'])
        try:
            self.saveFormat_var = tk.StringVar(value=data['saveFormat'])
        except:
            self.saveFormat_var = tk.StringVar(value=data_alt['saveFormat'])
        try:
            self.selectdownload_var = tk.StringVar(value=data['selectdownload'])
        except:
            self.selectdownload_var = tk.StringVar(value=data_alt['selectdownload'])
        try:
            self.segment_var = tk.StringVar(value=data['segment'])
        except:
            self.segment_var = tk.StringVar(value=data_alt['segment'])
        try:
            self.settest_var = tk.BooleanVar(value=data['settest'])
        except:
            self.settest_var = tk.BooleanVar(value=data_alt['settest'])
        try:
            self.shifts_b_var = tk.StringVar(value=data['shifts_b'])
        except:
            self.shifts_b_var = tk.StringVar(value=data_alt['shifts_b'])
        try:
            self.shifts_var = tk.StringVar(value=data['shifts'])
        except:
            self.shifts_var = tk.StringVar(value=data_alt['shifts'])
        try:
            self.split_mode_var = tk.BooleanVar(value=data['split_mode'])
        except:
            self.split_mode_var = tk.BooleanVar(value=data_alt['split_mode'])
        try:
            self.tta_var = tk.BooleanVar(value=data['tta'])
        except:
            self.tta_var = tk.BooleanVar(value=data_alt['tta'])
        try:
            self.voc_only_b_var = tk.BooleanVar(value=data['voc_only_b'])
        except:
            self.voc_only_b_var = tk.BooleanVar(value=data_alt['voc_only_b'])
        try:
            self.voc_only_var = tk.BooleanVar(value=data['voc_only'])
        except:
            self.voc_only_var = tk.BooleanVar(value=data_alt['voc_only'])
        try:
            self.vrensemchoose_a_var = tk.StringVar(value=data['vr_ensem_a'])
        except:
            self.vrensemchoose_a_var = tk.StringVar(value=data_alt['vr_ensem_a'])
        try:
            self.vrensemchoose_b_var = tk.StringVar(value=data['vr_ensem_b'])
        except:
            self.vrensemchoose_b_var = tk.StringVar(value=data_alt['vr_ensem_b'])
        try:
            self.vrensemchoose_c_var = tk.StringVar(value=data['vr_ensem_c'])
        except:
            self.vrensemchoose_c_var = tk.StringVar(value=data_alt['vr_ensem_c'])
        try:
            self.vrensemchoose_d_var = tk.StringVar(value=data['vr_ensem_d'])
        except:
            self.vrensemchoose_d_var = tk.StringVar(value=data_alt['vr_ensem_d'])
        try:
            self.vrensemchoose_e_var = tk.StringVar(value=data['vr_ensem_e'])
        except:
            self.vrensemchoose_e_var = tk.StringVar(value=data_alt['vr_ensem_e'])
        try:
            self.vr_multi_USER_model_param_1 = tk.StringVar(value=data['vr_multi_USER_model_param_1'])
        except:
            self.vr_multi_USER_model_param_1 = tk.StringVar(value=data_alt['vr_multi_USER_model_param_1'])
        try:
            self.vr_multi_USER_model_param_2 = tk.StringVar(value=data['vr_multi_USER_model_param_2'])
        except:
            self.vr_multi_USER_model_param_2 = tk.StringVar(value=data_alt['vr_multi_USER_model_param_2'])
        try:
            self.vr_multi_USER_model_param_3 = tk.StringVar(value=data['vr_multi_USER_model_param_3'])
        except:
            self.vr_multi_USER_model_param_3 = tk.StringVar(value=data_alt['vr_multi_USER_model_param_3'])
        try:
            self.vr_multi_USER_model_param_4 = tk.StringVar(value=data['vr_multi_USER_model_param_4'])
        except:
            self.vr_multi_USER_model_param_4 = tk.StringVar(value=data_alt['vr_multi_USER_model_param_4'])
        try:
            self.vr_basic_USER_model_param_1 = tk.StringVar(value=data['vr_basic_USER_model_param_1'])
        except:
            self.vr_basic_USER_model_param_1 = tk.StringVar(value=data_alt['vr_basic_USER_model_param_1'])
        try:
            self.vr_basic_USER_model_param_2 = tk.StringVar(value=data['vr_basic_USER_model_param_2'])
        except:
            self.vr_basic_USER_model_param_2 = tk.StringVar(value=data_alt['vr_basic_USER_model_param_2'])
        try:
            self.vr_basic_USER_model_param_3 = tk.StringVar(value=data['vr_basic_USER_model_param_3'])
        except:
            self.vr_basic_USER_model_param_3 = tk.StringVar(value=data_alt['vr_basic_USER_model_param_3'])
        try:
            self.vr_basic_USER_model_param_4 = tk.StringVar(value=data['vr_basic_USER_model_param_4'])
        except:
            self.vr_basic_USER_model_param_4 = tk.StringVar(value=data_alt['vr_basic_USER_model_param_4'])
        try:
            self.vr_basic_USER_model_param_5 = tk.StringVar(value=data['vr_basic_USER_model_param_5'])
        except:
            self.vr_basic_USER_model_param_5 = tk.StringVar(value=data_alt['vr_basic_USER_model_param_5'])
        try:
            self.vrensemchoose_mdx_a_var = tk.StringVar(value=data['vr_ensem_mdx_a'])
        except:
            self.vrensemchoose_mdx_a_var = tk.StringVar(value=data_alt['vr_ensem_mdx_a'])
        try:
            self.vrensemchoose_mdx_b_var = tk.StringVar(value=data['vr_ensem_mdx_b'])
        except:
            self.vrensemchoose_mdx_b_var = tk.StringVar(value=data_alt['vr_ensem_mdx_b'])
        try:
            self.vrensemchoose_mdx_c_var = tk.StringVar(value=data['vr_ensem_mdx_c'])
        except:
            self.vrensemchoose_mdx_c_var = tk.StringVar(value=data_alt['vr_ensem_mdx_c'])
        try:
            self.vrensemchoose_var = tk.StringVar(value=data['vr_ensem'])
        except:
            self.vrensemchoose_var = tk.StringVar(value=data_alt['vr_ensem'])
        try:
            self.wavtype_var = tk.StringVar(value=data['wavtype'])
        except:
            self.wavtype_var = tk.StringVar(value=data_alt['wavtype'])
        try:
            self.winSize_var = tk.StringVar(value=data['window_size'])
        except:
            self.winSize_var = tk.StringVar(value=data_alt['window_size'])
        try:
            self.lastDir = data['lastDir']
        except:
            self.lastDir = data_alt['lastDir']

        self.last_aiModel = self.aiModel_var.get()
        self.last_algo = self.aiModel_var.get()
        self.last_ensChoose = self.ensChoose_var.get()
        self.download_progress_var = tk.StringVar(value='')
        self.download_progress_bar_var = tk.StringVar(value='')
        self.download_progress_bar_zip_var = tk.IntVar(value=0)
        self.download_stop_var = tk.StringVar(value='') 
        self.progress_var = tk.IntVar(value=0)
        
        self.last_mdxnetModel = self.mdxnetModel_var.get()
        self.inputPathsEntry_var = tk.StringVar(value='')
        
        # Font
        pyglet.font.add_file('lib_v5/fonts/centurygothic/GOTHIC.TTF')
        self.font = tk.font.Font(family='Century Gothic', size=10)
        self.fontRadio = tk.font.Font(family='Century Gothic', size=8) 
        # --Widgets--
        self.create_widgets()
        self.bind_widgets()
        self.place_widgets()
        
        self.update_available_models()
        self.update_states()
        self.update_loop()
   
        global space_fill_wide
        global space_medium_l
        global space_medium
        global space_small
        global space_tiny
        global download_code_file
        global user_code_file
        global download_code_temp_dir

        download_code_file = 'lib_v5/filelists/download_codes/user_code_download.txt'
        user_code_file = 'lib_v5/filelists/download_codes/user_code.txt'
        download_code_temp_dir = 'lib_v5/filelists/download_codes/temp'

        space_fill_wide = '  '*32
        space_medium_l = '  '*20
        space_medium = '  '*17
        space_small = '  '*10
        space_tiny = '  '*5
   
    # -Widget Methods-
    
    def create_widgets(self):
        """Create window widgets"""
        self.title_Label = tk.Label(master=self,
                                    image=self.logo_img, compound=tk.TOP)
        self.filePaths_Frame = ttk.Frame(master=self)
        self.fill_filePaths_Frame()

        self.options_Frame = ttk.Frame(master=self)
        self.fill_options_Frame()

        self.conversion_Button = ttk.Button(master=self,
                                            text='Start Processing',
                                            command=self.start_conversion)
        self.stop_Button = ttk.Button(master=self,
                                         image=self.stop_img,
                                         command=self.stop_inf)
        self.mdx_stop_Button = ttk.Button(master=self,
                                         image=self.stop_img,
                                         command=self.stop_inf)
        self.settings_Button = ttk.Button(master=self,
                                         image=self.help_img,
                                         command=self.settings)

        #ttk.Button(win, text= "Open", command= open_popup).pack()
        
        self.efile_e_Button = ttk.Button(master=self,
                                         image=self.efile_img,
                                         command=self.open_exportPath_filedialog)
        
        self.efile_i_Button = ttk.Button(master=self,
                                         image=self.efile_img,
                                         command=self.open_inputPath_filedialog)
        
        self.progressbar = ttk.Progressbar(master=self, variable=self.progress_var)

        self.command_Text = ThreadSafeConsole(master=self,
                                              background='#0e0e0f',fg='#898b8e', font=('Century Gothic', 11),borderwidth=0)

        #self.command_Text.write(f'Ultimate Vocal Remover [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n')
        
        global current_version

        with open('uvr_patch_version.txt', 'r') as file :
            current_version = file.read()

        def start_target_update():
            
            def target_update():
                update_signal_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/update_patches.txt"
                url = update_signal_url
                label_set = " "
                try:
                    file = urllib.request.urlopen(url)
                    for line in file:
                        patch_name = line.decode("utf-8")
                        if patch_name == current_version:
                            self.command_Text.write(f'Ultimate Vocal Remover v5.4.0 [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
                        else:
                            label_set = f"New Update Found: {patch_name}\n\nClick the update button in the \"Settings\" menu to download and install!"
                
                            self.command_Text.write(f'Ultimate Vocal Remover v5.4.0 [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n\n{label_set}')
                except:
                        self.command_Text.write(f'Ultimate Vocal Remover v5.4.0 [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
                        
            rlg = KThread(target=target_update)
            rlg.start()

        start_target_update()


    def bind_widgets(self):
        """Bind widgets to the drag & drop mechanic"""
        self.filePaths_musicFile_Button.drop_target_register(DND_FILES)
        self.filePaths_musicFile_Entry.drop_target_register(DND_FILES)
        self.filePaths_saveTo_Button.drop_target_register(DND_FILES)
        self.filePaths_saveTo_Entry.drop_target_register(DND_FILES)        
        self.filePaths_musicFile_Button.dnd_bind('<<Drop>>',
                                                lambda e: drop(e, accept_mode='files'))
        self.filePaths_musicFile_Entry.dnd_bind('<<Drop>>',
                                                lambda e: drop(e, accept_mode='files'))
        self.filePaths_saveTo_Button.dnd_bind('<<Drop>>',
                                                lambda e: drop(e, accept_mode='folder'))
        self.filePaths_saveTo_Entry.dnd_bind('<<Drop>>',
                                                lambda e: drop(e, accept_mode='folder'))

    def place_widgets(self):
        """Place main widgets"""
        self.title_Label.place(x=-2, y=-2)
        self.filePaths_Frame.place(x=10, y=155, width=-20, height=self.FILEPATHS_HEIGHT,
                                   relx=0, rely=0, relwidth=1, relheight=0)
        self.options_Frame.place(x=10, y=250, width=-50, height=self.OPTIONS_HEIGHT,
                                 relx=0, rely=0, relwidth=1, relheight=0)
        self.conversion_Button.place(x=50, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.PADDING*2, width=-60 - 40, height=self.CONVERSIONBUTTON_HEIGHT,
                                     relx=0, rely=0, relwidth=1, relheight=0)
        self.efile_e_Button.place(x=-45, y=200, width=35, height=30,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        self.efile_i_Button.place(x=-45, y=160, width=35, height=30,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        
        self.stop_Button.place(x=-10 - 35, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.PADDING*2, width=35, height=self.CONVERSIONBUTTON_HEIGHT,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        self.settings_Button.place(x=-10 - 600, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.PADDING*2, width=35, height=self.CONVERSIONBUTTON_HEIGHT,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        self.command_Text.place(x=25, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.PADDING*3, width=-30, height=self.COMMAND_HEIGHT,
                                relx=0, rely=0, relwidth=1, relheight=0)
        self.progressbar.place(x=25, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PADDING*4, width=-50, height=self.PROGRESS_HEIGHT,
                               relx=0, rely=0, relwidth=1, relheight=0)

    def fill_filePaths_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
        # Save To Option
         # Select Music Files Option

        # Save To Option
        self.filePaths_saveTo_Button = ttk.Button(master=self.filePaths_Frame,
                                                  text='Select output',
                                                  command=self.open_export_filedialog)
        self.filePaths_saveTo_Entry = ttk.Entry(master=self.filePaths_Frame,

                                                textvariable=self.exportPath_var,
                                                state=tk.DISABLED
                                                )
        # Select Music Files Option
        self.filePaths_musicFile_Button = ttk.Button(master=self.filePaths_Frame,
                                                     text='Select input',
                                                     command=self.open_file_filedialog)
        self.filePaths_musicFile_Entry = ttk.Entry(master=self.filePaths_Frame,
                                                   textvariable=self.inputPathsEntry_var,
                                                   state=tk.DISABLED
                                                   )
       
       
        # -Place Widgets-
        
        # Select Music Files Option
        self.filePaths_musicFile_Button.place(x=0, y=5, width=0, height=-10,
                                              relx=0, rely=0, relwidth=0.3, relheight=0.5)                                              
        self.filePaths_musicFile_Entry.place(x=10, y=2.5, width=-50, height=-5,
                                              relx=0.3, rely=0, relwidth=0.7, relheight=0.5)                                             

        # Save To Option
        self.filePaths_saveTo_Button.place(x=0, y=5, width=0, height=-10,
                                           relx=0, rely=0.5, relwidth=0.3, relheight=0.5)
        self.filePaths_saveTo_Entry.place(x=10, y=2.5, width=-50, height=-5,
                                          relx=0.3, rely=0.5, relwidth=0.7, relheight=0.5)


    def fill_options_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
      

        # Save as wav
        self.options_wav_Radiobutton = ttk.Radiobutton(master=self.options_Frame,                                                
                                                text='WAV',
                                                variable=self.saveFormat_var,
                                                value='Wav'
                                                )
        
        # Save as flac 
        self.options_flac_Radiobutton = ttk.Radiobutton(master=self.options_Frame,                                                
                                                text='FLAC',                                                
                                                variable=self.saveFormat_var, 
                                                value='Flac'
                                                )
        
        # Save as mp3 
        self.options_mpThree_Radiobutton = ttk.Radiobutton(master=self.options_Frame,                                                
                                                text='MP3',                                                                                              
                                                variable=self.saveFormat_var, 
                                                value='Mp3', 
                                                )
        
        # -Column 1-
        
        # Choose Conversion Method
        self.options_aiModel_Label = tk.Button(master=self.options_Frame,
                                               text='Choose Process Method', anchor=tk.CENTER,
                                               background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.open_appdir_filedialog)
        self.options_aiModel_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                          self.aiModel_var, 
                                                          None, 'VR Architecture', 'MDX-Net', 'Demucs v3', 'Ensemble Mode')
        
        #  Choose Instrumental Model
        self.options_instrumentalModel_Label = tk.Button(master=self.options_Frame,
                                                        text='Choose Main Model',
                                                        background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.open_Modelfolder_vr)
        self.options_instrumentalModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                                   self.instrumentalModel_var)
        
        #  Choose Demucs Model
        self.options_DemucsModel_Label = tk.Button(master=self.options_Frame,
                                                        text='Choose Demucs Model',
                                                        background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.open_Modelfolder_de)
        self.options_DemucsModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                                   self.DemucsModel_var)
        
        #  Choose MDX-Net Model
        self.options_mdxnetModel_Label = tk.Button(master=self.options_Frame,
                                                        text='Choose MDX-Net Model', anchor=tk.CENTER,
                                                        background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.open_newModel_filedialog)
        
        self.options_mdxnetModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                             self.mdxnetModel_var)
        
        # Ensemble Mode
        self.options_ensChoose_Label = tk.Button(master=self.options_Frame,
                                               text='Choose Ensemble', anchor=tk.CENTER,
                                               background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.custom_ensemble)
        self.options_ensChoose_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                          self.ensChoose_var,
                                                          None, 'Multi-AI Ensemble', 'Basic VR Ensemble', 'Basic MD Ensemble', 'Manual Ensemble')
        
        # Choose Agorithim
        self.options_algo_Label = tk.Label(master=self.options_Frame,
                                               text='Choose Algorithm', anchor=tk.CENTER,
                                               background='#0e0e0f', font=self.font, foreground='#13a4c9')
        self.options_algo_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                          self.algo_var, 
                                                          None, 'Vocals (Max Spec)', 'Instrumentals (Min Spec)')#, 'Invert (Normal)', 'Invert (Spectral)')
        
        # Choose Demucs Stems
        self.options_demucs_stems_Label = tk.Button(master=self.options_Frame,
                                               text='Choose Stem(s)', anchor=tk.CENTER,
                                               background='#0e0e0f', font=self.font, foreground='#13a4c9', borderwidth=0, command=self.advanced_demucs_options)
        self.options_demucs_stems_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                          self.demucs_stems_var, 
                                                          None, 'All Stems', 'Vocals', 'Other', 'Bass', 'Drums')

        
        # -Column 2-
        
        # WINDOW SIZE
        self.options_winSize_Label = tk.Button(master=self.options_Frame,
                                              text='Window Size', anchor=tk.CENTER,
                                              background='#0e0e0f', font=self.font, foreground='#13a4c9', 
                                              borderwidth=0, command=self.advanced_vr_options)
        self.options_winSize_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.winSize_var,
                                                         None, '320', '512','1024')
        # MDX-chunks
        self.options_chunks_Label = tk.Label(master=self.options_Frame,
                                           text='Chunks',
                                           background='#0e0e0f', font=self.font, foreground='#13a4c9')
        self.options_chunks_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.chunks_var,
                                                         None, 'Auto', '1', '5', '10', '15', '20', 
                                                         '25', '30', '35', '40', '45', '50', 
                                                         '55', '60', '65', '70', '75', '80', 
                                                         '85', '90', '95', 'Full')
        
        # Demucs-Segment
        self.options_segment_Label = tk.Label(master=self.options_Frame,
                                           text='Segment',
                                           background='#0e0e0f', font=self.font, foreground='#13a4c9')
        self.options_segment_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.segment_var,
                                                         None, 'Default', '1', '5', '10', '15', '20', 
                                                         '25', '30', '35', '40', '45', '50', 
                                                         '55', '60', '65', '70', '75', '80', 
                                                         '85', '90', '95', '100')
        
        
        # Overlap
        self.options_overlap_b_Label = tk.Label(master=self.options_Frame,
                                              text='Overlap', anchor=tk.CENTER,
                                              background='#0e0e0f', font=self.font, foreground='#13a4c9')
        self.options_overlap_b_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.overlap_b_var,
                                                         0, 0.25, 0.50, 0.75, 0.99)
        
        # Shifts
        self.options_shifts_b_Label = tk.Label(master=self.options_Frame,
                                              text='Shifts', anchor=tk.CENTER,
                                              background='#0e0e0f', font=self.font, foreground='#13a4c9')
        self.options_shifts_b_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.shifts_b_var,
                                                         0, 0, 1, 2, 3, 4, 5, 
                                                         6, 7, 8, 9, 10, 11, 
                                                         12, 13, 14, 15, 16, 17, 
                                                         18, 19, 20)

        #Checkboxes
        # GPU Selection
        self.options_gpu_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='GPU Conversion',
                                                       variable=self.gpuConversion_var,
                                                       )
        
        # Vocal Only
        self.options_voc_only_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Save Vocals Only',
                                                variable=self.voc_only_var,
                                                )
        # Instrumental Only 
        self.options_inst_only_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Save Instrumental Only',
                                                variable=self.inst_only_var,
                                                )
        
        # Vocal Only
        self.options_voc_only_b_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Stem Only',
                                                variable=self.voc_only_b_var,
                                                )
        # Instrumental Only 
        self.options_inst_only_b_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Mix Without Stem Only',
                                                variable=self.inst_only_b_var,
                                                )
        
        # TTA
        self.options_tta_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='TTA',
                                                       variable=self.tta_var,
                                                       )

        # MDX-Auto-Chunk
        self.options_non_red_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Save Noisey Output',
                                                variable=self.non_red_var,
                                                )

        # Demucs Model VR
        self.options_postpro_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                        text='Post-Process',
                                                        variable=self.postprocessing_var,
                                                        )
        
        # Split Mode
        self.options_split_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                        text='Split Mode',
                                                        variable=self.split_mode_var,
                                                        )

        # -Column 3-

        # AGG
        self.options_agg_Label = tk.Button(master=self.options_Frame,
                                           text='Aggression Setting',
                                           background='#0e0e0f', font=self.font, foreground='#13a4c9', 
                                           borderwidth=0, command=self.advanced_vr_options)
        self.options_agg_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.agg_var,
                                                         None, '1', '2', '3', '4', '5', 
                                                         '6', '7', '8', '9', '10', '11', 
                                                         '12', '13', '14', '15', '16', '17', 
                                                         '18', '19', '20')

        # MDX-noisereduc_s
        self.options_noisereduc_s_Label = tk.Button(master=self.options_Frame,
                                           text='Noise Reduction',
                                           background='#0e0e0f', font=self.font, foreground='#13a4c9', 
                                           borderwidth=0, command=self.advanced_mdx_options)
        self.options_noisereduc_s_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.noisereduc_s_var,
                                                         None, 'None', '0', '1', '2', '3', '4', '5', 
                                                         '6', '7', '8', '9', '10')
        

        # Save Image
        self.options_image_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                         text='Output Image',
                                                         variable=self.outputImage_var,
                                                         )

        # MDX-Enable Demucs Model
        self.options_demucsmodel_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                               text='Demucs Model',
                                                               variable=self.demucsmodel_var,
                                                               )

        # MDX-Noise Reduction
        self.options_noisereduc_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                text='Noise Reduction',
                                                variable=self.noisereduc_var,
                                                )
        
        # Ensemble Save Ensemble Outputs
        self.options_save_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='Save All Outputs',
                                                       variable=self.save_var,
                                                       )
        
        # Model Test Mode
        self.options_modelFolder_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                               text='Model Test Mode',
                                                               variable=self.modelFolder_var,
                                                               )

        # -Place Widgets-

        # -Column 0-

        # Save as
        self.options_wav_Radiobutton.place(x=400, y=-5, width=0, height=6,
                                             relx=0, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_flac_Radiobutton.place(x=271, y=-5, width=0, height=6,
                                             relx=1/3, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_mpThree_Radiobutton.place(x=143, y=-5, width=0, height=6,
                                             relx=2/3, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # -Column 1-
        
        # Choose Conversion Method
        self.options_aiModel_Label.place(x=0, y=0, width=0, height=-10,
                                    relx=0, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_aiModel_Optionmenu.place(x=0, y=-2, width=0, height=7,
                                    relx=0, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # Choose Main Model
        self.options_instrumentalModel_Label.place(x=0, y=19, width=0, height=-10,
                                    relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_instrumentalModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                    relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        
        # Choose Demucs Model
        self.options_DemucsModel_Label.place(x=0, y=19, width=0, height=-10,
                                    relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_DemucsModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                    relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        
        # Choose MDX-Net Model
        self.options_mdxnetModel_Label.place(x=0, y=19, width=0, height=-10,
                                    relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_mdxnetModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                    relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # Choose Ensemble 
        self.options_ensChoose_Label.place(x=0, y=19, width=0, height=-10,
                                relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_ensChoose_Optionmenu.place(x=0, y=19, width=0, height=7,
                                relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        
        # Choose Algorithm
        self.options_algo_Label.place(x=20, y=0, width=0, height=-10,
                                relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_algo_Optionmenu.place(x=12, y=-2, width=0, height=7,
                                relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        
        # Choose Demucs Stems
        self.options_demucs_stems_Label.place(x=13, y=0, width=0, height=-10,
                                    relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_demucs_stems_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # -Column 2-
    
        # WINDOW
        self.options_winSize_Label.place(x=13, y=0, width=0, height=-10,
                                    relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_winSize_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #---MDX-Net Specific---
        # MDX-chunks
        self.options_chunks_Label.place(x=12, y=0, width=0, height=-10,
                                    relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_chunks_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        # Demucs-Segment
        self.options_segment_Label.place(x=12, y=0, width=0, height=-10,
                                    relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_segment_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        # Overlap
        self.options_overlap_b_Label.place(x=13, y=0, width=0, height=-10,
                                    relx=2/3, rely=4/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_overlap_b_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # Shifts
        self.options_shifts_b_Label.place(x=12, y=0, width=0, height=-10,
                                    relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_shifts_b_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        
        #Checkboxes
        
        #GPU Conversion
        self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Vocals Only
        self.options_voc_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Instrumental Only
        self.options_inst_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        #Vocals Only
        self.options_voc_only_b_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Instrumental Only
        self.options_inst_only_b_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Split Mode
        self.options_split_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        # TTA
        self.options_tta_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        # MDX-Keep Non_Reduced Vocal
        self.options_non_red_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # -Column 3-
        
        # AGG
        self.options_agg_Label.place(x=15, y=0, width=0, height=-10,
                                    relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_agg_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        # MDX-noisereduc_s
        self.options_noisereduc_s_Label.place(x=15, y=0, width=0, height=-10,
                                    relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_noisereduc_s_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                    relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Checkboxes
        #---MDX-Net Specific---
        # MDX-demucs Model
        self.options_demucsmodel_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        #---VR Architecture Specific---
        #Post-Process
        self.options_postpro_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #Save Image
        # self.options_image_Checkbutton.place(x=35, y=21, width=0, height=5,
        #                                     relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #---Ensemble Specific---
        #Ensemble Save Outputs
        self.options_save_Checkbutton.place(x=35, y=21, width=0, height=5,
                                    relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #---MDX-Net & VR Architecture Specific---
        #Model Test Mode
        self.options_modelFolder_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        
        # Change States
        self.aiModel_var.trace_add('write',
                                    lambda *args: self.deselect_models())
        self.ensChoose_var.trace_add('write',
                            lambda *args: self.update_states())
        
        self.inst_only_var.trace_add('write',
                    lambda *args: self.update_states())

        self.voc_only_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.inst_only_b_var.trace_add('write',
                    lambda *args: self.update_states())

        self.voc_only_b_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.demucs_stems_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.noisereduc_s_var.trace_add('write',
                    lambda *args: self.update_states())
        self.non_red_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.mdxnetModeltype_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.n_fft_scale_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.dim_f_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.demucsmodel_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.demucs_only_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.split_mode_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.chunks_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.chunks_d_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.margin_d_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.no_chunk_d_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.autocompensate_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.compensate_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.selectdownload_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.modeldownload_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.modeldownload_mdx_var.trace_add('write',
                    lambda *args: self.update_states())
        
        self.modeldownload_demucs_var.trace_add('write',
                    lambda *args: self.update_states())
        
    # Opening filedialogs
    def open_file_filedialog(self):
        """Make user select music files"""
        global dnd
        global nondnd
        
        if self.lastDir is not None:
            if not os.path.isdir(self.lastDir):
                self.lastDir = None

        paths = tk.filedialog.askopenfilenames(
            parent=self,
            title=f'Select Music Files',
            initialfile='',
            initialdir=self.lastDir,
        )
        if paths:  # Path selected
            self.inputPaths = paths
            dnd = 'no'
            self.update_inputPaths()
            nondnd = os.path.dirname(paths[0])

    def open_export_filedialog(self):
        """Make user select a folder to export the converted files in"""
        path = tk.filedialog.askdirectory(
            parent=self,
            title=f'Select Folder',)
        if path:  # Path selected
            self.exportPath_var.set(path)

    def open_exportPath_filedialog(self):
        filename = self.exportPath_var.get()

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_inputPath_filedialog(self):
        """Open Input Directory"""
         
        try: 
            if dnd == 'yes':  
                self.lastDir = str(dnddir)  
                filename = str(self.lastDir)
                if sys.platform == "win32":
                    os.startfile(filename)
            if dnd == 'no':
                self.lastDir = str(nondnd)
                filename = str(self.lastDir)
                
                if sys.platform == "win32":
                    os.startfile(filename)
        except:
            tbs = str(self.inputPathsEntry_var.get())
            head, sep, tail = tbs.partition(';')
            in_path=os.path.dirname(head)
            filename = str(in_path)
                
            if sys.platform == "win32":
                os.startfile(filename)

    def start_conversion(self):
        """
        Start the conversion for all the given mp3 and wav files
        """

        global stop_inf
        
        stop_inf = self.stop_inf_mdx

        # -Get all variables-
        export_path = self.exportPath_var.get()
        input_paths = self.inputPaths
        instrumentalModel_path = self.instrumentalLabel_to_path[self.instrumentalModel_var.get()]  # nopep8
        # mdxnetModel_path = self.mdxnetLabel_to_path[self.mdxnetModel_var.get()]
        # Get constants
        instrumental = self.instrumentalModel_var.get()
        try:
            if [bool(instrumental)].count(True) == 2: #CHECKTHIS
                window_size = DEFAULT_DATA['window_size']
                agg = DEFAULT_DATA['agg']
                chunks = DEFAULT_DATA['chunks']
                noisereduc_s = DEFAULT_DATA['noisereduc_s']
                mixing = DEFAULT_DATA['mixing']
            else:
                window_size = int(self.winSize_var.get())
                agg = int(self.agg_var.get())
                chunks = str(self.chunks_var.get())
                noisereduc_s = str(self.noisereduc_s_var.get())
                mixing = str(self.mixing_var.get())
                ensChoose = str(self.ensChoose_var.get())
                mdxnetModel = str(self.mdxnetModel_var.get())

        except SyntaxError:  # Non integer was put in entry box
            tk.messagebox.showwarning(master=self,
                                    title='Invalid Music File',
                                    message='You have selected an invalid music file!\nPlease make sure that your files still exist and ends with either ".mp3", ".mp4", ".m4a", ".flac", ".wav"')
            return

        # -Check for invalid inputs-
        
        for path in input_paths:
            if not os.path.isfile(path):
                    tk.messagebox.showwarning(master=self,
                                            title='Drag and Drop Feature Failed or Invalid Input',
                                            message='The input is invalid, or the drag and drop feature failed to select your files properly.\n\nPlease try the following:\n\n1. Select your inputs using the \"Select Input\" button\n2. Verify the input is valid.\n3. Then try again.')
                    return 


        if self.aiModel_var.get() == 'VR Architecture':
            if not os.path.isfile(instrumentalModel_path):
                    tk.messagebox.showwarning(master=self,
                                                title='Invalid Main Model File',
                                                message='You have selected an invalid main model file!\nPlease make sure that your model file still exists!')
                    return

        if not os.path.isdir(export_path):
            tk.messagebox.showwarning(master=self,
                                    title='Invalid Export Directory',
                                    message='You have selected an invalid export directory!\nPlease make sure that your directory still exists!')
            return

        if self.aiModel_var.get() == 'VR Architecture':
            inference = inference_v5
        elif self.aiModel_var.get() == 'Ensemble Mode':
            inference = inference_v5_ensemble
        elif self.aiModel_var.get() == 'MDX-Net':
            inference = inference_MDX
        elif self.aiModel_var.get() == 'Demucs v3':
            inference = inference_demucs
        else:
            raise TypeError('This error should not occur.')

        # -Run the algorithm-
        
        global inf
        inf = KThread(target=inference.main,
                        kwargs={
                            # Paths
                            'agg': agg,
                            'algo': self.algo_var.get(),
                            'appendensem': self.appendensem_var.get(),
                            'audfile': self.audfile_var.get(),
                            'aud_mdx': self.aud_mdx_var.get(),
                            'autocompensate': self.autocompensate_var.get(),
                            'break': False,
                            'button_widget_mdx_model_set': self.conversion_Button,
                            'button_widget': self.conversion_Button,
                            'channel': self.channel_var.get(),
                            'chunks': chunks,
                            'chunks_d': self.chunks_d_var.get(),
                            'compensate': self.compensate_var.get(),
                            'demucs_only': self.demucs_only_var.get(),
                            'demucs_stems': self.demucs_stems_var.get(),
                            'DemucsModel': self.DemucsModel_var.get(),
                            'demucsmodel': self.demucsmodel_var.get(),
                            'DemucsModel_MDX': self.DemucsModel_MDX_var.get(),
                            'demucsmodel_sel_VR': self.demucsmodel_sel_VR_var.get(),
                            'demucsmodelVR': self.demucsmodelVR_var.get(),
                            'dim_f': self.dim_f_var.get(),
                            'ensChoose': ensChoose,
                            'export_path': export_path,
                            'flactype': self.flactype_var.get(),
                            'gpu': 0 if self.gpuConversion_var.get() else -1,
                            'input_paths': input_paths,
                            'inst_menu': self.options_instrumentalModel_Optionmenu,
                            'inst_only': self.inst_only_var.get(),
                            'inst_only_b': self.inst_only_b_var.get(),
                            'instrumentalModel': instrumentalModel_path,
                            'margin': self.margin_var.get(),
                            'margin_d': self.margin_d_var.get(),
                            'mdx_ensem': self.mdxensemchoose_var.get(),
                            'mdx_ensem_b': self.mdxensemchoose_b_var.get(),
                            'mdx_only_ensem_a': self.mdx_only_ensem_a_var.get(),
                            'mdx_only_ensem_b': self.mdx_only_ensem_b_var.get(),
                            'mdx_only_ensem_c': self.mdx_only_ensem_c_var.get(),
                            'mdx_only_ensem_d': self.mdx_only_ensem_d_var.get(),
                            'mdx_only_ensem_e': self.mdx_only_ensem_e_var.get(),
                            'mdxnetModel': mdxnetModel,
                            'mdxnetModeltype': self.mdxnetModeltype_var.get(),
                            'mixing': mixing,
                            'modelFolder': self.modelFolder_var.get(),
                            'ModelParams': self.ModelParams_var.get(),
                            'mp3bit': self.mp3bit_var.get(),
                            'n_fft_scale': self.n_fft_scale_var.get(),
                            'no_chunk': self.no_chunk_var.get(),
                            'no_chunk_d': self.no_chunk_d_var.get(),
                            'noise_pro_select': self.noise_pro_select_var.get(),
                            'noise_reduc': self.noisereduc_var.get(),
                            'noisereduc_s': noisereduc_s,
                            'non_red': self.non_red_var.get(),
                            'nophaseinst': self.nophaseinst_var.get(),
                            'normalize': self.normalize_var.get(),
                            'output_image': self.outputImage_var.get(),
                            'overlap': self.overlap_var.get(),
                            'overlap_b': self.overlap_b_var.get(),
                            'postprocess': self.postprocessing_var.get(),
                            'progress_var': self.progress_var,
                            'save': self.save_var.get(),
                            'saveFormat': self.saveFormat_var.get(),
                            'selectdownload': self.selectdownload_var.get(),
                            'segment': self.segment_var.get(),
                            'settest': self.settest_var.get(),
                            'shifts': self.shifts_var.get(),
                            'shifts_b': self.shifts_b_var.get(),
                            'split_mode': self.split_mode_var.get(),
                            'text_widget': self.command_Text,
                            'tta': self.tta_var.get(),
                            'useModel': 'instrumental',  # Always instrumental
                            'voc_only': self.voc_only_var.get(),
                            'voc_only_b': self.voc_only_b_var.get(),
                            'vocalModel': '',  # Always not needed
                            'vr_ensem': self.vrensemchoose_var.get(),
                            'vr_ensem_a': self.vrensemchoose_a_var.get(),
                            'vr_ensem_b': self.vrensemchoose_b_var.get(),
                            'vr_ensem_c': self.vrensemchoose_c_var.get(),
                            'vr_ensem_d': self.vrensemchoose_d_var.get(),
                            'vr_ensem_e': self.vrensemchoose_e_var.get(),
                            'vr_ensem_mdx_a': self.vrensemchoose_mdx_a_var.get(),
                            'vr_ensem_mdx_b': self.vrensemchoose_mdx_b_var.get(),
                            'vr_ensem_mdx_c': self.vrensemchoose_mdx_c_var.get(),
                            'vr_multi_USER_model_param_1': self.vr_multi_USER_model_param_1.get(),
                            'vr_multi_USER_model_param_2': self.vr_multi_USER_model_param_2.get(),
                            'vr_multi_USER_model_param_3': self.vr_multi_USER_model_param_3.get(),
                            'vr_multi_USER_model_param_4': self.vr_multi_USER_model_param_4.get(),
                            'vr_basic_USER_model_param_1': self.vr_basic_USER_model_param_1.get(),
                            'vr_basic_USER_model_param_2': self.vr_basic_USER_model_param_2.get(),
                            'vr_basic_USER_model_param_3': self.vr_basic_USER_model_param_3.get(),
                            'vr_basic_USER_model_param_4': self.vr_basic_USER_model_param_4.get(),
                            'vr_basic_USER_model_param_5': self.vr_basic_USER_model_param_5.get(),
                            'wavtype': self.wavtype_var.get(),
                            'window': self,
                            'window_size': window_size,
                            'stop_thread': stop_inf,
                        },
                        daemon=True
                        )
        
        inf.start()
        
    def stop_inf(self):
        
        confirm = tk.messagebox.askyesno(title='Confirmation',
                message='You are about to stop all active processes.\n\nAre you sure you wish to continue?')

        if confirm:
            inf.kill()
            button_widget = self.conversion_Button
            button_widget.configure(state=tk.NORMAL)
            text = self.command_Text
            text.write('\n\nProcess stopped by user.')
            torch.cuda.empty_cache()
            importlib.reload(inference_v5)
            importlib.reload(inference_v5_ensemble)
            importlib.reload(inference_MDX)
            importlib.reload(inference_demucs)
            self.progress_var.set(0)
        else:
            pass
        
    def stop_inf_mdx(self):
        inf.kill()
        button_widget = self.conversion_Button
        button_widget.configure(state=tk.NORMAL)
        #text = self.command_Text
        #text.write('\n\nProcess stopped by user.')
        torch.cuda.empty_cache()
        importlib.reload(inference_v5)
        importlib.reload(inference_v5_ensemble)
        importlib.reload(inference_MDX)
        importlib.reload(inference_demucs)
        
    # Models
    def update_inputPaths(self):
        """Update the music file entry"""
        if self.inputPaths:
            # Non-empty Selection
            text = '; '.join(self.inputPaths)
        else:
            # Empty Selection
            text = ''
        self.inputPathsEntry_var.set(text)


    def update_loop(self):
        """Update the dropdown menu"""
        self.update_available_models()

        self.after(1000, self.update_loop)

    def update_available_models(self):
        """
        Loop through every VR model (.pth) in the models directory
        and add to the select your model list
        """
        temp_DemucsModels_dir = os.path.join(instrumentalModels_dir, 'Demucs_Models')
        new_DemucsModels = os.listdir(temp_DemucsModels_dir)
  
        temp_MDXModels_dir = os.path.join(instrumentalModels_dir, 'MDX_Net_Models')  # nopep8
        new_MDXModels = os.listdir(temp_MDXModels_dir)
        
        newmodels = [new_DemucsModels, new_MDXModels]

        temp_instrumentalModels_dir = os.path.join(instrumentalModels_dir, 'Main_Models')  # nopep8
        new_InstrumentalModels = os.listdir(temp_instrumentalModels_dir)
        
        if new_InstrumentalModels != self.lastInstrumentalModels_ensem:
            
            with open('lib_v5/filelists/ensemble_list/vr_en_list.txt', 'w') as f:
                f.write('No Model\nNo Model\n')
        
        if new_InstrumentalModels != self.lastInstrumentalModels:
            self.instrumentalLabel_to_path.clear()
            self.options_instrumentalModel_Optionmenu['menu'].delete(0, 'end')
            for file_name in natsort.natsorted(new_InstrumentalModels):
                if file_name.endswith('.pth'):
                    # Add Radiobutton to the Options Menu
                    self.options_instrumentalModel_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                                      command=tk._setit(self.instrumentalModel_var, file_name))
                    # Link the files name to its absolute path
                    self.instrumentalLabel_to_path[file_name] = os.path.join(temp_instrumentalModels_dir, file_name)  # nopep8
            self.lastInstrumentalModels = new_InstrumentalModels
            #print(self.instrumentalLabel_to_path)
            
        if new_InstrumentalModels != self.lastInstrumentalModels_ensem:
            
            for file_name_vr in natsort.natsorted(new_InstrumentalModels):
                if file_name_vr.endswith(".pth"):
                    b = [".pth"]
                    for char in b:
                        file_name_vr = file_name_vr.replace(char, "")

                    vr_list_en = file_name_vr
                    
                    with open('lib_v5/filelists/ensemble_list/vr_en_list.txt', 'a') as f:
                        f.write("{}\n".format(vr_list_en))
                    
            self.lastInstrumentalModels_ensem = new_InstrumentalModels

        """
        Loop through every MDX-Net model (.onnx) in the models directory
        and add to the select your model list
        """
        
        if newmodels != self.lastmdx_demuc_ensem:
            with open('lib_v5/filelists/ensemble_list/mdx_demuc_en_list.txt', 'w') as f:
                f.write('No Model\nNo Model\n')
        
        if new_MDXModels != self.lastMDXModels or newmodels != self.lastmdx_demuc_ensem:
            self.MDXLabel_to_path.clear()
            self.options_mdxnetModel_Optionmenu['menu'].delete(0, 'end')
            for file_name_1 in natsort.natsorted(new_MDXModels):
                if file_name_1.endswith(('.onnx')):
                    b = [".onnx"]
                    for char in b:
                        file_name_1 = file_name_1.replace(char, "")
                        
                    c = ["UVR_MDXNET_3_9662"]
                    for char in c:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 3") 
                        
                    d = ["UVR_MDXNET_2_9682"]
                    for char in d:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 2") 
                        
                    e = ["UVR_MDXNET_1_9703"]
                    for char in e:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 1") 
                        
                    f = ["UVR_MDXNET_9662"]
                    for char in f:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 3") 
                        
                    g = ["UVR_MDXNET_9682"]
                    for char in g:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 2") 
                        
                    h = ["UVR_MDXNET_9703"]
                    for char in h:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET 1") 
                        
                    i = ["UVR_MDXNET_KARA"]
                    for char in i:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET Karaoke") 
                        
                    j = ["UVR_MDXNET_Main"]
                    for char in j:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET Main") 
                        
                    k = ["UVR_MDXNET_Inst_1"]
                    for char in k:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET Inst 1") 
                        
                    l = ["UVR_MDXNET_Inst_2"]
                    for char in l:
                        file_name_1 = file_name_1.replace(char, "UVR-MDX-NET Inst 2") 
                    
                    self.options_mdxnetModel_Optionmenu['menu'].add_radiobutton(label=file_name_1,
                                                                        command=tk._setit(self.mdxnetModel_var, file_name_1))
                    
                    mdx_list_en = file_name_1
                    
                    self.ensemfiles = mdx_list_en
                    
                    with open('lib_v5/filelists/ensemble_list/mdx_demuc_en_list.txt', 'a') as f:
                        f.write("MDX-Net: {}\n".format(mdx_list_en))
                    
                    
            self.lastMDXModels = new_MDXModels
            
        """
        Loop through every Demucs model (.th, .pth) in the models directory
        and add to the select your model list
        """

        try:
            if new_DemucsModels != self.lastDemucsModels or newmodels != self.lastmdx_demuc_ensem:
                self.DemucsLabel_to_path.clear()
                self.options_DemucsModel_Optionmenu['menu'].delete(0, 'end')
                for file_name_2 in natsort.natsorted(new_DemucsModels):
                    if file_name_2.endswith(('.yaml', '.ckpt', '.gz', 'tasnet-beb46fac.th', 'tasnet_extra-df3777b2.th', 
                                             'demucs48_hq-28a1282c.th', 'demucs-e07c671f.th', 'demucs_extra-3646af93.th', 'demucs_unittest-09ebc15f.th', 
                                             'tasnet.th', 'tasnet_extra.th', 'demucs.th', 'demucs_extra.th', 'light.th', 'light_extra.th')):
                        #Demucs v3 Models
                        b = [".yaml"]
                        for char in b:
                            file_name_2 = file_name_2.replace(char, "") 
                        #Demucs v2 Models 
                        c = ["tasnet-beb46fac.th"]
                        for char in c:
                            file_name_2 = file_name_2.replace(char, "Tasnet v2")
                        d = ["tasnet_extra-df3777b2.th"]
                        for char in d:
                            file_name_2 = file_name_2.replace(char, "Tasnet_extra v2")
                        e = ["demucs48_hq-28a1282c.th"]
                        for char in e:
                            file_name_2 = file_name_2.replace(char, "Demucs48_hq v2")
                        f = ["demucs-e07c671f.th"]
                        for char in f:
                            file_name_2 = file_name_2.replace(char, "Demucs v2")
                        g = ["demucs_extra-3646af93.th"]
                        for char in g:
                            file_name_2 = file_name_2.replace(char, "Demucs_extra v2")
                        n = ["demucs_unittest-09ebc15f.th"]
                        for char in n:
                            file_name_2 = file_name_2.replace(char, "Demucs_unittest v2")
                        #Demucs v1 Models
                        h = ["tasnet.th"]
                        for char in h:
                            file_name_2 = file_name_2.replace(char, "Tasnet v1")
                        i = ["tasnet_extra.th"]
                        for char in i:
                            file_name_2 = file_name_2.replace(char, "Tasnet_extra v1")
                        j = ["demucs.th"]
                        for char in j:
                            file_name_2 = file_name_2.replace(char, "Demucs v1")
                        k = ["demucs_extra.th"]
                        for char in k:
                            file_name_2 = file_name_2.replace(char, "Demucs_extra v1")
                        l = ["light.th"]
                        for char in l:
                            file_name_2 = file_name_2.replace(char, "Light v1")
                        m = ["light_extra.th"]
                        for char in m:
                            file_name_2 = file_name_2.replace(char, "Light_extra v1")

                        
                        self.options_DemucsModel_Optionmenu['menu'].add_radiobutton(label=file_name_2,
                                                                            command=tk._setit(self.DemucsModel_var, file_name_2))
                        
                        demucs_list_en = file_name_2
                        
                        with open('lib_v5/filelists/ensemble_list/mdx_demuc_en_list.txt', 'a') as f:
                            f.writelines("Demucs: {}\n".format(demucs_list_en))
                        
                self.lastDemucsModels = new_DemucsModels
                
        except:
            pass
        
        try:
            if new_DemucsModels != self.lastDemucsModels or newmodels != self.lastmdx_demuc_ensem:
                #print(new_MDXModels)
                self.DemucsLabel_to_path.clear()
                self.options_DemucsModel_MDX_Optionmenu['menu'].delete(0, 'end')
                for file_name_3 in natsort.natsorted(new_DemucsModels):
                    if file_name_3.endswith(('.yaml', '.ckpt', '.gz', 'tasnet-beb46fac.th', 'tasnet_extra-df3777b2.th', 
                                             'demucs48_hq-28a1282c.th', 'demucs-e07c671f.th', 'demucs_extra-3646af93.th', 'demucs_unittest-09ebc15f.th', 
                                             'tasnet.th', 'tasnet_extra.th', 'demucs.th', 'demucs_extra.th', 'light.th', 'light_extra.th')):
                        #Demucs v3 Models
                        b = [".yaml"]
                        for char in b:
                            file_name_3 = file_name_3.replace(char, "") 
                        #Demucs v2 Models 
                        c = ["tasnet-beb46fac.th"]
                        for char in c:
                            file_name_3 = file_name_3.replace(char, "Tasnet v2")
                        d = ["tasnet_extra-df3777b2.th"]
                        for char in d:
                            file_name_3 = file_name_3.replace(char, "Tasnet_extra v2")
                        e = ["demucs48_hq-28a1282c.th"]
                        for char in e:
                            file_name_3 = file_name_3.replace(char, "Demucs48_hq v2")
                        f = ["demucs-e07c671f.th"]
                        for char in f:
                            file_name_3 = file_name_3.replace(char, "Demucs v2")
                        g = ["demucs_extra-3646af93.th"]
                        for char in g:
                            file_name_3 = file_name_3.replace(char, "Demucs_extra v2")
                        n = ["demucs_unittest-09ebc15f.th"]
                        for char in n:
                            file_name_3 = file_name_3.replace(char, "Demucs_unittest v2")
                        #Demucs v1 Models
                        h = ["tasnet.th"]
                        for char in h:
                            file_name_3 = file_name_3.replace(char, "Tasnet v1")
                        i = ["tasnet_extra.th"]
                        for char in i:
                            file_name_3 = file_name_3.replace(char, "Tasnet_extra v1")
                        j = ["demucs.th"]
                        for char in j:
                            file_name_3 = file_name_3.replace(char, "Demucs v1")
                        k = ["demucs_extra.th"]
                        for char in k:
                            file_name_3 = file_name_3.replace(char, "Demucs_extra v1")
                        l = ["light.th"]
                        for char in l:
                            file_name_3 = file_name_3.replace(char, "Light v1")
                        m = ["light_extra.th"]
                        for char in m:
                            file_name_3 = file_name_3.replace(char, "Light_extra v1")

                        
                        self.options_DemucsModel_MDX_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.DemucsModel_MDX_var, file_name_3))
                        
                self.lastDemucsModels = new_DemucsModels
                
        except:
            pass
        
        """
        Loop through every model param (.json) in the models directory
        and add to the select your model list
        """

        try:
            temp_ModelParams_dir = 'lib_v5\modelparams'  # nopep8
            new_ModelParams = os.listdir(temp_ModelParams_dir)
            
            if new_ModelParams != self.lastModelParams:
                self.ModelParamsLabel_to_path.clear()
                self.options_ModelParams_Optionmenu['menu'].delete(0, 'end')
                for file_name_3 in natsort.natsorted(new_ModelParams):
                    if file_name_3.endswith(('.json', 'Auto')):
                        
                        self.options_ModelParams_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.ModelParams_var, file_name_3))
                self.lastModelParams = new_ModelParams
        except:
            pass
        
        
        
        try:
            temp_ModelParams_dir = 'lib_v5\modelparams'  # nopep8
            new_ModelParams = os.listdir(temp_ModelParams_dir)
            
            if new_ModelParams != self.lastModelParams_ens:
                self.ModelParamsLabel_ens_to_path.clear()
                self.options_ModelParams_a_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_b_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_c_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_d_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_1_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_2_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_3_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_4_Optionmenu['menu'].delete(0, 'end')
                self.options_ModelParams_5_Optionmenu['menu'].delete(0, 'end')
                for file_name_3 in natsort.natsorted(new_ModelParams):
                    if file_name_3.endswith(('.json', 'Auto')):
                        
                        self.options_ModelParams_a_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_multi_USER_model_param_1, file_name_3))
                        self.options_ModelParams_b_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_multi_USER_model_param_2, file_name_3))
                        self.options_ModelParams_c_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_multi_USER_model_param_3, file_name_3))
                        self.options_ModelParams_d_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_multi_USER_model_param_4, file_name_3))
                        self.options_ModelParams_1_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_basic_USER_model_param_1, file_name_3))
                        self.options_ModelParams_2_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_basic_USER_model_param_2, file_name_3))
                        self.options_ModelParams_3_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_basic_USER_model_param_3, file_name_3))
                        self.options_ModelParams_4_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_basic_USER_model_param_4, file_name_3))
                        self.options_ModelParams_5_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.vr_basic_USER_model_param_5, file_name_3))
                self.lastModelParams_ens = new_ModelParams
        except:
            pass
        
        self.lastmdx_demuc_ensem = [new_DemucsModels, new_MDXModels]
            
    def update_states(self):
        """
        Vary the states for all widgets based
        on certain selections
        """

        if self.aiModel_var.get() == 'MDX-Net':
            # Place Widgets

            # Choose MDX-Net Model
            self.options_mdxnetModel_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            self.options_mdxnetModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            # MDX-chunks
            self.options_chunks_Label.place(x=12, y=0, width=0, height=-10,
                                        relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_chunks_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                        relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            # MDX-noisereduc_s
            self.options_noisereduc_s_Label.place(x=15, y=0, width=0, height=-10,
                                        relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_noisereduc_s_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                        relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #GPU Conversion
            self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
            self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Vocals Only
            self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
            self.options_voc_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Instrumental Only
            self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
            self.options_inst_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            # MDX-demucs Model
            self.options_demucsmodel_Checkbutton.configure(state=tk.NORMAL)
            self.options_demucsmodel_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            # MDX-Keep Non_Reduced Vocal
            self.options_non_red_Checkbutton.configure(state=tk.NORMAL)
            self.options_non_red_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Model Test Mode
            self.options_modelFolder_Checkbutton.configure(state=tk.NORMAL)
            self.options_modelFolder_Checkbutton.place(x=35, y=21, width=0, height=5,
                                                relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            # Forget widgets
            self.options_agg_Label.place_forget()
            self.options_agg_Optionmenu.place_forget()
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_demucs_stems_Label.place_forget()
            self.options_demucs_stems_Optionmenu.place_forget()
            self.options_DemucsModel_Label.place_forget()
            self.options_DemucsModel_Optionmenu.place_forget()
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_instrumentalModel_Label.place_forget()
            self.options_instrumentalModel_Optionmenu.place_forget()
            self.options_overlap_b_Label.place_forget()
            self.options_overlap_b_Optionmenu.place_forget()
            self.options_postpro_Checkbutton.configure(state=tk.DISABLED)
            self.options_postpro_Checkbutton.place_forget()
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_segment_Label.place_forget()
            self.options_segment_Optionmenu.place_forget()
            self.options_shifts_b_Label.place_forget()
            self.options_shifts_b_Optionmenu.place_forget()
            self.options_split_Checkbutton.configure(state=tk.DISABLED)
            self.options_split_Checkbutton.place_forget()
            self.options_tta_Checkbutton.configure(state=tk.DISABLED)
            self.options_tta_Checkbutton.place_forget()
            self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_b_Checkbutton.place_forget()
            self.options_winSize_Label.place_forget()
            self.options_winSize_Optionmenu.place_forget()
            
            
        elif self.aiModel_var.get() == 'VR Architecture':
            #Keep for Ensemble & VR Architecture Mode
            # Choose Main Model
            self.options_instrumentalModel_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            self.options_instrumentalModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            # WINDOW
            self.options_winSize_Label.place(x=13, y=0, width=0, height=-10,
                                        relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_winSize_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                        relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            # AGG
            self.options_agg_Label.place(x=15, y=0, width=0, height=-10,
                                        relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_agg_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                        relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #GPU Conversion
            self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
            self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Vocals Only
            self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
            self.options_voc_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Instrumental Only
            self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
            self.options_inst_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            # TTA
            self.options_tta_Checkbutton.configure(state=tk.NORMAL)
            self.options_tta_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Post-Process
            self.options_postpro_Checkbutton.configure(state=tk.NORMAL)
            self.options_postpro_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Save Image
            # self.options_image_Checkbutton.configure(state=tk.NORMAL)
            # self.options_image_Checkbutton.place(x=35, y=21, width=0, height=5,
            #                                     relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Model Test Mode
            self.options_modelFolder_Checkbutton.configure(state=tk.NORMAL)
            self.options_modelFolder_Checkbutton.place(x=35, y=21, width=0, height=5,
                                                relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Forget Widgets
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_chunks_Label.place_forget()
            self.options_chunks_Optionmenu.place_forget()
            self.options_demucs_stems_Label.place_forget()
            self.options_demucs_stems_Optionmenu.place_forget()
            self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
            self.options_demucsmodel_Checkbutton.place_forget()
            self.options_DemucsModel_Label.place_forget()
            self.options_DemucsModel_Optionmenu.place_forget()
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_mdxnetModel_Label.place_forget()
            self.options_mdxnetModel_Optionmenu.place_forget()
            self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
            self.options_noisereduc_Checkbutton.place_forget()
            self.options_noisereduc_s_Label.place_forget()
            self.options_noisereduc_s_Optionmenu.place_forget()
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.place_forget()
            self.options_non_red_Checkbutton.place_forget()
            self.options_overlap_b_Label.place_forget()
            self.options_overlap_b_Optionmenu.place_forget()
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_segment_Label.place_forget()
            self.options_segment_Optionmenu.place_forget()
            self.options_shifts_b_Label.place_forget()
            self.options_shifts_b_Optionmenu.place_forget()
            self.options_split_Checkbutton.configure(state=tk.DISABLED)
            self.options_split_Checkbutton.place_forget()
            self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_b_Checkbutton.place_forget()
            
        elif self.aiModel_var.get() == 'Demucs v3':
            #Keep for Ensemble & VR Architecture Mode
            # Choose Main Model
            self.options_DemucsModel_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            self.options_DemucsModel_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
            
            # Choose Stems
            self.options_demucs_stems_Label.place(x=13, y=0, width=0, height=-10,
                                        relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_demucs_stems_Optionmenu.place(x=55, y=-2, width=-85, height=7,
                                        relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            # Segment
            self.options_segment_Label.place(x=12, y=0, width=0, height=-10,
                                        relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_segment_Optionmenu.place(x=55, y=-2, width=-85, height=7,
                                        relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            # Shifts
            self.options_shifts_b_Label.place(x=12, y=0, width=0, height=-10,
                                        relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_shifts_b_Optionmenu.place(x=55, y=-2, width=-85, height=7,
                                        relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            # Overlap
            self.options_overlap_b_Label.place(x=13, y=0, width=0, height=-10,
                                        relx=2/3, rely=8/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_overlap_b_Optionmenu.place(x=55, y=-2, width=-85, height=7,
                                        relx=2/3, rely=9/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            #GPU Conversion
            self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
            self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Vocals Only
            self.options_voc_only_b_Checkbutton.configure(state=tk.NORMAL)
            self.options_voc_only_b_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            #Instrumental Only
            self.options_inst_only_b_Checkbutton.configure(state=tk.NORMAL)
            self.options_inst_only_b_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            #Split Mode
            self.options_split_Checkbutton.configure(state=tk.NORMAL)
            self.options_split_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=8/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

            # Forget Widgets
            # self.options_image_Checkbutton.configure(state=tk.DISABLED)
            # self.options_image_Checkbutton.place_forget()
            self.options_agg_Label.place_forget()
            self.options_agg_Optionmenu.place_forget()
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_chunks_Label.place_forget()
            self.options_chunks_Optionmenu.place_forget()
            self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
            self.options_demucsmodel_Checkbutton.place_forget()
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_inst_only_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_Checkbutton.place_forget()
            self.options_instrumentalModel_Label.place_forget()
            self.options_instrumentalModel_Optionmenu.place_forget()
            self.options_mdxnetModel_Label.place_forget()
            self.options_mdxnetModel_Optionmenu.place_forget()
            self.options_modelFolder_Checkbutton.place_forget()
            self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
            self.options_noisereduc_Checkbutton.place_forget()
            self.options_noisereduc_s_Label.place_forget()
            self.options_noisereduc_s_Optionmenu.place_forget()
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.place_forget()
            self.options_postpro_Checkbutton.configure(state=tk.DISABLED)
            self.options_postpro_Checkbutton.place_forget()
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_tta_Checkbutton.configure(state=tk.DISABLED)
            self.options_tta_Checkbutton.place_forget()
            self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_Checkbutton.place_forget()
            self.options_winSize_Label.place_forget()
            self.options_winSize_Optionmenu.place_forget()
            
        elif self.aiModel_var.get() == 'Ensemble Mode':
            if self.ensChoose_var.get() == 'Manual Ensemble':
                # Choose Algorithm
                self.options_algo_Label.place(x=20, y=0, width=0, height=-10,
                                        relx=1/3, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                self.options_algo_Optionmenu.place(x=12, y=-2, width=0, height=7,
                                        relx=1/3, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                # Choose Ensemble 
                self.options_ensChoose_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                self.options_ensChoose_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                # Forget Widgets
                # self.options_image_Checkbutton.configure(state=tk.DISABLED)
                # self.options_image_Checkbutton.place_forget()
                self.options_agg_Label.place_forget()
                self.options_agg_Optionmenu.place_forget()
                self.options_chunks_Label.place_forget()
                self.options_chunks_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
                self.options_demucsmodel_Checkbutton.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_gpu_Checkbutton.configure(state=tk.DISABLED)
                self.options_gpu_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_Checkbutton.place_forget()
                self.options_mdxnetModel_Label.place_forget()
                self.options_mdxnetModel_Optionmenu.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_noisereduc_s_Label.place_forget()
                self.options_noisereduc_s_Optionmenu.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_postpro_Checkbutton.configure(state=tk.DISABLED)
                self.options_postpro_Checkbutton.place_forget()
                self.options_save_Checkbutton.configure(state=tk.DISABLED)
                self.options_save_Checkbutton.place_forget()
                self.options_segment_Label.place_forget()
                self.options_segment_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
                self.options_tta_Checkbutton.configure(state=tk.DISABLED)
                self.options_tta_Checkbutton.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_Checkbutton.place_forget()
                self.options_winSize_Label.place_forget()
                self.options_winSize_Optionmenu.place_forget()
                

            elif self.ensChoose_var.get() == 'Multi-AI Ensemble':
                # Choose Ensemble 
                self.options_ensChoose_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                self.options_ensChoose_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                # MDX-chunks
                self.options_chunks_Label.place(x=12, y=0, width=0, height=-10,
                                            relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_chunks_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # MDX-noisereduc_s
                self.options_noisereduc_s_Label.place(x=15, y=0, width=0, height=-10,
                                            relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_noisereduc_s_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # WINDOW
                self.options_winSize_Label.place(x=13, y=-7, width=0, height=-10,
                                            relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_winSize_Optionmenu.place(x=71, y=-5, width=-118, height=7,
                                            relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # AGG
                self.options_agg_Label.place(x=15, y=-7, width=0, height=-10,
                                            relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_agg_Optionmenu.place(x=71, y=-5, width=-118, height=7,
                                            relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #GPU Conversion
                self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
                self.options_gpu_Checkbutton.place(x=35, y=3, width=0, height=5,
                                                relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Vocals Only
                self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_voc_only_Checkbutton.place(x=35, y=3, width=0, height=5,
                                            relx=1/3, rely=8/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Instrumental Only
                self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_inst_only_Checkbutton.place(x=35, y=3, width=0, height=5,
                                            relx=1/3, rely=9/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # MDX-demucs Model
                self.options_demucsmodel_Checkbutton.configure(state=tk.NORMAL)
                self.options_demucsmodel_Checkbutton.place(x=35, y=3, width=0, height=5,
                                            relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # TTA
                self.options_tta_Checkbutton.configure(state=tk.NORMAL)
                self.options_tta_Checkbutton.place(x=35, y=3, width=0, height=5,
                                                relx=2/3, rely=8/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Ensemble Save Outputs
                self.options_save_Checkbutton.configure(state=tk.NORMAL)
                self.options_save_Checkbutton.place(x=35, y=3, width=0, height=5,
                                            relx=2/3, rely=9/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # Forget Widgets
                # self.options_image_Checkbutton.configure(state=tk.DISABLED)
                # self.options_image_Checkbutton.place_forget()
                self.options_algo_Label.place_forget()
                self.options_algo_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_postpro_Checkbutton.configure(state=tk.DISABLED)
                self.options_postpro_Checkbutton.place_forget()
                self.options_segment_Label.place_forget()
                self.options_segment_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
            elif self.ensChoose_var.get() == 'Basic MD Ensemble':
                # Choose Ensemble 
                self.options_ensChoose_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                self.options_ensChoose_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                # MDX-chunks
                self.options_chunks_Label.place(x=12, y=0, width=0, height=-10,
                                            relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_chunks_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # MDX-noisereduc_s
                self.options_noisereduc_s_Label.place(x=15, y=0, width=0, height=-10,
                                            relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_noisereduc_s_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #GPU Conversion
                self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
                self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                                relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Vocals Only
                self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_voc_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Instrumental Only
                self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_inst_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # MDX-demucs Model
                self.options_demucsmodel_Checkbutton.configure(state=tk.NORMAL)
                self.options_demucsmodel_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                
                # Split Mode
                self.options_split_Checkbutton.configure(state=tk.NORMAL)
                self.options_split_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                
                #Ensemble Save Outputs
                self.options_save_Checkbutton.configure(state=tk.NORMAL)
                self.options_save_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                
                # Forget widgets
                # self.options_image_Checkbutton.configure(state=tk.DISABLED)
                # self.options_image_Checkbutton.place_forget()
                self.options_agg_Label.place_forget()
                self.options_agg_Optionmenu.place_forget()
                self.options_algo_Label.place_forget()
                self.options_algo_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_instrumentalModel_Label.place_forget()
                self.options_instrumentalModel_Optionmenu.place_forget()
                self.options_mdxnetModel_Label.place_forget()
                self.options_mdxnetModel_Optionmenu.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget() 
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_postpro_Checkbutton.configure(state=tk.DISABLED)
                self.options_postpro_Checkbutton.place_forget()
                self.options_segment_Label.place_forget()
                self.options_segment_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_tta_Checkbutton.configure(state=tk.DISABLED)
                self.options_tta_Checkbutton.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                self.options_winSize_Label.place_forget()
                self.options_winSize_Optionmenu.place_forget()
            else:
                # Choose Ensemble 
                self.options_ensChoose_Label.place(x=0, y=19, width=0, height=-10,
                                        relx=0, rely=6/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                self.options_ensChoose_Optionmenu.place(x=0, y=19, width=0, height=7,
                                        relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
                # WINDOW
                self.options_winSize_Label.place(x=13, y=0, width=0, height=-10,
                                            relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_winSize_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # AGG
                self.options_agg_Label.place(x=15, y=0, width=0, height=-10,
                                            relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                self.options_agg_Optionmenu.place(x=71, y=-2, width=-118, height=7,
                                            relx=2/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #GPU Conversion
                self.options_gpu_Checkbutton.configure(state=tk.NORMAL)
                self.options_gpu_Checkbutton.place(x=35, y=21, width=0, height=5,
                                                relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Vocals Only
                self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_voc_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Instrumental Only
                self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
                self.options_inst_only_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=1/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                # TTA
                self.options_tta_Checkbutton.configure(state=tk.NORMAL)
                self.options_tta_Checkbutton.place(x=35, y=21, width=0, height=5,
                                                relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Post-Process
                self.options_postpro_Checkbutton.configure(state=tk.NORMAL)
                self.options_postpro_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Save Image
                # self.options_image_Checkbutton.configure(state=tk.NORMAL)
                # self.options_image_Checkbutton.place(x=35, y=21, width=0, height=5,
                #                                     relx=2/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Ensemble Save Outputs
                self.options_save_Checkbutton.configure(state=tk.NORMAL)
                self.options_save_Checkbutton.place(x=35, y=21, width=0, height=5,
                                            relx=2/3, rely=7/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
                #Forget Widgets
                self.options_algo_Label.place_forget()
                self.options_algo_Optionmenu.place_forget()
                self.options_chunks_Label.place_forget()
                self.options_chunks_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
                self.options_demucsmodel_Checkbutton.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_instrumentalModel_Label.place_forget()
                self.options_mdxnetModel_Label.place_forget()
                self.options_mdxnetModel_Optionmenu.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_noisereduc_s_Label.place_forget()
                self.options_noisereduc_s_Optionmenu.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_segment_Label.place_forget()
                self.options_segment_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                
                
        if self.inst_only_var.get() == True:
            self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
            self.voc_only_var.set(False)
            #self.non_red_var.set(False)
        elif self.inst_only_var.get() == False:
            self.options_non_red_Checkbutton.configure(state=tk.NORMAL)
            self.options_voc_only_Checkbutton.configure(state=tk.NORMAL)
            
        if self.voc_only_var.get() == True:
            self.options_inst_only_Checkbutton.configure(state=tk.DISABLED)
            self.inst_only_var.set(False)
        elif self.voc_only_var.get() == False:
            self.options_inst_only_Checkbutton.configure(state=tk.NORMAL)
            
        if self.demucs_stems_var.get() == 'All Stems':
            self.voc_only_b_var.set(False)
            self.inst_only_b_var.set(False)
            self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
        elif self.demucs_stems_var.get() == 'Vocals':
            if self.inst_only_b_var.get() == True:
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.voc_only_b_var.set(False)
            elif self.inst_only_b_var.get() == False:
                self.options_voc_only_b_Checkbutton.configure(state=tk.NORMAL)
                
            if self.voc_only_b_var.get() == True:
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.inst_only_b_var.set(False)
            elif self.voc_only_b_var.get() == False:
                self.options_inst_only_b_Checkbutton.configure(state=tk.NORMAL)
        elif self.demucs_stems_var.get() == 'Other':
            if self.inst_only_b_var.get() == True:
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.voc_only_b_var.set(False)
            elif self.inst_only_b_var.get() == False:
                self.options_voc_only_b_Checkbutton.configure(state=tk.NORMAL)
                
            if self.voc_only_b_var.get() == True:
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.inst_only_b_var.set(False)
            elif self.voc_only_b_var.get() == False:
                self.options_inst_only_b_Checkbutton.configure(state=tk.NORMAL)
        elif self.demucs_stems_var.get() == 'Drums':
            if self.inst_only_b_var.get() == True:
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.voc_only_b_var.set(False)
            elif self.inst_only_b_var.get() == False:
                self.options_voc_only_b_Checkbutton.configure(state=tk.NORMAL)
            if self.voc_only_b_var.get() == True:
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.inst_only_b_var.set(False)
            elif self.voc_only_b_var.get() == False:
                self.options_inst_only_b_Checkbutton.configure(state=tk.NORMAL)
        elif self.demucs_stems_var.get() == 'Bass':
            if self.inst_only_b_var.get() == True:
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.voc_only_b_var.set(False)
            elif self.inst_only_b_var.get() == False:
                self.options_voc_only_b_Checkbutton.configure(state=tk.NORMAL)
                
            if self.voc_only_b_var.get() == True:
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.inst_only_b_var.set(False)
            elif self.voc_only_b_var.get() == False:
                self.options_inst_only_b_Checkbutton.configure(state=tk.NORMAL)
            
            
        if self.noisereduc_s_var.get() == 'None':
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.non_red_var.set(False)
        if not self.noisereduc_s_var.get() == 'None':
            self.options_non_red_Checkbutton.configure(state=tk.NORMAL)


        if self.autocompensate_var.get() == True:
            try:
                self.options_compensate.configure(state=tk.DISABLED)
            except:
                pass
            
        if self.autocompensate_var.get() == False:
            #self.compensate_var.set()
            try:
                self.options_compensate.configure(state=tk.NORMAL)
            except:
                pass

        if self.mdxnetModeltype_var.get() == 'Vocals (Default)':
            self.n_fft_scale_var.set('6144')
            self.dim_f_var.set('2048')
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.DISABLED)
                self.options_dim_f_Entry.configure(state=tk.DISABLED)
                self.options_n_fft_scale_Opt.configure(state=tk.DISABLED)
                self.options_dim_f_Opt.configure(state=tk.DISABLED)
            except:
                pass
            
        if self.mdxnetModeltype_var.get() == 'Other (Default)':
            self.n_fft_scale_var.set('8192')
            self.dim_f_var.set('2048')
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.DISABLED)
                self.options_dim_f_Entry.configure(state=tk.DISABLED)
                self.options_n_fft_scale_Opt.configure(state=tk.DISABLED)
                self.options_dim_f_Opt.configure(state=tk.DISABLED)
            except:
                pass

        if self.mdxnetModeltype_var.get() == 'Drums (Default)':
            self.n_fft_scale_var.set('4096')
            self.dim_f_var.set('2048')
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.DISABLED)
                self.options_dim_f_Entry.configure(state=tk.DISABLED)
                self.options_n_fft_scale_Opt.configure(state=tk.DISABLED)
                self.options_dim_f_Opt.configure(state=tk.DISABLED)
            except:
                pass
            
        if self.mdxnetModeltype_var.get() == 'Bass (Default)':
            self.n_fft_scale_var.set('16384')
            self.dim_f_var.set('2048')
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.DISABLED)
                self.options_dim_f_Entry.configure(state=tk.DISABLED)
                self.options_n_fft_scale_Opt.configure(state=tk.DISABLED)
                self.options_dim_f_Opt.configure(state=tk.DISABLED)
            except:
                pass
            
        if self.mdxnetModeltype_var.get() == 'Vocals (Custom)':
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.NORMAL)
                self.options_dim_f_Entry.configure(state=tk.NORMAL)
                self.options_n_fft_scale_Opt.configure(state=tk.NORMAL)
                self.options_dim_f_Opt.configure(state=tk.NORMAL)
            except:
                pass

        if self.mdxnetModeltype_var.get() == 'Other (Custom)':
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.NORMAL)
                self.options_dim_f_Entry.configure(state=tk.NORMAL)
                self.options_n_fft_scale_Opt.configure(state=tk.NORMAL)
                self.options_dim_f_Opt.configure(state=tk.NORMAL)
            except:
                pass

        if self.mdxnetModeltype_var.get() == 'Drums (Custom)':
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.NORMAL)
                self.options_dim_f_Entry.configure(state=tk.NORMAL)
                self.options_n_fft_scale_Opt.configure(state=tk.NORMAL)
                self.options_dim_f_Opt.configure(state=tk.NORMAL)
            except:
                pass
            
        if self.mdxnetModeltype_var.get() == 'Bass (Custom)':
            try:
                self.options_n_fft_scale_Entry.configure(state=tk.NORMAL)
                self.options_dim_f_Entry.configure(state=tk.NORMAL)
                self.options_n_fft_scale_Opt.configure(state=tk.NORMAL)
                self.options_dim_f_Opt.configure(state=tk.NORMAL)
            except:
                pass

        if self.selectdownload_var.get() == 'VR Arc':
            #self.modeldownload_var.clear()
            self.modeldownload_mdx_var.set('No Model Selected')
            self.modeldownload_demucs_var.set('No Model Selected')
            try:
                self.downloadmodelOptions.configure(state=tk.NORMAL)
                self.downloadmodelOptions_mdx.configure(state=tk.DISABLED)
                self.downloadmodelOptions_demucs.configure(state=tk.DISABLED)
            except:
                pass
        if self.selectdownload_var.get() == 'MDX-Net':
            #self.modeldownload_mdx_var.set('Full Model Pack')
            self.modeldownload_var.set('No Model Selected')
            self.modeldownload_demucs_var.set('No Model Selected')
            try:
                self.downloadmodelOptions.configure(state=tk.DISABLED)
                self.downloadmodelOptions_demucs.configure(state=tk.DISABLED)
                self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            except:
                pass
        if self.selectdownload_var.get() == 'Demucs':
            #self.modeldownload_demucs_var.set('Demucs v3: mdx')
            self.modeldownload_var.set('No Model Selected')
            self.modeldownload_mdx_var.set('No Model Selected')
            try:
                self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
                self.downloadmodelOptions.configure(state=tk.DISABLED)
                self.downloadmodelOptions_mdx.configure(state=tk.DISABLED)
            except:
                pass
            
        if self.no_chunk_d_var.get() == False:
            try:
                self.chunk_d_entry.configure(state=tk.DISABLED)
                self.margin_d_entry.configure(state=tk.DISABLED)
            except:
                pass
        elif self.no_chunk_d_var.get() == True:
            try:
                self.chunk_d_entry.configure(state=tk.NORMAL)
                self.margin_d_entry.configure(state=tk.NORMAL)
            except:
                pass

        if self.demucs_only_var.get() == True:
            self.demucsmodel_var.set(True)
            self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
        elif self.demucs_only_var.get() == False:
            self.options_demucsmodel_Checkbutton.configure(state=tk.NORMAL)

        self.update_inputPaths()

    def deselect_models(self):
        """
        Run this method on version change
        """
        if self.aiModel_var.get() == self.last_aiModel:
            return
        else:
            self.last_aiModel = self.aiModel_var.get()

        self.instrumentalModel_var.set('')
        self.ensChoose_var.set('Multi-AI Ensemble')
        self.mdxnetModel_var.set('UVR-MDX-NET 1')

        self.winSize_var.set(DEFAULT_DATA['window_size'])
        self.agg_var.set(DEFAULT_DATA['agg'])
        self.modelFolder_var.set(DEFAULT_DATA['modelFolder'])


        self.update_available_models()
        self.update_states()

    def restart(self):
        """
        Restart the application after asking for confirmation
        """
        confirm = tk.messagebox.askyesno(title='Restart Confirmation',
                message='This will restart the application and halt any running processes. Your current settings will be saved. \n\n Are you sure you wish to continue?')
        
        if confirm:
            self.save_values()
            try:
                subprocess.Popen(f'UVR_Launcher.exe')
            except:
                subprocess.Popen(f'python "{__file__}"', shell=True)
            exit()
        else:
            self.settings()
            pass
        
    def shutdown(self):
        """
        Shuts down the application after asking for confirmation
        """
        exit()

    def open_newModel_filedialog(self):
        """Let user paste an MDX-Net model to use for the vocal Separation"""
        
        filename = 'models\MDX_Net_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
            
    def delete_temps(self):  
        try:
            for basename in os.listdir(download_code_temp_dir):
                if basename.endswith('.aes'):
                    pathname = os.path.join(download_code_temp_dir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
                        
        except:
            pass
        
        try:
                        
            for basename in os.listdir(download_code_temp_dir):
                if basename.endswith('.txt'):
                    pathname = os.path.join(download_code_temp_dir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass
        
        try:
            srcdir = "models/Demucs_Models"

            for basename in os.listdir(srcdir):
                if basename.endswith('.tmp'):
                    pathname = os.path.join(srcdir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass   
        
        try:
            srcdir = "models/Demucs_Models/v3_repo"

            for basename in os.listdir(srcdir):
                if basename.endswith('.tmp'):
                    pathname = os.path.join(srcdir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass   

        try:
            srcdir = "models/Main_Models"

            for basename in os.listdir(srcdir):
                if basename.endswith('.tmp'):
                    pathname = os.path.join(srcdir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass  

        try:
            srcdir = "models/MDX_Net_Models"

            for basename in os.listdir(srcdir):
                if basename.endswith('.tmp'):
                    pathname = os.path.join(srcdir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass  
        
        try:
            srcdir = os.path.dirname(os.path.realpath(__file__))

            for basename in os.listdir(srcdir):
                if basename.endswith('.tmp'):
                    pathname = os.path.join(srcdir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        except:
            pass  
            
    def reset_to_defaults(self):
        self.agg_var.set(10)
        self.algo_var.set('Instrumentals (Min Spec)')
        self.appendensem_var.set(False)
        self.audfile_var.set(True)
        self.aud_mdx_var.set(True),
        self.autocompensate_var.set(True)
        self.channel_var.set(64)
        self.chunks_var.set('Auto')
        self.chunks_d_var.set('Full')
        self.compensate_var.set(1.03597672895)
        self.demucs_only_var.set(False)
        self.demucs_stems_var.set('All Stems')
        self.DemucsModel_var.set('mdx_extra')
        self.demucsmodel_var.set(False)
        self.DemucsModel_MDX_var.set('UVR_Demucs_Model_1')
        self.demucsmodel_sel_VR_var.set('UVR_Demucs_Model_1')
        self.demucsmodelVR_var.set(False)
        self.dim_f_var.set(2048)
        self.flactype_var.set('PCM_16')
        self.gpuConversion_var.set(False)
        self.inst_only_var.set(False)
        self.inst_only_b_var.set(False)
        self.margin_var.set(44100)
        self.margin_d_var.set(44100)
        self.mdxensemchoose_var.set('MDX-Net: UVR-MDX-NET Main')
        self.mdxensemchoose_b_var.set('No Model')
        self.mdx_only_ensem_a_var.set('MDX-Net: UVR-MDX-NET Main')
        self.mdx_only_ensem_b_var.set('MDX-Net: UVR-MDX-NET 1')
        self.mdx_only_ensem_c_var.set('No Model')
        self.mdx_only_ensem_d_var.set('No Model')
        self.mdx_only_ensem_e_var.set('No Model')  
        self.mdxnetModel_var.set('UVR-MDX-NET Main')        
        self.mdxnetModeltype_var.set('Vocals (Custom)')
        self.mixing_var.set('Default')
        self.modelFolder_var.set(False)
        self.instrumentalModel_var.set('')
        self.ModelParams_var.set('Auto')           
        self.mp3bit_var.set('320k')
        self.n_fft_scale_var.set(6144)
        self.no_chunk_var.set(False)
        self.no_chunk_d_var.set(True)
        self.noise_pro_select_var.set('Auto Select')
        self.noisereduc_var.set(True)
        self.noisereduc_s_var.set(3)
        self.non_red_var.set(False)
        self.nophaseinst_var.set(False)
        self.normalize_var.set(False)
        self.outputImage_var.set(False)
        self.overlap_var.set(0.25)
        self.overlap_b_var.set(0.25)
        self.postprocessing_var.set(False)
        self.save_var.set(True)
        self.saveFormat_var.set('Wav')
        self.segment_var.set(None)
        self.settest_var.set(False)
        self.shifts_var.set(2)
        self.shifts_b_var.set(2)
        self.split_mode_var.set(True)
        self.tta_var.set(False)
        self.voc_only_var.set(False)
        self.voc_only_b_var.set(False)
        self.vrensemchoose_var.set('2_HP-UVR')
        self.vrensemchoose_a_var.set('1_HP-UVR')
        self.vrensemchoose_b_var.set('2_HP-UVR')
        self.vrensemchoose_c_var.set('No Model')
        self.vrensemchoose_d_var.set('No Model')
        self.vrensemchoose_e_var.set('No Model')    
        self.vrensemchoose_mdx_a_var.set('No Model')
        self.vrensemchoose_mdx_b_var.set('No Model')
        self.vrensemchoose_mdx_c_var.set('No Model')
        self.vr_multi_USER_model_param_1.set('Auto')
        self.vr_multi_USER_model_param_2.set('Auto')
        self.vr_multi_USER_model_param_3.set('Auto')
        self.vr_multi_USER_model_param_4.set('Auto')
        self.vr_basic_USER_model_param_1.set('Auto')
        self.vr_basic_USER_model_param_2.set('Auto')
        self.vr_basic_USER_model_param_3.set('Auto')
        self.vr_basic_USER_model_param_4.set('Auto')
        self.vr_basic_USER_model_param_5.set('Auto')
        self.wavtype_var.set('PCM_16')
        self.winSize_var.set('512')
          
    def advanced_vr_options(self):
        """
        Open Advanced VR Options
        """     
           
        vr_opt=Toplevel(root)

        window_height = 630
        window_width = 500
        
        screen_width = vr_opt.winfo_screenwidth()
        screen_height = vr_opt.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        vr_opt.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        
        vr_opt.resizable(False, False)  # This code helps to disable windows from resizing
        
        x = root.winfo_x()
        y = root.winfo_y()
        vr_opt.geometry("+%d+%d" %(x+57,y+110))
        vr_opt.wm_transient(root)
        
        vr_opt.title("Advanced VR Options")
        
        #vr_opt.attributes("-topmost", True)

        # change title bar icon
        vr_opt.iconbitmap('img\\UVR-Icon-v2.ico')

        def close_win():
            vr_opt.destroy()
            self.settings()

        tabControl = ttk.Notebook(vr_opt)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Advanced Settings')
        tabControl.add(tab2, text ='Demucs Settings')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0,text="Advanced VR Options",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=tk.Label(frame0, text='Window Size (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.winSize_var, justify="center")
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Aggression Setting (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.agg_var, justify="center")
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Select Model Param\n(Can\'t change Model Params in Ensemble Mode)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        self.options_ModelParams_Optionmenu = l0=ttk.OptionMenu(frame0, self.ModelParams_var)  
        
        self.options_ModelParams_Optionmenu
        l0.grid(row=7,column=0,padx=0,pady=0) 
        
        l0=ttk.Checkbutton(frame0, text='Save Output Image(s) of Spectrogram(s)', variable=self.outputImage_var) 
        l0.grid(row=8,column=0,padx=0,pady=0)

        l0=ttk.Checkbutton(frame0, text='Demucs Model', variable=self.demucsmodelVR_var) 
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        def clear_cache():
            cachedir = "lib_v5/filelists/model_cache/vr_param_cache"

            for basename in os.listdir(cachedir):
                if basename.endswith('.txt'):
                    pathname = os.path.join(cachedir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        
        l0=ttk.Button(frame0,text='Clear Auto-Set Cache', command=clear_cache)
        l0.grid(row=10,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Open VR Models Folder', command=self.open_Modelfolder_vr)
        l0.grid(row=11,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=12,column=0,padx=0,pady=5)
        
        def close_win_self():
            vr_opt.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=13,column=0,padx=0,pady=5)
        
        self.ModelParamsLabel_to_path = defaultdict(lambda: '')
        self.lastModelParams = []
        
        frame0=Frame(tab2, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0,text='Demucs Model',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=5)
        
        l0=ttk.OptionMenu(frame0, self.demucsmodel_sel_VR_var, None, 'UVR_Demucs_Model_1', 'UVR_Demucs_Model_2', 'UVR_Demucs_Model_Bag')
        l0.grid(row=2,column=0,padx=0,pady=5) 
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=5)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_var, justify='center')
        l0.grid(row=4,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=5)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Segment', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=5)
        
        l0=ttk.Entry(frame0, textvariable=self.segment_var, justify='center')
        l0.grid(row=8,column=0,padx=0,pady=5)
        
        l0=ttk.Checkbutton(frame0, text='Split Mode', variable=self.split_mode_var) 
        l0.grid(row=9,column=0,padx=0,pady=5)
        
        #self.update_states()
          
    def advanced_demucs_options(self):
        """
        Open Advanced Demucs Options
        """
        demuc_opt= Toplevel(root)

        window_height = 750
        window_width = 500
        
        demuc_opt.title("Advanced Demucs Options")
        
        demuc_opt.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = demuc_opt.winfo_screenwidth()
        screen_height = demuc_opt.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        demuc_opt.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        
        #demuc_opt.attributes("-topmost", True)
        
        x = root.winfo_x()
        y = root.winfo_y()
        demuc_opt.geometry("+%d+%d" %(x+57,y+45))
        demuc_opt.wm_transient(root)

        # change title bar icon
        demuc_opt.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            demuc_opt.destroy()
            self.settings()

        tabControl = ttk.Notebook(demuc_opt)
        
        tabControl.pack(expand = 1, fill ="both")
        
        tabControl.grid_rowconfigure(0, weight=1)
        tabControl.grid_columnconfigure(0, weight=1)

        frame0=Frame(tabControl, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0,text="Advanced Demucs Options",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9", width=50)
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_b_var, justify='center')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_b_var, justify='center')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Segment', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.segment_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Chunks (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        self.chunk_d_entry=ttk.Entry(frame0, textvariable=self.chunks_d_var, justify='center')
        self.chunk_d_entry.grid(row=8,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Chunk Margin', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=0,padx=0,pady=10)
        
        self.margin_d_entry=ttk.Entry(frame0, textvariable=self.margin_d_var, justify='center')
        self.margin_d_entry.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Enable Chunks', variable=self.no_chunk_d_var) 
        l0.grid(row=11,column=0,padx=0,pady=10)
        
        l0=ttk.Checkbutton(frame0, text='Save Stems to Model & Track Name Directory', variable=self.audfile_var) 
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Open Demucs Model Folder', command=self.open_Modelfolder_de)
        l0.grid(row=13,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=14,column=0,padx=0,pady=0)
        
        def close_win_self():
            demuc_opt.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=15,column=0,padx=0,pady=10)
        
        l0=ttk.Label(frame0,text='\n')
        l0.grid(row=16,column=0,padx=0,pady=50)
        
        self.update_states()
        
    def advanced_mdx_options(self):
        """
        Open Advanced MDX Options
        """
        mdx_net_opt= Toplevel(root)

        window_height = 740
        window_width = 550
        
        mdx_net_opt.title("Advanced MDX-Net Options")
        
        mdx_net_opt.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = mdx_net_opt.winfo_screenwidth()
        screen_height = mdx_net_opt.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        mdx_net_opt.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        
        x = root.winfo_x()
        y = root.winfo_y()
        mdx_net_opt.geometry("+%d+%d" %(x+35,y+45))
        mdx_net_opt.wm_transient(root)

        # change title bar icon
        mdx_net_opt.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            mdx_net_opt.destroy()
            self.settings()

        tabControl = ttk.Notebook(mdx_net_opt)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Advanced Settings')
        tabControl.add(tab2, text ='Demucs Settings')
        tabControl.add(tab3, text ='Advanced ONNX Model Settings')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        tab3.grid_rowconfigure(0, weight=1)
        tab3.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0,text="Advanced MDX-Net Options",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=tk.Label(frame0, text='Chunks (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.chunks_var, justify='center')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Noise Reduction (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.noisereduc_s_var, justify='center')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Chunk Margin', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.margin_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Volume Compensation', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        self.options_compensate = l0=ttk.Entry(frame0, textvariable=self.compensate_var, justify='center')
        
        self.options_compensate
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Autoset Volume Compensation', variable=self.autocompensate_var) 
        l0.grid(row=9,column=0,padx=0,pady=5)
        
        l0=ttk.Checkbutton(frame0, text='Reduce Instrumental Noise Separately', variable=self.nophaseinst_var) 
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Noise Profile', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=11,column=0,padx=0,pady=5)
        
        l0=ttk.OptionMenu(frame0, self.noise_pro_select_var, None, 'Auto Select', 'MDX-NET_Noise_Profile_14_kHz', 'MDX-NET_Noise_Profile_17_kHz', 'MDX-NET_Noise_Profile_Full_Band')
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Open MDX-Net Models Folder', command=self.open_newModel_filedialog)
        l0.grid(row=13,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=14,column=0,padx=0,pady=0)
        
        def close_win_self():
            mdx_net_opt.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=15,column=0,padx=0,pady=10)
        
        frame0=Frame(tab2, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0, text='Choose Demucs Model', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=0,column=0,padx=0,pady=0)
        
        l0=tk.Button(frame0, text='(Click here to add more Demucs models)', font=("Century Gothic", "8"), foreground='#13a4c9', borderwidth=0, command=self.open_Modelfolder_de)
        l0.grid(row=1,column=0,padx=0,pady=0)
        
        self.options_DemucsModel_MDX_Optionmenu = l0=ttk.OptionMenu(frame0, self.DemucsModel_MDX_var)  
        
        self.options_DemucsModel_MDX_Optionmenu
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Mixing Algorithm', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=10)
        
        l0=ttk.OptionMenu(frame0, self.mixing_var, None, 'Default', 'Min_Mag', 'Max_Mag', 'Invert_p')   
        l0.grid(row=4,column=0,padx=0,pady=0) 
        
        l0=tk.Label(frame0, text='Segments\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.segment_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_var, justify='center')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_var, justify='center')
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Split Mode', variable=self.split_mode_var) 
        l0.grid(row=11,column=0,padx=0,pady=10)
        
        l0=ttk.Checkbutton(frame0, text='Enable Chunks', variable=self.no_chunk_var) 
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        self.update_states()
        
        frame0=Frame(tab3, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0, text=f'{space_small}Stem Type{space_small}', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxnetModeltype_var, None, 'Vocals (Custom)', 'Instrumental (Custom)', 'Other (Custom)', 'Bass (Custom)', 'Drums (Custom)')
        l0.grid(row=2,column=0,padx=0,pady=10)

        l0=tk.Label(frame0, text='N_FFT Scale', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='(Manual Set)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=1,padx=0,pady=0)
        
        self.options_n_fft_scale_Opt = l0=ttk.OptionMenu(frame0, self.n_fft_scale_var, None, '4096', '6144', '7680', '8192', '16384')
        
        self.options_n_fft_scale_Opt
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        self.options_n_fft_scale_Entry = l0=ttk.Entry(frame0, textvariable=self.n_fft_scale_var, justify='center')
        
        self.options_n_fft_scale_Entry
        l0.grid(row=4,column=1,padx=0,pady=0)

        l0=tk.Label(frame0, text='Dim_f', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='(Manual Set)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=1,padx=0,pady=0)
        
        self.options_dim_f_Opt = l0=ttk.OptionMenu(frame0, self.dim_f_var, None, '2048', '3072', '4096')
        
        self.options_dim_f_Opt
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        self.options_dim_f_Entry = l0=ttk.Entry(frame0, textvariable=self.dim_f_var, justify='center')
        
        self.options_dim_f_Entry
        l0.grid(row=6,column=1,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Auto-Set', variable=self.aud_mdx_var) 
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        
        def clear_cache():
            cachedir = "lib_v5/filelists/model_cache/mdx_model_cache"

            for basename in os.listdir(cachedir):
                if basename.endswith('.json'):
                    pathname = os.path.join(cachedir, basename)
                    if os.path.isfile(pathname):
                        os.remove(pathname)
        
        l0=ttk.Button(frame0, text='Clear Auto-Set Cache', command=clear_cache)
        
        l0.grid(row=8,column=0,padx=0,pady=10)
        
        self.DemucsLabel_to_path = defaultdict(lambda: '')
        self.lastmdx_demuc_ensem = []
        
        self.update_states()
        
    def custom_ensemble(self):
        """
        Open Ensemble Custom
        """
        custom_ens_opt= Toplevel(root)

        window_height = 680
        window_width = 900
        
        custom_ens_opt.title("Customize Ensemble")
        
        custom_ens_opt.resizable(False, False)  # This code helps to disable windows from resizing
        
        x = root.winfo_x()
        y = root.winfo_y()
        custom_ens_opt.geometry("+%d+%d" %(x+57,y+100))
        custom_ens_opt.wm_transient(root)
        
        screen_width = custom_ens_opt.winfo_screenwidth()
        screen_height = custom_ens_opt.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        custom_ens_opt.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        x = root.winfo_x()
        y = root.winfo_y()
        custom_ens_opt.geometry("+%d+%d" %(x-140,y+70))
        custom_ens_opt.wm_transient(root)

        # change title bar icon
        custom_ens_opt.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            custom_ens_opt.destroy()
            self.settings()

        tabControl = ttk.Notebook(custom_ens_opt)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Ensemble Options')
        tabControl.add(tab2, text ='More Options')
        tabControl.add(tab3, text ='VR Model Param Settings')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        
        tab3.grid_rowconfigure(0, weight=1)
        tab3.grid_columnconfigure(0, weight=1)
        
        mdx_demuc_en = ''
        mdx_demuc_en = lib_v5.filelist.get_mdx_demucs_en_list(mdx_demuc_en)
        mdx_demuc_en_list = mdx_demuc_en
        
        vr_en = ''
        vr_en = lib_v5.filelist.get_vr_en_list(vr_en)
        vr_en_list = vr_en

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  

        mdx_only_ensem_a = self.mdx_only_ensem_a_var.get()
        mdx_only_ensem_b = self.mdx_only_ensem_b_var.get()
        mdx_only_ensem_c = self.mdx_only_ensem_c_var.get()
        mdx_only_ensem_d = self.mdx_only_ensem_d_var.get()
        mdx_only_ensem_e = self.mdx_only_ensem_e_var.get()
        mdxensemchoose_b = self.mdxensemchoose_b_var.get()
        mdxensemchoose = self.mdxensemchoose_var.get()
        vrensemchoose_a = self.vrensemchoose_a_var.get()
        vrensemchoose_b = self.vrensemchoose_b_var.get()
        vrensemchoose_c = self.vrensemchoose_c_var.get()
        vrensemchoose_d = self.vrensemchoose_d_var.get()
        vrensemchoose_e = self.vrensemchoose_e_var.get()
        vrensemchoose_mdx_a = self.vrensemchoose_mdx_a_var.get()
        vrensemchoose_mdx_b = self.vrensemchoose_mdx_b_var.get()
        vrensemchoose_mdx_c = self.vrensemchoose_mdx_c_var.get()
        vrensemchoose = self.vrensemchoose_var.get()

        l0=tk.Label(frame0,text="Multi-AI Ensemble Options",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=0,padx=20,pady=10)
        
        l0=tk.Label(frame0,text=f"{space_small}MDX-Net or Demucs Model 1{space_small}\n",font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxensemchoose_var, *mdx_demuc_en_list)
        l0.grid(row=3,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxensemchoose_b_var, *mdx_demuc_en_list)
        l0.grid(row=5,column=0,padx=0,pady=0)

        l0=tk.Label(frame0,text='\nVR Model 1\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_var, *vr_en_list)
        l0.grid(row=7,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_a_var, *vr_en_list)
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_b_var, *vr_en_list)
        l0.grid(row=11,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_c_var, *vr_en_list)
        l0.grid(row=13,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text="Basic VR Ensemble Options",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=1,padx=20,pady=10)
        
        l0=tk.Label(frame0,text=f'{space_medium}VR Model 1{space_medium}\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_a_var, *vr_en_list)
        l0.grid(row=3,column=1,padx=0,pady=0)

        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_b_var, *vr_en_list)
        l0.grid(row=5,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_c_var, *vr_en_list)
        l0.grid(row=7,column=1,padx=0,pady=0) 
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_d_var, *vr_en_list)
        l0.grid(row=9,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 5\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_e_var, *vr_en_list)
        l0.grid(row=11,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text="Basic MD Ensemble Options",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=2,padx=20,pady=10)
        
        l0=tk.Label(frame0,text=f'{space_small}MDX-Net or Demucs Model 1{space_small}\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=2,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdx_only_ensem_a_var, *mdx_demuc_en_list)
        l0.grid(row=3,column=2,padx=0,pady=0)

        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=2,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdx_only_ensem_b_var, *mdx_demuc_en_list)
        l0.grid(row=5,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=2,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdx_only_ensem_c_var, *mdx_demuc_en_list)
        l0.grid(row=7,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=2,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdx_only_ensem_d_var, *mdx_demuc_en_list)
        l0.grid(row=9,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 5\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=2,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdx_only_ensem_e_var, *mdx_demuc_en_list)
        l0.grid(row=11,column=2,padx=0,pady=0)
        
        def close_win_self():
            custom_ens_opt.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=13,column=1,padx=20,pady=0)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=13,column=2,padx=20,pady=0)
        
        self.mdx_only_ensem_a_var.set(mdx_only_ensem_a)
        self.mdx_only_ensem_b_var.set(mdx_only_ensem_b)
        self.mdx_only_ensem_c_var.set(mdx_only_ensem_c)
        self.mdx_only_ensem_d_var.set(mdx_only_ensem_d)
        self.mdx_only_ensem_e_var.set(mdx_only_ensem_e)
        self.mdxensemchoose_b_var.set(mdxensemchoose_b)
        self.mdxensemchoose_var.set(mdxensemchoose)
        self.vrensemchoose_a_var.set(vrensemchoose_a)
        self.vrensemchoose_b_var.set(vrensemchoose_b)
        self.vrensemchoose_c_var.set(vrensemchoose_c)
        self.vrensemchoose_d_var.set(vrensemchoose_d)
        self.vrensemchoose_e_var.set(vrensemchoose_e)
        self.vrensemchoose_mdx_a_var.set(vrensemchoose_mdx_a)
        self.vrensemchoose_mdx_b_var.set(vrensemchoose_mdx_b)
        self.vrensemchoose_mdx_c_var.set(vrensemchoose_mdx_c)
        self.vrensemchoose_var.set(vrensemchoose)
        
        frame0=Frame(tab2, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0,text="Additional Options",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=1,padx=0,pady=10)
        
        l0=ttk.Checkbutton(frame0, text='Append Ensemble Name to Final Output', variable=self.appendensem_var) 
        l0.grid(row=2,column=1,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Open Models Directory', command=self.open_Modelfolder_filedialog)
        l0.grid(row=3,column=1,padx=0,pady=5)
        
        l0=tk.Label(frame0,text='Additional VR Architecture Options',font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=5,column=1,padx=0,pady=5)
        
        l0=ttk.Checkbutton(frame0, text='Post-Process', variable=self.postprocessing_var) 
        l0.grid(row=6,column=1,padx=0,pady=5)
        
        l0=ttk.Checkbutton(frame0, text='Save Output Image(s) of Spectrogram(s)', variable=self.outputImage_var) 
        l0.grid(row=7,column=1,padx=0,pady=5)
        
        l0=tk.Label(frame0,text='Additional Demucs Options',font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=8,column=1,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=1,padx=0,pady=5)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_var, justify='center')
        l0.grid(row=10,column=1,padx=0,pady=5)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=11,column=1,padx=0,pady=5)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_var, justify='center')
        l0.grid(row=12,column=1,padx=0,pady=5)
        
        l0=ttk.Checkbutton(frame0, text='Split Mode', variable=self.split_mode_var) 
        l0.grid(row=13,column=1,padx=0,pady=0)
        
        frame0=Frame(tab3, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  

        def set_auto():
            self.vr_multi_USER_model_param_1.set('Auto')
            self.vr_multi_USER_model_param_2.set('Auto')
            self.vr_multi_USER_model_param_3.set('Auto')
            self.vr_multi_USER_model_param_4.set('Auto')
            self.vr_basic_USER_model_param_1.set('Auto')
            self.vr_basic_USER_model_param_2.set('Auto')
            self.vr_basic_USER_model_param_3.set('Auto')
            self.vr_basic_USER_model_param_4.set('Auto')
            self.vr_basic_USER_model_param_5.set('Auto')

        l0=tk.Label(frame0,text="Multi-AI Ensemble VR Model Params",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=0,padx=20,pady=10)

        l0=tk.Label(frame0,text='\nVR Model 1\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        self.options_ModelParams_a_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_multi_USER_model_param_1)
        
        self.options_ModelParams_a_Optionmenu
        l0.grid(row=3,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        self.options_ModelParams_b_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_multi_USER_model_param_2)
        
        self.options_ModelParams_b_Optionmenu
        l0.grid(row=5,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        self.options_ModelParams_c_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_multi_USER_model_param_3)
        
        self.options_ModelParams_c_Optionmenu
        l0.grid(row=7,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        self.options_ModelParams_d_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_multi_USER_model_param_4)
        
        self.options_ModelParams_d_Optionmenu
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Set All to Auto', command=set_auto)
        l0.grid(row=11,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text="Basic VR Ensemble Model Params",font=("Century Gothic", "11", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=1,padx=20,pady=10)
        
        l0=tk.Label(frame0,text=f'{space_medium}VR Model 1{space_medium}\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=1,padx=0,pady=0)
        
        self.options_ModelParams_1_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_basic_USER_model_param_1)
        
        self.options_ModelParams_1_Optionmenu
        l0.grid(row=3,column=1,padx=0,pady=0)

        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=1,padx=0,pady=0)
        
        self.options_ModelParams_2_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_basic_USER_model_param_2)
        
        self.options_ModelParams_2_Optionmenu
        l0.grid(row=5,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=1,padx=0,pady=0)
        
        self.options_ModelParams_3_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_basic_USER_model_param_3)
        
        self.options_ModelParams_3_Optionmenu
        l0.grid(row=7,column=1,padx=0,pady=0) 
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=1,padx=0,pady=0)
        
        self.options_ModelParams_4_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_basic_USER_model_param_4)
        
        self.options_ModelParams_4_Optionmenu
        l0.grid(row=9,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 5\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=1,padx=0,pady=0)
        
        self.options_ModelParams_5_Optionmenu = l0=ttk.OptionMenu(frame0, self.vr_basic_USER_model_param_5)
        
        self.options_ModelParams_5_Optionmenu
        l0.grid(row=11,column=1,padx=0,pady=0)
        
        self.ModelParamsLabel_ens_to_path = defaultdict(lambda: '')
        self.lastModelParams_ens = []
        
        self.update_states()

    def help(self):
        """
        Open Help Guide
        """
        help_guide_opt = Toplevel(self)
        if GetSystemMetrics(1) >= 900:
            window_height = 810
            window_width = 1080
        elif GetSystemMetrics(1) <= 720:
            window_height = 640
            window_width = 930
        else:
            window_height = 670
            window_width = 930
        help_guide_opt.title("UVR Help Guide")
        
        help_guide_opt.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = help_guide_opt.winfo_screenwidth()
        screen_height = help_guide_opt.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        help_guide_opt.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        if GetSystemMetrics(1) >= 900:
            x = root.winfo_x()
            y = root.winfo_y()
            help_guide_opt.geometry("+%d+%d" %(x-220,y+5))
            help_guide_opt.wm_transient(root)

        # change title bar icon
        help_guide_opt.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            help_guide_opt.destroy()
            self.settings()

        tabControl = ttk.Notebook(help_guide_opt)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)
        tab5 = ttk.Frame(tabControl)
        tab6 = ttk.Frame(tabControl)
        tab7 = ttk.Frame(tabControl)
        tab8 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='General')
        tabControl.add(tab2, text ='VR Architecture')
        tabControl.add(tab3, text ='MDX-Net')
        tabControl.add(tab4, text ='Demucs v3')
        tabControl.add(tab5, text ='Ensemble Mode')
        tabControl.add(tab6, text ='Manual Ensemble')
        tabControl.add(tab7, text ='More Info')
        tabControl.add(tab8, text ='Credits')

        tabControl.pack(expand = 1, fill ="both")

        #Configure the row/col of our frame and root window to be resizable and fill all available space
        
        tab7.grid_rowconfigure(0, weight=1)
        tab7.grid_columnconfigure(0, weight=1)
        
        tab8.grid_rowconfigure(0, weight=1)
        tab8.grid_columnconfigure(0, weight=1)
        
        tk.Button(tab1, image=self.gen_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)

        tk.Button(tab2, image=self.vr_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)

        tk.Button(tab3, image=self.mdx_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)
        
        tk.Button(tab4, image=self.demucs_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)

        tk.Button(tab5, image=self.ense_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)

        tk.Button(tab6, image=self.user_ens_opt_img, borderwidth=0, command=close_win).grid(column = 0,
                                    row = 0, 
                                    padx = 87,
                                    pady = 30)

        #frame0
        frame0=Frame(tab7,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  

        if GetSystemMetrics(1) >= 900:
            l0=tk.Label(frame0,text="Notes",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=20,pady=10)

            l0=tk.Label(frame0,text="UVR is 100% free and open-source but MIT licensed.\nAll the models provided as part of UVR were trained by its core developers.\nPlease credit the core UVR developers if you choose to use any of our models or code for projects unrelated to UVR.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=2,column=0,padx=10,pady=7)
            
            l0=tk.Label(frame0,text="Resources",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=20,pady=7, sticky=N)
            
            link = Label(frame0, text="Ultimate Vocal Remover (Official GitHub)",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui"))
            
            l0=tk.Label(frame0,text="You can find updates, report issues, and give us a shout via our official GitHub.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=5,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="SoX - Sound eXchange",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=6,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download"))
            
            l0=tk.Label(frame0,text="UVR relies on SoX for Noise Reduction. It's automatically included via the UVR installer but not the developer build.\nIf you are missing SoX, please download it via the link and extract the SoX archive to the following directory - lib_v5/sox",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=7,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="FFmpeg",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=8,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://www.wikihow.com/Install-FFmpeg-on-Windows"))
            
            l0=tk.Label(frame0,text="UVR relies on FFmpeg for processing non-wav audio files.\nIt's automatically included via the UVR installer but not the developer build.\nIf you are missing FFmpeg, please see the installation guide via the link provided.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=9,column=0,padx=10,pady=7)

            link = Label(frame0, text="X-Minus AI",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=10,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://x-minus.pro/ai"))

            l0=tk.Label(frame0,text="Many of the models provided are also on X-Minus.\nThis resource primarily benefits users without the computing resources to run the GUI or models locally.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=11,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="Official UVR Patreon",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=12,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://www.patreon.com/uvr"))
            
            l0=tk.Label(frame0,text="If you wish to support and donate to this project, click the link above and become a Patreon!\nOfficial UVR Patreons will receive VIP access to additional models as well as pre-releases.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=13,column=0,padx=10,pady=7)
            
            frame0=Frame(tab8,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)

            #inside frame0    
            
            l0=tk.Label(frame0,text="Core UVR Developers",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=0,column=0,padx=20,pady=5, sticky=N)
            
            l0=tk.Label(frame0,image=self.credits_img,font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=1,column=0,padx=10,pady=5)

            l0=tk.Label(frame0,text="Anjok07\nAufr33",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=2,column=0,padx=10,pady=5)

            l0=tk.Label(frame0,text="Special Thanks",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=4,column=0,padx=20,pady=10)

            l0=tk.Label(frame0,text="DilanBoskan",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=5,column=0,padx=10,pady=5)

            l0=tk.Label(frame0,text="Your contributions at the start of this project were essential to the success of UVR. Thank you!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=6,column=0,padx=0,pady=0)

            link = Label(frame0, text="Tsurumeso",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=7,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/tsurumeso/vocal-remover"))
            
            l0=tk.Label(frame0,text="Developed the original VR Architecture AI code.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Kuielab & Woosung Choi",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=9,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/kuielab"))
            
            l0=tk.Label(frame0,text="Developed the original MDX-Net AI code.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            l0=tk.Label(frame0,text="Bas Curtiz",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=11,column=0,padx=10,pady=5)
            
            l0=tk.Label(frame0,text="Designed the official UVR logo, icon, banner, splash screen, and interface.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=12,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Adefossez & Demucs",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=13,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/facebookresearch/demucs"))
            
            l0=tk.Label(frame0,text="Core developer of Facebook's Demucs Music Source Separation.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=14,column=0,padx=0,pady=0)
            
            l0=tk.Label(frame0,text="Audio Separation Discord Community",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=15,column=0,padx=10,pady=5)
            
            l0=tk.Label(frame0,text="Thank you for the support!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=16,column=0,padx=0,pady=0)
            
            l0=tk.Label(frame0,text="CC Karokee & Friends Discord Community",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=17,column=0,padx=10,pady=5)
            
            l0=tk.Label(frame0,text="Thank you for the support!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=18,column=0,padx=0,pady=0)
            
        else:
            l0=tk.Label(frame0,text="Notes",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=5,pady=5)

            l0=tk.Label(frame0,text="UVR is 100% free and open-source but MIT licensed.\nPlease credit the core UVR developers if you choose to use any of our models or code for projects unrelated to UVR.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=2,column=0,padx=5,pady=5)
            
            l0=tk.Label(frame0,text="Resources",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=5,pady=5, sticky=N)
            
            link = Label(frame0, text="Ultimate Vocal Remover (Official GitHub)",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui"))
            
            l0=tk.Label(frame0,text="You can find updates, report issues, and give us a shout via our official GitHub.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=5,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="SoX - Sound eXchange",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=6,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download"))
            
            l0=tk.Label(frame0,text="UVR relies on SoX for Noise Reduction. It's automatically included via the UVR installer but not the developer build.\nIf you are missing SoX, please download it via the link and extract the SoX archive to the following directory - lib_v5/sox",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=7,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="FFmpeg",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=8,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://www.wikihow.com/Install-FFmpeg-on-Windows"))
            
            l0=tk.Label(frame0,text="UVR relies on FFmpeg for processing non-wav audio files.\nIf you are missing FFmpeg, please see the installation guide via the link provided.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=9,column=0,padx=5,pady=5)

            link = Label(frame0, text="X-Minus AI",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=10,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://x-minus.pro/ai"))

            l0=tk.Label(frame0,text="Many of the models provided are also on X-Minus.\nThis resource primarily benefits users without the computing resources to run the GUI or models locally.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=11,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="Official UVR Patreon",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=12,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://www.patreon.com/uvr"))
            
            l0=tk.Label(frame0,text="If you wish to support and donate to this project, click the link above and become a Patreon!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=13,column=0,padx=5,pady=5)
            
            frame0=Frame(tab8,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)

            #inside frame0    
            
            l0=tk.Label(frame0,text="Core UVR Developers",font=("Century Gothic", "12", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=0,column=0,padx=20,pady=5, sticky=N)
            
            l0=tk.Label(frame0,image=self.credits_img,font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=1,column=0,padx=5,pady=5)

            l0=tk.Label(frame0,text="Anjok07\nAufr33",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=2,column=0,padx=5,pady=5)

            l0=tk.Label(frame0,text="Special Thanks",font=("Century Gothic", "10", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=4,column=0,padx=20,pady=10)

            l0=tk.Label(frame0,text="DilanBoskan",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=5,column=0,padx=5,pady=5)

            l0=tk.Label(frame0,text="Your contributions at the start of this project were essential to the success of UVR. Thank you!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=6,column=0,padx=0,pady=0)

            link = Label(frame0, text="Tsurumeso",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=7,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/tsurumeso/vocal-remover"))
            
            l0=tk.Label(frame0,text="Developed the original VR Architecture AI code.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Kuielab & Woosung Choi",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=9,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/kuielab"))
            
            l0=tk.Label(frame0,text="Developed the original MDX-Net AI code.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            l0=tk.Label(frame0,text="Bas Curtiz",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=11,column=0,padx=5,pady=5)
            
            l0=tk.Label(frame0,text="Designed the official UVR logo, icon, banner, splash screen, and interface.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=12,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Adefossez & Demucs",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=13,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/facebookresearch/demucs"))
            
            l0=tk.Label(frame0,text="Core developer of Facebook's Demucs Music Source Separation.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=14,column=0,padx=0,pady=0)
            
            l0=tk.Label(frame0,text="Audio Separation and CC Karokee & Friends Discord Communities",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=15,column=0,padx=5,pady=5)
            
            l0=tk.Label(frame0,text="Thank you for the support!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=16,column=0,padx=0,pady=0)   
            
    def settings(self, choose=False):
        """
        Open Settings
        """
        
        self.delete_temps()

        update_var = tk.StringVar(value='')
        update_button_var = tk.StringVar(value='Check for Updates')
        update_set_var = tk.StringVar(value='UVR Version Current')

        settings_menu = Toplevel(self)

        window_height = 780
        window_width = 500
        
        settings_menu.title("Settings Guide")
        
        settings_menu.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = settings_menu.winfo_screenwidth()
        screen_height = settings_menu.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        settings_menu.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        x = root.winfo_x()
        y = root.winfo_y()
        settings_menu.geometry("+%d+%d" %(x+57,y+15))
        settings_menu.wm_transient(root)

        # change title bar icon
        settings_menu.iconbitmap('img\\UVR-Icon-v2.ico')

        def askyesorno():
            """
            Ask to Update
            """
            
            top_dialoge = Toplevel()

            window_height = 250
            window_width = 370
            
            top_dialoge.title("Update Found")
            
            top_dialoge.resizable(False, False)  # This code helps to disable windows from resizing
            
            top_dialoge.lift()
            
            top_dialoge.attributes("-topmost", True)
            
            settings_menu.attributes("-topmost", False)
            
            screen_width = top_dialoge.winfo_screenwidth()
            screen_height = top_dialoge.winfo_screenheight()

            x_cordinate = int((screen_width/2) - (window_width/2))
            y_cordinate = int((screen_height/2) - (window_height/2))

            top_dialoge.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

            # change title bar icon
            top_dialoge.iconbitmap('img\\UVR-Icon-v2.ico')
            
            tabControl = ttk.Notebook(top_dialoge)
            
            tabControl.pack(expand = 1, fill ="both")
            
            tabControl.grid_rowconfigure(0, weight=1)
            tabControl.grid_columnconfigure(0, weight=1)
            
            def no():
                settings_menu.attributes("-topmost", True)
                top_dialoge.destroy()
                
            def yes():
                download_update()
                settings_menu.attributes("-topmost", True)
                top_dialoge.destroy()
            
            frame0=Frame(tabControl,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=0)  
            
            l0=tk.Label(frame0, text='Update Found', font=("Century Gothic", "13", "underline"), foreground='#13a4c9')
            l0.grid(row=0,column=0,padx=0,pady=10)
            
            l0=tk.Label(frame0, text='Are you sure you want to continue?\n\nThe application will need to be restarted.\n', font=("Century Gothic", "11"), foreground='#13a4c9')
            l0.grid(row=1,column=0,padx=0,pady=5)
                    
            l0=ttk.Button(frame0, text='Yes', command=yes)

            l0.grid(row=2,column=0,padx=0,pady=5)

            l0=ttk.Button(frame0, text='No', command=no)

            l0.grid(row=3,column=0,padx=0,pady=5)

        def change_event():
            self.delete_temps()
            try:
                stop_thread()
            except:
                pass
            try:
                top_code.destroy()
            except:
                pass
            settings_menu.destroy()
        
        def close_win_custom_ensemble():
            change_event()
            self.custom_ensemble()
            
        def close_win_advanced_mdx_options():
            change_event()
            self.advanced_mdx_options()
            
        def close_win_advanced_demucs_options():
            change_event()
            self.advanced_demucs_options()
            
        def close_win_advanced_vr_options():
            change_event()
            self.advanced_vr_options()
            
        def close_win_error_log():
            change_event()
            self.error_log()
            
        def close_win_help():
            change_event()
            self.help()
            
        def close_win():
            change_event()

        def restart():
            settings_menu.destroy()
            self.restart()

        tabControl = ttk.Notebook(settings_menu)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Settings Guide')
        tabControl.add(tab2, text ='Audio Format Settings')
        tabControl.add(tab3, text ='Download Center')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)
        
        tab2.grid_rowconfigure(0, weight=1)
        tab2.grid_columnconfigure(0, weight=1)
        
        tab3.grid_rowconfigure(0, weight=1)
        tab3.grid_columnconfigure(0, weight=1)

        if choose:
            tabControl.select(tab3)

        frame0=Frame(tab1,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=tk.Label(frame0,text="Main Menu",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text="Ensemble Customization Options", command=close_win_custom_ensemble)
        l0.grid(row=1,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced MDX-Net Options", command=close_win_advanced_mdx_options)
        l0.grid(row=2,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced Demucs Options", command=close_win_advanced_demucs_options)
        l0.grid(row=3,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced VR Options", command=close_win_advanced_vr_options)
        l0.grid(row=4,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text="Open Help Guide", command=close_win_help)
        l0.grid(row=5,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Open Error Log', command=close_win_error_log)
        l0.grid(row=6,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0,text=f"Additional Options",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        l0=ttk.Checkbutton(frame0, text='Settings Test Mode', variable=self.settest_var) 
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Reset All Settings to Default', command=self.reset_to_defaults)
        l0.grid(row=9,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Open Application Directory', command=self.open_appdir_filedialog)
        l0.grid(row=10,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Restart Application', command=restart)
        l0.grid(row=11,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win)
        l0.grid(row=12,column=0,padx=0,pady=5)
        
        def start_target_update():
            def target_update():
                update_var.set(' Loading version information... ')
                update_signal_url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/update_patches.txt"
                url = update_signal_url
                try:
                    file = urllib.request.urlopen(url)
                    for line in file:
                        patch_name = line.decode("utf-8")
                        if patch_name == current_version:
                            update_var.set(' UVR Version Current ')
                            update_button_var.set('Check for Updates')
                        else:
                            label_set_a = f" Update Found: {patch_name} "
                            update_var.set(str(label_set_a))
                            update_button_var.set('Click Here to Update')
                except:
                        update_var.set(' Version Status: No Internet Connection ') 
            rlg = KThread(target=target_update)
            rlg.start()

        start_target_update()

        l0=tk.Label(frame0,text="Application Updates",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=13,column=0,padx=0,pady=10)
        
        def start_check_updates():
            def check_updates():
                try:
                    url = "https://raw.githubusercontent.com/TRvlvr/application_data/main/update_patches.txt"
                    file = urllib.request.urlopen(url)
                    for line in file:
                        patch_name = line.decode("utf-8")
                        if patch_name == current_version:
                            update_var.set(' UVR Version Current ')
                            update_button_var.set('Check for Updates')
                        else:
                            label_set = f"Update Found: {patch_name}"
                            update_button_var.set('Click Here to Update')
                            update_var.set(str(label_set))
                            update_set_var.set(str(label_set))
                            askyesorno()
                except:
                        update_var.set(' Version Status: No Internet Connection ') 

            rlg = KThread(target=check_updates)
            rlg.start()
        
        def open_bmac_m():
            settings_menu.attributes("-topmost", False)
            callback("https://www.buymeacoffee.com/uvr5")  
        
        l0=ttk.Button(frame0,text=update_button_var.get(), command=start_check_updates)
        l0.grid(row=14,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0,textvariable=update_var,font=("Century Gothic", "12"), justify="center", relief="ridge", fg="#13a4c9")
        l0.grid(row=15,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0, image=self.donate_img, command=open_bmac_m)
        l0.grid(row=16,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0,text=f"{space_small}{space_small}{space_small}{space_small}",font=("Century Gothic", "13"), justify="center", relief="flat", fg="#13a4c9")
        l0.grid(row=17,column=0,padx=0,pady=0)
        
        frame0=Frame(tab2,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=tk.Label(frame0,text="Audio Format Settings",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=tk.Label(frame0, text='Wav Type', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        l0=ttk.OptionMenu(frame0, self.wavtype_var, None, 'PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', '32-bit Float', '64-bit Float')
        l0.grid(row=2,column=0,padx=20,pady=0)
        
        l0=tk.Label(frame0, text='Mp3 Bitrate', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.OptionMenu(frame0, self.mp3bit_var, None, '96k', '128k', '160k', '224k', '256k', '320k')
        l0.grid(row=6,column=0,padx=20,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Normalize Outputs\n(Prevents clipping)', variable=self.normalize_var) 
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        frame0=Frame(tab3,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=tk.Label(frame0,text="Application Download Center",font=("Century Gothic", "13", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=20,pady=10)
        
        def user_code():
            """
            Input Code
            """
            
            okVar = tk.IntVar()
            
            try:
                with open(user_code_file, "r") as f:
                    code_read = f.read()
                user_code_var = tk.StringVar(value=code_read)
            except:
                user_code_var = tk.StringVar(value='')
                
            try:
                with open(download_code_file, "r") as f:
                    code_download_read = f.read()
                user_code_download_var = tk.StringVar(value=code_download_read)
            except:
                user_code_download_var = tk.StringVar(value='')
            
            global top_code
            
            top_code = Toplevel(settings_menu)

            window_height = 480
            window_width = 320
            
            top_code.title("User Download Codes")
            
            top_code.resizable(False, False)  # This code helps to disable windows from resizing
            
            # top_code.attributes("-topmost", True)
            
            # settings_menu.attributes("-topmost", False)
            
            screen_width = top_code.winfo_screenwidth()
            screen_height = top_code.winfo_screenheight()

            x_cordinate = int((screen_width/2) - (window_width/2))
            y_cordinate = int((screen_height/2) - (window_height/2))

            top_code.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

            x = settings_menu.winfo_x()
            y = settings_menu.winfo_y()
            top_code.geometry("+%d+%d" %(x+90,y+135))
            top_code.wm_transient(settings_menu)

            # change title bar icon
            top_code.iconbitmap('img\\UVR-Icon-v2.ico')
            
            tabControl = ttk.Notebook(top_code)
            
            tabControl.pack(expand = 1, fill ="both")
            
            tabControl.grid_rowconfigure(0, weight=1)
            tabControl.grid_columnconfigure(0, weight=1)
            
            frame0=Frame(tabControl,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=0)  
            
            def write_code():
                with open(user_code_file, 'w') as f:
                    user_type_code = user_code_var.get()
                    f.write(str(user_type_code))
                with open(download_code_file, 'w') as f:
                    user_download_code = user_code_download_var.get()
                    f.write(str(user_download_code))
                    top_code.destroy()
                    if user_type_code == 'VIP':
                        refresh_list_vip()
                    elif user_type_code == 'Developer':
                        refresh_list_dev()
                    else:
                        refresh_list()
            
            def open_patreon():
                top_code.attributes("-topmost", False)
                callback("https://www.patreon.com/uvr") 
                
            def open_bmac():
                top_code.attributes("-topmost", False)
                callback("https://www.buymeacoffee.com/uvr5")  
                         
            def quit():
                settings_menu.attributes("-topmost", True)
                top_code.destroy()
            
            l0=tk.Label(frame0, text=f'User Download Codes', font=("Century Gothic", "11", "underline"), foreground='#13a4c9')
            l0.grid(row=0,column=0,padx=0,pady=5)    
            
            l0=tk.Label(frame0, text=f'{space_medium}User Code{space_medium}', font=("Century Gothic", "9"), foreground='#13a4c9')
            l0.grid(row=1,column=0,padx=0,pady=5)       
                    
            l0=ttk.Entry(frame0, textvariable=user_code_var, justify='center')

            l0.grid(row=2,column=0,padx=0,pady=5)
            
            l0=tk.Label(frame0, text=f'Download Code', font=("Century Gothic", "9"), foreground='#13a4c9')
            l0.grid(row=3,column=0,padx=0,pady=5)       
                    
            l0=ttk.Entry(frame0, textvariable=user_code_download_var, justify='center')

            l0.grid(row=4,column=0,padx=0,pady=5)

            l0=ttk.Button(frame0, text='Confirm', command=write_code)

            l0.grid(row=5,column=0,padx=0,pady=5)
            
            l0=ttk.Button(frame0, text='Cancel', command=quit)

            l0.grid(row=6,column=0,padx=0,pady=5)
            
            l0=tk.Label(frame0, text=f'Support UVR', font=("Century Gothic", "11", "underline"), foreground='#13a4c9')
            l0.grid(row=7,column=0,padx=0,pady=5)    
            
            l0=tk.Label(frame0, text=f'Obtain codes by making a one-time donation\n via \"Buy Me a Coffee\" or by becoming a Patreon.\nClick one of the buttons below to donate or pledge!', font=("Century Gothic", "8"), foreground='#13a4c9')
            l0.grid(row=8,column=0,padx=0,pady=5)
            
            l0=ttk.Button(frame0, text='UVR Patreon Link', command=open_patreon)

            l0.grid(row=9,column=0,padx=0,pady=5)
            
            l0=ttk.Button(frame0, text='UVR \"Buy Me a Coffee\" Link', command=open_bmac)

            l0.grid(row=10,column=0,padx=0,pady=5)
            
        def download_code():
            """
            Input Download Code
            """
                
            try:
                with open(download_code_file, "r") as f:
                    code_download_read = f.read()
                user_code_download_var = tk.StringVar(value=code_download_read)
            except:
                user_code_download_var = tk.StringVar(value='')
            
            top_code= Toplevel()

            window_height = 300
            window_width = 410
            
            top_code.title("Invalid Download Code")
            
            top_code.resizable(False, False)  # This code helps to disable windows from resizing
            
            screen_width = top_code.winfo_screenwidth()
            screen_height = top_code.winfo_screenheight()

            x_cordinate = int((screen_width/2) - (window_width/2))
            y_cordinate = int((screen_height/2) - (window_height/2))

            top_code.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

            x = settings_menu.winfo_x()
            y = settings_menu.winfo_y()
            top_code.geometry("+%d+%d" %(x+43,y+220))
            top_code.wm_transient(settings_menu)

            # change title bar icon
            top_code.iconbitmap('img\\UVR-Icon-v2.ico')
            
            tabControl = ttk.Notebook(top_code)
            
            tabControl.pack(expand = 1, fill ="both")
            
            tabControl.grid_rowconfigure(0, weight=1)
            tabControl.grid_columnconfigure(0, weight=1)
            
            frame0=Frame(tabControl,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=0)  
            
            def write_code_p():
                with open(download_code_file, 'w') as f:
                    user_download_code = user_code_download_var.get()
                    f.write(str(user_download_code))
                    download_model()
                    top_code.destroy()
                    
            def quit():
                settings_menu.attributes("-topmost", True)
                top_code.destroy()
            
            l0=tk.Label(frame0, text=f'Invalid Download Code', font=("Century Gothic", "11", "underline"), foreground='#13a4c9')
            l0.grid(row=0,column=0,padx=0,pady=10)    
            
            l0=tk.Label(frame0, text=f'Provide the correct code below or make another selection.\n', font=("Century Gothic", "10"), foreground='#13a4c9')
            l0.grid(row=1,column=0,padx=0,pady=0)    
            
            l0=tk.Label(frame0, text=f'Download Code', font=("Century Gothic", "9"), foreground='#13a4c9')
            l0.grid(row=2,column=0,padx=0,pady=5)       
                    
            l0=ttk.Entry(frame0, textvariable=user_code_download_var, justify='center')

            l0.grid(row=3,column=0,padx=0,pady=5)

            l0=ttk.Button(frame0, text='Confirm', command=write_code_p)

            l0.grid(row=4,column=0,padx=0,pady=5)
            
            l0=ttk.Button(frame0, text='Cancel', command=quit)

            l0.grid(row=5,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, text=f"{space_fill_wide}Select Download{space_fill_wide}", font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        def download_progress_bar(current, total, width=80):

            progress = ('%s' % (100 * current // total))

            self.download_progress_bar_zip_var.set(int(progress))

            self.download_progress_var.set(progress + ' %')

        def change_state_complete():
            download_button.configure(state=tk.NORMAL)
            self.downloadmodelOptions.configure(state=tk.NORMAL)
            self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
            stop_button.configure(state=tk.DISABLED)
            self.download_stop_var.set(space_small)
            self.download_progress_bar_var.set('Download Complete') 
            self.delete_temps()
            
        def change_state_locked():
            download_button.configure(state=tk.NORMAL)
            self.downloadmodelOptions.configure(state=tk.NORMAL)
            self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
            stop_button.configure(state=tk.DISABLED)
            self.download_stop_var.set(space_small)
            self.download_progress_bar_var.set('Invalid Download Code')
            self.delete_temps()

            
        def change_state_already_found():
            download_button.configure(state=tk.NORMAL)
            self.downloadmodelOptions.configure(state=tk.NORMAL)
            self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
            stop_button.configure(state=tk.DISABLED)
            self.download_stop_var.set(space_small)
            self.download_progress_bar_var.set('Download Stopped') 
            self.delete_temps()
            
        def change_state_failed():
            download_button.configure(state=tk.NORMAL)
            self.downloadmodelOptions.configure(state=tk.NORMAL)
            self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
            stop_button.configure(state=tk.DISABLED)
            self.download_stop_var.set(space_small)
            self.download_progress_bar_var.set('Download Failed')
            self.delete_temps()

        def download_model():
            self.delete_temps()
            
            def begin_download_model():
                
                if not self.modeldownload_var.get() == 'No Model Selected':
                    model = self.modeldownload_var.get()
                    self.download_progress_bar_var.set('Downloading...')
                elif not self.modeldownload_mdx_var.get() == 'No Model Selected':
                    model = self.modeldownload_mdx_var.get()
                    self.download_progress_bar_var.set('Downloading...')
                elif not self.modeldownload_demucs_var.get() == 'No Model Selected':
                    model = self.modeldownload_demucs_var.get()
                    self.download_progress_bar_var.set('Downloading...')
                elif not update_set_var.get() == 'UVR Version Current':
                    model = update_set_var.get()
                    self.download_progress_bar_var.set('Downloading Update...')
                else:
                    self.download_progress_bar_var.set('No Model Selected')
                    
                    return
                   
                self.downloadmodelOptions.configure(state=tk.DISABLED)
                self.downloadmodelOptions_mdx.configure(state=tk.DISABLED)
                self.downloadmodelOptions_demucs.configure(state=tk.DISABLED)
                download_button.configure(state=tk.DISABLED)
                stop_button.configure(state=tk.NORMAL)
                self.download_stop_var.set('Stop Download')
                
                if model == 'Demucs v3: mdx':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v3: mdx')
                        
                        url_1 = links[0]
                        url_2 = links[1]
                        url_3 = links[2]
                        url_4 = links[3]
                        url_5 = links[4]
                        self.download_progress_bar_var.set('Downloading Model 1/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/0d19c1c6-0f06f20e.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_1, 'models/Demucs_Models/v3_repo/0d19c1c6-0f06f20e.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 2/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/7ecf8ec1-70f50cc9.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_2, 'models/Demucs_Models/v3_repo/7ecf8ec1-70f50cc9.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 3/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/c511e2ab-fe698775.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/v3_repo/c511e2ab-fe698775.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 4/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/7d865c68-3d5dd56b.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/v3_repo/7d865c68-3d5dd56b.th', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/v3_repo/mdx.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/v3_repo/mdx.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/mdx.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/mdx.yaml', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                if model == 'Demucs v3: mdx_q':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v3: mdx_q')
                        
                        url_1 = links[0]
                        url_2 = links[1]
                        url_3 = links[2]
                        url_4 = links[3]
                        url_5 = links[4]
                        self.download_progress_bar_var.set('Downloading Model 1/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/6b9c2ca1-3fd82607.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_1, 'models/Demucs_Models/v3_repo/6b9c2ca1-3fd82607.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 2/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/b72baf4e-8778635e.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_2, 'models/Demucs_Models/v3_repo/b72baf4e-8778635e.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 3/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/42e558d4-196e0e1b.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/v3_repo/42e558d4-196e0e1b.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 4/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/305bc58f-18378783.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/v3_repo/305bc58f-18378783.th', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/v3_repo/mdx_q.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/v3_repo/mdx_q.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/mdx_q.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/mdx_q.yaml', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                            
                            
                if model == 'Demucs v3: mdx_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v3: mdx_extra')
                        
                        url_1 = links[0]
                        url_2 = links[1]
                        url_3 = links[2]
                        url_4 = links[3]
                        url_5 = links[4]
                        self.download_progress_bar_var.set('Downloading Model 1/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/e51eebcc-c1b80bdd.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_1, 'models/Demucs_Models/v3_repo/e51eebcc-c1b80bdd.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 2/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/a1d90b5c-ae9d2452.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_2, 'models/Demucs_Models/v3_repo/a1d90b5c-ae9d2452.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 3/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/5d2d6c55-db83574e.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/v3_repo/5d2d6c55-db83574e.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 4/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/cfa93e08-61801ae1.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/v3_repo/cfa93e08-61801ae1.th', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/v3_repo/mdx_extra.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/v3_repo/mdx_extra.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/mdx_extra.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/mdx_extra.yaml', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v3: mdx_extra_q':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v3: mdx_extra_q')
                        
                        url_1 = links[0]
                        url_2 = links[1]
                        url_3 = links[2]
                        url_4 = links[3]
                        url_5 = links[4]
                        self.download_progress_bar_var.set('Downloading Model 1/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/83fc094f-4a16d450.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_1, 'models/Demucs_Models/v3_repo/83fc094f-4a16d450.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 2/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/464b36d7-e5a9386e.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_2, 'models/Demucs_Models/v3_repo/464b36d7-e5a9386e.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 3/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/14fc6a69-a89dd0ee.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/v3_repo/14fc6a69-a89dd0ee.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 4/4...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/7fd6ef75-a905dd85.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/v3_repo/7fd6ef75-a905dd85.th', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/v3_repo/mdx_extra_q.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/v3_repo/mdx_extra_q.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/mdx_extra_q.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/mdx_extra_q.yaml', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                
                if model == 'Demucs v3: UVR Models':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v3: UVR Models')
                        
                        url_1 = links[0]
                        url_2 = links[1]
                        url_3 = links[2]
                        url_4 = links[3]
                        url_5 = links[4]
                        self.download_progress_bar_var.set('Downloading Model 1/2...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/ebf34a2d.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_1, 'models/Demucs_Models/v3_repo/ebf34a2d.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model 2/2...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/ebf34a2db.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_2, 'models/Demucs_Models/v3_repo/ebf34a2db.th', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/UVR_Demucs_Model_1.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/v3_repo/UVR_Demucs_Model_1.yaml', bar=download_progress_bar)
                        self.download_progress_bar_var.set('Downloading Model...')
                        if os.path.isfile('models/Demucs_Models/v3_repo/UVR_Demucs_Model_2.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/v3_repo/UVR_Demucs_Model_2.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/v3_repo/UVR_Demucs_Model_Bag.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/v3_repo/UVR_Demucs_Model_Bag.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/UVR_Demucs_Model_1.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_3, 'models/Demucs_Models/UVR_Demucs_Model_1.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/UVR_Demucs_Model_2.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_4, 'models/Demucs_Models/UVR_Demucs_Model_2.yaml', bar=download_progress_bar)
                        if os.path.isfile('models/Demucs_Models/UVR_Demucs_Model_Bag.yaml'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url_5, 'models/Demucs_Models/UVR_Demucs_Model_Bag.yaml', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v2: demucs':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: demucs')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs-e07c671f.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs-e07c671f.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                if model == 'Demucs v2: demucs_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: demucs_extra')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs_extra-3646af93.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs_extra-3646af93.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v2: demucs48_hq':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: demucs48_hq')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs48_hq-28a1282c.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs48_hq-28a1282c.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v2: tasnet':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: tasnet')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/tasnet-beb46fac.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/tasnet-beb46fac.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v2: tasnet_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: tasnet_extra')
                        
                        url = links    
                                            
                        if os.path.isfile('models/Demucs_Models/tasnet_extra-df3777b2.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/tasnet_extra-df3777b2.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v2: demucs_unittest':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v2: demucs_unittest')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs_unittest-09ebc15f.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs_unittest-09ebc15f.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: demucs':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: demucs')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: demucs_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: demucs_extra')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/demucs_extra.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/demucs_extra.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: light':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: light')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/light.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/light.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: light_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: light_extra')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/light_extra.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/light_extra.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: tasnet':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: tasnet')
                        
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/tasnet.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/tasnet.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'Demucs v1: tasnet_extra':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='Demucs v1: tasnet_extra')
                        url = links                        
                        if os.path.isfile('models/Demucs_Models/tasnet_extra.th'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Demucs_Models/tasnet_extra.th', bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
         
                if model == 'VR Arch Model Pack v5: HP2 Models':
                    try:
                        links = []
                        links = lib_v5.filelist.get_download_links(links, downloads='model_repo')
                        url = f"{links}uvr_v5_hp2_models.zip"
                        if os.path.isfile('models/Main_Models/uvr_v5_hp2_models.zip'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Main_Models/uvr_v5_hp2_models.zip', bar=download_progress_bar)
                        with zipfile.ZipFile('models/Main_Models/uvr_v5_hp2_models.zip', 'r') as zip_ref:
                            zip_ref.extractall('models/Main_Models')
                        os.remove('models/Main_Models/uvr_v5_hp2_models.zip')
                        change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if model == 'VR Arch Model Pack v4: Main Models':
                    links = []
                    links = lib_v5.filelist.get_download_links(links, downloads='model_repo')
                    url = f"{links}uvr_v4_models.zip"
                    try:
                        if os.path.isfile('models/Main_Models/4_models.zip'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Main_Models/4_models.zip', bar=download_progress_bar)
                        with zipfile.ZipFile('models/Main_Models/4_models.zip', 'r') as zip_ref:
                            zip_ref.extractall('models/Main_Models')
                        os.remove('models/Main_Models/4_models.zip')
                        change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                if model == 'VR Arch Model Pack v5: SP Models':
                    links = []
                    links = lib_v5.filelist.get_download_links(links, downloads='model_repo')
                    url = f"{links}uvr_v5_sp_models.zip"
                    try:
                        if os.path.isfile('models/Main_Models/uvr_v5_sp_models.zip'):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, 'models/Main_Models/uvr_v5_sp_models.zip', bar=download_progress_bar)
                        with zipfile.ZipFile('models/Main_Models/uvr_v5_sp_models.zip', 'r') as zip_ref:
                            zip_ref.extractall('models/Main_Models')
                        os.remove('models/Main_Models/uvr_v5_sp_models.zip')
                        change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                if 'VR Arch Single Model v5:' in model or 'VR Arch Single Model v4:' in model:
                    
                    if 'VR Arch Single Model v5:' in model:
                        model_name = model
                        head, sep, tail = model_name.partition('VR Arch Single Model v5: ')
                        model_name = tail
                    if 'VR Arch Single Model v4:' in model:
                        model_name = model
                        head, sep, tail = model_name.partition('VR Arch Single Model v4: ')
                        model_name = tail
                    if 'VR Arch Single Model v4:' in model:
                        model_name = model
                        head, sep, tail = model_name.partition('VR Arch Single Model v4: ')
                        model_name = tail
                        
                    links = []
                    links = lib_v5.filelist.get_download_links(links, downloads='single_model_repo')
                    m_url = f"{links}{model_name}.pth"
                    url = m_url
                    
                    try:
                        if os.path.isfile(f"models/Main_Models/{model_name}.pth"):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, f"models/Main_Models/{model_name}.pth", bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                
                if 'MDX-Net Model: ' in model:
                    
                    model_name = model
                    head, sep, tail = model_name.partition('MDX-Net Model: ')
                    model_name = tail
                    links = []
                    links = lib_v5.filelist.get_download_links(links, downloads='single_model_repo')
                    m_url = f"{links}{model_name}.onnx"
                    
                    #print(m_url)
                    url = m_url
                    
                    try:
                        if os.path.isfile(f"models/MDX_Net_Models/{model_name}.onnx"):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(url, f"models/MDX_Net_Models/{model_name}.onnx", bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                            
                if 'MDX-Net Model VIP:' in model:
                    
                    model_name = model
                    head, sep, tail = model_name.partition('MDX-Net Model VIP: ')
                    model_name = tail
    
                    url_code = f"https://github.com/TRvlvr/application_data/raw/main/filelists/aes_vip/{model_name}.txt.aes"
                    
                    encrypted_file_code_vip = f"lib_v5/filelists/download_codes/temp/{model_name}.aes"
                    file_code_vip = f"lib_v5/filelists/download_codes/temp/{model_name}.txt"
                    
                    try:
                        wget.download(url_code, encrypted_file_code_vip, bar=download_progress_bar)
                        with open(download_code_file, "r") as f:
                            user_download_code_read = f.read()
                        
                        bufferSize = 128 * 1024
                        password = user_download_code_read
                        
                        try:
                            pyAesCrypt.decryptFile(encrypted_file_code_vip, file_code_vip, password, bufferSize)
                        except:
                            try:
                                url_v_key = f"https://github.com/TRvlvr/application_data/raw/main/filelists/aes_dev/vip_key.txt.aes"
                                wget.download(url_v_key, 'lib_v5/filelists/download_codes/temp/vip_key.aes', bar=download_progress_bar)
                                pyAesCrypt.decryptFile('lib_v5/filelists/download_codes/temp/vip_key.aes', 
                                                    'lib_v5/filelists/download_codes/temp/vip_key.txt', password, bufferSize)
                                
                                with open('lib_v5/filelists/download_codes/temp/vip_key.txt', "r") as f:
                                    vip_code_read = f.read()
                                
                                password = vip_code_read

                                pyAesCrypt.decryptFile(encrypted_file_code_vip, file_code_vip, password, bufferSize)
                            except:
                                download_code()
                                change_state_locked()
                                return
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                    
                    with open(file_code_vip, "r") as f:
                        link=f.read()  
                    
                    m_url = link
                    
                    try:
                        if os.path.isfile(f"models/MDX_Net_Models/{model_name}.onnx"):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(m_url, f"models/MDX_Net_Models/{model_name}.onnx", bar=download_progress_bar)
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                            
                            
                if 'Developer Pack:' in model:
                        
                    pack_name = model
                    head, sep, tail = pack_name.partition('Developer Pack: ')
                    pack_name = tail
                    
                    url_code = f"https://github.com/TRvlvr/application_data/raw/main/filelists/aes_dev/{pack_name}.txt.aes"
                    
                    #print(url_code)
                    
                    encrypted_file_code = f"lib_v5/filelists/download_codes/temp/{pack_name}.aes"
                    file_code = f"lib_v5/filelists/download_codes/temp/{pack_name}.txt"
                    
                    try:
                        wget.download(url_code, encrypted_file_code, bar=download_progress_bar)
                        with open(download_code_file, "r") as f:
                            user_download_code_read = f.read()
                        
                        bufferSize = 128 * 1024
                        password = user_download_code_read
                        # encrypt
                        
                        try:
                            pyAesCrypt.decryptFile(encrypted_file_code, file_code, password, bufferSize)
                        except:
                            download_code()
                            change_state_locked()
                            return
                        
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error) 
                    
                    with open(file_code, "r") as f:
                        link=f.read()
                        
                    m_url = link
                        
                    try:
                        if os.path.isfile(f"models/MDX_Net_Models/{pack_name}.zip"):
                                self.download_progress_var.set('File already exists')
                                change_state_already_found()
                                pass
                        else:
                                wget.download(m_url, f"models/MDX_Net_Models/{pack_name}.zip", bar=download_progress_bar)
                                with zipfile.ZipFile(f'models/MDX_Net_Models/{pack_name}.zip', 'r') as zip_ref:
                                    zip_ref.extractall('models/MDX_Net_Models')
                                try:
                                    os.remove(f'models/MDX_Net_Models/{pack_name}.zip')
                                except:
                                    pass
                                change_state_complete()
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error)            
                            
                if 'Update Found:' in model:
                    pack_name = model
                    head, sep, tail = pack_name.partition('Update Found: ')
                    pack_name = tail
                    cwd_path = os.path.dirname(os.path.realpath(__file__))
                    #print('cwd_path ', cwd_path)
                    
                    links = []
                    links = lib_v5.filelist.get_download_links(links, downloads='app_patch')
                    url_link = f"{links}{pack_name}.exe"
                    #print(url_link)
                    settings_menu.attributes("-topmost", False)
                    try:
                        if os.path.isfile(f"{cwd_path}/{pack_name}.exe"):
                                self.download_progress_var.set('File already exists')
                                subprocess.Popen(f"{cwd_path}/{pack_name}.exe")
                        else:
                            wget.download(url_link, f"{cwd_path}/{pack_name}.exe", bar=download_progress_bar)
                            subprocess.Popen(f"{cwd_path}/{pack_name}.exe")
                    except Exception as e:
                        short_error = f'{e}'
                        change_state_failed()
                        if '[Errno 11001] getaddrinfo failed' in short_error:
                            self.download_progress_var.set('No Internet Connection Detected')
                        else: 
                            self.download_progress_var.set(short_error)

                self.update_states()           
        
            global th
            
            th = KThread(target=begin_download_model)
            th.start()
    
        def stop_thread():
            th.kill()
            download_button.configure(state=tk.NORMAL)
            self.downloadmodelOptions.configure(state=tk.NORMAL)
            self.downloadmodelOptions_mdx.configure(state=tk.NORMAL)
            self.downloadmodelOptions_demucs.configure(state=tk.NORMAL)
            stop_button.configure(state=tk.DISABLED)
            self.download_stop_var.set(space_small)
            self.update_states()
            self.download_progress_bar_var.set('Download Stopped')
            self.delete_temps()
            
            
        def download_update():
            self.modeldownload_var.set('No Model Selected')
            self.modeldownload_mdx_var.set('No Model Selected')
            self.modeldownload_demucs_var.set('No Model Selected')
            tabControl.select(tab3)
            download_model()
            
        vr_download_list_file = "lib_v5/filelists/download_lists/vr_download_list.txt"
        mdx_download_list_file = "lib_v5/filelists/download_lists/mdx_download_list.txt"
        demucs_download_list_file = "lib_v5/filelists/download_lists/demucs_download_list.txt"
        
        vr_download_list_temp_file = "lib_v5/filelists/download_lists/temp/vr_download_list.txt"
        mdx_download_list_temp_file = "lib_v5/filelists/download_lists/temp/mdx_download_list.txt"
        demucs_download_list_temp_file = "lib_v5/filelists/download_lists/temp/demucs_download_list.txt"
        
        mdx_new_hashes = "lib_v5/filelists/hashes/mdx_new_hashes.txt"
        mdx_new_inst_hashes = "lib_v5/filelists/hashes/mdx_new_inst_hashes.txt"
        mdx_original_hashes = "lib_v5/filelists/hashes/mdx_original_hashes.txt"
        download_links_file = "lib_v5/filelists/download_lists/download_links.json"
        
        mdx_new_hashes_temp = "lib_v5/filelists/hashes/temp/mdx_new_hashes.txt"
        mdx_new_inst_hashes_temp = "lib_v5/filelists/hashes/temp/mdx_new_inst_hashes.txt"
        mdx_original_hashes_temp = "lib_v5/filelists/hashes/temp/mdx_original_hashes.txt"
        download_links_file_temp = "lib_v5/filelists/download_lists/temp/download_links.json"
            
        def move_lists_from_temp():
            try:
                shutil.move(vr_download_list_temp_file, vr_download_list_file)
                shutil.move(mdx_download_list_temp_file, mdx_download_list_file)
                shutil.move(demucs_download_list_temp_file, demucs_download_list_file)
                shutil.move(mdx_new_hashes_temp, mdx_new_hashes)
                shutil.move(mdx_new_inst_hashes_temp, mdx_new_inst_hashes)
                shutil.move(mdx_original_hashes_temp, mdx_original_hashes)
                shutil.move(download_links_file_temp, download_links_file)
            except:
                pass
            
        def remove_lists_temp():
            try:
                os.remove(vr_download_list_temp_file)
                os.remove(mdx_download_list_temp_file)
                os.remove(demucs_download_list_temp_file)
                os.remove(mdx_new_hashes_temp)
                os.remove(mdx_new_inst_hashes_temp)
                os.remove(mdx_original_hashes_temp)
                os.remove(download_links_file_temp)
            except:
                pass
            
        def refresh_download_list_only():
            download_links = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/download_links.json"
            def begin_refresh_list():
                try:
                    url_1 = download_links
                    shutil.move(download_links_file, download_links_file_temp)
                    wget.download(url_1, download_links_file, bar=download_progress_bar)
                    os.remove(download_links_file_temp)
                except Exception as e:
                    try:
                        shutil.move(download_links_file_temp, download_links_file)
                    except:
                        pass
            
            rlg = KThread(target=begin_refresh_list)
            rlg.start()
            
        def refresh_list():
            download_links = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/download_links.json"
            def begin_refresh_list():
                try:
                    url_1 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/gen_vr_download_list.txt'
                    url_2 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/gen_mdx_download_list.txt'
                    url_3 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/gen_demucs_download_list.txt'
                    url_4 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_hashes.txt'
                    url_5 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_inst_hashes.txt'
                    url_6 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_original_hashes.txt'
                    url_7 = download_links
                    wget.download(url_1, vr_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_2, mdx_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_3, demucs_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_4, mdx_new_hashes_temp, bar=download_progress_bar)
                    wget.download(url_5, mdx_new_inst_hashes_temp, bar=download_progress_bar)
                    wget.download(url_6, mdx_original_hashes_temp, bar=download_progress_bar)
                    wget.download(url_7, download_links_file_temp, bar=download_progress_bar)
                    move_lists_from_temp()
                    self.download_progress_bar_var.set('Download list\'s refreshed!')
                    settings_menu.destroy()
                    self.settings(choose=True)
                except Exception as e:
                    short_error = f'{e}'
                    self.download_progress_bar_var.set('Refresh failed')
                    if '[Errno 11001] getaddrinfo failed' in short_error:
                        self.download_progress_var.set('No Internet Connection Detected')
                    else: 
                        self.download_progress_var.set(short_error)
                    try:
                        remove_lists_temp()
                    except:
                        pass
            
            rlg = KThread(target=begin_refresh_list)
            rlg.start()
                
        def refresh_list_vip():
            download_links = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/download_links.json"
            def begin_refresh_list_vip():
                try:
                    url_1 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/vip_vr_download_list.txt'
                    url_2 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/vip_mdx_download_list.txt'
                    url_3 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/vip_demucs_download_list.txt'
                    url_4 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_hashes.txt'
                    url_5 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_inst_hashes.txt'
                    url_6 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_original_hashes.txt'
                    url_7 = download_links
                    wget.download(url_1, vr_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_2, mdx_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_3, demucs_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_4, mdx_new_hashes_temp, bar=download_progress_bar)
                    wget.download(url_5, mdx_new_inst_hashes_temp, bar=download_progress_bar)
                    wget.download(url_6, mdx_original_hashes_temp, bar=download_progress_bar)
                    wget.download(url_7, download_links_file_temp, bar=download_progress_bar)
                    move_lists_from_temp()
                    self.download_progress_bar_var.set('VIP: Download list\'s refreshed!')
                    settings_menu.destroy()
                    self.settings(choose=True)
                except Exception as e:
                    short_error = f'{e}'
                    self.download_progress_bar_var.set('Refresh failed')
                    if '[Errno 11001] getaddrinfo failed' in short_error:
                        self.download_progress_var.set('No Internet Connection Detected')
                    else: 
                        self.download_progress_var.set(short_error)
                    try:
                        remove_lists_temp()
                    except:
                        pass
            
            rlv = KThread(target=begin_refresh_list_vip)
            rlv.start()

        vr_download_list_dev = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/dev_vr_download_list.txt'
        mdx_download_list_dev = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/dev_mdx_download_list.txt'
        demucs_download_list_dev = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/dev_demucs_download_list.txt'

        def refresh_list_dev():
            download_links = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_lists/download_links.json"
            def begin_refresh_list_dev():
                try:
                    url_1 = vr_download_list_dev
                    url_2 = mdx_download_list_dev
                    url_3 = demucs_download_list_dev
                    url_4 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_hashes.txt'
                    url_5 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_new_inst_hashes.txt'
                    url_6 = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/hashes/mdx_original_hashes.txt'
                    url_7 = download_links
                    wget.download(url_1, vr_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_2, mdx_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_3, demucs_download_list_temp_file, bar=download_progress_bar)
                    wget.download(url_4, mdx_new_hashes_temp, bar=download_progress_bar)
                    wget.download(url_5, mdx_new_inst_hashes_temp, bar=download_progress_bar)
                    wget.download(url_6, mdx_original_hashes_temp, bar=download_progress_bar)
                    wget.download(url_7, download_links_file_temp, bar=download_progress_bar)
                    move_lists_from_temp()
                    self.download_progress_bar_var.set('Developer: Download list\'s refreshed!')
                    settings_menu.destroy()
                    self.settings(choose=True)
                except Exception as e:
                    short_error = f'{e}'
                    self.download_progress_bar_var.set('Refresh failed')
                    if '[Errno 11001] getaddrinfo failed' in short_error:
                        self.download_progress_var.set('No Internet Connection Detected')
                    else: 
                        self.download_progress_var.set(short_error)
                    try:
                        remove_lists_temp()
                    except:
                        pass
            
            rld = KThread(target=begin_refresh_list_dev)
            rld.start()
            
        vr_list = ''
        vr_list = lib_v5.filelist.get_vr_download_list(vr_list)
        vr_download_list = vr_list
        
        mdx_list = ''
        mdx_list = lib_v5.filelist.get_mdx_download_list(mdx_list)
        mdx_download_list = mdx_list
        
        demucs_list = ''
        demucs_list = lib_v5.filelist.get_demucs_download_list(demucs_list)
        demucs_download_list = demucs_list
        
        ach_radio = l0=ttk.Radiobutton(frame0, text='VR Arch', variable=self.selectdownload_var, value='VR Arc')
        l0.grid(row=3,column=0,padx=0,pady=5)
        
        self.downloadmodelOptions = l0=ttk.OptionMenu(frame0, self.modeldownload_var, *vr_download_list)
        l0.grid(row=4,column=0,padx=0,pady=5)
        
        mdx_radio = l0=ttk.Radiobutton(frame0, text='MDX-Net', variable=self.selectdownload_var, value='MDX-Net')
        l0.grid(row=5,column=0,padx=0,pady=5)
        
        self.downloadmodelOptions_mdx = l0=ttk.OptionMenu(frame0, self.modeldownload_mdx_var, *mdx_download_list)
        l0.grid(row=6,column=0,padx=0,pady=5)
        
        demucs_radio = l0=ttk.Radiobutton(frame0, text='Demucs', variable=self.selectdownload_var, value='Demucs')
        l0.grid(row=7,column=0,padx=0,pady=5)
        
        self.downloadmodelOptions_demucs = l0=ttk.OptionMenu(frame0, self.modeldownload_demucs_var, *demucs_download_list)
        l0.grid(row=8,column=0,padx=0,pady=5)
        
        download_button = l0=ttk.Button(frame0, image=self.download_img, command=download_model)
        l0.grid(row=9,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, textvariable=self.download_progress_bar_var, font=("Century Gothic", "9"), foreground='#13a4c9', borderwidth=0)
        l0.grid(row=10,column=0,padx=0,pady=5)
        
        l0=tk.Label(frame0, textvariable=self.download_progress_var, font=("Century Gothic", "9"), wraplength=350, foreground='#13a4c9')
        l0.grid(row=11,column=0,padx=0,pady=5)
        
        l0=ttk.Progressbar(frame0, variable=self.download_progress_bar_zip_var)
        l0.grid(row=12,column=0,padx=0,pady=5)
        
        stop_button = l0=ttk.Button(frame0, textvariable=self.download_stop_var, command=stop_thread)
        l0.grid(row=13,column=0,padx=0,pady=5)
        
        try:
            with open(user_code_file, "r") as f:
                code_read = f.read()
                if code_read == 'VIP':
                    l0=ttk.Button(frame0, text='Refresh List', command=refresh_list_vip)
                elif code_read == 'Developer':
                    l0=ttk.Button(frame0, text='Refresh List', command=refresh_list_dev)
                else:
                    code_read = 'None'
                    l0=ttk.Button(frame0, text='Refresh List', command=refresh_list)
        except:
            code_read = 'None'
            l0=ttk.Button(frame0, text='Refresh List', command=refresh_list)
            
        l0.grid(row=14,column=0,padx=0,pady=5)
        
        l0=ttk.Button(frame0, image=self.key_img, command=user_code)
        l0.grid(row=15,column=0,padx=0,pady=5)
        
        stop_button.configure(state=tk.DISABLED)
        
        if choose:
            pass
        else:
            self.download_progress_bar_var.set('')
            
        self.download_progress_var.set('')
        self.download_stop_var.set(space_small)
        
        settings_menu.protocol("WM_DELETE_WINDOW", change_event)
        
        self.update_states()      

    def error_log(self):
        """
        Open Error Log
        """
        error_log_screen= Toplevel(root)
        
        if GetSystemMetrics(1) >= 900:
            window_height = 810
            window_width = 1080
        elif GetSystemMetrics(1) <= 720:
            window_height = 640
            window_width = 930
        else:
            window_height = 670
            window_width = 930
            
        error_log_screen.title("UVR Help Guide")
        
        error_log_screen.resizable(False, False)  # This code helps to disable windows from resizing
        
        #error_log_screen.attributes("-topmost", True)
        
        screen_width = error_log_screen.winfo_screenwidth()
        screen_height = error_log_screen.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        error_log_screen.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        if GetSystemMetrics(1) >= 900:
            x = root.winfo_x()
            y = root.winfo_y()
            error_log_screen.geometry("+%d+%d" %(x-220,y+5))
            error_log_screen.wm_transient(root)

        # x = root.winfo_x()
        # y = root.winfo_y()
        # error_log_screen.geometry("+%d+%d" %(x+43,y+220))
        # error_log_screen.wm_transient(root)

        # change title bar icon
        error_log_screen.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            error_log_screen.destroy()
            self.settings()
            
        def close_win_self():
            error_log_screen.destroy()

        tabControl = ttk.Notebook(error_log_screen)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Error Log')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=tk.Label(frame0,text="Error Details",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=0,padx=20,pady=10)
        
        l0=tk.Label(frame0,text="This tab will show the raw details of the last error received.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text="(Click the error console below to copy the error)\n",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
        l0.grid(row=3,column=0,padx=0,pady=0)
        
        with open("errorlog.txt", "r") as f:
            l0=Button(frame0,text=f.read(),font=("Century Gothic", "7"), command=self.copy_clip, justify="left", wraplength=1000, fg="#FF0000", bg="black", relief="sunken")
            l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text="Back to Main Menu", command=close_win)
        l0.grid(row=5,column=0,padx=20,pady=10)
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=6,column=0,padx=20,pady=0)

    def copy_clip(self):
            copy_t = open("errorlog.txt", "r").read()
            pyperclip.copy(copy_t)
            
    def copy_vr_list(self):
            copy_t = open("lib_v5/vr_download_list.txt", "r").read()
            pyperclip.copy(copy_t)
            
    def open_Modelfolder_filedialog(self):
        """Let user paste a ".pth" model to use for the vocal Separation"""
        filename = 'models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_Modelfolder_vr(self):
        """Let user paste a ".pth" model to use for the vocal Separation"""
        filename = 'models\Main_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_Modelfolder_de(self):
        """Let user paste a ".pth" model to use for the vocal Separation"""
        filename = 'models\Demucs_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_appdir_filedialog(self):
        
        pathname = '.'
        
        if sys.platform == "win32":
            os.startfile(pathname)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
        
    def save_values(self):
        """
        Save the data of the application
        """
        # Get constants
        instrumental = self.instrumentalModel_var.get()
        if [bool(instrumental)].count(True) == 2: #Checkthis
            window_size = DEFAULT_DATA['window_size']
            agg = DEFAULT_DATA['agg']
            chunks = DEFAULT_DATA['chunks']
            noisereduc_s = DEFAULT_DATA['noisereduc_s']
            mixing = DEFAULT_DATA['mixing']
        else:
            window_size = self.winSize_var.get()
            agg = self.agg_var.get()
            chunks = self.chunks_var.get()
            noisereduc_s = self.noisereduc_s_var.get()
            mixing = self.mixing_var.get()

        # -Save Data-
        save_data(data={
            'agg': agg,
            'aiModel': self.aiModel_var.get(),
            'algo': self.algo_var.get(),
            'appendensem': self.appendensem_var.get(),
            'audfile': self.audfile_var.get(),
            'aud_mdx': self.aud_mdx_var.get(),
            'autocompensate': self.autocompensate_var.get(),
            'channel': self.channel_var.get(),
            'chunks': chunks,
            'chunks_d': self.chunks_d_var.get(),
            'compensate': self.compensate_var.get(),
            'demucs_only': self.demucs_only_var.get(),
            'demucs_stems': self.demucs_stems_var.get(),
            'DemucsModel': self.DemucsModel_var.get(),
            'demucsmodel': self.demucsmodel_var.get(),
            'DemucsModel_MDX': self.DemucsModel_MDX_var.get(),
            'demucsmodel_sel_VR': self.demucsmodel_sel_VR_var.get(),
            'demucsmodelVR': self.demucsmodelVR_var.get(),
            'dim_f': self.dim_f_var.get(),
            'ensChoose': self.ensChoose_var.get(),
            'exportPath': self.exportPath_var.get(),
            'flactype': self.flactype_var.get(),
            'gpu': self.gpuConversion_var.get(),
            'inputPaths': self.inputPaths,
            'inst_only': self.inst_only_var.get(),
            'inst_only_b': self.inst_only_b_var.get(),
            'lastDir': self.lastDir,
            'margin': self.margin_var.get(),
            'margin_d': self.margin_d_var.get(),
            'mdx_ensem': self.mdxensemchoose_var.get(),
            'mdx_ensem_b': self.mdxensemchoose_b_var.get(),
            'mdx_only_ensem_a': self.mdx_only_ensem_a_var.get(),
            'mdx_only_ensem_b': self.mdx_only_ensem_b_var.get(),
            'mdx_only_ensem_c': self.mdx_only_ensem_c_var.get(),
            'mdx_only_ensem_d': self.mdx_only_ensem_d_var.get(),
            'mdx_only_ensem_e': self.mdx_only_ensem_e_var.get(),
            'mdxnetModel': self.mdxnetModel_var.get(),
            'mdxnetModeltype': self.mdxnetModeltype_var.get(),
            'mixing': mixing,
            'modeldownload': 'No Model Selected',
            'modeldownload_mdx': 'No Model Selected',
            'modeldownload_demucs': 'No Model Selected',
            'modeldownload_type': 'VR Arc',
            'modelFolder': self.modelFolder_var.get(),
            'modelInstrumentalLabel': self.instrumentalModel_var.get(),
            'ModelParams': self.ModelParams_var.get(),
            'mp3bit': self.mp3bit_var.get(),
            'n_fft_scale': self.n_fft_scale_var.get(),
            'no_chunk': self.no_chunk_var.get(),
            'no_chunk_d': self.no_chunk_d_var.get(),
            'noise_pro_select': self.noise_pro_select_var.get(),
            'noise_reduc': self.noisereduc_var.get(),
            'noisereduc_s': noisereduc_s,
            'non_red': self.non_red_var.get(),
            'nophaseinst': self.nophaseinst_var.get(),
            'normalize': self.normalize_var.get(),
            'output_image': self.outputImage_var.get(),
            'overlap': self.overlap_var.get(),
            'overlap_b': self.overlap_b_var.get(),
            'postprocess': self.postprocessing_var.get(),
            'save': self.save_var.get(),
            'saveFormat': self.saveFormat_var.get(),
            'selectdownload': self.selectdownload_var.get(),
            'segment': self.segment_var.get(),
            'settest': self.settest_var.get(),
            'shifts': self.shifts_var.get(),
            'shifts_b': self.shifts_b_var.get(),
            'split_mode': self.split_mode_var.get(),
            'tta': self.tta_var.get(),
            'useModel': 'instrumental',
            'voc_only': self.voc_only_var.get(),
            'voc_only_b': self.voc_only_b_var.get(),
            'vr_ensem': self.vrensemchoose_var.get(),
            'vr_ensem_a': self.vrensemchoose_a_var.get(),
            'vr_ensem_b': self.vrensemchoose_b_var.get(),
            'vr_ensem_c': self.vrensemchoose_c_var.get(),
            'vr_ensem_d': self.vrensemchoose_d_var.get(),
            'vr_ensem_e': self.vrensemchoose_e_var.get(),
            'vr_ensem_mdx_a': self.vrensemchoose_mdx_a_var.get(),
            'vr_ensem_mdx_b': self.vrensemchoose_mdx_b_var.get(),
            'vr_ensem_mdx_c': self.vrensemchoose_mdx_c_var.get(),
            'vr_multi_USER_model_param_1': self.vr_multi_USER_model_param_1.get(),
            'vr_multi_USER_model_param_2': self.vr_multi_USER_model_param_2.get(),
            'vr_multi_USER_model_param_3': self.vr_multi_USER_model_param_3.get(),
            'vr_multi_USER_model_param_4': self.vr_multi_USER_model_param_4.get(),
            'vr_basic_USER_model_param_1': self.vr_basic_USER_model_param_1.get(),
            'vr_basic_USER_model_param_2': self.vr_basic_USER_model_param_2.get(),
            'vr_basic_USER_model_param_3': self.vr_basic_USER_model_param_3.get(),
            'vr_basic_USER_model_param_4': self.vr_basic_USER_model_param_4.get(),
            'vr_basic_USER_model_param_5': self.vr_basic_USER_model_param_5.get(),
            'wavtype': self.wavtype_var.get(),
            'window_size': window_size,
        }, 
        )
        
        self.destroy()

if __name__ == "__main__":

    root = MainWindow()

    root.tk.call(
    'wm', 
    'iconphoto', 
    root._w, 
    tk.PhotoImage(file='img\\GUI-icon.png')
    )

    lib_v5.sv_ttk.set_theme("dark")
    lib_v5.sv_ttk.use_dark_theme()  # Set dark theme

    #Define a callback function
    def callback(url):
        webbrowser.open_new_tab(url)
        
    root.mainloop()
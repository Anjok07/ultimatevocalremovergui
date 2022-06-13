# GUI modules
import os
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
# Other Modules

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


from win32api import GetSystemMetrics

try:
    with open(os.path.join(os.getcwd(), 'tmp', 'splash.txt'), 'w') as f:
        f.write('1')
except:
    pass

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

#Images
instrumentalModels_dir = os.path.join(base_path, 'models')
banner_path = os.path.join(base_path, 'img', 'UVR-banner.png')
efile_path = os.path.join(base_path, 'img', 'file.png')
stop_path = os.path.join(base_path, 'img', 'stop.png')
help_path = os.path.join(base_path, 'img', 'help.png')
gen_opt_path = os.path.join(base_path, 'img', 'gen_opt.png')
mdx_opt_path = os.path.join(base_path, 'img', 'mdx_opt.png')
vr_opt_path = os.path.join(base_path, 'img', 'vr_opt.png')
demucs_opt_path = os.path.join(base_path, 'img', 'demucs_opt.png')
ense_opt_path = os.path.join(base_path, 'img', 'ense_opt.png')
user_ens_opt_path = os.path.join(base_path, 'img', 'user_ens_opt.png')
credits_path = os.path.join(base_path, 'img', 'credits.png')

DEFAULT_DATA = {
    'exportPath': '',
    'inputPaths': [],
    'saveFormat': 'Wav',
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
    'gpu': False,
    'postprocess': False,
    'tta': False,
    'save': True,
    'output_image': False,
    'window_size': '512',
    'agg': 10,
    'modelFolder': False,
    'modelInstrumentalLabel': '',
    'aiModel': 'MDX-Net',
    'algo': 'Instrumentals (Min Spec)',
    'demucs_stems': 'All Stems',
    'ensChoose': 'Multi-AI Ensemble',
    'useModel': 'instrumental',
    'lastDir': None,
    'break': False,
    #Advanced Options
    'appendensem': False,
    'demucs_only': False,
    'split_mode': True,
    #MDX-Net
    'demucsmodel': True,
    'non_red': False,
    'noise_reduc': True,
    'voc_only': False,
    'inst_only': False,
    'voc_only_b': False,
    'inst_only_b': False,
    'audfile': True,
    'chunks': 'Auto',
    'n_fft_scale': 6144,
    'dim_f': 2048,
    'noise_pro_select': 'Auto Select',
    'overlap': 0.25,
    'shifts': 2,
    'overlap_b': 0.25,
    'shifts_b': 2,
    'margin': 44100,
    'channel': 64,
    'compensate': 1.03597672895,
    'mdxnetModeltype': 'Vocals (Custom)',
    'noisereduc_s': '3',
    'mixing': 'Default',
    'mdxnetModel': 'UVR-MDX-NET 1',
    'DemucsModel': 'mdx_extra',
    'DemucsModel_MDX': 'UVR_Demucs_Model_1',
    'ModelParams': 'Auto',
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
        print('dnddir ', str(dnddir))
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
        self.MDXLabel_to_path = defaultdict(lambda: '')
        self.lastMDXModels = []
        self.ModelParamsLabel_to_path = defaultdict(lambda: '')
        self.lastModelParams = []
        self.DemucsLabel_to_path = defaultdict(lambda: '')
        self.lastDemucsModels = []
        self.DemucsLabel_to_path_MDX = defaultdict(lambda: '')
        self.lastDemucsModels_MDX = []

        # -Tkinter Value Holders-
        data = load_data()
        # Paths
        self.inputPaths = data['inputPaths']
        self.inputPathop_var = tk.StringVar(value=data['inputPaths'])
        self.exportPath_var = tk.StringVar(value=data['exportPath'])
        self.saveFormat_var = tk.StringVar(value=data['saveFormat'])
        self.vrensemchoose_var = tk.StringVar(value=data['vr_ensem'])
        self.vrensemchoose_a_var = tk.StringVar(value=data['vr_ensem_a'])
        self.vrensemchoose_b_var = tk.StringVar(value=data['vr_ensem_b'])
        self.vrensemchoose_c_var = tk.StringVar(value=data['vr_ensem_c'])
        self.vrensemchoose_d_var = tk.StringVar(value=data['vr_ensem_d'])
        
        self.vrensemchoose_e_var = tk.StringVar(value=data['vr_ensem_e'])
        self.vrensemchoose_mdx_a_var = tk.StringVar(value=data['vr_ensem_mdx_a'])
        self.vrensemchoose_mdx_b_var = tk.StringVar(value=data['vr_ensem_mdx_b'])
        self.vrensemchoose_mdx_c_var = tk.StringVar(value=data['vr_ensem_mdx_c'])
        self.mdxensemchoose_var = tk.StringVar(value=data['mdx_ensem'])
        self.mdxensemchoose_b_var = tk.StringVar(value=data['mdx_ensem_b'])
        #Advanced Options
        self.appendensem_var = tk.BooleanVar(value=data['appendensem'])
        self.demucs_only_var = tk.BooleanVar(value=data['demucs_only'])
        self.split_mode_var = tk.BooleanVar(value=data['split_mode'])
        # Processing Options
        self.gpuConversion_var = tk.BooleanVar(value=data['gpu'])
        self.postprocessing_var = tk.BooleanVar(value=data['postprocess'])
        self.tta_var = tk.BooleanVar(value=data['tta'])
        self.save_var = tk.BooleanVar(value=data['save'])
        self.outputImage_var = tk.BooleanVar(value=data['output_image'])
        # MDX-NET Specific Processing Options
        self.demucsmodel_var = tk.BooleanVar(value=data['demucsmodel'])
        self.non_red_var = tk.BooleanVar(value=data['non_red'])
        self.noisereduc_var = tk.BooleanVar(value=data['noise_reduc'])
        self.chunks_var = tk.StringVar(value=data['chunks'])
        self.noisereduc_s_var = tk.StringVar(value=data['noisereduc_s'])
        self.mixing_var = tk.StringVar(value=data['mixing']) #dropdown
        # Models
        self.instrumentalModel_var = tk.StringVar(value=data['modelInstrumentalLabel'])
        # Model Test Mode
        self.modelFolder_var = tk.BooleanVar(value=data['modelFolder'])
        # Constants
        self.winSize_var = tk.StringVar(value=data['window_size'])
        self.agg_var = tk.StringVar(value=data['agg'])
        self.n_fft_scale_var = tk.StringVar(value=data['n_fft_scale'])
        self.dim_f_var = tk.StringVar(value=data['dim_f'])
        self.noise_pro_select_var = tk.StringVar(value=data['noise_pro_select'])
        self.overlap_var = tk.StringVar(value=data['overlap'])
        self.shifts_var = tk.StringVar(value=data['shifts'])
        self.overlap_b_var = tk.StringVar(value=data['overlap_b'])
        self.shifts_b_var = tk.StringVar(value=data['shifts_b'])
        
        self.channel_var = tk.StringVar(value=data['channel'])
        self.margin_var = tk.StringVar(value=data['margin'])
        self.compensate_var = tk.StringVar(value=data['compensate'])
        self.mdxnetModeltype_var = tk.StringVar(value=data['mdxnetModeltype'])
        # Instrumental or Vocal Only
        self.voc_only_var = tk.BooleanVar(value=data['voc_only'])
        self.inst_only_var = tk.BooleanVar(value=data['inst_only'])
        self.voc_only_b_var = tk.BooleanVar(value=data['voc_only_b'])
        self.inst_only_b_var = tk.BooleanVar(value=data['inst_only_b'])
        self.audfile_var = tk.BooleanVar(value=data['audfile'])
        # Choose Conversion Method
        self.aiModel_var = tk.StringVar(value=data['aiModel'])
        self.last_aiModel = self.aiModel_var.get()
        # Choose Conversion Method
        self.algo_var = tk.StringVar(value=data['algo'])
        self.last_algo = self.aiModel_var.get()
        self.demucs_stems_var = tk.StringVar(value=data['demucs_stems'])
        # Choose Ensemble
        self.ensChoose_var = tk.StringVar(value=data['ensChoose'])
        self.last_ensChoose = self.ensChoose_var.get()
        # Choose MDX-NET Model
        self.mdxnetModel_var = tk.StringVar(value=data['mdxnetModel'])
        self.DemucsModel_var = tk.StringVar(value=data['DemucsModel'])
        self.DemucsModel_MDX_var = tk.StringVar(value=data['DemucsModel_MDX'])
        self.ModelParams_var = tk.StringVar(value=data['ModelParams'])
        self.last_mdxnetModel = self.mdxnetModel_var.get()
        # Other
        self.inputPathsEntry_var = tk.StringVar(value='')
        self.lastDir = data['lastDir']  # nopep8
        self.progress_var = tk.IntVar(value=0)
        # Font
        pyglet.font.add_file('lib_v5/fonts/centurygothic/GOTHIC.TTF')
        self.font = tk.font.Font(family='Century Gothic', size=10)
        self.fontRadio = tk.font.Font(family='Century Gothic', size=8) 
        # --Widgets--
        self.create_widgets()
        self.configure_widgets()
        self.bind_widgets()
        self.place_widgets()
        
        self.update_available_models()
        self.update_states()
        self.update_loop()
   
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
                                         command=self.restart)
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

        self.command_Text.write(f'Ultimate Vocal Remover [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]\n')
       
    def configure_widgets(self):
        """Change widget styling and appearance"""

        #ttk.Style().configure('TCheckbutton', background='#0e0e0f',
        #                      font=self.font, foreground='#d4d4d4')
        #ttk.Style().configure('TRadiobutton', background='#0e0e0f',
        #                      font=("Century Gothic", "11", "bold"), foreground='#d4d4d4')
        #ttk.Style().configure('T', font=self.font, foreground='#d4d4d4')  

        #s = ttk.Style()
        #s.configure('TButton', background='blue', foreground='black', font=('Century Gothic', '9', 'bold'), relief="groove")
        

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
                                                          None, 'Multi-AI Ensemble', 'Basic VR Ensemble', 'Vocal Models', 'Manual Ensemble')
        
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
                                                text='Save Noisey Vocal',
                                                variable=self.non_red_var,
                                                )

        # Postprocessing
        self.options_post_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
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
        self.options_post_Checkbutton.place(x=35, y=21, width=0, height=5,
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
            print('last dir', self.lastDir)

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
            filename = str(self.lastDir)
            
            if sys.platform == "win32":
                os.startfile(filename)

    def start_conversion(self):
        """
        Start the conversion for all the given mp3 and wav files
        """

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
        threading.Thread(target=inference.main,
                        kwargs={
                            # Paths
                            'input_paths': input_paths,
                            'export_path': export_path,
                            'saveFormat': self.saveFormat_var.get(),
                            'vr_ensem': self.vrensemchoose_var.get(),
                            'vr_ensem_a': self.vrensemchoose_a_var.get(),
                            'vr_ensem_b': self.vrensemchoose_b_var.get(),
                            'vr_ensem_c': self.vrensemchoose_c_var.get(),
                            'vr_ensem_d': self.vrensemchoose_d_var.get(),
                            
                            'vr_ensem_e': self.vrensemchoose_e_var.get(),
                            'vr_ensem_mdx_a': self.vrensemchoose_mdx_a_var.get(),
                            'vr_ensem_mdx_b': self.vrensemchoose_mdx_b_var.get(),
                            'vr_ensem_mdx_c': self.vrensemchoose_mdx_c_var.get(),
                            
                            'mdx_ensem': self.mdxensemchoose_var.get(),
                            'mdx_ensem_b': self.mdxensemchoose_b_var.get(),
                            # Processing Options
                            'gpu': 0 if self.gpuConversion_var.get() else -1,
                            'postprocess': self.postprocessing_var.get(),
                            'appendensem': self.appendensem_var.get(),
                            'demucs_only': self.demucs_only_var.get(),
                            'split_mode': self.split_mode_var.get(),
                            'tta': self.tta_var.get(),
                            'save': self.save_var.get(),
                            'output_image': self.outputImage_var.get(),
                            'algo': self.algo_var.get(),
                            'demucs_stems': self.demucs_stems_var.get(),
                            # Models
                            'instrumentalModel': instrumentalModel_path,
                            'vocalModel': '',  # Always not needed
                            'useModel': 'instrumental',  # Always instrumental
                            # Model Folder
                            'modelFolder': self.modelFolder_var.get(),
                            # Constants
                            'window_size': window_size,
                            'agg': agg,
                            'break': False,
                            'ensChoose': ensChoose,
                            'mdxnetModel': mdxnetModel,
                            'DemucsModel': self.DemucsModel_var.get(),
                            'DemucsModel_MDX': self.DemucsModel_MDX_var.get(),
                            'ModelParams': self.ModelParams_var.get(),
                            # Other Variables (Tkinter)
                            'window': self,
                            'text_widget': self.command_Text,
                            'button_widget': self.conversion_Button,
                            'inst_menu': self.options_instrumentalModel_Optionmenu,
                            'progress_var': self.progress_var,
                            # MDX-Net Specific
                            'demucsmodel': self.demucsmodel_var.get(),
                            'non_red': self.non_red_var.get(),
                            'noise_reduc': self.noisereduc_var.get(),
                            'voc_only': self.voc_only_var.get(),
                            'inst_only': self.inst_only_var.get(),
                            'voc_only_b': self.voc_only_b_var.get(),
                            'inst_only_b': self.inst_only_b_var.get(),
                            'audfile': self.audfile_var.get(),
                            'chunks': chunks,
                            'noisereduc_s': noisereduc_s,
                            'mixing': mixing,
                            'n_fft_scale': self.n_fft_scale_var.get(),
                            'dim_f': self.dim_f_var.get(),
                            'noise_pro_select': self.noise_pro_select_var.get(),
                            'overlap': self.overlap_var.get(),
                            'shifts': self.shifts_var.get(),
                            'overlap_b': self.overlap_b_var.get(),
                            'shifts_b': self.shifts_b_var.get(),
                            'margin': self.margin_var.get(),
                            'channel': self.channel_var.get(),
                            'compensate': self.compensate_var.get(),
                            'mdxnetModeltype': self.mdxnetModeltype_var.get(),
                        },
                        daemon=True
                        ).start()
        
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
        temp_instrumentalModels_dir = os.path.join(instrumentalModels_dir, 'Main_Models')  # nopep8

        # Main models
        new_InstrumentalModels = os.listdir(temp_instrumentalModels_dir)
        
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
            
        """
        Loop through every MDX-Net model (.onnx) in the models directory
        and add to the select your model list
        """
  
        temp_MDXModels_dir = os.path.join(instrumentalModels_dir, 'MDX_Net_Models')  # nopep8

        # MDX-Net
        new_MDXModels = os.listdir(temp_MDXModels_dir)
        
        if new_MDXModels != self.lastMDXModels:
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
                    
                    self.options_mdxnetModel_Optionmenu['menu'].add_radiobutton(label=file_name_1,
                                                                        command=tk._setit(self.mdxnetModel_var, file_name_1))
            self.lastMDXModels = new_MDXModels
        
        
        """
        Loop through every Demucs model (.th, .pth) in the models directory
        and add to the select your model list
        """

        try:
            temp_DemucsModels_dir = os.path.join(instrumentalModels_dir, 'Demucs_Models')  # nopep8
            new_DemucsModels = os.listdir(temp_DemucsModels_dir)
            
            if new_DemucsModels != self.lastDemucsModels:
                #print(new_MDXModels)
                self.DemucsLabel_to_path.clear()
                self.options_DemucsModel_Optionmenu['menu'].delete(0, 'end')
                for file_name_2 in natsort.natsorted(new_DemucsModels):
                    if file_name_2.endswith(('.yaml')):
                        b = [".yaml"]
                        for char in b:
                            file_name_2 = file_name_2.replace(char, "")

                        
                        self.options_DemucsModel_Optionmenu['menu'].add_radiobutton(label=file_name_2,
                                                                            command=tk._setit(self.DemucsModel_var, file_name_2))
                self.lastDemucsModels = new_DemucsModels
        except:
            pass
        
        try:
            temp_DemucsModels_MDX_dir = os.path.join(instrumentalModels_dir, 'Demucs_Models')  # nopep8
            new_DemucsModels_MDX = os.listdir(temp_DemucsModels_MDX_dir)
            
            if new_DemucsModels_MDX != self.lastDemucsModels_MDX:
                #print(new_MDXModels)
                self.DemucsLabel_MDX_to_path.clear()
                self.options_DemucsModel_MDX_Optionmenu['menu'].delete(0, 'end')
                for file_name_3 in natsort.natsorted(new_DemucsModels_MDX):
                    if file_name_3.endswith(('.yaml')):
                        b = [".yaml"]
                        for char in b:
                            file_name_3 = file_name_3.replace(char, "")
                        
                        self.options_DemucsModel_MDX_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.DemucsModel_MDX_var, file_name_3))
                self.lastDemucsModels_MDX = new_DemucsModels_MDX
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
                #print(new_MDXModels)
                self.ModelParamsLabel_to_path.clear()
                self.options_ModelParams_Optionmenu['menu'].delete(0, 'end')
                for file_name_3 in natsort.natsorted(new_ModelParams):
                    if file_name_3.endswith(('.json', 'Auto')):
                        
                        self.options_ModelParams_Optionmenu['menu'].add_radiobutton(label=file_name_3,
                                                                            command=tk._setit(self.ModelParams_var, file_name_3))
                self.lastModelParams = new_ModelParams
        except:
            pass

            
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
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_instrumentalModel_Label.place_forget()
            self.options_instrumentalModel_Optionmenu.place_forget()
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_post_Checkbutton.configure(state=tk.DISABLED)
            self.options_post_Checkbutton.place_forget()
            self.options_tta_Checkbutton.configure(state=tk.DISABLED)
            self.options_tta_Checkbutton.place_forget()
            # self.options_image_Checkbutton.configure(state=tk.DISABLED)
            # self.options_image_Checkbutton.place_forget()
            self.options_winSize_Label.place_forget()
            self.options_winSize_Optionmenu.place_forget()
            self.options_agg_Label.place_forget()
            self.options_agg_Optionmenu.place_forget()
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_DemucsModel_Label.place_forget()
            self.options_DemucsModel_Optionmenu.place_forget()
            self.options_demucs_stems_Label.place_forget()
            self.options_demucs_stems_Optionmenu.place_forget()
            self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_b_Checkbutton.place_forget()
            self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_overlap_b_Label.place_forget()
            self.options_overlap_b_Optionmenu.place_forget()
            self.options_shifts_b_Label.place_forget()
            self.options_shifts_b_Optionmenu.place_forget()
            self.options_split_Checkbutton.configure(state=tk.DISABLED)
            self.options_split_Checkbutton.place_forget()
            
            
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
            self.options_post_Checkbutton.configure(state=tk.NORMAL)
            self.options_post_Checkbutton.place(x=35, y=21, width=0, height=5,
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
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_chunks_Label.place_forget()
            self.options_chunks_Optionmenu.place_forget()
            self.options_noisereduc_s_Label.place_forget()
            self.options_noisereduc_s_Optionmenu.place_forget()
            self.options_mdxnetModel_Label.place_forget()
            self.options_mdxnetModel_Optionmenu.place_forget()
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.place_forget()
            self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
            self.options_noisereduc_Checkbutton.place_forget()
            self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
            self.options_demucsmodel_Checkbutton.place_forget()
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.place_forget()
            self.options_DemucsModel_Label.place_forget()
            self.options_DemucsModel_Optionmenu.place_forget()
            self.options_demucs_stems_Label.place_forget()
            self.options_demucs_stems_Optionmenu.place_forget()
            self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_b_Checkbutton.place_forget()
            self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_b_Checkbutton.place_forget()
            self.options_overlap_b_Label.place_forget()
            self.options_overlap_b_Optionmenu.place_forget()
            self.options_shifts_b_Label.place_forget()
            self.options_shifts_b_Optionmenu.place_forget()
            self.options_split_Checkbutton.configure(state=tk.DISABLED)
            self.options_split_Checkbutton.place_forget()
            
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
            
            # Chunks
            self.options_chunks_Label.place(x=12, y=0, width=0, height=-10,
                                        relx=2/3, rely=2/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            self.options_chunks_Optionmenu.place(x=55, y=-2, width=-85, height=7,
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
            
            #Instrumental Only
            self.options_split_Checkbutton.configure(state=tk.NORMAL)
            self.options_split_Checkbutton.place(x=35, y=21, width=0, height=5,
                                        relx=1/3, rely=8/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

            # Forget Widgets
            self.options_save_Checkbutton.configure(state=tk.DISABLED)
            self.options_save_Checkbutton.place_forget()
            self.options_post_Checkbutton.configure(state=tk.DISABLED)
            self.options_post_Checkbutton.place_forget()
            self.options_tta_Checkbutton.configure(state=tk.DISABLED)
            self.options_tta_Checkbutton.place_forget()
            # self.options_image_Checkbutton.configure(state=tk.DISABLED)
            # self.options_image_Checkbutton.place_forget()
            self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
            self.options_demucsmodel_Checkbutton.place_forget()
            self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
            self.options_noisereduc_Checkbutton.place_forget()
            self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
            self.options_non_red_Checkbutton.place_forget()
            self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
            self.options_voc_only_Checkbutton.place_forget()
            self.options_inst_only_Checkbutton.configure(state=tk.DISABLED)
            self.options_inst_only_Checkbutton.place_forget()
            self.options_noisereduc_s_Label.place_forget()
            self.options_noisereduc_s_Optionmenu.place_forget()
            self.options_mdxnetModel_Label.place_forget()
            self.options_mdxnetModel_Optionmenu.place_forget()
            self.options_winSize_Label.place_forget()
            self.options_winSize_Optionmenu.place_forget()
            self.options_agg_Label.place_forget()
            self.options_agg_Optionmenu.place_forget()
            self.options_ensChoose_Label.place_forget()
            self.options_ensChoose_Optionmenu.place_forget()
            self.options_instrumentalModel_Label.place_forget()
            self.options_instrumentalModel_Optionmenu.place_forget()
            self.options_algo_Label.place_forget()
            self.options_algo_Optionmenu.place_forget()
            self.options_modelFolder_Checkbutton.place_forget()
            
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
                self.options_save_Checkbutton.configure(state=tk.DISABLED)
                self.options_save_Checkbutton.place_forget()
                self.options_post_Checkbutton.configure(state=tk.DISABLED)
                self.options_post_Checkbutton.place_forget()
                self.options_tta_Checkbutton.configure(state=tk.DISABLED)
                self.options_tta_Checkbutton.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget()
                # self.options_image_Checkbutton.configure(state=tk.DISABLED)
                # self.options_image_Checkbutton.place_forget()
                self.options_gpu_Checkbutton.configure(state=tk.DISABLED)
                self.options_gpu_Checkbutton.place_forget()
                self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_Checkbutton.place_forget()
                self.options_inst_only_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_Checkbutton.place_forget()
                self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
                self.options_demucsmodel_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_chunks_Label.place_forget()
                self.options_chunks_Optionmenu.place_forget()
                self.options_noisereduc_s_Label.place_forget()
                self.options_noisereduc_s_Optionmenu.place_forget()
                self.options_mdxnetModel_Label.place_forget()
                self.options_mdxnetModel_Optionmenu.place_forget()
                self.options_winSize_Label.place_forget()
                self.options_winSize_Optionmenu.place_forget()
                self.options_agg_Label.place_forget()
                self.options_agg_Optionmenu.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
                

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
                self.options_post_Checkbutton.configure(state=tk.DISABLED)
                self.options_post_Checkbutton.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_modelFolder_Checkbutton.place_forget()
                # self.options_image_Checkbutton.configure(state=tk.DISABLED)
                # self.options_image_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_algo_Label.place_forget()
                self.options_algo_Optionmenu.place_forget()
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
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
                self.options_post_Checkbutton.configure(state=tk.NORMAL)
                self.options_post_Checkbutton.place(x=35, y=21, width=0, height=5,
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
                self.options_instrumentalModel_Label.place_forget()
                self.options_instrumentalModel_Optionmenu.place_forget()
                self.options_chunks_Label.place_forget()
                self.options_chunks_Optionmenu.place_forget()
                self.options_noisereduc_s_Label.place_forget()
                self.options_noisereduc_s_Optionmenu.place_forget()
                self.options_mdxnetModel_Label.place_forget()
                self.options_mdxnetModel_Optionmenu.place_forget()
                self.options_modelFolder_Checkbutton.place_forget()
                self.options_modelFolder_Checkbutton.configure(state=tk.DISABLED)
                self.options_noisereduc_Checkbutton.place_forget()
                self.options_noisereduc_Checkbutton.configure(state=tk.DISABLED)
                self.options_demucsmodel_Checkbutton.place_forget()
                self.options_demucsmodel_Checkbutton.configure(state=tk.DISABLED)
                self.options_non_red_Checkbutton.place_forget()
                self.options_non_red_Checkbutton.configure(state=tk.DISABLED)
                self.options_DemucsModel_Label.place_forget()
                self.options_DemucsModel_Optionmenu.place_forget()
                self.options_demucs_stems_Label.place_forget()
                self.options_demucs_stems_Optionmenu.place_forget()
                self.options_voc_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_voc_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.configure(state=tk.DISABLED)
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_inst_only_b_Checkbutton.place_forget()
                self.options_overlap_b_Label.place_forget()
                self.options_overlap_b_Optionmenu.place_forget()
                self.options_shifts_b_Label.place_forget()
                self.options_shifts_b_Optionmenu.place_forget()
                self.options_split_Checkbutton.configure(state=tk.DISABLED)
                self.options_split_Checkbutton.place_forget()
                
                
        if self.inst_only_var.get() == True:
            self.options_voc_only_Checkbutton.configure(state=tk.DISABLED)
            self.voc_only_var.set(False)
            self.non_red_var.set(False)
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
            pass

    def open_newModel_filedialog(self):
        """Let user paste an MDX-Net model to use for the vocal seperation"""
        
        filename = 'models\MDX_Net_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def advanced_vr_options(self):
        """
        Open Advanced VR Options
        """
        top=Toplevel(self)

        top.geometry("600x500")
        window_height = 600
        window_width = 500
        
        top.title("Advanced VR Options")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')

        def close_win():
            top.destroy()
            self.settings()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Advanced Settings')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=Label(frame0,text="Advanced VR Options",font=("Century Gothic", "13", "bold", "underline"), justify="center", fg="#13a4c9")
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
        l0.grid(row=8,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text='Open VR Models Folder', command=self.open_Modelfolder_vr)
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=10,column=0,padx=0,pady=10)
        
        def close_win_self():
            top.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=11,column=0,padx=0,pady=0)
        
        self.ModelParamsLabel_to_path = defaultdict(lambda: '')
        self.lastModelParams = []
        
        self.update_states()
        
        
    def advanced_demucs_options(self):
        """
        Open Advanced Demucs Options
        """
        top= Toplevel(self)

        top.geometry("670x500")
        window_height = 670
        window_width = 500
        
        top.title("Advanced Demucs Options")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            top.destroy()
            self.settings()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Advanced Settings')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=Label(frame0,text="Advanced Demucs Options",font=("Century Gothic", "13", "bold", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=tk.Label(frame0, text='Chunks (Set Manually)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.chunks_var, justify='center')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Chunk Margin', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=3,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.margin_var, justify='center')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_b_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_b_var, justify='center')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Save Stems to Model Name Directory', variable=self.audfile_var) 
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Settings Test Mode', variable=self.modelFolder_var) 
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        # l0=ttk.Checkbutton(frame0, text='Basic Prediction', variable=self.audfile_var) 
        # l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Open Demucs Model Folder', command=self.open_Modelfolder_de)
        l0.grid(row=11,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=12,column=0,padx=0,pady=10)
        
        def close_win_self():
            top.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=13,column=0,padx=0,pady=0)
        
            
    def advanced_mdx_options(self):
        """
        Open Advanced MDX Options
        """
        top= Toplevel(self)

        top.geometry("670x550")
        window_height = 670
        window_width = 550
        
        top.title("Advanced MDX-Net Options")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            top.destroy()
            self.settings()

        tabControl = ttk.Notebook(top)
  
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
        
        l0=Label(frame0,text="Advanced MDX-Net Options",font=("Century Gothic", "13", "bold", "underline"), justify="center", fg="#13a4c9")
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
        
        l0=ttk.Entry(frame0, textvariable=self.compensate_var, justify='center')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Noise Profile', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=0,padx=0,pady=10)
        
        l0=ttk.OptionMenu(frame0, self.noise_pro_select_var, None, 'Auto Select', 'MDX-NET_Noise_Profile_14_kHz', 'MDX-NET_Noise_Profile_17_kHz', 'MDX-NET_Noise_Profile_Full_Band')
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Open MDX-Net Models Folder', command=self.open_newModel_filedialog)
        l0.grid(row=11,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        def close_win_self():
            top.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=13,column=0,padx=0,pady=10)
        
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
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources and increase processing times)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=5,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_var, justify='center')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_var, justify='center')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Split Mode', variable=self.split_mode_var) 
        l0.grid(row=9,column=0,padx=0,pady=10)
        
        self.update_states()
        
        frame0=Frame(tab3, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  
        
        l0=tk.Label(frame0, text='Stem Type', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=1,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxnetModeltype_var, None, 'Vocals (Default)', 'Other (Default)', 'Bass (Default)', 'Drums (Default)', 'Vocals (Custom)', 'Other (Custom)', 'Bass (Custom)', 'Drums (Custom)')
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
        
        self.DemucsLabel_MDX_to_path = defaultdict(lambda: '')
        self.lastDemucsModels_MDX = []
        
        self.update_states()
        

    def custom_ensemble(self):
        """
        Open Ensemble Custom
        """
        top= Toplevel(self)

        top.geometry("800x680")
        window_height = 680
        window_width = 800
        
        top.title("Customize Ensemble")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            top.destroy()
            self.settings()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Ensemble Options')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1, highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=30)  

        l0=Label(frame0,text="Multi-AI Ensemble Options",font=("Century Gothic", "10", "bold", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=0,padx=20,pady=10)
        
        l0=tk.Label(frame0,text='MDX-Net or Demucs Model 1\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxensemchoose_var, None, 'UVR-MDX-NET 1', 'UVR-MDX-NET 2', 'UVR-MDX-NET 3', 
                            'UVR-MDX-NET Karaoke', 'Demucs UVR Model 1', 'Demucs UVR Model 2', 'Demucs mdx_extra', 'Demucs mdx_extra_q')
        l0.grid(row=3,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nMDX-Net or Demucs Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.mdxensemchoose_b_var, None, 'No Model', 'UVR-MDX-NET 1', 'UVR-MDX-NET 2', 'UVR-MDX-NET 3', 
                            'UVR-MDX-NET Karaoke', 'Demucs UVR Model 1', 'Demucs UVR Model 2', 'Demucs mdx_extra', 'Demucs mdx_extra_q')
        l0.grid(row=5,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 1\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=7,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_a_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=9,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_b_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=11,column=0,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=12,column=0,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_mdx_c_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=13,column=0,padx=0,pady=0)
        
        l0=Label(frame0,text="Basic VR Ensemble Options",font=("Century Gothic", "10", "bold", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=1,padx=20,pady=10)
        
        l0=tk.Label(frame0,text='VR Model 1\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=2,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_a_var, None, '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=3,column=1,padx=0,pady=0)

        l0=tk.Label(frame0,text='\nVR Model 2\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=4,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_b_var, None, '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=5,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 3\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=6,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_c_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=7,column=1,padx=0,pady=0) 
        
        l0=tk.Label(frame0,text='\nVR Model 4\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=8,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_d_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=9,column=1,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='\nVR Model 5\n',font=("Century Gothic", "9"), justify="center", foreground='#13a4c9')
        l0.grid(row=10,column=1,padx=0,pady=0)
        
        l0=ttk.OptionMenu(frame0, self.vrensemchoose_e_var, None, 'No Model', '1_HP-UVR', '2_HP-UVR', '3_HP-Vocal-UVR', 
                            '4_HP-Vocal-UVR', '5_HP-Karaoke-UVR', '6_HP-Karaoke-UVR', '7_HP2-UVR', '8_HP2-UVR', 
                            '9_HP2-UVR', '10_SP-UVR-2B-32000-1', '11_SP-UVR-2B-32000-2', '12_SP-UVR-3B-44100', '13_SP-UVR-4B-44100-1',
                            '14_SP-UVR-4B-44100-2', '15_SP-UVR-MID-44100-1', '16_SP-UVR-MID-44100-2',
                            'MGM_MAIN_v4', 'MGM_HIGHEND_v4', 'MGM_LOWEND_A_v4', 'MGM_LOWEND_B_v4')
        l0.grid(row=11,column=1,padx=0,pady=0)
        
        def close_win_self():
            top.destroy()
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win_self)
        l0.grid(row=13,column=1,padx=20,pady=0)
        
        l0=Label(frame0,text="Additional Options",font=("Century Gothic", "10", "bold", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=2,padx=0,pady=10)
        
        l0=ttk.Checkbutton(frame0, text='Append Ensemble Name to Final Output', variable=self.appendensem_var) 
        l0.grid(row=2,column=2,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Open Models Directory', command=self.open_Modelfolder_filedialog)
        l0.grid(row=3,column=2,padx=0,pady=0)
        
        l0=ttk.Button(frame0,text='Back to Main Menu', command=close_win)
        l0.grid(row=4,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='Additional VR Architecture Options',font=("Century Gothic", "10", "bold", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=5,column=2,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Post-Process', variable=self.postprocessing_var) 
        l0.grid(row=6,column=2,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Save Output Image(s) of Spectrogram(s)', variable=self.outputImage_var) 
        l0.grid(row=7,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0,text='Additional Demucs v3 Options',font=("Century Gothic", "10", "bold", "underline"), justify="center", fg="#f4f4f4")
        l0.grid(row=8,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Shifts\n(Higher values use more resources)', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=9,column=2,padx=0,pady=0)
        
        l0=ttk.Entry(frame0, textvariable=self.shifts_var, justify='center')
        l0.grid(row=10,column=2,padx=0,pady=0)
        
        l0=tk.Label(frame0, text='Overlap', font=("Century Gothic", "9"), foreground='#13a4c9')
        l0.grid(row=11,column=2,padx=0,pady=0)
        
        l0=ttk.Entry(frame0, textvariable=self.overlap_var, justify='center')
        l0.grid(row=12,column=2,padx=0,pady=0)
        
        l0=ttk.Checkbutton(frame0, text='Split Mode', variable=self.split_mode_var) 
        l0.grid(row=13,column=2,padx=0,pady=0)

    def help(self):
        """
        Open Help Guide
        """
        top= Toplevel(self)
        if GetSystemMetrics(1) >= 900:
            top.geometry("1080x810")
            window_height = 810
            window_width = 1080
        elif GetSystemMetrics(1) <= 720:
            top.geometry("930x640")
            window_height = 640
            window_width = 930
        else:
            top.geometry("930x670")
            window_height = 670
            window_width = 930
        top.title("UVR Help Guide")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            top.destroy()
            self.settings()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        tab3 = ttk.Frame(tabControl)
        tab4 = ttk.Frame(tabControl)
        tab5 = ttk.Frame(tabControl)
        tab6 = ttk.Frame(tabControl)
        tab7 = ttk.Frame(tabControl)
        tab8 = ttk.Frame(tabControl)
        tab9 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='General')
        tabControl.add(tab2, text ='VR Architecture')
        tabControl.add(tab3, text ='MDX-Net')
        tabControl.add(tab4, text ='Demucs v3')
        tabControl.add(tab5, text ='Ensemble Mode')
        tabControl.add(tab6, text ='Manual Ensemble')
        tabControl.add(tab7, text ='More Info')
        tabControl.add(tab8, text ='Credits')
        tabControl.add(tab9, text ='Updates')

        tabControl.pack(expand = 1, fill ="both")

        #Configure the row/col of our frame and root window to be resizable and fill all available space
        
        tab7.grid_rowconfigure(0, weight=1)
        tab7.grid_columnconfigure(0, weight=1)
        
        tab8.grid_rowconfigure(0, weight=1)
        tab8.grid_columnconfigure(0, weight=1)
        
        tab9.grid_rowconfigure(0, weight=1)
        tab9.grid_columnconfigure(0, weight=1)
        
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
            l0=Label(frame0,text="Notes",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=20,pady=10)

            l0=Label(frame0,text="UVR is 100% free and open-source but MIT licensed.\nAll the models provided as part of UVR were trained by its core developers.\nPlease credit the core UVR developers if you choose to use any of our models or code for projects unrelated to UVR.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=2,column=0,padx=10,pady=7)
            
            l0=Label(frame0,text="Resources",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=20,pady=7, sticky=N)
            
            link = Label(frame0, text="Ultimate Vocal Remover (Official GitHub)",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui"))
            
            l0=Label(frame0,text="You can find updates, report issues, and give us a shout via our official GitHub.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=5,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="SoX - Sound eXchange",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=6,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download"))
            
            l0=Label(frame0,text="UVR relies on SoX for Noise Reduction. It's automatically included via the UVR installer but not the developer build.\nIf you are missing SoX, please download it via the link and extract the SoX archive to the following directory - lib_v5/sox",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=7,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="FFmpeg",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=8,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://www.wikihow.com/Install-FFmpeg-on-Windows"))
            
            l0=Label(frame0,text="UVR relies on FFmpeg for processing non-wav audio files.\nIt's automatically included via the UVR installer but not the developer build.\nIf you are missing FFmpeg, please see the installation guide via the link provided.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=9,column=0,padx=10,pady=7)

            link = Label(frame0, text="X-Minus AI",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=10,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://x-minus.pro/ai"))

            l0=Label(frame0,text="Many of the models provided are also on X-Minus.\nThis resource primarily benefits users without the computing resources to run the GUI or models locally.",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=11,column=0,padx=10,pady=7)
            
            link = Label(frame0, text="Official UVR Patreon",font=("Century Gothic", "14", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=12,column=0,padx=10,pady=7)
            link.bind("<Button-1>", lambda e:
            callback("https://www.patreon.com/uvr"))
            
            l0=Label(frame0,text="If you wish to support and donate to this project, click the link above and become a Patreon!",font=("Century Gothic", "13"), justify="center", fg="#F6F6F7")
            l0.grid(row=13,column=0,padx=10,pady=7)
            
            frame0=Frame(tab8,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)

            #inside frame0    
            
            l0=Label(frame0,text="Core UVR Developers",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=0,column=0,padx=20,pady=5, sticky=N)
            
            l0=Label(frame0,image=self.credits_img,font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=1,column=0,padx=10,pady=5)

            l0=Label(frame0,text="Anjok07\nAufr33",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=2,column=0,padx=10,pady=5)

            l0=Label(frame0,text="Special Thanks",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=4,column=0,padx=20,pady=10)

            l0=Label(frame0,text="DilanBoskan",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=5,column=0,padx=10,pady=5)

            l0=Label(frame0,text="Your contributions at the start of this project were essential to the success of UVR. Thank you!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=6,column=0,padx=0,pady=0)

            link = Label(frame0, text="Tsurumeso",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=7,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/tsurumeso/vocal-remover"))
            
            l0=Label(frame0,text="Developed the original VR Architecture AI code.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Kuielab & Woosung Choi",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=9,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/kuielab"))
            
            l0=Label(frame0,text="Developed the original MDX-Net AI code.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="Bas Curtiz",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=11,column=0,padx=10,pady=5)
            
            l0=Label(frame0,text="Designed the official UVR logo, icon, banner, splash screen, and interface.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=12,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Adefossez & Demucs",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=13,column=0,padx=10,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/facebookresearch/demucs"))
            
            l0=Label(frame0,text="Core developer of Facebook's Demucs Music Source Separation.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=14,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="Audio Separation Discord Community",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=15,column=0,padx=10,pady=5)
            
            l0=Label(frame0,text="Thank you for the support!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=16,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="CC Karokee & Friends Discord Community",font=("Century Gothic", "13", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=17,column=0,padx=10,pady=5)
            
            l0=Label(frame0,text="Thank you for the support!",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
            l0.grid(row=18,column=0,padx=0,pady=0)
            
            frame0=Frame(tab9,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)  
            
            l0=Label(frame0,text="Update Details",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=20,pady=10)
            
            l0=Label(frame0,text="Installing Model Expansion Pack",font=("Century Gothic", "13", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=2,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="1. Download the model expansion pack via the provided link below.\n2. Once the download has completed, click the \"Open Models Directory\" button below.\n3. Extract the \'Main Models\' folder within the downloaded archive to the opened \"models\" directory.\n4. Without restarting the application, you will now see the new models appear under the VR Architecture model selection list.",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Model Expansion Pack",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=10,pady=10)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.2.0"))
            
            l0=ttk.Button(frame0,text='Open Models Directory', command=self.open_Modelfolder_filedialog)
            l0.grid(row=5,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="\n\nBackward Compatibility",font=("Century Gothic", "13", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=6,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="The v4 Models are fully compatible with this GUI. \n1. If you already have them on your system, click the \"Open Models Directory\" button below. \n2. Place the files with extension \".pth\" into the \"Main Models\" directory. \n3. Now they will automatically appear in the VR Architecture model selection list.\n Note: The v2 models are not compatible with this GUI.\n",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=7,column=0,padx=0,pady=0)
            
            l0=ttk.Button(frame0,text='Open Models Directory', command=self.open_Modelfolder_filedialog)
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="\n\nInstalling Future Updates",font=("Century Gothic", "13", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=9,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="New updates and patches for this application can be found on the official UVR Releases GitHub page (link below).\nAny new update instructions will likely require the use of the \"Open Application Directory\" button below.",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="UVR Releases GitHub Page",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=11,column=0,padx=10,pady=10)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui/releases"))
            
            l0=ttk.Button(frame0,text='Open Application Directory', command=self.open_appdir_filedialog)
            l0.grid(row=12,column=0,padx=0,pady=0) 
        else:
            l0=Label(frame0,text="Notes",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=5,pady=5)

            l0=Label(frame0,text="UVR is 100% free and open-source but MIT licensed.\nPlease credit the core UVR developers if you choose to use any of our models or code for projects unrelated to UVR.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=2,column=0,padx=5,pady=5)
            
            l0=Label(frame0,text="Resources",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=5,pady=5, sticky=N)
            
            link = Label(frame0, text="Ultimate Vocal Remover (Official GitHub)",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui"))
            
            l0=Label(frame0,text="You can find updates, report issues, and give us a shout via our official GitHub.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=5,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="SoX - Sound eXchange",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=6,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download"))
            
            l0=Label(frame0,text="UVR relies on SoX for Noise Reduction. It's automatically included via the UVR installer but not the developer build.\nIf you are missing SoX, please download it via the link and extract the SoX archive to the following directory - lib_v5/sox",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=7,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="FFmpeg",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=8,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://www.wikihow.com/Install-FFmpeg-on-Windows"))
            
            l0=Label(frame0,text="UVR relies on FFmpeg for processing non-wav audio files.\nIf you are missing FFmpeg, please see the installation guide via the link provided.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=9,column=0,padx=5,pady=5)

            link = Label(frame0, text="X-Minus AI",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=10,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://x-minus.pro/ai"))

            l0=Label(frame0,text="Many of the models provided are also on X-Minus.\nThis resource primarily benefits users without the computing resources to run the GUI or models locally.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=11,column=0,padx=5,pady=5)
            
            link = Label(frame0, text="Official UVR Patreon",font=("Century Gothic", "11", "underline"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=12,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://www.patreon.com/uvr"))
            
            l0=Label(frame0,text="If you wish to support and donate to this project, click the link above and become a Patreon!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=13,column=0,padx=5,pady=5)
            
            frame0=Frame(tab8,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)

            #inside frame0    
            
            l0=Label(frame0,text="Core UVR Developers",font=("Century Gothic", "12", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=0,column=0,padx=20,pady=5, sticky=N)
            
            l0=Label(frame0,image=self.credits_img,font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=1,column=0,padx=5,pady=5)

            l0=Label(frame0,text="Anjok07\nAufr33",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=2,column=0,padx=5,pady=5)

            l0=Label(frame0,text="Special Thanks",font=("Century Gothic", "10", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=4,column=0,padx=20,pady=10)

            l0=Label(frame0,text="DilanBoskan",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=5,column=0,padx=5,pady=5)

            l0=Label(frame0,text="Your contributions at the start of this project were essential to the success of UVR. Thank you!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=6,column=0,padx=0,pady=0)

            link = Label(frame0, text="Tsurumeso",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=7,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/tsurumeso/vocal-remover"))
            
            l0=Label(frame0,text="Developed the original VR Architecture AI code.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Kuielab & Woosung Choi",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=9,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/kuielab"))
            
            l0=Label(frame0,text="Developed the original MDX-Net AI code.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="Bas Curtiz",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=11,column=0,padx=5,pady=5)
            
            l0=Label(frame0,text="Designed the official UVR logo, icon, banner, splash screen, and interface.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=12,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Adefossez & Demucs",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=13,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/facebookresearch/demucs"))
            
            l0=Label(frame0,text="Core developer of Facebook's Demucs Music Source Separation.",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=14,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="Audio Separation and CC Karokee & Friends Discord Communities",font=("Century Gothic", "11", "bold"), justify="center", fg="#13a4c9")
            l0.grid(row=15,column=0,padx=5,pady=5)
            
            l0=Label(frame0,text="Thank you for the support!",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
            l0.grid(row=16,column=0,padx=0,pady=0)
            
            frame0=Frame(tab9,highlightbackground='red',highlightthicknes=0)
            frame0.grid(row=0,column=0,padx=0,pady=30)  
            
            l0=Label(frame0,text="Update Details",font=("Century Gothic", "12", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=1,column=0,padx=20,pady=5)
            
            l0=Label(frame0,text="Installing Model Expansion Pack",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=2,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="1. Download the model expansion pack via the provided link below.\n2. Once the download has completed, click the \"Open Models Directory\" button below.\n3. Extract the \'Main Models\' folder within the downloaded archive to the opened \"models\" directory.\n4. Without restarting the application, you will now see the new models appear under the VR Architecture model selection list.",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=3,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="Model Expansion Pack",font=("Century Gothic", "10", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=4,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.2.0"))
            
            l0=Button(frame0,text='Open Models Directory',font=("Century Gothic", "8"), command=self.open_Modelfolder_filedialog, justify="left", wraplength=1000, bg="black", relief="ridge")
            l0.grid(row=5,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="\nBackward Compatibility",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=6,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="The v4 Models are fully compatible with this GUI. \n1. If you already have them on your system, click the \"Open Models Directory\" button below. \n2. Place the files with extension \".pth\" into the \"Main Models\" directory. \n3. Now they will automatically appear in the VR Architecture model selection list.\n Note: The v2 models are not compatible with this GUI.\n",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=7,column=0,padx=0,pady=0)
            
            l0=Button(frame0,text='Open Models Directory',font=("Century Gothic", "8"), command=self.open_Modelfolder_filedialog, justify="left", wraplength=1000, bg="black", relief="ridge")
            l0.grid(row=8,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="\nInstalling Future Updates",font=("Century Gothic", "11", "bold"), justify="center", fg="#f4f4f4")
            l0.grid(row=9,column=0,padx=0,pady=0)
            
            l0=Label(frame0,text="New updates and patches for this application can be found on the official UVR Releases GitHub page (link below).\nAny new update instructions will likely require the use of the \"Open Application Directory\" button below.",font=("Century Gothic", "11"), justify="center", fg="#f4f4f4")
            l0.grid(row=10,column=0,padx=0,pady=0)
            
            link = Label(frame0, text="UVR Releases GitHub Page",font=("Century Gothic", "10", "bold"), justify="center", fg="#13a4c9", cursor="hand2")
            link.grid(row=11,column=0,padx=5,pady=5)
            link.bind("<Button-1>", lambda e:
            callback("https://github.com/Anjok07/ultimatevocalremovergui/releases"))
            
            l0=Button(frame0,text='Open Application Directory',font=("Century Gothic", "8"), command=self.open_appdir_filedialog, justify="left", wraplength=1000, bg="black", relief="ridge")
            l0.grid(row=12,column=0,padx=0,pady=0)
            
            
    def settings(self):
        """
        Open Settings
        """
        top= Toplevel(self)

        top.geometry("600x500")
        window_height = 600
        window_width = 500
        
        top.title("Settings Guide")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win_custom_ensemble():
            top.destroy()
            self.custom_ensemble()
            
        def close_win_advanced_mdx_options():
            top.destroy()
            self.advanced_mdx_options()
            
        def close_win_advanced_demucs_options():
            top.destroy()
            self.advanced_demucs_options()
            
        def close_win_advanced_vr_options():
            top.destroy()
            self.advanced_vr_options()
            
        def close_win_error_log():
            top.destroy()
            self.error_log()
            
        def close_win_help():
            top.destroy()
            self.help()
            
        def close_win():
            top.destroy()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Settings Guide')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=Label(frame0,text="Main Menu",font=("Century Gothic", "13", "bold", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=0,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text="Ensemble Customization Options", command=close_win_custom_ensemble)
        l0.grid(row=1,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced MDX-Net Options", command=close_win_advanced_mdx_options)
        l0.grid(row=2,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced Demucs Options", command=close_win_advanced_demucs_options)
        l0.grid(row=3,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text="Advanced VR Options", command=close_win_advanced_vr_options)
        l0.grid(row=4,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text="Open Help Guide", command=close_win_help)
        l0.grid(row=5,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text='Open Error Log', command=close_win_error_log)
        l0.grid(row=6,column=0,padx=20,pady=5)
        
        l0=Label(frame0,text="Additional Options",font=("Century Gothic", "13", "bold", "underline"), justify="center", fg="#13a4c9")
        l0.grid(row=7,column=0,padx=0,pady=10)
        
        l0=ttk.Button(frame0,text='Open Application Directory', command=self.open_appdir_filedialog)
        l0.grid(row=8,column=0,padx=20,pady=5)
        
        l0=ttk.Button(frame0,text='Close Window', command=close_win)
        l0.grid(row=9,column=0,padx=20,pady=5)
        
        
    def error_log(self):
        """
        Open Error Log
        """
        top= Toplevel(self)
        if GetSystemMetrics(1) >= 900:
            top.geometry("1080x810")
            window_height = 810
            window_width = 1080
        elif GetSystemMetrics(1) <= 720:
            top.geometry("930x640")
            window_height = 640
            window_width = 930
        else:
            top.geometry("930x670")
            window_height = 670
            window_width = 930
            
        top.title("UVR Help Guide")
        
        top.resizable(False, False)  # This code helps to disable windows from resizing
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))

        top.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        # change title bar icon
        top.iconbitmap('img\\UVR-Icon-v2.ico')
        
        def close_win():
            top.destroy()
            self.settings()
            
        def close_win_self():
            top.destroy()

        tabControl = ttk.Notebook(top)
  
        tab1 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Error Log')

        tabControl.pack(expand = 1, fill ="both")
        
        tab1.grid_rowconfigure(0, weight=1)
        tab1.grid_columnconfigure(0, weight=1)

        frame0=Frame(tab1,highlightbackground='red',highlightthicknes=0)
        frame0.grid(row=0,column=0,padx=0,pady=0)  
        
        l0=Label(frame0,text="Error Details",font=("Century Gothic", "16", "bold"), justify="center", fg="#f4f4f4")
        l0.grid(row=1,column=0,padx=20,pady=10)
        
        l0=Label(frame0,text="This tab will show the raw details of the last error received.",font=("Century Gothic", "12"), justify="center", fg="#F6F6F7")
        l0.grid(row=2,column=0,padx=0,pady=0)
        
        l0=Label(frame0,text="(Click the error console below to copy the error)\n",font=("Century Gothic", "10"), justify="center", fg="#F6F6F7")
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
            
    def open_Modelfolder_filedialog(self):
        """Let user paste a ".pth" model to use for the vocal seperation"""
        filename = 'models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_Modelfolder_vr(self):
        """Let user paste a ".pth" model to use for the vocal seperation"""
        filename = 'models\Main_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_Modelfolder_de(self):
        """Let user paste a ".pth" model to use for the vocal seperation"""
        filename = 'models\Demucs_Models'

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
            
    def open_appdir_filedialog(self):
        
        pathname = '.'
        
        print(pathname)
        
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
            'exportPath': self.exportPath_var.get(),
            'inputPaths': self.inputPaths,
            'saveFormat': self.saveFormat_var.get(),
            'vr_ensem': self.vrensemchoose_var.get(),
            'vr_ensem_a': self.vrensemchoose_a_var.get(),
            'vr_ensem_b': self.vrensemchoose_b_var.get(),
            'vr_ensem_c': self.vrensemchoose_c_var.get(),
            'vr_ensem_d': self.vrensemchoose_d_var.get(),
            'vr_ensem_e': self.vrensemchoose_e_var.get(),
            'vr_ensem_mdx_a': self.vrensemchoose_mdx_a_var.get(),
            'vr_ensem_mdx_b': self.vrensemchoose_mdx_b_var.get(),
            'vr_ensem_mdx_c': self.vrensemchoose_mdx_c_var.get(),
            'mdx_ensem': self.mdxensemchoose_var.get(),
            'mdx_ensem_b': self.mdxensemchoose_b_var.get(),
            'gpu': self.gpuConversion_var.get(),
            'appendensem': self.appendensem_var.get(),
            'demucs_only': self.demucs_only_var.get(),
            'split_mode': self.split_mode_var.get(),
            'postprocess': self.postprocessing_var.get(),
            'tta': self.tta_var.get(),
            'save': self.save_var.get(),
            'output_image': self.outputImage_var.get(),
            'window_size': window_size,
            'agg': agg,
            'useModel': 'instrumental',
            'lastDir': self.lastDir,
            'modelFolder': self.modelFolder_var.get(),
            'modelInstrumentalLabel': self.instrumentalModel_var.get(),
            'aiModel': self.aiModel_var.get(),
            'algo': self.algo_var.get(),
            'demucs_stems': self.demucs_stems_var.get(),
            'ensChoose': self.ensChoose_var.get(),
            'mdxnetModel': self.mdxnetModel_var.get(),
            'DemucsModel': self.DemucsModel_var.get(),
            'DemucsModel_MDX': self.DemucsModel_MDX_var.get(),
            'ModelParams': self.ModelParams_var.get(),
            #MDX-Net
            'demucsmodel': self.demucsmodel_var.get(),
            'non_red': self.non_red_var.get(),
            'noise_reduc': self.noisereduc_var.get(),
            'voc_only': self.voc_only_var.get(),
            'inst_only': self.inst_only_var.get(),
            'voc_only_b': self.voc_only_b_var.get(),
            'inst_only_b': self.inst_only_b_var.get(),
            'audfile': self.audfile_var.get(),
            'chunks': chunks,
            'n_fft_scale': self.n_fft_scale_var.get(),
            'dim_f': self.dim_f_var.get(),
            'noise_pro_select': self.noise_pro_select_var.get(),
            'overlap': self.overlap_var.get(),
            'shifts': self.shifts_var.get(),
            'overlap_b': self.overlap_b_var.get(),
            'shifts_b': self.shifts_b_var.get(),
            'margin': self.margin_var.get(),
            'channel': self.channel_var.get(),
            'compensate': self.compensate_var.get(),
            'mdxnetModeltype': self.mdxnetModeltype_var.get(),
            'noisereduc_s': noisereduc_s,
            'mixing': mixing,
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
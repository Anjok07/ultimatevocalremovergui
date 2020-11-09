# GUI modules
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.font
from tkinterdnd2 import TkinterDnD, DND_FILES  # Enable Drag & Drop
from datetime import datetime
# Images
from PIL import Image
from PIL import ImageTk
import pickle  # Save Data
# Other Modules
import subprocess  # Run python file
# Pathfinding
import pathlib
import sys
import os
from collections import defaultdict
# Used for live text displaying
import queue
import threading  # Run the algorithm inside a thread

import inference_v2
import inference_v4

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

instrumentalModels_dir = os.path.join(base_path, 'models')
vocalModels_dir = os.path.join(base_path, 'models')
stackedModels_dir = os.path.join(base_path, 'models')
logo_path = os.path.join(base_path, 'img', 'UVR-logo.png')
refresh_path = os.path.join(base_path, 'img', 'refresh.png')
DEFAULT_DATA = {
    'export_path': '',
    'gpu': False,
    'postprocess': False,
    'tta': False,
    'output_image': False,
    'sr': 44100,
    'hop_length': 1024,
    'window_size': 512,
    'n_fft': 2048,
    'stack': False,
    'stackPasses': 1,
    'stackOnly': False,
    'saveAllStacked': False,
    'modelFolder': False,
    'aiModel': 'v4',

    'useModel': 'instrumental',
    'lastDir': None,
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


def get_model_values(model_name):
    text = model_name.replace('.pth', '')
    text_parts = text.split('_')[1:]
    model_values = {}

    for text_part in text_parts:
        if 'sr' in text_part:
            text_part = text_part.replace('sr', '')
            if text_part.isdecimal():
                try:
                    model_values['sr'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'hl' in text_part:
            text_part = text_part.replace('hl', '')
            if text_part.isdecimal():
                try:
                    model_values['hop_length'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'w' in text_part:
            text_part = text_part.replace('w', '')
            if text_part.isdecimal():
                try:
                    model_values['window_size'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass
        if 'nf' in text_part:
            text_part = text_part.replace('nf', '')
            if text_part.isdecimal():
                try:
                    model_values['n_fft'] = int(text_part)
                    continue
                except ValueError:
                    # Cannot convert string to int
                    pass

    return model_values


def drop(var, event, accept_mode: str = 'files'):
    """
    Drag & Drop verification process
    """
    path = event.data

    if accept_mode == 'folder':
        path = path.replace('{', '').replace('}', '')
        if not os.path.isdir(path):
            tk.messagebox.showerror(title='Invalid Folder',
                                    message='Your given export path is not a valid folder!')
            return
    elif accept_mode == 'files':
        # Clean path text and set path to the list of paths
        path = path[:-1]
        path = path.replace('{', '')
        path = path.split('} ')
    else:
        # Invalid accept mode
        return

    var.set(path)


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
    IMAGE_HEIGHT = 140
    FILEPATHS_HEIGHT = 90
    OPTIONS_HEIGHT = 280
    CONVERSIONBUTTON_HEIGHT = 35
    COMMAND_HEIGHT = 200
    PROGRESS_HEIGHT = 26
    PADDING = 10

    COL1_ROWS = 10
    COL2_ROWS = 8
    COL3_ROWS = 7

    def __init__(self):
        # Run the __init__ method on the TkinterDnD.Tk class
        super().__init__()
        # Calculate window height
        height = self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT
        height += self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PROGRESS_HEIGHT
        height += self.PADDING * 5  # Padding

        # --Window Settings--
        self.title('Vocal Remover')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=590,
            height=height,
            xpad=int(self.winfo_screenwidth()/2 - 550/2),
            ypad=int(self.winfo_screenheight()/2 - height/2 - 30)))
        self.configure(bg='#000000')  # Set background color to black
        self.protocol("WM_DELETE_WINDOW", self.save_values)
        self.resizable(False, False)
        self.update()

        # --Variables--
        self.logo_img = open_image(path=logo_path,
                                   size=(self.winfo_width(), 9999))
        self.refresh_img = open_image(path=refresh_path,
                                      size=(20, 20))
        self.instrumentalLabel_to_path = defaultdict(lambda: '')
        self.vocalLabel_to_path = defaultdict(lambda: '')
        self.stackedLabel_to_path = defaultdict(lambda: '')
        self.lastInstrumentalModels = []
        self.lastVocalModels = []
        self.lastStackedModels = []
        # -Tkinter Value Holders-
        data = load_data()
        # Paths
        self.exportPath_var = tk.StringVar(value=data['export_path'])
        self.inputPaths_var = tk.StringVar(value='')
        # Processing Options
        self.gpuConversion_var = tk.BooleanVar(value=data['gpu'])
        self.postprocessing_var = tk.BooleanVar(value=data['postprocess'])
        self.tta_var = tk.BooleanVar(value=data['tta'])
        self.outputImage_var = tk.BooleanVar(value=data['output_image'])
        # Models
        self.useModel_var = tk.StringVar(value=data['useModel'])
        self.instrumentalModel_var = tk.StringVar(value='')
        self.vocalModel_var = tk.StringVar(value='')
        self.stackedModel_var = tk.StringVar(value='')
        # Stacked Options
        self.stack_var = tk.BooleanVar(value=data['stack'])
        self.stackLoops_var = tk.StringVar(value=data['stackPasses'])
        self.stackOnly_var = tk.BooleanVar(value=data['stackOnly'])
        self.saveAllStacked_var = tk.BooleanVar(value=data['saveAllStacked'])
        self.modelFolder_var = tk.BooleanVar(value=data['modelFolder'])
        # Constants
        self.srValue_var = tk.StringVar(value=data['sr'])
        self.hopValue_var = tk.StringVar(value=data['hop_length'])
        self.winSize_var = tk.StringVar(value=data['window_size'])
        self.nfft_var = tk.StringVar(value=data['n_fft'])
        # AI model
        self.aiModel_var = tk.StringVar(value=data['aiModel'])
        self.last_aiModel = self.aiModel_var.get()
        # Other
        self.lastDir = data['lastDir']  # nopep8
        self.progress_var = tk.IntVar(value=0)
        # Font
        self.font = tk.font.Font(family='Helvetica', size=9, weight='bold')
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
        self.title_Label = tk.Label(master=self, bg='black',
                                    image=self.logo_img, compound=tk.TOP)
        self.filePaths_Frame = tk.Frame(master=self, bg='black')
        self.fill_filePaths_Frame()

        self.options_Frame = tk.Frame(master=self, bg='black')
        self.fill_options_Frame()

        self.conversion_Button = ttk.Button(master=self,
                                            text='Start Conversion',
                                            command=self.start_conversion)
        self.refresh_Button = ttk.Button(master=self,
                                         image=self.refresh_img,
                                         command=self.restart)

        self.progressbar = ttk.Progressbar(master=self,
                                           variable=self.progress_var)

        self.command_Text = ThreadSafeConsole(master=self,
                                              background='#a0a0a0',
                                              borderwidth=0,)
        self.command_Text.write(f'COMMAND LINE [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')  # nopep8

    def configure_widgets(self):
        """Change widget styling and appearance"""

        ttk.Style().configure('TCheckbutton', background='black',
                              font=self.font, foreground='white')
        ttk.Style().configure('TRadiobutton', background='black',
                              font=self.font, foreground='white')
        ttk.Style().configure('T', font=self.font, foreground='white')

    def bind_widgets(self):
        """Bind widgets to the drag & drop mechanic"""
        self.filePaths_saveTo_Button.drop_target_register(DND_FILES)
        self.filePaths_saveTo_Entry.drop_target_register(DND_FILES)
        self.filePaths_musicFile_Button.drop_target_register(DND_FILES)
        self.filePaths_musicFile_Entry.drop_target_register(DND_FILES)
        self.filePaths_saveTo_Button.dnd_bind('<<Drop>>',
                                              lambda e, var=self.exportPath_var: drop(var, e, accept_mode='folder'))
        self.filePaths_saveTo_Entry.dnd_bind('<<Drop>>',
                                             lambda e, var=self.exportPath_var: drop(var, e, accept_mode='folder'))
        self.filePaths_musicFile_Button.dnd_bind('<<Drop>>',
                                                 lambda e, var=self.inputPaths_var: drop(var, e, accept_mode='files'))
        self.filePaths_musicFile_Entry.dnd_bind('<<Drop>>',
                                                lambda e, var=self.inputPaths_var: drop(var, e, accept_mode='files'))

    def place_widgets(self):
        """Place main widgets"""
        self.title_Label.place(x=-2, y=-2)
        self.filePaths_Frame.place(x=10, y=self.IMAGE_HEIGHT, width=-20, height=self.FILEPATHS_HEIGHT,
                                   relx=0, rely=0, relwidth=1, relheight=0)
        self.options_Frame.place(x=25, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.PADDING, width=-50, height=self.OPTIONS_HEIGHT,
                                 relx=0, rely=0, relwidth=1, relheight=0)
        self.conversion_Button.place(x=10, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.PADDING*2, width=-20 - 40, height=self.CONVERSIONBUTTON_HEIGHT,
                                     relx=0, rely=0, relwidth=1, relheight=0)
        self.refresh_Button.place(x=-10 - 35, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.PADDING*2, width=35, height=self.CONVERSIONBUTTON_HEIGHT,
                                  relx=1, rely=0, relwidth=0, relheight=0)
        self.command_Text.place(x=15, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.PADDING*3, width=-30, height=self.COMMAND_HEIGHT,
                                relx=0, rely=0, relwidth=1, relheight=0)
        self.progressbar.place(x=25, y=self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT + self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PADDING*4, width=-50, height=self.PROGRESS_HEIGHT,
                               relx=0, rely=0, relwidth=1, relheight=0)

    def fill_filePaths_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
        # Save To Option
        self.filePaths_saveTo_Button = ttk.Button(master=self.filePaths_Frame,
                                                  text='Save to',
                                                  command=self.open_export_filedialog)
        self.filePaths_saveTo_Entry = ttk.Entry(master=self.filePaths_Frame,

                                                textvariable=self.exportPath_var,
                                                state=tk.DISABLED
                                                )
        # Select Music Files Option
        self.filePaths_musicFile_Button = ttk.Button(master=self.filePaths_Frame,
                                                     text='Select Your Audio File(s)',
                                                     command=self.open_file_filedialog)
        self.filePaths_musicFile_Entry = ttk.Entry(master=self.filePaths_Frame,
                                                   textvariable=self.inputPaths_var,
                                                   state=tk.DISABLED
                                                   )
        # -Place Widgets-
        # Save To Option
        self.filePaths_saveTo_Button.place(x=0, y=5, width=0, height=-10,
                                           relx=0, rely=0, relwidth=0.3, relheight=0.5)
        self.filePaths_saveTo_Entry.place(x=10, y=7, width=-20, height=-14,
                                          relx=0.3, rely=0, relwidth=0.7, relheight=0.5)
        # Select Music Files Option
        self.filePaths_musicFile_Button.place(x=0, y=5, width=0, height=-10,
                                              relx=0, rely=0.5, relwidth=0.4, relheight=0.5)
        self.filePaths_musicFile_Entry.place(x=10, y=7, width=-20, height=-14,
                                             relx=0.4, rely=0.5, relwidth=0.6, relheight=0.5)

    def fill_options_Frame(self):
        """Fill Frame with neccessary widgets"""
        # -Create Widgets-
        # -Column 1-
        # GPU Selection
        self.options_gpu_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='GPU Conversion',
                                                       variable=self.gpuConversion_var,
                                                       )
        # Postprocessing
        self.options_post_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                        text='Post-Process',
                                                        variable=self.postprocessing_var,
                                                        )
        # TTA
        self.options_tta_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='TTA',
                                                       variable=self.tta_var,
                                                       )
        # Save Image
        self.options_image_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                         text='Output Image',
                                                         variable=self.outputImage_var,
                                                         )
        # Use Instrumental Model
        self.options_instrumental_Radiobutton = ttk.Radiobutton(master=self.options_Frame,
                                                                text='Use Instrumental Model',
                                                                variable=self.useModel_var,
                                                                value='instrumental',
                                                                )
        # Use Vocal Model
        self.options_vocal_Radiobutton = ttk.Radiobutton(master=self.options_Frame,
                                                         text='Use Vocal Model',
                                                         variable=self.useModel_var,
                                                         value='vocal',
                                                         )
        # Stack Loops
        self.options_stack_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                         text='Stack Passes',
                                                         variable=self.stack_var,
                                                         )
        self.options_stack_Entry = ttk.Entry(master=self.options_Frame,
                                             textvariable=self.stackLoops_var,)
        # Stack Only
        self.options_stackOnly_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                             text='Stack Conversion Only',
                                                             variable=self.stackOnly_var,
                                                             )
        # Save All Stacked Outputs
        self.options_saveStack_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                             text='Save All Stacked Outputs',
                                                             variable=self.saveAllStacked_var,
                                                             )
        self.options_modelFolder_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                               text='Model Test Mode',
                                                               variable=self.modelFolder_var,
                                                               )
        # -Column 2-
        # SR
        self.options_sr_Entry = ttk.Entry(master=self.options_Frame,
                                          textvariable=self.srValue_var,)
        self.options_sr_Label = tk.Label(master=self.options_Frame,
                                         text='SR', anchor=tk.W,
                                         background='#63605f', font=self.font, foreground='white', relief="sunken")
        # HOP LENGTH
        self.options_hop_Entry = ttk.Entry(master=self.options_Frame,
                                           textvariable=self.hopValue_var,)
        self.options_hop_Label = tk.Label(master=self.options_Frame,
                                          text='HOP LENGTH', anchor=tk.W,
                                          background='#63605f', font=self.font, foreground='white', relief="sunken")
        # WINDOW SIZE
        self.options_winSize_Entry = ttk.Entry(master=self.options_Frame,
                                               textvariable=self.winSize_var,)
        self.options_winSize_Label = tk.Label(master=self.options_Frame,
                                              text='WINDOW SIZE', anchor=tk.W,
                                              background='#63605f', font=self.font, foreground='white', relief="sunken")
        # N_FFT
        self.options_nfft_Entry = ttk.Entry(master=self.options_Frame,
                                            textvariable=self.nfft_var,)
        self.options_nfft_Label = tk.Label(master=self.options_Frame,
                                           text='N_FFT', anchor=tk.W,
                                           background='#63605f', font=self.font, foreground='white', relief="sunken")
        # AI model
        self.options_aiModel_Label = tk.Label(master=self.options_Frame,
                                              text='Choose AI Engine', anchor=tk.CENTER,
                                              background='#63605f', font=self.font, foreground='white', relief="sunken")
        self.options_aiModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                         self.aiModel_var,
                                                         None, 'v2', 'v4',)
        #  "Save to", "Select Your Audio File(s)"", and "Start Conversion" Button Style
        s = ttk.Style()
        s.configure('TButton', background='blue', foreground='black', font=('Verdana', '9', 'bold'), relief="sunken")

        # -Column 3-
        #  Choose Instrumental Model
        self.options_instrumentalModel_Label = tk.Label(master=self.options_Frame,
                                                        text='Choose Instrumental Model',
                                                        background='#a7a7a7', font=self.font, relief="ridge")
        self.options_instrumentalModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                                   self.instrumentalModel_var)
        # Choose Vocal Model
        self.options_vocalModel_Label = tk.Label(master=self.options_Frame,
                                                 text='Choose Vocal Model',
                                                 background='#a7a7a7', font=self.font, relief="ridge")
        self.options_vocalModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                            self.vocalModel_var)
        # Choose Stacked Model
        self.options_stackedModel_Label = tk.Label(master=self.options_Frame,
                                                   text='Choose Stacked Model',
                                                   background='#a7a7a7', font=self.font, relief="ridge")
        self.options_stackedModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                              self.stackedModel_var,)
        self.options_model_Button = ttk.Button(master=self.options_Frame,
                                               text='Add New Model(s)',
                                               style="Bold.TButton",
                                               command=self.open_newModel_filedialog)
        # -Place Widgets-
        # -Column 1-
        self.options_gpu_Checkbutton.place(x=0, y=0, width=0, height=0,
                                           relx=0, rely=0, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_post_Checkbutton.place(x=0, y=0, width=0, height=0,
                                            relx=0, rely=1/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_tta_Checkbutton.place(x=0, y=0, width=0, height=0,
                                           relx=0, rely=2/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_image_Checkbutton.place(x=0, y=0, width=0, height=0,
                                             relx=0, rely=3/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # Model
        self.options_instrumental_Radiobutton.place(x=0, y=0, width=0, height=0,
                                                    relx=0, rely=4/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_vocal_Radiobutton.place(x=0, y=0, width=0, height=0,
                                             relx=0, rely=5/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # Stacks
        self.options_stack_Checkbutton.place(x=0, y=0, width=0, height=0,
                                             relx=0, rely=6/self.COL1_ROWS, relwidth=1/3/4*3, relheight=1/self.COL1_ROWS)
        self.options_stack_Entry.place(x=0, y=3, width=0, height=-6,
                                       relx=1/3/4*2.4, rely=6/self.COL1_ROWS, relwidth=1/3/4*0.9, relheight=1/self.COL1_ROWS)
        self.options_stackOnly_Checkbutton.place(x=0, y=0, width=0, height=0,
                                                 relx=0, rely=7/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        self.options_saveStack_Checkbutton.place(x=0, y=0, width=0, height=0,
                                                 relx=0, rely=8/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # Model Folder
        self.options_modelFolder_Checkbutton.place(x=0, y=0, width=0, height=0,
                                                   relx=0, rely=9/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)
        # -Column 2-
        # SR
        self.options_sr_Label.place(x=5, y=4, width=5, height=-8,
                                    relx=1/3, rely=0, relwidth=1/3/2, relheight=1/self.COL2_ROWS)
        self.options_sr_Entry.place(x=15, y=4, width=5, height=-8,
                                    relx=1/3 + 1/3/2, rely=0, relwidth=1/3/4, relheight=1/self.COL2_ROWS)
        # HOP LENGTH
        self.options_hop_Label.place(x=5, y=4, width=5, height=-8,
                                     relx=1/3, rely=1/self.COL2_ROWS, relwidth=1/3/2, relheight=1/self.COL2_ROWS)
        self.options_hop_Entry.place(x=15, y=4, width=5, height=-8,
                                     relx=1/3 + 1/3/2, rely=1/self.COL2_ROWS, relwidth=1/3/4, relheight=1/self.COL2_ROWS)
        # WINDOW SIZE
        self.options_winSize_Label.place(x=5, y=4, width=5, height=-8,
                                         relx=1/3, rely=2/self.COL2_ROWS, relwidth=1/3/2, relheight=1/self.COL2_ROWS)
        self.options_winSize_Entry.place(x=15, y=4, width=5, height=-8,
                                         relx=1/3 + 1/3/2, rely=2/self.COL2_ROWS, relwidth=1/3/4, relheight=1/self.COL2_ROWS)
        # N_FFT
        self.options_nfft_Label.place(x=5, y=4, width=5, height=-8,
                                      relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3/2, relheight=1/self.COL2_ROWS)
        self.options_nfft_Entry.place(x=15, y=4, width=5, height=-8,
                                      relx=1/3 + 1/3/2, rely=3/self.COL2_ROWS, relwidth=1/3/4, relheight=1/self.COL2_ROWS)
        # AI model
        self.options_aiModel_Label.place(x=5, y=4, width=-30, height=-8,
                                         relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_aiModel_Optionmenu.place(x=5, y=4, width=-30, height=-8,
                                              relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # -Column 3-
        # Choose Model
        self.options_instrumentalModel_Label.place(x=0, y=-5, width=0, height=-10,
                                                   relx=2/3, rely=0, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_instrumentalModel_Optionmenu.place(x=15, y=-10, width=-30, height=-2,
                                                        relx=2/3, rely=1/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_vocalModel_Label.place(x=0, y=-5, width=0, height=-10,
                                            relx=2/3, rely=2/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_vocalModel_Optionmenu.place(x=15, y=-10, width=-30, height=-2,
                                                 relx=2/3, rely=3/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_stackedModel_Label.place(x=0, y=-5, width=0, height=-10,
                                              relx=2/3, rely=4/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_stackedModel_Optionmenu.place(x=15, y=-10, width=-30, height=-2,
                                                   relx=2/3, rely=5/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_model_Button.place(x=15, y=0, width=-30, height=1,
                                        relx=2/3, rely=6/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)

        # -Update Binds-
        self.options_stackOnly_Checkbutton.configure(command=self.update_states)  # nopep8
        self.options_stack_Checkbutton.configure(command=self.update_states)  # nopep8
        self.options_stack_Entry.bind('<FocusOut>',
                                      lambda e: self.update_states())
        self.options_instrumental_Radiobutton.configure(command=self.update_states)  # nopep8
        self.options_vocal_Radiobutton.configure(command=self.update_states)
        # Model name decoding
        self.instrumentalModel_var.trace_add('write',
                                             lambda *args: self.decode_modelNames())
        self.vocalModel_var.trace_add('write',
                                      lambda *args: self.decode_modelNames())
        self.stackedModel_var.trace_add('write',
                                        lambda *args: self.decode_modelNames())
        # Model deselect
        self.aiModel_var.trace_add('write',
                                   lambda *args: self.deselect_models())

    # Opening filedialogs
    def open_file_filedialog(self):
        """Make user select music files"""
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

            self.inputPaths_var.set(paths)
            self.lastDir = os.path.dirname(paths[0])

    def open_export_filedialog(self):
        """Make user select a folder to export the converted files in"""
        path = tk.filedialog.askdirectory(
            parent=self,
            title=f'Select Folder',)
        if path:  # Path selected
            self.exportPath_var.set(path)

    def open_newModel_filedialog(self):
        """Let user paste a ".pth" model to use for the vocal seperation"""
        os.startfile('models')

    def start_conversion(self):
        """
        Start the conversion for all the given mp3 and wav files
        """
        # -Get all variables-
        export_path = self.exportPath_var.get()
        input_paths = self.inputPaths_var.get()
        instrumentalModel_path = self.instrumentalLabel_to_path[self.instrumentalModel_var.get()]  # nopep8
        vocalModel_path = self.vocalLabel_to_path[self.vocalModel_var.get()]
        stackedModel_path = self.stackedLabel_to_path[self.stackedModel_var.get()]  # nopep8
        # Get constants
        instrumental = get_model_values(self.instrumentalModel_var.get())
        vocal = get_model_values(self.vocalModel_var.get())
        stacked = get_model_values(self.stackedModel_var.get())
        try:
            if [bool(instrumental), bool(vocal), bool(stacked)].count(True) == 2:
                sr = DEFAULT_DATA['sr']
                hop_length = DEFAULT_DATA['hop_length']
                window_size = DEFAULT_DATA['window_size']
                n_fft = DEFAULT_DATA['n_fft']
            else:
                sr = int(self.srValue_var.get())
                hop_length = int(self.hopValue_var.get())
                window_size = int(self.winSize_var.get())
                n_fft = int(self.nfft_var.get())
            stackPasses = int(self.stackLoops_var.get())
        except ValueError:  # Non integer was put in entry box
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Input',
                                      message='Please make sure you only input integer numbers!')
            return
        except SyntaxError:  # Non integer was put in entry box
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Music File',
                                      message='You have selected an invalid music file!\nPlease make sure that your files still exist and ends with either ".mp3", ".mp4", ".m4a", ".flac", ".wav"')
            return

        # -Check for invalid inputs-
        for path in input_paths:
            if not os.path.isfile(path):
                tk.messagebox.showwarning(master=self,
                                          title='Invalid Music File',
                                          message='You have selected an invalid music file! Please make sure that the file still exists!',
                                          detail=f'File path: {path}')
                return
        if not os.path.isdir(export_path):
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Export Directory',
                                      message='You have selected an invalid export directory!\nPlease make sure that your directory still exists!')
            return
        if not self.stackOnly_var.get():
            if self.useModel_var.get() == 'instrumental':
                if not os.path.isfile(instrumentalModel_path):
                    tk.messagebox.showwarning(master=self,
                                              title='Invalid Instrumental Model File',
                                              message='You have selected an invalid instrumental model file!\nPlease make sure that your model file still exists!')
                    return
            elif self.useModel_var.get() == 'vocal':
                if not os.path.isfile(vocalModel_path):
                    tk.messagebox.showwarning(master=self,
                                              title='Invalid Vocal Model File',
                                              message='You have selected an invalid vocal model file!\nPlease make sure that your model file still exists!')
                    return
            else:
                print('THIS SHOULD NOT HAPPEN')
                exit()
        if (self.stackOnly_var.get() or
                stackPasses > 0):
            if not os.path.isfile(stackedModel_path):
                tk.messagebox.showwarning(master=self,
                                          title='Invalid Stacked Model File',
                                          message='You have selected an invalid stacked model file!\nPlease make sure that your model file still exists!')
                return

        if self.aiModel_var.get() == 'v2':
            inference = inference_v2
        elif self.aiModel_var.get() == 'v4':
            inference = inference_v4
        else:
            raise TypeError('This error should not occur.')

        # -Run the algorithm-
        threading.Thread(target=inference.main,
                         kwargs={
                             # Paths
                             'input_paths': input_paths,
                             'export_path': export_path,
                             # Processing Options
                             'gpu': 0 if self.gpuConversion_var.get() else -1,
                             'postprocess': self.postprocessing_var.get(),
                             'tta': self.tta_var.get(),  # not needed for v2
                             'output_image': self.outputImage_var.get(),
                             # Models
                             'instrumentalModel': instrumentalModel_path,
                             'vocalModel': vocalModel_path,
                             'stackModel': stackedModel_path,
                             'useModel': self.useModel_var.get(),
                             # Stack Options
                             'stackPasses': stackPasses,
                             'stackOnly': self.stackOnly_var.get(),
                             'saveAllStacked': self.saveAllStacked_var.get(),
                             # Model Folder
                             'modelFolder': self.modelFolder_var.get(),
                             # Constants
                             'sr': sr,
                             'hop_length': hop_length,
                             'window_size': window_size,
                             'n_fft': n_fft,  # not needed for v2
                             # Other Variables (Tkinter)
                             'window': self,
                             'text_widget': self.command_Text,
                             'button_widget': self.conversion_Button,
                             'progress_var': self.progress_var,
                         },
                         daemon=True
                         ).start()

    # Models
    def decode_modelNames(self):
        """
        Enable/Disable the 4 constants based on the selected model names
        """
        # Check state of model selectors
        instrumental_selectable = bool(str(self.options_instrumentalModel_Optionmenu.cget('state')) == 'normal')
        vocal_selectable = bool(str(self.options_vocalModel_Optionmenu.cget('state')) == 'normal')
        stacked_selectable = bool(str(self.options_stackedModel_Optionmenu.cget('state')) == 'normal')

        # Extract data from models name
        instrumental = get_model_values(self.instrumentalModel_var.get())
        vocal = get_model_values(self.vocalModel_var.get())
        stacked = get_model_values(self.stackedModel_var.get())

        # Assign widgets to constants
        widgetsVars = {
            'sr': [self.options_sr_Entry, self.srValue_var],
            'hop_length': [self.options_hop_Entry, self.hopValue_var],
            'window_size': [self.options_winSize_Entry, self.winSize_var],
            'n_fft': [self.options_nfft_Entry, self.nfft_var],
        }
        # Obtain data from instrumental or vocal (based on what is selected)
        modelData = instrumental if bool(instrumental) else vocal
        modelData_selectable = (instrumental_selectable or vocal_selectable)

        # Loop through each constant (key) and its widgets
        for key, (widget, var) in widgetsVars.items():
            if stacked_selectable:
                if modelData_selectable:
                    if (key in modelData.keys() and
                            key in stacked.keys()):
                        # Both models have set constants
                        widget.configure(state=tk.DISABLED)
                        var.set('%d/%d' % (modelData[key], stacked[key]))
                        continue
                else:
                    if key in stacked.keys():
                        # Only stacked selectable
                        widget.configure(state=tk.DISABLED)
                        var.set(stacked[key])
                        continue
            else:
                # Stacked model can not be selected
                if (key in modelData.keys() and
                        modelData_selectable):
                    widget.configure(state=tk.DISABLED)
                    var.set(modelData[key])
                    continue
            # If widget is already enabled, no need to reset the value
            if str(widget.cget('state')) != 'normal':
                widget.configure(state=tk.NORMAL)
                var.set(DEFAULT_DATA[key])

    def update_loop(self):
        """Update the dropdown menu"""
        self.update_available_models()

        self.after(3000, self.update_loop)

    def update_available_models(self):
        """
        Loop through every model (.pth) in the models directory
        and add to the select your model list
        """
        temp_instrumentalModels_dir = os.path.join(instrumentalModels_dir, self.aiModel_var.get(), 'Instrumental Models')  # nopep8
        temp_vocalModels_dir = os.path.join(vocalModels_dir, self.aiModel_var.get(), 'Vocal Models')
        temp_stackedModels_dir = os.path.join(stackedModels_dir, self.aiModel_var.get(), 'Stacked Models')
        # Instrumental models
        new_InstrumentalModels = os.listdir(temp_instrumentalModels_dir)
        if new_InstrumentalModels != self.lastInstrumentalModels:
            self.instrumentalLabel_to_path.clear()
            self.options_instrumentalModel_Optionmenu['menu'].delete(0, 'end')
            for file_name in new_InstrumentalModels:
                if file_name.endswith('.pth'):
                    # Add Radiobutton to the Options Menu
                    self.options_instrumentalModel_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                                      command=tk._setit(self.instrumentalModel_var, file_name))
                    # Link the files name to its absolute path
                    self.instrumentalLabel_to_path[file_name] = os.path.join(temp_instrumentalModels_dir, file_name)  # nopep8
            self.lastInstrumentalModels = new_InstrumentalModels
        # Vocal models
        new_VocalModels = os.listdir(temp_vocalModels_dir)
        if new_VocalModels != self.lastVocalModels:
            self.vocalLabel_to_path.clear()
            self.options_vocalModel_Optionmenu['menu'].delete(0, 'end')
            for file_name in new_VocalModels:
                if file_name.endswith('.pth'):
                    # Add Radiobutton to the Options Menu
                    self.options_vocalModel_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                               command=tk._setit(self.vocalModel_var, file_name))
                    # Link the files name to its absolute path
                    self.vocalLabel_to_path[file_name] = os.path.join(temp_vocalModels_dir, file_name)  # nopep8
            self.lastVocalModels = new_VocalModels
        # Stacked models
        new_stackedModels = os.listdir(temp_stackedModels_dir)
        if new_stackedModels != self.lastStackedModels:
            self.stackedLabel_to_path.clear()
            self.options_stackedModel_Optionmenu['menu'].delete(0, 'end')
            for file_name in new_stackedModels:
                if file_name.endswith('.pth'):
                    # Add Radiobutton to the Options Menu
                    self.options_stackedModel_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                                 command=tk._setit(self.stackedModel_var, file_name))
                    # Link the files name to its absolute path
                    self.stackedLabel_to_path[file_name] = os.path.join(temp_stackedModels_dir, file_name)  # nopep8
            self.lastStackedModels = new_stackedModels

    def update_states(self):
        """
        Vary the states for all widgets based
        on certain selections
        """
        try:
            stackLoops = int(self.stackLoops_var.get())
        except ValueError:
            stackLoops = 0

        # Stack Passes
        if self.stack_var.get():
            self.options_stack_Entry.configure(state=tk.NORMAL)
            if stackLoops <= 0:
                self.stackLoops_var.set(1)
                stackLoops = 1
        else:
            self.options_stack_Entry.configure(state=tk.DISABLED)
            self.stackLoops_var.set(0)
            stackLoops = 0

        # Radiobuttons
        if self.stackOnly_var.get():
            self.options_instrumental_Radiobutton.configure(text='Stack Instrumental')
            self.options_vocal_Radiobutton.configure(text='Stack Vocal')
        else:
            self.options_instrumental_Radiobutton.configure(text='Use Instrumental Model')
            self.options_vocal_Radiobutton.configure(text='Use Vocal Model')

        # Stack Only and Save All Outputs
        if stackLoops > 0:
            self.options_stackOnly_Checkbutton.configure(state=tk.NORMAL)
            self.options_saveStack_Checkbutton.configure(state=tk.NORMAL)
        else:
            self.options_stackOnly_Checkbutton.configure(state=tk.DISABLED)
            self.options_saveStack_Checkbutton.configure(state=tk.DISABLED)
            self.saveAllStacked_var.set(False)
            self.stackOnly_var.set(False)

        # Models
        if self.stackOnly_var.get():
            # Instrumental Model
            self.options_instrumentalModel_Label.configure(foreground='#777')
            self.options_instrumentalModel_Optionmenu.configure(state=tk.DISABLED)  # nopep8
            self.instrumentalModel_var.set('')
            # Vocal Model
            self.options_vocalModel_Label.configure(foreground='#777')
            self.options_vocalModel_Optionmenu.configure(state=tk.DISABLED)
            self.vocalModel_var.set('')
            # Stack Model
            self.options_stackedModel_Label.configure(foreground='#000')
            self.options_stackedModel_Optionmenu.configure(state=tk.NORMAL)  # nopep8
        elif self.useModel_var.get() == 'instrumental':
            # Instrumental Model
            self.options_instrumentalModel_Label.configure(foreground='#000')
            self.options_instrumentalModel_Optionmenu.configure(state=tk.NORMAL)  # nopep8
            # Vocal Model
            self.options_vocalModel_Label.configure(foreground='#777')
            self.options_vocalModel_Optionmenu.configure(state=tk.DISABLED)
            self.vocalModel_var.set('')
            # Stack Model
            if stackLoops > 0:
                self.options_stackedModel_Label.configure(foreground='#000')
                self.options_stackedModel_Optionmenu.configure(state=tk.NORMAL)  # nopep8
            else:
                self.options_stackedModel_Label.configure(foreground='#777')
                self.options_stackedModel_Optionmenu.configure(state=tk.DISABLED)  # nopep8
                self.stackedModel_var.set('')
        else:
            # Instrumental Model
            self.options_instrumentalModel_Label.configure(foreground='#777')
            self.options_instrumentalModel_Optionmenu.configure(state=tk.DISABLED)  # nopep8
            self.instrumentalModel_var.set('')
            # Vocal Model
            self.options_vocalModel_Label.configure(foreground='#000')
            self.options_vocalModel_Optionmenu.configure(state=tk.NORMAL)
            # Stack Model
            if stackLoops > 0:
                self.options_stackedModel_Label.configure(foreground='#000')
                self.options_stackedModel_Optionmenu.configure(state=tk.NORMAL)  # nopep8
            else:
                self.options_stackedModel_Label.configure(foreground='#777')
                self.options_stackedModel_Optionmenu.configure(state=tk.DISABLED)  # nopep8
                self.stackedModel_var.set('')

        if self.aiModel_var.get() == 'v2':
            self.options_tta_Checkbutton.configure(state=tk.DISABLED)
            self.options_nfft_Label.place_forget()
            self.options_nfft_Entry.place_forget()
        else:
            self.options_tta_Checkbutton.configure(state=tk.NORMAL)
            self.options_nfft_Label.place(x=5, y=4, width=5, height=-8,
                                          relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3/2, relheight=1/self.COL2_ROWS)
            self.options_nfft_Entry.place(x=15, y=4, width=5, height=-8,
                                          relx=1/3 + 1/3/2, rely=3/self.COL2_ROWS, relwidth=1/3/4, relheight=1/self.COL2_ROWS)

        self.decode_modelNames()

    def deselect_models(self):
        """
        Run this method on version change
        """
        if self.aiModel_var.get() == self.last_aiModel:
            return
        else:
            self.last_aiModel = self.aiModel_var.get()

        self.instrumentalModel_var.set('')
        self.vocalModel_var.set('')
        self.stackedModel_var.set('')

        self.srValue_var.set(DEFAULT_DATA['sr'])
        self.hopValue_var.set(DEFAULT_DATA['hop_length'])
        self.winSize_var.set(DEFAULT_DATA['window_size'])
        self.nfft_var.set(DEFAULT_DATA['n_fft'])

        self.update_available_models()
        self.update_states()

    def restart(self):
        """
        Restart the application after asking for confirmation
        """
        save = tk.messagebox.askyesno(title='Confirmation',
                                      message='The application will restart. Do you want to save the data?')
        if save:
            self.save_values()
        subprocess.Popen(f'python "{__file__}"', shell=True)
        exit()

    def save_values(self):
        """
        Save the data of the application
        """
        export_path = self.exportPath_var.get()
        # Get constants
        instrumental = get_model_values(self.instrumentalModel_var.get())
        vocal = get_model_values(self.vocalModel_var.get())
        stacked = get_model_values(self.stackedModel_var.get())
        if [bool(instrumental), bool(vocal), bool(stacked)].count(True) == 2:
            sr = DEFAULT_DATA['sr']
            hop_length = DEFAULT_DATA['hop_length']
            window_size = DEFAULT_DATA['window_size']
            n_fft = DEFAULT_DATA['n_fft']
        else:
            sr = self.srValue_var.get()
            hop_length = self.hopValue_var.get()
            window_size = self.winSize_var.get()
            n_fft = self.nfft_var.get()

        # -Save Data-
        save_data(data={
            'export_path': export_path,
            'gpu': self.gpuConversion_var.get(),
            'postprocess': self.postprocessing_var.get(),
            'tta': self.tta_var.get(),
            'output_image': self.outputImage_var.get(),
            'stack': self.stack_var.get(),
            'stackOnly': self.stackOnly_var.get(),
            'stackPasses': self.stackLoops_var.get(),
            'saveAllStacked': self.saveAllStacked_var.get(),
            'sr': sr,
            'hop_length': hop_length,
            'window_size': window_size,
            'n_fft': n_fft,
            'useModel': self.useModel_var.get(),
            'lastDir': self.lastDir,
            'modelFolder': self.modelFolder_var.get(),
            'aiModel': self.aiModel_var.get(),
        })

        self.destroy()


if __name__ == "__main__":
    root = MainWindow()

    root.mainloop()

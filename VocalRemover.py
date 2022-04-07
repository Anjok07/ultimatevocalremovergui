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
import subprocess
from collections import defaultdict
# Used for live text displaying
import queue
import threading  # Run the algorithm inside a thread

from pathlib import Path


import inference_v5

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
banner_path = os.path.join(base_path, 'img', 'UVR-banner.png')
refresh_path = os.path.join(base_path, 'img', 'refresh.png')
DEFAULT_DATA = {
    'exportPath': '',
    'inputPaths': [],
    'gpu': False,
    'postprocess': False,
    'tta': False,
    'output_image': False,
    'window_size': '512',
    'agg': 10,
    'modelFolder': False,
    'modelInstrumentalLabel': '',
    #'aiModel': 'v5',
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


def drop(event, accept_mode: str = 'files'):
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
        # Set Variables
        root.exportPath_var.set(path)
    elif accept_mode == 'files':
        # Clean path text and set path to the list of paths
        path = path.replace('{', '')
        path = path.split('} ')
        path[-1] = path[-1].replace('}', '')
        # Set Variables
        root.inputPaths = path
        root.update_inputPaths()
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
    IMAGE_HEIGHT = 140
    FILEPATHS_HEIGHT = 80
    OPTIONS_HEIGHT = 190
    CONVERSIONBUTTON_HEIGHT = 35
    COMMAND_HEIGHT = 200
    PROGRESS_HEIGHT = 26
    PADDING = 10

    COL1_ROWS = 6
    COL2_ROWS = 5
    COL3_ROWS = 6

    def __init__(self):
        # Run the __init__ method on the tk.Tk class
        super().__init__()
        # Calculate window height
        height = self.IMAGE_HEIGHT + self.FILEPATHS_HEIGHT + self.OPTIONS_HEIGHT
        height += self.CONVERSIONBUTTON_HEIGHT + self.COMMAND_HEIGHT + self.PROGRESS_HEIGHT
        height += self.PADDING * 5  # Padding

        # --Window Settings--
        self.title('Vocal Remover')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=620,
            height=height,
            xpad=int(self.winfo_screenwidth()/2 - 550/2),
            ypad=int(self.winfo_screenheight()/2 - height/2 - 30)))
        self.configure(bg='#000000')  # Set background color to black
        self.protocol("WM_DELETE_WINDOW", self.save_values)
        self.resizable(False, False)
        self.update()

        # --Variables--
        self.logo_img = open_image(path=banner_path,
                                   size=(self.winfo_width(), 9999))
        self.refresh_img = open_image(path=refresh_path,
                                      size=(20, 20))
        self.instrumentalLabel_to_path = defaultdict(lambda: '')
        self.lastInstrumentalModels = []
        # -Tkinter Value Holders-
        data = load_data()
        # Paths
        self.exportPath_var = tk.StringVar(value=data['exportPath'])
        self.inputPaths = data['inputPaths']
        # Processing Options
        self.gpuConversion_var = tk.BooleanVar(value=data['gpu'])
        self.postprocessing_var = tk.BooleanVar(value=data['postprocess'])
        self.tta_var = tk.BooleanVar(value=data['tta'])
        self.outputImage_var = tk.BooleanVar(value=data['output_image'])
        # Models
        self.instrumentalModel_var = tk.StringVar(value=data['modelInstrumentalLabel'])
        # Model Test Mode
        self.modelFolder_var = tk.BooleanVar(value=data['modelFolder'])
        # Constants
        self.winSize_var = tk.StringVar(value=data['window_size'])
        self.agg_var = tk.StringVar(value=data['agg'])
        # AI model
        #self.aiModel_var = tk.StringVar(value=data['aiModel'])
        #self.last_aiModel = self.aiModel_var.get()
        # Other
        self.inputPathsEntry_var = tk.StringVar(value='')
        self.lastDir = data['lastDir']  # nopep8
        self.progress_var = tk.IntVar(value=0)
        # Font
        self.font = tk.font.Font(family='Microsoft JhengHei', size=9, weight='bold')
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
                                              lambda e: drop(e, accept_mode='folder'))
        self.filePaths_saveTo_Entry.dnd_bind('<<Drop>>',
                                             lambda e: drop(e, accept_mode='folder'))
        self.filePaths_musicFile_Button.dnd_bind('<<Drop>>',
                                                 lambda e: drop(e, accept_mode='files'))
        self.filePaths_musicFile_Entry.dnd_bind('<<Drop>>',
                                                lambda e: drop(e, accept_mode='files'))

    def place_widgets(self):
        """Place main widgets"""
        self.title_Label.place(x=-2, y=-2)
        self.filePaths_Frame.place(x=10, y=155, width=-20, height=self.FILEPATHS_HEIGHT,
                                   relx=0, rely=0, relwidth=1, relheight=0)
        self.options_Frame.place(x=25, y=250, width=-50, height=self.OPTIONS_HEIGHT,
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
                                                   textvariable=self.inputPathsEntry_var,
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
        self.options_modelFolder_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                               text='Model Test Mode',
                                                               variable=self.modelFolder_var,
                                                               )
        # -Column 2-

        # WINDOW SIZE
        self.options_winSize_Label = tk.Label(master=self.options_Frame,
                                              text='Window Size', anchor=tk.CENTER,
                                              background='#404040', font=self.font, foreground='white', relief="groove")
        self.options_winSize_Optionmenu = ttk.OptionMenu(self.options_Frame, 
                                                         self.winSize_var, 
                                                         None, '320', '512','1024')

        # AGG
        self.options_agg_Entry = ttk.Entry(master=self.options_Frame,
                                            textvariable=self.agg_var, justify='center')
        self.options_agg_Label = tk.Label(master=self.options_Frame,
                                           text='Aggression Setting',
                                           background='#404040', font=self.font, foreground='white', relief="groove")
        
        # AI model
        # self.options_aiModel_Label = tk.Label(master=self.options_Frame,
        #                                       text='Choose AI Engine', anchor=tk.CENTER,
        #                                       background='#63605f', font=self.font, foreground='white', relief="sunken")
        # self.options_aiModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
        #                                                  self.aiModel_var,
        #                                                  None, 'v5')


        #  "Save to", "Select Your Audio File(s)"", and "Start Conversion" Button Style
        s = ttk.Style()
        s.configure('TButton', background='blue', foreground='black', font=('Microsoft JhengHei', '9', 'bold'), relief="groove")

        # -Column 3-
        #  Choose Instrumental Model
        self.options_instrumentalModel_Label = tk.Label(master=self.options_Frame,
                                                        text='Choose Main Model',
                                                        background='#404040', font=self.font, foreground='white', relief="groove")
        self.options_instrumentalModel_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                                   self.instrumentalModel_var)
        
        # Add New Model Button
        self.options_model_Button = ttk.Button(master=self.options_Frame,
                                               text='Open Export Directory',
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
        self.options_modelFolder_Checkbutton.place(x=0, y=0, width=0, height=0,
                                            relx=0, rely=4/self.COL1_ROWS, relwidth=1/3, relheight=1/self.COL1_ROWS)

        # -Column 2-

        self.options_instrumentalModel_Label.place(x=-15, y=6, width=0, height=-10,
                                    relx=1/3, rely=0/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_instrumentalModel_Optionmenu.place(x=-15, y=6, width=0, height=-10,
                                    relx=1/3, rely=1/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        self.options_model_Button.place(x=0, y=0, width=-30, height=-8,
                                    relx=1/3, rely=3/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # -Column 3-
        
        # WINDOW
        self.options_winSize_Label.place(x=35, y=6, width=-40, height=-10,
                                                   relx=2/3, rely=0, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_winSize_Optionmenu.place(x=80, y=6, width=-133, height=-10,
                                                        relx=2/3, rely=1/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        
        # AGG
        self.options_agg_Label.place(x=35, y=6, width=-40, height=-10,
                                    relx=2/3, rely=2/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
        self.options_agg_Entry.place(x=80, y=6, width=-133, height=-10,
                                    relx=2/3, rely=3/self.COL3_ROWS, relwidth=1/3, relheight=1/self.COL3_ROWS)
    
        
        # AI model
        # self.options_aiModel_Label.place(x=5, y=-5, width=-30, height=-8,
        #                                  relx=1/3, rely=5/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        # self.options_aiModel_Optionmenu.place(x=5, y=-5, width=-30, height=-8,
        #                                       relx=1/3, rely=6/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)

        # Model deselect
        # self.aiModel_var.trace_add('write',
        #                            lambda *args: self.deselect_models())

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
            self.inputPaths = paths
            self.update_inputPaths()
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
        filename = self.exportPath_var.get()

        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    def start_conversion(self):
        """
        Start the conversion for all the given mp3 and wav files
        """
        # -Get all variables-
        export_path = self.exportPath_var.get()
        input_paths = self.inputPaths
        instrumentalModel_path = self.instrumentalLabel_to_path[self.instrumentalModel_var.get()]  # nopep8
        # Get constants
        instrumental = self.instrumentalModel_var.get()
        try:
            if [bool(instrumental)].count(True) == 2: #CHECKTHIS
                window_size = DEFAULT_DATA['window_size']
                agg = DEFAULT_DATA['agg']
            else:
                window_size = int(self.winSize_var.get())
                agg = int(self.agg_var.get())
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

        # if self.aiModel_var.get() == 'v4':
        #     inference = inference_v4
        # elif self.aiModel_var.get() == 'v5':
        #     inference = inference_v5
        # else:
        #     raise TypeError('This error should not occur.')
        
        inference = inference_v5

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
                             'vocalModel': '',  # Always not needed
                             'useModel': 'instrumental',  # Always instrumental
                             # Model Folder
                             'modelFolder': self.modelFolder_var.get(),
                             # Constants
                             'window_size': window_size,
                             'agg': agg,
                             # Other Variables (Tkinter)
                             'window': self,
                             'text_widget': self.command_Text,
                             'button_widget': self.conversion_Button,
                             'progress_var': self.progress_var,
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

        self.after(3000, self.update_loop)

    def update_available_models(self):
        """
        Loop through every model (.pth) in the models directory
        and add to the select your model list
        """
        #temp_instrumentalModels_dir = os.path.join(instrumentalModels_dir, self.aiModel_var.get(), 'Main Models')  # nopep8
        temp_instrumentalModels_dir = os.path.join(instrumentalModels_dir, 'Main Models')  # nopep8

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


    def update_states(self):
        """
        Vary the states for all widgets based
        on certain selections
        """

        # Models
        # self.options_instrumentalModel_Label.configure(foreground='#000')
        # self.options_instrumentalModel_Optionmenu.configure(state=tk.NORMAL)  # nopep8

        # if self.aiModel_var.get() == 'v5':
        #     self.options_tta_Checkbutton.configure(state=tk.NORMAL)
        #     self.options_agg_Label.place(x=5, y=-5, width=-30, height=-8,
        #                             relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #     self.options_agg_Entry.place(x=5, y=-4, width=-30, height=-8,
        #                             relx=1/3, rely=4/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
        # else:
        #     self.options_tta_Checkbutton.configure(state=tk.NORMAL)
        #     self.options_agg_Label.place(x=5, y=-5, width=-30, height=-8,
        #                             relx=1/3, rely=3/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
        #     self.options_agg_Entry.place(x=5, y=-4, width=-30, height=-8,
        #                             relx=1/3, rely=4/self.COL2_ROWS, relwidth=1/3, relheight=1/self.COL2_ROWS)
            
            
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

        self.winSize_var.set(DEFAULT_DATA['window_size'])
        self.agg_var.set(DEFAULT_DATA['agg'])

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
        # Get constants
        instrumental = self.instrumentalModel_var.get()
        if [bool(instrumental)].count(True) == 2: #Checkthis
            window_size = DEFAULT_DATA['window_size']
            agg = DEFAULT_DATA['agg']
        else:
            window_size = self.winSize_var.get()
            agg = self.agg_var.get()

        # -Save Data-
        save_data(data={
            'exportPath': self.exportPath_var.get(),
            'inputPaths': self.inputPaths,
            'gpu': self.gpuConversion_var.get(),
            'postprocess': self.postprocessing_var.get(),
            'tta': self.tta_var.get(),
            'output_image': self.outputImage_var.get(),
            'window_size': window_size,
            'agg': agg,
            'useModel': 'instrumental',
            'lastDir': self.lastDir,
            'modelFolder': self.modelFolder_var.get(),
            'modelInstrumentalLabel': self.instrumentalModel_var.get(),
            #'aiModel': self.aiModel_var.get(),
        })

        self.destroy()


if __name__ == "__main__":
    root = MainWindow()

    root.mainloop()

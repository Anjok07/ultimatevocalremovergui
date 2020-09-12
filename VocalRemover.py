# GUI modules
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
import tkinter.font
from datetime import datetime
# Images
from PIL import Image
from PIL import ImageTk
import pickle  # Save Data
# Other Modules
import subprocess  # Run python file
# Pathfinding
import pathlib
import os
from collections import defaultdict
# Used for live text displaying
import queue
import threading  # Run the algorithm inside a thread

import torch
import inference

# --Global Variables--
base_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_path)  # Change the current working directory to the base path
models_dir = os.path.join(base_path, 'models')
logo_path = os.path.join(base_path, 'Images/UVR-logo.png')
DEFAULT_DATA = {
    'exportPath': '',
    'gpuConversion': False,
    'postprocessing': False,
    'mask': False,
    'stackLoops': False,
    'srValue': 44100,
    'hopValue': 1024,
    'stackLoopsNum': 1,
    'winSize': 512,
}
# Supported Music Files
AVAILABLE_FORMATS = ['.mp3', '.mp4', '.m4a', '.flac', '.wav']


def open_image(path: str, size: tuple = None, keep_aspect: bool = True, rotate: int = 0) -> tuple:
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
    Returns(tuple):
        (ImageTk.PhotoImage, Image)
    """
    img = Image.open(path)
    ratio = img.height/img.width
    img = img.rotate(angle=-rotate)
    if size is not None:
        size = (int(size[0]), int(size[1]))
        if keep_aspect:
            img = img.resize((size[0], int(size[0] * ratio)), Image.ANTIALIAS)
        else:
            img = img.resize(size, Image.ANTIALIAS)
    img = img.convert(mode='RGBA')
    return ImageTk.PhotoImage(img), img


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


class MainWindow(tk.Tk):
    # --Constants--
    # None

    def __init__(self):
        # Run the __init__ method on the tk.Tk class
        super().__init__()

        # --Window Settings--
        self.title('Desktop Application')
        # Set Geometry and Center Window
        self.geometry('{width}x{height}+{xpad}+{ypad}'.format(
            width=530,
            height=690,
            xpad=int(self.winfo_screenwidth()/2 - 530/2),
            ypad=int(self.winfo_screenheight()/2 - 690/2)))
        self.configure(bg='#FFFFFF')  # Set background color to white
        self.resizable(False, False)
        self.update()

        # --Variables--
        self.logo_img = open_image(path=logo_path,
                                   size=(self.winfo_width(), 9999),
                                   keep_aspect=True)[0]
        self.label_to_path = defaultdict(lambda: '')
        # -Tkinter Value Holders-
        data = load_data()
        self.exportPath_var = tk.StringVar(value=data['exportPath'])
        self.filePaths = ''
        self.gpuConversion_var = tk.BooleanVar(value=data['gpuConversion'])
        self.postprocessing_var = tk.BooleanVar(value=data['postprocessing'])
        self.mask_var = tk.BooleanVar(value=data['mask'])
        self.stackLoops_var = tk.IntVar(value=data['stackLoops'])
        self.srValue_var = tk.IntVar(value=data['srValue'])
        self.hopValue_var = tk.IntVar(value=data['hopValue'])
        self.winSize_var = tk.IntVar(value=data['winSize'])
        self.stackLoopsNum_var = tk.IntVar(value=data['stackLoopsNum'])
        self.model_var = tk.StringVar(value='')

        self.progress_var = tk.IntVar(value=0)

        # --Widgets--
        self.create_widgets()
        self.configure_widgets()
        self.place_widgets()

        self.update_available_models()
        self.update_stack_state()

    # -Widget Methods-
    def create_widgets(self):
        """Create window widgets"""
        self.title_Label = tk.Label(master=self, bg='white',
                                    image=self.logo_img, compound=tk.TOP)
        self.filePaths_Frame = tk.Frame(master=self, bg='white')
        self.fill_filePaths_Frame()

        self.options_Frame = tk.Frame(master=self, bg='white')
        self.fill_options_Frame()

        self.conversion_Button = ttk.Button(master=self,
                                            text='Start Conversion',
                                            command=self.start_conversion)

        self.progressbar = ttk.Progressbar(master=self,
                                           variable=self.progress_var)

        self.command_Text = ThreadSafeConsole(master=self,
                                              background='#EFEFEF',
                                              borderwidth=0,)
        self.command_Text.write(f'COMMAND LINE [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')  # nopep8

    def configure_widgets(self):
        """Change widget styling and appearance"""
        ttk.Style().configure('TCheckbutton', background='white')

    def place_widgets(self):
        """Place main widgets"""
        self.title_Label.place(x=-2, y=-2)

        self.filePaths_Frame.place(x=10, y=0, width=-20, height=0,
                                   relx=0, rely=0.19, relwidth=1, relheight=0.14)
        self.options_Frame.place(x=25, y=15, width=-50, height=-30,
                                 relx=0, rely=0.33, relwidth=1, relheight=0.23)
        self.conversion_Button.place(x=10, y=5, width=-20, height=-10,
                                     relx=0, rely=0.56, relwidth=1, relheight=0.07)
        self.command_Text.place(x=15, y=10, width=-30, height=-10,
                                relx=0, rely=0.63, relwidth=1, relheight=0.28)
        self.progressbar.place(x=25, y=15, width=-50, height=-30,
                               relx=0, rely=0.91, relwidth=1, relheight=0.09)

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
                                                   text=self.filePaths,
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
        # GPU Selection
        self.options_gpu_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                       text='GPU Conversion',
                                                       variable=self.gpuConversion_var,
                                                       )
        # Postprocessing
        self.options_post_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                        text='Post-Process (Dev Opt)',
                                                        variable=self.postprocessing_var,
                                                        )
        # Mask
        self.options_mask_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                        text='Save Mask PNG',
                                                        variable=self.mask_var,
                                                        )
        # SR
        self.options_sr_Entry = ttk.Entry(master=self.options_Frame,
                                          textvariable=self.srValue_var,)
        self.options_sr_Label = tk.Label(master=self.options_Frame,
                                         text='SR', anchor=tk.W,
                                         background='white')
        # HOP LENGTH
        self.options_hop_Entry = ttk.Entry(master=self.options_Frame,
                                           textvariable=self.hopValue_var,)
        self.options_hop_Label = tk.Label(master=self.options_Frame,
                                          text='HOP LENGTH', anchor=tk.W,
                                          background='white')
        # WINDOW SIZE
        self.options_winSize_Entry = ttk.Entry(master=self.options_Frame,
                                               textvariable=self.winSize_var,)
        self.options_winSize_Label = tk.Label(master=self.options_Frame,
                                              text='WINDOW SIZE', anchor=tk.W,
                                              background='white')
        # Stack Loops
        self.options_stack_Checkbutton = ttk.Checkbutton(master=self.options_Frame,
                                                         text='Stack Passes',
                                                         variable=self.stackLoops_var,
                                                         )
        self.options_stack_Entry = ttk.Entry(master=self.options_Frame,
                                             textvariable=self.stackLoopsNum_var,)
        self.options_stack_Checkbutton.configure(command=self.update_stack_state)  # nopep8
        # Choose Model
        self.options_model_Label = tk.Label(master=self.options_Frame,
                                            text='Choose Your Model',
                                            background='white')
        self.options_model_Optionmenu = ttk.OptionMenu(self.options_Frame,
                                                       self.model_var,
                                                       1,
                                                       *[1, 2])
        self.options_model_Button = ttk.Button(master=self.options_Frame,
                                               text='Add Your Own Model',
                                               command=self.open_newModel_filedialog)
        # -Place Widgets-
        # GPU Selection
        self.options_gpu_Checkbutton.place(x=0, y=0, width=0, height=0,
                                           relx=0, rely=0, relwidth=1/3, relheight=1/4)
        self.options_post_Checkbutton.place(x=0, y=0, width=0, height=0,
                                            relx=0, rely=1/4, relwidth=1/3, relheight=1/4)
        self.options_mask_Checkbutton.place(x=0, y=0, width=0, height=0,
                                            relx=0, rely=2/4, relwidth=1/3, relheight=1/4)
        # Stack Loops
        self.options_stack_Checkbutton.place(x=0, y=0, width=0, height=0,
                                             relx=0, rely=3/4, relwidth=1/3/4*3, relheight=1/4)
        self.options_stack_Entry.place(x=0, y=4, width=0, height=-8,
                                       relx=1/3/4*2.4, rely=3/4, relwidth=1/3/4*0.9, relheight=1/4)
        # SR
        self.options_sr_Entry.place(x=-5, y=4, width=5, height=-8,
                                    relx=1/3, rely=0, relwidth=1/3/4, relheight=1/4)
        self.options_sr_Label.place(x=10, y=4, width=-10, height=-8,
                                    relx=1/3/4 + 1/3, rely=0, relwidth=1/3/4*3, relheight=1/4)
        # HOP LENGTH
        self.options_hop_Entry.place(x=-5, y=4, width=5, height=-8,
                                     relx=1/3, rely=1/4, relwidth=1/3/4, relheight=1/4)
        self.options_hop_Label.place(x=10, y=4, width=-10, height=-8,
                                     relx=1/3/4 + 1/3, rely=1/4, relwidth=1/3/4*3, relheight=1/4)
        # WINDOW SIZE
        self.options_winSize_Entry.place(x=-5, y=4, width=5, height=-8,
                                         relx=1/3, rely=2/4, relwidth=1/3/4, relheight=1/4)
        self.options_winSize_Label.place(x=10, y=4, width=-10, height=-8,
                                         relx=1/3/4 + 1/3, rely=2/4, relwidth=1/3/4*3, relheight=1/4)
        # Choose Model
        self.options_model_Label.place(x=0, y=0, width=0, height=-10,
                                       relx=2/3, rely=0, relwidth=1/3, relheight=1/3)
        self.options_model_Optionmenu.place(x=15, y=-2.5, width=-30, height=-10,
                                            relx=2/3, rely=1/3, relwidth=1/3, relheight=1/3)
        self.options_model_Button.place(x=15, y=0, width=-30, height=-5,
                                        relx=2/3, rely=2/3, relwidth=1/3, relheight=1/3)

    # Opening filedialogs
    def open_file_filedialog(self):
        """Make user select music files"""
        paths = tk.filedialog.askopenfilenames(
            parent=self,
            title=f'Select Music Files',
            initialdir='/',
            initialfile='',
            filetypes=[
                ('; '.join(AVAILABLE_FORMATS).replace('.', ''),
                 '*' + ' *'.join(AVAILABLE_FORMATS)),
            ])
        if paths:  # Path selected
            for path in paths:
                if not path.lower().endswith(tuple(AVAILABLE_FORMATS)):
                    tk.messagebox.showerror(master=self,
                                            title='Invalid File',
                                            message='Please select a \"{}\" audio file!'.format('" or "'.join(AVAILABLE_FORMATS)),  # nopep8
                                            detail=f'File: {path}')
                    return
            self.filePaths = paths
            # Change the entry text
            self.filePaths_musicFile_Entry.configure(state=tk.NORMAL)
            self.filePaths_musicFile_Entry.delete(0, tk.END)
            self.filePaths_musicFile_Entry.insert(0, self.filePaths)
            self.filePaths_musicFile_Entry.configure(state=tk.DISABLED)

    def open_export_filedialog(self):
        """Make user select a folder to export the converted files in"""
        path = tk.filedialog.askdirectory(
            parent=self,
            title=f'Select Folder',
            initialdir='/',)
        if path:  # Path selected
            self.exportPath_var.set(path)

    def open_newModel_filedialog(self):
        """Make user select a ".pth" model to use for the vocal removing"""
        path = tk.filedialog.askopenfilename(
            parent=self,
            title=f'Select Model File',
            initialdir='/',
            initialfile='',
            filetypes=[
                ('pth', '*.pth'),
            ])

        if path:  # Path selected
            if path.lower().endswith(('.pth')):
                self.add_available_model(abs_path=path)
            else:
                tk.messagebox.showerror(master=self,
                                        title='Invalid File',
                                        message=f'Please select a PyTorch model file ".pth"!',
                                        detail=f'File: {path}')
                return

    def start_conversion(self):
        """
        Start the conversion for all the given mp3 and wav files
        """
        # -Get all variables-
        input_paths = self.filePaths
        export_path = self.exportPath_var.get()
        model_path = self.label_to_path[self.model_var.get()]
        try:
            sr = self.srValue_var.get()
            hop_length = self.hopValue_var.get()
            window_size = self.winSize_var.get()
            loops_num = self.stackLoopsNum_var.get()
        except tk.TclError:  # Non integer was put in entry box
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Input',
                                      message='Please make sure you only input integer numbers!')
            return
        except SyntaxError:  # Non integer was put in entry box
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Music File',
                                      message='You have selected an invalid music file!\nPlease make sure that your files still exist and end with either ".mp3", ".mp4", ".m4a", ".flac", ".wav"')
            return

        # -Check for invalid inputs-
        if not any([(os.path.isfile(path) and path.endswith(('.mp3', '.mp4', '.m4a', '.flac', '.wav')))
                    for path in input_paths]):
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Music File',
                                      message='You have selected an invalid music file!\nPlease make sure that your files still exist and end with either ".mp3", ".mp4", ".m4a", ".flac", ".wav"')
            return
        if not os.path.isdir(export_path):
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Export Directory',
                                      message='You have selected an invalid export directory!\nPlease make sure that your directory still exists!')
            return
        if not os.path.isfile(model_path):
            tk.messagebox.showwarning(master=self,
                                      title='Invalid Model File',
                                      message='You have selected an invalid model file!\nPlease make sure that your model file still exists!')
            return

        # -Save Data-
        save_data(data={
            'exportPath': export_path,
            'gpuConversion': self.gpuConversion_var.get(),
            'postprocessing': self.postprocessing_var.get(),
            'mask': self.mask_var.get(),
            'stackLoops': self.stackLoops_var.get(),
            'gpuConversion': self.gpuConversion_var.get(),
            'srValue': sr,
            'hopValue': hop_length,
            'winSize': window_size,
            'stackLoopsNum': loops_num,
        })

        # -Run the algorithm-
        threading.Thread(target=inference.main,
                         kwargs={
                             'input_paths': input_paths,
                             'gpu': 0 if self.gpuConversion_var.get() else -1,
                             'postprocess': self.postprocessing_var.get(),
                             'out_mask': self.mask_var.get(),
                             'model': model_path,
                             'sr': sr,
                             'hop_length': hop_length,
                             'window_size': window_size,
                             'export_path': export_path,
                             'loops': loops_num,
                             # Other Variables (Tkinter)
                             'window': self,
                             'command_widget': self.command_Text,
                             'button_widget': self.conversion_Button,
                             'progress_var': self.progress_var,
                         },
                         daemon=True
                         ).start()

    # Models
    def update_available_models(self):
        """
        Loop through every model (.pth) in the models directory
        and add to the select your model list
        """
        # Delete all previous options
        self.model_var.set('')
        self.options_model_Optionmenu['menu'].delete(0, 'end')

        for file_name in os.listdir(models_dir):
            if file_name.endswith('.pth'):
                # Add Radiobutton to the Options Menu
                self.options_model_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                      command=tk._setit(self.model_var, file_name))
                # Link the files name to its absolute path
                self.label_to_path[file_name] = os.path.join(models_dir, file_name)  # nopep8

    def add_available_model(self, abs_path: str):
        """
        Add the given absolute path of the file (.pth) to the available options
        and set the currently selected model to this one
        """
        if abs_path.endswith('.pth'):
            file_name = f'[CUSTOM] {os.path.basename(abs_path)}'
            # Add Radiobutton to the Options Menu
            self.options_model_Optionmenu['menu'].add_radiobutton(label=file_name,
                                                                  command=tk._setit(self.model_var, file_name))
            # Set selected model to the newly added one
            self.model_var.set(file_name)
            # Link the files name to its absolute path
            self.label_to_path[file_name] = abs_path  # nopep8
        else:
            tk.messagebox.showerror(master=self,
                                    title='Invalid File',
                                    message='Please select a model file with the ".pth" ending!',
                                    detail=f'File: {abs_path}')

    def update_stack_state(self):
        """
        Vary the stack Entry fro disabled/enabled based on the
        stackLoops variable, which is connected to the checkbutton
        """
        if self.stackLoops_var.get():
            self.options_stack_Entry.configure(state=tk.NORMAL)
        else:
            self.options_stack_Entry.configure(state=tk.DISABLED)
            self.stackLoopsNum_var.set(1)


if __name__ == "__main__":
    root = MainWindow()

    root.mainloop()

import os
import platform
from screeninfo import get_monitors
from PIL import Image
from PIL import ImageTk

OPERATING_SYSTEM = platform.system()

def get_screen_height():
    monitors = get_monitors()
    if len(monitors) == 0:
        raise Exception("Failed to get screen height")
    return monitors[0].height, monitors[0].width

def scale_values(value):
    if not SCALE_WIN_SIZE == 1920:
        ratio = SCALE_WIN_SIZE/1920  # Approx. 1.3333 for 2K
        return value * ratio
    else:
        return value

SCREEN_HIGHT, SCREEN_WIDTH = get_screen_height()
SCALE_WIN_SIZE = 1920

SCREEN_SIZE_VALUES = {
        "normal": {
            "credits_img":(100, 100),
            ## App Size
            'IMAGE_HEIGHT': 140, 
            'FILEPATHS_HEIGHT': 75, 
            'OPTIONS_HEIGHT': 262, 
            'CONVERSIONBUTTON_HEIGHT': 30,
            'COMMAND_HEIGHT': 141, 
            'PROGRESS_HEIGHT': 25, 
            'PADDING': 7, 
            'WIDTH': 680
        },
        "small": {
            "credits_img":(50, 50),
            'IMAGE_HEIGHT': 140, 
            'FILEPATHS_HEIGHT': 75, 
            'OPTIONS_HEIGHT': 262, 
            'CONVERSIONBUTTON_HEIGHT': 30, 
            'COMMAND_HEIGHT': 80, 
            'PROGRESS_HEIGHT': 25, 
            'PADDING': 5, 
            'WIDTH': 680
        },
        "medium": {
            "credits_img":(50, 50),
            ## App Size
            'IMAGE_HEIGHT': 140, 
            'FILEPATHS_HEIGHT': 75, 
            'OPTIONS_HEIGHT': 262, 
            'CONVERSIONBUTTON_HEIGHT': 30, 
            'COMMAND_HEIGHT': 115, 
            'PROGRESS_HEIGHT': 25, 
            'PADDING': 7, 
            'WIDTH': 680
        },
}

try:
    if SCREEN_HIGHT >= 900:
        determined_size = SCREEN_SIZE_VALUES["normal"]
    elif SCREEN_HIGHT <= 720:
        determined_size = SCREEN_SIZE_VALUES["small"]
    else:
        determined_size = SCREEN_SIZE_VALUES["medium"]
except:
        determined_size = SCREEN_SIZE_VALUES["normal"]

image_scale_1, image_scale_2 = 20, 30

class ImagePath():
    def __init__(self, base_path):
        img_path = os.path.join(base_path, 'gui_data', 'img')
        credits_path = os.path.join(img_path, 'credits.png')
        donate_path = os.path.join(img_path, 'donate.png')
        download_path = os.path.join(img_path, 'download.png')
        efile_path = os.path.join(img_path, 'File.png')
        help_path = os.path.join(img_path, 'help.png')
        key_path = os.path.join(img_path, 'key.png')
        stop_path = os.path.join(img_path, 'stop.png')
        play_path = os.path.join(img_path, 'play.png')
        pause_path = os.path.join(img_path, 'pause.png')
        up_img_path = os.path.join(img_path, "up.png")
        down_img_path = os.path.join(img_path, "down.png")
        left_img_path = os.path.join(img_path, "left.png")
        right_img_path = os.path.join(img_path, "right.png")
        clear_img_path = os.path.join(img_path, "clear.png")
        copy_img_path = os.path.join(img_path, "copy.png")
        self.banner_path = os.path.join(img_path, 'UVR-banner.png')

        self.efile_img = self.open_image(path=efile_path,size=(image_scale_1, image_scale_1))
        self.stop_img = self.open_image(path=stop_path, size=(image_scale_1, image_scale_1))
        self.play_img = self.open_image(path=play_path, size=(image_scale_1, image_scale_1))
        self.pause_img = self.open_image(path=pause_path, size=(image_scale_1, image_scale_1))
        self.help_img = self.open_image(path=help_path, size=(image_scale_1, image_scale_1))
        self.download_img = self.open_image(path=download_path, size=(image_scale_2, image_scale_2))       
        self.donate_img = self.open_image(path=donate_path, size=(image_scale_2, image_scale_2))    
        self.key_img = self.open_image(path=key_path, size=(image_scale_2, image_scale_2))     
        self.up_img = self.open_image(path=up_img_path, size=(image_scale_2, image_scale_2))
        self.down_img = self.open_image(path=down_img_path, size=(image_scale_2, image_scale_2))       
        self.left_img = self.open_image(path=left_img_path, size=(image_scale_2, image_scale_2))    
        self.right_img = self.open_image(path=right_img_path, size=(image_scale_2, image_scale_2))   
        self.clear_img = self.open_image(path=clear_img_path, size=(image_scale_2, image_scale_2))
        self.copy_img = self.open_image(path=copy_img_path, size=(image_scale_2, image_scale_2))
        self.credits_img = self.open_image(path=credits_path, size=determined_size["credits_img"])

    def open_image(self, path: str, size: tuple = None, keep_aspect: bool = True, rotate: int = 0) -> ImageTk.PhotoImage:
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

#All Sizes Below Calibrated to 1080p!

if OPERATING_SYSTEM=="Darwin":
   FONT_SIZE_F1 = 13
   FONT_SIZE_F2 = 11
   FONT_SIZE_F3 = 12
   FONT_SIZE_0 = 9
   FONT_SIZE_1 = 11
   FONT_SIZE_2 = 12
   FONT_SIZE_3 = 13
   FONT_SIZE_4 = 14
   FONT_SIZE_5 = 15
   FONT_SIZE_6 = 17
   HELP_HINT_CHECKBOX_WIDTH = 13
   MDX_CHECKBOXS_WIDTH = 14
   VR_CHECKBOXS_WIDTH = 14
   ENSEMBLE_CHECKBOXS_WIDTH = 18
   DEMUCS_CHECKBOXS_WIDTH = 14
   DEMUCS_PRE_CHECKBOXS_WIDTH = 20
   GEN_SETTINGS_WIDTH = 17
   MENU_COMBOBOX_WIDTH = 16
   MENU_OPTION_WIDTH = 12
   READ_ONLY_COMBO_WIDTH = 35
   SETTINGS_BUT_WIDTH = 19
   VR_BUT_WIDTH = 16
   SET_MENUS_CHECK_WIDTH = 12
   COMBO_WIDTH = 14
   SET_VOC_SPLIT_CHECK_WIDTH = 21
elif OPERATING_SYSTEM=="Linux":
   HELP_HINT_CHECKBOX_WIDTH = 15
   MDX_CHECKBOXS_WIDTH = 16
   VR_CHECKBOXS_WIDTH = 16
   ENSEMBLE_CHECKBOXS_WIDTH = 20
   DEMUCS_CHECKBOXS_WIDTH = 16
   DEMUCS_PRE_CHECKBOXS_WIDTH = 24
   GEN_SETTINGS_WIDTH = 20
   MENU_COMBOBOX_WIDTH = 18
   MENU_OPTION_WIDTH = 12
   READ_ONLY_COMBO_WIDTH = 40
   SETTINGS_BUT_WIDTH = 23
   VR_BUT_WIDTH = 18
   SET_MENUS_CHECK_WIDTH = 13
   COMBO_WIDTH = 16
   SET_VOC_SPLIT_CHECK_WIDTH = 25                      
   FONT_SIZE_F1 = 10
   FONT_SIZE_F2 = 8
   FONT_SIZE_F3 = 9
   FONT_SIZE_0 = 7
   FONT_SIZE_1 = 8
   FONT_SIZE_2 = 9
   FONT_SIZE_3 = 10
   FONT_SIZE_4 = 11
   FONT_SIZE_5 = 13
   FONT_SIZE_6 = 15
elif OPERATING_SYSTEM=="Windows":
   HELP_HINT_CHECKBOX_WIDTH = 15
   MDX_CHECKBOXS_WIDTH = 14
   VR_CHECKBOXS_WIDTH = 14
   ENSEMBLE_CHECKBOXS_WIDTH = 20
   DEMUCS_CHECKBOXS_WIDTH = 14
   DEMUCS_PRE_CHECKBOXS_WIDTH = 20
   GEN_SETTINGS_WIDTH = 18
   MENU_COMBOBOX_WIDTH = 16
   MENU_OPTION_WIDTH = 12
   READ_ONLY_COMBO_WIDTH = 35
   SETTINGS_BUT_WIDTH = 20
   VR_BUT_WIDTH = 16
   SET_MENUS_CHECK_WIDTH = 13
   COMBO_WIDTH = 14
   SET_VOC_SPLIT_CHECK_WIDTH = 23                      
   FONT_SIZE_F1 = 10
   FONT_SIZE_F2 = 8
   FONT_SIZE_F3 = 9
   FONT_SIZE_0 = 7
   FONT_SIZE_1 = 8
   FONT_SIZE_2 = 9
   FONT_SIZE_3 = 10
   FONT_SIZE_4 = 11
   FONT_SIZE_5 = 13
   FONT_SIZE_6 = 15

#Main Size Values:
IMAGE_HEIGHT = determined_size["IMAGE_HEIGHT"]
FILEPATHS_HEIGHT = determined_size["FILEPATHS_HEIGHT"]
OPTIONS_HEIGHT = determined_size["OPTIONS_HEIGHT"]
CONVERSIONBUTTON_HEIGHT = determined_size["CONVERSIONBUTTON_HEIGHT"]
COMMAND_HEIGHT = determined_size["COMMAND_HEIGHT"]
PROGRESS_HEIGHT = determined_size["PROGRESS_HEIGHT"]
PADDING = determined_size["PADDING"]
WIDTH = determined_size["WIDTH"]

# IMAGE_HEIGHT = 140
# FILEPATHS_HEIGHT = 75
# OPTIONS_HEIGHT = 262
# CONVERSIONBUTTON_HEIGHT = 30
# COMMAND_HEIGHT = 141
# PROGRESS_HEIGHT = 25
# PADDING = 7
# WIDTH = 680

MENU_PADDING_1 = 3
MENU_PADDING_2 = 10
MENU_PADDING_3 = 15
MENU_PADDING_4 = 3

#Main Frame Sizes 
X_CONVERSION_BUTTON_1080P = 50
WIDTH_CONVERSION_BUTTON_1080P = -100
HEIGHT_GENERIC_BUTTON_1080P = 35
X_STOP_BUTTON_1080P = -10 - 35
X_SETTINGS_BUTTON_1080P = -670
X_PROGRESSBAR_1080P = 25
WIDTH_PROGRESSBAR_1080P = -50
X_CONSOLE_FRAME_1080P = 15
WIDTH_CONSOLE_FRAME_1080P = -30
HO_S = 7

#File Frame Sizes
FILEPATHS_FRAME_X = 10
FILEPATHS_FRAME_Y = 155
FILEPATHS_FRAME_WIDTH = -20
MUSICFILE_BUTTON_X = 0
MUSICFILE_BUTTON_Y = 5
MUSICFILE_BUTTON_WIDTH = 0
MUSICFILE_BUTTON_HEIGHT = -5
MUSICFILE_ENTRY_X = 7.5
MUSICFILE_ENTRY_WIDTH = -50
MUSICFILE_ENTRY_HEIGHT = -5
MUSICFILE_OPEN_X = -45
MUSICFILE_OPEN_Y = 160
MUSICFILE_OPEN_WIDTH = 35
MUSICFILE_OPEN_HEIGHT = 33
SAVETO_BUTTON_X = 0
SAVETO_BUTTON_Y = 5
SAVETO_BUTTON_WIDTH = 0
SAVETO_BUTTON_HEIGHT = -5
SAVETO_ENTRY_X = 7.5
OPEN_BUTTON_X = 427.1
OPEN_BUTTON_WIDTH = -427.4
SAVETO_ENTRY_WIDTH = -50
SAVETO_ENTRY_HEIGHT = -5
SAVETO_OPEN_X = -45
SAVETO_OPEN_Y = 197.5
SAVETO_OPEN_WIDTH = 35
SAVETO_OPEN_HEIGHT = 32

#Main Option menu
OPTIONS_FRAME_X = 10
OPTIONS_FRAME_Y = 250
OPTIONS_FRAME_WIDTH = -20
FILEONE_LABEL_X = -28
FILEONE_LABEL_WIDTH = -38
FILETWO_LABEL_X = -32
FILETWO_LABEL_WIDTH = -20
TIME_WINDOW_LABEL_X = -43
TIME_WINDOW_LABEL_WIDTH = 0
INTRO_ANALYSIS_LABEL_X = -83
INTRO_ANALYSIS_LABEL_WIDTH = -50
INTRO_ANALYSIS_OPTION_X = -68
DB_ANALYSIS_LABEL_X = 62
DB_ANALYSIS_LABEL_WIDTH = -34
DB_ANALYSIS_OPTION_X = 86
WAV_TYPE_SET_LABEL_X = -43
WAV_TYPE_SET_LABEL_WIDTH = 0
ENTRY_WIDTH = 222

# Constants for the ensemble_listbox_Frame
ENSEMBLE_LISTBOX_FRAME_X = -25
ENSEMBLE_LISTBOX_FRAME_Y = -20
ENSEMBLE_LISTBOX_FRAME_WIDTH = 0
ENSEMBLE_LISTBOX_FRAME_HEIGHT = 67

# Constants for the ensemble_listbox_scroll
ENSEMBLE_LISTBOX_SCROLL_X = 195
ENSEMBLE_LISTBOX_SCROLL_Y = -20
ENSEMBLE_LISTBOX_SCROLL_WIDTH = -48
ENSEMBLE_LISTBOX_SCROLL_HEIGHT = 69

# Constants for Radio Buttons
RADIOBUTTON_X_WAV = 457
RADIOBUTTON_X_FLAC = 300
RADIOBUTTON_X_MP3 = 143
RADIOBUTTON_Y = -5
RADIOBUTTON_WIDTH = 0
RADIOBUTTON_HEIGHT = 6
MAIN_ROW_Y_1 = -15
MAIN_ROW_Y_2 = -17
MAIN_ROW_X_1 = -4
MAIN_ROW_X_2 = 21
MAIN_ROW_2_Y_1 = -15
MAIN_ROW_2_Y_2 = -17
MAIN_ROW_2_X_1 = -28
MAIN_ROW_2_X_2 = 1
LOW_MENU_Y_1 = 18
LOW_MENU_Y_2 = 16
SUB_ENT_ROW_X = -2
MAIN_ROW_WIDTH = -53
MAIN_ROW_ALIGN_WIDTH = -86
CHECK_BOX_Y = 0
CHECK_BOX_X = 20
CHECK_BOX_WIDTH = -49
CHECK_BOX_HEIGHT = 2
LEFT_ROW_WIDTH = -10
LABEL_HEIGHT = -5
OPTION_HEIGHT = 8
LABEL_X_OFFSET = -28
LABEL_WIDTH = -38
ENTRY_WIDTH = 179.5
ENTRY_OPEN_BUTT_WIDTH = -185
ENTRY_OPEN_BUTT_X_OFF = 405
UPDATE_LABEL_WIDTH = 35 if OPERATING_SYSTEM == 'Linux' else 32

HEIGHT_CONSOLE_FRAME_1080P = COMMAND_HEIGHT + HO_S
LOW_MENU_Y = LOW_MENU_Y_1, LOW_MENU_Y_2
MAIN_ROW_Y = MAIN_ROW_Y_1, MAIN_ROW_Y_2
MAIN_ROW_X = MAIN_ROW_X_1, MAIN_ROW_X_2
MAIN_ROW_2_Y = MAIN_ROW_2_Y_1, MAIN_ROW_2_Y_2
MAIN_ROW_2_X = MAIN_ROW_2_X_1, MAIN_ROW_2_X_2

LABEL_Y = MAIN_ROW_Y[0]
ENTRY_Y = MAIN_ROW_Y[1]

BUTTON_Y_1080P = IMAGE_HEIGHT + FILEPATHS_HEIGHT + OPTIONS_HEIGHT - 8 + PADDING*2
HEIGHT_PROGRESSBAR_1080P = PROGRESS_HEIGHT
Y_OFFSET_PROGRESS_BAR_1080P = IMAGE_HEIGHT + FILEPATHS_HEIGHT + OPTIONS_HEIGHT + CONVERSIONBUTTON_HEIGHT + COMMAND_HEIGHT + PADDING*4
Y_OFFSET_CONSOLE_FRAME_1080P = IMAGE_HEIGHT + FILEPATHS_HEIGHT + OPTIONS_HEIGHT + CONVERSIONBUTTON_HEIGHT + PADDING + X_PROGRESSBAR_1080P

LABEL_Y_OFFSET = MAIN_ROW_Y[0]
ENTRY_X_OFFSET = SUB_ENT_ROW_X
ENTRY_Y_OFFSET = MAIN_ROW_Y[1]
OPTION_WIDTH = MAIN_ROW_ALIGN_WIDTH

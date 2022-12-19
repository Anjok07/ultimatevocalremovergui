import os
from screeninfo import get_monitors
from PIL import Image
from PIL import ImageTk

MAC = False

def get_screen_height():
    monitors = get_monitors()
    if len(monitors) == 0:
        raise Exception("Failed to get screen height")
    return monitors[0].height

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
        },
        "small": {
            "credits_img":(50, 50),
            ## App Size
            'IMAGE_HEIGHT': 135, 
            'FILEPATHS_HEIGHT': 85, 
            'OPTIONS_HEIGHT': 274, 
            'CONVERSIONBUTTON_HEIGHT': 35, 
            'COMMAND_HEIGHT': 80, 
            'PROGRESS_HEIGHT': 6, 
            'PADDING': 5, 
        },
        "medium": {
            "credits_img":(50, 50),
            ## App Size
            'IMAGE_HEIGHT': 135, 
            'FILEPATHS_HEIGHT': 85, 
            'OPTIONS_HEIGHT': 274, 
            'CONVERSIONBUTTON_HEIGHT': 20, 
            'COMMAND_HEIGHT': 115, 
            'PROGRESS_HEIGHT': 9, 
            'PADDING': 7, 
        },
        "mac": {
            "credits_img":(200, 200),
            ## App Size
            'IMAGE_HEIGHT': 135, 
            'FILEPATHS_HEIGHT': 75, 
            'OPTIONS_HEIGHT': 262, 
            'CONVERSIONBUTTON_HEIGHT': 30, 
            'COMMAND_HEIGHT': 141, 
            'PROGRESS_HEIGHT': 25, 
            'PADDING': 5, 
        },
}

if MAC:
    determined_size = SCREEN_SIZE_VALUES["mac"]
    normal_screen = True
else:
    try:
        if get_screen_height() >= 900:
            determined_size = SCREEN_SIZE_VALUES["normal"]
            normal_screen = True
        elif get_screen_height() <= 720:
            determined_size = SCREEN_SIZE_VALUES["small"]
            normal_screen = False
        else:
            determined_size = SCREEN_SIZE_VALUES["medium"]
            normal_screen = False
    except:
            determined_size = SCREEN_SIZE_VALUES["normal"]
            normal_screen = False

class ImagePath():
    def __init__(self, base_path):
        
        img_path = os.path.join(base_path, 'gui_data', 'img')
        credits_path = os.path.join(img_path, 'credits.png')
        donate_path = os.path.join(img_path, 'donate.png')
        download_path = os.path.join(img_path, 'download.png')
        efile_path = os.path.join(img_path, 'file.png')
        help_path = os.path.join(img_path, 'help.png')
        key_path = os.path.join(img_path, 'key.png')
        stop_path = os.path.join(img_path, 'stop.png')
        play_path = os.path.join(img_path, 'play.png')
        pause_path = os.path.join(img_path, 'pause.png')
        self.banner_path = os.path.join(img_path, 'UVR-banner.png')

        self.efile_img = self.open_image(path=efile_path,size=(20, 20))
        self.stop_img = self.open_image(path=stop_path, size=(20, 20))
        self.play_img = self.open_image(path=play_path, size=(20, 20))
        self.pause_img = self.open_image(path=pause_path, size=(20, 20))
        self.help_img = self.open_image(path=help_path, size=(20, 20))
        self.download_img = self.open_image(path=download_path, size=(30, 30))       
        self.donate_img = self.open_image(path=donate_path, size=(30, 30))    
        self.key_img = self.open_image(path=key_path, size=(30, 30))     
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

class AdjustedValues():
    IMAGE_HEIGHT = determined_size["IMAGE_HEIGHT"]
    FILEPATHS_HEIGHT = determined_size["FILEPATHS_HEIGHT"]
    OPTIONS_HEIGHT = determined_size["OPTIONS_HEIGHT"]
    CONVERSIONBUTTON_HEIGHT = determined_size["CONVERSIONBUTTON_HEIGHT"]
    COMMAND_HEIGHT = determined_size["COMMAND_HEIGHT"]
    PROGRESS_HEIGHT = determined_size["PROGRESS_HEIGHT"]
    PADDING = determined_size["PADDING"]
    normal_screen = normal_screen
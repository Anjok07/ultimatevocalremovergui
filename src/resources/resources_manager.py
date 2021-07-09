"""
Manager of folders and images

Also catches
"""
import os
import sys
from pathlib import Path
# Logging
import datetime as dt
import logging
from logging.handlers import RotatingFileHandler
# Get the absolute path to this file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    main_path = os.path.dirname(sys.executable)
    abs_path = os.path.join(main_path, 'resources')
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))


IMAGE_FOLDER = 'images'
TRANSLATIONS_FOLDER = 'translations'
USER_FOLDER = 'user'
LOGS_FOLDER = 'logs'
MODELS_FOLDER = 'models'
THEMES_FOLDER = 'themes'
INSTRUMENTAL_FOLDER_NAME = 'Main Models'
VOCAL_FOLDER_NAME = 'Vocal Models'
TEMP_MUSIC_FILES_FOLDER = 'tmp'


class ResourcePaths:
    """
    Get access to all resources used in the application
    through this class
    """
    class images:
        refresh = os.path.join(abs_path, IMAGE_FOLDER, 'refresh.png')
        showcase = os.path.join(abs_path, IMAGE_FOLDER, 'showcase.png')
        icon = os.path.join(abs_path, IMAGE_FOLDER, 'icon.ico')
        banner = os.path.join(abs_path, IMAGE_FOLDER, 'banner.png')
        settings = os.path.join(abs_path, IMAGE_FOLDER, 'settings.png')
        folder = os.path.join(abs_path, IMAGE_FOLDER, 'folder.png')
        audio_play = os.path.join(abs_path, IMAGE_FOLDER, 'audio_play.png')
        audio_pause = os.path.join(abs_path, IMAGE_FOLDER, 'audio_pause.png')
        save = os.path.join(abs_path, IMAGE_FOLDER, 'save.png')
        menu = os.path.join(abs_path, IMAGE_FOLDER, 'menu.png')
        playpause_gif = os.path.join(abs_path, IMAGE_FOLDER, 'playpause.gif')
        upload = os.path.join(abs_path, IMAGE_FOLDER, 'upload.png')
        download = os.path.join(abs_path, IMAGE_FOLDER, 'download.png')

    class themes:
        dark_path = os.path.join(abs_path, THEMES_FOLDER, 'dark.qss')
        light_path = os.path.join(abs_path, THEMES_FOLDER, 'light.qss')

        with open(dark_path, 'r') as f:
            dark = f.read()
        with open(light_path, 'r') as f:
            light = f.read()

    localizationDir = os.path.join(abs_path, TRANSLATIONS_FOLDER)
    instrumentalModelsDir = os.path.join(abs_path, MODELS_FOLDER, INSTRUMENTAL_FOLDER_NAME)
    vocalModelsDir = os.path.join(abs_path, MODELS_FOLDER, VOCAL_FOLDER_NAME)
    # -User Folder-
    logsDir = os.path.join(abs_path, USER_FOLDER, LOGS_FOLDER)
    modelsDir = os.path.join(abs_path, USER_FOLDER, MODELS_FOLDER)
    customInstrumentalModelsDir = os.path.join(abs_path, USER_FOLDER, MODELS_FOLDER, INSTRUMENTAL_FOLDER_NAME)
    customVocalModelsDir = os.path.join(abs_path, USER_FOLDER, MODELS_FOLDER, VOCAL_FOLDER_NAME)
    tempDir = os.path.join(abs_path, USER_FOLDER, TEMP_MUSIC_FILES_FOLDER)
    settingsIniFile = os.path.join(abs_path, USER_FOLDER, 'settings.ini')


class Logger(logging.Logger):
    """
    This code is a mess but it works
    """
    MAX_BACKUP_FILES = 3

    def __init__(self):
        super().__init__('mylogger')
        # Create directory if not available
        if not os.path.exists(ResourcePaths.logsDir):
            os.makedirs(ResourcePaths.logsDir)

        # Base file path
        self.filePath = os.path.join(ResourcePaths.logsDir, 'logger.log')
        self.currentIndent = 0
        # -Create Log Formatter-
        self.format_str = "%(levelname)s | %(message)s"
        self.default_formatter = logging.Formatter(self.format_str)
        # -Create File Handler-
        self.logging_fileHandler = RotatingFileHandler(filename=self.filePath, mode='w',
                                                       backupCount=self.MAX_BACKUP_FILES,
                                                       delay=True)
        self.logging_fileHandler.setFormatter(self.default_formatter)
        self.logging_fileHandler.namer = self.file_namer
        if os.path.isfile(self.filePath):
            # Log already exists, rollover
            self.logging_fileHandler.doRollover()

        # -Set the handler to basicConfig and instance of logger-
        # to prevent issues with renaming the file internally
        logging.basicConfig(handlers=[self.logging_fileHandler])
        self.logger = logging.getLogger()
        self.setLevel(logging.DEBUG)
        self.addHandler(self.logging_fileHandler)

        # -Other-
        # Ensure MAX_BACKUP_FILES stays true
        self.cleanup_logDir()
        # First message
        self.logging_fileHandler.setFormatter(logging.Formatter("%(message)s"))
        self.info(f'{dt.datetime.now().strftime("--- %d-%m-%Y [%H:%M]")} Initialized Logger ---')
        self.update_formatter()

    def indent_forward(self):
        self.currentIndent += 1
        self.update_formatter()

    def indent_backwards(self):
        self.currentIndent -= 1
        # Never let indent go below 0
        self.currentIndent = max(0, self.currentIndent)
        self.update_formatter()

    def end_indent(self):
        self.currentIndent = 0
        self.update_formatter()

    def update_formatter(self):
        tabs = '\t' * self.currentIndent
        fmt = tabs + self.format_str
        self.currentFormatter = logging.Formatter(fmt)
        self.logging_fileHandler.setFormatter(self.currentFormatter)

    def cleanup_logDir(self):
        """
        Cleanups logs directory if there are more files in
        it than MAX_BACKUP_FILES allows
        """
        # Sorted by creation date newest to oldest
        filePaths = sorted(Path(ResourcePaths.logsDir).iterdir(), key=os.path.getmtime, reverse=True)

        for path in filePaths[self.MAX_BACKUP_FILES:]:
            os.remove(path)

    def file_namer(self, filename: str) -> str:
        """
        Rename the log files correctly

        (e.x):
            file_namer("logger.log.1")
            1. Strip down to "logger"
            2. Add date and ".log" extension
            returns "logger-00.00.00-10.02.2021.log"
        """
        # Original filename
        # Remove the .log. from
        basename = os.path.basename(self.filePath)
        basename = basename[:len(os.path.splitext(basename)[0])]
        filename = os.path.join(ResourcePaths.logsDir, basename)
        date = dt.datetime.now().strftime('-%H.%M.%S-%d-%m-%Y')
        filename = filename + '-BACKUP' + date + '.log'
        return filename

    def info(self, msg, *args, indent_forwards: bool = False, **kwargs):
        super().info(msg, *args, **kwargs)
        if indent_forwards:
            self.indent_forward()

    def warning(self, msg, *args, indent_forwards: bool = False, **kwargs):
        super().warning(msg, *args, **kwargs)
        if indent_forwards:
            self.indent_forward()

    def debug(self, msg, *args, indent_forwards: bool = False, **kwargs):
        super().debug(msg, *args, **kwargs)
        if indent_forwards:
            self.indent_forward()

    def error(self, msg, *args, indent_forwards: bool = False, **kwargs):
        super().error(msg, *args, **kwargs)
        if indent_forwards:
            self.indent_forward()

    def exception(self, msg, *args, indent_forwards: bool = False, **kwargs):
        super().exception(msg, *args, **kwargs)
        if indent_forwards:
            self.indent_forward()


logging.getLogger().__class__ = Logger

if __name__ == "__main__":
    """Print all resources"""

    print('-- Images --')
    for img, img_path in vars(ResourcePaths.images).items():
        if os.path.isfile(str(img_path)):
            print(f'{img} -> {img_path}')

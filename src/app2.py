"""
Main Application
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide6.QtCore import (Qt, QThreadPool, QSize, QDir,)
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QWidget, QPushButton, QFileDialog, QWidget)
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
from PySide6.QtUiTools import QUiLoader
# -Root imports-
from .resources.resources_manager import ResourcePaths
from .data.data_manager import DataManager
# -Other-
import os
import sys
# Code annotation
from typing import (Dict)

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    base_path = sys._MEIPASS  # pylint: disable=no-member
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(base_path)  # Change the current working directory to the base path

settingsManager = DataManager(default_data={
    'seperation_data':
        {'exportPath': '',
         'inputPaths': [],
         'gpu': False,
         'postprocess': False,
         'tta': False,
         'output_image': False,
         'sr': 44100,
         'hop_length': 1024,
         'window_size': 320,
         'n_fft': 2048,
         'stack': False,
         'stackPasses': 1,
         'stackOnly': False,
         'saveAllStacked': False,
         'modelFolder': False,
         'modelInstrumentalLabel': '',
         'modelStackedLabel': '',
         'aiModel': 'v4',
         'resType': 'Kaiser Fast',
         'manType': False,

         'useModel': 'instrumental',
         'lastDir': None,
         },
    'window_geometries': {}})
app: QApplication


def load_windows() -> dict:
    """
    Load all windows of this application and return
    a dictionary with their instances
    """
    global loader
    loader = QUiLoader()
    window_paths = {'main': ResourcePaths.ui_files.mainwindow,
                    }
    windows: Dict[str, QWidget] = {}

    for win_name, path in window_paths.items():
        window = loader.load(path, None)
        if win_name != 'main':
            # Window is not main
            window.setWindowFlag(Qt.WindowStaysOnTopHint)
        windows[win_name] = window

    return windows


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()


class WindowManager:
    def __init__(self, windows: Dict[str, QWidget]):
        self.windows = windows
        # -Variables-
        self.threadpool = QThreadPool()
        # -Setup-
        icon = QtGui.QPixmap(ResourcePaths.images.settings)
        self.windows['main'].pushButton_settings.setIcon(icon)
        self.windows['main'].pushButton_settings.setIconSize(QSize(30, 30))

        # -Other-
        # Show window
        self.windows['main'].show()
        self.windows['main'].resizeEvent = lambda: print('F')
        # Focus window
        self.windows['main'].activateWindow()
        self.windows['main'].raise_()


def run():
    """Start the application\n
    Run 'sys.exit(app.exec_())' after this method has been called
    """
    global app
    global winManager
    # Suppress QT Warnings
    os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    app = QApplication
    # ...Application settings here...
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = app(sys.argv)
    windows = load_windows()
    # Create Manager
    winManager = WindowManager(windows)

"""
Main Application
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
from PySide6.QtGui import Qt
# -Root imports-
from .resources.resources_manager import ResourcePaths
from .data.data_manager import DataManager
from .windows import (mainwindow, settingswindow)
# -Other-
import os
import sys
# Code annotation
from typing import (Dict, Optional)

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

settingsManager = DataManager(default_data={'seperation_data': {'exportPath': '',
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
                                            'window_geometries': {},
                                            })


class CustomApplication(QtWidgets.QApplication):
    def __init__(self):
        # -Init Application-
        # Suppress QT Warnings
        os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
        os.environ["QT_SCALE_FACTOR"] = "1"
        # ...Application settings here...
        self.setAttribute(Qt.AA_UseHighDpiPixmaps)
        super(CustomApplication, self).__init__(sys.argv)

        # -Create Managers-
        self.settings = QtCore.QSettings('UVR', 'Ultimate Vocal Remover')
        self.settings.remove("")
        self.resources = ResourcePaths()
        self.languageManager = LanguageManager(self)
        self.threadpool = QtCore.QThreadPool(self)
        # -Load Windows-
        self.windows = {
            # Collection of windows
            'main': MainWindow(self),
        }

        self.windows['main'].show()


class LanguageManager:
    def __init__(self, app: CustomApplication):
        self.app = app
        self.translator = QtCore.QTranslator(self.app)
        # Load language -> if not specified, try loading system language
        self.load_language(self.app.settings.value('language',
                                                   'en'))
        # defaultValue=QtCore.QLocale.system().name()))

    def load_language(self, language: Optional[str] = None):
        """
        Load specified language by file name

        Default is english
        """
        if (not language or
                'en' in language.lower()):
            # Language was not specified
            self.app.removeTranslator(self.translator)
        else:
            # Language was specified
            translation_path = os.path.join(self.app.resources.localizationDir, f'{language}.qm')
            if not os.path.isfile(translation_path):
                # Translation does not exist
                # Load default language (english)
                self.load_language()

            self.translator.load(translation_path)
            self.app.installTranslator(self.translator)

        if hasattr(self.app, 'windows'):
            # Application already initialized windows
            for window in self.app.windows.values():
                window.update_translation()


class MainWindow(QtWidgets.QWidget):
    def __init__(self, app: CustomApplication):
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.app = app

        self.load_settings()

    def load_settings(self):
        """
        Load the saved settings for this window
        """
        # -Default Settings-
        # Window is centered on primary window
        default_geometry = self.geometry()
        point = QtCore.QPoint()
        point.setX(self.app.primaryScreen().size().width() / 2)
        point.setY(self.app.primaryScreen().size().height() / 2)
        default_geometry.moveCenter(point)

        # -Load Settings-
        self.app.settings.beginGroup('mainwindow')
        geometry = self.app.settings.value('geometry',
                                           default_geometry)
        self.app.settings.endGroup()

        # -Apply Settings-
        self.setGeometry(geometry)

    def save_settings(self):
        """
        Save the settings for this window
        """
        # -Save Settings-
        self.app.settings.beginGroup('mainwindow')
        self.app.settings.setValue('geometry',
                                   self.geometry())
        self.app.settings.endGroup()
        # Commit Save
        self.app.settings.sync()

    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        self.save_settings()

        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.ui.retranslateUi(self)


def run():
    """Start the application\n
    Run 'sys.exit(app.exec_())' after this method has been called
    """
    app = CustomApplication()

    sys.exit(app.exec_())
    # Create Manager

    # winManager = WindowManager(windows)

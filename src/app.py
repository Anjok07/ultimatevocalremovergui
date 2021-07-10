"""
Main Application
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
# -Root imports-
from .resources.resources_manager import (ResourcePaths, Logger)
from .inference import converter
from .translator import Translator
from . import constants as const
# -Other-
# Logging
import logging
from collections import OrderedDict
# System
import os
import sys
# Code annotation
from typing import (Dict, List,)
# Debugging
import pprint
# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_path)  # Change the current working directory to the base path


class CustomApplication(QtWidgets.QApplication):
    """Application of the GUI

    The class contains instances of all windows and performs
    general tasks like setting up the windows, saving data,
    or improving the functionality of widgets, across all windows.
    """

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
        self.logger = Logger()
        self.settings = QtCore.QSettings(ResourcePaths.settingsIniFile, QtCore.QSettings.Format.IniFormat)
        # self.settings.clear()
        self.resources = ResourcePaths()
        self.translator = Translator(self)
        self.themeManager = ThemeManager(self)
        self.threadpool = QtCore.QThreadPool(self)
        # -Load Windows-
        # Workaround for circular dependency
        from .windows import (mainwindow, settingswindow, presetseditorwindow, infowindow)
        # Collection of windows
        self.windows: Dict[str, QtWidgets.QWidget] = {
            'main': mainwindow.MainWindow(self),
            'settings': settingswindow.SettingsWindow(self),
            'presetsEditor': presetseditorwindow.PresetsEditorWindow(self),
            'info': infowindow.InfoWindow(self),
        }
        self.mainWindow: mainwindow.MainWindow = self.windows['main']
        self.settingsWindow: settingswindow.SettingsWindow = self.windows['settings']
        self.presetsEditorWindow: presetseditorwindow.PresetsEditorWindow = self.windows['presetsEditor']
        self.infoWindow: infowindow.InfoWindow = self.windows['info']

        self.logger.info('--- Setting up application ---',
                         indent_forwards=True)
        self.setup_application()
        self.logger.indent_backwards()
        self.logger.info('--- Finished setup ---')

    def setup_application(self):
        """Set up the windows of this application

        - Execute setup methods of the windows
        - Improve widgets across all windows
        - Load some user data
        - Show the main window
        """
        def setup_windows():
            """
            Setup all windows in this application
            """
            for window in self.windows.values():
                window.setup_window()

        def improve_comboboxes():
            """
            Improvements:
                - Always show full contents of popup
                - Center align editable comboboxes
            """
            for window in self.windows.values():
                for combobox in window.findChildren(QtWidgets.QComboBox):
                    # Monkeypatch showPopup function
                    combobox.showPopup = lambda wig=combobox, func=combobox.showPopup: self.improved_combobox_showPopup(wig, func)  # nopep8
                    if combobox.isEditable():
                        # Align editable comboboxes to center
                        combobox.lineEdit().setAlignment(Qt.AlignCenter)

        def assign_lineEdit_validators():
            """
            Set the validators for the required lineEdits
            """
            validator = QtGui.QIntValidator(self)
            for widget in self.windows['settings'].ui.frame_constants.findChildren(QtWidgets.QLineEdit):
                widget.setValidator(validator)

        def bind_info_boxes():
            def show_info(title: str, text: str):
                """Show info to user with QMessageBox

                Args:
                    title (str): Title of Message
                    text (str): Content of Message
                """
                self.infoWindow.update_info(title, text)
                self.infoWindow.show()

            self.settingsWindow.ui.info_conversion.clicked.connect(lambda: show_info(self.tr("Conversion Info"),
                                                                                     self.translator.loaded_language.settings_conversion))

        # -Before-

        # -Setup-
        setup_windows()
        improve_comboboxes()
        assign_lineEdit_validators()
        bind_info_boxes()

        # -After-
        # Load language
        lang_code = self.settings.value('user/language',
                                        const.DEFAULT_SETTINGS['language'])
        self.translator.load_language(self.translator.LANGUAGES[lang_code])  # nopep8
        # Load Theme
        theme = self.settings.value('user/theme',
                                    const.DEFAULT_SETTINGS['theme'])
        self.themeManager.load_theme(theme)
        # Check for first startup
        if not self.settings.allKeys():
            self.first_startup()
        # with open(os.path.join(os.getcwd(), '..', 'startup', 'run.txt'), 'w') as f:
        #     f.write('1')

    def first_startup(self):
        """
        First time user started the application or he reset the app.
        """
        print("FIRST STARTUP")

    @staticmethod
    def improved_combobox_showPopup(widget: QtWidgets.QComboBox, showPopup: QtWidgets.QComboBox.showPopup):
        """Extend functionality for the QComboBox.showPopup function

        Improve the QComboBox by overriding the showPopup function to
        adjust the size of the view (list that opens on click)
        to the contents before showing

        Args:
            widget (QtWidgets.QComboBox): Widget to apply improvement on
            showPopup (QtWidgets.QComboBox.showPopup): showPopup function of the given widget
        """
        # Get variables
        view = widget.view()
        fm = widget.fontMetrics()
        widths = [fm.size(0, widget.itemText(i)).width()
                  for i in range(widget.count())]
        if widths:
            # Combobox has content
            # + 30 as a buffer for the scrollbar
            view.setMinimumWidth(max(widths) + 30)

        showPopup()

    def extract_seperation_data(self) -> dict:
        """Collects the settings required for seperation

        Returns:
            dict: Seperation settings
        """

        seperation_data = OrderedDict()

        # Input/Export
        # self.windows['settings'].inputPaths
        seperation_data['input_paths'] = self.windows['main'].inputPaths
        seperation_data['export_path'] = self.windows['settings'].exportDirectory
        # -Simple Variables (Easy to extract)-
        # Checkbox
        seperation_data['gpuConversion'] = self.windows['settings'].ui.checkBox_gpuConversion.isChecked()
        seperation_data['postProcess'] = self.windows['settings'].ui.checkBox_postProcess.isChecked()
        seperation_data['tta'] = self.windows['settings'].ui.checkBox_tta.isChecked()
        seperation_data['outputImage'] = self.windows['settings'].ui.checkBox_outputImage.isChecked()
        seperation_data['modelFolder'] = self.windows['settings'].ui.checkBox_modelFolder.isChecked()
        seperation_data['deepExtraction'] = self.windows['settings'].ui.checkBox_deepExtraction.isChecked()
        seperation_data['multithreading'] = self.windows['settings'].ui.checkBox_multithreading.isChecked()
        seperation_data['save_instrumentals'] = self.windows['settings'].ui.checkBox_autoSaveInstrumentals.isChecked()
        seperation_data['save_vocals'] = self.windows['settings'].ui.checkBox_autoSaveVocals.isChecked()
        # Combobox
        seperation_data['model'] = self.windows['settings'].ui.comboBox_instrumental.currentData()['path']
        seperation_data['modelDataPath'] = r"D:\Dilan\GitHub\ultimatevocalremovergui\src\inference\modelparams\2band_48000.json"
        seperation_data['isVocal'] = False
        # Lineedit (Constants)
        seperation_data['window_size'] = int(self.windows['settings'].ui.comboBox_winSize.currentText())
        # Other
        seperation_data['highEndProcess'] = self.windows['settings'].ui.comboBox_highEndProcess.currentText()
        seperation_data['aggressiveness'] = self.windows['settings'].ui.doubleSpinBox_aggressiveness.value()
        # -Complex variables (Difficult to extract)-

        if set(seperation_data.keys()) != set(converter.default_data.keys()):
            msg = (
                'Extracted Keys do not equal keys set by default converter!\n'
                f'\tExtracted Keys: {sorted(list(seperation_data.keys()))}\n'
                f'\tShould be Keys: {sorted(list(converter.default_data.keys()))}\n'
                f'\tExtracted Values:\n\t{pprint.pformat(seperation_data)}'
            )
            self.logger.debug(msg)
        else:
            msg = (
                'Successful extraction of seperation data!\n'
                f'\tExtracted Values:\n\t{pprint.pformat(seperation_data, compact=True)}'
            )
            self.logger.info(msg)

        return seperation_data

    def save_application(self):
        """Save the data for the application

        This includes widget states as well as user specific
        data like presets, selected language, ect.
        """
        def save_user_data():
            """Save user specific data"""
            self.settings.setValue('user/exportDirectory',
                                   self.windows['settings'].exportDirectory)
            self.settings.setValue('user/language',
                                   self.translator.loaded_language.code)
            self.settings.setValue('user/inputPaths',
                                   self.windows['main'].inputPaths)
            self.settings.setValue('user/inputsDirectory',
                                   self.windows['main'].inputsDirectory)
            self.settings.setValue('user/presets',
                                   list(self.windows['presetsEditor'].get_presets().items()))
            self.settings.setValue('user/presets_loadDir',
                                   self.windows['presetsEditor'].presets_loadDir)
            self.settings.setValue('user/presets_saveDir',
                                   self.windows['presetsEditor'].presets_saveDir)
            self.settings.setValue('user/theme',
                                   self.themeManager.loaded_theme)

        def save_widgets_data():
            """Save widget states

            Uses the save_window function for each window
            in the application
            """
            for window in self.windows.values():
                window.save_window()

        # Save
        save_widgets_data()
        save_user_data()

    def closeAllWindows(self):
        """Capture application close to save data"""
        self.logger.info('--- Closing application ---',
                         indent_forwards=True)

        # Save Application
        self.save_application()

        # Close Windows
        self.logger.info('Closing windows...')
        super().closeAllWindows()

        # Finish
        self.settings.sync()
        self.logger.indent_backwards()
        self.logger.info('--- Done! ---')


class ThemeManager:
    """Theme Manager for the application

    Manages the look of the widgets in the application

    Args:
        loaded_theme (str):
            Currently loaded theme in the application. To change, run method load_theme.
    """

    def __init__(self, app: CustomApplication):
        self.app = app
        self.logger = app.logger
        self.loaded_theme: str
        self.app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    def load_theme(self, theme: str = 'dark') -> bool:
        """Load a theme for the whole application

        Note:
            theme arg info:
                Casing is ignored

        Args:
            theme (str): Either "dark" or "light".  Defaults to "dark".

        Returns:
            bool: Whether the theme was successfully changed
        """
        def load_dark_theme():
            palette = QtGui.QPalette()

            darkColor = QtGui.QColor(45, 45, 45)
            disabledColor = QtGui.QColor(127, 127, 127)
            palette.setColor(QtGui.QPalette.Window, darkColor)
            palette.setColor(QtGui.QPalette.WindowText, Qt.white)
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(18, 18, 18))
            palette.setColor(QtGui.QPalette.AlternateBase, darkColor)
            palette.setColor(QtGui.QPalette.ToolTipBase, Qt.white)
            palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
            palette.setColor(QtGui.QPalette.Text, Qt.white)
            palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabledColor)
            palette.setColor(QtGui.QPalette.Button, darkColor)
            palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
            palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabledColor)
            palette.setColor(QtGui.QPalette.BrightText, Qt.red)
            palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
            palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, disabledColor)

            self.app.setPalette(palette)
            self.loaded_theme = 'dark'

        def load_light_theme():
            self.app.setPalette(self.app.style().standardPalette())
            self.loaded_theme = 'light'

        if theme == "dark":
            stylesheet = ResourcePaths.themes.dark
            load_dark_theme()
        elif theme == "light":
            stylesheet = ResourcePaths.themes.light
            load_light_theme()

        for window in self.app.windows.values():
            self.app.setStyleSheet(stylesheet)


def run():
    """Start UVR

    This function is executed by the main.py
    file one directory up
    """
    app = CustomApplication()
    sys.exit(app.exec_())

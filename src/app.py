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
from .inference import converter_v4
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
        self.settings = QtCore.QSettings(const.APPLICATION_SHORTNAME, const.APPLICATION_NAME)
        # self.settings.clear()
        self.resources = ResourcePaths()
        self.translator = Translator(self)
        self.threadpool = QtCore.QThreadPool(self)
        # -Load Windows-
        # Workaround for circular dependency
        from .windows import (mainwindow, settingswindow, presetseditorwindow)
        # Collection of windows
        self.windows: Dict[str, QtWidgets.QWidget] = {
            'main': mainwindow.MainWindow(self),
            'settings': settingswindow.SettingsWindow(self),
            'presetsEditor': presetseditorwindow.PresetsEditorWindow(self),
        }
        self.mainWindow = self.windows['main']
        self.settingsWindow = self.windows['settings']
        self.presetsEditorWindow = self.windows['presetsEditor']

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

        # -Before-

        # -Setup-
        setup_windows()
        improve_comboboxes()
        assign_lineEdit_validators()

        # -After-
        # Load language
        language = QtCore.QLocale(self.settings.value('user/language',
                                                      const.DEFAULT_SETTINGS['language'])).language()
        self.translator.load_language(language)
        # Raise main window
        self.windows['main'].activateWindow()
        self.windows['main'].raise_()

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
        seperation_data['stackOnly'] = self.windows['settings'].ui.checkBox_stackOnly.isChecked()
        seperation_data['saveAllStacked'] = self.windows['settings'].ui.checkBox_saveAllStacked.isChecked()
        seperation_data['modelFolder'] = self.windows['settings'].ui.checkBox_modelFolder.isChecked()
        seperation_data['customParameters'] = self.windows['settings'].ui.checkBox_customParameters.isChecked()
        seperation_data['multithreading'] = self.windows['settings'].ui.checkBox_multiThreading.isChecked()
        seperation_data['save_instrumentals'] = self.windows['settings'].ui.checkBox_autoSaveInstrumentals.isChecked()
        seperation_data['save_vocals'] = self.windows['settings'].ui.checkBox_autoSaveVocals.isChecked()
        # Combobox
        seperation_data['useModel'] = 'instrumental'
        seperation_data['instrumentalModel'] = self.windows['settings'].ui.comboBox_instrumental.currentData()
        seperation_data['vocalModel'] = ""
        seperation_data['stackModel'] = self.windows['settings'].ui.comboBox_stacked.currentData()
        # Lineedit (Constants)
        seperation_data['sr'] = int(self.windows['settings'].ui.lineEdit_sr.text())
        seperation_data['hop_length'] = int(self.windows['settings'].ui.lineEdit_hopLength.text())
        seperation_data['window_size'] = int(self.windows['settings'].ui.comboBox_winSize.currentText())
        seperation_data['n_fft'] = int(self.windows['settings'].ui.lineEdit_nfft.text())
        seperation_data['sr_stacked'] = int(self.windows['settings'].ui.lineEdit_sr_stacked.text())
        seperation_data['hop_length_stacked'] = int(self.windows['settings'].ui.lineEdit_hopLength_stacked.text())
        seperation_data['window_size_stacked'] = int(self.windows['settings'].ui.comboBox_winSize_stacked.currentText())
        seperation_data['n_fft_stacked'] = int(self.windows['settings'].ui.lineEdit_nfft_stacked.text())
        # -Complex variables (Difficult to extract)-
        # Stack passes
        stackpasses = 0
        if self.windows['settings'].ui.checkBox_stackPasses.isChecked():
            # Stack passes checkbox is checked so extract number from combobox
            stackpasses = int(self.windows['settings'].ui.comboBox_stackPasses.currentText())
        seperation_data['stackPasses'] = stackpasses
        # Resolution Type
        resType = self.windows['settings'].ui.comboBox_resType.currentText()
        resType = resType.lower().replace(' ', '_')
        seperation_data['resType'] = resType

        if set(seperation_data.keys()) != set(converter_v4.default_data.keys()):
            msg = (
                'Extracted Keys do not equal keys set by default converter!\n'
                f'\tExtracted Keys: {sorted(list(seperation_data.keys()))}\n'
                f'\tShould be Keys: {sorted(list(converter_v4.default_data.keys()))}\n'
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
                                   self.translator.loaded_language)
            self.settings.setValue('user/inputPaths',
                                   self.windows['main'].inputPaths)
            self.settings.setValue('user/inputsDirectory',
                                   self.windows['main'].inputsDirectory)
            self.settings.setValue('user/presets',
                                   self.windows['presetsEditor'].get_presets())
            self.settings.setValue('user/presets_loadDir',
                                   self.windows['presetsEditor'].presets_loadDir)
            self.settings.setValue('user/presets_saveDir',
                                   self.windows['presetsEditor'].presets_saveDir)
        
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


class Translator:
    """Localizer for the application

    Manages the languages for the applications

    Args:
        loaded_language (str):
            Currently loaded language in the application. To change, run method load_language.
    """
    def __init__(self, app: CustomApplication):
        self.app = app
        self.logger = app.logger
        self.loaded_language: str
        self._translator = QtCore.QTranslator(self.app)

    def load_language(self, language: QtCore.QLocale.Language = QtCore.QLocale.English) -> bool:
        """Load a language on the application

        Note:
            language arg info:
                If the language given is not supported, a warning message will be reported to the logger
                and the language will be set to english

        Args:
            language (QtCore.QLocale.Language, optional): Language to load. Defaults to English.

        Returns:
            bool: Whether the applications language was successfully changed to the given language
        """
        language_str = QtCore.QLocale.languageToString(language).lower()
        self.logger.info(f'Translating to "{language_str.capitalize()}"...',
                         indent_forwards=True)
        # Get path where translation file should be
        translation_path = os.path.join(self.app.resources.localizationDir, f'{language_str}.qm')
        if (not os.path.isfile(translation_path) and
                language != QtCore.QLocale.English):
            # Translation file does not exist
            # Load default language (english)
            self.logger.warning(f'Translation file does not exist! Switching to English. Language: {language_str}')
            self.logger.indent_backwards()
            self.load_language()
            return False
        # get language name to later store in settings
        self.loaded_language = QtCore.QLocale(language).name()
        # Load language
        if language == QtCore.QLocale.English:
            # English is base language so remove translator
            self.app.removeTranslator(self._translator)
        else:
            self._translator.load(translation_path)
            self.app.installTranslator(self._translator)

        # -Windows are initialized-
        # Update translation on all windows
        for window in self.app.windows.values():
            window.update_translation()
        # Update settings window
        for button in self.app.windows['settings'].ui.frame_languages.findChildren(QtWidgets.QPushButton):
            language_str = QtCore.QLocale.languageToString(language).lower()
            button_name = f'pushButton_{language_str}'
            if button.objectName() == button_name:
                # Language found
                button.setChecked(True)
            else:
                # Not selected language
                button.setChecked(False)
        
        self.logger.indent_backwards()
        return True


def run():
    """Start UVR

    This function is executed by the main.py
    file one directory up
    """
    app = CustomApplication()
    sys.exit(app.exec_())
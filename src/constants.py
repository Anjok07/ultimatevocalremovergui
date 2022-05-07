"""
Store Appliation info and default data
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtCore
# -Root imports-
from .inference import converter
from .translator import Translator
from collections import OrderedDict
import torch

__is_light_theme = bool(QtCore.QSettings("HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize",
                        QtCore.QSettings.NativeFormat).value("AppsUseLightTheme"))

VERSION = "0.0.5"
APPLICATION_SHORTNAME = 'UVR'
APPLICATION_NAME = 'Ultimate Vocal Remover'
# Values that are inside a preset with their corresponding label
# Key -> json key
# Value -> widget object name
JSON_TO_NAME = OrderedDict(**{
    # -Conversion-
    # Boolean
    'postProcess': 'checkBox_postProcess',
    'tta': 'checkBox_tta',
    'outputImage': 'checkBox_outputImage',
    'modelFolder': 'checkBox_modelFolder',
    'deepExtraction': 'checkBox_deepExtraction',
    # Number
    'aggressiveness': 'doubleSpinBox_aggressiveness',
    'highEndProcess': 'comboBox_highEndProcess',
    'windowSize': 'comboBox_winSize',
    # -Models-
    'ensemble': 'checkBox_ensemble',
    'instrumentalModelName': 'comboBox_instrumental',
    'vocalModelName': 'comboBox_vocal',
})
DEFAULT_SETTINGS = {
    # --Independent Data (Data not directly connected with widgets)--
    'inputPaths': [],
    # Directory to open when selecting a music file (Default: desktop)
    'inputsDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Export path (Default: desktop)
    'exportDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Language in format {language}_{country} (Default: system language)
    'language': Translator.SUPPORTED_LANGUAGES[QtCore.QLocale.system().language().name.decode('utf-8').lower()],
    # Presets for seperations
    'presets': [
        ['ALL', {
            # -Conversion-
            # Boolean
            'postProcess': True,
            'tta': True,
            'outputImage': True,
            'modelFolder': True,
            'deepExtraction': True,
            'windowSize': 1024,
            # Number
            'aggressiveness': 0.1,
            'highEndProcess': 'Bypass',
            # -Models-
            'ensemble': True,
        }],
        ['NONE', {
            # -Conversion-
            # Boolean
            'postProcess': False,
            'tta': False,
            'outputImage': False,
            'modelFolder': False,
            'deepExtraction': False,
            'windowSize': 352,
            # Number
            'aggressiveness': -0.1,
            'highEndProcess': 'Mirroring',
            # -Models-
            'ensemble': False,
        }]
    ],
    # Presets save directory (Default: desktop)
    'presets_saveDir': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Presets load directory (Default: desktop)
    'presets_loadDir': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # --Settings window -> Seperation Settings--
    # -Conversion-
    # Boolean
    'checkBox_gpuConversion': torch.cuda.is_available(),
    'checkBox_postProcess': converter.default_data['postProcess'],
    'checkBox_tta': converter.default_data['tta'],
    'checkBox_outputImage': converter.default_data['outputImage'],
    'checkBox_modelFolder': converter.default_data['modelFolder'],
    'checkBox_deepExtraction': converter.default_data['deepExtraction'],
    # Number
    'doubleSpinBox_aggressiveness': converter.default_data['aggressiveness'],
    'comboBox_highEndProcess': converter.default_data['highEndProcess'],
    # -Models-
    'checkBox_ensemble': False,
    'comboBox_instrumental': '',
    'comboBox_vocal': '',
    'comboBox_winSize': converter.default_data['window_size'],
    # -Presets-
    'comboBox_presets': '',
    # --Settings window -> Customization--
    'theme': 'light' if __is_light_theme else 'dark',
    # --Settings window -> Preferences--
    # -Settings-
    # Command off
    'comboBox_command': 'Off',
    # Notify on seperation finish
    'checkBox_notifiyOnFinish': False,
    # Notify on application updates
    'checkBox_notifyUpdates': True,
    # Open settings on startup
    'checkBox_settingsStartup': False,
    # Disable animations
    'checkBox_enableAnimations': True,
    # Disable Shortcuts
    'checkBox_showInfoButtons': True,
    # Process multiple files at once
    'checkBox_multithreading': converter.default_data['multithreading'],
    # -Export Settings-
    # Autosave Instrumentals/Vocals
    'checkBox_autoSaveInstrumentals': True,
    'checkBox_autoSaveVocals': True,
}


class QWinTaskbar_PLACEHOLDER:
    def __init__(self, parent):
        self.parent = parent
        self.progress_class = QWinTaskbarProgress_PLACEHOLDER(self.parent)

    def clearOverlayIcon(self):
        pass

    def eventFilter(self, arg__1, arg__2):
        pass

    def overlayAccessibleDescription(self):
        pass

    def overlayIcon(self):
        pass

    def progress(self):
        return self.progress_class

    def setOverlayAccessibleDescription(self, description):
        pass

    def setOverlayIcon(self, icon):
        pass

    def setWindow(self, window):
        pass

    def window(self):
        pass


class QWinTaskbarProgress_PLACEHOLDER:

    def __init__(self, parent):
        self.parent = parent

    def hide(self):
        pass

    def isPaused(self):
        pass

    def isStopped(self):
        pass

    def isVisible(self):
        pass

    def maximum(self):
        pass

    def minimum(self):
        pass

    def pause(self):
        pass

    def reset(self):
        pass

    def resume(self):
        pass

    def setMaximum(self, maximum):
        pass

    def setMinimum(self, minimum):
        pass

    def setPaused(self, paused):
        pass

    def setRange(self, minimum, maximum):
        pass

    def setValue(self, value):
        pass

    def setVisible(self, visible):
        pass

    def show(self):
        pass

    def stop(self):
        pass

    def value(self):
        pass

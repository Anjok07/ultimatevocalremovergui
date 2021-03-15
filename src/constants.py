"""
Store Appliation info and default data
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtCore
# -Root imports-
from .inference import converter_v4
from collections import OrderedDict

VERSION = "0.0.5"
APPLICATION_SHORTNAME = 'UVR'
APPLICATION_NAME = 'Ultimate Vocal Remover'
# Values that are inside a preset with their corresponding label
# Key -> json key
# Value -> widget object name
JSON_TO_NAME = OrderedDict(**{
    # -Conversion-
    # Boolean
    'gpuConversion': 'checkBox_gpuConversion',
    'postProcess': 'checkBox_postProcess',
    'tta': 'checkBox_tta',
    'outputImage': 'checkBox_outputImage',
    'useStackPasses': 'checkBox_stackPasses',
    'stackOnly': 'checkBox_stackOnly',
    'saveAllStacked': 'checkBox_saveAllStacked',
    'modelFolder': 'checkBox_modelFolder',
    'customParameters': 'checkBox_customParameters',
    # Combobox
    'stackPassesNum': 'comboBox_stackPasses',
    # -Engine-
    'aiEngine': 'comboBox_engine',
    'resolutionType': 'comboBox_resType',
    # -Models-
    'instrumentalModelName': 'comboBox_instrumental',
    'stackedModelName': 'comboBox_stacked',
    # Sampling Rate (SR)
    'sr': 'lineEdit_sr',
    'srStacked': 'lineEdit_sr_stacked',
    # Hop Length
    'hopLength': 'lineEdit_hopLength',
    'hopLengthStacked': 'lineEdit_hopLength_stacked',
    # Window size
    'windowSize': 'comboBox_winSize',
    'windowSizeStacked': 'comboBox_winSize_stacked',
    # NFFT
    'nfft': 'lineEdit_nfft',
    'nfftStacked': 'lineEdit_nfft_stacked',
})

DEFAULT_SETTINGS = {
    # --Independent Data (Data not directly connected with widgets)--
    'inputPaths': [],
    # Directory to open when selecting a music file (Default: desktop)
    'inputsDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Export path (Default: desktop)
    'exportDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Language in format {language}_{country} (Default: system language)
    'language': QtCore.QLocale.system().name(),
    # Presets for seperations
    'presets': {},
    # Presets save directory (Default: desktop)
    'presets_saveDir': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Presets load directory (Default: desktop)
    'presets_loadDir': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # --Settings window -> Seperation Settings--
    # -Conversion-
    # Boolean
    'checkBox_gpuConversion': converter_v4.default_data['gpuConversion'],
    'checkBox_postProcess': converter_v4.default_data['postProcess'],
    'checkBox_tta': converter_v4.default_data['tta'],
    'checkBox_outputImage': converter_v4.default_data['outputImage'],
    'checkBox_stackOnly': converter_v4.default_data['stackOnly'],
    'checkBox_saveAllStacked': converter_v4.default_data['saveAllStacked'],
    'checkBox_modelFolder': converter_v4.default_data['modelFolder'],
    'checkBox_customParameters': converter_v4.default_data['customParameters'],
    'checkBox_stackPasses': True,
    # Combobox
    'comboBox_stackPasses': 1,
    # -Engine-
    'comboBox_engine': 'v4',
    'comboBox_resType': 'Kaiser Fast',
    # -Models-
    'comboBox_instrumental': '',
    'comboBox_stacked': '',
    # Sampling Rate (SR)
    'lineEdit_sr': converter_v4.default_data['sr'],
    'lineEdit_sr_stacked': converter_v4.default_data['sr'],
    # Hop Length
    'lineEdit_hopLength': converter_v4.default_data['hop_length'],
    'lineEdit_hopLength_stacked': converter_v4.default_data['hop_length'],
    # Window size
    'comboBox_winSize': converter_v4.default_data['window_size'],
    'comboBox_winSize_stacked': converter_v4.default_data['window_size'],
    # NFFT
    'lineEdit_nfft': converter_v4.default_data['n_fft'],
    'lineEdit_nfft_stacked': converter_v4.default_data['n_fft'],
    # -Presets-
    'comboBox_presets': '',
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
    'checkBox_disableAnimations': False,
    # Disable Shortcuts
    'checkBox_disableShortcuts': False,
    # Process multiple files at once
    'checkBox_multiThreading': False,
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

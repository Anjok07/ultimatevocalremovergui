"""
Main Application
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
from PySide2.QtWinExtras import (QWinTaskbarButton)
# -Root imports-
from .resources.resources_manager import ResourcePaths
from .windows import (mainwindow, settingswindow)
from .inference import converter_v4_copy as converter_v4
# -Other-
import datetime as dt
from collections import defaultdict
from collections import OrderedDict
import os
import sys
# Code annotation
from typing import (Dict, List, Tuple, Optional)
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
APPLICATION_SHORTNAME = 'UVR'
APPLICATION_NAME = 'Ultimate Vocal Remover'
DEFAULT_SETTINGS = {
    # --Independent Data (Data not directly connected with widgets)--
    'inputPaths': [],
    # Default export path (Desktop)
    'exportDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Default language in format {language}_{country}
    'language': QtCore.QLocale.system().name(),
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
    'checkBox_stackPasses': False,
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

    # --Settings window -> Preferences--
    # -Settings-
    # Command off
    'comboBox_command': 'Off',
    # Notify on seperation finish
    'checkBox_notifiyOnFinish': False,
    # Notify on application updates
    'checkBox_notifyUpdates': True,
    # Open settings on startup
    'checkBox_settingsStartup': True,
    # Disable animations
    'checkBox_disableAnimations': False,
    # Disable Shortcuts
    'checkBox_disableShortcuts': False,
    # -Export Settings-
    # Autosave Instrumentals/Vocals
    'checkBox_autoSaveInstrumentals': False,
    'checkBox_autoSaveVocals': False,
}


def dict_to_HTMLtable(x: dict, header: str) -> str:
    """
    Convert a 1D-dictionary into an HTML table
    """
    # Generate table contents
    values = ''
    for i, (key, value) in enumerate(x.items()):
        if i % 2:
            color = "#000"
        else:
            color = "#333"

        values += f'<tr style="background-color:{color};border:none;"><td>{key}</td><td>{value}</td></tr>\n'
    # HTML string
    htmlTable = """
    <table border="1" cellspacing="0" cellpadding="0" width="100%">
        <tr><th colspan="2" style="background-color:#555500">{header}</th></tr>
        {values}
    </table>
    """.format(header=header,
               values=values)
    # Return HTML string
    return htmlTable

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
        self.settings = QtCore.QSettings(APPLICATION_SHORTNAME, APPLICATION_NAME)
        # self.settings.clear()
        self.resources = ResourcePaths()
        self.translator = Translator(self)
        self.threadpool = QtCore.QThreadPool(self)
        # -Variables/Functions-
        self.in_debug_mode = lambda: self.windows['settings'].ui.comboBox_command.currentIndex() == 2
        # -Load Windows-
        # Collection of windows
        self.windows: Dict[str, QtWidgets.QWidget] = {
            'main': MainWindow(self),
            'settings': SettingsWindow(self),
        }
        self.windows['main'].show()

        self.load_settings()
        self.late_setup()

    def load_settings(self):
        """
        Load the settings saved
        """
        language = QtCore.QLocale(self.settings.value('settingswindow/language',
                                                      DEFAULT_SETTINGS['language'])).language()
        # Load language -> if not specified, try loading system language
        self.translator.load_language(language)

        open_settings = self.settings.value('settingswindow/checkBox_settingsStartup',
                                            DEFAULT_SETTINGS['checkBox_settingsStartup'],
                                            bool)
        if open_settings:
            self.windows['settings'].show()

    def late_setup(self):
        """
        Update windows, set up binds and images for widgets that interact
        with widgets/methods from other windows

        (Binds cannot be set to widgets/methods that have not been initialized yet)
        """
        def improve_comboboxes(window: QtWidgets.QWidget):
            """
            Improvements:
                - Always show full contents of popup
                - Center align editable comboboxes
            """
            for combobox in window.findChildren(QtWidgets.QComboBox):
                combobox.showPopup = lambda wig=combobox, func=combobox.showPopup: self.improved_combobox_showPopUp(wig, func)  # nopep8

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

        for window in self.windows.values():
            window.setup_window()
            improve_comboboxes(window)

        assign_lineEdit_validators()

    @staticmethod
    def improved_combobox_showPopUp(widget: QtWidgets.QComboBox, popup_func: QtWidgets.QComboBox.showPopup):
        """
        Improve the QComboBox by overriding the showPopup function to
        also adjust the size of the view to its contents before showing 
        """
        view = widget.view()
        fm = widget.fontMetrics()
        widths = [fm.size(0, widget.itemText(i)).width()
                  for i in range(widget.count())]
        if widths:
            # Combobox has contents
            # + 30 as a buffer for the scrollbar
            view.setMinimumWidth(max(widths) + 30)
        popup_func()

    def extract_seperation_data(self) -> dict:
        """
        Extract the saved seperation data
        """
        seperation_data = OrderedDict()

        # Input/Export
        # self.windows['settings'].inputPaths
        seperation_data['input_paths'] = ['B:/boska/Desktop/Test inference/test.mp3']
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

        if set(converter_v4.default_data.keys()) != set(seperation_data.keys()):
            self.debug_to_command(f'Extracted Keys do not equal keys set by default converter!\nExtracted Keys: {sorted(list(seperation_data.keys()))}\nShould be Keys: {sorted(list(converter_v4.default_data.keys()))}',
                                  priority=1)

        return seperation_data

    def debug_to_command(self, text: str, priority: Optional[int] = 'default', debug_prefix: bool = True):
        """
        Shortcut function for mainwindow write to gui

        Priority:
            1: Red Color
            2: Orange Color
            'default': Default color
        """
        if hasattr(self, 'windows'):
            debug_colours = {1: QtGui.QColor("#EE3737"),
                             2: QtGui.QColor("#FFAE42"),
                             'default': QtGui.QColor("#CCC")}
            if self.in_debug_mode():
                self.windows['main'].ui.textBrowser_command.setTextColor(debug_colours[priority])
                if debug_prefix:
                    text = f'DEBUG: {text}'
                self.windows['main'].write_to_command(text)
                self.windows['main'].ui.textBrowser_command.setTextColor(debug_colours['default'])
        else:
            # Windows not initialized yet
            print(f'Debug Text: {text}\nPriority: {priority}')

    def closeAllWindows(self):
        """
        Capture application close to save data
        """
        # --Settings Window--
        self.settings.beginGroup('settingswindow')
        # Widgets
        setting_widgets = [*self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QCheckBox),
                           *self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QComboBox),
                           *self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QLineEdit), ]
        for widget in setting_widgets:
            widgetObjectName = widget.objectName()
            if not widgetObjectName in DEFAULT_SETTINGS:
                # Default settings do not exist
                self.debug_to_command(text=f'"{widgetObjectName}" does not have a default setting!',
                                      priority=2)
                continue
            if isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
                self.settings.setValue(widgetObjectName,
                                       value)
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
                self.settings.setValue(widgetObjectName,
                                       value)
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
                self.settings.setValue(widgetObjectName,
                                       value)
        # Back-end Data
        self.settings.setValue('exportDirectory',
                               self.windows['settings'].exportDirectory)
        self.settings.setValue('language',
                               self.translator.loaded_language)
        self.settings.endGroup()
        self.settings.sync()

        super().closeAllWindows()


class Translator:
    def __init__(self, app: CustomApplication):
        self.app = app
        self.loaded_language: str
        self._translator = QtCore.QTranslator(self.app)

    def load_language(self, language: QtCore.QLocale.Language = QtCore.QLocale.English):
        """
        Load specified language by file name

        Default is english
        """
        # Get path where translation file should be
        language_str = QtCore.QLocale.languageToString(language).lower()
        translation_path = os.path.join(self.app.resources.localizationDir, f'{language_str}.qm')
        if (not os.path.isfile(translation_path) and
                language != QtCore.QLocale.English):
            # Translation file does not exist
            # Load default language (english)
            self.app.debug_to_command(f'Translation file does not exist! Switching to English.\nLanguage: {language_str}\nPath: {translation_path}',
                                      priority=1)
            self.load_language()
            return
        # get language name to later store in settings
        self.loaded_language = QtCore.QLocale(language).name()
        # Load language
        if language == QtCore.QLocale.English:
            # English is base language so remove translator
            self.app.debug_to_command(f'Translating to english...')
            self.app.removeTranslator(self._translator)
        else:
            self.app.debug_to_command(f'Translating to {language_str}...')
            self._translator.load(translation_path)
            self.app.installTranslator(self._translator)

        if hasattr(self.app, 'windows'):
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


class MainWindow(QtWidgets.QWidget):
    def __init__(self, app: CustomApplication):
        # -Window setup-
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.app = app
        self.settings = QtCore.QSettings(APPLICATION_SHORTNAME, APPLICATION_NAME)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.icon))

        # -Other Variables-
        # Independent data
        self.inputPaths = self.settings.value('inputPaths',
                                              DEFAULT_SETTINGS['inputPaths'],
                                              type=list)

    # -Initialization methods-
    def setup_window(self):
        """
        Set up the window with binds, images, saved settings

        (Only run right after window initialization of main and settings window)
        """
        def load_geometry():
            """
            Load the geometry of this window
            """
            # Window is centered on primary window
            default_geometry = self.geometry()
            point = QtCore.QPoint()
            point.setX(self.app.primaryScreen().size().width() / 2)
            point.setY(self.app.primaryScreen().size().height() / 2)
            default_geometry.moveCenter(point)
            # Get geometry
            geometry = self.settings.value('mainwindow/geometry',
                                           default_geometry)
            self.setGeometry(geometry)

        def load_images():
            """
            Load the images for this window and assign them to their widgets
            """
            # Settings button
            icon = QtGui.QPixmap(ResourcePaths.images.settings)
            self.ui.pushButton_settings.setIcon(icon)
            self.ui.pushButton_settings.setIconSize(QtCore.QSize(25, 25))

        def bind_widgets():
            """
            Bind the widgets here
            """
            # -Override binds-
            # Music file drag & drop
            self.ui.label_musicFiles.dragEnterEvent = self.label_musicFiles_dragEnterEvent
            self.ui.label_musicFiles.dropEvent = self.label_musicFiles_dropEvent
            # -Pushbuttons-
            self.ui.pushButton_settings.clicked.connect(self.pushButton_settings_clicked)
            self.ui.pushButton_seperate.clicked.connect(self.pushButton_seperate_clicked)

        # -Before setup-
        self.winTaskbar = QWinTaskbarButton(self)
        self.winTaskbar.setWindow(self.windowHandle())
        self.winTaskbar_progress = self.winTaskbar.progress()

        # -Setup-
        load_geometry()
        load_images()
        bind_widgets()

        # -After setup-
        # Load saved settings for widgets
        self._load_data()

    def _load_data(self, default: bool = False):
        """
        Load the data for this window

        (Only run right after window initialization or to reset settings)

        Parameters:
            default(bool):
                Reset to the default settings
        """
        self.settings.beginGroup('mainwindow')
        if default:
            # Delete settings group
            self.settings.remove('mainwindow')

        # -Load Settings-
        # None

        # -Done-
        self.settings.endGroup()
        self._late_update()

    def _late_update(self):
        """
        Late update cross-windows

        (Only run right after window initialization)
        """
        self.update_window()

    # -Widget Binds-
    def pushButton_settings_clicked(self):
        """
        Open the settings window
        """
        self.app.debug_to_command('Opening settings window...')
        # Reshow window
        self.app.windows['settings'].setWindowState(Qt.WindowNoState)
        self.app.windows['settings'].show()
        # Focus window
        self.app.windows['settings'].activateWindow()
        self.app.windows['settings'].raise_()

        self.app.windows['settings'].update_window()

    def pushButton_seperate_clicked(self):
        """
        Seperate given files
        """
        # -Extract seperation info from GUI-
        seperation_data = self.app.extract_seperation_data()
        self.app.debug_to_command(dict_to_HTMLtable(seperation_data, 'Seperation Data'),
                                  debug_prefix=False)
        # -Seperation-
        # Create instance
        vocalRemover = converter_v4.VocalRemoverWorker(seperation_data=seperation_data,)
        # Bind events
        vocalRemover.signals.start.connect(self.seperation_start)
        vocalRemover.signals.message.connect(self.seperation_write)
        vocalRemover.signals.progress.connect(self.seperation_update_progress)
        vocalRemover.signals.error.connect(self.seperation_error)
        vocalRemover.signals.finished.connect(self.seperation_finish)
        # Start seperation
        self.app.threadpool.start(vocalRemover)

    def label_musicFiles_dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Check whether the files the user is dragging over the widget
        is valid or not
        """
        if event.mimeData().urls():
            # URLs dragged
            event.accept()
        else:
            event.ignore()

    def label_musicFiles_dropEvent(self, event: QtGui.QDropEvent):
        """
        Assign dropped paths to list
        """
        inputPaths = []
        for url in event.mimeData().urls():
            inputPaths.append(url.toLocalFile())

        self.ui.label_musicFiles.setText(repr(inputPaths))
        self.inputPaths = inputPaths

    # -Seperation Methods-
    def seperation_start(self):
        """
        Seperation has started
        """
        self.app.debug_to_command(f'The seperation has started.',
                                  priority=2)
        # Disable Seperation Button
        self.ui.pushButton_seperate.setEnabled(False)
        # Setup WinTaskbar
        self.winTaskbar.setOverlayAccessibleDescription('Seperating...')
        self.winTaskbar.setOverlayIcon(QtGui.QIcon(ResourcePaths.images.folder))
        self.winTaskbar_progress.setVisible(True)

    def seperation_write(self, text: str, priority: Optional[int] = 'default'):
        """
        Write to GUI
        """
        self.write_to_command(text)

    def seperation_update_progress(self, progress: int):
        """
        Update both progressbars in Taskbar and GUI
        with the given progress
        """
        self.app.debug_to_command(f'Updating progress: {progress}%')
        self.winTaskbar_progress.setValue(progress)
        self.ui.progressBar.setValue(progress)

    def seperation_error(self, message: Tuple[str, str]):
        """
        Error occured while seperating

        Parameters:
            message(tuple):
                Index 0: Error Message
                Index 1: Detailed Message
        """
        self.app.debug_to_command(f'An error occured!\nMessage: {message[0]}\nDetailed Message: {message[1]}',
                                  priority=1)
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('An Error Occurred')
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg.setText(
            message[0] + '\n\nIf the issue is not clear, please contact the creator and attach a screenshot of the detailed message with the file and settings that caused it!')
        msg.setDetailedText(message[1])
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.setWindowFlag(Qt.WindowStaysOnTopHint)
        msg.exec_()

        self.seperation_finish(failed=True)

    def seperation_finish(self, elapsed_time: str = '1:02:03', failed: bool = False):
        """
        Finished seperation
        """
        def seperation_reset():
            """
            Reset the progress bars and WinTaskbar
            """
            self.winTaskbar.setOverlayAccessibleDescription('')
            self.winTaskbar.clearOverlayIcon()
            self.winTaskbar_progress.setVisible(False)
            self.seperation_update_progress(0)
        self.app.debug_to_command(f'The seperation has finished.',
                                  priority=2)

        # -Create MessageBox-
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Seperation Complete')
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setText(f"UVR:\nYour seperation has finished!\n\nTime elapsed: {elapsed_time}")  # nopep8
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.setWindowFlag(Qt.WindowStaysOnTopHint)
        msg.exec_()
        # -Reset progress-
        seperation_reset()
        # -Focus main window-
        self.activateWindow()
        self.raise_()
        # -Enable Seperation Button
        self.ui.pushButton_seperate.setEnabled(True)

    # -Other Methods-

    def update_window(self):
        """
        Update the text and states of widgets
        in this window
        """
        self.ui.label_musicFiles.setText(repr(self.inputPaths))

    def write_to_command(self, text: str):
        """
        Write to the command line

        Parameters:
            text(str):
                Text written in the command on a new line
        """
        if hasattr(self.app, 'windows'):
            self.app.windows['settings'].ui.pushButton_clearCommand.setVisible(True)
        self.ui.textBrowser_command.append(text)

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        # -Save the geometry for this window-
        self.settings.beginGroup('mainwindow')
        self.settings.setValue('geometry',
                               self.geometry())
        self.settings.endGroup()
        # Commit Save
        self.settings.sync()
        # -Close all windows-
        self.app.closeAllWindows()
        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.ui.retranslateUi(self)


class SettingsWindow(QtWidgets.QWidget):
    def __init__(self, app: CustomApplication):
        super(SettingsWindow, self).__init__()
        self.ui = settingswindow.Ui_SettingsWindow()
        self.ui.setupUi(self)
        self.app = app
        self.settings = QtCore.QSettings(APPLICATION_SHORTNAME, APPLICATION_NAME)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.settings))

        # -Other Variables-
        # Independent data
        self.exportDirectory = self.settings.value('settingswindow/exportDirectory',
                                                   DEFAULT_SETTINGS['exportDirectory'],
                                                   type=str)
    # -Initialization methods-

    def setup_window(self):
        """
        Set up the window with binds, images, saved settings

        (Only run right after window initialization of main and settings window)
        """
        def load_geometry():
            """
            Load the geometry of this window
            """
            # Window is centered on primary window
            default_geometry = self.geometry()
            point = QtCore.QPoint()
            point.setX(self.app.primaryScreen().size().width() / 2)
            point.setY(self.app.primaryScreen().size().height() / 2)
            default_geometry.moveCenter(point)
            # Get geometry
            geometry = self.settings.value('settingswindow/geometry',
                                           default_geometry)
            self.setGeometry(geometry)

        def load_images():
            """
            Load the images for this window and assign them to their widgets
            """
            # Flag images
            for button in self.ui.frame_languages.findChildren(QtWidgets.QPushButton):
                # Loop through every button in the languages frame
                language = button.objectName().split('_')[1]
                button.setText('')

                # -Prepare rounded image-
                # Load original image
                img_path = getattr(ResourcePaths.images.flags, language)
                origin_img = QtGui.QPixmap(img_path)
                origin_img = origin_img.scaled(button.width(), button.height(),
                                               mode=Qt.TransformationMode.SmoothTransformation)
                # Create new image based on origins size
                rounded = QtGui.QPixmap(origin_img.size())
                rounded.fill(Qt.transparent)
                # Add rounded clip area
                path = QtGui.QPainterPath()
                path.addRoundedRect(2.75, 2.75, rounded.width() - 5.5, rounded.height() - 5.5, 8, 8)
                # Paint original image on new image
                painter = QtGui.QPainter(rounded)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                painter.setClipPath(path)
                painter.drawPixmap(0, 0, origin_img.width(), origin_img.height(), origin_img)
                painter.end()

                # Set new image to icon
                button.setIcon(QtGui.QIcon(rounded))
                button.setIconSize(rounded.size())
            # Folder icon
            folder_img = QtGui.QPixmap(ResourcePaths.images.folder)
            self.ui.pushButton_exportDirectory.setIcon(folder_img)
            self.ui.pushButton_exportDirectory.setIconSize(QtCore.QSize(18, 18))

        def bind_widgets():
            """
            Bind the widgets here
            """
            # -Main buttons-
            # Main control
            self.ui.pushButton_resetDefault.clicked.connect(self.pushButton_resetDefault_clicked)
            # Menu
            self.menu_group.buttonClicked.connect(lambda btn:
                                                  self.menu_loadPage(index=self.menu_group.id(btn)))
            # -Seperation Settings Page-
            # Checkboxes
            self.ui.checkBox_stackPasses.stateChanged.connect(self.update_page_seperationSettings)
            self.ui.checkBox_stackOnly.stateChanged.connect(self.update_page_seperationSettings)
            self.ui.checkBox_customParameters.stateChanged.connect(self.update_page_seperationSettings)
            # Comboboxes
            self.ui.comboBox_instrumental.currentIndexChanged.connect(self.update_page_seperationSettings)
            self.ui.comboBox_stacked.currentIndexChanged.connect(self.update_page_seperationSettings)
            self.ui.comboBox_engine.currentIndexChanged.connect(self._update_selectable_models)
            # -Preferences Page-
            # Language
            for button in self.ui.frame_languages.findChildren(QtWidgets.QPushButton):
                # Loop through every button in the languages frame
                # Get capitalized language from button name
                language_str = button.objectName().split('_')[1].capitalize()
                # Get language as QtCore.QLocale.Language
                language: QtCore.QLocale.Language = getattr(QtCore.QLocale, language_str)
                # Call load language on button click
                button.clicked.connect(lambda *args, lan=language: self.app.translator.load_language(lan))
            # Pushbuttons
            self.ui.pushButton_exportDirectory.clicked.connect(self.pushButton_exportDirectory_clicked)
            self.ui.pushButton_clearCommand.clicked.connect(self.pushButton_clearCommand_clicked)
            # Comboboxes
            self.ui.comboBox_command.currentIndexChanged.connect(self.comboBox_command_currentIndexChanged)

        # -Before setup-
        # Connect button group for menu together
        self.menu_group = QtWidgets.QButtonGroup(self)  # Menu group
        self.menu_group.addButton(self.ui.radioButton_seperationSettings,
                                  id=0)
        self.menu_group.addButton(self.ui.radioButton_shortcuts,
                                  id=1)
        self.menu_group.addButton(self.ui.radioButton_customization,
                                  id=2)
        self.menu_group.addButton(self.ui.radioButton_preferences,
                                  id=3)

        # -Setup-
        load_geometry()
        load_images()
        bind_widgets()

        # -After setup-
        self._update_selectable_models()
        # Load saved settings for widgets
        self._load_data()

    def _load_data(self, default: bool = False):
        """
        Load the data for this window

        (Only run right after window initialization or to reset settings)

        Parameters:
            default(bool):
                Reset to the default settings
        """
        self.settings.beginGroup('settingswindow')
        if default:
            # Delete settings group
            self.settings.remove("")
        # -Load Settings-
        # Widgets
        setting_widgets = [*self.ui.stackedWidget.findChildren(QtWidgets.QCheckBox),
                           *self.ui.stackedWidget.findChildren(QtWidgets.QComboBox),
                           *self.ui.stackedWidget.findChildren(QtWidgets.QLineEdit), ]
        for widget in setting_widgets:
            widgetObjectName = widget.objectName()
            # -Errors-
            if not widgetObjectName in DEFAULT_SETTINGS:
                if not widgetObjectName:
                    # Empty object name no need to notify
                    continue
                # Default settings do not exist
                self.app.debug_to_command(text=f'"{widgetObjectName}" does not have a default setting!',
                                          priority=2)
                continue

            # -Finding the instance and loading appropiately-
            if isinstance(widget, QtWidgets.QCheckBox):
                value = self.settings.value(widgetObjectName,
                                            defaultValue=DEFAULT_SETTINGS[widgetObjectName],
                                            type=bool)
                widget.setChecked(value)
            elif isinstance(widget, QtWidgets.QComboBox):
                value = self.settings.value(widgetObjectName,
                                            defaultValue=DEFAULT_SETTINGS[widgetObjectName],
                                            type=str)
                if widget.isEditable():
                    # Allows self-typing
                    widget.setCurrentText(value)
                else:
                    # Only allows a list to choose from
                    all_items = [widget.itemText(i) for i in range(widget.count())]
                    for i, item in enumerate(all_items):
                        if item == value:
                            # Both have the same text
                            widget.setCurrentIndex(i)
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = self.settings.value(widgetObjectName,
                                            defaultValue=DEFAULT_SETTINGS[widgetObjectName],
                                            type=str)
                widget.setText(value)

        # -Done-
        self.settings.endGroup()
        self._late_update()

    def _late_update(self):
        """
        Late update cross-windows

        (Only run right after window initialization)
        """
        self.comboBox_command_currentIndexChanged()
        self.update_window()

    # -Widget Binds-
    def pushButton_clearCommand_clicked(self):
        """
        Clear the command line and append
        """
        self.app.debug_to_command(text='Clearing Command Line...')
        self.ui.pushButton_clearCommand.setVisible(False)

        index = self.ui.comboBox_command.currentIndex()
        current_date = dt.datetime.now().strftime("%H:%M:%S")

        self.app.windows['main'].ui.textBrowser_command.clear()
        if index == 1:
            self.app.windows['main'].ui.textBrowser_command.append(f'Command Line [{current_date}]')
        elif index == 2:
            self.app.windows['main'].ui.textBrowser_command.append(f'Command Line (DEBUG MODE) [{current_date}]')

    def pushButton_exportDirectory_clicked(self):
        """
        Let user select an export directory for the seperation
        """
        self.app.debug_to_command(text='Selecting Export Directory...')
        # Ask for directory
        filepath = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                              caption='Select Export Directory',
                                                              dir=self.settings.value('seperation/export_path',
                                                                                      DEFAULT_SETTINGS['exportDirectory'])
                                                              )

        if not filepath:
            # No directory specified
            return
        # Update export path value
        self.exportDirectory = filepath
        self.update_page_preferences()

    def pushButton_resetDefault_clicked(self):
        """
        Reset settings to default
        """
        self._load_data(default=True)

    def comboBox_command_currentIndexChanged(self):
        """
        Changed mode for command line
        """
        self.pushButton_clearCommand_clicked()
        self.update_page_preferences()

    # -Update and Save Methods-
    # Whole window (All widgets)
    def update_window(self):
        """
        Update the values and states of all widgets
        in this window
        """
        self.update_page_seperationSettings()
        self.update_page_shortcuts()
        self.update_page_customization()
        self.update_page_preferences()

    def save_window(self):
        """
        Save all values of the widgets in the
        settings window
        """

    # Seperation Settings Page
    def update_page_seperationSettings(self):
        """
        Update values and states of all widgets in the
        seperation settings page
        """
        def subgroup_model_update_constants(model_basename: str, widgets: Dict[str, QtWidgets.QLineEdit]):
            """
            Update values and states of the constant widgets
            in the model subgroup
            """
            def extract_constants() -> Dict[str, int]:
                """
                Extract SR, HOP_LENGTH, WINDOW_SIZE and NFFT
                from the model's name
                """
                text_parts = model_basename.split('_')[1:]
                model_values = {}

                for text_part in text_parts:
                    if 'sr' in text_part:
                        text_part = text_part.replace('sr', '')
                        if text_part.isdecimal():
                            try:
                                model_values['sr'] = int(text_part)
                                continue
                            except ValueError:
                                # Cannot convert string to int
                                pass
                    if 'hl' in text_part:
                        text_part = text_part.replace('hl', '')
                        if text_part.isdecimal():
                            try:
                                model_values['hop_length'] = int(text_part)
                                continue
                            except ValueError:
                                # Cannot convert string to int
                                pass
                    if 'w' in text_part:
                        text_part = text_part.replace('w', '')
                        if text_part.isdecimal():
                            try:
                                model_values['window_size'] = int(text_part)
                                continue
                            except ValueError:
                                # Cannot convert string to int
                                pass
                    if 'nf' in text_part:
                        text_part = text_part.replace('nf', '')
                        if text_part.isdecimal():
                            try:
                                model_values['n_fft'] = int(text_part)
                                continue
                            except ValueError:
                                # Cannot convert string to int
                                pass
                return model_values

            if self.ui.checkBox_customParameters.isChecked():
                # Manually types constants
                for widget in widgets.values():
                    # Enable all widgets
                    widget.setEnabled(True)
                return

            constants = extract_constants()
            for key, widget in widgets.items():
                # Enable all widgets
                if not key in constants.keys():
                    # Model did not contain this constant
                    # so make enable the entry
                    widget.setEnabled(True)
                    continue
                widget.setEnabled(False)
                widget.setText(str(constants[key]))

        # -Conversion Subgroup-
        # Stack Passes
        if self.ui.checkBox_stackPasses.isChecked():
            self.ui.comboBox_stackPasses.setVisible(True)
            self.ui.checkBox_saveAllStacked.setVisible(True)
            self.ui.checkBox_stackOnly.setVisible(True)
        else:
            self.ui.comboBox_stackPasses.setVisible(False)
            self.ui.checkBox_saveAllStacked.setVisible(False)
            self.ui.checkBox_saveAllStacked.setChecked(False)
            self.ui.checkBox_stackOnly.setVisible(False)
            self.ui.checkBox_stackOnly.setChecked(False)

        # -Models Subgroup-
        # Get selected models
        instrumental_model = self.ui.comboBox_instrumental.currentText()
        instrumental_widgets = {'sr': self.ui.lineEdit_sr,
                                'hop_length': self.ui.lineEdit_hopLength,
                                'window_size': self.ui.comboBox_winSize.lineEdit(),
                                'n_fft': self.ui.lineEdit_nfft, }
        stacked_model = self.ui.comboBox_stacked.currentText()
        stacked_widgets = {'sr': self.ui.lineEdit_sr_stacked,
                           'hop_length': self.ui.lineEdit_hopLength_stacked,
                           'window_size': self.ui.comboBox_winSize_stacked.lineEdit(),
                           'n_fft': self.ui.lineEdit_nfft_stacked, }

        if not self.ui.checkBox_stackOnly.isChecked():
            # Show widgets
            self.ui.frame_instrumentalComboBox.setVisible(True)
            self.ui.comboBox_winSize.setVisible(True)
            for widget in instrumental_widgets.values():
                widget.setVisible(True)
            # Update entries
            subgroup_model_update_constants(model_basename=instrumental_model,
                                            widgets=instrumental_widgets)
        else:
            # Hide widgets
            self.ui.frame_instrumentalComboBox.setVisible(False)
            self.ui.comboBox_winSize.setVisible(False)
            for widget in instrumental_widgets.values():
                widget.setVisible(False)

        if self.ui.checkBox_stackPasses.isChecked():
            # Show widgets
            self.ui.frame_stackComboBox.setVisible(True)
            self.ui.comboBox_winSize_stacked.setVisible(True)
            for widget in stacked_widgets.values():
                widget.setVisible(True)
            # Update entries
            subgroup_model_update_constants(model_basename=stacked_model,
                                            widgets=stacked_widgets)
        else:
            # Hide widgets
            self.ui.frame_stackComboBox.setVisible(False)
            self.ui.comboBox_winSize_stacked.setVisible(False)
            for widget in stacked_widgets.values():
                widget.setVisible(False)

    def save_page_seperationSettings(self):
        """
        Save the values of the widgets in the
        seperation settings page
        """

    def _update_selectable_models(self):
        """
        Update the list of models to select from in the
        seperation settings page based on the selected AI Engine
        """
        def fill_model_comboBox(widget: QtWidgets.QComboBox, folder: str):
            """
            Fill the combobox for the model
            """
            currently_selected_model_name = widget.currentText()
            widget.clear()
            for index, f in enumerate(os.listdir(folder)):
                if not f.endswith('.pth'):
                    # File is not a model file, so skip
                    continue
                # Get data
                full_path = os.path.join(folder, f)
                model_name = os.path.splitext(os.path.basename(f))[0]
                # Add item to combobox
                widget.addItem(model_name,
                               full_path)
                if model_name == currently_selected_model_name:
                    # This model was selected before clearing the
                    # QComboBox, so reselect
                    widget.setCurrentIndex(index)

        # Get selected engine
        engine = self.ui.comboBox_engine.currentText()
        # Generate paths
        instrumental_folder = os.path.join(ResourcePaths.modelsDir, engine, ResourcePaths.instrumentalDirName)
        stacked_folder = os.path.join(ResourcePaths.modelsDir, engine, ResourcePaths.stackedDirName)

        # Fill Comboboxes
        fill_model_comboBox(widget=self.ui.comboBox_instrumental,
                            folder=instrumental_folder)
        fill_model_comboBox(widget=self.ui.comboBox_stacked,
                            folder=stacked_folder)

    # Shortcuts Page
    def update_page_shortcuts(self):
        """
        Update values and states of all widgets in the
        shortcuts page
        """
        pass

    def save_page_shortcuts(self):
        """
        Save the values of the widgets in the
        shortcuts page
        """

    # Customization Page
    def update_page_customization(self):
        """
        Update values and states of all widgets in the
        customization page
        """
        pass

    def save_page_customization(self):
        """
        Save the values of the widgets in the
        customization page
        """

    # Preferences Page
    def update_page_preferences(self):
        """
        Update values and states of all widgets in the
        preferences page
        """
        # -Command Line-
        # Index:
        # 0 = off
        # 1 = on
        # 2 = debug
        index = self.ui.comboBox_command.currentIndex()

        if index == 0:
            # Adjust window size to size without the textbrowser
            if self.app.windows['main'].ui.textBrowser_command.isVisible():
                win_size = QtCore.QSize(self.app.windows['main'].width() - self.app.windows['main'].ui.textBrowser_command.width(),
                                        self.app.windows['main'].height())
                # Call after 1 ms to prevent window not being resized due to unknown reasons
                # (Workaround)
                QtCore.QTimer.singleShot(1, lambda: self.app.windows['main'].resize(win_size))
            # Hide Textbrowser
            self.app.windows['main'].ui.textBrowser_command.setVisible(False)
        else:
            # Show Textbrowser
            self.app.windows['main'].ui.textBrowser_command.setVisible(True)

        # -Export Directory-
        self.ui.label_exportDirectory.setText(self.exportDirectory)

    def save_page_preferences(self):
        """
        Save the values of the widgets in the
        seperation settings page
        """

    # -Other-
    def decode_modelNames(self):
        """
        Decode the selected model file names and adjust states
        and cosntants of widgets accordingly
        """

    def menu_loadPage(self, index: int):
        """
        Load the given menu page by index and
        adjust minimum size of window

        Parameters:
            index(int):
                0 = Seperation Settings
                1 = Shortcuts
                2 = Customization
                3 = Preferences
        """
        # Load Page
        stackedWidget = self.ui.stackedWidget
        stackedWidget.setCurrentIndex(index)
        # Check Radiobutton
        self.menu_group.button(index).setChecked(True)

        # Find Frame which specifies the minimum width
        page = stackedWidget.currentWidget()
        min_width = page.property('minimumFrameWidth')
        self.ui.frame_14.setMinimumWidth(min_width)

        # Update states and values of widgets
        self.update_window()

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        # -Save the geometry for this window-
        self.settings.beginGroup('settingswindow')
        self.settings.setValue('geometry',
                               self.geometry())
        self.settings.endGroup()
        # Commit Save
        self.settings.sync()
        # -Close Window-
        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.ui.retranslateUi(self)


def run():
    """
    Start the application
    """
    app = CustomApplication()
    sys.exit(app.exec_())

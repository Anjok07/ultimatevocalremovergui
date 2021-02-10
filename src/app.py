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
from .resources.resources_manager import (ResourcePaths, Logger)
from .windows import (mainwindow, settingswindow)
from .inference import converter_v4
# -Other-
# Logging
import logging
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
    # Directory to open when selecting a music file (Default: desktop)
    'inputsDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Export path (Default: desktop)
    'exportDirectory': QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation),
    # Language in format {language}_{country} (Default: system language)
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
    # Process multiple files at once
    'checkBox_multiThreading': False,
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
        self.logger = Logger()
        self.settings = QtCore.QSettings(APPLICATION_SHORTNAME, APPLICATION_NAME)
        # self.settings.clear()
        self.resources = ResourcePaths()
        self.translator = Translator(self)
        self.threadpool = QtCore.QThreadPool(self)
        # -Load Windows-
        # Collection of windows
        self.windows: Dict[str, QtWidgets.QWidget] = {
            'main': MainWindow(self),
            'settings': SettingsWindow(self),
        }
        self.windows['main'].show()

        self.logger.info('--- Setting up application ---',
                         indent_forwards=True)
        self.setup_application()
        # Raise main window
        self.windows['main'].activateWindow()
        self.windows['main'].raise_()
        # self.windows['main'].pushButton_seperate_clicked()
        self.logger.indent_backwards()
        self.logger.info('--- Finished setup ---')

    def setup_application(self):
        """
        Update windows
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

        # -Before-
        # None

        # -Setup-
        setup_windows()
        improve_comboboxes()
        assign_lineEdit_validators()

        # -After-
        # Open settings window on startup
        open_settings = self.settings.value('settingswindow/checkBox_settingsStartup',
                                            DEFAULT_SETTINGS['checkBox_settingsStartup'],
                                            bool)
        if open_settings:
            self.windows['main'].pushButton_settings_clicked()
        # Load language
        language = QtCore.QLocale(self.settings.value('settingswindow/language',
                                                      DEFAULT_SETTINGS['language'])).language()
        self.translator.load_language(language)

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

    def closeAllWindows(self):
        """
        Capture application close to save data
        """
        self.logger.info('--- Closing application ---',
                         indent_forwards=True)
        # --Settings Window--
        self.settings.beginGroup('settingswindow')
        # Widgets
        self.logger.info('Saving application data...')
        setting_widgets = [*self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QCheckBox),
                           *self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QComboBox),
                           *self.windows['settings'].ui.stackedWidget.findChildren(QtWidgets.QLineEdit), ]
        for widget in setting_widgets:
            widgetObjectName = widget.objectName()
            if not widgetObjectName in DEFAULT_SETTINGS:
                if not widgetObjectName:
                    # Empty object name no need to notify
                    continue
                # Default settings do not exist
                self.logger.warn(f'"{widgetObjectName}"; {widget.__class__} does not have a default setting!')
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

        self.logger.info('Closing windows...')
        super().closeAllWindows()
        self.logger.indent_backwards()
        self.logger.info('--- Done! ---')


class Translator:
    def __init__(self, app: CustomApplication):
        self.app = app
        self.logger = app.logger
        self.loaded_language: str
        self._translator = QtCore.QTranslator(self.app)

    def load_language(self, language: QtCore.QLocale.Language = QtCore.QLocale.English):
        """
        Load specified language by file name

        Default is english
        """
        language_str = QtCore.QLocale.languageToString(language).lower()
        self.logger.info(f'Translating to {language_str}...',
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
            return
        # get language name to later store in settings
        self.loaded_language = QtCore.QLocale(language).name()
        # Load language
        if language == QtCore.QLocale.English:
            # English is base language so remove translator
            self.app.removeTranslator(self._translator)
        else:
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

        self.logger.indent_backwards()


class MainWindow(QtWidgets.QWidget):
    def __init__(self, app: CustomApplication):
        # -Window setup-
        super(MainWindow, self).__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
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
            default_size = self.size()
            default_pos = QtCore.QPoint()
            default_pos.setX((self.app.primaryScreen().size().width() / 2) - default_size.width() / 2)
            default_pos.setY((self.app.primaryScreen().size().height() / 2) - default_size.height() / 2)
            # Get geometry
            size = self.settings.value('mainwindow/size',
                                       default_size)
            pos = self.settings.value('mainwindow/pos',
                                      default_pos)
            self.resize(size)
            self.move(pos)

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
            self.ui.frame_musicFiles.dragEnterEvent = self.frame_musicFiles_dragEnterEvent
            self.ui.frame_musicFiles.dropEvent = self.frame_musicFiles_dropEvent
            # -Pushbuttons-
            self.ui.pushButton_settings.clicked.connect(self.pushButton_settings_clicked)
            self.ui.pushButton_seperate.clicked.connect(self.pushButton_seperate_clicked)

        def create_animation_objects():
            """
            Create the animation objects that are used
            multiple times here
            """
            def style_progressbar():
                """
                Style pogressbar manually as when styled in Qt Designer
                a bug occurs that prevents smooth animation of progressbar
                """
                self.ui.progressBar.setStyleSheet("""QProgressBar:horizontal {
                    border: 0px solid gray;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0.0795455 rgba(33, 147, 176, 255), stop:1 rgba(109, 213, 237, 255));
                    border-top-right-radius: 2px;
                    border-bottom-right-radius: 2px;
                } """)
                self.seperation_update_progress(0)
            self.pbar_animation = QtCore.QPropertyAnimation(self.ui.progressBar, b"value",
                                                            parent=self)
            # This is all to prevent the progressbar animation not working propertly
            self.pbar_animation.setDuration(8)
            self.pbar_animation.setStartValue(0)
            self.pbar_animation.setEndValue(8)
            self.pbar_animation.start()
            self.pbar_animation.setDuration(500)
            QtCore.QTimer.singleShot(1000, lambda: style_progressbar())

        # -Before setup-
        self.logger.info('Main -> Setting up',
                         indent_forwards=True)
        # Load saved settings for widgets
        self._load_data()
        # Create WinTaskbar
        self.winTaskbar = QWinTaskbarButton(self)
        self.winTaskbar.setWindow(self.windowHandle())
        self.winTaskbar_progress = self.winTaskbar.progress()

        # -Setup-
        load_geometry()
        load_images()
        bind_widgets()
        create_animation_objects()

        # -After setup-
        # Create instance
        self.vocalRemoverRunnable = converter_v4.VocalRemoverWorker(logger=self.logger)
        # Bind events
        self.vocalRemoverRunnable.signals.start.connect(self.seperation_start)
        self.vocalRemoverRunnable.signals.message.connect(self.seperation_write)
        self.vocalRemoverRunnable.signals.progress.connect(self.seperation_update_progress)
        self.vocalRemoverRunnable.signals.error.connect(self.seperation_error)
        self.vocalRemoverRunnable.signals.finished.connect(self.seperation_finish)
        # Late update
        self.update_window()
        self.logger.indent_backwards()

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

    # -Widget Binds-
    def pushButton_settings_clicked(self):
        """
        Open the settings window
        """
        self.logger.info('Opening settings window...')
        # Reshow window
        self.app.windows['settings'].setWindowState(Qt.WindowNoState)
        self.app.windows['settings'].show()
        # Focus window
        self.app.windows['settings'].activateWindow()
        self.app.windows['settings'].raise_()

    def pushButton_seperate_clicked(self):
        """
        Seperate given files
        """
        # -Extract seperation info from GUI-
        self.app.logger.info('Seperation button pressed')
        seperation_data = self.app.extract_seperation_data()
        self.vocalRemoverRunnable.seperation_data = seperation_data.copy()
        # Start seperation
        self.app.threadpool.start(self.vocalRemoverRunnable)

    def frame_musicFiles_dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Check whether the files the user is dragging over the widget
        is valid or not
        """
        if event.mimeData().urls():
            # URLs dragged
            event.accept()
        else:
            event.ignore()

    def frame_musicFiles_dropEvent(self, event: QtGui.QDropEvent):
        """
        Assign dropped paths to list
        """
        urls = event.mimeData().urls()
        self.logger.info(f'Selected {len(urls)} files',
                         indent_forwards=True)
        inputPaths = []
        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path):
                inputPaths.append(path)

            self.logger.info(repr(path))
        self.inputPaths = inputPaths
        self.update_window()
        self.logger.indent_backwards()

    # -Seperation Methods-
    def seperation_start(self):
        """
        Seperation has started
        """
        # Disable Seperation Button
        self.ui.pushButton_seperate.setEnabled(False)
        # Setup WinTaskbar
        self.winTaskbar.setOverlayAccessibleDescription('Seperating...')
        self.winTaskbar.setOverlayIcon(QtGui.QIcon(ResourcePaths.images.folder))
        self.winTaskbar_progress.setVisible(True)

    def seperation_write(self, text: str):
        """
        Write to GUI
        """
        self.logger.info(text)
        self.write_to_command(text)

    def seperation_update_progress(self, progress: int):
        """
        Update both progressbars in Taskbar and GUI
        with the given progress
        """
        # self.logger.info(f'Updating progress: {progress}%')
        # # Given progress is (0-100) but (0-200) is needed
        progress *= 2
        cur_progress = self.ui.progressBar.value()
        self.pbar_animation.stop()
        self.pbar_animation.setStartValue(cur_progress)
        self.pbar_animation.setEndValue(progress)
        self.pbar_animation.start()
        self.winTaskbar_progress.setValue(progress)

    def seperation_error(self, message: Tuple[str, str]):
        """
        Error occured while seperating

        Parameters:
            message(tuple):
                Index 0: Error Message
                Index 1: Detailed Message
        """
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('An Error Occurred')
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg.setText(message[0] + '\n\nIf the issue is not clear, please contact the creator and attach a screenshot of the detailed message with the file and settings that caused it!')  # nopep8
        msg.setDetailedText(message[1])
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.setWindowFlag(Qt.WindowStaysOnTopHint)
        msg.exec_()

        self.seperation_finish(failed=True)

    def seperation_finish(self, elapsed_time: str = 'N/A', failed: bool = False):
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
            QtCore.QTimer.singleShot(2500, lambda: self.ui.progressBar.setValue(0))

        # -Create MessageBox-
        if failed:
            # Error message was already displayed in the seperation_error function
            self.logger.warn(msg=f'----- The seperation has failed! -----')
        else:
            self.logger.info(msg=f'----- The seperation has finished! (in {elapsed_time}) -----')
            if self.app.windows['settings'].ui.checkBox_notifiyOnFinish.isChecked():
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle('Seperation Complete')
                msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
                msg.setText(f"UVR:\nYour seperation has finished!\n\nTime elapsed: {elapsed_time}")  # nopep8
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.setWindowFlag(Qt.WindowStaysOnTopHint)
                # Focus main window
                self.activateWindow()
                self.raise_()
                # Show messagebox
                msg.exec_()
            else:
                # Highlight window in taskbar
                self.app.alert(self)

        # -Reset progress-
        seperation_reset()
        # -Enable Seperation Button
        self.ui.pushButton_seperate.setEnabled(True)

    # -Other Methods-
    def update_window(self):
        """
        Update the text and states of widgets
        in this window
        """
        self.logger.info('Updating main window...',
                         indent_forwards=True)

        if self.inputPaths:
            self.listWidget_musicFiles_update()
            self.ui.listWidget_musicFiles.setVisible(True)
            self.ui.pushButton_musicFiles.setVisible(False)
        else:
            self.ui.listWidget_musicFiles.setVisible(False)
            self.ui.pushButton_musicFiles.setVisible(True)
        self.logger.indent_backwards()

    def listWidget_musicFiles_update(self):
        """
        Write to the list view
        """
        self.ui.listWidget_musicFiles.clear()
        self.ui.listWidget_musicFiles.addItems(self.inputPaths)
        self.ui.listWidget_musicFiles.setFixedSize(self.ui.listWidget_musicFiles.sizeHintForColumn(0) + 2 * self.ui.listWidget_musicFiles.frameWidth(
        ), self.ui.listWidget_musicFiles.sizeHintForRow(0) * self.ui.listWidget_musicFiles.count() + 2 * self.ui.listWidget_musicFiles.frameWidth())

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
        self.settings.setValue('mainwindow/size',
                               self.size())
        self.settings.setValue('mainwindow/pos',
                               self.pos())
        # Commit Save
        self.settings.sync()
        # -Close all windows-
        self.app.closeAllWindows()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.logger.info('Main: Retranslating UI')
        self.ui.retranslateUi(self)


class SettingsWindow(QtWidgets.QWidget):
    def __init__(self, app: CustomApplication):
        super(SettingsWindow, self).__init__()
        self.ui = settingswindow.Ui_SettingsWindow()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
        self.settings = QtCore.QSettings(APPLICATION_SHORTNAME, APPLICATION_NAME)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.settings))

        # -Other Variables-
        self.menu_update_methods = {
            0: self.update_page_seperationSettings,
            1: self.update_page_shortcuts,
            2: self.update_page_customization,
            3: self.update_page_preferences,
        }
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
        self.logger.info('Settings -> Setting up',
                         indent_forwards=True)

        def load_geometry():
            """
            Load the geometry of this window
            """
            # Window is centered on primary window
            default_size = self.size()
            default_pos = QtCore.QPoint()
            default_pos.setX((self.app.primaryScreen().size().width() / 2) - default_size.width() / 2)
            default_pos.setY((self.app.primaryScreen().size().height() / 2) - default_size.height() / 2)
            # Get geometry
            size = self.settings.value('settingswindow/size',
                                       default_size)
            pos = self.settings.value('settingswindow/pos',
                                      default_pos)
            self.resize(size)
            self.move(pos)

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
        # Load saved settings for widgets
        self._load_data()
        # Update available model lists
        self._update_selectable_models()
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
        # Clear command
        self.pushButton_clearCommand_clicked()
        # Load menu (Preferences)
        self.menu_loadPage(3)
        self.update_window()
        self.logger.indent_backwards()

    def _load_data(self, default: bool = False):
        """
        Load the data for this window

        (Only run right after window initialization or to reset settings)

        Parameters:
            default(bool):
                Reset to the default settings
        """
        self.logger.info('Loading data...',
                         indent_forwards=True)
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
                self.logger.warn(f'"{widgetObjectName}"; {widget.__class__} does not have a default setting!')
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
        self.logger.indent_backwards()

    # -Widget Binds-
    def pushButton_clearCommand_clicked(self):
        """
        Clear the command line and append
        """
        self.logger.info('Clearing Command Line...')
        self.ui.pushButton_clearCommand.setVisible(False)
        self.app.windows['main'].ui.textBrowser_command.clear()

        index = self.ui.comboBox_command.currentIndex()
        if index:
            current_date = dt.datetime.now().strftime("%H:%M:%S")
            self.app.windows['main'].ui.textBrowser_command.append(f'Command Line [{current_date}]')

    def pushButton_exportDirectory_clicked(self):
        """
        Let user select an export directory for the seperation
        """
        self.logger.info('Selecting Export Directory...',
                         indent_forwards=True)
        # Ask for directory
        filepath = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                              caption='Select Export Directory',
                                                              dir=self.exportDirectory,
                                                              )

        if not filepath:
            # No directory specified
            self.logger.info('No file selected!',)
            self.logger.indent_backwards()
            return
        # Update export path value
        self.logger.info(f'Selected Path: "{filepath}"',)
        self.logger.indent_backwards()
        self.exportDirectory = filepath
        self.update_page_preferences()

    def pushButton_resetDefault_clicked(self):
        """
        Reset settings to default
        """
        self.logger.info('Resetting to default settings...',
                         indent_forwards=True)
        self._load_data(default=True)
        self.logger.indent_backwards()

    def comboBox_command_currentIndexChanged(self):
        """
        Changed mode for command line
        """
        self.logger.info('Changing Command mode...',
                         indent_forwards=True)
        self.update_page_preferences()
        self.pushButton_clearCommand_clicked()
        self.logger.indent_backwards()

    # -Update Methods-
    # Whole window (All widgets)
    def update_window(self):
        """
        Update the values and states of all widgets
        in this window
        """
        self.logger.info('Updating settings window...',
                         indent_forwards=True)
        self.update_page_seperationSettings()
        self.update_page_shortcuts()
        self.update_page_customization()
        self.update_page_preferences()
        self.logger.indent_backwards()

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

        self.logger.info('Updating: "Seperation Settings" page',
                         indent_forwards=True)
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
        self.logger.indent_backwards()

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
        self.logger.info('Updating: "Shortcuts" page',
                         indent_forwards=True)
        self.logger.indent_backwards()

    # Customization Page
    def update_page_customization(self):
        """
        Update values and states of all widgets in the
        customization page
        """
        self.logger.info('Updating: "Customization" page',
                         indent_forwards=True)
        self.logger.indent_backwards()

    # Preferences Page
    def update_page_preferences(self):
        """
        Update values and states of all widgets in the
        preferences page
        """
        self.logger.info('Updating: "Preferences" page',
                         indent_forwards=True)
        # -Command Line-
        # Index:
        # 0 = off
        # 1 = on
        index = self.ui.comboBox_command.currentIndex()

        if not index:
            # Hide Textbrowser
            self.app.windows['main'].ui.textBrowser_command.setVisible(False)
        else:
            # Show Textbrowser
            self.app.windows['main'].ui.textBrowser_command.setVisible(True)

        # -Export Directory-
        self.ui.label_exportDirectory.setText(self.exportDirectory)
        self.logger.indent_backwards()

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
        self.logger.info(f'Loading page with index {index}',
                         indent_forwards=True)
        # Load Page
        stackedWidget = self.ui.stackedWidget
        stackedWidget.setCurrentIndex(index)
        # Check Radiobutton
        self.menu_group.button(index).setChecked(True)

        # Find Frame which specifies the minimum width
        page = stackedWidget.currentWidget()
        min_width = page.property('minimumFrameWidth')
        self.ui.frame_14.setMinimumWidth(min_width)

        # Update page based on index
        self.menu_update_methods[index]()
        self.logger.indent_backwards()

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        # -Save the geometry for this window-
        self.settings.setValue('settingswindow/size',
                               self.size())
        self.settings.setValue('settingswindow/pos',
                               self.pos())
        # Commit Save
        self.settings.sync()
        # -Close Window-
        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.logger.info('Settings: Retranslating UI')
        self.ui.retranslateUi(self)


def run():
    """
    Start the application
    """
    app = CustomApplication()
    sys.exit(app.exec_())

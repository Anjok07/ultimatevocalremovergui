# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
from PySide2.QtWinExtras import (QWinTaskbarButton)
from PySide2 import QtMultimedia
# -Root imports-
from ..resources.resources_manager import ResourcePaths
from ..inference import converter_v4
from .. import constants as const
from .design import mainwindow_ui
# -Other-
# System
import os
# Code annotation
from typing import (Tuple,)


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


class AudioPlayer(QtMultimedia.QMediaPlayer):
    def __init__(self, parent, wig_play_pause: QtWidgets.QPushButton, wig_slider: QtWidgets.QSlider):
        super().__init__(parent=parent)
        self.logger = parent.logger
        self.wig_play_pause = wig_play_pause
        self.wig_slider = wig_slider
        self.last_state = self.state()

        # -Images-
        self.play_img = QtGui.QPixmap(ResourcePaths.images.audio_play)
        self.pause_img = QtGui.QPixmap(ResourcePaths.images.audio_pause)
        self.wig_play_pause.setIconSize(QtCore.QSize(23, 23))
        self.pause()

        # -Binds-
        # Player
        self.error.connect(self.error_occurred)
        self.durationChanged.connect(self.update_slider_max)
        self.positionChanged.connect(self.update_slider)
        # Widgets
        self.wig_play_pause.pressed.connect(self.play_or_pause)
        self.wig_slider.sliderPressed.connect(self.sliderPressed)
        self.wig_slider.sliderReleased.connect(self.sliderReleased)

    def play(self):
        self.wig_play_pause.setIcon(self.pause_img)

        super().play()

    def pause(self):
        self.wig_play_pause.setIcon(self.play_img)
        super().pause()

    def play_or_pause(self):
        if self.state() != QtMultimedia.QMediaPlayer.PlayingState:
            # Not playing -> Play
            self.play()
        else:
            # Playing -> Pause
            self.pause()

    def sliderPressed(self):
        self.last_state = self.state()
        self.pause()

    def sliderReleased(self):
        if self.last_state == QtMultimedia.QMediaPlayer.PlayingState:
            self.play()
        self.setPosition(self.wig_slider.value())

    def set_song(self, path: str):
        self.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(path)))

    def update_slider_max(self, duration):
        self.wig_slider.setMaximum(duration)

    def update_slider(self, position):
        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.wig_slider.blockSignals(True)
        self.wig_slider.setValue(position)
        self.wig_slider.blockSignals(False)

    def error_occurred(self, *args):
        self.logger.info(args)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, app: QtWidgets.QApplication):
        # -Window setup-
        super(MainWindow, self).__init__()
        self.ui = mainwindow_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
        self.settings = QtCore.QSettings(const.APPLICATION_SHORTNAME, const.APPLICATION_NAME)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.icon))

        # -Other Variables-
        # Independent data
        self.inputPaths = self.settings.value('inputPaths',
                                              const.DEFAULT_SETTINGS['inputPaths'],
                                              type=list)

        self.instrumentals_audioPlayer = AudioPlayer(self,
                                                     self.ui.pushButton_play_instrumentals,
                                                     self.ui.horizontalSlider_instrumentals,)
        self.vocals_audioPlayer = AudioPlayer(self,
                                              self.ui.pushButton_play_vocals,
                                              self.ui.horizontalSlider_vocals)

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
            self.ui.stackedWidget_musicFiles.dragEnterEvent = self.stackedWidget_musicFiles_dragEnterEvent
            self.ui.stackedWidget_musicFiles.dropEvent = self.stackedWidget_musicFiles_dropEvent
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

    def stackedWidget_musicFiles_dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Check whether the files the user is dragging over the widget
        is valid or not
        """
        if event.mimeData().urls():
            # URLs dragged
            event.accept()
        else:
            event.ignore()

    def stackedWidget_musicFiles_dropEvent(self, event: QtGui.QDropEvent):
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
        # Media player
        self.deactivate_audio_players()

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
        # -Activate Audio Players-
        QtCore.QTimer.singleShot(1000, lambda: self.activate_audio_players())
        # -Enable Seperation Button
        self.ui.pushButton_seperate.setEnabled(True)

    def activate_audio_players(self):
        self.ui.stackedWidget_instrumentals.setCurrentIndex(1)
        self.ui.stackedWidget_vocals.setCurrentIndex(1)
        self.instrumentals_audioPlayer.set_song(ResourcePaths.temp_instrumental)
        self.vocals_audioPlayer.set_song(ResourcePaths.temp_vocal)

    def deactivate_audio_players(self):
        self.ui.stackedWidget_instrumentals.setCurrentIndex(0)
        self.ui.stackedWidget_vocals.setCurrentIndex(0)
        self.instrumentals_audioPlayer.stop()
        self.vocals_audioPlayer.stop()
        # Remove all connections to previous media content
        self.instrumentals_audioPlayer.setMedia(QtMultimedia.QMediaContent())
        self.vocals_audioPlayer.setMedia(QtMultimedia.QMediaContent())

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
            self.ui.stackedWidget_musicFiles.setCurrentIndex(1)
        else:
            self.ui.stackedWidget_musicFiles.setCurrentIndex(0)
        self.logger.indent_backwards()

    def listWidget_musicFiles_update(self):
        """
        Write to the list view
        """
        self.ui.listWidget_musicFiles.clear()
        self.ui.listWidget_musicFiles.addItems(self.inputPaths)

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

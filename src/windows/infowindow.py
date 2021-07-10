# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
from PySide2 import QtMultimedia
# -Root imports-
from ..resources.resources_manager import (ResourcePaths)
from ..inference import converter
from ..app import CustomApplication
from .. import constants as const
from .design import infowindow_ui
# -Other-
import subprocess
# System
import os
# Code annotation
from typing import (Tuple, Optional)


class InfoWindow(QtWidgets.QWidget):
    """
    Main Window of UVR where seperation, progress and embedded seperated song-playing takes place
    """

    def __init__(self, app: CustomApplication):
        # -Window setup-
        super(InfoWindow, self).__init__()
        self.ui = infowindow_ui.Ui_InfoWindow()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
        self.settings = self.app.settings

        # -Other Variables-

    # -Widget Binds-

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
            # Get data
            self.settings.beginGroup(self.__class__.__name__.lower())
            size = self.settings.value('size',
                                       default_size)
            pos = self.settings.value('pos',
                                      default_pos)
            isMaximized = self.settings.value('isMaximized',
                                              False,
                                              type=bool)
            self.settings.endGroup()
            # Apply data
            self.move(pos)
            if isMaximized:
                self.setWindowState(Qt.WindowMaximized)
            else:
                self.resize(size)

        def load_images():
            """
            Load the images for this window and assign them to their widgets
            """

        def bind_widgets():
            """
            Bind the widgets here
            """

        # -Before setup-
        self.logger.info('InfoWindow -> Setting up',
                         indent_forwards=True)
        # -Setup-
        load_geometry()
        load_images()
        bind_widgets()

        # -After setup-
        # Late update
        self.update_window()
        self.logger.indent_backwards()

    # -Other Methods-
    def update_window(self):
        """
        Update the text and states of widgets
        in this window
        """
        self.logger.info('Updating info window...',
                         indent_forwards=True)

        self.logger.indent_backwards()

    def save_window(self):
        """Save window

        Save states of the widgets in this window
        """

    def update_info(self, title: str, text: str):
        self.setWindowTitle(title)
        self.ui.textEdit.setMarkdown(text)

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        # -Close all windows-
        event.accept()
        # -Save the geometry for this window-
        self.settings.beginGroup(self.__class__.__name__.lower())
        self.settings.setValue('size',
                               self.size())
        self.settings.setValue('pos',
                               self.pos())
        self.settings.setValue('isMaximized',
                               self.isMaximized())
        self.settings.endGroup()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.logger.info('InfoWindow: Retranslating UI')
        self.ui.retranslateUi(self)

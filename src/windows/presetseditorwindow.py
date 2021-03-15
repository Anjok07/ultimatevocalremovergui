# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
# -Root imports-
from ..resources.resources_manager import (ResourcePaths)
from ..app import CustomApplication
from .. import constants as const
from .design import presetseditorwindow_ui
# -Other-
# File saving
from collections import OrderedDict
import json
# System
import os
import sys
# Code annotation
from typing import (Tuple, Optional)


class PresetsEditorWindow(QtWidgets.QWidget):
    """
    Window for editing presets for the seperation settings
    """
    PRESET_PREFIX = 'UVR_'

    def __init__(self, app: CustomApplication):
        # -Window setup-
        super(PresetsEditorWindow, self).__init__()
        self.ui = presetseditorwindow_ui.Ui_PresetsEditor()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
        self.settings = QtCore.QSettings(const.APPLICATION_SHORTNAME, const.APPLICATION_NAME)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # -Other Variables-
        # Independent data
        self.presets_saveDir: str = self.settings.value('user/presets_saveDir',
                                                        const.DEFAULT_SETTINGS['presets_saveDir'])
        self.presets_loadDir: str = self.settings.value('user/presets_loadDir',
                                                        const.DEFAULT_SETTINGS['presets_loadDir'],
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
            default_size = self.size()
            default_pos = QtCore.QPoint()
            default_pos.setX((self.app.primaryScreen().size().width() / 2) - default_size.width() / 2)
            default_pos.setY((self.app.primaryScreen().size().height() / 2) - default_size.height() / 2)
            # Get geometry
            size = self.settings.value('presetseditorwindow/size',
                                       default_size)
            pos = self.settings.value('presetseditorwindow/pos',
                                      default_pos)
            self.resize(size)
            self.move(pos)

        def load_images():
            """
            Load the images for this window and assign them to their widgets
            """
            upload_img = QtGui.QPixmap(ResourcePaths.images.upload)
            download_img = QtGui.QPixmap(":/img/images/download.png")
            self.ui.pushButton_export.setIcon(upload_img)
            self.ui.pushButton_import.setIcon(download_img)
            self.ui.pushButton_export.setIconSize(QtCore.QSize(18, 18))
            self.ui.pushButton_import.setIconSize(QtCore.QSize(18, 18))

        def bind_widgets():
            """
            Bind the widgets here
            """
            self.ui.listWidget_presets.itemChanged.connect(
                lambda: self.app.settingsWindow.update_page_seperationSettings())
            self.ui.pushButton_add.clicked.connect(self.pushButton_add_clicked)
            self.ui.pushButton_delete.clicked.connect(self.pushButton_delete_clicked)
            self.ui.pushButton_export.clicked.connect(self.pushButton_export_clicked)
            self.ui.pushButton_import.clicked.connect(self.pushButton_import_clicked)

        def fill_presetsList():
            """
            Fill the main table in the window
            """
            self.ui.listWidget_presets.clear()
            presets = self.settings.value('user/presets',
                                          const.DEFAULT_SETTINGS['presets'])
            for label, settings in presets.items():
                self.pushButton_add_clicked(label,
                                            settings)

        # -Before setup-

        # -Setup-
        load_geometry()
        load_images()
        fill_presetsList()
        bind_widgets()

        # -After setup-
        # Late update
        self.update_window()

    # -Widget Binds-
    def pushButton_add_clicked(self, label: Optional[str] = None, settings: Optional[dict] = None):
        """
        Add current settings as a preset
        """
        # -Create item-
        item = QtWidgets.QListWidgetItem('')
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setSizeHint(QtCore.QSize(item.sizeHint().width(), 25))
        # Insert item at top and enter edit mode
        self.ui.listWidget_presets.insertItem(0, item)
        # -Obtain data-
        if settings is None:
            settingsManager = self.app.settingsWindow.settingsManager
            # Get current
            settings = settingsManager.get_settings(page_idx=0)
            del settings['comboBox_presets']
            name_to_json = {v: k for k, v in const.JSON_TO_NAME.items()}  # Invert dict
            for widget_objectName in list(settings.keys()):
                json_key = name_to_json[widget_objectName]
                settings[json_key] = settings.pop(widget_objectName)
        if label is None:
            # Generate generic name for preset
            i = self.ui.listWidget_presets.count() + 1
            label = f'Preset {i}'
            # Set into edit mode
            self.ui.listWidget_presets.editItem(item)
        # -Set data-
        item.setText(label)
        item.setData(Qt.UserRole, settings.copy())
        # -Update settings window-
        self.app.settingsWindow.update_page_seperationSettings()

    def pushButton_delete_clicked(self):
        """
        Delete selected presets after asking for
        confirmation
        """
        selected_items = self.ui.listWidget_presets.selectedItems()
        # Some paths already selected
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle(self.tr('Confirmation'))
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg.setText(f'You will delete {len(selected_items)} items. Do you wish to continue?')
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setWindowFlag(Qt.WindowStaysOnTopHint)
        val = msg.exec_()

        if val == QtWidgets.QMessageBox.No:
            # Cancel
            return

        for item in self.ui.listWidget_presets.selectedItems():
            row = self.ui.listWidget_presets.row(item)
            self.ui.listWidget_presets.takeItem(row)

        # -Update settings window-
        self.app.settingsWindow.update_page_seperationSettings()

    def pushButton_export_clicked(self):
        """
        Export selected preset as a json file
        """
        self.logger.info('Exporting Preset...',
                         indent_forwards=True)
        selected_items = self.ui.listWidget_presets.selectedItems()
        if not selected_items:
            # No item selected
            self.logger.info('No item selected')
            self.logger.indent_backwards()
            return

        item = selected_items[0]
        itemText = item.text().replace(' ', '_')
        file_name = f'{self.PRESET_PREFIX}{itemText}.json'
        path = QtWidgets.QFileDialog().getSaveFileName(parent=self,
                                                       caption='Save Preset',
                                                       dir=os.path.join(self.presets_saveDir,
                                                                        file_name),
                                                       filter='JSON File (*.json)',
                                                       )[0]

        if not path:
            # No files specified
            self.logger.info('Canceled preset export!',)
            self.logger.indent_backwards()
            return

        self.presets_saveDir = os.path.dirname(path)

        settings = item.data(Qt.UserRole)
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)

    def pushButton_import_clicked(self):
        """
        Import a .json preset file
        """
        self.logger.info('Importing Preset Files...',
                         indent_forwards=True)
        paths = QtWidgets.QFileDialog.getOpenFileNames(parent=self,
                                                       caption='Select Presets',
                                                       filter='JSON Files (*.json)',
                                                       dir=self.presets_loadDir,
                                                       )[0]

        if not paths:
            # No files specified
            self.logger.info('No presets selected!',)
            self.logger.indent_backwards()
            return
        self.presets_loadDir = os.path.dirname(paths[0])

        for path in paths:
            with open(path, 'r') as f:
                settings = json.load(f, object_pairs_hook=OrderedDict)
            file_name = os.path.splitext(os.path.basename(path))[0]
            file_name = file_name.replace(self.PRESET_PREFIX, '')
            file_name = file_name.replace('_', ' ')

            self.pushButton_add_clicked(label=file_name,
                                        settings=settings)

        self.logger.indent_backwards()

    # -Other Methods-
    def update_window(self):
        """
        Update the text and states of widgets
        in this window
        """

    def save_window(self):
        """Save window

        Save states of the widgets in this window
        """

    def get_presets(self) -> dict:
        """
        Obtain the presets from the window

        (Used for saving)
        """
        presets = {}
        for idx in range(self.ui.listWidget_presets.count()):
            item = self.ui.listWidget_presets.item(idx)
            presets[item.text()] = item.data(Qt.UserRole)

        return presets

    def get_settings(self, name: str):
        """
        Get settings of a preset by name
        """
        presets = self.get_presets()
        if name in presets:
            return presets[name]
        else:
            return {}

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """
        Catch close event of this window to save data
        """
        # -Save the geometry for this window-
        self.settings.setValue('presetseditorwindow/size',
                               self.size())
        self.settings.setValue('presetseditorwindow/pos',
                               self.pos())
        # Commit Save
        self.settings.sync()
        # -Close window-
        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.logger.info('Presets: Retranslating UI')
        self.ui.retranslateUi(self)

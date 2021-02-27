# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
# -Root imports-
from ..inference.lib.model_param_init import ModelParameters
from ..resources.resources_manager import ResourcePaths
from ..app import CustomApplication
from .. import constants as const
from .design import settingswindow_ui
# -Other-
import datetime as dt
# System
import os
# Code annotation
from typing import (Dict, )


class SettingsWindow(QtWidgets.QWidget):
    """
    Settings window for UVR, available sections are:
        - Seperation Settings: Modify settings, like model and ai engine for the seperation of audio files
        - Shortcuts: Set shortcuts to quickly change settings or select other music files/new export directory
        - Customization: Select from different themes for the application
        - Preferences: Change personalised settings like language, export directory,
                       and whether to show the command line
    """

    def __init__(self, app: CustomApplication):
        # -Window setup-
        super(SettingsWindow, self).__init__()
        self.ui = settingswindow_ui.Ui_SettingsWindow()
        self.ui.setupUi(self)
        self.app = app
        self.logger = app.logger
        self.settings = QtCore.QSettings(const.APPLICATION_SHORTNAME, const.APPLICATION_NAME)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.settings))

        # -Other Variables-
        self.pageSwitchTimer = QtCore.QTimer(self)
        self.menu_update_methods = {
            0: self.update_page_seperationSettings,
            1: self.update_page_shortcuts,
            2: self.update_page_customization,
            3: self.update_page_preferences,
        }
        # Independent data
        self.exportDirectory = self.settings.value('settingswindow/exportDirectory',
                                                   const.DEFAULT_SETTINGS['exportDirectory'],
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
            # -Override binds-
            # Music file drag & drop
            self.ui.groupBox_export.dragEnterEvent = self.groupBox_export_dragEnterEvent
            self.ui.groupBox_export.dropEvent = self.groupBox_export_dropEvent
            # -Main buttons-
            # Main control
            self.ui.pushButton_resetDefault.clicked.connect(self.pushButton_resetDefault_clicked)
            # Menu
            self.menu_group.buttonClicked.connect(lambda btn:
                                                  self.menu_loadPage(index=self.menu_group.id(btn)))
            # -Seperation Settings Page-
            self.ui.pushButton_presetsEdit.clicked.connect(self.pushButton_presetsEdit_clicked)
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

        def create_animation_objects():
            """
            Create the animation objects that are used
            multiple times here
            """
            self.effect = QtWidgets.QGraphicsOpacityEffect(self)
            self.effect.setOpacity(1)
            self.ui.stackedWidget.setGraphicsEffect(self.effect)
            # Stackedwidget
            self.pages_ani = QtCore.QPropertyAnimation(self.effect, b'opacity')
            self.pages_ani.setDuration(400)
            # self.pages_ani.setEasingCurve(QtCore.QEasingCurve.OutBack)
            self.pages_ani.setStartValue(1)
            self.pages_ani.setKeyValueAt(0.5, 0)
            self.pages_ani.setEndValue(1)
            self.pages_ani.setLoopCount(1)
            self.pageSwitchTimer.setSingleShot(True)
            self.pageSwitchTimer.setInterval(self.pages_ani.duration() / 2)

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
        # Open settings window on startup
        open_settings = self.settings.value('settingswindow/checkBox_settingsStartup',
                                            const.DEFAULT_SETTINGS['checkBox_settingsStartup'],
                                            bool)

        # -Setup-
        load_geometry()
        load_images()
        bind_widgets()
        create_animation_objects()
        if open_settings:
            self.app.windows['main'].pushButton_settings_clicked()

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
            if not widgetObjectName in const.DEFAULT_SETTINGS:
                if not widgetObjectName:
                    # Empty object name no need to notify
                    continue
                # Default settings do not exist
                self.logger.warn(f'"{widgetObjectName}"; {widget.__class__} does not have a default setting!')
                continue

            # -Finding the instance and loading appropiately-
            if isinstance(widget, QtWidgets.QCheckBox):
                value = self.settings.value(widgetObjectName,
                                            defaultValue=const.DEFAULT_SETTINGS[widgetObjectName],
                                            type=bool)
                widget.setChecked(value)
            elif isinstance(widget, QtWidgets.QComboBox):
                value = self.settings.value(widgetObjectName,
                                            defaultValue=const.DEFAULT_SETTINGS[widgetObjectName],
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
                                            defaultValue=const.DEFAULT_SETTINGS[widgetObjectName],
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
        path = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                          caption='Select Export Directory',
                                                          dir=self.exportDirectory,
                                                          )

        if not path:
            # No directory specified
            self.logger.info('No directory selected!',)
            self.logger.indent_backwards()
            return
        # Update export path value
        self.logger.info(repr(path))
        self.exportDirectory = path
        self.update_page_preferences()

        self.logger.indent_backwards()

    def pushButton_presetsEdit_clicked(self):
        """
        Open the presets editor window
        """
        # Reshow window
        self.app.windows['presetsEditor'].setWindowState(Qt.WindowNoState)
        self.app.windows['presetsEditor'].show()
        # Focus window
        self.app.windows['presetsEditor'].activateWindow()
        self.app.windows['presetsEditor'].raise_()

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

    def groupBox_export_dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """
        Check whether the files the user is dragging over the widget
        is valid or not
        """
        # -Get Path-
        urls = event.mimeData().urls()
        if not urls:
            # No urls given
            event.ignore()
            return
        # Get first given path
        path = urls[0].toLocalFile()
        # Convert to file info
        fileInfo = QtCore.QFileInfo(path)
        if fileInfo.isShortcut():
            # Path is shortcut -> Resolve shortcut
            path = fileInfo.symLinkTarget()
        else:
            # Path is not shortcut
            path = fileInfo.absoluteFilePath()

        # -Check Path-
        if os.path.isdir(path):
            # File is a folder
            event.accept()
        else:
            event.ignore()

    def groupBox_export_dropEvent(self, event: QtGui.QDropEvent):
        """
        Assign dropped paths to list
        """
        self.logger.info('Dragged Export Directory...',
                         indent_forwards=True)
        # -Get Path-
        # Get first given path
        path = event.mimeData().urls()[0].toLocalFile()
        # Convert to file info
        fileInfo = QtCore.QFileInfo(path)
        if fileInfo.isShortcut():
            # Path is shortcut -> Resolve shortcut
            path = fileInfo.symLinkTarget()
        else:
            # Path is not shortcut
            path = fileInfo.absoluteFilePath()

        # -Update path-
        self.logger.info(repr(path))
        self.exportDirectory = path
        self.update_page_preferences()

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
        
        # -Presets subgroup-
        last_text = self.ui.comboBox_presets.currentText()
        self.ui.comboBox_presets.clear()
        for idx in range(self.app.windows['presetsEditor'].ui.listWidget_presets.count()):
            # Loop through every preset in the list on the window
            # Get item by index
            item = self.app.windows['presetsEditor'].ui.listWidget_presets.item(idx)
            # Get text
            text = item.text()
            # Add text to combobox
            self.ui.comboBox_presets.addItem(text)
            if text == last_text:
                self.ui.comboBox_presets.setCurrentText(text)

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
    def get_seperation_settings(self) -> dict:
        """
        Get the currently selected seperation settings

        (Used for presets)
        """
        settings = {
            'instrumentalModel_id': self.get_model_id(self.ui.comboBox_instrumental.currentData()),
            'stackedModel_id': self.get_model_id(self.ui.comboBox_stacked.currentData()),
        }
        return settings

    def set_seperation_settings(self, settings: dict):
        """
        Set the seperation settings

        (Used for presets)
        """

    def get_model_id(self, model_path: str):
        """
        Get the models id

        If no id has been found return the filename
        """
        # print(model_path)
        # model_params = ModelParameters(model_path)
        # print(model_params.param['id'])
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        return model_name

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

        def menu_loadPage():
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

        if not self.ui.checkBox_disableAnimations.isChecked():
            # Animations enabled
            self.pages_ani.start()
            # On half of whole aniamtion loaad new window
            self.pageSwitchTimer.timeout.connect(menu_loadPage)
            self.pageSwitchTimer.start()
        else:
            menu_loadPage()

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
        self.app.windows['presetsEditor'].hide()
        event.accept()

    def update_translation(self):
        """
        Update translation of this window
        """
        self.logger.info('Settings: Retranslating UI')
        self.ui.retranslateUi(self)

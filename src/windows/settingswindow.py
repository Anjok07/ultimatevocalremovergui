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
from collections import OrderedDict
# System
import os
# Code annotation
from typing import (Dict, Union, Optional)


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
        self.settingsManager = SettingsManager(win=self)
        self.setWindowIcon(QtGui.QIcon(ResourcePaths.images.settings))

        # -Other Variables-
        self.menu_update_methods = {
            0: self.update_page_seperationSettings,
            1: self.update_page_shortcuts,
            2: self.update_page_customization,
            3: self.update_page_preferences,
        }
        # Independent data
        self.exportDirectory = self.settings.value('user/exportDirectory',
                                                   const.DEFAULT_SETTINGS['exportDirectory'],
                                                   type=str)
        self.search_for_preset = True

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

    def comboBox_presets_currentIndexChanged(self):
        """
        Changed preset
        """
        self.search_for_preset = False
        name = self.ui.comboBox_presets.currentText()
        settings = self.app.windows['presetsEditor'].get_settings(name)
        for json_key in list(settings.keys()):
            widget_objectName = const.JSON_TO_NAME[json_key]
            settings[widget_objectName] = settings.pop(json_key)
        self.settingsManager.set_settings(settings)
        self.search_for_preset = True

    def frame_export_dragEnterEvent(self, event: QtGui.QDragEnterEvent):
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

    def frame_export_dropEvent(self, event: QtGui.QDropEvent):
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

    def settings_changed(self):
        if self.search_for_preset:
            current_settings = self.settingsManager.get_settings(0)
            presets: dict = self.app.windows['presetsEditor'].get_presets()

            for preset_name, settings in presets.items():
                for json_key, value in settings.items():
                    if (current_settings[const.JSON_TO_NAME[json_key]] != value):
                        break
                else:
                    self.ui.comboBox_presets.setCurrentText(preset_name)
                    break
            else:
                self.ui.comboBox_presets.setCurrentIndex(0)

    # -Window Setup Methods-

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
            # Get data
            self.settings.beginGroup(self.__class__.__name__.lower())
            size = self.settings.value('size',
                                       default_size)
            pos = self.settings.value('pos',
                                      default_pos)
            self.settings.endGroup()
            # Apply data
            self.move(pos)
            self.resize(size)

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
            def bind_settings_changed():
                """
                Bind all the widgets that affect the preset to the
                settings changed method
                """
                for widget_objectName in const.JSON_TO_NAME.values():
                    widget = self.findChild(QtCore.QObject, widget_objectName)
                    if isinstance(widget, QtWidgets.QCheckBox):
                        widget.stateChanged.connect(self.settings_changed)
                    elif isinstance(widget, QtWidgets.QComboBox):
                        widget.currentTextChanged.connect(self.settings_changed)
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.textChanged.connect(self.settings_changed)
                    elif (isinstance(widget, QtWidgets.QDoubleSpinBox) or
                            isinstance(widget, QtWidgets.QSpinBox)):
                        widget.valueChanged.connect(self.settings_changed)
                    else:
                        raise TypeError('Invalid widget type that is not supported!\nWidget: ', widget)

            # -Override binds-
            # Music file drag & drop
            self.ui.frame_export.dragEnterEvent = self.frame_export_dragEnterEvent
            self.ui.frame_export.dropEvent = self.frame_export_dropEvent
            # -Main buttons-
            # Menu
            self.menu_group.buttonClicked.connect(lambda btn:
                                                  self.menu_loadPage(page_idx=self.menu_group.id(btn)))
            # -Seperation Settings Page-
            self.ui.pushButton_presetsEdit.clicked.connect(self.pushButton_presetsEdit_clicked)
            # Comboboxes
            self.ui.comboBox_instrumental.currentIndexChanged.connect(self.update_page_seperationSettings)
            self.ui.comboBox_vocal.currentIndexChanged.connect(self.update_page_seperationSettings)
            self.ui.comboBox_presets.currentIndexChanged.connect(self.comboBox_presets_currentIndexChanged)
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

            self.ui.radioButton_lightTheme.clicked.connect(lambda: self.app.themeManager.load_theme('light'))
            self.ui.radioButton_darkTheme.clicked.connect(lambda: self.app.themeManager.load_theme('dark'))

            bind_settings_changed()

        def create_animation_objects():
            """
            Create the animation objects that are used
            multiple times here
            """

        # -Before setup-
        self.search_for_preset = False
        # Load saved settings for widgets
        self.settingsManager.load_window()
        # Update available model lists
        self._update_selectable_models()
        # Update available model lists
        # Connect button group for menu together
        self.menu_group = QtWidgets.QButtonGroup(self)  # Menu group
        self.menu_group.addButton(self.ui.radioButton_separationSettings,
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
        self.update_window()
        self.menu_loadPage(0, True)
        self.search_for_preset = True
        self.settings_changed()
        self.logger.indent_backwards()

    def load_window(self):
        """Load window

        Load states of the widgets in this window
        from the settings
        """
        self.settingsManager.load_window()

    def save_window(self):
        """Save window

        Save states of the widgets in this window
        """
        self.settingsManager.save_window()

    # -Update Methods-
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

    # Seperation Settings Page
    def update_page_seperationSettings(self):
        """
        Update values and states of all widgets in the
        seperation settings page
        """
        self.logger.info('Updating: "Seperation Settings" page',
                         indent_forwards=True)
        # -Presets subgroup-
        self.ui.comboBox_presets.blockSignals(True)
        last_text = self.ui.comboBox_presets.currentText()
        self.ui.comboBox_presets.clear()
        self.ui.comboBox_presets.addItem('Custom')
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
        self.ui.comboBox_presets.blockSignals(False)
        self.settings_changed()

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

        # Fill Comboboxes
        fill_model_comboBox(widget=self.ui.comboBox_instrumental,
                            folder=ResourcePaths.instrumentalModelsDir)
        fill_model_comboBox(widget=self.ui.comboBox_vocal,
                            folder=ResourcePaths.vocalModelsDir)

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
    def menu_loadPage(self, page_idx: int, force: bool = False):
        """Load the given menu page by index

        Note:
            Also adjust the minimum size of the window
            based on the stored minimum width in the page

            If the page is already loaded, the function to
            load the menu will not be executed (see force)

        Args:
            page_idx (int):
                Which page to load.

                0 - Seperation Settings Page
                1 - Shortcuts Page
                2 - Customization Page
                3 - Preferences Page
            force (bool):
                Force the menu load
        """
        self.logger.info(f'Loading page with index {page_idx}',
                         indent_forwards=True)

        def menu_loadPage():
            # Load Page
            stackedWidget = self.ui.stackedWidget
            stackedWidget.setCurrentIndex(page_idx)
            # Check Radiobutton
            self.menu_group.button(page_idx).setChecked(True)

            # Get specified minimum width from page
            page = stackedWidget.currentWidget()
            min_width = page.property('minimumFrameWidth')
            self.ui.frame_14.setMinimumWidth(min_width)

            # Update page based on index
            self.menu_update_methods[page_idx]()

        if (self.ui.stackedWidget.currentIndex() == page_idx and
                not force):
            # Trying to load same page
            self.logger.info('Skipping load -> page already loaded')
            self.logger.indent_backwards()
            return

        menu_loadPage()

        self.logger.indent_backwards()

    # -Overriden methods-
    def closeEvent(self, event: QtCore.QEvent):
        """Catch close event of this window to save data"""
        # -Save the geometry for this window-
        self.settings.beginGroup(self.__class__.__name__.lower())
        self.settings.setValue('size',
                               self.size())
        self.settings.setValue('pos',
                               self.pos())
        self.settings.endGroup()
        # Commit Save
        self.settings.sync()
        # -Close Window-
        # Hide the presets editor window (child window)
        self.app.presetsEditorWindow.hide()
        event.accept()

    def update_translation(self):
        """Update translation of this window"""
        self.logger.info('Settings: Retranslating UI')
        self.ui.retranslateUi(self)


class SettingsManager:
    """Manage the states of the widgets in the SettingsWindow

    Attributes:
        win (SettingsWindow): Settings window that is being managed
        save_widgets (dict): Configurable widgets that the window contains
            Key - Page number
            Value - Widgets
    """

    def __init__(self, win: SettingsWindow):
        """Inits SettingsManager

        Args:
            win (SettingsWindow): Settings window that is to be managed
        """
        self.win = win
        self.save_widgets = {
            0: [],
            1: [],
            2: [],
            3: [],
        }
        self.fill_save_widgets()

    def fill_save_widgets(self):
        """Update the save_widgets variable

        Assign all instances of the widgets on the
        settings window to their corresponding page.

        Note:
            Only run right after window initialization
        """
        # Get widgets
        seperation_settings_widgets = [
            # -Conversion-
            # Checkbox
            self.win.ui.checkBox_gpuConversion,
            self.win.ui.checkBox_tta,
            self.win.ui.checkBox_modelFolder,
            self.win.ui.checkBox_outputImage,
            self.win.ui.checkBox_postProcess,
            self.win.ui.checkBox_deepExtraction,
            # -Models-
            # Combobox
            self.win.ui.comboBox_instrumental,
            self.win.ui.comboBox_vocal,
            self.win.ui.comboBox_winSize,
            # SpinBox
            self.win.ui.doubleSpinBox_aggressiveness,
            # -Presets-
            # Combobox
            self.win.ui.comboBox_presets, ]
        shortcuts_widgets = []
        customization_widgets = [
            # self.win.ui.radioButton_lightTheme,
            # self.win.ui.radioButton_darkTheme,
        ]
        preferences_widgets = [
            # -Settings-
            # Checkbox
            self.win.ui.checkBox_notifiyOnFinish,
            self.win.ui.checkBox_notifyUpdates,
            self.win.ui.checkBox_settingsStartup,
            self.win.ui.checkBox_disableAnimations,
            self.win.ui.checkBox_disableShortcuts,
            self.win.ui.checkBox_multithreading,
            # Combobox
            self.win.ui.comboBox_command,
            # -Export Settings-
            # Checkbox
            self.win.ui.checkBox_autoSaveInstrumentals,
            self.win.ui.checkBox_autoSaveVocals, ]

        # Assign to save_widgets
        self.save_widgets[0] = seperation_settings_widgets
        self.save_widgets[1] = shortcuts_widgets
        self.save_widgets[2] = customization_widgets
        self.save_widgets[3] = preferences_widgets

    def get_settings(self, page_idx: Optional[int] = None) -> Dict[str, Union[bool, str, float]]:
        """Obtain states of the widgets

        Args:
            page_idx (Optional[int], optional):
                Which page to load the widgets from to get the settings.
                Defaults to None.

                0 - Seperation Settings Page
                1 - Shortcuts Page
                2 - Customization Page
                3 - Preferences Page
                None - All widgets

        Raises:
            TypeError: Invalid widget type in the widgets (has to be either: QCheckBox, QRadioButton, QLineEdit or QComboBox)

        Returns:
            Dict[str, Union[bool, str, float]]: Widget states
                Key - Widget object name
                Value - State of the widget
        """
        settings = OrderedDict()

        save_widgets = self.get_widgets(page_idx=page_idx)

        for widget in save_widgets:
            # Get value
            if (isinstance(widget, QtWidgets.QCheckBox) or
                    isinstance(widget, QtWidgets.QRadioButton)):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
            elif (isinstance(widget, QtWidgets.QDoubleSpinBox) or
                    isinstance(widget, QtWidgets.QSpinBox)):
                value = widget.value()
            else:
                raise TypeError('Invalid widget type that is not supported!\nWidget: ', widget)

            # Set value
            settings[widget.objectName()] = value

        return settings

    def set_settings(self, settings: Dict[str, Union[bool, str, float]]):
        """Update states of the widgets

        The given dict's key should be the widgets object name
        and its value a valid state for that widget.

        Note:
            settings arg info:
                Key - Widget object name
                Value - State of widget

                There are given expected value types for the
                settings argument depending on the widgets type:

                    QCheckBox - bool
                    QRadioButton - bool
                    QLineEdit - str
                    QComboBox - str


        Args:
            settings (Dict[str, Union[bool, str, float]]): States of the widgets to update

        Raises:
            TypeError: Invalid widget type in the widgets (has to be either: QCheckBox, QRadioButton, QLineEdit or QComboBox)
        """
        for widget_objectName, value in settings.items():
            # Get widget
            widget = self.win.findChild(QtCore.QObject, widget_objectName)

            # Set value
            if (isinstance(widget, QtWidgets.QCheckBox) or
                    isinstance(widget, QtWidgets.QRadioButton)):
                widget.setChecked(value)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QtWidgets.QComboBox):
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
            elif (isinstance(widget, QtWidgets.QDoubleSpinBox) or
                    isinstance(widget, QtWidgets.QSpinBox)):
                widget.setValue(value)
            else:
                raise TypeError('Invalid widget type that is not supported!\nWidget: ', widget)
        self.win.update_window()

    def load_window(self):
        """Load states of the widgets of the window

        Raises:
            TypeError: Invalid widget type in the widgets (has to be either: QCheckBox, QRadioButton, QLineEdit or QComboBox)
        """
        # Before
        self.win.logger.info('Settings: Loading window')
        # -Load states-
        self.win.settings.beginGroup('settingswindow')
        for widget in self.get_widgets():
            # Get widget name
            widget_objectName = widget.objectName()
            if not widget_objectName in const.DEFAULT_SETTINGS:
                # Default setting does not exist for this widget
                self.win.logger.warn(f'"{widget_objectName}"; {widget.__class__} does not have a default setting!')
                # Skip saving
                continue

            # Get value from settings and set to widget
            if (isinstance(widget, QtWidgets.QCheckBox) or
                    isinstance(widget, QtWidgets.QRadioButton)):
                value = self.win.settings.value(widget_objectName,
                                                defaultValue=const.DEFAULT_SETTINGS[widget_objectName],
                                                type=bool)
                widget.setChecked(value)
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = self.win.settings.value(widget_objectName,
                                                defaultValue=const.DEFAULT_SETTINGS[widget_objectName],
                                                type=str)
                widget.setText(value)
            elif isinstance(widget, QtWidgets.QComboBox):
                value = self.win.settings.value(widget_objectName,
                                                defaultValue=const.DEFAULT_SETTINGS[widget_objectName],
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
            elif (isinstance(widget, QtWidgets.QDoubleSpinBox) or
                    isinstance(widget, QtWidgets.QSpinBox)):
                value = self.win.settings.value(widget_objectName,
                                                defaultValue=const.DEFAULT_SETTINGS[widget_objectName],
                                                type=float)
                widget.setValue(value)
            else:
                raise TypeError('Invalid widget type that is not supported!\nWidget: ', widget)

        self.win.settings.endGroup()

    def save_window(self):
        """Save states of the widgets of the window

        Raises:
            TypeError: Invalid widget type in the widgets (has to be either: QCheckBox, QRadioButton, QLineEdit or QComboBox)
        """
        # Before
        self.win.logger.info('Settings: Saving window')

        # -Save states-
        self.win.settings.beginGroup('settingswindow')
        for widget in self.get_widgets():
            # Get widget name
            widget_objectName = widget.objectName()
            if not widget_objectName in const.DEFAULT_SETTINGS:
                # Default setting does not exist for this widget (so state is not saved)
                self.win.logger.warn(f'"{widget_objectName}"; {widget.__class__} does not have a default setting!')
                # Skip saving
                continue
            # Get value
            if (isinstance(widget, QtWidgets.QCheckBox) or
                    isinstance(widget, QtWidgets.QRadioButton)):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
            elif (isinstance(widget, QtWidgets.QDoubleSpinBox) or
                    isinstance(widget, QtWidgets.QSpinBox)):
                value = widget.value()
            else:
                raise TypeError('Invalid widget type that is not supported!\nWidget: ', widget)
            # Save value
            self.win.settings.setValue(widget_objectName,
                                       value)
        self.win.settings.endGroup()

    def get_widgets(self, page_idx: Optional[int] = None) -> list:
        """Obtain the configurable widgets in the window

        Args:
            page_idx (Optional[int], optional):
                Which page to load the widgets from.
                Defaults to None.

                0 - Seperation Settings Page
                1 - Shortcuts Page
                2 - Customization Page
                3 - Preferences Page
                None - All widgets

        Returns:
            list: Widgets of the given page
        """
        if page_idx is None:
            # Load all widgets
            widgets = []
            for widget_list in self.save_widgets.values():
                widgets.extend(widget_list)
        else:
            # Load one page of widgets
            assert page_idx in self.save_widgets.keys(), "Invalid page index!"
            widgets = self.save_widgets[page_idx]
        return widgets

"""
Translator Class File
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
# -Root imports-
# None
# -Other-
from typing import DefaultDict, Dict
# System
import os
import sys


class Language:
    """
    Single language in the UVR System
    """

    def __init__(self, name: str, code: str, folder_path: str):
        self.name = name
        self.code = code
        self.folder_path = folder_path
        self.flag_path = os.path.join(folder_path, 'flag.png')
        self.qm_path = os.path.join(folder_path, f'{code}.qm')
        self.load_infos()

    def load_infos(self):
        with open(os.path.join(self.folder_path, 'infos', 'settings_conversion.md')) as f:
            self.settings_conversion = "\n".join(f.readlines())


class Translator:
    """Localizer for the application

    Manages the languages for the applications

    Args:
        loaded_language (str):
            Currently loaded language in the application. To change, run method load_language.
    """
    SUPPORTED_LANGUAGES = DefaultDict(lambda: "en", **{
        'english': 'en',
        'german': 'de',
        'japanese': 'ja',
        'filipino': 'fil',
        'russian': 'ru',
        'turkish': 'tr',
    })
    LANGUAGES: Dict[str, Language] = {}

    def __init__(self, app):
        self.app = app
        self.logger = app.logger
        self.loaded_language: Language
        self._translator = QtCore.QTranslator(self.app)

        for lang_name, lang_code in self.SUPPORTED_LANGUAGES.items():
            self.LANGUAGES[lang_code] = Language(lang_name, lang_code,
                                                 os.path.join(self.app.resources.localizationDir, lang_code))

    def load_language(self, language: Language) -> bool:
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
        self.logger.info(f'Translating to "{language.name}"...',
                         indent_forwards=True)
        # Load language
        self._translator.load(language.qm_path)
        self.app.installTranslator(self._translator)

        # -Windows are initialized-
        # Update translation on all windows
        for window in self.app.windows.values():
            window.update_translation()
        # Update settings window
        for button in self.app.windows['settings'].ui.frame_languages.findChildren(QtWidgets.QPushButton):
            button_name = f'pushButton_{language.code}'
            if button.objectName() == button_name:
                # Language found
                button.setChecked(True)
            else:
                # Not selected language
                button.setChecked(False)

        self.logger.indent_backwards()
        self.loaded_language = language
        return True

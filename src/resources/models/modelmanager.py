"""
Class for managing models
"""
# pylint: disable=no-name-in-module, import-error
# -GUI-
from typing import Dict
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtGui import Qt
# -Root imports-
# None
# -Other-
import hashlib
# System
import os
import sys

# Get the absolute path to this file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    main_path = os.path.dirname(sys.executable)
    abs_path = os.path.join(main_path, 'models')
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))


MODEL_TYPES = [
    "Instrumental",
    "Vocal",
    "Karaoke",
    "Custom"
]


class Model:
    def __init__(self, model_type: str, path: str):
        self.type = model_type
        self.path = path
        self.id = self.get_model_id(path)

    @staticmethod
    def get_model_id(path: str) -> str:
        buffer_size = 65536
        sha1 = hashlib.sha1()

        with open(path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha1.update(data)

        return sha1.hexdigest()

    def __repr__(self) -> str:
        return f"Model(type:{self.type},"


class ModelManager:

    def __init__(self):
        self.model_dirs = {}
        self.available_models = {}
        self._callback = None
        self.create_model_folders()

    def search_for_models(self, force_callback: bool = False):
        """Search for the models in each model type folder"""
        new_available_models = {}

        for model_type, model_dir in self.model_dirs.items():
            for index, f in enumerate(os.listdir(model_dir)):
                if not f.endswith('.pth'):
                    # File is not a model file, so skip
                    continue
                # Get data
                path = os.path.join(model_dir, f)
                new_available_models[model_type] = Model(model_type, path)

        if (new_available_models != self.available_models or
                force_callback):
            self.available_models = new_available_models
            if self._callback is not None:
                self._callback()
        else:
            self.available_models = new_available_models

    def set_callback(self, callback):
        """Set a callback function that will be called
        when the avaiable models have changed

        Args:
            callback (function): Callback funtion
        """
        self._callback = callback

    def create_model_folders(self):
        """Create the folders for each model type"""
        self.model_dirs.clear()
        for model in MODEL_TYPES:
            model_dir = os.path.join(abs_path, model)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
            self.model_dirs[model] = model_dir

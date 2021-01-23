import pickle
import os
import sys

# Get the absolute path to this file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    abs_path = sys._MEIPASS  # pylint: disable=no-member
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))


class DataManager:
    """
    Load/Save/Delete Data Manager
    """

    def __init__(self, default_data: dict = {}, file_path: str = None):
        self.default_data = default_data
        self._data = self.default_data
        self.file_path = os.path.join(abs_path, 'data', 'data.pkl') if file_path is None else file_path
        self.save_folder = os.path.dirname(self.file_path)
        self._load_file()

    @property
    def data(self) -> dict:
        return self._data

    @data.setter
    def data(self, data: dict):
        """
        Update the new saved data
        """
        assert isinstance(data, dict)
        self._data = data
        self.save_file()

    @data.deleter
    def data(self):
        self.data = self.default_data
        self.save_file()

    def value(self, key):
        """
        Return value with the key
        """
        if not key in self._data.keys():
            raise TypeError('Specified key not in data! Key:', key)
        return self._data[key]

    def setValue(self, key, value):
        """
        Set value of key
        """
        self._data[key] = value
        self.save_file()

    def save_file(self):
        """
        Saves given data as a .pkl (pickle) file
        """
        # Open data file, create it if it does not exist
        with open(self.file_path, 'wb') as data_file:
            pickle.dump(self.data, data_file)

    def _load_file(self):
        """
        Loads saved pkl file and sets it to the data variable
        """
        try:
            with open(self.file_path, 'rb') as data_file:  # Open data file
                self._data = pickle.load(data_file)
        except (ValueError, FileNotFoundError):
            # Data File is corrupted or not found so recreate it
            self._data = self.default_data
            self.save_file()
            self._load_file()

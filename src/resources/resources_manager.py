import os
import sys

# Get the absolute path to this file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    main_path = sys._MEIPASS  # pylint: disable=no-member
    abs_path = os.path.join(main_path, 'resources')
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))


class ResourcePaths:
    """
    Get access to all resources used in the application
    through this class
    """
    class images:
        _IMAGE_FOLDER = 'images'
        refresh = os.path.join(abs_path, _IMAGE_FOLDER, 'refresh.png')
        showcase = os.path.join(abs_path, _IMAGE_FOLDER, 'showcase.png')
        icon = os.path.join(abs_path, _IMAGE_FOLDER, 'icon.ico')
        banner = os.path.join(abs_path, _IMAGE_FOLDER, 'banner.png')
        settings = os.path.join(abs_path, _IMAGE_FOLDER, 'settings.png')

    class ui_files:
        _UI_FOLDER = 'ui_files'
        mainwindow = os.path.join(abs_path, _UI_FOLDER, 'mainwindow.ui')

    class models:
        _MODELS_FOLDER = 'models'
        modelsFolder = os.path.join(abs_path, _MODELS_FOLDER)


if __name__ == "__main__":
    """Print all resources"""

    print('-- Images --')
    for img, img_path in vars(ResourcePaths.images).items():
        if os.path.isfile(str(img_path)):
            print(f'{img} -> {img_path}')
    print('-- UI Files --')
    for ui, ui_path in vars(ResourcePaths.ui_files).items():
        if os.path.isfile(str(ui_path)):
            print(f'{ui} -> {ui_path}')

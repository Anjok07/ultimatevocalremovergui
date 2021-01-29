import os
import sys

# Get the absolute path to this file
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    main_path = os.path.dirname(sys.executable)
    abs_path = os.path.join(main_path, 'resources')
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))


IMAGE_FOLDER = 'images'
MODELS_FOLDER = 'models'
TRANSLATIONS_FOLDER = 'models'


class ResourcePaths:
    """
    Get access to all resources used in the application
    through this class
    """
    class images:
        refresh = os.path.join(abs_path, IMAGE_FOLDER, 'refresh.png')
        showcase = os.path.join(abs_path, IMAGE_FOLDER, 'showcase.png')
        icon = os.path.join(abs_path, IMAGE_FOLDER, 'icon.ico')
        banner = os.path.join(abs_path, IMAGE_FOLDER, 'banner.png')
        settings = os.path.join(abs_path, IMAGE_FOLDER, 'settings.png')
        folder = os.path.join(abs_path, IMAGE_FOLDER, 'folder.png')

        class flags:
            _FLAG_FOLDER = 'flags'
            english = os.path.join(abs_path, IMAGE_FOLDER, _FLAG_FOLDER, 'english.png')
            german = os.path.join(abs_path, IMAGE_FOLDER, _FLAG_FOLDER, 'german.png')
            japanese = os.path.join(abs_path, IMAGE_FOLDER, _FLAG_FOLDER, 'japan.png')
            filipino = os.path.join(abs_path, IMAGE_FOLDER, _FLAG_FOLDER, 'filipino.png')


    modelsDir = os.path.join(abs_path, MODELS_FOLDER)
    localizationDir = os.path.join(abs_path, TRANSLATIONS_FOLDER)


if __name__ == "__main__":
    """Print all resources"""

    print('-- Images --')
    for img, img_path in vars(ResourcePaths.images).items():
        if os.path.isfile(str(img_path)):
            print(f'{img} -> {img_path}')

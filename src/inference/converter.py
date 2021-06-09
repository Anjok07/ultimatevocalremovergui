default_data = {
    # Paths
    'input_paths': [],  # List of paths
    'export_path': '',  # Export path
    # Processing Options
    'gpuConversion': False,
    'postProcess': True,
    'tta': True,
    'outputImage': False,
    # Models
    'instrumentalModel': '',  # Path to instrumental (not needed if not used)
    'vocalModel': '',  # Path to vocal model (not needed if not used)
    'isVocal': False,
    # Model Folder
    'modelFolder': False,  # Model Test Mode
    # Constants
    'window_size': 320,
    'deepExtraction': True,
    'aggressiveness': 0.02,
    # Allows to process multiple music files at once
    'multithreading': False,
    # What to save
    'save_instrumentals': True,
    'save_vocals': True,
}

"""
Non-GUI version of UVR

To see more info on configurable options for the VocalRemover, go to inference_v4.py and see variable 'default_data'
"""
from inference.converter_v4 import (VocalRemover, default_data)

seperation_data = default_data.copy()
seperation_data['input_paths'] = [r'']
seperation_data['export_path'] = r''
seperation_data['instrumentalModel'] = r''
seperation_data['stackPasses'] = 0
seperation_data['stackModel'] = r''

vocal_remover = VocalRemover(data=seperation_data)
vocal_remover.seperate_files()

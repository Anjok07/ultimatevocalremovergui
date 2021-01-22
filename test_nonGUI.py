"""
Test the non GUI version of UVR

To see configurable options for the VocalRemover, go to inference_v4.py and see variable 'default_data'
"""
from inference_v4 import (VocalRemover, default_data)

seperation_data = default_data.copy()
seperation_data['input_paths'] = ['fe']
seperation_data['export_path'] = ''
seperation_data['instrumentalModel'] = r'B:\boska\Documents\Dilan\Dropbox\Github\ultimatevocalremovergui\models\v4\Main Models\MGM_LOWEND_B_v4_sr33075_hl384_nf2048.pth'
seperation_data['stackPasses'] = 2


vocal_remover = VocalRemover(data=seperation_data)
vocal_remover.seperate_files()

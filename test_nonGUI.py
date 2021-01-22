"""
Test the non GUI version of UVR

To see configurable options for the VocalRemover, go to inference_v4.py and see variable 'default_data'
"""
from inference_v4 import (VocalRemover, default_data)

seperation_data = default_data.copy()
seperation_data['input_paths'] = [r'B:\boska\Desktop\Test inference\test.mp3']
seperation_data['export_path'] = r'B:\boska\Desktop\Test inference'
seperation_data['instrumentalModel'] = r'B:\boska\Documents\Dilan\Dropbox\Github\ultimatevocalremovergui\models\v4\Main Models\MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
seperation_data['modelFolder'] = True
seperation_data['tta'] = False


vocal_remover = VocalRemover(data=seperation_data)
vocal_remover.seperate_files()

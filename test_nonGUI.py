"""
Test the non GUI version of UVR
"""
from inference_v4 import (VocalRemover, default_data)

seperation_data = default_data.copy()
seperation_data['input_paths'] = []
seperation_data['export_path'] = ''
seperation_data['instrumentalModel'] = 'models/v4/Main Models/MGM_HIGHEND_v4_sr44100_hl1024_nf2048.pth'
seperation_data['useModel'] = 'instrumental'
seperation_data['stackModel'] = 'models/v4/Stacked Models/StackedMGM_LL_v4_sr32000_hl512_nf2048.pth'
seperation_data['stackPasses'] = 0

vocal_remover = VocalRemover(data=seperation_data)
vocal_remover.seperate_files()

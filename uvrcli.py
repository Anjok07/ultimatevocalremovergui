"""
This is the official CLI version of UVR (beta).
Currently only v4 is supported
"""
import argparse
import sys
from converter_v4 import (VocalRemover, default_data)


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
optional = parser._action_groups.pop()  # Edited this line
required = parser.add_argument_group('required arguments')
# -IO-
required.add_argument('-i', '--inputs', metavar='Input Path/s', type=str, nargs='+',
                      required=True, dest='input_paths',
                      help='Path to music file/s')
required.add_argument('-o', '--export', metavar='\b', type=str,
                      required=True, dest='export_path',
                      help='Path to export directory')
# -Models-
# Only v4 supported for now
# parser.add_argument('-engine', '--engine', metavar='\b', type=str,
#                     required=False, dest='engine', default='v4',
#                     help='AI Engine (default: v4)')
required.add_argument('-inst', '--instrumentalModel', metavar='\b', type=str,
                      required=True, dest='instrumentalModel',
                      help='Path to instrumental model. Constants are extracted from the file name (to disable, set --customParameters to True)')
# -Settings-
arguments = {
    'gpuConversion': {
        'flags': ['-gpu', '--gpuConversion'],
        'help': 'Use GPU acceleration (CUDA).',
        'type': bool,
    },
    'postProcess': {
        'flags': ['-post', '--postProcess'],
        'help': '[Only v4] This option can potentially identify leftover instrumental artifacts within the vocal outputs. This option may improve the separation on some songs.',
        'type': bool,
    },
    'tta': {
        'flags': ['-tta', '--tta'],
        'help': 'This option performs Test-Time-Augmentation to improve the separation quality.',
        'type': bool,
    },
    'outputImage': {
        'flags': ['-image', '--outputImage'],
        'help': 'Selecting this option will include the spectrograms in .jpg format for the instrumental & vocal audio outputs.',
        'type': bool,
    },
    'resType': {
        'flags': ['-res', '--resolutionType'],
        'help': "(Choose from 'kaiser_fast', 'kaiser_best', 'scipy') Type of spectogram used by the program.",
        'type': str,
        'choices': ['kaiser_fast', 'kaiser_best', 'scipy']
    },
    'modelFolder': {
        'flags': ['-modelFolder', '--modelTestMode'],
        'help': 'This option structures the model testing process by creating a folder in the export path named after the model/s used in the seperation and saving the audio files there.',
        'type': bool,
    },
    'stackModel': {
        'flags': ['-stack', '--stackModel'],
        'help': 'Path to stack model used in the stack passes. Constants are extracted from the file name (to disable, set --customParameters to True)',
        'type': str,
    },
    'stackPasses': {
        'flags': ['-stackPasses', '--stackPasses'],
        'help': 'Set the number of times a track runs through a stacked model.',
        'type': int,
    },
    'stackOnly': {
        'flags': ['-stackOnly', '--stackOnly'],
        'help': 'Selecting this option allows the user to bypass the main model and run a track through a stacked model only.',
        'type': bool,
    },
    'saveAllStacked': {
        'flags': ['-stackSave', '--saveAllStacked'],
        'help': 'Having this option selected will auto-generate a new folder named after the track being processed to your export path. The new folder will contain all of the outputs that were generated after each stack pass.',
        'type': bool,
    },
    'customParameters': {
        'flags': ['-custom', '--customParameters'],
        'help': 'Allows you to set custom parameters (SR, HOP LENGNTH, & N_FFT) instead of using the ones appended to the model filenames',
        'type': bool,
    },
    'sr': {
        'flags': ['-sr', '--samplingRate'],
        'help': '[Required: --customParameters is True] Sampling Rate',
        'type': int,
    },
    'sr_stacked': {
        'flags': ['-sr_stack', '--samplingRate_stack'],
        'help': '[Required: --customParameters is True] Sampling Rate for stack model',
        'type': int,
    },
    'hop_length': {
        'flags': ['-hopLength', '--hopLength'],
        'help': '[Required: --customParameters is True] Hop Length',
        'type': int,
    },
    'hop_length_stacked': {
        'flags': ['-hopLength_stack', '--hopLength_stack'],
        'help': '[Required: --customParameters is True] Hop Length for stack model',
        'type': int,
    },
    'window_size': {
        'flags': ['-winSize', '--windowSize'],
        'help': '[Required: --customParameters is True] Window Size',
        'type': int,
    },
    'window_size_stacked': {
        'flags': ['-winSize_stack', '--windowSize_stack'],
        'help': '[Required: --customParameters is True] Window Size for stack model',
        'type': int,
    },
    'n_fft': {
        'flags': ['-nfft', '--nfft'],
        'help': '[Required: --customParameters is True] NFFT',
        'type': int,
    },
    'n_fft_stacked': {
        'flags': ['-nfft_stack', '--nfft_stack'],
        'help': '[Required: --customParameters is True] NFFT for stack model',
        'type': int,
    },
}
for argument_name, options in arguments.items():
    default = default_data[argument_name]
    required = False
    choices = None
    if 'default' in options:
        default = options['default']
    if 'required' in options:
        required = options['required']
    if 'choices' in options:
        choices = options['choices']

    optional.add_argument(*options['flags'], metavar='\b', type=options['type'],
                          required=required, dest=argument_name, default=default,
                          choices=choices,
                          help=options['help'] + f' (default: "{default}")')
parser._action_groups.append(optional)  # added this line
args = parser.parse_args()
seperation_data = dict(args._get_kwargs())

seperation_data['useModel'] = 'instrumental'
seperation_data['vocalModel'] = 'vocalModel'

if set(seperation_data.keys()) != set(default_data.keys()):
    raise TypeError(
        f'Extracted Keys do not equal keys set by default converter!\nExtracted Keys: {sorted(list(seperation_data.keys()))}\nShould be Keys: {sorted(list(default_data.keys()))}')

aiEngine = 'v4'
if aiEngine == 'v4':
    vocal_remover = VocalRemover(seperation_data=seperation_data)
    vocal_remover.seperate_files()
else:
    print('Not supported AI Engine!')

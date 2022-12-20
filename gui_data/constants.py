#Model Types
VR_ARCH_TYPE = 'VR Arc'
MDX_ARCH_TYPE = 'MDX-Net'
DEMUCS_ARCH_TYPE = 'Demucs'
VR_ARCH_PM = 'VR Architecture'
ENSEMBLE_MODE = 'Ensemble Mode'
ENSEMBLE_STEM_CHECK = 'Ensemble Stem'
SECONDARY_MODEL = 'Secondary Model'
DEMUCS_6_STEM_MODEL = 'htdemucs_6s'

DEMUCS_V3_ARCH_TYPE = 'Demucs v3'
DEMUCS_V4_ARCH_TYPE = 'Demucs v4'
DEMUCS_NEWER_ARCH_TYPES = [DEMUCS_V3_ARCH_TYPE, DEMUCS_V4_ARCH_TYPE]

DEMUCS_V1 = 'v1'
DEMUCS_V2 = 'v2'
DEMUCS_V3 = 'v3'
DEMUCS_V4 = 'v4'

DEMUCS_V1_TAG = 'v1 | '
DEMUCS_V2_TAG = 'v2 | '
DEMUCS_V3_TAG = 'v3 | '
DEMUCS_V4_TAG = 'v4 | '
DEMUCS_NEWER_TAGS = [DEMUCS_V3_TAG, DEMUCS_V4_TAG]

DEMUCS_VERSION_MAPPER = {
            DEMUCS_V1:DEMUCS_V1_TAG,
            DEMUCS_V2:DEMUCS_V2_TAG,
            DEMUCS_V3:DEMUCS_V3_TAG,
            DEMUCS_V4:DEMUCS_V4_TAG}

#Download Center
DOWNLOAD_FAILED = 'Download Failed'
DOWNLOAD_STOPPED = 'Download Stopped'
DOWNLOAD_COMPLETE = 'Download Complete'
DOWNLOAD_UPDATE_COMPLETE = 'Update Download Complete'
SETTINGS_MENU_EXIT = 'exit'
NO_CONNECTION = 'No Internet Connection'
VIP_SELECTION = 'VIP:'
DEVELOPER_SELECTION = 'VIP:'
NO_NEW_MODELS = 'All Available Models Downloaded'
ENSEMBLE_PARTITION = ': '
NO_MODEL = 'No Model Selected'
CHOOSE_MODEL = 'Choose Model'
SINGLE_DOWNLOAD = 'Downloading Item 1/1...'
DOWNLOADING_ITEM = 'Downloading Item'
FILE_EXISTS = 'File already exists!'
DOWNLOADING_UPDATE = 'Downloading Update...'
DOWNLOAD_MORE = 'Download More Models'

#Menu Options

AUTO_SELECT = 'Auto'

#LINKS
DOWNLOAD_CHECKS = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
MDX_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data.json"
VR_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data.json"
DONATE_LINK_BMAC = "https://www.buymeacoffee.com/uvr5"
DONATE_LINK_PATREON = "https://www.patreon.com/uvr"

#DOWNLOAD REPOS
NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UPDATE_REPO = "https://github.com/TRvlvr/model_repo/releases/download/uvr_update_patches/"
ISSUE_LINK = 'https://github.com/Anjok07/ultimatevocalremovergui/issues/new'
VIP_REPO = b'\xf3\xc2W\x19\x1foI)\xc2\xa9\xcc\xb67(Z\xf5',\
           b'gAAAAABjQAIQ-NpNMMxMedpKHHb7ze_nqB05hw0YhbOy3pFzuzDrfqumn8_qvraxEoUpZC5ZXC0gGvfDxFMqyq9VWbYKlA67SUFI_wZB6QoVyGI581vs7kaGfUqlXHIdDS6tQ_U-BfjbEAK9EU_74-R2zXjz8Xzekw=='
NO_CODE = 'incorrect_code'

#Extensions

ONNX = '.onnx'
YAML = '.yaml'
PTH = '.pth'
JSON = '.json'

#GUI Buttons

START_PROCESSING = 'Start Processing'
WAIT_PROCESSING = 'Please wait...'
STOP_PROCESSING = 'Halting process, please wait...'
LOADING_MODELS = 'Loading models...'

#---Messages and Logs----

MISSING_MODEL = 'missing'
MODEL_PRESENT = 'present'

UNRECOGNIZED_MODEL = 'Unrecognized Model Detected', ' is an unrecognized model.\n\n' + \
                     'Would you like to select the correct parameters before continuing?'
                     
STOP_PROCESS_CONFIRM = 'Confirmation', 'You are about to stop all active processes.\n\nAre you sure you wish to continue?'
NO_ENSEMBLE_SELECTED = 'No Models Selected', 'Please select ensemble and try again.'
PICKLE_CORRU = 'File Corrupted', 'Unable to load this ensemble.\n\n' + \
               'Would you like to remove this ensemble from your list?'
DELETE_ENS_ENTRY = 'Confirm Removal', 'Are you sure you want to remove this entry?'

ALL_STEMS = 'All Stems'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
OTHER_STEM = 'Other'
BASS_STEM = 'Bass'
DRUM_STEM = 'Drums'
GUITAR_STEM = 'Guitar'
PIANO_STEM = 'Piano'
SYNTH_STEM = 'Synthesizer'
STRINGS_STEM = 'Strings'
WOODWINDS_STEM = 'Woodwinds'
NO_OTHER_STEM = 'No Other'
NO_BASS_STEM = 'No Bass'
NO_DRUM_STEM = 'No Drums'
NO_GUITAR_STEM = 'No Guitar'
NO_PIANO_STEM = 'No Piano'
NO_SYNTH_STEM = 'No Synthesizer'
NO_STRINGS_STEM = 'No Strings'
NO_WOODWINDS_STEM = 'No Woodwinds'
PRIMARY_STEM = 'Primary Stem'
SECONDARY_STEM = 'Secondary Stem'

#Other Constants
DEMUCS_2_SOURCE = ["instrumental", "vocals"]
DEMUCS_4_SOURCE = ["drums", "bass", "other", "vocals"]

DEMUCS_2_SOURCE_MAPPER = {
                        INST_STEM: 0,
                        VOCAL_STEM: 1}

DEMUCS_4_SOURCE_MAPPER = {
                        BASS_STEM: 0,
                        DRUM_STEM: 1,
                        OTHER_STEM: 2,
                        VOCAL_STEM: 3}

DEMUCS_6_SOURCE_MAPPER = {
                        BASS_STEM: 0,
                        DRUM_STEM: 1,
                        OTHER_STEM: 2,
                        VOCAL_STEM: 3,
                        GUITAR_STEM:4,
                        PIANO_STEM:5}

DEMUCS_4_SOURCE_LIST = [BASS_STEM, DRUM_STEM, OTHER_STEM, VOCAL_STEM]
DEMUCS_6_SOURCE_LIST = [BASS_STEM, DRUM_STEM, OTHER_STEM, VOCAL_STEM, GUITAR_STEM, PIANO_STEM]

DEMUCS_UVR_MODEL = 'UVR_Model'

CHOOSE_STEM_PAIR = 'Choose Stem Pair'

STEM_SET_MENU = (VOCAL_STEM, 
                 INST_STEM, 
                 OTHER_STEM, 
                 BASS_STEM, 
                 DRUM_STEM, 
                 GUITAR_STEM, 
                 PIANO_STEM, 
                 SYNTH_STEM, 
                 STRINGS_STEM, 
                 WOODWINDS_STEM, 
                 NO_OTHER_STEM, 
                 NO_BASS_STEM, 
                 NO_DRUM_STEM, 
                 NO_GUITAR_STEM, 
                 NO_PIANO_STEM, 
                 NO_SYNTH_STEM, 
                 NO_STRINGS_STEM, 
                 NO_WOODWINDS_STEM)

STEM_PAIR_MAPPER = {
            VOCAL_STEM: INST_STEM,
            INST_STEM: VOCAL_STEM,
            OTHER_STEM: NO_OTHER_STEM,
            BASS_STEM: NO_BASS_STEM,
            DRUM_STEM: NO_DRUM_STEM,
            GUITAR_STEM: NO_GUITAR_STEM,
            PIANO_STEM: NO_PIANO_STEM,
            NO_OTHER_STEM: OTHER_STEM,
            NO_BASS_STEM: BASS_STEM,
            NO_DRUM_STEM: DRUM_STEM,
            PRIMARY_STEM: SECONDARY_STEM,
            NO_GUITAR_STEM: GUITAR_STEM,
            NO_PIANO_STEM: PIANO_STEM,
            SYNTH_STEM: NO_SYNTH_STEM,
            STRINGS_STEM: NO_STRINGS_STEM,
            WOODWINDS_STEM: NO_WOODWINDS_STEM}

MDX_NET_FREQ_CUT = [VOCAL_STEM, INST_STEM]

DEMUCS_4_STEM_OPTIONS = (ALL_STEMS, VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM)
DEMUCS_6_STEM_OPTIONS = (ALL_STEMS, VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM)
DEMUCS_2_STEM_OPTIONS = (VOCAL_STEM, INST_STEM)
DEMUCS_4_STEM_CHECK = (OTHER_STEM, BASS_STEM, DRUM_STEM)

#Menu Dropdowns

VOCAL_PAIR = f'{VOCAL_STEM}/{INST_STEM}'
INST_PAIR = f'{INST_STEM}/{VOCAL_STEM}'
OTHER_PAIR = f'{OTHER_STEM}/{NO_OTHER_STEM}'
DRUM_PAIR = f'{DRUM_STEM}/{NO_DRUM_STEM}'
BASS_PAIR = f'{BASS_STEM}/{NO_BASS_STEM}'
FOUR_STEM_ENSEMBLE = '4 Stem Ensemble'

ENSEMBLE_MAIN_STEM = (CHOOSE_STEM_PAIR, VOCAL_PAIR, OTHER_PAIR, DRUM_PAIR, BASS_PAIR, FOUR_STEM_ENSEMBLE)

MIN_SPEC = 'Min Spec'
MAX_SPEC = 'Max Spec'
AUDIO_AVERAGE = 'Average'

MAX_MIN = f'{MAX_SPEC}/{MIN_SPEC}'
MAX_MAX = f'{MAX_SPEC}/{MAX_SPEC}'
MAX_AVE = f'{MAX_SPEC}/{AUDIO_AVERAGE}'
MIN_MAX = f'{MIN_SPEC}/{MAX_SPEC}'
MIN_MIX = f'{MIN_SPEC}/{MIN_SPEC}'
MIN_AVE = f'{MIN_SPEC}/{AUDIO_AVERAGE}'
AVE_MAX = f'{AUDIO_AVERAGE}/{MAX_SPEC}'
AVE_MIN = f'{AUDIO_AVERAGE}/{MIN_SPEC}'
AVE_AVE = f'{AUDIO_AVERAGE}/{AUDIO_AVERAGE}'

ENSEMBLE_TYPE = (MAX_MIN, MAX_MAX, MAX_AVE, MIN_MAX, MIN_MIX, MIN_AVE, AVE_MAX, AVE_MIN, AVE_AVE)
ENSEMBLE_TYPE_4_STEM = (MAX_SPEC, MIN_SPEC, AUDIO_AVERAGE)

CHUNKS = (AUTO_SELECT, '1', '5', '10', '15', '20', 
          '25', '30', '35', '40', '45', '50', 
          '55', '60', '65', '70', '75', '80', 
          '85', '90', '95', 'Full')

VOL_COMPENSATION = (AUTO_SELECT, '1.035', '1.08')

MARGIN_SIZE = ('44100', '22050', '11025')

AUDIO_TOOLS = 'Audio Tools'

MANUAL_ENSEMBLE = 'Manual Ensemble'
TIME_STRETCH = 'Time Stretch'
CHANGE_PITCH = 'Change Pitch'
ALIGN_INPUTS = 'Align Inputs'

AUDIO_TOOL_OPTIONS = (MANUAL_ENSEMBLE, ALIGN_INPUTS)

MANUAL_ENSEMBLE_OPTIONS = (MIN_SPEC, MAX_SPEC, AUDIO_AVERAGE)

PROCESS_METHODS = (VR_ARCH_PM, MDX_ARCH_TYPE, DEMUCS_ARCH_TYPE, ENSEMBLE_MODE, AUDIO_TOOLS)

DEMUCS_SEGMENTS = ('Default', '1', '5', '10', '15', '20', 
                  '25', '30', '35', '40', '45', '50', 
                  '55', '60', '65', '70', '75', '80', 
                  '85', '90', '95', '100')

DEMUCS_SHIFTS = (0, 1, 2, 3, 4, 5, 
                 6, 7, 8, 9, 10, 11, 
                 12, 13, 14, 15, 16, 17, 
                 18, 19, 20)

DEMUCS_OVERLAP = (0.25, 0.50, 0.75, 0.99)

VR_AGGRESSION = (1, 2, 3, 4, 5, 
                 6, 7, 8, 9, 10, 11, 
                 12, 13, 14, 15, 16, 17, 
                 18, 19, 20)

VR_WINDOW = ('320', '512','1024')
VR_CROP = ('256', '512', '1024')
VR_BATCH = ('4', '6', '8')

MDX_POP_PRO = ('MDX-NET_Noise_Profile_14_kHz', 'MDX-NET_Noise_Profile_17_kHz', 'MDX-NET_Noise_Profile_Full_Band')
MDX_POP_STEMS = ('Vocals', 'Instrumental', 'Other', 'Drums', 'Bass')
MDX_POP_NFFT = ('4096', '5120', '6144', '7680', '8192', '16384')
MDX_POP_DIMF = ('2048', '3072', '4096')

SAVE_ENSEMBLE = 'Save Ensemble'
CLEAR_ENSEMBLE = 'Clear Selection(s)'
MENU_SEPARATOR = 35*'•'
CHOOSE_ENSEMBLE_OPTION = 'Choose Option'

INVALID_ENTRY = 'Invalid Input, Please Try Again'
ENSEMBLE_INPUT_RULE = '1. Only letters, numbers, spaces, and dashes allowed.\n2. No dashes or spaces at the start or end of input.'

ENSEMBLE_OPTIONS = (SAVE_ENSEMBLE, CLEAR_ENSEMBLE)
ENSEMBLE_CHECK = 'ensemble check'

SELECT_SAVED_ENSEMBLE = 'Select Saved Ensemble'
SELECT_SAVED_SETTING = 'Select Saved Setting'
ENSEMBLE_OPTION = "Ensemble Customization Options"
MDX_OPTION = "Advanced MDX-Net Options"
DEMUCS_OPTION = "Advanced Demucs Options"
VR_OPTION = "Advanced VR Options"
HELP_OPTION = "Open Information Guide"
ERROR_OPTION = "Open Error Log"
VERIFY_BEGIN = 'Verifying file '
SAMPLE_BEGIN = 'Creating Sample '
MODEL_MISSING_CHECK = 'Model Missing:'

# Audio Player

PLAYING_SONG = ": Playing"
PAUSE_SONG = ": Paused"
STOP_SONG = ": Stopped"

SELECTED_VER = 'Selected'
DETECTED_VER = 'Detected'

SAMPLE_MODE_CHECKBOX = lambda v:f'Sample Mode ({v}s)'
REMOVED_FILES = lambda r, e:f'Audio Input Verification Report:\n\nRemoved Files:\n\n{r}\n\nError Details:\n\n{e}'
ADVANCED_SETTINGS = (ENSEMBLE_OPTION, MDX_OPTION, DEMUCS_OPTION, VR_OPTION, HELP_OPTION, ERROR_OPTION)

WAV = 'WAV'
FLAC = 'FLAC'
MP3 = 'MP3'

MP3_BIT_RATES = ('96k', '128k', '160k', '224k', '256k', '320k')
WAV_TYPE = ('PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', '32-bit Float', '64-bit Float')

SELECT_SAVED_SET = 'Choose Option'
SAVE_SETTINGS = 'Save Current Settings'
RESET_TO_DEFAULT = 'Reset to Default'
RESET_FULL_TO_DEFAULT = 'Reset to Default'
RESET_PM_TO_DEFAULT = 'Reset All Application Settings to Default'

SAVE_SET_OPTIONS = (SAVE_SETTINGS, RESET_TO_DEFAULT)

TIME_PITCH = ('1.0', '2.0', '3.0', '4.0')
TIME_TEXT = '_time_stretched'
PITCH_TEXT = '_pitch_shifted'

#RegEx Input Validation

REG_TIME_PITCH = r'^[-+]?(1[0]|[0-9]([.][0-9]*)?)$'
REG_COMPENSATION = r'\b^(1[0]|[0-9]([.][0-9]*)?|Auto|None)$\b'
REG_COMPENSATION = r'\b^(1[0]|[0-9]([.][0-9]*)?|Auto|None)$\b'
REG_CHUNKS = r'\b^(200|1[0-9][0-9]|[1-9][0-9]?|Auto|Full)$\b'
REG_MARGIN = r'\b^[0-9]*$\b'
REG_SEGMENTS = r'\b^(200|1[0-9][0-9]|[1-9][0-9]?|Default)$\b'
REG_SAVE_INPUT = r'\b^([a-zA-Z0-9 -]{0,25})$\b'
REG_AGGRESSION = r'^[-+]?[0-9]\d*?$'
REG_WINDOW = r'\b^[0-9]{0,4}$\b'
REG_SHIFTS = r'\b^[0-9]*$\b'
REG_OVERLAP = r'\b^([0]([.][0-9]{0,6})?|None)$\b'

# Sub Menu

VR_ARCH_SETTING_LOAD = 'Load for VR Arch'
MDX_SETTING_LOAD = 'Load for MDX-Net'
DEMUCS_SETTING_LOAD = 'Load for Demucs'
ALL_ARCH_SETTING_LOAD = 'Load for Full Application'

# Mappers

MDX_NAME_SELECT = {
                "UVR_MDXNET_1_9703": 'UVR-MDX-NET 1',
                "UVR_MDXNET_2_9682": 'UVR-MDX-NET 2',
                "UVR_MDXNET_3_9662": 'UVR-MDX-NET 3',
                "UVR_MDXNET_KARA": 'UVR-MDX-NET Karaoke',
                "UVR_MDXNET_Main": 'UVR-MDX-NET Main',
                "UVR-MDX-NET-Inst_1": 'UVR-MDX-NET Inst 1',
                "UVR-MDX-NET-Inst_2": 'UVR-MDX-NET Inst 2',
                "UVR-MDX-NET-Inst_3": 'UVR-MDX-NET Inst 3',
                "UVR-MDX-NET-Inst_Main": 'UVR-MDX-NET Inst Main'}

DEMUCS_NAME_SELECT = {
                'tasnet.th': 'v1 | Tasnet',
                'tasnet_extra.th': 'v1 | Tasnet_extra',
                'demucs.th': 'v1 | Demucs',
                'demucs_extra.th': 'v1 | Demucs_extra',
                'light.th': 'v1 | Light',
                'light_extra.th': 'v1 | Light_extra',
                'tasnet.th.gz': 'v1 | Tasnet.gz',
                'tasnet_extra.th.gz': 'v1 | Tasnet_extra.gz',
                'demucs.th.gz': 'v1 | Demucs_extra.gz',
                'light.th.gz': 'v1 | Light.gz',
                'light_extra.th.gz': "v1 | Light_extra.gz'",
                'tasnet-beb46fac.th': 'v2 | Tasnet',
                'tasnet_extra-df3777b2.th': 'v2 | Tasnet_extra',
                'demucs48_hq-28a1282c.th': 'v2 | Demucs48_hq',
                'demucs_extra-3646af93.th': 'v2 | Demucs_extra',
                'demucs_unittest-09ebc15f.th': 'v2 | Demucs_unittest',
                'mdx.yaml': 'v3 | mdx',
                'mdx_extra.yaml': 'v3 | mdx_extra',
                'mdx_extra_q.yaml': 'v3 | mdx_extra_q',
                'mdx_q.yaml': 'v3 | mdx_q',
                'repro_mdx_a.yaml': 'v3 | repro_mdx_a',
                'repro_mdx_a_hybrid_only.yaml': 'v3 | repro_mdx_a_hybrid',
                'repro_mdx_a_time_only.yaml': 'v3 | repro_mdx_a_time',
                'UVR_Demucs_Model_1.yaml': 'v3 | UVR_Model_1',
                'UVR_Demucs_Model_2.yaml': 'v3 | UVR_Model_2',
                'UVR_Demucs_Model_Bag.yaml': 'v3 | UVR_Model_Bag',
                'hdemucs_mmi.yaml': 'v4 | hdemucs_mmi',
                'htdemucs.yaml': 'v4 | htdemucs',
                'htdemucs_ft.yaml': 'v4 | htdemucs_ft',
                'htdemucs_6s.yaml': 'v4 | htdemucs_6s'
                }

DEFAULT_DATA = {
    
        'chosen_process_method': MDX_ARCH_TYPE,
        'vr_model': CHOOSE_MODEL,
        'aggression_setting': 10,
        'window_size': 512,
        'batch_size': 4,
        'crop_size': 256, 
        'is_tta': False,
        'is_output_image': False,
        'is_post_process': False,
        'is_high_end_process': False,
        'vr_voc_inst_secondary_model': NO_MODEL,
        'vr_other_secondary_model': NO_MODEL,
        'vr_bass_secondary_model': NO_MODEL,
        'vr_drums_secondary_model': NO_MODEL,
        'vr_is_secondary_model_activate': False,        
        'vr_voc_inst_secondary_model_scale': 0.9,
        'vr_other_secondary_model_scale': 0.7,
        'vr_bass_secondary_model_scale': 0.5,
        'vr_drums_secondary_model_scale': 0.5,
        'demucs_model': CHOOSE_MODEL, 
        'demucs_stems': ALL_STEMS,
        'segment': DEMUCS_SEGMENTS[0],
        'overlap': DEMUCS_OVERLAP[0],
        'shifts': 2,
        'chunks_demucs': CHUNKS[0],
        'margin_demucs': 44100,
        'is_chunk_demucs': False,
        'is_primary_stem_only_Demucs': False,
        'is_secondary_stem_only_Demucs': False,
        'is_split_mode': True,
        'is_demucs_combine_stems': True,
        'demucs_voc_inst_secondary_model': NO_MODEL,
        'demucs_other_secondary_model': NO_MODEL,
        'demucs_bass_secondary_model': NO_MODEL,
        'demucs_drums_secondary_model': NO_MODEL,
        'demucs_is_secondary_model_activate': False,        
        'demucs_voc_inst_secondary_model_scale': 0.9,
        'demucs_other_secondary_model_scale': 0.7,
        'demucs_bass_secondary_model_scale': 0.5,
        'demucs_drums_secondary_model_scale': 0.5,
        'demucs_stems': ALL_STEMS,
        'demucs_pre_proc_model': NO_MODEL,
        'is_demucs_pre_proc_model_activate': False,
        'is_demucs_pre_proc_model_inst_mix': False,
        'mdx_net_model': CHOOSE_MODEL,
        'chunks': CHUNKS[0],
        'margin': 44100,
        'compensate': AUTO_SELECT,
        'is_denoise': False,
        'is_invert_spec': False, 
        'mdx_voc_inst_secondary_model': NO_MODEL,
        'mdx_other_secondary_model': NO_MODEL,
        'mdx_bass_secondary_model': NO_MODEL,
        'mdx_drums_secondary_model': NO_MODEL,
        'mdx_is_secondary_model_activate': False,        
        'mdx_voc_inst_secondary_model_scale': 0.9,
        'mdx_other_secondary_model_scale': 0.7,
        'mdx_bass_secondary_model_scale': 0.5,
        'mdx_drums_secondary_model_scale': 0.5,
        'is_save_all_outputs_ensemble': True,
        'is_append_ensemble_name': False,
        'chosen_audio_tool': AUDIO_TOOL_OPTIONS[0],
        'choose_algorithm': MANUAL_ENSEMBLE_OPTIONS[0],
        'time_stretch_rate': 2.0,
        'pitch_rate': 2.0,
        'is_gpu_conversion': False,
        'is_primary_stem_only': False,
        'is_secondary_stem_only': False,
        'is_testing_audio': False,
        'is_add_model_name': False,
        'is_accept_any_input': False,
        'is_task_complete': False,
        'is_normalization': False,
        'is_create_model_folder': False,
        'mp3_bit_set': '320k',
        'save_format': WAV,
        'wav_type_set': 'PCM_16',
        'user_code': '',
        'export_path': '',
        'input_paths': [],
        'lastDir': None,
        'export_path': '',
        'model_hash_table': None,
        'help_hints_var': False,
        'model_sample_mode': False,
        'model_sample_mode_duration': 30
}

# Message Box Text

INVALID_INPUT = 'Invalid Input', 'The input is invalid.\n\nPlease verify the input still exists or is valid and try again.'
INVALID_EXPORT = 'Invalid Export Directory', 'You have selected an invalid export directory.\n\nPlease make sure the selected directory still exists.'
INVALID_ENSEMBLE = 'Not Enough Models', 'You must select 2 or more models to run ensemble.'
INVALID_MODEL = 'No Model Chosen', 'You must select an model to continue.'
MISSING_MODEL = 'Model Missing', 'The selected model is missing or not valid.'
ERROR_OCCURED = 'Error Occured', '\n\nWould you like to open the error log for more details?\n'

# GUI Text Constants

BACK_TO_MAIN_MENU = 'Back to Main Menu'

# Help Hint Text

INTERNAL_MODEL_ATT = 'Internal model attribute. \n\n ***Do not change this setting if you are unsure!***'
STOP_HELP = 'Halts any running processes. \n A pop-up window will ask the user to confirm the action.'
SETTINGS_HELP = 'Opens the main settings guide. This window includes the \"Download Center\"'
COMMAND_TEXT_HELP = 'Provides information on the progress of the current process.'
SAVE_CURRENT_SETTINGS_HELP = 'Allows the user to open any saved settings or save the current application settings.'
CHUNKS_HELP = ('This option allows the user to reduce (or increase) RAM or V-RAM usage.\n\n' + \
                '• Smaller chunk sizes use less RAM or V-RAM but can also increase processing times.\n' + \
                '• Larger chunk sizes use more RAM or V-RAM but can also reduce processing times.\n' + \
                '• Selecting \"Auto\" calculates an appropriate chuck size based on how much RAM or V-RAM your system has.\n' + \
                '• Selecting \"Full\" will process the track as one whole chunk.\n' + \
                '• This option is only recommended for those with powerful PCs.\n' +\
                '• The default selection is \"Auto\".')
MARGIN_HELP = 'Selects the frequency margins to slice the chunks from.\n\n• The recommended margin size is 44100.\n• Other values can give unpredictable results.'
AGGRESSION_SETTING_HELP = ('This option allows you to set how strong the primary stem extraction will be.\n\n' + \
                           '• The range is 0-100.\n' + \
                           '• Higher values perform deeper extractions.\n' + \
                           '• The default is 10 for instrumental & vocal models.\n' + \
                           '• Values over 10 can result in muddy-sounding instrumentals for the non-vocal models')
WINDOW_SIZE_HELP = ('The smaller your window size, the better your conversions will be. \nHowever, a smaller window means longer conversion times and heavier resource usage.\n\n' + \
                    'Breakdown of the selectable window size values:\n' + \
                    '• 1024 - Low conversion quality, shortest conversion time, low resource usage.\n' + \
                    '• 512 - Average conversion quality, average conversion time, normal resource usage.\n' + \
                    '• 320 - Better conversion quality.')
DEMUCS_STEMS_HELP = ('Here, you can choose which stem to extract using the selected model.\n\n' +\
                     'Stem Selections:\n\n' +\
                     '• All Stems - Saves all of the stems the model is able to extract.\n' +\
                     '• Vocals - Pulls vocal stem only.\n' +\
                     '• Other - Pulls other stem only.\n' +\
                     '• Bass - Pulls bass stem only.\n' +\
                     '• Drums - Pulls drum stem only.\n')
SEGMENT_HELP = ('This option allows the user to reduce (or increase) RAM or V-RAM usage.\n\n' + \
                '• Smaller segment sizes use less RAM or V-RAM but can also increase processing times.\n' + \
                '• Larger segment sizes use more RAM or V-RAM but can also reduce processing times.\n' + \
                '• Selecting \"Default\" uses the recommended segment size.\n' + \
                '• It is recommended that you not use segments with \"Chunking\".')
ENSEMBLE_MAIN_STEM_HELP = 'Allows the user to select the type of stems they wish to ensemble.\n\nOptions:\n\n' +\
                          f'• {VOCAL_PAIR} - The primary stem will be the vocals and the secondary stem will be the the instrumental\n' +\
                          f'• {OTHER_PAIR} - The primary stem will be other and the secondary stem will be no other (the mixture without the \'other\' stem)\n' +\
                          f'• {BASS_PAIR} - The primary stem will be bass and the secondary stem will be no bass (the mixture without the \'bass\' stem)\n' +\
                          f'• {DRUM_PAIR} - The primary stem will be drums and the secondary stem will be no drums (the mixture without the \'drums\' stem)\n' +\
                          f'• {FOUR_STEM_ENSEMBLE} - This option will gather all the 4 stem Demucs models and ensemble all of the outputs.\n'
ENSEMBLE_TYPE_HELP = 'Allows the user to select the ensemble algorithm to be used to generate the final output.\n\nExample & Other Note:\n\n' +\
                     f'• {MAX_MIN} - If this option is chosen, the primary stem outputs will be processed through \nthe \'Max Spec\' algorithm, and the secondary stem will be processed through the \'Min Spec\' algorithm.\n' +\
                     f'• Only a single algorithm will be shown when the \'4 Stem Ensemble\' option is chosen.\n\nAlgorithm Details:\n\n' +\
                     f'• {MAX_SPEC} - This algorithm combines the final results and generates the highest possible output from them.\nFor example, if this algorithm were processing vocal stems, you would get the fullest possible \n' +\
                        'result making the ensembled vocal stem sound cleaner. However, it might result in more unwanted artifacts.\n' +\
                     f'• {MIN_SPEC} - This algorithm combines the results and generates the lowest possible output from them.\nFor example, if this algorithm were processing instrumental stems, you would get the cleanest possible result \n' +\
                        'result, eliminating more unwanted artifacts. However, the result might also sound \'muddy\' and lack a fuller sound.\n' +\
                     f'• {AUDIO_AVERAGE} - This algorithm simply combines the results and averages all of them together. \n'
ENSEMBLE_LISTBOX_HELP = 'List of the all the models available for the main stem pair selected.'
IS_GPU_CONVERSION_HELP = ('When checked, the application will attempt to use your GPU (if you have one).\n' +\
                         'If you do not have a GPU but have this checked, the application will default to your CPU.\n\n' +\
                         'Note: CPU conversions are much slower than those processed through the GPU.')
SAVE_STEM_ONLY_HELP = 'Allows the user to save only the selected stem.'
IS_NORMALIZATION_HELP = 'Normalizes output to prevent clipping.'
CROP_SIZE_HELP = '**Only compatible with select models only!**\n\n Setting should match training crop-size value. Leave as is if unsure.'
BATCH_SIZE_HELP = '**Only compatible with select models only!**\n\n Lower values allows for less resource usage but longer conversion times.'
IS_TTA_HELP = ('This option performs Test-Time-Augmentation to improve the separation quality.\n\n' +\
               'Note: Having this selected will increase the time it takes to complete a conversion')
IS_POST_PROCESS_HELP = ('This option can potentially identify leftover instrumental artifacts within the vocal outputs. \nThis option may improve the separation of some songs.\n\n' +\
                       'Note: Selecting this option can adversely affect the conversion process, depending on the track. Because of this, it is only recommended as a last resort.')
IS_HIGH_END_PROCESS_HELP = 'The application will mirror the missing frequency range of the output.'
SHIFTS_HELP = ('Performs multiple predictions with random shifts of the input and averages them.\n\n' +\
              '• The higher number of shifts, the longer the prediction will take. \n- Not recommended unless you have a GPU.')
OVERLAP_HELP = 'This option controls the amount of overlap between prediction windows (for Demucs one window is 10 seconds)'
IS_CHUNK_DEMUCS_HELP = '• Enables the using \"Chunks\".\n• We recommend you not enable this option with \"Split Mode\" enabled or with the Demucs v4 Models.'
IS_SPLIT_MODE_HELP = ('• Enables \"Segments\". \n• We recommend you not enable this option with \"Enable Chunks\".\n' +\
                      '• Deselecting this option is only recommended for those with powerful PCs or if using \"Chunk\" mode instead.')
IS_DEMUCS_COMBINE_STEMS_HELP = 'The application will create the secondary stem by combining the remaining stems \ninstead of inverting the primary stem with the mixture.'
COMPENSATE_HELP = 'Compensates the audio of the primary stems to allow for a better secondary stem.'
IS_DENOISE_HELP = '• This option removes a majority of the noise generated by the MDX-Net models.\n• The conversion will take nearly twice as long with this enabled.'
CLEAR_CACHE_HELP = 'Clears any user selected model settings for previously unrecognized models.'
IS_SAVE_ALL_OUTPUTS_ENSEMBLE_HELP = 'Enabling this option will keep all indivudual outputs generated by an ensemble.'
IS_APPEND_ENSEMBLE_NAME_HELP = 'The application will append the ensemble name to the final output \nwhen this option is enabled.'
DONATE_HELP = 'Takes the user to an external web-site to donate to this project!'
IS_INVERT_SPEC_HELP = '• This option may produce a better secondary stem.\n• Inverts primary stem with mixture using spectragrams instead of wavforms.\n• This inversion method is slightly slower.'
IS_TESTING_AUDIO_HELP = 'Appends a unique 10 digit number to output files so the user \ncan compare results with different settings.'
IS_MODEL_TESTING_AUDIO_HELP = 'Appends the model name to output files so the user \ncan compare results with different settings.'
IS_ACCEPT_ANY_INPUT_HELP = 'The application will accept any input when enabled, even if it does not have an audio format extension.\n\nThis is for experimental purposes, and having it enabled is not recommended.'
IS_TASK_COMPLETE_HELP = 'When enabled, chimes will be heard when a process completes or fails.'
IS_CREATE_MODEL_FOLDER_HELP = 'Two new directories will be generated for the outputs in \nthe export directory after each conversion.\n\n' +\
                              '• First directory - Named after the model.\n' +\
                              '• Second directory - Named after the track.\n\n' +\
                              '• Example: \n\n' +\
                              '─ Export Directory\n' +\
                              '   └── First Directory\n' +\
                              '           └── Second Directory\n' +\
                              '                    └── Output File(s)'
DELETE_YOUR_SETTINGS_HELP = 'This menu contains your saved settings. You will be asked to \nconfirm if you wish to delete the selected setting.'
SET_STEM_NAME_HELP = 'Choose the primary stem for the selected model.'
MDX_DIM_T_SET_HELP = INTERNAL_MODEL_ATT
MDX_DIM_F_SET_HELP = INTERNAL_MODEL_ATT
MDX_N_FFT_SCALE_SET_HELP = 'Set the N_FFT size the model was trained with.'
POPUP_COMPENSATE_HELP = f'Choose the appropriate voluem compensattion for the selected model\n\nReminder: {COMPENSATE_HELP}'
VR_MODEL_PARAM_HELP = 'Choose the parameters needed to run the selected model.'
CHOSEN_ENSEMBLE_HELP = 'Select saved enselble or save current ensemble.\n\nDefault Selections:\n\n• Save the current ensemble.\n• Clears all current model selections.'
CHOSEN_PROCESS_METHOD_HELP = 'Here, you choose between different Al networks and algorithms to process your track.\n\n' +\
                             'There are four options:\n\n' +\
                             '• VR Architecture - These models use magnitude spectrograms for Source Separation.\n' +\
                             '• MDX-Net - These models use Hybrid Spectrogram/Waveform for Source Separation.\n' +\
                             '• Demucs v3 - These models use Hybrid Spectrogram/Waveform for Source Separation.\n' +\
                             '• Ensemble Mode - Here, you can get the best results from multiple models and networks.\n' +\
                             '• Audio Tools - These are additional tools for added convenience.'
INPUT_FOLDER_ENTRY_HELP = 'Select Input:\n\nHere is where you select the audio files(s) you wish to process.'
OUTPUT_FOLDER_ENTRY_HELP = 'Select Output:\n\nHere is where you select the directory where your processed files are to be saved.'
INPUT_FOLDER_BUTTON_HELP = 'Open Input Folder Button: \n\nOpens the directory containing the selected input audio file(s).'
OUTPUT_FOLDER_BUTTON_HELP = 'Open Output Folder Button: \n\nOpens the selected output folder.'
CHOOSE_MODEL_HELP = 'Each process method comes with its own set of options and models.\n\nHere is where you choose the model associated with the selected process method.'
FORMAT_SETTING_HELP = 'Save outputs as '
SECONDARY_MODEL_ACTIVATE_HELP = 'When enabled, the application will run an additional inference with the selected model(s) above.'
SECONDARY_MODEL_HELP = 'Choose the secondary model associated with this stem you wish to run with the current process method.'
SECONDARY_MODEL_SCALE_HELP = 'The scale determines how the final audio outputs will be averaged between the primary and secondary models.\n\nFor example:\n\n' +\
                             '• 10% - 10 percent of the main model result will be factored into the final result.\n' +\
                             '• 50% - The results from the main and secondary models will be averaged evenly.\n' +\
                             '• 90% - 90 percent of the main model result will be factored into the final result.'
PRE_PROC_MODEL_ACTIVATE_HELP = 'The application will run an inference with the selected model above, pulling only the instrumental stem when enabled. \nFrom there, all of the non-vocal stems will be pulled from the generated instrumental.\n\nNotes:\n\n' +\
                               '• This option can significantly reduce vocal bleed within the non-vocal stems.\n' +\
                               '• It is only available in Demucs.\n' +\
                               '• It is only compatible with non-vocal and non-instrumental stem outputs.\n' +\
                               '• This will increase thetotal processing time.\n' +\
                               '• Only VR and MDX-Net Vocal or Instrumental models are selectable above.'

AUDIO_TOOLS_HELP = 'Here, you choose between different audio tools to process your track.\n\n' +\
                               '• Manual Ensemble - You must have 2 or more files selected as your inputs. Allows the user to run their tracks through \nthe same algorithms used in Ensemble Mode.\n' +\
                               '• Align Inputs - You must have exactly 2 files selected as your inputs. The second input will be aligned with the first input.\n' +\
                               '• Time Stretch - The user can speed up or slow down the selected inputs.\n' +\
                               '• Change Pitch - The user can change the pitch for the selected inputs.\n'
PRE_PROC_MODEL_INST_MIX_HELP = 'When enabled, the application will generate a third output without the selected stem and vocals.'         
MODEL_SAMPLE_MODE_HELP = 'Allows the user to process only part of a track to sample settings or a model without \nrunning a full conversion.\n\nNotes:\n\n' +\
                         '• The number in the parentheses is the current number of seconds the generated sample will be.\n' +\
                         '• You can choose the number of seconds to extract from the track in the \"Additional Settings\" menu.'
                         
# Warning Messages

STORAGE_ERROR = 'Insufficient Storage', 'There is not enough storage on main drive to continue. Your main drive must have at least 3 GB\'s of storage in order for this application function properly. \n\nPlease ensure your main drive has at least 3 GB\'s of storage and try again.\n\n'
STORAGE_WARNING = 'Available Storage Low', 'Your main drive is running low on storage. Your main drive must have at least 3 GB\'s of storage in order for this application function properly.\n\n'
CONFIRM_WARNING = '\nAre you sure you wish to continue?'
PROCESS_FAILED = 'Process failed, please see error log\n'
EXIT_PROCESS_ERROR = 'Active Process', 'Please stop the active process or wait for it to complete before you exit.'
EXIT_HALTED_PROCESS_ERROR = 'Halting Process', 'Please wait for the application to finish halting the process before exiting.'
EXIT_DOWNLOAD_ERROR = 'Active Download', 'Please stop the download or wait for it to complete before you exit.'
SET_TO_DEFAULT_PROCESS_ERROR = 'Active Process', 'You cannot reset all of the application settings during an active process.'
SET_TO_ANY_PROCESS_ERROR = 'Active Process', 'You cannot reset the application settings during an active process.'
RESET_ALL_TO_DEFAULT_WARNING = 'Reset Settings Confirmation', 'All application settings will be set to factory default.\n\nAre you sure you wish to continue?'
AUDIO_VERIFICATION_CHECK = lambda i, e:f'++++++++++++++++++++++++++++++++++++++++++++++++++++\n\nBroken File Removed: \n\n{i}\n\nError Details:\n\n{e}\n++++++++++++++++++++++++++++++++++++++++++++++++++++'

# Separation Text

LOADING_MODEL = 'Loading model...'
INFERENCE_STEP_1 = 'Running inference...'
INFERENCE_STEP_1_SEC = 'Running inference (secondary model)...'
INFERENCE_STEP_1_4_STEM = lambda stem:f'Running inference (secondary model for {stem})...'
INFERENCE_STEP_1_PRE = 'Running inference (pre-process model)...'
INFERENCE_STEP_2_PRE = lambda pm, m:f'Loading pre-process model ({pm}: {m})...'
INFERENCE_STEP_2_SEC = lambda pm, m:f'Loading secondary model ({pm}: {m})...'
INFERENCE_STEP_2_SEC_CACHED_MODOEL = lambda pm, m:f'Secondary model ({pm}: {m}) cache loaded.\n'
INFERENCE_STEP_2_PRE_CACHED_MODOEL = lambda pm, m:f'Pre-process model ({pm}: {m}) cache loaded.\n'
INFERENCE_STEP_2_SEC_CACHED = 'Loading cached secondary model source(s)... Done!\n'
INFERENCE_STEP_2_PRIMARY_CACHED = 'Model cache loaded.\n'
INFERENCE_STEP_2 = 'Inference complete.'
SAVING_STEM = 'Saving ', ' stem...'
SAVING_ALL_STEMS = 'Saving all stems...'
ENSEMBLING_OUTPUTS = 'Ensembling outputs...'
DONE = ' Done!\n'
ENSEMBLES_SAVED = 'Ensembled outputs saved!\n\n'
NEW_LINES = "\n\n"
NEW_LINE = "\n"
NO_LINE = ''

# Widget Placements

MAIN_ROW_Y = -15, -17
MAIN_ROW_X = -4, 21
MAIN_ROW_WIDTH = -53
MAIN_ROW_2_Y = -15, -17
MAIN_ROW_2_X = -28, 1
CHECK_BOX_Y = 0
CHECK_BOX_X = 20
CHECK_BOX_WIDTH = -50
CHECK_BOX_HEIGHT = 2
LEFT_ROW_WIDTH = -10
LABEL_HEIGHT = -5
OPTION_HEIGHT = 7
LOW_MENU_Y = 18, 16
FFMPEG_EXT = (".aac", ".aiff", ".alac" ,".flac", ".mov", ".mp4", 
              ".m4a", ".mp2", ".mp3", ".mp4", ".mpc", ".mpc8", 
              ".mpeg", ".ogg", ".tta", ".wav", ".wma")
FFMPEG_MORE_EXT = (".aa", ".aac", ".ac3", ".aiff", ".alac", ".avi", ".f4v",".flac", ".flic", ".flv",
              ".m4v",".mlv", ".mov", ".mp4", ".m4a", ".mp2", ".mp3", ".mp4", ".mpc", ".mpc8", 
              ".mpeg", ".ogg", ".tta", ".tty", ".vcd", ".wav", ".wma")
ANY_EXT = ""

# Secondary Menu Constants

VOCAL_PAIR_PLACEMENT = 1, 2, 3, 4
OTHER_PAIR_PLACEMENT = 5, 6, 7, 8
BASS_PAIR_PLACEMENT = 9, 10, 11, 12
DRUMS_PAIR_PLACEMENT = 13, 14, 15, 16
LICENSE_TEXT = lambda a, p:f'Current Application Version: Ultimate Vocal Remover {a}\n' +\
                f'Current Patch Version: {p}\n\n' +\
                'Copyright (c) 2022 Ultimate Vocal Remover\n\n' +\
                'UVR is free and open-source, but MIT licensed. Please credit us if you use our\n' +\
                'models or code for projects unrelated to UVR.\n\n' +\
                '• This application is intended for those running macOS Monterey and above.\n' +\
                '• Application functionality for systems running macOS BigSur or lower.\n' +\
                '• Application functionality for older or budget Mac systems is not guaranteed.\n\n' +\
                'This bundle contains the UVR interface, Python, PyTorch, and other\n' +\
                'dependencies needed to run the application effectively.\n\n' +\
                'Website Links: This application, System or Service(s) may contain links to\n' +\
                'other websites and downloads, and they are solely provided to you as an\n' +\
                'additional convenience. You understand and acknowledge that by clicking\n' +\
                'or activating such links you are accessing a site or service outside of\n' +\
                'this application, and that we do not screen, review, approve, or otherwise\n' +\
                'endorse any content or information contained in these linked websites.\n' +\
                'You acknowledge and agree that we, our affiliates and partners are not\n' +\
                'responsible for the contents of any of these linked websites, including\n' +\
                'the accuracy or availability of information provided by the linked websites,\n' +\
                'and we make no representations or warranties regarding your use of\n' +\
                'the linked websites.\n\n' +\
                'This application is MIT Licensed\n\n' +\
                'Permission is hereby granted, free of charge, to any person obtaining a copy\n' +\
                'of this software and associated documentation files (the "Software"), to deal\n' +\
                'in the Software without restriction, including without limitation the rights\n' +\
                'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n' +\
                'copies of the Software, and to permit persons to whom the Software is\n' +\
                'furnished to do so, subject to the following conditions:\n\n' +\
                'The above copyright notice and this permission notice shall be included in all\n' +\
                'copies or substantial portions of the Software.\n\n' +\
                'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n' +\
                'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n' +\
                'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n' +\
                'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n' +\
                'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n' +\
                'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n' +\
                'SOFTWARE.'

# Font Sizes

FONT_SIZE_0 = 9
FONT_SIZE_1 = 11
FONT_SIZE_2 = 11
FONT_SIZE_3 = 12
FONT_SIZE_4 = 13
FONT_SIZE_5 = 14
FONT_SIZE_6 = 17

GEN_SETTINGS_WIDTH = 17
# FONT_SIZE_0 = 7
# FONT_SIZE_1 = 8
# FONT_SIZE_2 = 9
# FONT_SIZE_3 = 10
# FONT_SIZE_4 = 11
# FONT_SIZE_5 = 12
# FONT_SIZE_6 = 15

# INTERNAL_MODEL_ATT = '内部模型属性 \n\n ***如果不确定，请勿更改此设置！***'
# STOP_HELP = '停止任何正在运行的进程 \n 弹出窗口将要求用户确认操作'
# SETTINGS_HELP = '打开设置指南此窗口包括\"下载中心\"'
# COMMAND_TEXT_HELP = '提供有关当前进程进度的信息'
# SAVE_CURRENT_SETTINGS_HELP = '允许用户打开任何保存的设置或保存当前应用程序设置'
# CHUNKS_HELP = ('此选项允许用户减少（或增加）RAM或VRAM\n\n' + \
#                 '• 较小的块大小使用较少的RAM或VRAM，但也会增加处理时间\n' + \
#                 '• 较大的块大小使用更多的RAM或VRAM，但也可以减少处理时间\n' + \
#                 '• 选择“自动”可根据系统的RAM或VRAM大小计算适当的运行内存\n' + \
#                 '• 选择“完整”将使用全部电脑可用资源处理曲目\n' + \
#                 '• 此选项仅适用于具有强大pc的用户,不要对自己电脑过于自信\n' +\
#                 '• 默认选择为“自动”.')
# MARGIN_HELP = '选择要从中分割块的频率\n\n- 建议的频率大小为44100\n- 其他值可能会产生不可预测的结果'
# AGGRESSION_SETTING_HELP = ('该选项允许您设置主轨道提取的强度\n\n' + \
#                            '• 范围为0-100\n' + \
#                            '• 值越高，提取程度越高\n' + \
#                            '• 乐器和声乐模型的默认值为10\n' + \
#                            '• 超过10的值可能会导致非发声模型的乐器发出浑浊的声音')
# WINDOW_SIZE_HELP = ('分块大小越小，转换效果越好 \n然而，较小的分块意味着更长的转换时间和更重的资源使用\n\n' + \
#                     '可选窗口大小值的细分：\n' + \
#                     '• 1024 - 转换质量低，转换时间短，资源使用率低\n' + \
#                     '• 512 - 平均转换质量、平均转换时间、正常资源使用\n' + \
#                     '• 320 - 更好的转换质量')
# DEMUCS_STEMS_HELP = ('在这里，您可以选择使用所选模型提取某个轨道\n\n' +\
#                      '轨道选择：\n\n' +\
#                      '• All Stems - 保存模型能够提取的所有轨道.\n' +\
#                      '• Vocals -仅人声轨道.\n' +\
#                      '• Other - 仅其他轨道.\n' +\
#                      '• Bass - 仅贝斯轨道.\n' +\
#                      '• Drums - 仅鼓轨道.\n')
# SEGMENT_HELP = ('此选项允许用户减少（或增加）RAM或VRAM使用\n\n' + \
#                 '• 较小的段大小使用较少的RAM或VRAM，但也会增加处理时间.\n' + \
#                 '• 较大的段大小使用更多的RAM或VRAM，但也可以减少处理时间\n' + \
#                 '• 选择“默认值”使用建议的段大小\n' + \
#                 '• 建议不要使用带有“分段”的段".')
# ENSEMBLE_MAIN_STEM_HELP = '允许用户选择要集成的阀杆类型\n\n示例：主阀杆/次阀杆'
# ENSEMBLE_TYPE_HELP = '允许用户选择用于生成最终输出的集成算法'
# ENSEMBLE_LISTBOX_HELP = '所选主阀杆对的所有可用型号列表'
# IS_GPU_CONVERSION_HELP = ('选中后，应用程序将尝试使用您的GPU（如果您有）.\n' +\
#                          '如果您没有GPU，但选中了此项，则应用程序将默认为CPU\n\n' +\
#                          '注：CPU转换比通过GPU处理的转换慢得多.')
# SAVE_STEM_ONLY_HELP = '允许用户仅保存选定的阀杆'
# IS_NORMALIZATION_HELP = '规格化输出以防止剪裁'
# CROP_SIZE_HELP = '**仅与部分型号兼容！**\n\n 设置应与训练作物大小值相匹配，如果不确定，则保持原样'
# BATCH_SIZE_HELP = '**仅与部分型号兼容！**\n\n 值越低，资源使用量越少，但转换时间越长'
# IS_TTA_HELP = ('此选项执行测试时间增强以提高分离质量\n\n' +\
#                '注意：选择此选项将增加完成转换所需的时间')
# IS_POST_PROCESS_HELP = ('该选项可以潜在地识别声音输出中残留的乐器伪影 \n此选项可能会改进某些歌曲的分离.\n\n' +\
#                        '注意：选择此选项可能会对转换过程产生不利影响，具体取决于曲目。因此，建议将其作为最后救命稻草')
# IS_HIGH_END_PROCESS_HELP = '应用程序将镜像输出的缺失频率范围'
# SHIFTS_HELP = ('使用输入的随机移位执行多个预测，并对其进行平均.\n\n' +\
#               '• 移位次数越多，预测所需时间越长\n- 除非您有GPU最低8g，否则别瞎选电脑爆炸概不负责')
# OVERLAP_HELP = '此选项控制预测窗口之间的重叠量（对于demucs，一个窗口为10秒）'
# IS_CHUNK_DEMUCS_HELP = '启用使用“块”.\n\n请注意：我们建议您不要在启用“拆分模式”的情况下启用此选项'
# IS_SPLIT_MODE_HELP = ('启用“分段”. \n\n请注意：我们建议您不要使用“启用区块”来启用此选项.\n' +\
#                      '仅建议具有强大pc或使用“块”模式.再次提醒别瞎点,要对自己电脑负责.别选!不负责任的狗男人')
# IS_DEMUCS_COMBINE_STEMS_HELP = '应用程序将通过组合剩余的阀杆来创建第二阀杆\n而不是用混合物反转主茎'
# COMPENSATE_HELP = '补偿主杆的音频，以获得更好的辅助杆'
# IS_DENOISE_HELP = '该选项消除了MDX-NET模型产生的大部分噪声\n\n请注意：启用此选项后，转换所需的时间几乎是原来的两倍'
# CLEAR_CACHE_HELP = '清除以前无法识别的模型的任何用户选择的模型设置'
# IS_SAVE_ALL_OUTPUTS_ENSEMBLE_HELP = '启用此选项将保留集成生成的所有单独输出'
# IS_APPEND_ENSEMBLE_NAME_HELP = '应用程序将在最终输出中附加集成名称 \n启用此选项时'
# DONATE_HELP = '将用户带到外部网站为该项目捐款！'
# IS_INVERT_SPEC_HELP = '相反，使用光谱图用混合物反转主阀杆 \n这种反演方法稍慢'
# IS_TESTING_AUDIO_HELP = '在输出文件中附加一个唯一的10位数字，以便用户\nc不同设置的比较结果'
# IS_CREATE_MODEL_FOLDER_HELP = '将为中的输出生成两个新目录 \n每次转换后的导出目录'
# DELETE_YOUR_SETTINGS_HELP = '此菜单包含您保存的设置，系统将要求您\n确认是否要删除所选设置'
# SET_STEM_NAME_HELP = '为所选模型选择主阀杆'
# MDX_DIM_T_SET_HELP = INTERNAL_MODEL_ATT
# MDX_DIM_F_SET_HELP = INTERNAL_MODEL_ATT
# MDX_N_FFT_SCALE_SET_HELP = '设置训练模型的N_FFT大小'
# POPUP_COMPENSATE_HELP = f'为所选模型选择适当的体积补偿\n\n提醒： {COMPENSATE_HELP}'
# VR_MODEL_PARAM_HELP = '选择运行所选模型所需的参数'
# CHOSEN_ENSEMBLE_HELP = '选择保存的集合或保存当前集合\n\n默认选择：\n\n- 保存当前集合\n- 清除所有当前模型选择'
# CHOSEN_PROCESS_METHOD_HELP = '选择要运行曲目的进程'
# FORMAT_SETTING_HELP = '将输出另存为'
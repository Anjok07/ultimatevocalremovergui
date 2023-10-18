import platform

#Platform Details
OPERATING_SYSTEM = platform.system()
SYSTEM_ARCH = platform.platform()
SYSTEM_PROC = platform.processor()
ARM = 'arm'

is_macos = False

CPU = 'cpu'
CUDA_DEVICE = 'cuda'
DIRECTML_DEVICE = "privateuseone"

#MAIN_FONT_NAME = "Century Gothic"
OPT_SEPARATOR_SAVE = '─'*25
BG_COLOR = '#0e0e0f'
FG_COLOR = '#13849f'

#Model Types
VR_ARCH_TYPE = 'VR Arc'
MDX_ARCH_TYPE = 'MDX-Net'
DEMUCS_ARCH_TYPE = 'Demucs'
VR_ARCH_PM = 'VR Architecture'
ENSEMBLE_MODE = 'Ensemble Mode'
ENSEMBLE_STEM_CHECK = 'Ensemble Stem'
SECONDARY_MODEL = 'Secondary Model'
DEMUCS_6_STEM_MODEL = 'htdemucs_6s'
DEFAULT = "Default"
ALIGNMENT_TOOL = 'Alignment Tool Options'

SINGLE_FILE = 'SINGLE_FILE'
MULTIPLE_FILE = 'MULTI_FILE'
MAIN_MULTIPLE_FILE = 'MAIN_MULTI_FILE'
CHOOSE_EXPORT_FIR = 'CHOOSE_EXPORT_FIR'

DUAL = "dual"
FOUR_STEM = "fourstem"
ANY_STEM = "Any Stem"

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
IS_KARAOKEE = "is_karaoke"
IS_BV_MODEL = "is_bv_model"
IS_BV_MODEL_REBAL = "is_bv_model_rebalanced"
INPUT_STEM_NAME = 'Input Stem Name'

#Menu Options

AUTO_SELECT = 'Auto'

#LINKS
DOWNLOAD_CHECKS = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
MDX_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json"
VR_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json"
MDX23_CONFIG_CHECKS = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"
BULLETIN_CHECK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/bulletin.txt"

DEMUCS_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/demucs_model_data/model_name_mapper.json"
MDX_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_name_mapper.json"

DONATE_LINK_BMAC = "https://www.buymeacoffee.com/uvr5"
DONATE_LINK_PATREON = "https://www.patreon.com/uvr"

#DOWNLOAD REPOS
NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UPDATE_REPO = "https://github.com/TRvlvr/model_repo/releases/download/uvr_update_patches/"

UPDATE_MAC_ARM_REPO = "https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_arm64.dmg"
UPDATE_MAC_X86_64_REPO = "https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_x86_64.dmg"
UPDATE_LINUX_REPO = "https://github.com/Anjok07/ultimatevocalremovergui#linux-installation"

ISSUE_LINK = 'https://github.com/Anjok07/ultimatevocalremovergui/issues/new'
VIP_REPO = b'\xf3\xc2W\x19\x1foI)\xc2\xa9\xcc\xb67(Z\xf5',\
           b'gAAAAABjQAIQ-NpNMMxMedpKHHb7ze_nqB05hw0YhbOy3pFzuzDrfqumn8_qvraxEoUpZC5ZXC0gGvfDxFMqyq9VWbYKlA67SUFI_wZB6QoVyGI581vs7kaGfUqlXHIdDS6tQ_U-BfjbEAK9EU_74-R2zXjz8Xzekw=='
NO_CODE = 'incorrect_code'

#Extensions
ONNX = '.onnx'
CKPT = '.ckpt'
CKPT_C = '.ckptc'
YAML = '.yaml'
PTH = '.pth'
TH_EXT = '.th'
JSON = '.json'

#GUI Buttons
START_PROCESSING = 'Start Processing'
WAIT_PROCESSING = 'Please wait...'
STOP_PROCESSING = 'Halting process, please wait...'
LOADING_MODELS = 'Loading models...'

#---Messages and Logs----

MISSING_MODEL = 'missing'
MODEL_PRESENT = 'present'

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
BRASS_STEM = 'Brass'
WIND_INST_STEM = 'Wind Inst'
NO_OTHER_STEM = 'No Other'
NO_BASS_STEM = 'No Bass'
NO_DRUM_STEM = 'No Drums'
NO_GUITAR_STEM = 'No Guitar'
NO_PIANO_STEM = 'No Piano'
NO_SYNTH_STEM = 'No Synthesizer'
NO_STRINGS_STEM = 'No Strings'
NO_WOODWINDS_STEM = 'No Woodwinds'
NO_WIND_INST_STEM = 'No Wind Inst'
NO_BRASS_STEM = 'No Brass'
PRIMARY_STEM = 'Primary Stem'
SECONDARY_STEM = 'Secondary Stem'
LEAD_VOCAL_STEM = 'lead_only'
BV_VOCAL_STEM = 'backing_only'
LEAD_VOCAL_STEM_I = 'with_lead_vocals'
BV_VOCAL_STEM_I = 'with_backing_vocals'
LEAD_VOCAL_STEM_LABEL = 'Lead Vocals'
BV_VOCAL_STEM_LABEL = 'Backing Vocals'

VOCAL_STEM_ONLY = f'{VOCAL_STEM} Only'
INST_STEM_ONLY = f'{INST_STEM} Only'
PRIMARY_STEM_ONLY = f'{PRIMARY_STEM} Only'

IS_SAVE_INST_ONLY = f'save_only_inst'
IS_SAVE_VOC_ONLY = f'save_only_voc'

DEVERB_MAPPER = {'Main Vocals Only':VOCAL_STEM, 
                 'Lead Vocals Only':LEAD_VOCAL_STEM_LABEL, 
                 'Backing Vocals Only':BV_VOCAL_STEM_LABEL, 
                 'All Vocal Types':'ALL'}

BALANCE_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
                        BASS_STEM:0,
                        DRUM_STEM:1,
                        OTHER_STEM:2,
                        VOCAL_STEM:3,
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
                 BRASS_STEM, 
                 WIND_INST_STEM)

STEM_SET_MENU_ONLY = list(STEM_SET_MENU) + [OPT_SEPARATOR_SAVE, INPUT_STEM_NAME]

STEM_SET_MENU_2 = (
                 OTHER_STEM, 
                 BASS_STEM, 
                 DRUM_STEM, 
                 GUITAR_STEM, 
                 PIANO_STEM, 
                 SYNTH_STEM, 
                 STRINGS_STEM, 
                 WOODWINDS_STEM, 
                 BRASS_STEM, 
                 WIND_INST_STEM,
                 "Noise",
                 "Reverb")

STEM_PAIR_MAPPER = {
            VOCAL_STEM: INST_STEM,
            INST_STEM: VOCAL_STEM,
            LEAD_VOCAL_STEM: BV_VOCAL_STEM,
            BV_VOCAL_STEM: LEAD_VOCAL_STEM,
            PRIMARY_STEM: SECONDARY_STEM}

STEM_PAIR_MAPPER_FULL = {
            VOCAL_STEM: INST_STEM,
            INST_STEM: VOCAL_STEM,
            OTHER_STEM: NO_OTHER_STEM,
            BASS_STEM: NO_BASS_STEM,
            DRUM_STEM: NO_DRUM_STEM,
            GUITAR_STEM: NO_GUITAR_STEM,
            PIANO_STEM: NO_PIANO_STEM,
            SYNTH_STEM: NO_SYNTH_STEM,
            STRINGS_STEM: NO_STRINGS_STEM,
            WOODWINDS_STEM: NO_WOODWINDS_STEM,
            BRASS_STEM: NO_BRASS_STEM,
            WIND_INST_STEM: NO_WIND_INST_STEM,
            NO_OTHER_STEM: OTHER_STEM,
            NO_BASS_STEM: BASS_STEM,
            NO_DRUM_STEM: DRUM_STEM,
            NO_GUITAR_STEM: GUITAR_STEM,
            NO_PIANO_STEM: PIANO_STEM,
            NO_SYNTH_STEM: SYNTH_STEM,
            NO_STRINGS_STEM: STRINGS_STEM,
            NO_WOODWINDS_STEM: WOODWINDS_STEM,
            NO_BRASS_STEM: BRASS_STEM,
            NO_WIND_INST_STEM: WIND_INST_STEM,
            PRIMARY_STEM: SECONDARY_STEM}

NO_STEM = "No "

NON_ACCOM_STEMS = (
            VOCAL_STEM,
            OTHER_STEM,
            BASS_STEM,
            DRUM_STEM,
            GUITAR_STEM,
            PIANO_STEM,
            SYNTH_STEM,
            STRINGS_STEM,
            WOODWINDS_STEM,
            BRASS_STEM,
            WIND_INST_STEM)

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
MULTI_STEM_ENSEMBLE = 'Multi-stem Ensemble'

ENSEMBLE_MAIN_STEM = (CHOOSE_STEM_PAIR, VOCAL_PAIR, OTHER_PAIR, DRUM_PAIR, BASS_PAIR, FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE)

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

BATCH_MODE = 'Batch Mode'
BETA_VERSION = 'BETA'
DEF_OPT = 'Default'
USER_INPUT = "User Input"
OPT_SEPARATOR = '─'*65

CHUNKS = (AUTO_SELECT, '1', '5', '10', '15', '20', 
          '25', '30', '35', '40', '45', '50', 
          '55', '60', '65', '70', '75', '80', 
          '85', '90', '95', 'Full')

BATCH_SIZE = (DEF_OPT, '2', '3', '4', '5', 
          '6', '7', '8', '9', '10')

VOL_COMPENSATION = (AUTO_SELECT, '1.035', '1.08')

MARGIN_SIZE = ('44100', '22050', '11025')

AUDIO_TOOLS = 'Audio Tools'

MANUAL_ENSEMBLE = 'Manual Ensemble'
TIME_STRETCH = 'Time Stretch'
CHANGE_PITCH = 'Change Pitch'
ALIGN_INPUTS = 'Align Inputs'
MATCH_INPUTS = 'Matchering'
COMBINE_INPUTS = 'Combine Inputs'

if OPERATING_SYSTEM == 'Windows' or OPERATING_SYSTEM == 'Darwin':  
   AUDIO_TOOL_OPTIONS = (MANUAL_ENSEMBLE, TIME_STRETCH, CHANGE_PITCH, ALIGN_INPUTS, MATCH_INPUTS)
else:
   AUDIO_TOOL_OPTIONS = (MANUAL_ENSEMBLE, ALIGN_INPUTS, MATCH_INPUTS)

MANUAL_ENSEMBLE_OPTIONS = (MIN_SPEC, MAX_SPEC, AUDIO_AVERAGE, COMBINE_INPUTS)

PROCESS_METHODS = (VR_ARCH_PM, MDX_ARCH_TYPE, DEMUCS_ARCH_TYPE, ENSEMBLE_MODE, AUDIO_TOOLS)

DEMUCS_SEGMENTS = (DEF_OPT, '1', '5', '10', '15', '20', 
                  '25', '30', '35', '40', '45', '50', 
                  '55', '60', '65', '70', '75', '80', 
                  '85', '90', '95', '100')

DEMUCS_SHIFTS = (0, 1, 2, 3, 4, 5, 
                 6, 7, 8, 9, 10, 11, 
                 12, 13, 14, 15, 16, 17, 
                 18, 19, 20)
SEMI_DEF = ['0']
SEMITONE_SEL = (-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12)

NOUT_SEL = (8, 16, 32, 48, 64)
NOUT_LSTM_SEL = (64, 128)

DEMUCS_OVERLAP = (0.25, 0.50, 0.75, 0.99)
MDX_OVERLAP = (DEF_OPT, 0.25, 0.50, 0.75, 0.99)
MDX23_OVERLAP = range(2, 51)
VR_AGGRESSION = range(0, 51)

TIME_WINDOW_MAPPER = {
            "None": None,
            "1": [0.0625],
            "2": [0.125],
            "3": [0.25],
            "4": [0.5],
            "5": [0.75],
            "6": [1],
            "7": [2],
            "Shifts: Low": [0.0625, 0.5],
            "Shifts: Medium": [0.0625, 0.125, 0.5],
            "Shifts: High": [0.0625, 0.125, 0.25, 0.5]
            #"Shifts: Very High": [0.0625, 0.125, 0.25, 0.5, 0.75, 1],
}

INTRO_MAPPER = {
            "Default": [10],
            "1": [8],
            "2": [6],
            "3": [4],
            "4": [2],
            "Shifts: Low": [1, 10],
            "Shifts: Medium": [1, 10, 8],
            "Shifts: High": [1, 10, 8, 6, 4]
            }

VOLUME_MAPPER = {
            "None": (0, [0]),
            "Low": (-4, range(0, 8)),
            "Medium": (-6, range(0, 12)),
            "High": (-6, [x * 0.5 for x in range(0, 25)]),
            "Very High": (-10, [x * 0.5 for x in range(0, 41)])}
            #"Max": (-10, [x * 0.3 for x in range(0, int(20 / 0.3) + 1)])}

PHASE_MAPPER = {
            "None": [0],
            "Shifts Low": [0, 180],
            "Shifts Medium": [0],
            "Shifts High": [0],
            "Shifts Very High": [0],}

NONE_P = "None"
VLOW_P = "Shifts: Very Low"
LOW_P = "Shifts: Low"
MED_P = "Shifts: Medium"
HIGH_P = "Shifts: High"
VHIGH_P = "Shifts: Very High"
VMAX_P = "Shifts: Maximum"

PHASE_SHIFTS_OPT = {
                     NONE_P:190,
                     VLOW_P:180,
                     LOW_P:90,
                     MED_P:45,
                     HIGH_P:20,
                     VHIGH_P:10,
                     VMAX_P:1,}

VR_WINDOW = ('320', '512','1024')
VR_CROP = ('256', '512', '1024')
POST_PROCESSES_THREASHOLD_VALUES = ('0.1', '0.2', '0.3')

MDX_POP_PRO = ('MDX-NET_Noise_Profile_14_kHz', 'MDX-NET_Noise_Profile_17_kHz', 'MDX-NET_Noise_Profile_Full_Band')
MDX_POP_STEMS = ('Vocals', 'Instrumental', 'Other', 'Drums', 'Bass')
MDX_POP_NFFT = ('4096', '5120', '6144', '7680', '8192', '16384')
MDX_POP_DIMF = ('2048', '3072', '4096')
DENOISE_NONE, DENOISE_S, DENOISE_M = 'None', 'Standard', 'Denoise Model'
MDX_DENOISE_OPTION = [DENOISE_NONE, DENOISE_S, DENOISE_M]
MDX_SEGMENTS = list(range(32, 4000+1, 32))

SAVE_ENSEMBLE = 'Save Ensemble'
CLEAR_ENSEMBLE = 'Clear Selection(s)'
MENU_SEPARATOR = 35*'•'
CHOOSE_ENSEMBLE_OPTION = 'Choose Option'
ALL_TYPES = 'ALL'
INVALID_ENTRY = 'Invalid Input, Please Try Again'
ENSEMBLE_INPUT_RULE = '1. Only letters, numbers, spaces, and dashes allowed.\n2. No dashes or spaces at the start or end of input.'
STEM_INPUT_RULE = '1. Only words with no spaces are allowed.\n2. No spaces, numbers, or special characters.'

ENSEMBLE_OPTIONS = [OPT_SEPARATOR_SAVE, SAVE_ENSEMBLE, CLEAR_ENSEMBLE]
ENSEMBLE_CHECK = 'ensemble check'
KARAOKEE_CHECK = 'kara check'

AUTO_PHASE = "Automatic"
POSITIVE_PHASE = "Positive Phase"
NEGATIVE_PHASE = "Negative Phase"
OFF_PHASE = "Native Phase"

ALIGN_PHASE_OPTIONS = [AUTO_PHASE, POSITIVE_PHASE, NEGATIVE_PHASE, OFF_PHASE]

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
OPTION_LIST = [VR_OPTION, MDX_OPTION, DEMUCS_OPTION, ENSEMBLE_OPTION, ALIGNMENT_TOOL, HELP_OPTION, ERROR_OPTION]

#Menu Strings
VR_MENU ='VR Menu'
DEMUCS_MENU ='Demucs Menu'
MDX_MENU ='MDX-Net Menu'
ENSEMBLE_MENU ='Ensemble Menu'
HELP_MENU ='Help Menu'
ERROR_MENU ='Error Log'
INPUTS_MENU ='Inputs Menu'
ALIGN_MENU ='Align Menu'

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
GPU_DEVICE_NUM_OPTS = (DEFAULT, '0', '1', '2', '3', '4', '5', '6', '7', '8')

SELECT_SAVED_SET = 'Choose Option'
SAVE_SETTINGS = 'Save Current Settings'
RESET_TO_DEFAULT = 'Reset to Default'
RESET_FULL_TO_DEFAULT = 'Reset to Default'
RESET_PM_TO_DEFAULT = 'Reset All Application Settings to Default'

SAVE_SET_OPTIONS = [OPT_SEPARATOR_SAVE, SAVE_SETTINGS, RESET_TO_DEFAULT]

TIME_PITCH = ('1.0', '2.0', '3.0', '4.0')
TIME_TEXT = '_time_stretched'
PITCH_TEXT = '_pitch_shifted'

#RegEx Input Validation
REG_PITCH = r'^[-+]?(1[0]|[0-9]([.][0-9]*)?)$'
REG_TIME = r'^[+]?(1[0]|[0-9]([.][0-9]*)?)$'
REG_COMPENSATION = r'\b^(1[0]|[0-9]([.][0-9]*)?|Auto|None)$\b'
REG_THES_POSTPORCESS = r'\b^([0]([.][0-9]{0,6})?)$\b'
REG_CHUNKS = r'\b^(200|1[0-9][0-9]|[1-9][0-9]?|Auto|Full)$\b'
REG_CHUNKS_DEMUCS = r'\b^(200|1[0-9][0-9]|[1-9][0-9]?|Auto|Full)$\b'
REG_MARGIN = r'\b^[0-9]*$\b'
REG_SEGMENTS = r'\b^(200|1[0-9][0-9]|[1-9][0-9]?|Default)$\b'
REG_SAVE_INPUT = r'\b^([a-zA-Z0-9 -]{0,25})$\b'
REG_INPUT_STEM_NAME = r'^(Wind Inst|[a-zA-Z]{1,25})$'
REG_SEMITONES = r'^-?(20\.00|[01]?\d(\.\d{1,2})?|20)$'
REG_AGGRESSION = r'^[-+]?[0-9]\d*?$'
REG_WINDOW = r'\b^[0-9]{0,4}$\b'
REG_SHIFTS = r'\b^[0-9]*$\b'
REG_BATCHES = r'\b^([0-9]*?|Default)$\b'
REG_OVERLAP = r'\b^([0]([.][0-9]{0,6})?|Default)$\b'#r"(Default|[0-9]+(\.[0-9]+)?)"#
REG_OVERLAP23 = r'\b^([1][0-9]|[2-9][0-9]*|Default)$\b'#r'\b^([2-9][0-9]*?|Default)$\b'
REG_MDX_SEG = r'\b(?:' + '|'.join([str(num) for num in range(32, 1000001, 32)]) + r')\b'
REG_ALIGN = r'^[-+]?[0-9]\d*?$'
REG_VOL_COMP = r'^\d+\.\d{1,9}$'

# Sub Menu
VR_ARCH_SETTING_LOAD = 'Load for VR Arch'
MDX_SETTING_LOAD = 'Load for MDX-Net'
DEMUCS_SETTING_LOAD = 'Load for Demucs'
ALL_ARCH_SETTING_LOAD = 'Load for Full Application'

# Mappers

DEFAULT_DATA = {
        'chosen_process_method': MDX_ARCH_TYPE,
        'vr_model': CHOOSE_MODEL,
        'aggression_setting': 5,
        'window_size': 512,
        'mdx_segment_size': 256,
        'batch_size': DEF_OPT,
        'crop_size': 256, 
        'is_tta': False,
        'is_output_image': False,
        'is_post_process': False,
        'is_high_end_process': False,
        'post_process_threshold': 0.2,
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
        'segment': DEMUCS_SEGMENTS[0],
        'overlap': DEMUCS_OVERLAP[0],
        'overlap_mdx': MDX_OVERLAP[0],
        'overlap_mdx23': '8',
        'shifts': 2,
        'chunks_demucs': CHUNKS[0],
        'margin_demucs': 44100,
        'is_chunk_demucs': False,
        'is_chunk_mdxnet': False,
        'is_primary_stem_only_Demucs': False,
        'is_secondary_stem_only_Demucs': False,
        'is_split_mode': True,
        'is_demucs_combine_stems': True,#
        'is_mdx23_combine_stems': True,#
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
        'is_denoise': False,#
        'denoise_option': 'None',#
        'phase_option': AUTO_PHASE,
        'phase_shifts': NONE_P,#
        'is_save_align': False,#, 
        'is_match_frequency_pitch': True,#
        'is_match_silence': True,#
        'is_spec_match': False,#
        'is_mdx_c_seg_def': False,
        'is_invert_spec': False, #
        'is_deverb_vocals': False, #
        'deverb_vocal_opt': 'Main Vocals Only', #
        'voc_split_save_opt': 'Lead Only', #
        'is_mixer_mode': False, 
        'mdx_batch_size': DEF_OPT,
        'mdx_voc_inst_secondary_model': NO_MODEL,
        'mdx_other_secondary_model': NO_MODEL,
        'mdx_bass_secondary_model': NO_MODEL,
        'mdx_drums_secondary_model': NO_MODEL,
        'mdx_is_secondary_model_activate': False,        
        'mdx_voc_inst_secondary_model_scale': 0.9,
        'mdx_other_secondary_model_scale': 0.7,
        'mdx_bass_secondary_model_scale': 0.5,
        'mdx_drums_secondary_model_scale': 0.5,
        'mdx_stems': ALL_STEMS,
        'is_save_all_outputs_ensemble': True,
        'is_append_ensemble_name': False,
        'chosen_audio_tool': AUDIO_TOOL_OPTIONS[0],
        'choose_algorithm': MANUAL_ENSEMBLE_OPTIONS[0],
        'time_stretch_rate': 2.0,
        'pitch_rate': 2.0,
        'is_time_correction': True,
        'is_gpu_conversion': False,
        'is_primary_stem_only': False,
        'is_secondary_stem_only': False,
        'is_testing_audio': False,#
        'is_auto_update_model_params': True,#
        'is_add_model_name': False,
        'is_accept_any_input': False,
        'is_task_complete': False,
        'is_normalization': False,
        'is_use_opencl': False,
        'is_wav_ensemble': False,
        'is_create_model_folder': False,
        'mp3_bit_set': '320k',#
        'semitone_shift': '0',#
        'save_format': WAV,
        'wav_type_set': 'PCM_16',
        'device_set': DEFAULT,
        'user_code': '',
        'export_path': '',
        'input_paths': [],
        'lastDir': None,
        'time_window': "3",
        'intro_analysis': DEFAULT,
        'db_analysis': "Medium",
        'fileOneEntry': '',
        'fileOneEntry_Full': '',
        'fileTwoEntry': '',
        'fileTwoEntry_Full': '',
        'DualBatch_inputPaths': [],
        'model_hash_table': {},
        'help_hints_var': True,
        'set_vocal_splitter': NO_MODEL,
        'is_set_vocal_splitter': False,#
        'is_save_inst_set_vocal_splitter': False,#
        'model_sample_mode': False,
        'model_sample_mode_duration': 30
}

SETTING_CHECK = ('vr_model',
               'aggression_setting',
               'window_size',
               'mdx_segment_size',
               'batch_size',
               'crop_size',
               'is_tta',
               'is_output_image',
               'is_post_process',
               'is_high_end_process',
               'post_process_threshold',
               'vr_voc_inst_secondary_model',
               'vr_other_secondary_model',
               'vr_bass_secondary_model',
               'vr_drums_secondary_model',
               'vr_is_secondary_model_activate',
               'vr_voc_inst_secondary_model_scale',
               'vr_other_secondary_model_scale',
               'vr_bass_secondary_model_scale',
               'vr_drums_secondary_model_scale',
               'demucs_model',
               'segment',
               'overlap',
               'overlap_mdx',
               'shifts',
               'chunks_demucs',
               'margin_demucs',
               'is_chunk_demucs',
               'is_primary_stem_only_Demucs',
               'is_secondary_stem_only_Demucs',
               'is_split_mode',
               'is_demucs_combine_stems',#
               'is_mdx23_combine_stems',#
               'demucs_voc_inst_secondary_model',
               'demucs_other_secondary_model',
               'demucs_bass_secondary_model',
               'demucs_drums_secondary_model',
               'demucs_is_secondary_model_activate',
               'demucs_voc_inst_secondary_model_scale',
               'demucs_other_secondary_model_scale',
               'demucs_bass_secondary_model_scale',
               'demucs_drums_secondary_model_scale',
               'demucs_stems',
               'mdx_net_model',
               'chunks',
               'margin',
               'compensate',
               'is_denoise',#
               'denoise_option',#
               'phase_option',#
               'phase_shifts',#
               'is_save_align',#,
               'is_match_silence',
               'is_spec_match',#,
               'is_match_frequency_pitch',#
               'is_mdx_c_seg_def',
               'is_invert_spec',#
               'is_deverb_vocals',#
               'deverb_vocal_opt',#
               'voc_split_save_opt',#
               'mdx_batch_size',
               'mdx_voc_inst_secondary_model',
               'mdx_other_secondary_model',
               'mdx_bass_secondary_model',
               'mdx_drums_secondary_model',
               'mdx_is_secondary_model_activate',
               'mdx_voc_inst_secondary_model_scale',
               'mdx_other_secondary_model_scale',
               'mdx_bass_secondary_model_scale',
               'mdx_drums_secondary_model_scale',
               'is_save_all_outputs_ensemble',
               'is_append_ensemble_name',
               'chosen_audio_tool',
               'choose_algorithm',
               'time_stretch_rate',
               'pitch_rate',
               'is_time_correction',
               'is_primary_stem_only',
               'is_secondary_stem_only',
               'is_testing_audio',#
               'is_auto_update_model_params',#
               'is_add_model_name',
               "is_accept_any_input",
               'is_task_complete',
               'is_create_model_folder',
               'mp3_bit_set',#
               'semitone_shift',#
               'save_format',
               'wav_type_set',
               'device_set',
               'user_code',
               'is_gpu_conversion',
               'is_normalization',
               'is_use_opencl',
               'is_wav_ensemble',
               'help_hints_var',
               'set_vocal_splitter',
               'is_set_vocal_splitter',#
               'is_save_inst_set_vocal_splitter',#
               'model_sample_mode',
               'model_sample_mode_duration',
               'time_window',
               'intro_analysis',
               'db_analysis',
               'fileOneEntry',
               'fileOneEntry_Full',
               'fileTwoEntry',
               'fileTwoEntry_Full',
               'DualBatch_inputPaths'
               )

NEW_LINES = "\n\n"
NEW_LINE = "\n"
NO_LINE = ''

FFMPEG_EXT = (".aac", ".aiff", ".alac" ,".flac", ".FLAC", ".mov", ".mp4", ".MP4", 
              ".m4a", ".M4A", ".mp2", ".mp3", "MP3", ".mpc", ".mpc8", 
              ".mpeg", ".ogg", ".OGG", ".tta", ".wav", ".wave", ".WAV", ".WAVE", ".wma", ".webm", ".eac3", ".mkv", ".opus", ".OPUS")

FFMPEG_MORE_EXT = (".aa", ".aac", ".ac3", ".aiff", ".alac", ".avi", ".f4v",".flac", ".flic", ".flv",
              ".m4v",".mlv", ".mov", ".mp4", ".m4a", ".mp2", ".mp3", ".mp4", ".mpc", ".mpc8", 
              ".mpeg", ".ogg", ".tta", ".tty", ".vcd", ".wav", ".wma")
ANY_EXT = ""

# Secondary Menu Constants

VOCAL_PAIR_PLACEMENT = 1, 2, 3, 4
OTHER_PAIR_PLACEMENT = 5, 6, 7, 8
BASS_PAIR_PLACEMENT = 9, 10, 11, 12
DRUMS_PAIR_PLACEMENT = 13, 14, 15, 16

# Drag n Drop String Checks

DOUBLE_BRACKET = "} {"
RIGHT_BRACKET = "}"
LEFT_BRACKET = "{"
#DND CONSTS

MAC_DND_CHECK = ('/Users/',
                 '/Applications/',
                 '/Library/',
                 '/System/')
LINUX_DND_CHECK = ('/home/',
                   '/usr/')
WINDOWS_DND_CHECK = ('A:', 'B:', 'C:', 'D:', 'E:', 'F:', 'G:', 'H:', 'I:', 'J:', 'K:', 'L:', 'M:', 'N:', 'O:', 'P:', 'Q:', 'R:', 'S:', 'T:', 'U:', 'V:', 'W:', 'X:', 'Y:', 'Z:')

WOOD_INST_MODEL_HASH = '0ec76fd9e65f81d8b4fbd13af4826ed8'
WOOD_INST_PARAMS = {
    "vr_model_param": "4band_v3",
    "primary_stem": NO_WIND_INST_STEM
                     }

READ_ONLY = 'readonly'

FILE_1 = 'file1'
FILE_2 = 'file2'

FILE_1_LB = 'file1_lb'
FILE_2_LB = 'file1_2b'
BATCH_MODE_DUAL = " : Batch Mode"

CODEC_DICT = {
    'PCM_U8':   {"sample_width": 1, "codec": None},        # 8-bit unsigned PCM
    'PCM_16':   {"sample_width": 2, "codec": None},        # 16-bit signed PCM
    'PCM_24':   {"sample_width": 3, "codec": None},        # 24-bit signed PCM
    'PCM_32':   {"sample_width": 4, "codec": None},        # 32-bit signed PCM
    'FLOAT32':  {"sample_width": None, "codec": "pcm_f32le"},  # 32-bit float
    'FLOAT64':  {"sample_width": None, "codec": "pcm_f64le"}   # 64-bit float
}


# Manual Downloads
VR_PLACEMENT_TEXT = 'Place models in \"models/VR_Models\" directory.'
MDX_PLACEMENT_TEXT = 'Place models in \"models/MDX_Net_Models\" directory.'
DEMUCS_PLACEMENT_TEXT = 'Place models in \"models/Demucs_Models\" directory.'
DEMUCS_V3_V4_PLACEMENT_TEXT = 'Place items in \"models/Demucs_Models/v3_v4_repo\" directory.'
MDX_23_NAME = "MDX23C Model"

# Liscense info
if OPERATING_SYSTEM=="Darwin":
   is_macos = True
   LICENSE_OS_SPECIFIC_TEXT = '• This application is intended for those running macOS Catalina and above.\n' +\
                              '• Application functionality for systems running macOS Mojave or lower is not guaranteed.\n' +\
                              '• Application functionality for older or budget Mac systems is not guaranteed.\n\n'
elif OPERATING_SYSTEM=="Linux":
   LICENSE_OS_SPECIFIC_TEXT = '• This application is intended for those running Linux Ubuntu 18.04+.\n' +\
                              '• Application functionality for systems running other Linux platforms is not guaranteed.\n' +\
                              '• Application functionality for older or budget systems is not guaranteed.\n\n'
elif OPERATING_SYSTEM=="Windows":
   LICENSE_OS_SPECIFIC_TEXT = '• This application is intended for those running Windows 10 or higher.\n' +\
                              '• Application functionality for systems running Windows 7 or lower is not guaranteed.\n' +\
                              '• Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.\n\n'

LICENSE_TEXT = lambda a, p:f'Current Application Version: Ultimate Vocal Remover {a}\n' +\
               f'Current Patch Version: {p}\n\n' +\
               'Copyright (c) 2022 Ultimate Vocal Remover\n\n' +\
               'UVR is free and open-source, but MIT licensed. Please credit us if you use our\n' +\
               f'models or code for projects unrelated to UVR.\n\n{LICENSE_OS_SPECIFIC_TEXT}' +\
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

# Message Box Text
INVALID_INPUT = 'Invalid Input', 'The input is invalid.\n\nPlease verify the input still exists or is valid and try again.'
INVALID_EXPORT = 'Invalid Export Directory', 'You have selected an invalid export directory.\n\nPlease make sure the selected directory still exists.'
INVALID_ENSEMBLE = 'Not Enough Models', 'You must select 2 or more models to run ensemble.'
INVALID_MODEL = 'No Model Chosen', 'You must select an model to continue.'
MISSING_MODEL = 'Model Missing', 'The selected model is missing or not valid.'
ERROR_OCCURED = 'Error Occured', '\n\nWould you like to open the error log for more details?\n'
PROCESS_COMPLETE = '\nProcess complete\n'
PROCESS_COMPLETE_2 = 'Process complete\n'

# GUI Text Constants
BACK_TO_MAIN_MENU = 'Back to Main Menu'

# Help Hint Text
INTERNAL_MODEL_ATT = 'This is an internal model setting. \n\n***Avoid changing it unless you\'re certain about it!***'
STOP_HELP = 'Stops ongoing tasks.\n• A confirmation pop-up will appear before stopping.'
SETTINGS_HELP = 'Accesses the main settings and the "Download Center."'
COMMAND_TEXT_HELP = 'Shows the status and progress of ongoing tasks.'
SAVE_CURRENT_SETTINGS_HELP = 'Load or save the app\'s settings.'
PITCH_SHIFT_HELP = ('Choose the pitch for processing tracks:\n\n'
                '• Whole numbers indicate semitones.\n'
                '• Using higher pitches may cut the upper bandwidth, even in high-quality models.\n'
                '• Upping the pitch can be better for tracks with deeper vocals.\n'
                '• Dropping the pitch may take more processing time but works well for tracks with high-pitched vocals.')
AGGRESSION_SETTING_HELP = ('Adjust the intensity of primary stem extraction:\n\n'
                           '• It ranges from -100 - 100.\n'
                           '• Bigger values mean deeper extractions.\n' 
                           '• Typically, it\'s set to 5 for vocals & instrumentals. \n' 
                           '• Values beyond 5 might muddy the sound for non-vocal models.')
WINDOW_SIZE_HELP = ('Select window size to balance quality and speed:\n\n'
                    '• 1024 - Quick but lesser quality.\n'
                    '• 512 - Medium speed and quality.\n'
                    '• 320 - Takes longer but may offer better quality.')
MDX_SEGMENT_SIZE_HELP = ('Pick a segment size to balance speed, resource use, and quality:\n'
                         '• Smaller sizes consume less resources.\n'
                         '• Bigger sizes consume more resources, but may provide better results.\n'
                         '• Default size is 256. Quality can change based on your pick.')
DEMUCS_STEMS_HELP = ('Select a stem for extraction with the chosen model:\n\n'
                     '• All Stems - Extracts all available stems.\n'
                     '• Vocals - Only the "vocals" stem.\n'
                     '• Other - Only the "other" stem.\n'
                     '• Bass - Only the "bass" stem.\n'
                     '• Drums - Only the "drums" stem.')
SEGMENT_HELP = ('Adjust segments to manage RAM or V-RAM usage:\n\n'
               '• Smaller sizes consume less resources.\n'
               '• Bigger sizes consume more resources, but may provide better results.\n'
               '• "Default" picks the optimal size.')

ENSEMBLE_MAIN_STEM_HELP = (
    'Select the stem type for ensembling:\n\n'
    
    f'• {VOCAL_PAIR}:\n'
    '  - Primary Stem: Vocals\n'
    '  - Secondary Stem: Instrumental (mixture minus vocals)\n\n'
    
    f'• {OTHER_PAIR}:\n'
    '  - Primary Stem: Other\n'
    '  - Secondary Stem: No Other (mixture minus "other")\n\n'
    
    f'• {BASS_PAIR}:\n'
    '  - Primary Stem: Bass\n'
    '  - Secondary Stem: No Bass (mixture minus bass)\n\n'
    
    f'• {DRUM_PAIR}:\n'
    '  - Primary Stem: Drums\n'
    '  - Secondary Stem: No Drums (mixture minus drums)\n\n'
    
    f'• {FOUR_STEM_ENSEMBLE}:\n'
    '  - Gathers all 4-stem Demucs models and ensembles all outputs.\n\n'
    
    f'• {MULTI_STEM_ENSEMBLE}:\n'
    '  - The "Jungle Ensemble" gathers all models and ensembles any related outputs.'
)

ENSEMBLE_TYPE_HELP = (
    'Choose the ensemble algorithm for generating the final output:\n\n'
    
    f'• {MAX_MIN}:\n'
    '  - Primary stem processed with "Max Spec" algorithm.\n'
    '  - Secondary stem processed with "Min Spec" algorithm.\n\n'
    
    'Note: For the "4 Stem Ensemble" option, only one algorithm will be displayed.\n\n'
    
    'Algorithm Details:\n'
    
    f'• {MAX_SPEC}:\n'
    '  - Produces the highest possible output.\n'
    '  - Ideal for vocal stems for a fuller sound, but might introduce unwanted artifacts.\n'
    '  - Works well with instrumental stems, but avoid using VR Arch models in the ensemble.\n\n'
    
    f'• {MIN_SPEC}:\n'
    '  - Produces the lowest possible output.\n'
    '  - Ideal for instrumental stems for a cleaner result. Might result in a "muddy" sound.\n\n'
    
    f'• {AUDIO_AVERAGE}:\n'
    '  - Averages all results together for the final output.'
)

ENSEMBLE_LISTBOX_HELP = (
    'Displays all available models for the chosen main stem pair.'
)

if OPERATING_SYSTEM == 'darwin':
   IS_GPU_CONVERSION_HELP = (
      '• Use GPU for Processing (if available):\n'
      '  - If checked, the application will attempt to use your GPU for faster processing.\n'
      '  - If a GPU is not detected, it will default to CPU processing.\n'
      '  - GPU processing for MacOS only works with VR Arch models.\n\n'
      '• Please Note:\n'
      '  - CPU processing is significantly slower than GPU processing.\n'
      '  - Only Macs with M1 chips can be used for GPU processing.'
   )
else:
   IS_GPU_CONVERSION_HELP = (
      '• Use GPU for Processing (if available):\n'
      '  - If checked, the application will attempt to use your GPU for faster processing.\n'
      '  - If a GPU is not detected, it will default to CPU processing.\n\n'
      '• Please Note:\n'
      '  - CPU processing is significantly slower than GPU processing.\n'
      '  - Only Nvidia GPUs can be used for GPU processing.'
   )

IS_TIME_CORRECTION_HELP = ('When checked, the output will retain the original BPM of the input.')
SAVE_STEM_ONLY_HELP = 'Allows the user to save only the selected stem.'
IS_NORMALIZATION_HELP = 'Normalizes output to prevent clipping.'
IS_CUDA_SELECT_HELP = "If you have more than one GPU, you can pick which one to use for processing."
CROP_SIZE_HELP = '**Only compatible with select models only!**\n\n Setting should match training crop-size value. Leave as is if unsure.'
IS_TTA_HELP = ('This option performs Test-Time-Augmentation to improve the separation quality.\n\n'
               'Note: Having this selected will increase the time it takes to complete a conversion')
IS_POST_PROCESS_HELP = ('This option can potentially identify leftover instrumental artifacts within the vocal outputs. \nThis option may improve the separation of some songs.\n\n' +\
                       'Note: Selecting this option can adversely affect the conversion process, depending on the track. Because of this, it is only recommended as a last resort.')
IS_HIGH_END_PROCESS_HELP = 'The application will mirror the missing frequency range of the output.'
SHIFTS_HELP = ('Performs multiple predictions with random shifts of the input and averages them.\n\n'
              '• The higher number of shifts, the longer the prediction will take. \n- Not recommended unless you have a GPU.')
OVERLAP_HELP = ('• This option controls the amount of overlap between prediction windows.\n'
               '       - Higher values can provide better results, but will lead to longer processing times.\n'
               '       - You can choose between 0.001-0.999')
MDX_OVERLAP_HELP = ('• This option controls the amount of overlap between prediction windows.\n'
               '       - Higher values can provide better results, but will lead to longer processing times.\n'
               '       - For Non-MDX23C models: You can choose between 0.001-0.999')
OVERLAP_23_HELP = ('• This option controls the amount of overlap between prediction windows.\n'
                  '       - Higher values can provide better results, but will lead to longer processing times.')
IS_SEGMENT_DEFAULT_HELP = '• The segment size is set based on the value provided in a chosen model\'s associated \nconfig file (yaml).'
IS_SPLIT_MODE_HELP = '• Enables \"Segments\". \n• Deselecting this option is only recommended for those with powerful PCs.'
IS_DEMUCS_COMBINE_STEMS_HELP = 'The application will create the secondary stem by combining the remaining stems \ninstead of inverting the primary stem with the mixture.'
COMPENSATE_HELP = 'Compensates the audio of the primary stems to allow for a better secondary stem.'
IS_DENOISE_HELP = ('• Standard: This setting reduces the noise created by MDX-Net models.\n' 
                   '       - This option only reduces noise in non-MDX23 models.\n' 
                   '• Denoise Model: This setting employs a special denoise model to eliminate noise produced by any MDX-Net model.\n'
                   '       - This option works on all MDX-Net models.\n'
                   '       - You must have the "UVR-DeNoise-Lite" VR Arch model installed to use this option.\n'
                   '• Please Note: Both options will increase separation time.')

VOC_SPLIT_MODEL_SELECT_HELP = '• Select a model from the list of lead and backing vocal models to run through vocal stems automatically.'
IS_VOC_SPLIT_INST_SAVE_SELECT_HELP = '• When activated, you will receive extra instrumental outputs that include: one with just the lead vocals and another with only the backing vocals.'
IS_VOC_SPLIT_MODEL_SELECT_HELP = ('• When activated, this option auto-processes generated vocal stems, using either a karaoke model to remove lead vocals or another to remove backing vocals.\n'
                                 '       - This option splits the vocal track into two separate parts: lead vocals and backing vocals, providing two extra vocal outputs.\n'
                                 '       - The results will be organized in the same way, whether you use a karaoke model or a background vocal model.\n'
                                 '       - This option does not work in ensemble mode at this time.')
IS_DEVERB_OPT_HELP = ('• Select the vocal type you wish to deverb automatically.\n'
                     '       - Example: Choosing "Lead Vocals Only" will only remove reverb from a lead vocal stem.')
IS_DEVERB_VOC_HELP = ('• This option removes reverb from a vocal stem.\n'
                     '       - You must have the "UVR-DeEcho-DeReverb" VR Arch model installed to use this option.\n'
                     '       - This option does not work in ensemble mode at this time.')
IS_FREQUENCY_MATCH_HELP = 'Matches the frequency cut-off of the primary stem to that of the secondary stem.'
CLEAR_CACHE_HELP = 'Clears settings for unrecognized models chosen by the user.'
IS_SAVE_ALL_OUTPUTS_ENSEMBLE_HELP = 'If enabled, all individual ensemble-generated outputs are retained.'
IS_APPEND_ENSEMBLE_NAME_HELP = 'When enabled, the ensemble name is added to the final output.'
IS_WAV_ENSEMBLE_HELP = (
    'Processes ensemble algorithms with waveforms instead of spectrograms when activated:\n'
    '• Might lead to increased distortion.\n'
    '• Waveform ensembling is faster than spectrogram ensembling.'
)
DONATE_HELP = 'Opens official UVR "Buy Me a Coffee" external link for project donations!'
IS_INVERT_SPEC_HELP = (
    'Potentially enhances the secondary stem quality:\n'
    '• Inverts primary stem using spectrograms, instead of waveforms.\n'
    '• Slightly slower inversion method.'
)
IS_TESTING_AUDIO_HELP = 'Appends a 10-digit number to saved files to avoid accidental overwrites.'
IS_MODEL_TESTING_AUDIO_HELP = 'Appends the model name to outputs for comparison across different models.'
IS_ACCEPT_ANY_INPUT_HELP = (
    'Allows all types of inputs when enabled, even non-audio formats.\n'
    'For experimental use only. Not recommended for regular use.'
)
IS_TASK_COMPLETE_HELP = 'Plays a chime upon process completion or failure when activated.'
DELETE_YOUR_SETTINGS_HELP = (
    'Contains your saved settings. Confirmation will be requested before deleting a selected setting.'
)
SET_STEM_NAME_HELP = 'Select the primary stem for the given model.'
IS_CREATE_MODEL_FOLDER_HELP = ('Two new directories will be generated for the outputs in the export directory after each conversion.\n\n'
                              '• Example: \n'
                              '─ Export Directory\n'
                              '   └── First Directory (Named after the model)\n'
                              '           └── Second Directory (Named after the track)\n'
                              '                    └── Output File(s)')
MDX_DIM_T_SET_HELP = INTERNAL_MODEL_ATT
MDX_DIM_F_SET_HELP = INTERNAL_MODEL_ATT

MDX_N_FFT_SCALE_SET_HELP = 'Specify the N_FFT size used during model training.'
POPUP_COMPENSATE_HELP = (
    f'Select the appropriate volume compensation for the chosen model.\n'
    f'Reminder: {COMPENSATE_HELP}'
)
VR_MODEL_PARAM_HELP = 'Select the required parameters to run the chosen model.'
CHOSEN_ENSEMBLE_HELP = (
    'Default Ensemble Selections:\n'
    '• Save the current ensemble configuration.\n'
    '• Clear all selected models.\n'
    'Note: You can also select previously saved ensembles.'
)
CHOSEN_PROCESS_METHOD_HELP = (
    'Choose a Processing Method:\n'
    'Select from various AI networks and algorithms to process your track:\n'
    '\n'
    '• VR Architecture: Uses magnitude spectrograms for source separation.\n'
    '• MDX-Net: Employs a Hybrid Spectrogram network for source separation.\n'
    '• Demucs v3: Also utilizes a Hybrid Spectrogram network for source separation.\n'
    '• Ensemble Mode: Combine results from multiple models and networks for optimal results.\n'
    '• Audio Tools: Additional utilities for added convenience.'
)        

INPUT_FOLDER_ENTRY_HELP = (
    'Select Input:\n'
    'Choose the audio file(s) you want to process.'
)
INPUT_FOLDER_ENTRY_HELP_2 = (
    'Input Option Menu:\n'
    'Click to access the input option menu.'
)
OUTPUT_FOLDER_ENTRY_HELP = (
    'Select Output:\n'
    'Choose the directory where the processed files will be saved.'
)
INPUT_FOLDER_BUTTON_HELP = (
    'Open Input Folder Button:\n'
    'Open the directory containing the selected input audio file(s).'
)
OUTPUT_FOLDER_BUTTON_HELP = (
    'Open Output Folder Button:\n'
    'Open the selected output folder.'
)
CHOOSE_MODEL_HELP = (
    'Each processing method has its own set of options and models.\n'
    'Choose the model associated with the selected processing method here.'
)
FORMAT_SETTING_HELP = 'Save Outputs As: '
SECONDARY_MODEL_ACTIVATE_HELP = (
    'When enabled, the application will perform an additional inference using the selected model(s) above.'
)
SECONDARY_MODEL_HELP = (
    'Choose the Secondary Model:\n'
    'Select the secondary model associated with the stem you want to process with the current method.'
)

INPUT_SEC_FIELDS_HELP = (
    'Right click here to choose your inputs!'
)

SECONDARY_MODEL_SCALE_HELP = ('The scale determines how the final audio outputs will be averaged between the primary and secondary models.\n\nFor example:\n\n'
                             '• 10% - 10 percent of the main model result will be factored into the final result.\n'
                             '• 50% - The results from the main and secondary models will be averaged evenly.\n'
                             '• 90% - 90 percent of the main model result will be factored into the final result.')
PRE_PROC_MODEL_ACTIVATE_HELP = (
    'When enabled, the application will use the selected model to isolate the instrumental stem.\n'
    'Subsequently, all non-vocal stems will be extracted from this generated instrumental.\n'
    '\n'
    'Key Points:\n'
    '• This feature can significantly reduce vocal bleed in non-vocal stems.\n'
    '• Available exclusively in the Demucs tool.\n'
    '• Compatible only with non-vocal and non-instrumental stem outputs.\n'
    '• Expect an increase in total processing time.\n'
    '• Only the VR or MDX-Net Vocal Instrumental/Vocals models can be chosen for this process.'
)
      
AUDIO_TOOLS_HELP = (
    'Select from various audio tools to process your track:\n'
    '\n'
    '• Manual Ensemble: Requires 2 or more selected files as inputs. This allows tracks to be processed using the algorithms from Ensemble Mode.\n'
    '• Time Stretch: Adjust the playback speed of the selected inputs to be faster or slower.\n'
    '• Change Pitch: Modify the pitch of the selected inputs.\n'
    '• Align Inputs: Choose 2 audio file and the application will align them and provide the difference in alignment.\n'
    '    - This tool provides similar functionality to "Utagoe."\n'
    '    - Primary Audio: This is usually a mixture.\n'
    '    - Secondary Audio: This is usually an instrumental.\n'
    '• Matchering: Choose 2 audio files. The matchering algorithm will master the target audio to have the same RMS, FR, peak amplitude, and stereo width as the reference audio.'
)
             
PRE_PROC_MODEL_INST_MIX_HELP = 'When enabled, the application will generate a third output without the selected stem and vocals.'         
MODEL_SAMPLE_MODE_HELP = ('Allows the user to process only part of a track to sample settings or a model without running a full conversion.\n\nNotes:\n\n'
                         '• The number in the parentheses is the current number of seconds the generated sample will be.\n'
                         '• You can choose the number of seconds to extract from the track in the \"Additional Settings\" menu.')
                    
POST_PROCESS_THREASHOLD_HELP = ('Allows the user to control the intensity of the Post_process option.\n\nNotes:\n\n'
                               '• Higher values potentially remove more artifacts. However, bleed might increase.\n'
                               '• Lower values limit artifact removal.')

BATCH_SIZE_HELP = ('Specify the number of batches to be processed at a time.\n\nNotes:\n\n'
                               '• Higher values mean more RAM usage but slightly faster processing times.\n'
                               '• Lower values mean less RAM usage but slightly longer processing times.\n'
                               '• Batch size value has no effect on output quality.')
         
VR_MODEL_NOUT_HELP = ""
VR_MODEL_NOUT_LSTM_HELP = ""
  
IS_PHASE_HELP = 'Select the phase for the secondary audio.\n• Note: Using the "Automatic" option is strongly recommended.'
IS_ALIGN_TRACK_HELP = 'Enable this to save the secondary track once aligned.'
IS_MATCH_SILENCE_HELP = (
    'Aligns the initial silence of the secondary audio with the primary audio.\n'
    '• Note: Avoid using this option if the primary audio begins solely with vocals.'
)
IS_MATCH_SPEC_HELP = 'Align the secondary audio based on the primary audio\'s spectrogram.\n• Note: This may enhance alignment in specific cases.'

TIME_WINDOW_ALIGN_HELP = (
                           'This setting determines the window size for alignment analysis, especially for pairs with minor timing variations:\n'
                           '\n'
                           '• None: Disables time window analysis.\n'
                           '• 1: Analyzes pair by 0.0625-second windows.\n'
                           '• 2: Analyzes pair by 0.125-second windows.\n'
                           '• 3: Analyzes pair by 0.25-second windows.\n'
                           '• 4: Analyzes pair by 0.50-second windows.\n'
                           '• 5: Analyzes pair by 0.75-second windows.\n'
                           '• 6: Analyzes pair by 1-second windows.\n'
                           '• 7: Analyzes pair by 2-second windows.\n'
                           '\n'
                           'Shifts Options:\n'
                           '• Low: Cycles through 0.0625 and 0.5-second windows to find an optimal match.\n'
                           '• Medium: Cycles through 0.0625, 0.125, and 0.5-second windows to find an optimal match.\n'
                           '• High: Cycles through 0.0625, 0.125, 0.25, and 0.5-second windows to find an optimal match.\n'
                           '\n'
                           'Important Points to Consider:\n'
                           '    - Using the "Shifts" option may require more processing time and might not guarantee better results.\n'
                           '    - Opting for smaller analysis windows can increase processing times.\n'
                           '    - The best settings are likely to vary based on the specific tracks being processed.'
)
INTRO_ANALYSIS_ALIGN_HELP = (
                           'This setting determines the portion of the audio input to be analyzed for initial alignment.\n'
                           '\n'
                           '• Default: Analyzes 10% (or 1/10th) of the audio\'s total length.\n'
                           '• 1: Analyzes 12.5% (or 1/8th) of the audio\'s total length.\n'
                           '• 2: Analyzes 16.67% (or 1/6th) of the audio\'s total length.\n'
                           '• 3: Analyzes 25% (or 1/4th) of the audio\'s total length.\n'
                           '• 4: Analyzes 50% (or half) of the audio\'s total length.\n'
                           '\n'
                           'Shifts Options:\n'
                           '• Low: Cycles through 2 intro analysis values.\n'
                           '• Medium: Cycles through 3 intro analysis values.\n'
                           '• High: Cycles through 5 intro analysis values.\n'
                           '\n'
                           'Important Points to Consider:\n'
                           '    - Using the "Shifts" option will require more processing time and might not guarantee better results.\n'
                           '    - Optimal settings may vary depending on the specific tracks being processed.'
)

VOLUME_ANALYSIS_ALIGN_HELP = (
                           'This setting specifies the volume adjustments to be made on the secondary input:\n'
                           '\n'
                           '• None: No volume adjustments are made.\n'
                           '• Low: Analyzes the audio within a 4dB range, adjusting in 1dB increments.\n'
                           '• Medium: Analyzes the audio within a 6dB range, adjusting in 1dB increments.\n'
                           '• High: Analyzes the audio within a 6dB range, adjusting in 0.5dB increments.\n'
                           '• Very High: Analyzes the audio within a 10dB range, adjusting in 0.5dB increments.\n'
                           '\n'
                           'Important Points to Consider:\n'
                           '    - Selecting more extensive analysis options (e.g., High, Very High) will lead to longer processing times.\n'
                           '    - Optimal settings might vary based on the specific tracks being processed.'
)

PHASE_SHIFTS_ALIGN_HELP = (
                           'This setting specifies the phase adjustments to be made on the secondary input:\n'
                           '\n'
                           'Shifts Options:\n'
                           '• None: No phase adjustments are made.\n'
                           '• Very Low: Analyzes the audio within range of 2 different phase positions.\n'
                           '• Low: Analyzes the audio within range of 4 different phase positions.\n'
                           '• Medium: Analyzes the audio within range of 8 different phase positions.\n'
                           '• High: Analyzes the audio within range of 18 different phase positions.\n'
                           '• Very High: Analyzes the audio within range of 36 different phase positions.\n'
                           '• Maximum: Analyzes the audio in all 360 phase positions.\n'
                           '\n'
                           'Important Points to Consider:\n'
                           '    - This option only works with time correction.\n'
                           '    - This option can be helpful if one of the inputs were from an analog source.\n'
                           '    - Selecting more extensive analysis options (e.g., High, Very High) will lead to longer processing times.\n'
                           '    - Selecting "Maximum" can take hours to process.\n'
                           '    - Optimal settings might vary based on the specific tracks being processed.'
)

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
INVALID_ONNX_MODEL_ERROR = 'Invalid Model', 'The file selected is not a valid MDX-Net model. Please see the error log for more information.'
INVALID_PARAM_MODEL_ERROR = 'Select Model Param', 'Please choose a model param or click \'Cancel\'.'
UNRECOGNIZED_MODEL = 'Unrecognized Model Detected', ' is an unrecognized model.\n\n' + \
                     'Would you like to select the correct parameters before continuing?'
STOP_PROCESS_CONFIRM = 'Confirmation', 'You are about to stop all active processes.\n\nAre you sure you wish to continue?'
NO_ENSEMBLE_SELECTED = 'No Models Selected', 'Please select ensemble and try again.'
PICKLE_CORRU = 'File Corrupted', 'Unable to load this ensemble.\n\n' + \
               'Would you like to remove this ensemble from your list?'
DELETE_ENS_ENTRY = 'Confirm Removal', 'Are you sure you want to remove this entry?'

# Separation Text
LOADING_MODEL = 'Loading model...'
INFERENCE_STEP_1 = 'Running inference...'
INFERENCE_STEP_1_SEC = 'Running inference (secondary model)...'
INFERENCE_STEP_1_4_STEM = lambda stem:f'Running inference (secondary model for {stem})...'
INFERENCE_STEP_1_PRE = 'Running inference (pre-process model)...'
INFERENCE_STEP_1_VOC_S = 'Splitting vocals...'
INFERENCE_STEP_2_PRE = lambda pm, m:f'Loading pre-process model ({pm}: {m})...'
INFERENCE_STEP_2_SEC = lambda pm, m:f'Loading secondary model ({pm}: {m})...'
INFERENCE_STEP_2_VOC_S = lambda pm, m:f'Loading vocal splitter model ({pm}: {m})...'
INFERENCE_STEP_2_SEC_CACHED_MODOEL = lambda pm, m:f'Secondary model ({pm}: {m}) cache loaded.\n'
INFERENCE_STEP_2_PRE_CACHED_MODOEL = lambda pm, m:f'Pre-process model ({pm}: {m}) cache loaded.\n'
INFERENCE_STEP_2_SEC_CACHED = 'Loading cached secondary model source(s)... Done!\n'
INFERENCE_STEP_2_PRIMARY_CACHED = ' Model cache loaded.\n'
INFERENCE_STEP_2 = 'Inference complete.'
INFERENCE_STEP_DEVERBING = ' Deverbing...'
SAVING_STEM = 'Saving ', ' stem...'
SAVING_ALL_STEMS = 'Saving all stems...'
ENSEMBLING_OUTPUTS = 'Ensembling outputs...'
DONE = ' Done!\n'
ENSEMBLES_SAVED = 'Ensembled outputs saved!\n\n'

#Additional Text
CHOOSE_PROC_METHOD_MAIN_LABEL = 'CHOOSE PROCESS METHOD'
SELECT_SAVED_SETTINGS_MAIN_LABEL = 'SELECT SAVED SETTINGS'
CHOOSE_MDX_MODEL_MAIN_LABEL = 'CHOOSE MDX-NET MODEL'
BATCHES_MDX_MAIN_LABEL = 'BATCH SIZE'
VOL_COMP_MDX_MAIN_LABEL = 'VOLUME COMPENSATION'
SEGMENT_MDX_MAIN_LABEL = 'SEGMENT SIZE'
SELECT_VR_MODEL_MAIN_LABEL = 'CHOOSE VR MODEL'
AGGRESSION_SETTING_MAIN_LABEL = 'AGGRESSION SETTING'
WINDOW_SIZE_MAIN_LABEL = 'WINDOW SIZE'
CHOOSE_DEMUCS_MODEL_MAIN_LABEL = 'CHOOSE DEMUCS MODEL'
CHOOSE_STEMS_MAIN_LABEL = 'CHOOSE STEM(S)'
CHOOSE_SEGMENT_MAIN_LABEL = 'SEGMENT'
ENSEMBLE_OPTIONS_MAIN_LABEL = 'ENSEMBLE OPTIONS'
CHOOSE_MAIN_PAIR_MAIN_LABEL = 'MAIN STEM PAIR'
CHOOSE_ENSEMBLE_ALGORITHM_MAIN_LABEL = 'ENSEMBLE ALGORITHM'
AVAILABLE_MODELS_MAIN_LABEL = 'AVAILABLE MODELS'
CHOOSE_AUDIO_TOOLS_MAIN_LABEL = 'CHOOSE AUDIO TOOL'
CHOOSE_MANUAL_ALGORITHM_MAIN_LABEL = 'CHOOSE ALGORITHM'
CHOOSE_RATE_MAIN_LABEL = 'RATE'
CHOOSE_SEMITONES_MAIN_LABEL = 'SEMITONES'
GPU_CONVERSION_MAIN_LABEL = 'GPU Conversion'
CHANGE_LOG_HEADER = lambda patch:f"Patch Version:\n\n{patch}"
INVALID_INPUT_E = ' Invalid input! '
LB_UP = "Move Selection Up"
LB_DOWN = "Move Selection Down"
LB_CLEAR = "Clear Box"
LB_MOVE_OVER_P = "Move Selection to Secondary List"
LB_MOVE_OVER_S = "Move Selection to Primary List"
FILE_ONE_MAIN_LABEL = "PRIMARY AUDIO"
FILE_TWO_MAIN_LABEL = "SECONDARY AUDIO"
FILE_ONE_MATCH_MAIN_LABEL = "TARGET AUDIO"
FILE_TWO_MATCH_MAIN_LABEL = "REFERENCE AUDIO"
TIME_WINDOW_MAIN_LABEL = "TIME ADJUSTMENT"
INTRO_ANALYSIS_MAIN_LABEL = "INTRO ANALYSIS"
VOLUME_ADJUSTMENT_MAIN_LABEL = "VOLUME ADJUSTMENT"
SELECT_INPUTS = "Select Input(s)"
SELECTED_INPUTS = 'Selected Inputs'
WIDEN_BOX = 'Widen Box'
CONFIRM_ENTRIES = 'Confirm Entries'
CLOSE_WINDOW = 'Close Window'
DUAL_AUDIO_PROCESSING = 'Dual Audio Batch Processing'
CANCEL_TEXT = "Cancel"
CONFIRM_TEXT = "Confirm"
SELECT_MODEL_TEXT = 'Select Model'
NONE_SELECTED = 'None Selected'
SAVE_TEXT = 'Save'
OVERLAP_TEXT = 'Overlap'
ACCEPT_ANY_INPUT_TEXT = 'Accept Any Input'
ACTIVATE_PRE_PROCESS_MODEL_TEXT = 'Activate Pre-process Model'
ACTIVATE_SECONDARY_MODEL_TEXT = 'Activate Secondary Model'
ADDITIONAL_MENUS_INFORMATION_TEXT = 'Additional Menus & Information'
ADDITIONAL_SETTINGS_TEXT = 'Additional Settings'
ADVANCED_ALIGN_TOOL_OPTIONS_TEXT = 'Advanced Align Tool Options'
ADVANCED_DEMUCS_OPTIONS_TEXT = 'Advanced Demucs Options'
ADVANCED_ENSEMBLE_OPTIONS_TEXT = 'Advanced Ensemble Options'
ADVANCED_MDXNET23_OPTIONS_TEXT = 'Advanced MDX-NET23 Options'
ADVANCED_MDXNET_OPTIONS_TEXT = 'Advanced MDX-Net Options'
ADVANCED_OPTION_MENU_TEXT = 'Advanced Option Menu'
ADVANCED_VR_OPTIONS_TEXT = 'Advanced VR Options'
AGGRESSION_SETTING_TEXT = 'Aggression Setting'
APPEND_ENSEMBLE_NAME_TEXT = 'Append Ensemble Name'
APPLICATION_DOWNLOAD_CENTER_TEXT = 'Application Download Center'
APPLICATION_UPDATES_TEXT = 'Application Updates'
AUDIO_FORMAT_SETTINGS_TEXT = 'Audio Format Settings'
BALANCE_VALUE_TEXT = 'Balance Value'
BATCH_SIZE_TEXT = 'Batch Size'
BV_MODEL_TEXT = 'BV Model'
CHANGE_MODEL_DEFAULT_TEXT = 'Change Model Default'
CHANGE_MODEL_DEFAULTS_TEXT = 'Change Model Defaults'
CHANGE_PARAMETERS_TEXT = 'Change Parameters'
CHOOSE_ADVANCED_MENU_TEXT = 'Choose Advanced Menu' 
CHOOSE_MODEL_PARAM_TEXT = 'Choose Model Param'
CLEAR_AUTOSET_CACHE_TEXT = 'Clear Auto-Set Cache'
COMBINE_STEMS_TEXT = 'Combine Stems'
CONFIRM_UPDATE_TEXT = 'Confirm Update'
COPIED_TEXT = 'Copied!'
COPY_ALL_TEXT_TEXT = 'Copy All Text'
DEFINED_PARAMETERS_DELETED_TEXT = 'Defined Parameters Deleted'
DELETE_PARAMETERS_TEXT = 'Delete Parameters'
DELETE_USER_SAVED_SETTING_TEXT = 'Delete User Saved Setting'
DEMUCS_TEXT = 'Demucs'
DENOISE_OUTPUT_TEXT = 'Denoise Output'
DEVERB_VOCALS_TEXT = 'Deverb Vocals'
DONE_TEXT = 'Done'
DOWNLOAD_CENTER_TEXT = 'Download Center'
DOWNLOAD_CODE_TEXT = 'Download Code'
DOWNLOAD_LINKS_TEXT = 'Download Link(s)'
DOWNLOAD_UPDATE_IN_APPLICATION_TEXT = 'Download Update in Application'
ENABLE_HELP_HINTS_TEXT = 'Enable Help Hints'
ENABLE_TTA_TEXT = 'Enable TTA'
ENABLE_VOCAL_SPLIT_MODE_TEXT = 'Enable Vocal Split Mode'
ENSEMBLE_NAME_TEXT = 'Ensemble Name'
ENSEMBLE_WAVFORMS_TEXT = 'Ensemble Wavforms'
ERROR_CONSOLE_TEXT = 'Error Console'
GENERAL_MENU_TEXT = 'General Menu'
GENERAL_PROCESS_SETTINGS_TEXT = 'General Process Settings'
GENERATE_MODEL_FOLDER_TEXT = 'Generate Model Folder'
HIGHEND_PROCESS_TEXT = 'High-End Process'
INPUT_CODE_TEXT = 'Input Code'
INPUT_STEM_NAME_TEXT = 'Input Stem Name'
INPUT_UNIQUE_STEM_NAME_TEXT = 'Input Unique Stem Name'
IS_INVERSE_STEM_TEXT = 'Is Inverse Stem'
KARAOKE_MODEL_TEXT = 'Karaoke Model'
MANUAL_DOWNLOADS_TEXT = 'Manual Downloads'
MATCH_FREQ_CUTOFF_TEXT = 'Match Freq Cut-off'
MDXNET_C_MODEL_PARAMETERS_TEXT = 'MDX-Net C Model Parameters'
MDXNET_MODEL_SETTINGS_TEXT = 'MDX-Net Model Settings'
MDXNET_TEXT = 'MDX-Net'
MODEL_PARAMETERS_CHANGED_TEXT = 'Model Parameters Changed'
MODEL_SAMPLE_MODE_SETTINGS_TEXT = 'Model Sample Mode Settings'
MODEL_TEST_MODE_TEXT = 'Model Test Mode'
MP3_BITRATE_TEXT = 'Mp3 Bitrate'
NAME_SETTINGS_TEXT = 'Name Settings'
NO_DEFINED_PARAMETERS_FOUND_TEXT = 'No Defined Parameters Found'
NO_TEXT = 'No'
NORMALIZE_OUTPUT_TEXT = 'Normalize Output'
USE_OPENCL_TEXT = 'Use OpenCL'
NOT_ENOUGH_MODELS_TEXT = 'Not Enough Models'
NOTIFICATION_CHIMES_TEXT = 'Notification Chimes'
OPEN_APPLICATION_DIRECTORY_TEXT = 'Open Application Directory'
OPEN_LINK_TO_MODEL_TEXT = 'Open Link to Model'
OPEN_MODEL_DIRECTORY_TEXT = 'Open Model Directory'
OPEN_MODEL_FOLDER_TEXT = 'Open Model Folder'
OPEN_MODELS_FOLDER_TEXT = 'Open Models Folder'
PHASE_SHIFTS_TEXT = 'Phase Shifts'
POST_PROCESS_TEXT = 'Post-Process'
POST_PROCESS_THRESHOLD_TEXT = 'Post-process Threshold'
PREPROCESS_MODEL_CHOOSE_TEXT = 'Pre-process Model'
PRIMARY_STEM_TEXT = 'Primary Stem'
REFRESH_LIST_TEXT = 'Refresh List'
REMOVE_SAVED_ENSEMBLE_TEXT = 'Remove Saved Ensemble'
REPORT_ISSUE_TEXT = 'Report Issue'
RESET_ALL_SETTINGS_TO_DEFAULT_TEXT = 'Reset All Settings to Default'
RESTART_APPLICATION_TEXT = 'Restart Application'
SAMPLE_CLIP_DURATION_TEXT = 'Sample Clip Duration'
SAVE_ALIGNED_TRACK_TEXT = 'Save Aligned Track'
SAVE_ALL_OUTPUTS_TEXT = 'Save All Outputs'
SAVE_CURRENT_ENSEMBLE_TEXT = 'Save Current Ensemble'
SAVE_CURRENT_SETTINGS_TEXT = 'Save Current Settings'
SAVE_INSTRUMENTAL_MIXTURE_TEXT = 'Save Instrumental Mixture'
SAVE_SPLIT_VOCAL_INSTRUMENTALS_TEXT = 'Save Split Vocal Instrumentals'
SECONDARY_MODEL_TEXT = 'Secondary Model'
SECONDARY_PHASE_TEXT = 'Secondary Phase'
SECONDS_TEXT = 'Seconds'
SEGMENT_DEFAULT_TEXT = 'Segment Default'
SEGMENT_SIZE_TEXT = 'Segment Size'
SEGMENTS_TEXT = 'Segments'
SELECT_DOWNLOAD_TEXT = 'Select Download'
SELECT_MODEL_PARAM_TEXT = 'Select Model Param'
SELECT_VOCAL_TYPE_TO_DEVERB_TEXT = 'Select Vocal Type to Deverb'
SELECTED_MODEL_PLACEMENT_PATH_TEXT = 'Selected Model Placement Path'
SETTINGS_GUIDE_TEXT = 'Settings Guide'
SETTINGS_TEST_MODE_TEXT = 'Settings Test Mode'
SHIFT_CONVERSION_PITCH_TEXT = 'Shift Conversion Pitch'
SHIFTS_TEXT = 'Shifts'
SILENCE_MATCHING_TEXT = 'Silence Matching'
SPECIFY_MDX_NET_MODEL_PARAMETERS_TEXT = 'Specify MDX-Net Model Parameters'
SPECIFY_PARAMETERS_TEXT = 'Specify Parameters'
SPECIFY_VR_MODEL_PARAMETERS_TEXT = 'Specify VR Model Parameters'
SPECTRAL_INVERSION_TEXT = 'Spectral Inversion'
SPECTRAL_MATCHING_TEXT = 'Spectral Matching'   
SPLIT_MODE_TEXT = 'Split Mode'
STEM_NAME_TEXT = 'Stem Name'
STOP_DOWNLOAD_TEXT = 'Stop Download'
SUPPORT_UVR_TEXT = 'Support UVR'
TRY_MANUAL_DOWNLOAD_TEXT = 'Try Manual Download'
UPDATE_FOUND_TEXT = 'Update Found'
USER_DOWNLOAD_CODES_TEXT = 'User Download Codes'
UVR_BUY_ME_A_COFFEE_LINK_TEXT = 'UVR \'Buy Me a Coffee\' Link'
UVR_ERROR_LOG_TEXT = 'UVR Error Log'
UVR_PATREON_LINK_TEXT = 'UVR Patreon Link'
VOCAL_DEVERB_OPTIONS_TEXT = 'Vocal Deverb Options'
VOCAL_SPLIT_MODE_OPTIONS_TEXT = 'Vocal Split Mode Options'
VOCAL_SPLIT_OPTIONS_TEXT = 'Vocal Split Options'
VOLUME_COMPENSATION_TEXT = 'Volume Compensation'
VR_51_MODEL_TEXT = 'VR 5.1 Model'
VR_ARCH_TEXT = 'VR Arch'
WAV_TYPE_TEXT = 'Wav Type'
CUDA_NUM_TEXT = 'GPU Device'
WINDOW_SIZE_TEXT = 'Window Size'
YES_TEXT = 'Yes'
VERIFY_INPUTS_TEXT = 'Verify Inputs'
AUDIO_INPUT_TOTAL_TEXT = 'Audio Input Total'
MDX23C_ONLY_OPTIONS_TEXT = 'MDXNET23 Only Options'
PROCESS_STARTING_TEXT = 'Process starting... '
MISSING_MESS_TEXT = 'is missing or currupted.'
SIMILAR_TEXT = "are the same."
LOADING_VERSION_INFO_TEXT = 'Loading version information...'
CHECK_FOR_UPDATES_TEXT = 'Check for Updates'
INFO_UNAVAILABLE_TEXT = "Information unavailable."
UPDATE_CONFIRMATION_TEXT = 'Are you sure you want to continue?\n\nThe application will need to be restarted.\n'
BROKEN_OR_INCOM_TEXT = 'Broken or Incompatible File(s) Removed. Check Error Log for details.'
BMAC_UVR_TEXT = 'UVR \"Buy Me a Coffee\" Link'
MDX_MENU_WAR_TEXT = '(Leave this setting as is if you are unsure.)'
NO_FILES_TEXT = 'No Files'
CHOOSE_INPUT_TEXT = 'Choose Input'
OPEN_INPUT_DIR_TEXT = 'Open Input Directory'
BATCH_PROCESS_MENU_TEXT = 'Batch Process Menu'
TEMP_FILE_DELETION_TEXT = 'Temp File Deletion'
VOCAL_SPLITTER_OPTIONS_TEXT = 'Vocal Splitter Options'
WAVEFORM_ENSEMBLE_TEXT = 'Waveform Ensemble'
SELECT_INPUT_TEXT = 'Select Input'
SELECT_OUTPUT_TEXT = 'Select Output'
TIME_CORRECTION_TEXT = 'Time Correction'
UVR_LIS_INFO_TEXT = 'UVR License Information'
ADDITIONAL_RES_CREDITS_TEXT = 'Additional Resources & Credits'
SAVE_INST_MIXTURE_TEXT = 'Save Instrumental Mixture'
DOWNLOAD_UPDATE_IN_APP_TEXT = 'Download Update in Application'
WAVE_TYPE_TEXT = 'WAVE TYPE'
OPEN_LINK_TO_MODEL_TEXT = "Open Link to Model"
OPEN_MODEL_DIRECTORY = "Open Model Directory"
SELECTED_MODEL_PLACE_PATH_TEXT = 'Selected Model Placement Path'
IS_INVERSE_STEM_TEXT = "Is Inverse Stem"
INPUT_STEM_NAME_TEXT = "Input Stem Name"
INPUT_UNIQUE_STEM_NAME_TEXT = "Input Unique Stem Name"
DONE_MENU_TEXT = "Done"
OK_TEXT = "Ok"
ENSEMBLE_WARNING_NOT_ENOUGH_SHORT_TEXT = "Not Enough Models"
ENSEMBLE_WARNING_NOT_ENOUGH_TEXT = "You must select 2 or more models to save an ensemble."
NOT_ENOUGH_ERROR_TEXT = "Not enough files to process.\n"
INVALID_FOLDER_ERROR_TEXT = 'Invalid Folder', 'Your given export path is not a valid folder!'

GET_DL_VIP_CODE_TEXT = ("Obtain codes by visiting one of the following links below."
                        "\nFrom there you can donate, pledge, "
                        "or just obatain the code!\n (Donations are not required to obtain VIP code)")
CONFIRM_RESTART_TEXT = 'Restart Confirmation', 'This will restart the application and halt any running processes. Your current settings will be saved. \n\n Are you sure you wish to continue?'
ERROR_LOADING_FILE_TEXT = 'Error Loading the Following File', 'Raw Error Details'
LOADING_MODEL_TEXT = 'Loading model'
FULL_APP_SET_TEXT = 'Full Application Settings'
PROCESS_STARTING_TEXT = 'Process starting... '
PROCESS_STOPPED_BY_USER = '\n\nProcess stopped by user.'
NEW_UPDATE_FOUND_TEXT = lambda version:f"\n\nNew Update Found: {version}\n\nClick the update button in the \"Settings\" menu to download and install!"
ROLL_BACK_TEXT = 'Click Here to Roll Back'

def secondary_stem(stem:str):
    """Determines secondary stem"""
    
    stem = stem if stem else NO_STEM
    
    if stem in STEM_PAIR_MAPPER.keys():
        for key, value in STEM_PAIR_MAPPER.items():
            if stem in key:
                secondary_stem = value
    else:
        secondary_stem = stem.replace(NO_STEM, "") if NO_STEM in stem else f"{NO_STEM}{stem}"
    
    return secondary_stem

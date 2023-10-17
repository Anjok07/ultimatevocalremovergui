import audioread
import librosa
import numpy as np
import soundfile as sf
import math
import platform
import traceback
from . import pyrb
from scipy.signal import correlate, hilbert
import io

OPERATING_SYSTEM = platform.system()
SYSTEM_ARCH = platform.platform()
SYSTEM_PROC = platform.processor()
ARM = 'arm'

AUTO_PHASE = "Automatic"
POSITIVE_PHASE = "Positive Phase"
NEGATIVE_PHASE = "Negative Phase"
NONE_P = "None",
LOW_P = "Shifts: Low",
MED_P = "Shifts: Medium",
HIGH_P = "Shifts: High",
VHIGH_P = "Shifts: Very High"
MAXIMUM_P = "Shifts: Maximum"

progress_value = 0
last_update_time = 0
is_macos = False

if OPERATING_SYSTEM == 'Windows':
    from pyrubberband import pyrb
else:
    from . import pyrb

if OPERATING_SYSTEM == 'Darwin':
    wav_resolution = "polyphase" if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else "sinc_fastest" 
    wav_resolution_float_resampling = "kaiser_best" if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else wav_resolution 
    is_macos = True
else:
    wav_resolution = "sinc_fastest"
    wav_resolution_float_resampling = wav_resolution 

MAX_SPEC = 'Max Spec'
MIN_SPEC = 'Min Spec'
LIN_ENSE = 'Linear Ensemble'

MAX_WAV = MAX_SPEC
MIN_WAV = MIN_SPEC

AVERAGE = 'Average'

def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError('h1_shape[3] must be greater than h2_shape[3]')

    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1

def preprocess(X_spec):
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    return X_mag, X_phase

def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size

def normalize(wave, is_normalize=False):
    """Normalize audio"""

    maxv = np.abs(wave).max()
    if maxv > 1.0:
        if is_normalize:
            print("Above clipping threshold.")
            wave /= maxv
    
    return wave
    
def auto_transpose(audio_array:np.ndarray):
    """
    Ensure that the audio array is in the (channels, samples) format.

    Parameters:
        audio_array (ndarray): Input audio array.

    Returns:
        ndarray: Transposed audio array if necessary.
    """
    
    # If the second dimension is 2 (indicating stereo channels), transpose the array
    if audio_array.shape[1] == 2:
        return audio_array.T
    return audio_array

def write_array_to_mem(audio_data, subtype):
    if isinstance(audio_data, np.ndarray):
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, 44100, subtype=subtype, format='WAV')
        audio_buffer.seek(0)
        return audio_buffer
    else:
        return audio_data

def spectrogram_to_image(spec, mode='magnitude'):
    if mode == 'magnitude':
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        y = np.log10(y ** 2 + 1e-8)
    elif mode == 'phase':
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([
            np.max(img, axis=2, keepdims=True), img
        ], axis=2)

    return img

def reduce_vocal_aggressively(X, y, softmask):
    v = X - y
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(v)

    v_mask = v_mag_tmp > y_mag_tmp
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)

    return y_mag * np.exp(1.j * np.angle(y))

def merge_artifacts(y_mask, thres=0.01, min_range=64, fade_size=32):
    mask = y_mask
    
    try:
        if min_range < fade_size * 2:
            raise ValueError('min_range must be >= fade_size * 2')

        idx = np.where(y_mask.min(axis=(0, 1)) > thres)[0]
        start_idx = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
        end_idx = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
        artifact_idx = np.where(end_idx - start_idx > min_range)[0]
        weight = np.zeros_like(y_mask)
        if len(artifact_idx) > 0:
            start_idx = start_idx[artifact_idx]
            end_idx = end_idx[artifact_idx]
            old_e = None
            for s, e in zip(start_idx, end_idx):
                if old_e is not None and s - old_e < fade_size:
                    s = old_e - fade_size * 2

                if s != 0:
                    weight[:, :, s:s + fade_size] = np.linspace(0, 1, fade_size)
                else:
                    s -= fade_size

                if e != y_mask.shape[2]:
                    weight[:, :, e - fade_size:e] = np.linspace(1, 0, fade_size)
                else:
                    e += fade_size

                weight[:, :, s + fade_size:e - fade_size] = 1
                old_e = e

        v_mask = 1 - y_mask
        y_mask += weight * v_mask
        
        mask = y_mask
    except Exception as e:
        error_name = f'{type(e).__name__}'
        traceback_text = ''.join(traceback.format_tb(e.__traceback__))
        message = f'{error_name}: "{e}"\n{traceback_text}"'
        print('Post Process Failed: ', message)
        
    return mask

def align_wave_head_and_tail(a, b):
    l = min([a[0].size, b[0].size])  
    
    return a[:l,:l], b[:l,:l]
    
def convert_channels(spec, mp, band):
    cc = mp.param['band'][band].get('convert_channels')

    if 'mid_side_c' == cc:
        spec_left = np.add(spec[0], spec[1] * .25)
        spec_right = np.subtract(spec[1], spec[0] * .25)
    elif 'mid_side' == cc:
        spec_left = np.add(spec[0], spec[1]) / 2
        spec_right = np.subtract(spec[0], spec[1])
    elif 'stereo_n' == cc:
        spec_left = np.add(spec[0], spec[1] * .25) / 0.9375
        spec_right = np.add(spec[1], spec[0] * .25) / 0.9375
    else:
        return spec
        
    return np.asfortranarray([spec_left, spec_right])
    
def combine_spectrograms(specs, mp, is_v51_model=False):
    l = min([specs[i].shape[2] for i in specs])    
    spec_c = np.zeros(shape=(2, mp.param['bins'] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param['band'])
    
    for d in range(1, bands_n + 1):
        h = mp.param['band'][d]['crop_stop'] - mp.param['band'][d]['crop_start']
        spec_c[:, offset:offset+h, :l] = specs[d][:, mp.param['band'][d]['crop_start']:mp.param['band'][d]['crop_stop'], :l]
        offset += h
        
    if offset > mp.param['bins']:
        raise ValueError('Too much bins')
        
    # lowpass fiter
    
    if mp.param['pre_filter_start'] > 0:
        if is_v51_model:
            spec_c *= get_lp_filter_mask(spec_c.shape[1], mp.param['pre_filter_start'], mp.param['pre_filter_stop'])
        else:
            if bands_n == 1:
                spec_c = fft_lp_filter(spec_c, mp.param['pre_filter_start'], mp.param['pre_filter_stop'])
            else:
                gp = 1        
                for b in range(mp.param['pre_filter_start'] + 1, mp.param['pre_filter_stop']):
                    g = math.pow(10, -(b - mp.param['pre_filter_start']) * (3.5 - gp) / 20.0)
                    gp = g
                    spec_c[:, b, :] *= g
                
    return np.asfortranarray(spec_c)
    
def wave_to_spectrogram(wave, hop_length, n_fft, mp, band, is_v51_model=False):

    if wave.ndim == 1:
        wave = np.asfortranarray([wave,wave])

    if not is_v51_model:
        if mp.param['reverse']:
            wave_left = np.flip(np.asfortranarray(wave[0]))
            wave_right = np.flip(np.asfortranarray(wave[1]))
        elif mp.param['mid_side']:
            wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
        elif mp.param['mid_side_b2']:
            wave_left = np.asfortranarray(np.add(wave[1], wave[0] * .5))
            wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * .5))
        else:
            wave_left = np.asfortranarray(wave[0])
            wave_right = np.asfortranarray(wave[1])
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    
    spec = np.asfortranarray([spec_left, spec_right])

    if is_v51_model:
        spec = convert_channels(spec, mp, band)

    return spec

def spectrogram_to_wave(spec, hop_length=1024, mp={}, band=0, is_v51_model=True):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    
    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    
    if is_v51_model:
        cc = mp.param['band'][band].get('convert_channels')
        if 'mid_side_c' == cc:
            return np.asfortranarray([np.subtract(wave_left / 1.0625, wave_right / 4.25), np.add(wave_right / 1.0625, wave_left / 4.25)])    
        elif 'mid_side' == cc:
            return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif 'stereo_n' == cc:
            return np.asfortranarray([np.subtract(wave_left, wave_right * .25), np.subtract(wave_right, wave_left * .25)])
    else:
        if mp.param['reverse']:
            return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
        elif mp.param['mid_side']:
            return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
        elif mp.param['mid_side_b2']:
            return np.asfortranarray([np.add(wave_right / 1.25, .4 * wave_left), np.subtract(wave_left / 1.25, .4 * wave_right)])
    
    return np.asfortranarray([wave_left, wave_right])
    
def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None, is_v51_model=False):
    bands_n = len(mp.param['band'])    
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param['band'][d]
        spec_s = np.ndarray(shape=(2, bp['n_fft'] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp['crop_stop'] - bp['crop_start']
        spec_s[:, bp['crop_start']:bp['crop_stop'], :] = spec_m[:, offset:offset+h, :]
                
        offset += h
        if d == bands_n: # higher
            if extra_bins_h: # if --high_end_process bypass
                max_bin = bp['n_fft'] // 2
                spec_s[:, max_bin-extra_bins_h:max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp['hpf_start'] > 0:
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp['hpf_start'], bp['hpf_stop'] - 1)
                else:
                    spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model)
            else:
                wave = np.add(wave, spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model))
        else:
            sr = mp.param['band'][d+1]['sr']
            if d == 1: # lower
                if is_v51_model:
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp['lpf_start'], bp['lpf_stop'])
                else:
                    spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave = librosa.resample(spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model), bp['sr'], sr, res_type=wav_resolution)
            else: # mid
                if is_v51_model:
                    spec_s *= get_hp_filter_mask(spec_s.shape[1], bp['hpf_start'], bp['hpf_stop'] - 1)
                    spec_s *= get_lp_filter_mask(spec_s.shape[1], bp['lpf_start'], bp['lpf_stop'])
                else:
                    spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
                    spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                    
                wave2 = np.add(wave, spectrogram_to_wave(spec_s, bp['hl'], mp, d, is_v51_model))
                wave = librosa.resample(wave2, bp['sr'], sr, res_type=wav_resolution)
        
    return wave

def get_lp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate([
        np.ones((bin_start - 1, 1)),
        np.linspace(1, 0, bin_stop - bin_start + 1)[:, None],
        np.zeros((n_bins - bin_stop, 1))
    ], axis=0)

    return mask
    
def get_hp_filter_mask(n_bins, bin_start, bin_stop):
    mask = np.concatenate([
        np.zeros((bin_stop + 1, 1)),
        np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None],
        np.ones((n_bins - bin_start - 2, 1))
    ], axis=0)

    return mask

def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]
        
    spec[:, bin_stop:, :] *= 0

    return spec

def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]
    
    spec[:, 0:bin_stop+1, :] *= 0

    return spec

def spectrogram_to_wave_old(spec, hop_length=1024):
    if spec.ndim == 2:
        wave = librosa.istft(spec, hop_length=hop_length)
    elif spec.ndim == 3:
        spec_left = np.asfortranarray(spec[0])
        spec_right = np.asfortranarray(spec[1])

        wave_left = librosa.istft(spec_left, hop_length=hop_length)
        wave_right = librosa.istft(spec_right, hop_length=hop_length)
        wave = np.asfortranarray([wave_left, wave_right])

    return wave
    
def wave_to_spectrogram_old(wave, hop_length, n_fft):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    
    spec = np.asfortranarray([spec_left, spec_right])

    return spec

def mirroring(a, spec_m, input_high_end, mp):
    if 'mirroring' == a:
        mirror = np.flip(np.abs(spec_m[:, mp.param['pre_filter_start']-10-input_high_end.shape[1]:mp.param['pre_filter_start']-10, :]), 1)
        mirror = mirror * np.exp(1.j * np.angle(input_high_end))
        
        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)
        
    if 'mirroring2' == a:
        mirror = np.flip(np.abs(spec_m[:, mp.param['pre_filter_start']-10-input_high_end.shape[1]:mp.param['pre_filter_start']-10, :]), 1)
        mi = np.multiply(mirror, input_high_end * 1.7)
        
        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)

def adjust_aggr(mask, is_non_accom_stem, aggressiveness):
    aggr = aggressiveness['value'] * 2

    if aggr != 0:
        if is_non_accom_stem:
            aggr = 1 - aggr
    
        aggr = [aggr, aggr]
    
        if aggressiveness['aggr_correction'] is not None:
            aggr[0] += aggressiveness['aggr_correction']['left']
            aggr[1] += aggressiveness['aggr_correction']['right']

        for ch in range(2):
            mask[ch, :aggressiveness['split_bin']] = np.power(mask[ch, :aggressiveness['split_bin']], 1 + aggr[ch] / 3)
            mask[ch, aggressiveness['split_bin']:] = np.power(mask[ch, aggressiveness['split_bin']:], 1 + aggr[ch])

    return mask

def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec

def istft(spec, hl):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hl)
    wave_right = librosa.istft(spec_right, hop_length=hl)
    wave = np.asfortranarray([wave_left, wave_right])

    return wave

def spec_effects(wave, algorithm='Default', value=None):
    spec = [stft(wave[0],2048,1024), stft(wave[1],2048,1024)]
    if algorithm == 'Min_Mag':
        v_spec_m = np.where(np.abs(spec[1]) <= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'Max_Mag':
        v_spec_m = np.where(np.abs(spec[1]) >= np.abs(spec[0]), spec[1], spec[0])
        wave = istft(v_spec_m,1024)
    elif algorithm == 'Default':
        wave = (wave[1] * value) + (wave[0] * (1-value))
    elif algorithm == 'Invert_p':
        X_mag = np.abs(spec[0])
        y_mag = np.abs(spec[1])            
        max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
        v_spec = spec[1] - max_mag * np.exp(1.j * np.angle(spec[0]))
        wave = istft(v_spec,1024)
            
    return wave      

def spectrogram_to_wave_no_mp(spec, n_fft=2048, hop_length=1024):
    wave = librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)
    
    if wave.ndim == 1:
        wave = np.asfortranarray([wave,wave])

    return wave

def wave_to_spectrogram_no_mp(wave):
    
    spec = librosa.stft(wave, n_fft=2048, hop_length=1024)
    
    if spec.ndim == 1:
        spec = np.asfortranarray([spec,spec])

    return spec

def invert_audio(specs, invert_p=True):
    
    ln = min([specs[0].shape[2], specs[1].shape[2]])
    specs[0] = specs[0][:,:,:ln]
    specs[1] = specs[1][:,:,:ln]
        
    if invert_p:
        X_mag = np.abs(specs[0])
        y_mag = np.abs(specs[1])            
        max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
        v_spec = specs[1] - max_mag * np.exp(1.j * np.angle(specs[0]))
    else:
        specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
        v_spec = specs[0] - specs[1]

    return v_spec

def invert_stem(mixture, stem):
    mixture = wave_to_spectrogram_no_mp(mixture)
    stem = wave_to_spectrogram_no_mp(stem)
    output = spectrogram_to_wave_no_mp(invert_audio([mixture, stem]))

    return -output.T

def ensembling(a, inputs, is_wavs=False): 

    for i in range(1, len(inputs)):
        if i == 1:
            input = inputs[0]

        if is_wavs:
            ln = min([input.shape[1], inputs[i].shape[1]])
            input = input[:,:ln]
            inputs[i] = inputs[i][:,:ln]
        else:
            ln = min([input.shape[2], inputs[i].shape[2]])
            input = input[:,:,:ln]
            inputs[i] = inputs[i][:,:,:ln]
        
        if MIN_SPEC == a:
            input = np.where(np.abs(inputs[i]) <= np.abs(input), inputs[i], input)
        if MAX_SPEC == a:
            input = np.where(np.abs(inputs[i]) >= np.abs(input), inputs[i], input)  

    #linear_ensemble
    #input = ensemble_wav(inputs, split_size=1)

    return input

def ensemble_for_align(waves):
    
    specs = []
    
    for wav in waves:
        spec = wave_to_spectrogram_no_mp(wav.T)
        specs.append(spec)
        
    wav_aligned = spectrogram_to_wave_no_mp(ensembling(MIN_SPEC, specs)).T
    wav_aligned = match_array_shapes(wav_aligned, waves[1], is_swap=True)    
   
    return wav_aligned
    
def ensemble_inputs(audio_input, algorithm, is_normalization, wav_type_set, save_path, is_wave=False, is_array=False):

    wavs_ = []
    
    if algorithm == AVERAGE:
        output = average_audio(audio_input)
        samplerate = 44100
    else:
        specs = []
        
        for i in range(len(audio_input)):  
            wave, samplerate = librosa.load(audio_input[i], mono=False, sr=44100)
            wavs_.append(wave)
            spec = wave if is_wave else wave_to_spectrogram_no_mp(wave)
            specs.append(spec)
        
        wave_shapes = [w.shape[1] for w in wavs_]
        target_shape = wavs_[wave_shapes.index(max(wave_shapes))]
        
        if is_wave:
            output = ensembling(algorithm, specs, is_wavs=True)
        else:
            output = spectrogram_to_wave_no_mp(ensembling(algorithm, specs))
            
        output = to_shape(output, target_shape.shape)

    sf.write(save_path, normalize(output.T, is_normalization), samplerate, subtype=wav_type_set)

def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = (target_dim - x_dim)
        pad_tuple = ((0, pad_value))
        padding_list.append(pad_tuple)
    
    return np.pad(x, tuple(padding_list), mode='constant')

def to_shape_minimize(x: np.ndarray, target_shape):
    
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = (target_dim - x_dim)
        pad_tuple = ((0, pad_value))
        padding_list.append(pad_tuple)
    
    return np.pad(x, tuple(padding_list), mode='constant')

def detect_leading_silence(audio, sr, silence_threshold=0.007, frame_length=1024):
    """
    Detect silence at the beginning of an audio signal.

    :param audio: np.array, audio signal
    :param sr: int, sample rate
    :param silence_threshold: float, magnitude threshold below which is considered silence
    :param frame_length: int, the number of samples to consider for each check

    :return: float, duration of the leading silence in milliseconds
    """
    
    if len(audio.shape) == 2:
        # If stereo, pick the channel with more energy to determine the silence
        channel = np.argmax(np.sum(np.abs(audio), axis=1))
        audio = audio[channel]
    
    for i in range(0, len(audio), frame_length):
        if np.max(np.abs(audio[i:i+frame_length])) > silence_threshold:
            return (i / sr) * 1000

    return (len(audio) / sr) * 1000

def adjust_leading_silence(target_audio, reference_audio, silence_threshold=0.01, frame_length=1024):
    """
    Adjust the leading silence of the target_audio to match the leading silence of the reference_audio.

    :param target_audio: np.array, audio signal that will have its silence adjusted
    :param reference_audio: np.array, audio signal used as a reference
    :param sr: int, sample rate
    :param silence_threshold: float, magnitude threshold below which is considered silence
    :param frame_length: int, the number of samples to consider for each check

    :return: np.array, target_audio adjusted to have the same leading silence as reference_audio
    """
    
    def find_silence_end(audio):
        if len(audio.shape) == 2:
            # If stereo, pick the channel with more energy to determine the silence
            channel = np.argmax(np.sum(np.abs(audio), axis=1))
            audio_mono = audio[channel]
        else:
            audio_mono = audio

        for i in range(0, len(audio_mono), frame_length):
            if np.max(np.abs(audio_mono[i:i+frame_length])) > silence_threshold:
                return i
        return len(audio_mono)

    ref_silence_end = find_silence_end(reference_audio)
    target_silence_end = find_silence_end(target_audio)
    silence_difference = ref_silence_end - target_silence_end

    try:
        ref_silence_end_p = (ref_silence_end / 44100) * 1000
        target_silence_end_p = (target_silence_end / 44100) * 1000
        silence_difference_p = ref_silence_end_p - target_silence_end_p
        print("silence_difference: ", silence_difference_p)
    except Exception as e:
        pass

    if silence_difference > 0:  # Add silence to target_audio
        if len(target_audio.shape) == 2:  # stereo
            silence_to_add = np.zeros((target_audio.shape[0], silence_difference))
        else:  # mono
            silence_to_add = np.zeros(silence_difference)
        return np.hstack((silence_to_add, target_audio))
    elif silence_difference < 0:  # Remove silence from target_audio
        if len(target_audio.shape) == 2:  # stereo
            return target_audio[:, -silence_difference:]
        else:  # mono
            return target_audio[-silence_difference:]
    else:  # No adjustment needed
        return target_audio

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray, is_swap=False):
    
    if is_swap:
        array_1, array_2 = array_1.T, array_2.T
    
    #print("before", array_1.shape, array_2.shape)
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:,:array_2.shape[1]] 
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    
    #print("after", array_1.shape, array_2.shape)
    
    if is_swap:
        array_1, array_2 = array_1.T, array_2.T
        
    return array_1

def match_mono_array_shapes(array_1: np.ndarray, array_2: np.ndarray):
    
    if len(array_1) > len(array_2):
        array_1 = array_1[:len(array_2)]
    elif len(array_1) < len(array_2):
        padding = len(array_2) - len(array_1)
        array_1 = np.pad(array_1, (0, padding), 'constant', constant_values=0)
        
    return array_1

def change_pitch_semitones(y, sr, semitone_shift):
    factor = 2 ** (semitone_shift / 12)  # Convert semitone shift to factor for resampling
    y_pitch_tuned = []
    for y_channel in y:
        y_pitch_tuned.append(librosa.resample(y_channel, sr, sr*factor, res_type=wav_resolution_float_resampling))
    y_pitch_tuned = np.array(y_pitch_tuned)
    new_sr = sr * factor
    return y_pitch_tuned, new_sr

def augment_audio(export_path, audio_file, rate, is_normalization, wav_type_set, save_format=None, is_pitch=False, is_time_correction=True):

    wav, sr = librosa.load(audio_file, sr=44100, mono=False)

    if wav.ndim == 1:
        wav = np.asfortranarray([wav,wav])

    if not is_time_correction:
        wav_mix = change_pitch_semitones(wav, 44100, semitone_shift=-rate)[0]
    else:
        if is_pitch:
            wav_1 = pyrb.pitch_shift(wav[0], sr, rate, rbargs=None)
            wav_2 = pyrb.pitch_shift(wav[1], sr, rate, rbargs=None)
        else:
            wav_1 = pyrb.time_stretch(wav[0], sr, rate, rbargs=None)
            wav_2 = pyrb.time_stretch(wav[1], sr, rate, rbargs=None)

        if wav_1.shape > wav_2.shape:
            wav_2 = to_shape(wav_2, wav_1.shape)
        if wav_1.shape < wav_2.shape:
            wav_1 = to_shape(wav_1, wav_2.shape)
            
        wav_mix = np.asfortranarray([wav_1, wav_2])
    
    sf.write(export_path, normalize(wav_mix.T, is_normalization), sr, subtype=wav_type_set)
    save_format(export_path)
    
def average_audio(audio):
    
    waves = []
    wave_shapes = []
    final_waves = []

    for i in range(len(audio)):
        wave = librosa.load(audio[i], sr=44100, mono=False)
        waves.append(wave[0])
        wave_shapes.append(wave[0].shape[1])

    wave_shapes_index = wave_shapes.index(max(wave_shapes))
    target_shape = waves[wave_shapes_index]
    waves.pop(wave_shapes_index)
    final_waves.append(target_shape)

    for n_array in waves:
        wav_target = to_shape(n_array, target_shape.shape)
        final_waves.append(wav_target)

    waves = sum(final_waves)
    waves = waves/len(audio)

    return waves
    
def average_dual_sources(wav_1, wav_2, value):
    
    if wav_1.shape > wav_2.shape:
        wav_2 = to_shape(wav_2, wav_1.shape)
    if wav_1.shape < wav_2.shape:
        wav_1 = to_shape(wav_1, wav_2.shape)

    wave = (wav_1 * value) + (wav_2 * (1-value))

    return wave
       
def reshape_sources(wav_1: np.ndarray, wav_2: np.ndarray):
    
    if wav_1.shape > wav_2.shape:
        wav_2 = to_shape(wav_2, wav_1.shape)
    if wav_1.shape < wav_2.shape:
        ln = min([wav_1.shape[1], wav_2.shape[1]])
        wav_2 = wav_2[:,:ln]

    ln = min([wav_1.shape[1], wav_2.shape[1]])
    wav_1 = wav_1[:,:ln]
    wav_2 = wav_2[:,:ln]

    return wav_2
    
def reshape_sources_ref(wav_1_shape, wav_2: np.ndarray):
    
    if wav_1_shape > wav_2.shape:
        wav_2 = to_shape(wav_2, wav_1_shape)

    return wav_2
    
def combine_arrarys(audio_sources, is_swap=False):
    source = np.zeros_like(max(audio_sources, key=np.size))
    
    for v in audio_sources:
        v = match_array_shapes(v, source, is_swap=is_swap)
        source += v
        
    return source
    
def combine_audio(paths: list, audio_file_base=None, wav_type_set='FLOAT', save_format=None):
    
    source = combine_arrarys([load_audio(i) for i in paths])
    save_path = f"{audio_file_base}_combined.wav"
    sf.write(save_path, source.T, 44100, subtype=wav_type_set)
    save_format(save_path)
    
def reduce_mix_bv(inst_source, voc_source, reduction_rate=0.9):
    # Reduce the volume
    inst_source = inst_source * (1 - reduction_rate)

    mix_reduced = combine_arrarys([inst_source, voc_source], is_swap=True)

    return mix_reduced
    
def organize_inputs(inputs):
    input_list = {
        "target":None,
        "reference":None,
        "reverb":None,
        "inst":None
    }
    
    for i in inputs:
        if i.endswith("_(Vocals).wav"):
            input_list["reference"] = i
        elif "_RVC_" in i:
            input_list["target"] = i
        elif i.endswith("reverbed_stem.wav"):
            input_list["reverb"] = i
        elif i.endswith("_(Instrumental).wav"):
            input_list["inst"] = i
            
    return input_list
      
def check_if_phase_inverted(wav1, wav2, is_mono=False):
    # Load the audio files
    if not is_mono:
        wav1 = np.mean(wav1, axis=0)
        wav2 = np.mean(wav2, axis=0)
    
    # Compute the correlation
    correlation = np.corrcoef(wav1[:1000], wav2[:1000])
    
    return correlation[0,1] < 0
         
def align_audio(file1, 
                file2, 
                file2_aligned, 
                file_subtracted, 
                wav_type_set, 
                is_save_aligned, 
                command_Text, 
                save_format, 
                align_window:list, 
                align_intro_val:list,
                db_analysis:tuple,
                set_progress_bar,
                phase_option,
                phase_shifts,
                is_match_silence,
                is_spec_match):
    
    global progress_value
    progress_value = 0
    is_mono = False
    
    def get_diff(a, b):
        corr = np.correlate(a, b, "full")
        diff = corr.argmax() - (b.shape[0] - 1)

        return diff

    def progress_bar(length):
        global progress_value
        progress_value += 1

        if (0.90/length*progress_value) >= 0.9:
            length = progress_value + 1

        set_progress_bar(0.1, (0.9/length*progress_value))
    
    # read tracks
    
    if file1.endswith(".mp3") and is_macos:
        length1 = rerun_mp3(file1)
        wav1, sr1 = librosa.load(file1, duration=length1, sr=44100, mono=False)
    else:
        wav1, sr1 = librosa.load(file1, sr=44100, mono=False)

    if file2.endswith(".mp3") and is_macos:
        length2 = rerun_mp3(file2)
        wav2, sr2 = librosa.load(file2, duration=length2, sr=44100, mono=False)
    else:
        wav2, sr2 = librosa.load(file2, sr=44100, mono=False)

    if wav1.ndim == 1 and wav2.ndim == 1:
         is_mono = True
    elif wav1.ndim == 1:
        wav1 = np.asfortranarray([wav1,wav1])
    elif wav2.ndim == 1:
        wav2 = np.asfortranarray([wav2,wav2])
    
    # Check if phase is inverted
    if phase_option == AUTO_PHASE:
        if check_if_phase_inverted(wav1, wav2, is_mono=is_mono):
            wav2 = -wav2
    elif phase_option == POSITIVE_PHASE:
        wav2 = +wav2
    elif phase_option == NEGATIVE_PHASE:
        wav2 = -wav2
    
    if is_match_silence:
        wav2 = adjust_leading_silence(wav2, wav1)
    
    wav1_length = int(librosa.get_duration(y=wav1, sr=44100))
    wav2_length = int(librosa.get_duration(y=wav2, sr=44100))
    
    if not is_mono:
        wav1 = wav1.transpose()
        wav2 = wav2.transpose()

    wav2_org = wav2.copy()
    
    command_Text("Processing files... \n")
    seconds_length = min(wav1_length, wav2_length)
    
    wav2_aligned_sources = []
    
    for sec_len in align_intro_val:
        # pick a position at 1 second in and get diff
        sec_seg = 1 if sec_len == 1 else int(seconds_length // sec_len)
        index = sr1*sec_seg  # 1 second in, assuming sr1 = sr2 = 44100

        if is_mono:
            samp1, samp2 = wav1[index : index + sr1], wav2[index : index + sr1]
            diff = get_diff(samp1, samp2)
            #print(f"Estimated difference: {diff}\n")
        else:
            index = sr1*sec_seg  # 1 second in, assuming sr1 = sr2 = 44100
            samp1, samp2 = wav1[index : index + sr1, 0], wav2[index : index + sr1, 0]
            samp1_r, samp2_r = wav1[index : index + sr1, 1], wav2[index : index + sr1, 1]
            diff, diff_r = get_diff(samp1, samp2), get_diff(samp1_r, samp2_r)
            #print(f"Estimated difference Left Channel: {diff}\nEstimated difference Right Channel: {diff_r}\n")
        
        # make aligned track 2
        if diff > 0:
            zeros_to_append = np.zeros(diff) if is_mono else np.zeros((diff, 2))
            wav2_aligned = np.append(zeros_to_append, wav2_org, axis=0)
        elif diff < 0:
            wav2_aligned = wav2_org[-diff:]
        else:
            wav2_aligned = wav2_org
            #command_Text(f"Audio files already aligned.\n")
            
        if not any(np.array_equal(wav2_aligned, source) for source in wav2_aligned_sources):
            wav2_aligned_sources.append(wav2_aligned)

    #print("Unique Sources: ", len(wav2_aligned_sources))
    
    unique_sources = len(wav2_aligned_sources)
    
    sub_mapper_big_mapper = {}

    for s in wav2_aligned_sources:
        wav2_aligned = match_mono_array_shapes(s, wav1) if is_mono else match_array_shapes(s, wav1, is_swap=True)
        
        if align_window:
            wav_sub = time_correction(wav1, wav2_aligned, seconds_length, align_window=align_window, db_analysis=db_analysis, progress_bar=progress_bar, unique_sources=unique_sources, phase_shifts=phase_shifts)
            wav_sub_size = np.abs(wav_sub).mean()  
            sub_mapper_big_mapper = {**sub_mapper_big_mapper, **{wav_sub_size:wav_sub}}
        else:
            wav2_aligned = wav2_aligned * np.power(10, db_analysis[0] / 20)
            db_range = db_analysis[1]
            
            for db_adjustment in db_range:
                # Adjust the dB of track2
                s_adjusted = wav2_aligned * (10 ** (db_adjustment / 20))
                wav_sub = wav1 - s_adjusted
                wav_sub_size = np.abs(wav_sub).mean() 
                sub_mapper_big_mapper = {**sub_mapper_big_mapper, **{wav_sub_size:wav_sub}}
            
        #print(sub_mapper_big_mapper.keys(), min(sub_mapper_big_mapper.keys()))
    
    sub_mapper_value_list = list(sub_mapper_big_mapper.values())
    
    if is_spec_match and len(sub_mapper_value_list) >= 2:
        #print("using spec ensemble with align")
        wav_sub = ensemble_for_align(list(sub_mapper_big_mapper.values()))
    else:
        #print("using linear ensemble with align")
        wav_sub = ensemble_wav(list(sub_mapper_big_mapper.values()))
         
    #print(f"Mix Mean: {np.abs(wav1).mean()}\nInst Mean: {np.abs(wav2).mean()}")
    #print('Final: ', np.abs(wav_sub).mean())
    wav_sub = np.clip(wav_sub, -1, +1)
    
    command_Text(f"Saving inverted track... ")

    if is_save_aligned or is_spec_match:
        wav1 = match_mono_array_shapes(wav1, wav_sub) if is_mono else match_array_shapes(wav1, wav_sub, is_swap=True)
        wav2_aligned = wav1 - wav_sub

        if is_spec_match:
            if wav1.ndim == 1 and wav2.ndim == 1:
                wav2_aligned = np.asfortranarray([wav2_aligned, wav2_aligned]).T
                wav1 = np.asfortranarray([wav1, wav1]).T
            
            wav2_aligned = ensemble_for_align([wav2_aligned, wav1])
            wav_sub = wav1 - wav2_aligned
        
        if is_save_aligned:
            sf.write(file2_aligned, wav2_aligned, sr1, subtype=wav_type_set)
            save_format(file2_aligned)

    sf.write(file_subtracted, wav_sub, sr1, subtype=wav_type_set)
    save_format(file_subtracted)

def phase_shift_hilbert(signal, degree):
    analytic_signal = hilbert(signal)
    return np.cos(np.radians(degree)) * analytic_signal.real - np.sin(np.radians(degree)) * analytic_signal.imag

def get_phase_shifted_tracks(track, phase_shift):
    if phase_shift == 180:
        return [track, -track]

    step = phase_shift
    end = 180 - (180 % step) if 180 % step == 0 else 181
    phase_range = range(step, end, step)
    
    flipped_list = [track, -track]
    for i in phase_range:
        flipped_list.extend([phase_shift_hilbert(track, i), phase_shift_hilbert(track, -i)])

    return flipped_list

def time_correction(mix:np.ndarray, instrumental:np.ndarray, seconds_length, align_window, db_analysis, sr=44100, progress_bar=None, unique_sources=None, phase_shifts=NONE_P):
    # Function to align two tracks using cross-correlation

    def align_tracks(track1, track2):
        # A dictionary to store each version of track2_shifted and its mean absolute value
        shifted_tracks = {}

        # Loop to adjust dB of track2
        track2 = track2 * np.power(10, db_analysis[0] / 20)
        db_range = db_analysis[1]
        
        if phase_shifts == 190:
            track2_flipped = [track2]
        else:
            track2_flipped = get_phase_shifted_tracks(track2, phase_shifts)
            
        for db_adjustment in db_range:
            for t in track2_flipped:
                # Adjust the dB of track2
                track2_adjusted = t * (10 ** (db_adjustment / 20))
                corr = correlate(track1, track2_adjusted)
                delay = np.argmax(np.abs(corr)) - (len(track1) - 1)
                track2_shifted = np.roll(track2_adjusted, shift=delay)

                # Compute the mean absolute value of track2_shifted
                track2_shifted_sub = track1 - track2_shifted
                mean_abs_value = np.abs(track2_shifted_sub).mean()

                # Store track2_shifted and its mean absolute value in the dictionary
                shifted_tracks[mean_abs_value] = track2_shifted

        # Return the version of track2_shifted with the smallest mean absolute value
                
        return shifted_tracks[min(shifted_tracks.keys())]

    # Make sure the audio files have the same shape
    
    assert mix.shape == instrumental.shape, f"Audio files must have the same shape - Mix: {mix.shape}, Inst: {instrumental.shape}"
    
    seconds_length = seconds_length // 2

    sub_mapper = {}
    
    progress_update_interval = 120
    total_iterations = 0
    
    if len(align_window) > 2:
        progress_update_interval = 320
    
    for secs in align_window:
        step = secs / 2
        window_size = int(sr * secs)
        step_size = int(sr * step)
        
        if len(mix.shape) == 1:
            total_mono = (len(range(0, len(mix) - window_size, step_size))//progress_update_interval)*unique_sources
            total_iterations += total_mono
        else:
            total_stereo_ = len(range(0, len(mix[:, 0]) - window_size, step_size))*2
            total_stereo = (total_stereo_//progress_update_interval) * unique_sources
            total_iterations += total_stereo
    
    #print(total_iterations)
    
    for secs in align_window:
        sub = np.zeros_like(mix)
        divider = np.zeros_like(mix)
        step = secs / 2
        window_size = int(sr * secs)
        step_size = int(sr * step)
        window = np.hanning(window_size)

        # For the mono case:
        if len(mix.shape) == 1:
            # The files are mono
            counter = 0
            for i in range(0, len(mix) - window_size, step_size):
                counter += 1
                if counter % progress_update_interval == 0:
                    progress_bar(total_iterations)
                window_mix = mix[i:i+window_size] * window
                window_instrumental = instrumental[i:i+window_size] * window
                window_instrumental_aligned = align_tracks(window_mix, window_instrumental)
                sub[i:i+window_size] += window_mix - window_instrumental_aligned
                divider[i:i+window_size] += window
        else:
            # The files are stereo
            counter = 0
            for ch in range(mix.shape[1]):
                for i in range(0, len(mix[:, ch]) - window_size, step_size):
                    counter += 1
                    if counter % progress_update_interval == 0:
                        progress_bar(total_iterations)
                    window_mix = mix[i:i+window_size, ch] * window
                    window_instrumental = instrumental[i:i+window_size, ch] * window
                    window_instrumental_aligned = align_tracks(window_mix, window_instrumental)
                    sub[i:i+window_size, ch] += window_mix - window_instrumental_aligned
                    divider[i:i+window_size, ch] += window

        # Normalize the result by the overlap count
        sub = np.where(divider > 1e-6, sub / divider, sub)
        sub_size = np.abs(sub).mean()
        sub_mapper = {**sub_mapper, **{sub_size: sub}}

    #print("SUB_LEN", len(list(sub_mapper.values())))

    sub = ensemble_wav(list(sub_mapper.values()), split_size=12)
          
    return sub

def ensemble_wav(waveforms, split_size=240):
    # Create a dictionary to hold the thirds of each waveform and their mean absolute values
    waveform_thirds = {i: np.array_split(waveform, split_size) for i, waveform in enumerate(waveforms)}

    # Initialize the final waveform
    final_waveform = []

    # For chunk
    for third_idx in range(split_size):
        # Compute the mean absolute value of each third from each waveform
        means = [np.abs(waveform_thirds[i][third_idx]).mean() for i in range(len(waveforms))]

        # Find the index of the waveform with the lowest mean absolute value for this third
        min_index = np.argmin(means)

        # Add the least noisy third to the final waveform
        final_waveform.append(waveform_thirds[min_index][third_idx])

    # Concatenate all the thirds to create the final waveform
    final_waveform = np.concatenate(final_waveform)

    return final_waveform

def ensemble_wav_min(waveforms):
    for i in range(1, len(waveforms)):
        if i == 1:
            wave = waveforms[0]

        ln = min(len(wave), len(waveforms[i]))
        wave = wave[:ln]
        waveforms[i] = waveforms[i][:ln]

        wave = np.where(np.abs(waveforms[i]) <= np.abs(wave), waveforms[i], wave)
        
    return wave

def align_audio_test(wav1, wav2, sr1=44100):
    def get_diff(a, b):
        corr = np.correlate(a, b, "full")
        diff = corr.argmax() - (b.shape[0] - 1)
        return diff
  
    # read tracks
    wav1 = wav1.transpose()
    wav2 = wav2.transpose()

    #print(f"Audio file shapes: {wav1.shape} / {wav2.shape}\n")
    
    wav2_org = wav2.copy()
    
    # pick a position at 1 second in and get diff
    index = sr1#*seconds_length  # 1 second in, assuming sr1 = sr2 = 44100
    samp1 = wav1[index : index + sr1, 0] # currently use left channel
    samp2 = wav2[index : index + sr1, 0]
    diff = get_diff(samp1, samp2)
    
  # make aligned track 2
    if diff > 0:
        wav2_aligned = np.append(np.zeros((diff, 1)), wav2_org, axis=0)
    elif diff < 0:
        wav2_aligned = wav2_org[-diff:]
    else:
        wav2_aligned = wav2_org
        
    return wav2_aligned

def load_audio(audio_file):
    wav, sr = librosa.load(audio_file, sr=44100, mono=False)

    if wav.ndim == 1:
        wav = np.asfortranarray([wav,wav])
        
    return wav

def rerun_mp3(audio_file):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return track_length

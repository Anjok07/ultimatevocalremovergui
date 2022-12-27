import librosa
import numpy as np
import soundfile as sf
import math
import random
import pyrubberband
import math
import platform

OPERATING_SYSTEM = platform.system()

if OPERATING_SYSTEM == 'Windows':
    wav_resolution = "sinc_fastest"
else:
    wav_resolution = "polyphase"

MAX_SPEC = 'Max Spec'
MIN_SPEC = 'Min Spec'
AVERAGE = 'Average'

def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError('h1_shape[3] must be greater than h2_shape[3]')

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
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

def wave_to_spectrogram(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * .5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * .5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    
    spec = np.asfortranarray([spec_left, spec_right])

    return spec
   
def wave_to_spectrogram_mt(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    import threading

    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * .5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * .5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])
   
    def run_thread(**kwargs):
        global spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(target=run_thread, kwargs={'y': wave_left, 'n_fft': n_fft, 'hop_length': hop_length})
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    thread.join()   
    
    spec = np.asfortranarray([spec_left, spec_right])

    return spec
    
def normalize(wave, is_normalize=False):
    """Save output music files"""
    maxv = np.abs(wave).max()
    if maxv > 1.0:
        print(f"\nNormalization Set {is_normalize}: Input above threshold for clipping. Max:{maxv}")
        if is_normalize:
            print(f"The result was normalized.")
            wave /= maxv
    else:
        print(f"\nNormalization Set {is_normalize}: Input not above threshold for clipping. Max:{maxv}")
    
    return wave
    
def normalize_two_stem(wave, mix, is_normalize=False):
    """Save output music files"""
    
    maxv = np.abs(wave).max()
    max_mix = np.abs(mix).max()
    
    if maxv > 1.0:
        print(f"\nNormalization Set {is_normalize}: Primary source above threshold for clipping. The result was normalized. Max:{maxv}")
        print(f"\nNormalization Set {is_normalize}: Mixture above threshold for clipping. The result was normalized. Max:{max_mix}")
        if is_normalize:
            wave /= maxv
            mix /= maxv
    else:
        print(f"\nNormalization Set {is_normalize}: Input not above threshold for clipping. Max:{maxv}")
    
    
    print(f"\nNormalization Set {is_normalize}: Primary source - Max:{np.abs(wave).max()}")
    print(f"\nNormalization Set {is_normalize}: Mixture - Max:{np.abs(mix).max()}")
    
    return wave, mix    

def combine_spectrograms(specs, mp):
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
    if mp.param['pre_filter_start'] > 0: # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:   
        if bands_n == 1:
            spec_c = fft_lp_filter(spec_c, mp.param['pre_filter_start'], mp.param['pre_filter_stop'])
        else:
            gp = 1        
            for b in range(mp.param['pre_filter_start'] + 1, mp.param['pre_filter_stop']):
                g = math.pow(10, -(b - mp.param['pre_filter_start']) * (3.5 - gp) / 20.0)
                gp = g
                spec_c[:, b, :] *= g
                
    return np.asfortranarray(spec_c)
    
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

    return y_mask

def mask_silence(mag, ref, thres=0.1, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError('min_range must be >= fade_area * 2')

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s:s + fade_size] += weight * ref[:, :, s:s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size:e] += weight * ref[:, :, e - fade_size:e]
            else:
                e += fade_size

            mag[:, :, s + fade_size:e - fade_size] += ref[:, :, s + fade_size:e - fade_size]
            old_e = e

    return mag
    
def align_wave_head_and_tail(a, b):
    l = min([a[0].size, b[0].size])  
    
    return a[:l,:l], b[:l,:l]
    
def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse, clamp=False):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return np.asfortranarray([np.add(wave_right / 1.25, .4 * wave_left), np.subtract(wave_left / 1.25, .4 * wave_right)])
    else:
        return np.asfortranarray([wave_left, wave_right])
    
def spectrogram_to_wave_mt(spec, hop_length, mid_side, reverse, mid_side_b2):
    import threading

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    
    def run_thread(**kwargs):
        global wave_left
        wave_left = librosa.istft(**kwargs)
        
    thread = threading.Thread(target=run_thread, kwargs={'stft_matrix': spec_left, 'hop_length': hop_length})
    thread.start()
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    thread.join()   
    
    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return np.asfortranarray([np.add(wave_right / 1.25, .4 * wave_left), np.subtract(wave_left / 1.25, .4 * wave_right)])
    else:
        return np.asfortranarray([wave_left, wave_right])
    
def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
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
                spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(spec_s, bp['hl'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse'])
            else:
                wave = np.add(wave, spectrogram_to_wave(spec_s, bp['hl'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse']))
        else:
            sr = mp.param['band'][d+1]['sr']
            if d == 1: # lower
                spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave = librosa.resample(spectrogram_to_wave(spec_s, bp['hl'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse']), bp['sr'], sr, res_type=wav_resolution)
            else: # mid
                spec_s = fft_hp_filter(spec_s, bp['hpf_start'], bp['hpf_stop'] - 1)
                spec_s = fft_lp_filter(spec_s, bp['lpf_start'], bp['lpf_stop'])
                wave2 = np.add(wave, spectrogram_to_wave(spec_s, bp['hl'], mp.param['mid_side'], mp.param['mid_side_b2'], mp.param['reverse']))
                wave = librosa.resample(wave2, bp['sr'], sr, res_type=wav_resolution)
        
    return wave

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

def mirroring(a, spec_m, input_high_end, mp):
    if 'mirroring' == a:
        mirror = np.flip(np.abs(spec_m[:, mp.param['pre_filter_start']-10-input_high_end.shape[1]:mp.param['pre_filter_start']-10, :]), 1)
        mirror = mirror * np.exp(1.j * np.angle(input_high_end))
        
        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)
        
    if 'mirroring2' == a:
        mirror = np.flip(np.abs(spec_m[:, mp.param['pre_filter_start']-10-input_high_end.shape[1]:mp.param['pre_filter_start']-10, :]), 1)
        mi = np.multiply(mirror, input_high_end * 1.7)
        
        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)

def adjust_aggr(mask, is_vocal_model, aggressiveness):
    aggr = aggressiveness.get('value', 0.0) * 4

    if aggr != 0:
        if is_vocal_model:
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

def spectrogram_to_wave_bare(spec, hop_length=1024):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    wave = np.asfortranarray([wave_left, wave_right])

    return wave

def spectrogram_to_wave_no_mp(spec, hop_length=1024):
    if spec.ndim == 2:
        wave = librosa.istft(spec, hop_length=hop_length)
    elif spec.ndim == 3:
        spec_left = np.asfortranarray(spec[0])
        spec_right = np.asfortranarray(spec[1])

        wave_left = librosa.istft(spec_left, hop_length=hop_length)
        wave_right = librosa.istft(spec_right, hop_length=hop_length)
        wave = np.asfortranarray([wave_left, wave_right])

    return wave

def wave_to_spectrogram_no_mp(wave):
    
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=2048, hop_length=1024)
    spec_right = librosa.stft(wave_right, n_fft=2048, hop_length=1024)
    spec = np.asfortranarray([spec_left, spec_right])

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

def ensembling(a, specs):   
    for i in range(1, len(specs)):
        if i == 1:
            spec = specs[0]

        ln = min([spec.shape[2], specs[i].shape[2]])
        spec = spec[:,:,:ln]
        specs[i] = specs[i][:,:,:ln]
        
        if MIN_SPEC == a:
            spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
        if MAX_SPEC == a:
            spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)  
        if AVERAGE == a:
            spec = np.where(np.abs(specs[i]) == np.abs(spec), specs[i], spec)  

    return spec

def ensemble_inputs(audio_input, algorithm, is_normalization, wav_type_set, save_path):
    
    if algorithm == AVERAGE:
        output = average_audio(audio_input)
        samplerate = 44100
    else:
        specs = []
        
        for i in range(len(audio_input)):  
            wave, samplerate = librosa.load(audio_input[i], mono=False, sr=44100)
            spec = wave_to_spectrogram_no_mp(wave)
            specs.append(spec)
        
        output = spectrogram_to_wave_no_mp(ensembling(algorithm, specs))

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

def augment_audio(export_path, audio_file, rate, is_normalization, wav_type_set, save_format=None, is_pitch=False):
    
    wav, sr = librosa.load(audio_file, sr=44100, mono=False)

    if wav.ndim == 1:
        wav = np.asfortranarray([wav,wav])

    if is_pitch:
        wav_1 = pyrubberband.pyrb.pitch_shift(wav[0], sr, rate, rbargs=None)
        wav_2 = pyrubberband.pyrb.pitch_shift(wav[1], sr, rate, rbargs=None)
    else:
        wav_1 = pyrubberband.pyrb.time_stretch(wav[0], sr, rate, rbargs=None)
        wav_2 = pyrubberband.pyrb.time_stretch(wav[1], sr, rate, rbargs=None)

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
    
def align_audio(file1, file2, file2_aligned, file_subtracted, wav_type_set, is_normalization, command_Text, progress_bar_main_var, save_format):
    def get_diff(a, b):
        corr = np.correlate(a, b, "full")
        diff = corr.argmax() - (b.shape[0] - 1)
        return diff
  
    progress_bar_main_var.set(10)
    
    # read tracks
    wav1, sr1 = librosa.load(file1, sr=44100, mono=False)
    wav2, sr2 = librosa.load(file2, sr=44100, mono=False)
    wav1 = wav1.transpose()
    wav2 = wav2.transpose()

    command_Text(f"Audio file shapes: {wav1.shape} / {wav2.shape}\n")
    
    wav2_org = wav2.copy()
    progress_bar_main_var.set(20)
    
    command_Text("Processing files... \n")
    
  # pick random position and get diff
    
    counts = {}       # counting up for each diff value
    progress = 20
    
    check_range = 64

    base = (64 / check_range)

    for i in range(check_range):
        index = int(random.uniform(44100 * 2, min(wav1.shape[0], wav2.shape[0]) - 44100 * 2))
        shift = int(random.uniform(-22050,+22050))
        samp1 = wav1[index      :index      +44100, 0]          # currently use left channel
        samp2 = wav2[index+shift:index+shift+44100, 0]
        progress += 1 * base
        progress_bar_main_var.set(progress)
        diff = get_diff(samp1, samp2)
        diff -= shift
        
    if abs(diff) < 22050:
        if not diff in counts:
            counts[diff] = 0
        counts[diff] += 1
  
  # use max counted diff value
    max_count = 0
    est_diff  = 0
    for diff in counts.keys():
        if counts[diff] > max_count:
            max_count = counts[diff]
            est_diff = diff
    
    command_Text(f"Estimated difference is {est_diff} (count: {max_count})\n")

    progress_bar_main_var.set(90)
    
    audio_files = []

    def save_aligned_audio(wav2_aligned):
        command_Text(f"Aligned File 2 with File 1.\n")
        command_Text(f"Saving files... ")
        sf.write(file2_aligned, normalize(wav2_aligned, is_normalization), sr2, subtype=wav_type_set)
        save_format(file2_aligned)
        min_len = min(wav1.shape[0], wav2_aligned.shape[0])
        wav_sub = wav1[:min_len] - wav2_aligned[:min_len]
        audio_files.append(file2_aligned)
        return min_len, wav_sub
    
  # make aligned track 2
    if est_diff > 0:
        wav2_aligned = np.append(np.zeros((est_diff, 2)), wav2_org, axis=0)
        min_len, wav_sub = save_aligned_audio(wav2_aligned)
    elif est_diff < 0:
        wav2_aligned = wav2_org[-est_diff:]
        min_len, wav_sub = save_aligned_audio(wav2_aligned)
    else:
        command_Text(f"Audio files already aligned.\n")
        command_Text(f"Saving inverted track... ")
        min_len = min(wav1.shape[0], wav2.shape[0])
        wav_sub = wav1[:min_len] - wav2[:min_len]

    wav_sub = np.clip(wav_sub, -1, +1)
  
    sf.write(file_subtracted, normalize(wav_sub, is_normalization), sr1, subtype=wav_type_set)
    save_format(file_subtracted)
  
    progress_bar_main_var.set(95)
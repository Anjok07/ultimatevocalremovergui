import os

import librosa
import numpy as np
import soundfile as sf


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


def wave_to_spectrogram(wave, hop_length, n_fft):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec


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


def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
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


def align_wave_head_and_tail(a, b, sr):
    a, _ = librosa.effects.trim(a)
    b, _ = librosa.effects.trim(b)

    a_mono = a[:, :sr * 4].sum(axis=0)
    b_mono = b[:, :sr * 4].sum(axis=0)

    a_mono -= a_mono.mean()
    b_mono -= b_mono.mean()

    offset = len(a_mono) - 1
    delay = np.argmax(np.correlate(a_mono, b_mono, 'full')) - offset

    if delay > 0:
        a = a[:, delay:]
    else:
        b = b[:, np.abs(delay):]

    if a.shape[1] < b.shape[1]:
        b = b[:, :a.shape[1]]
    else:
        a = a[:, :b.shape[1]]

    return a, b


def cache_or_load(mix_path, inst_path, sr, hop_length, n_fft):
    mix_basename = os.path.splitext(os.path.basename(mix_path))[0]
    inst_basename = os.path.splitext(os.path.basename(inst_path))[0]

    cache_dir = 'sr{}_hl{}_nf{}'.format(sr, hop_length, n_fft)
    mix_cache_dir = os.path.join(os.path.dirname(mix_path), cache_dir)
    inst_cache_dir = os.path.join(os.path.dirname(inst_path), cache_dir)
    os.makedirs(mix_cache_dir, exist_ok=True)
    os.makedirs(inst_cache_dir, exist_ok=True)

    mix_cache_path = os.path.join(mix_cache_dir, mix_basename + '.npy')
    inst_cache_path = os.path.join(inst_cache_dir, inst_basename + '.npy')

    if os.path.exists(mix_cache_path) and os.path.exists(inst_cache_path):
        X = np.load(mix_cache_path)
        y = np.load(inst_cache_path)
    else:
        X, _ = librosa.load(
            mix_path, sr, False, dtype=np.float32, res_type='kaiser_fast')
        y, _ = librosa.load(
            inst_path, sr, False, dtype=np.float32, res_type='kaiser_fast')

        X, y = align_wave_head_and_tail(X, y, sr)

        X = wave_to_spectrogram(X, hop_length, n_fft)
        y = wave_to_spectrogram(y, hop_length, n_fft)

        _, ext = os.path.splitext(mix_path)
        np.save(mix_cache_path, X)
        np.save(inst_cache_path, y)

    return X, y


def spectrogram_to_wave(spec, hop_length=1024):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    wave = np.asfortranarray([wave_left, wave_right])

    return wave


if __name__ == "__main__":
    import cv2
    import sys

    X, _ = librosa.load(
        sys.argv[1], 44100, False, dtype=np.float32, res_type='kaiser_fast')
    y, _ = librosa.load(
        sys.argv[2], 44100, False, dtype=np.float32, res_type='kaiser_fast')

    X, y = align_wave_head_and_tail(X, y, 44100)

    X_spec = wave_to_spectrogram(X, 1024, 2048)
    y_spec = wave_to_spectrogram(y, 1024, 2048)

    y_spec = reduce_vocal_aggressively(X_spec, y_spec, 0.2)
    v_spec = X_spec - y_spec

    # v_mask = np.abs(v_spec) > np.abs(y_spec)
    # y_spec = X_spec - v_spec * v_mask
    # v_spec = X_spec - y_spec

    X_mag = np.abs(X_spec)
    y_mag = np.abs(y_spec)
    v_mag = np.abs(v_spec)

    X_image = spectrogram_to_image(X_mag)
    y_image = spectrogram_to_image(y_mag)
    v_image = spectrogram_to_image(v_mag)

    cv2.imwrite('test_X.jpg', X_image)
    cv2.imwrite('test_y.jpg', y_image)
    cv2.imwrite('test_v.jpg', v_image)

    sf.write('test_X.wav', spectrogram_to_wave(X_spec).T, 44100)
    sf.write('test_y.wav', spectrogram_to_wave(y_spec).T, 44100)
    sf.write('test_v.wav', spectrogram_to_wave(v_spec).T, 44100)

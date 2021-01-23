import os

import librosa
import numpy as np
import soundfile as sf
import torch


def crop_center(h1, h2, concat=True):
    # s_freq = (h2.shape[2] - h1.shape[2]) // 2
    # e_freq = s_freq + h1.shape[2]
    h1_shape = h1.size()
    h2_shape = h2.size()
    if h2_shape[3] < h1_shape[3]:
        raise ValueError('h2_shape[3] must be greater than h1_shape[3]')
    s_time = (h2_shape[3] - h1_shape[3]) // 2
    e_time = s_time + h1_shape[3]
    h2 = h2[:, :, :, s_time:e_time]
    if concat:
        return torch.cat([h1, h2], dim=1)
    else:
        return h2


def calc_spec(X, hop_length):
    n_fft = (hop_length - 1) * 2
    audio_left = np.asfortranarray(X[0])
    audio_right = np.asfortranarray(X[1])
    spec_left = librosa.stft(audio_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(audio_right, n_fft, hop_length=hop_length)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def mask_uninformative(mask, ref, thres=0.3, min_range=64, fade_area=32):
    if min_range < fade_area * 2:
        raise ValueError('min_range must be >= fade_area * 2')
    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_area:
                s = old_e - fade_area * 2
            elif s != 0:
                start_mask = mask[:, :, s:s + fade_area]
                np.clip(
                    start_mask + np.linspace(0, 1, fade_area), 0, 1,
                    out=start_mask)
            if e != mask.shape[2]:
                end_mask = mask[:, :, e - fade_area:e]
                np.clip(
                    end_mask + np.linspace(1, 0, fade_area), 0, 1,
                    out=end_mask)
            mask[:, :, s + fade_area:e - fade_area] = 1
            old_e = e

    return mask


def align_wave_head_and_tail(a, b, sr):
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


def cache_or_load(mix_path, inst_path, sr, hop_length):
    _, mix_ext = os.path.splitext(mix_path)
    _, inst_ext = os.path.splitext(inst_path)
    spec_mix_path = mix_path.replace(mix_ext, '.npy')
    spec_inst_path = inst_path.replace(inst_ext, '.npy')

    if os.path.exists(spec_mix_path) and os.path.exists(spec_inst_path):
        X = np.load(spec_mix_path)
        y = np.load(spec_inst_path)
    else:
        X, _ = librosa.load(
            mix_path, sr, False, dtype=np.float32, res_type='kaiser_fast')
        y, _ = librosa.load(
            inst_path, sr, False, dtype=np.float32, res_type='kaiser_fast')
        X, _ = librosa.effects.trim(X)
        y, _ = librosa.effects.trim(y)
        X, y = align_wave_head_and_tail(X, y, sr)

        X = np.abs(calc_spec(X, hop_length))
        y = np.abs(calc_spec(y, hop_length))

        _, ext = os.path.splitext(mix_path)
        np.save(spec_mix_path, X)
        np.save(spec_inst_path, y)

    return X, y


def spec_to_wav(mag, phase, hop_length):
    spec = mag * phase
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wav_left = librosa.istft(spec_left, hop_length=hop_length)
    wav_right = librosa.istft(spec_right, hop_length=hop_length)
    wav = np.asfortranarray([wav_left, wav_right])

    return wav


if __name__ == "__main__":
    import sys
    X, _ = librosa.load(
        sys.argv[1], 44100, False, dtype=np.float32, res_type='kaiser_fast')
    y, _ = librosa.load(
        sys.argv[2], 44100, False, dtype=np.float32, res_type='kaiser_fast')
    X, _ = librosa.effects.trim(X)
    y, _ = librosa.effects.trim(y)
    X, y = align_wave_head_and_tail(X, y, 44100)
    sf.write('test_i.wav', y.T, 44100)
    sf.write('test_m.wav', X.T, 44100)
    sf.write('test_v.wav', (X - y).T, 44100)

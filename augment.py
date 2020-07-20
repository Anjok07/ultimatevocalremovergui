import argparse
import os
import subprocess

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib import spec_utils


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--pitch', '-p', type=int, default=-2)
    p.add_argument('--mixture_dataset', '-m', required=True)
    p.add_argument('--instrumental_dataset', '-i', required=True)
    args = p.parse_args()

    input_exts = ['.wav', '.m4a', '.3gp', '.oma', '.mp3', '.mp4']
    X_list = sorted([
        os.path.join(args.mixture_dataset, fname)
        for fname in os.listdir(args.mixture_dataset)
        if os.path.splitext(fname)[1] in input_exts])
    y_list = sorted([
        os.path.join(args.instrumental_dataset, fname)
        for fname in os.listdir(args.instrumental_dataset)
        if os.path.splitext(fname)[1] in input_exts])

    input_i = 'input_i_{}.wav'.format(args.pitch)
    input_v = 'input_v_{}.wav'.format(args.pitch)
    output_i = 'output_i_{}.wav'.format(args.pitch)
    output_v = 'output_v_{}.wav'.format(args.pitch)
    cmd_i = 'soundstretch {} {} -pitch={}'.format(input_i, output_i, args.pitch)
    cmd_v = 'soundstretch {} {} -pitch={}'.format(input_v, output_v, args.pitch)
    suffix = '_pitch{}.npy'.format(args.pitch)

    filelist = list(zip(X_list, y_list))
    for mix_path, inst_path in tqdm(filelist):
        X, _ = librosa.load(
            mix_path, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        y, _ = librosa.load(
            inst_path, args.sr, False, dtype=np.float32, res_type='kaiser_fast')

        X, _ = librosa.effects.trim(X)
        y, _ = librosa.effects.trim(y)
        X, y = spec_utils.align_wave_head_and_tail(X, y, args.sr)

        v = X - y
        sf.write(input_i, y.T, args.sr)
        sf.write(input_v, v.T, args.sr)
        subprocess.call(cmd_i, stderr=subprocess.DEVNULL)
        subprocess.call(cmd_v, stderr=subprocess.DEVNULL)

        y, _ = librosa.load(
            output_i, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        v, _ = librosa.load(
            output_v, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        X = y + v

        spec = spec_utils.calc_spec(X, args.hop_length)
        basename, _ = os.path.splitext(os.path.basename(mix_path))
        outpath = os.path.join(args.mixture_dataset, basename + suffix)
        np.save(outpath, np.abs(spec))

        spec = spec_utils.calc_spec(y, args.hop_length)
        basename, _ = os.path.splitext(os.path.basename(inst_path))
        outpath = os.path.join(args.instrumental_dataset, basename + suffix)
        np.save(outpath, np.abs(spec))

        os.remove(input_i)
        os.remove(input_v)
        os.remove(output_i)
        os.remove(output_v)

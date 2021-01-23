import os

import numpy as np
import torch
from tqdm import tqdm

from . import spec_utils


class VocalRemoverValidationSet(torch.utils.data.Dataset):

    def __init__(self, filelist):
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        path = self.filelist[idx]
        data = np.load(path)

        return data['X'], data['y']


def mixup_generator(X, y, rate, alpha):
    perm = np.random.permutation(len(X))[:int(len(X) * rate)]
    for i in range(len(perm) - 1):
        lam = np.random.beta(alpha, alpha)
        X[perm[i]] = lam * X[perm[i]] + (1 - lam) * X[perm[i + 1]]
        y[perm[i]] = lam * y[perm[i]] + (1 - lam) * y[perm[i + 1]]

    return X, y


def get_oracle_data(X, y, instance_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    idx = np.argsort(instance_loss)[::-1][:k]
    idx = np.random.choice(idx, n, replace=False)
    oracle_X = X[idx].copy()
    oracle_y = y[idx].copy()

    return oracle_X, oracle_y, idx


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def make_training_set(filelist, cropsize, patches, sr, hop_length, offset):
    len_dataset = patches * len(filelist)
    X_dataset = np.zeros(
        (len_dataset, 2, hop_length, cropsize), dtype=np.float32)
    y_dataset = np.zeros(
        (len_dataset, 2, hop_length, cropsize), dtype=np.float32)
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        p = np.random.uniform()
        if p < 0.1:
            X_path.replace(os.path.splitext(X_path)[1], '_pitch-1.wav')
            y_path.replace(os.path.splitext(y_path)[1], '_pitch-1.wav')
        elif p < 0.2:
            X_path.replace(os.path.splitext(X_path)[1], '_pitch1.wav')
            y_path.replace(os.path.splitext(y_path)[1], '_pitch1.wav')

        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length)
        coeff = np.max([X.max(), y.max()])
        X, y = X / coeff, y / coeff

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode='constant')

        starts = np.random.randint(0, X_pad.shape[2] - cropsize, patches)
        ends = starts + cropsize
        for j in range(patches):
            idx = i * patches + j
            X_dataset[idx] = X_pad[:, :, starts[j]:ends[j]]
            y_dataset[idx] = y_pad[:, :, starts[j]:ends[j]]
            if np.random.uniform() < 0.5:
                # swap channel
                X_dataset[idx] = X_dataset[idx, ::-1]
                y_dataset[idx] = y_dataset[idx, ::-1]

    return X_dataset, y_dataset


def make_validation_set(filelist, cropsize, sr, hop_length, offset):
    patch_list = []
    outdir = 'cs{}_sr{}_hl{}_of{}'.format(cropsize, sr, hop_length, offset)
    os.makedirs(outdir, exist_ok=True)
    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length)
        coeff = np.max([X.max(), y.max()])
        X, y = X / coeff, y / coeff

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(outdir, '{}_p{}.npz'.format(basename, j))
            start = j * roi_size
            if not os.path.exists(outpath):
                np.savez(
                    outpath,
                    X=X_pad[:, :, start:start + cropsize],
                    y=y_pad[:, :, start:start + cropsize])
            patch_list.append(outpath)

    return VocalRemoverValidationSet(patch_list)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    original_device = x.device
    try:
        z = th.stft(x,
                    n_fft * (1 + pad),
                    hop_length or n_fft // 4,
                    window=th.hann_window(n_fft).to(x),
                    win_length=n_fft,
                    normalized=True,
                    center=True,
                    return_complex=True,
                    pad_mode='reflect')
    except Exception:
        x = x.cpu()
        z = th.stft(x,
                    n_fft * (1 + pad),
                    hop_length or n_fft // 4,
                    window=th.hann_window(n_fft).to(x),
                    win_length=n_fft,
                    normalized=True,
                    center=True,
                    return_complex=True,
                    pad_mode='reflect')
        if original_device.type == "cuda":
            try:
                z = z.to(original_device)
            except Exception:
                pass
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    original_device = z.device
    try:
        x = th.istft(z,
                     n_fft,
                     hop_length,
                     window=th.hann_window(win_length).to(z.real),
                     win_length=win_length,
                     normalized=True,
                     length=length,
                     center=True)
    except Exception:
        z = z.cpu()
        x = th.istft(z,
                     n_fft,
                     hop_length,
                     window=th.hann_window(win_length).to(z.real),
                     win_length=win_length,
                     normalized=True,
                     length=length,
                     center=True)
        if original_device.type in ["cuda", "mps"]:
            try:
                x = x.to(original_device)
            except Exception:
                pass
    _, length = x.shape
    return x.view(*other, length)

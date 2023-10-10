# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Conveniance wrapper to perform STFT and iSTFT"""

import torch as th


def spectro(x, n_fft=512, hop_length=None, pad=0):
    """
    Perform Short-Time Fourier Transform (STFT) on the input signal 'x'.

    This function computes the STFT of the input signal using the provided parameters.

    Parameters:
        x (torch.Tensor): The input signal.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples between successive frames.
        pad (int): The amount of padding to apply to the signal.

    Returns:
        torch.Tensor: The spectrogram of the input signal.
    """
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
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
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    """
    Perform inverse Short-Time Fourier Transform (iSTFT) on the input spectrogram 'z'.

    This function computes the inverse STFT of the input spectrogram using the provided parameters.

    Parameters:
        z (torch.Tensor): The input spectrogram.
        hop_length (int): The number of samples between successive frames.
        length (int): The desired length of the output signal.
        pad (int): The amount of padding to apply to the signal.

    Returns:
        torch.Tensor: The inverse STFT of the input spectrogram.
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)

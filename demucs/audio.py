# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import subprocess as sp
from pathlib import Path

import lameenc
import julius
import numpy as np
import torch
import torchaudio as ta

from .utils import temp_filenames


def _read_info(path):
    stdout_data = sp.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
    return json.loads(stdout_data.decode('utf-8'))


class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None,
             temp_folder=None):
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.
            temp_folder (str or Path or None): temporary folder to use for decoding.


        """
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels()).t()
                if channels is not None:
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels):
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


def i16_pcm(wav):
    """Convert audio to 16 bits integer PCM format."""
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav


def f32_pcm(wav):
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    else:
        return wav.float() / (2**15 - 1)


def as_dtype_pcm(wav, dtype):
    """Convert audio to either f32 pcm or i16 pcm depending on the given dtype."""
    if wav.dtype.is_floating_point:
        return f32_pcm(wav)
    else:
        return i16_pcm(wav)


def encode_mp3(wav, path, samplerate=44100, bitrate=320, verbose=False):
    """Save given audio as mp3. This should work on all OSes."""
    C, T = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(2)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav, path, samplerate, bitrate=320, clip='rescale',
               bits_per_sample=16, as_float=False):
    """Save audio file, automatically preventing clipping if necessary
    based on the given `clip` strategy. If the path ends in `.mp3`, this
    will save as mp3 with the given `bitrate`.
    """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")

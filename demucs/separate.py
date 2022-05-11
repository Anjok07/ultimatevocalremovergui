# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

import julius
import torch as th
import torchaudio as ta

from .audio import AudioFile, convert_audio_channels
from .pretrained import is_pretrained, load_pretrained
from .utils import apply_model, load_model


def load_track(track, device, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels).to(device)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio_channels(wav, audio_channels)
            wav = wav.to(device)
            wav = julius.resample_frac(wav, sr, samplerate)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def encode_mp3(wav, path, bitrate=320, samplerate=44100, channels=2, verbose=False):
    try:
        import lameenc
    except ImportError:
        print("Failed to call lame encoder. Maybe it is not installed? "
              "On windows, run `python.exe -m pip install -U lameenc`, "
              "on OSX/Linux, run `python3 -m pip install -U lameenc`, "
              "then try again.", file=sys.stderr)
        sys.exit(1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def main():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    parser.add_argument("-n",
                        "--name",
                        default="demucs_quantized",
                        help="Model name. See README.md for the list of pretrained models. "
                             "Default is demucs_quantized.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("models"),
                        help="Path to trained models. "
                        "Also used to store downloaded pretrained models")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=0,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    parser.add_argument("--no-split",
                        action="store_false",
                        dest="split",
                        default=True,
                        help="Doesn't split audio in chunks. This can use large amounts of memory.")
    parser.add_argument("--float32",
                        action="store_true",
                        help="Convert the output wavefile to use pcm f32 format instead of s16. "
                        "This should not make a difference if you just plan on listening to the "
                        "audio but might be needed to compute exactly metrics like SDR etc.")
    parser.add_argument("--int16",
                        action="store_false",
                        dest="float32",
                        help="Opposite of --float32, here for compatibility.")
    parser.add_argument("--mp3", action="store_true",
                        help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")

    args = parser.parse_args()
    name = args.name + ".th"
    model_path = args.models / name
    if model_path.is_file():
        model = load_model(model_path)
    else:
        if is_pretrained(args.name):
            model = load_pretrained(args.name)
        else:
            print(f"No pre-trained model {args.name}", file=sys.stderr)
            sys.exit(1)
    model.to(args.device)

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        print(f"Separating track {track}")
        wav = load_track(track, args.device, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav, shifts=args.shifts, split=args.split,
                              overlap=args.overlap, progress=True)
        sources = sources * ref.std() + ref.mean()

        track_folder = out / track.name.rsplit(".", 1)[0]
        track_folder.mkdir(exist_ok=True)
        for source, name in zip(sources, model.sources):
            source = source / max(1.01 * source.abs().max(), 1)
            if args.mp3 or not args.float32:
                source = (source * 2**15).clamp_(-2**15, 2**15 - 1).short()
            source = source.cpu()
            stem = str(track_folder / name)
            if args.mp3:
                encode_mp3(source, stem + ".mp3",
                           bitrate=args.mp3_bitrate,
                           samplerate=model.samplerate,
                           channels=model.audio_channels,
                           verbose=args.verbose)
            else:
                wavname = str(track_folder / f"{name}.wav")
                ta.save(wavname, source, sample_rate=model.samplerate)


if __name__ == "__main__":
    main()

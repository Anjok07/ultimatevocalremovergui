# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("demucs", description="Train and evaluate Demucs.")
    default_raw = None
    default_musdb = None
    if 'DEMUCS_RAW' in os.environ:
        default_raw = Path(os.environ['DEMUCS_RAW'])
    if 'DEMUCS_MUSDB' in os.environ:
        default_musdb = Path(os.environ['DEMUCS_MUSDB'])
    parser.add_argument(
        "--raw",
        type=Path,
        default=default_raw,
        help="Path to raw audio, can be faster, see python3 -m demucs.raw to extract.")
    parser.add_argument("--no_raw", action="store_const", const=None, dest="raw")
    parser.add_argument("-m",
                        "--musdb",
                        type=Path,
                        default=default_musdb,
                        help="Path to musdb root")
    parser.add_argument("--is_wav", action="store_true",
                        help="Indicate that the MusDB dataset is in wav format (i.e. MusDB-HQ).")
    parser.add_argument("--metadata", type=Path, default=Path("metadata/"),
                        help="Folder where metadata information is stored.")
    parser.add_argument("--wav", type=Path,
                        help="Path to a wav dataset. This should contain a 'train' and a 'valid' "
                             "subfolder.")
    parser.add_argument("--samplerate", type=int, default=44100)
    parser.add_argument("--audio_channels", type=int, default=2)
    parser.add_argument("--samples",
                        default=44100 * 10,
                        type=int,
                        help="number of samples to feed in")
    parser.add_argument("--data_stride",
                        default=44100,
                        type=int,
                        help="Stride for chunks, shorter = longer epochs")
    parser.add_argument("-w", "--workers", default=10, type=int, help="Loader workers")
    parser.add_argument("--eval_workers", default=2, type=int, help="Final evaluation workers")
    parser.add_argument("-d",
                        "--device",
                        help="Device to train on, default is cuda if available else cpu")
    parser.add_argument("--eval_cpu", action="store_true", help="Eval on test will be run on cpu.")
    parser.add_argument("--dummy", help="Dummy parameter, useful to create a new checkpoint file")
    parser.add_argument("--test", help="Just run the test pipeline + one validation. "
                                       "This should be a filename relative to the models/ folder.")
    parser.add_argument("--test_pretrained", help="Just run the test pipeline + one validation, "
                                                  "on a pretrained model. ")

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")

    parser.add_argument("--checkpoints",
                        type=Path,
                        default=Path("checkpoints"),
                        help="Folder where to store checkpoints etc")
    parser.add_argument("--evals",
                        type=Path,
                        default=Path("evals"),
                        help="Folder where to store evals and waveforms")
    parser.add_argument("--save",
                        action="store_true",
                        help="Save estimated for the test set waveforms")
    parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs"),
                        help="Folder where to store logs")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("models"),
                        help="Folder where to store trained models")
    parser.add_argument("-R",
                        "--restart",
                        action='store_true',
                        help='Restart training, ignoring previous run')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=180, help="Number of epochs")
    parser.add_argument("-r",
                        "--repeat",
                        type=int,
                        default=2,
                        help="Repeat the train set, longer epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mse", action="store_true", help="Use MSE instead of L1")
    parser.add_argument("--init", help="Initialize from a pre-trained model.")

    # Augmentation options
    parser.add_argument("--no_augment",
                        action="store_false",
                        dest="augment",
                        default=True,
                        help="No basic data augmentation.")
    parser.add_argument("--repitch", type=float, default=0.2,
                        help="Probability to do tempo/pitch change")
    parser.add_argument("--max_tempo", type=float, default=12,
                        help="Maximum relative tempo change in %% when using repitch.")

    parser.add_argument("--remix_group_size",
                        type=int,
                        default=4,
                        help="Shuffle sources using group of this size. Useful to somewhat "
                        "replicate multi-gpu training "
                        "on less GPUs.")
    parser.add_argument("--shifts",
                        type=int,
                        default=10,
                        help="Number of random shifts used for the shift trick.")
    parser.add_argument("--overlap",
                        type=float,
                        default=0.25,
                        help="Overlap when --split_valid is passed.")

    # See model.py for doc
    parser.add_argument("--growth",
                        type=float,
                        default=2.,
                        help="Number of channels between two layers will increase by this factor")
    parser.add_argument("--depth",
                        type=int,
                        default=6,
                        help="Number of layers for the encoder and decoder")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of layers for the LSTM")
    parser.add_argument("--channels",
                        type=int,
                        default=64,
                        help="Number of channels for the first encoder layer")
    parser.add_argument("--kernel_size",
                        type=int,
                        default=8,
                        help="Kernel size for the (transposed) convolutions")
    parser.add_argument("--conv_stride",
                        type=int,
                        default=4,
                        help="Stride for the (transposed) convolutions")
    parser.add_argument("--context",
                        type=int,
                        default=3,
                        help="Context size for the decoder convolutions "
                        "before the transposed convolutions")
    parser.add_argument("--rescale",
                        type=float,
                        default=0.1,
                        help="Initial weight rescale reference")
    parser.add_argument("--no_resample", action="store_false",
                        default=True, dest="resample",
                        help="No Resampling of the input/output x2")
    parser.add_argument("--no_glu",
                        action="store_false",
                        default=True,
                        dest="glu",
                        help="Replace all GLUs by ReLUs")
    parser.add_argument("--no_rewrite",
                        action="store_false",
                        default=True,
                        dest="rewrite",
                        help="No 1x1 rewrite convolutions")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no_norm_wav", action="store_false", dest='norm_wav', default=True)

    # Tasnet options
    parser.add_argument("--tasnet", action="store_true")
    parser.add_argument("--split_valid",
                        action="store_true",
                        help="Predict chunks by chunks for valid and test. Required for tasnet")
    parser.add_argument("--X", type=int, default=8)

    # Other options
    parser.add_argument("--show",
                        action="store_true",
                        help="Show model architecture, size and exit")
    parser.add_argument("--save_model", action="store_true",
                        help="Skip traning, just save final model "
                             "for the current checkpoint value.")
    parser.add_argument("--save_state",
                        help="Skip training, just save state "
                             "for the current checkpoint value. You should "
                             "provide a model name as argument.")

    # Quantization options
    parser.add_argument("--q-min-size", type=float, default=1,
                        help="Only quantize layers over this size (in MB)")
    parser.add_argument(
        "--qat", type=int, help="If provided, use QAT training with that many bits.")

    parser.add_argument("--diffq", type=float, default=0)
    parser.add_argument(
        "--ms-target", type=float, default=162,
        help="Model size target in MB, when using DiffQ. Best model will be kept "
             "only if it is smaller than this target.")

    return parser


def get_name(parser, args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    ignore_args = set([
        "checkpoints",
        "deterministic",
        "eval",
        "evals",
        "eval_cpu",
        "eval_workers",
        "logs",
        "master",
        "rank",
        "restart",
        "save",
        "save_model",
        "save_state",
        "show",
        "workers",
        "world_size",
    ])
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in ignore_args:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}={value.name}")
            else:
                parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    return name

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from fractions import Fraction
from concurrent import futures

import musdb
from torch import distributed

from .audio import AudioFile


def get_musdb_tracks(root, *args, **kwargs):
    mus = musdb.DB(root, *args, **kwargs)
    return {track.name: track.path for track in mus}


class StemsSet:
    def __init__(self, tracks, metadata, duration=None, stride=1,
                 samplerate=44100, channels=2, streams=slice(None)):

        self.metadata = []
        for name, path in tracks.items():
            meta = dict(metadata[name])
            meta["path"] = path
            meta["name"] = name
            self.metadata.append(meta)
            if duration is not None and meta["duration"] < duration:
                raise ValueError(f"Track {name} duration is too small {meta['duration']}")
        self.metadata.sort(key=lambda x: x["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate
        self.streams = streams

    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            return meta

    def __getitem__(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            streams = AudioFile(meta["path"]).read(seek_time=index * self.stride,
                                                   duration=self.duration,
                                                   channels=self.channels,
                                                   samplerate=self.samplerate,
                                                   streams=self.streams)
            return (streams - meta["mean"]) / meta["std"]


def _get_track_metadata(path):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path)
    mix = audio.read(streams=0, channels=1, samplerate=44100)
    return {"duration": audio.duration, "std": mix.std().item(), "mean": mix.mean().item()}


def _build_metadata(tracks, workers=10):
    pendings = []
    with futures.ProcessPoolExecutor(workers) as pool:
        for name, path in tracks.items():
            pendings.append((name, pool.submit(_get_track_metadata, path)))
    return {name: p.result() for name, p in pendings}


def _build_musdb_metadata(path, musdb, workers):
    tracks = get_musdb_tracks(musdb)
    metadata = _build_metadata(tracks, workers)
    path.parent.mkdir(exist_ok=True, parents=True)
    json.dump(metadata, open(path, "w"))


def get_compressed_datasets(args, samples):
    metadata_file = args.metadata / "musdb.json"
    if not metadata_file.is_file() and args.rank == 0:
        _build_musdb_metadata(metadata_file, args.musdb, args.workers)
    if args.world_size > 1:
        distributed.barrier()
    metadata = json.load(open(metadata_file))
    duration = Fraction(samples, args.samplerate)
    stride = Fraction(args.data_stride, args.samplerate)
    train_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="train"),
                         metadata,
                         duration=duration,
                         stride=stride,
                         streams=slice(1, None),
                         samplerate=args.samplerate,
                         channels=args.audio_channels)
    valid_set = StemsSet(get_musdb_tracks(args.musdb, subsets=["train"], split="valid"),
                         metadata,
                         samplerate=args.samplerate,
                         channels=args.audio_channels)
    return train_set, valid_set

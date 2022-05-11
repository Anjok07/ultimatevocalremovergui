# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict, namedtuple
from pathlib import Path

import musdb
import numpy as np
import torch as th
import tqdm
from torch.utils.data import DataLoader

from .audio import AudioFile

ChunkInfo = namedtuple("ChunkInfo", ["file_index", "offset", "local_index"])


class Rawset:
    """
    Dataset of raw, normalized, float32 audio files
    """
    def __init__(self, path, samples=None, stride=None, channels=2, streams=None):
        self.path = Path(path)
        self.channels = channels
        self.samples = samples
        if stride is None:
            stride = samples if samples is not None else 0
        self.stride = stride
        entries = defaultdict(list)
        for root, folders, files in os.walk(self.path, followlinks=True):
            folders.sort()
            files.sort()
            for file in files:
                if file.endswith(".raw"):
                    path = Path(root) / file
                    name, stream = path.stem.rsplit('.', 1)
                    entries[(path.parent.relative_to(self.path), name)].append(int(stream))

        self._entries = list(entries.keys())

        sizes = []
        self._lengths = []
        ref_streams = sorted(entries[self._entries[0]])
        assert ref_streams == list(range(len(ref_streams)))
        if streams is None:
            self.streams = ref_streams
        else:
            self.streams = streams
        for entry in sorted(entries.keys()):
            streams = entries[entry]
            assert sorted(streams) == ref_streams
            file = self._path(*entry)
            length = file.stat().st_size // (4 * channels)
            if samples is None:
                sizes.append(1)
            else:
                if length < samples:
                    self._entries.remove(entry)
                    continue
                sizes.append((length - samples) // stride + 1)
            self._lengths.append(length)
        if not sizes:
            raise ValueError(f"Empty dataset {self.path}")
        self._cumulative_sizes = np.cumsum(sizes)
        self._sizes = sizes

    def __len__(self):
        return self._cumulative_sizes[-1]

    @property
    def total_length(self):
        return sum(self._lengths)

    def chunk_info(self, index):
        file_index = np.searchsorted(self._cumulative_sizes, index, side='right')
        if file_index == 0:
            local_index = index
        else:
            local_index = index - self._cumulative_sizes[file_index - 1]
        return ChunkInfo(offset=local_index * self.stride,
                         file_index=file_index,
                         local_index=local_index)

    def _path(self, folder, name, stream=0):
        return self.path / folder / (name + f'.{stream}.raw')

    def __getitem__(self, index):
        chunk = self.chunk_info(index)
        entry = self._entries[chunk.file_index]

        length = self.samples or self._lengths[chunk.file_index]
        streams = []
        to_read = length * self.channels * 4
        for stream_index, stream in enumerate(self.streams):
            offset = chunk.offset * 4 * self.channels
            file = open(self._path(*entry, stream=stream), 'rb')
            file.seek(offset)
            content = file.read(to_read)
            assert len(content) == to_read
            content = np.frombuffer(content, dtype=np.float32)
            content = content.copy()  # make writable
            streams.append(th.from_numpy(content).view(length, self.channels).t())
        return th.stack(streams, dim=0)

    def name(self, index):
        chunk = self.chunk_info(index)
        folder, name = self._entries[chunk.file_index]
        return folder / name


class MusDBSet:
    def __init__(self, mus, streams=slice(None), samplerate=44100, channels=2):
        self.mus = mus
        self.streams = streams
        self.samplerate = samplerate
        self.channels = channels

    def __len__(self):
        return len(self.mus.tracks)

    def __getitem__(self, index):
        track = self.mus.tracks[index]
        return (track.name, AudioFile(track.path).read(channels=self.channels,
                                                       seek_time=0,
                                                       streams=self.streams,
                                                       samplerate=self.samplerate))


def build_raw(mus, destination, normalize, workers, samplerate, channels):
    destination.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(MusDBSet(mus, channels=channels, samplerate=samplerate),
                        batch_size=1,
                        num_workers=workers,
                        collate_fn=lambda x: x[0])
    for name, streams in tqdm.tqdm(loader):
        if normalize:
            ref = streams[0].mean(dim=0)  # use mono mixture as reference
            streams = (streams - ref.mean()) / ref.std()
        for index, stream in enumerate(streams):
            open(destination / (name + f'.{index}.raw'), "wb").write(stream.t().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser('rawset')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--samplerate', type=int, default=44100)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('musdb', type=Path)
    parser.add_argument('destination', type=Path)

    args = parser.parse_args()

    build_raw(musdb.DB(root=args.musdb, subsets=["train"], split="train"),
              args.destination / "train",
              normalize=True,
              channels=args.channels,
              samplerate=args.samplerate,
              workers=args.workers)
    build_raw(musdb.DB(root=args.musdb, subsets=["train"], split="valid"),
              args.destination / "valid",
              normalize=True,
              samplerate=args.samplerate,
              channels=args.channels,
              workers=args.workers)


if __name__ == "__main__":
    main()

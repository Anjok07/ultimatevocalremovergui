# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#cython: language_level=3

import cython
from libc.stdint cimport uint64_t, int16_t
import math
import typing as tp
import numpy as np
import torch


@cython.boundscheck(False)
@cython.wraparound(False)
def _pack(int16_t[::1] indexes, int nbits=0, int block_size=32):
    if nbits == 0:
        # automatically chose bitwidth
        nbits = int(math.ceil(math.log2(1 + np.max(indexes))))
    cdef int le = len(indexes)
    cdef int storage = 64
    assert le % (storage * block_size) == 0
    cdef int lines = le // (storage * block_size)
    out = np.zeros((lines, nbits, block_size), dtype=np.uint64)
    cdef uint64_t[:, :, ::1] out_view = out
    cdef int bit_in, bit_out, index, line
    cdef int16_t x
    cdef uint64_t tmp
    for line in range(lines):
        for bit_out in range(storage):
            for bit_in in range(nbits):
                for index in range(block_size):
                    x = indexes[line * block_size * storage + bit_out * block_size + index]
                    tmp = (x >> bit_in) & 1
                    out_view[line, bit_in, index] |= tmp << bit_out
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def _unpack(uint64_t[:, :, ::1] packed):
    cdef int lines = packed.shape[0]
    cdef int nbits = packed.shape[1]
    cdef int block_size = packed.shape[2]
    cdef int storage = 64
    out = np.zeros((lines * block_size * storage), dtype=np.int16)
    cdef int16_t[::1] out_view = out
    cdef int bit_in, bit_out, index, line
    cdef int16_t x
    cdef uint64_t tmp
    for line in range(lines):
        for bit_in in range(storage):
            for bit_out in range(nbits):
                for index in range(block_size):
                    tmp = packed[line, bit_out, index]
                    x = (tmp >> bit_in) & 1
                    out_view[line * block_size * storage + bit_in * block_size + index] |= x << bit_out
    return out


def pack(x: torch.Tensor, nbits: int = 0, block_size: int = 32) -> torch.Tensor:
    assert not x.dtype.is_floating_point
    if x.numel() > 0:
        assert x.max().item() < 2 ** 15
        assert x.min().item() >= 0
    x = x.short().reshape(-1).cpu()

    ideal_size = int(math.ceil(len(x) / (64 * block_size)) * (64 * block_size))
    if ideal_size != len(x):
        x = torch.nn.functional.pad(x, (0, ideal_size - len(x)))
    out = _pack(x.numpy(), nbits, block_size)
    # We explicitely stay in PyTorch Tensor as this will be more optimally stored
    # on disk with torch.save.
    return torch.from_numpy(out.view(np.int64))


def unpack(packed: torch.Tensor, length: tp.Optional[int] = None):
    assert packed.dtype == torch.int64
    packed_np = packed.numpy().view(np.uint64)
    out = _unpack(packed_np)
    if length is not None:
        out = out[:length]
    return torch.from_numpy(out)

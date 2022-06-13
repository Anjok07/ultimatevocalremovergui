# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bit packing in pure PyTorch.
Slower than bitpack.pyx but compatible with torchscript.
"""
import math
import typing as tp
import torch
from torch.nn import functional as F


def as_rectangle(p: torch.Tensor, side: int):
    """Reshape as rectangle, using padding when necessary so that out shape is [*, side]"""
    p_flat = p.view(-1)
    ideal_length = int(math.ceil(len(p_flat) / side) * side)
    p_flat_pad = F.pad(p_flat, (0, ideal_length - len(p_flat)))
    return p_flat_pad.view(side, -1)


def _storage_size(dtype: torch.dtype):
    if dtype == torch.int64:
        return 64
    elif dtype == torch.int32:
        return 32
    elif dtype == torch.int16:
        return 16
    elif dtype == torch.uint8:
        return 8
    else:
        raise ValueError("Invalid bitpacking storage type")


def pack(indexes, nbits: int = 0, storage_dtype: torch.dtype = torch.int16):
    """You can think of indexes as a "Tensor" of bits of shape [L, nbits].
    Instead of concatenating naively as [L * nbits], we instead look at it transposed as
    [nbits, L]. For L = 16 * G, we get [nbits, G, 16] which is trivial to store
    efficiently on int16 integers.
    There will be overhead if L is far from a multiple of 16 (e.g. 1) but for large
    model layers this is acceptable. Storage type can be changed.

    `nbits` should be the number of bits on which the indexes are coded, and will
    actually be determined automatically if set to 0.
    """
    assert not indexes.dtype.is_floating_point
    if indexes.numel() > 0:
        assert indexes.max().item() < 2 ** 15
        assert indexes.min().item() >= 0
        if nbits == 0:
            nbits = int(math.ceil(math.log2(1 + (indexes.max()))))
        else:
            assert indexes.max().item() < 2 ** nbits

    indexes = indexes.reshape(-1)
    storage_size = _storage_size(storage_dtype)
    rect = as_rectangle(indexes, storage_size)
    out = torch.zeros(nbits, rect.shape[1], dtype=storage_dtype, device=indexes.device)
    for in_bit in range(nbits):
        for out_bit in range(storage_size):
            d = ((rect[out_bit] >> in_bit) & 1).to(out.dtype) << out_bit
            out[in_bit, :] |= d
    return out


def unpack(packed: torch.Tensor, length: tp.Optional[int] = None):
    """Opposite of `pack`. You might need to specify the original length."""
    storage_size = _storage_size(packed.dtype)
    nbits, groups = packed.shape
    out = torch.zeros(storage_size, groups, dtype=torch.int16, device=packed.device)
    for in_bit in range(storage_size):
        for out_bit in range(nbits):
            bit_value = (packed[out_bit, :] >> in_bit) & 1
            out[in_bit, :] = out[in_bit, :] | (bit_value.to(out) << out_bit)
    out = out.view(-1)
    if length is not None:
        out = out[:length]
    return out

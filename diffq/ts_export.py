# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""TorchScript export support.
We have to do a lot of black magic for TorchScript to be happy
because we cannot dynamically allocate new weights when loading the model.

Here is how it works:
- we generate code in a temporary python file for the given model that explicitely
    override all the weights on the first forward from their packed version.
    This is because TorchScript does not let us iterate over parameters in a generic manner.
- we zero out all the original weights. We cannot simply remove those weights
    because TorchScript won't let us recreate them.
- A TorchScript file is just a zip file, but stored without compression.
    In order to remove the cost of storing the zeroed out weights, we unzip the file,
    and zip it again with compression.
"""
import importlib
import os
from pathlib import Path
import random
import sys
import typing as tp
import tempfile
import zipfile

import torch
from torch import jit

from .diffq import DiffQuantizer
from .uniform import uniform_unquantize
from .torch_pack import unpack

_DiffQPacked = tp.Tuple[
    tp.List[tp.Optional[torch.Tensor]], tp.Tuple[float, float],
    torch.Tensor, tp.List[int]]

# This is the template for the generated class.
TEMPLATE = '''
import typing as tp
import torch
from torch import jit

from diffq.ts_export import _unpack_param, _DiffQPacked

from {module} import {klass}


class DiffQTSModel(torch.nn.Module):
    def __init__(self, model: {klass}, group_size: int, min_bits: int,
                 packed: tp.List[_DiffQPacked]):
        super().__init__()
        self.group_size = group_size
        self.min_bits = min_bits
        self.model = model
        self._unpacked = False
        self._packed = packed

    @jit.export
    def unpack(self):
        """
        Unpack the weights, automatically called on the first forward,
        or explicitely."""
        if self._unpacked:
            return
{unpack_assigns}
        self._unpacked = True

    def forward(self, x: torch.Tensor):
        self.unpack()
        return self.model.forward(x)
'''

# those are the assignments for each quantized weight.
UNPACK_ASSIGN = (' ' * 8) + ('self.model{full_name}.data[:] = '
                             '_unpack_param(self._packed[{index}], '
                             'group_size=self.group_size, min_bits=self.min_bits)')
UNPACK_ASSIGN_SAME = (' ' * 8) + 'self.model{full_name} = self.model{other_name}'


def export(quantizer: DiffQuantizer, path: tp.Union[str, Path]):
    """Export the given quantized model to the given path.
    We must save the quantized model ourselves, as we need to recompress
    the zip archive afterwards.
    """
    packed: tp.List[_DiffQPacked] = []
    uniq_name = ''.join([random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12)])
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.path.insert(0, tmpdir)
        try:
            code = _codegen(quantizer)
            with open(Path(tmpdir) / f'{uniq_name}.py', 'w') as f:
                f.write(code)
            module = importlib.import_module(uniq_name)
            ts_klass = module.DiffQTSModel
            state = quantizer.get_quantized_state(packed=True, torch_pack=True)
            quantized = state["quantized"]
            for qparam in quantizer._qparams:
                if qparam.other is None:
                    levels, scales, bits = quantized.pop(0)
                    size = qparam.param.size()
                    packed.append((levels, scales, bits, list(size)))
                    qparam.param.data.zero_()
            quantizer.detach()
            ts_premodel = ts_klass(quantizer.model, quantizer.group_size,
                                   quantizer.min_bits, packed)
            ts_model = jit.script(ts_premodel)
            if path is not None:
                jit.save(ts_model, path)
                recompress(path)
        finally:
            sys.path.pop(0)

    return ts_model


def _unpack_param(packed: _DiffQPacked, group_size: int, min_bits: int) -> torch.Tensor:
    """Function called from TorchScript on the first forward to decode the
    packed weights to FP32.
    """
    packed_all_levels, scales, packed_bits, shape = packed
    numel = 1
    for dim in shape:
        numel *= dim
    bits = unpack(packed_bits, numel // group_size) + min_bits
    levels = torch.empty(bits.numel(), group_size, dtype=torch.short)
    for idx, packed_levels in enumerate(packed_all_levels):
        bit = idx + 1
        if packed_levels is not None:
            sub_levels = levels[bits == bit]
            levels[bits == bit] = unpack(packed_levels, sub_levels.numel()).view_as(sub_levels)
    bits = bits[:, None]
    unquant = uniform_unquantize(levels, scales, bits)
    if len(shape) == 4:
        return unquant.view(shape[0], shape[1], shape[2], shape[3])
    elif len(shape) == 3:
        return unquant.view(shape[0], shape[1], shape[2])
    elif len(shape) == 2:
        return unquant.view(shape[0], shape[1])
    elif len(shape) == 1:
        return unquant.view(shape[0])
    else:
        raise RuntimeError("Invalid numbr of dim")


def recompress(path: tp.Union[str, Path]):
    """After having saved the torchscript file, this will recompress it
    to make sure all the zeroed out parameters don't actually take any space.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path) as zipin:
            zipin.extractall(tmpdir)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as zipout:
            for root, folders, files in os.walk(tmpdir):
                for file in files:
                    fp = Path(root) / file
                    name = fp.relative_to(tmpdir)
                    zipout.write(fp, name)


def _get_full_name_access(full_name):
    # When generating code, we need to handle attributes vs. indexing.
    parts = []
    for part in full_name.split("."):
        try:
            index = int(part)
        except ValueError:
            parts.append("." + part)
        else:
            parts.append(f"[{index}]")
    return "".join(parts)


def _codegen(quantizer: DiffQuantizer):
    # Generates the code for the given quantizer
    module = quantizer.model.__class__.__module__
    klass = quantizer.model.__class__.__name__
    model = quantizer.model

    assert not quantizer.float16
    names = {}
    for mod_name, mod in model.named_modules():
        names[mod] = mod_name
    unpack_assigns = []

    index = 0
    for qparam in quantizer._qparams:
        mod_name = names[qparam.module]
        if mod_name == '':
            full_name = qparam.name
        else:
            full_name = mod_name + '.' + qparam.name
        full_name = _get_full_name_access(full_name)
        if qparam.other is None:
            unpack_assigns.append(UNPACK_ASSIGN.format(full_name=full_name, index=index))
            index += 1
        else:
            other_name = names[(qparam.other.module, qparam.other.name)]
            other_name = _get_full_name_access(other_name)
            unpack_assigns.append(
                UNPACK_ASSIGN_SAME.format(full_name=full_name, other_name=other_name))

    return TEMPLATE.format(
        module=module,
        klass=klass,
        unpack_assigns='\n'.join(unpack_assigns))

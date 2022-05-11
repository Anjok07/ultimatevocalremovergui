# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from concurrent import futures
from fnmatch import fnmatch
from functools import partial
import io
import math
from multiprocessing import cpu_count
import typing as tp
import zlib

import torch


class BaseQuantizer:
    @dataclass
    class _QuantizedParam:
        name: str
        param: torch.nn.Parameter
        module: torch.nn.Module
        # If a Parameter is used multiple times, `other` can be used
        # to share state between the different Quantizers
        other: tp.Optional[tp.Any]

    def __init__(self, model: torch.nn.Module, min_size: float = 0.01, float16: bool = False,
                 exclude: tp.Optional[tp.List[str]] = [], detect_bound: bool = True):
        self.model = model
        self.min_size = min_size
        self.float16 = float16
        self.exclude = exclude
        self.detect_bound = detect_bound
        self._quantized = False
        self._pre_handle = self.model.register_forward_pre_hook(self._forward_pre_hook)
        self._post_handle = self.model.register_forward_hook(self._forward_hook)

        self._quantized_state = None
        self._qparams = []
        self._float16 = []
        self._others = []
        self._rnns = []

        self._saved = []

        self._find_params()

    def _find_params(self):
        min_params = self.min_size * 2**20 // 4
        previous = {}
        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.RNNBase):
                self._rnns.append(module)
            for name, param in list(module.named_parameters(recurse=False)):
                full_name = f"{module_name}.{name}"
                matched = False
                for pattern in self.exclude:
                    if fnmatch(full_name, pattern) or fnmatch(name, pattern):
                        matched = True
                        break

                if param.numel() <= min_params or matched:
                    if id(param) in previous:
                        continue
                    if self.detect_bound:
                        previous[id(param)] = None
                    if self.float16:
                        self._float16.append(param)
                    else:
                        self._others.append(param)
                else:
                    qparam = self._register_param(name, param, module, previous.get(id(param)))
                    if self.detect_bound:
                        previous[id(param)] = qparam
                    self._qparams.append(qparam)

    def _register_param(self, name, param, module, other):
        return self.__class__._QuantizedParam(name, param, module, other)

    def _forward_pre_hook(self, module, input):
        if self.model.training:
            self._quantized_state = None
            if self._quantized:
                self.unquantize()
            if self._pre_forward_train():
                self._fix_rnns()
        else:
            self.quantize()

    def _forward_hook(self, module, input, output):
        if self.model.training:
            if self._post_forward_train():
                self._fix_rnns(flatten=False)  # Hacky, next forward will flatten

    def quantize(self, save=True):
        """
        Immediately apply quantization to the model parameters.
        If `save` is True, save a copy of the unquantized parameters, that can be
        restored with `unquantize()`.
        """
        if self._quantized:
            return
        if save:
            self._saved = [qp.param.data.to('cpu', copy=True)
                           for qp in self._qparams if qp.other is None]
        self.restore_quantized_state(self.get_quantized_state())
        self._quantized = True
        self._fix_rnns()

    def unquantize(self):
        """
        Revert a previous call to `quantize()`.
        """
        if not self._quantized:
            raise RuntimeError("Can only be called on a quantized model.")
        if not self._saved:
            raise RuntimeError("Nothing to restore.")
        for qparam in self._qparams:
            if qparam.other is None:
                qparam.param.data[:] = self._saved.pop(0)
        assert len(self._saved) == 0
        self._quantized = False
        self._fix_rnns()

    def _pre_forward_train(self) -> bool:
        """
        Called once before each forward for continuous quantization.
        Should return  True if parameters were changed.
        """
        return False

    def _post_forward_train(self) -> bool:
        """
        Called once after each forward (to restore state for instance).
        Should return True if parameters were changed.
        """
        return False

    def _fix_rnns(self, flatten=True):
        """
        To be called after quantization happened to fix RNNs.
        """
        for rnn in self._rnns:
            rnn._flat_weights = [
                (lambda wn: getattr(rnn, wn) if hasattr(rnn, wn) else None)(wn)
                for wn in rnn._flat_weights_names]
            if flatten:
                rnn.flatten_parameters()

    def get_quantized_state(self):
        """
        Returns sufficient quantized information to rebuild the model state.

        ..Note::
            To achieve maximum compression, you should compress this with
            gzip or other, as quantized weights are not optimally coded!
        """
        if self._quantized_state is None:
            self._quantized_state = self._get_quantized_state()
        return self._quantized_state

    def _get_quantized_state(self):
        """
        Actual implementation for `get_quantized_state`.
        """
        float16_params = []
        for p in self._float16:
            q = p.data.half()
            float16_params.append(q)

        return {
            "quantized": [self._quantize_param(qparam) for qparam in self._qparams
                          if qparam.other is None],
            "float16": float16_params,
            "others": [p.data.clone() for p in self._others],
        }

    def _quantize_param(self, qparam: _QuantizedParam) -> tp.Any:
        """
        To be overriden.
        """
        raise NotImplementedError()

    def _unquantize_param(self, qparam: _QuantizedParam, quantized: tp.Any) -> torch.Tensor:
        """
        To be overriden.
        """
        raise NotImplementedError()

    def restore_quantized_state(self, state) -> None:
        """
        Restore the state of the model from the quantized state.
        """
        for p, q in zip(self._float16, state["float16"]):
            p.data[:] = q.to(p)

        for p, q in zip(self._others, state["others"]):
            p.data[:] = q

        remaining = list(state["quantized"])
        for qparam in self._qparams:
            if qparam.other is not None:
                # Only unquantize first appearance of nn.Parameter.
                continue
            quantized = remaining.pop(0)
            qparam.param.data[:] = self._unquantize_param(qparam, quantized)
        self._fix_rnns()

    def detach(self) -> None:
        """
        Detach from the model, removes hooks and anything else.
        """
        self._pre_handle.remove()
        self._post_handle.remove()

    def model_size(self) -> torch.Tensor:
        """
        Returns an estimate of the quantized model size.
        """
        total = torch.tensor(0.)
        for p in self._float16:
            total += 16 * p.numel()
        for p in self._others:
            total += 32 * p.numel()
        return total / 2**20 / 8  # bits to MegaBytes

    def true_model_size(self) -> float:
        """
        Return the true quantized model size, in MB, without extra
        compression.
        """
        return self.model_size().item()

    def compressed_model_size(self, compress_level=-1, num_workers=8) -> float:
        """
        Return the compressed quantized model size, in MB.

        Args:
            compress_level (int): compression level used with zlib,
                see `zlib.compress` for details.
            num_workers (int): will split the final big byte representation in that
                many chunks processed in parallels.
        """
        out = io.BytesIO()
        torch.save(self.get_quantized_state(), out)
        ms = _parallel_compress_len(out.getvalue(), compress_level, num_workers)
        return ms / 2 ** 20


def _compress_len(data, compress_level):
    return len(zlib.compress(data, level=compress_level))


def _parallel_compress_len(data, compress_level, num_workers):
    num_workers = min(cpu_count(), num_workers)
    chunk_size = int(math.ceil(len(data) / num_workers))
    chunks = [data[offset:offset + chunk_size] for offset in range(0, len(data), chunk_size)]
    with futures.ProcessPoolExecutor(num_workers) as pool:
        return sum(pool.map(partial(_compress_len, compress_level=compress_level), chunks))

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Classic uniform quantization over n bits.
"""
from typing import Tuple
import torch

from .base import BaseQuantizer
from .utils import simple_repr


def uniform_quantize(p: torch.Tensor, bits: torch.Tensor = torch.tensor(8.)):
    """
    Quantize the given weights over `bits` bits.

    Returns:
        - quantized levels
        - (min, max) range.

    """
    assert (bits >= 1).all() and (bits <= 15).all()
    num_levels = (2 ** bits.float()).long()
    mn = p.min().item()
    mx = p.max().item()
    p = (p - mn) / (mx - mn)  # put p in [0, 1]
    unit = 1 / (num_levels - 1)  # quantization unit
    levels = (p / unit).round()
    if (bits <= 8).all():
        levels = levels.byte()
    else:
        levels = levels.short()
    return levels, (mn, mx)


def uniform_unquantize(levels: torch.Tensor, scales: Tuple[float, float],
                       bits: torch.Tensor = torch.tensor(8.)):
    """
    Unquantize the weights from the levels and scale. Return a float32 tensor.
    """
    mn, mx = scales
    num_levels = 2 ** bits.float()
    unit = 1 / (num_levels - 1)
    levels = levels.float()
    p = levels * unit  # in [0, 1]
    return p * (mx - mn) + mn


class UniformQuantizer(BaseQuantizer):
    def __init__(self, model: torch.nn.Module, bits: float = 8., min_size: float = 0.01,
                 float16: bool = False, qat: bool = False, exclude=[], detect_bound=True):
        """
        Args:
            model (torch.nn.Module): model to quantize
            bits (float): number of bits to quantize over.
            min_size (float): minimum size in MB of a parameter to be quantized.
            float16 (bool): if a layer is smaller than min_size, should we still do float16?
            qat (bool): perform quantized aware training.
            exclude (list[str]): list of patterns used to match parameters to exclude.
                For instance `['bias']` to exclude all bias terms.
            detect_bound (bool): if True, will detect bound parameters and reuse
                the same quantized tensor for both.
        """
        self.bits = float(bits)
        self.qat = qat

        super().__init__(model, min_size, float16, exclude, detect_bound)

    def __repr__(self):
        return simple_repr(self, )

    def _pre_forward_train(self):
        if self.qat:
            for qparam in self._qparams:
                if qparam.other is not None:
                    new_param = qparam.other.module._parameters[qparam.other.name]
                else:
                    quantized = self._quantize_param(qparam)
                    qvalue = self._unquantize_param(qparam, quantized)
                    new_param = qparam.param + (qvalue - qparam.param).detach()
                qparam.module._parameters[qparam.name] = new_param
            return True
        return False

    def _post_forward_train(self):
        if self.qat:
            for qparam in self._qparams:
                qparam.module._parameters[qparam.name] = qparam.param
            return True
        return False

    def _quantize_param(self, qparam):
        levels, scales = uniform_quantize(qparam.param.data, torch.tensor(self.bits))
        return (levels, scales)

    def _unquantize_param(self, qparam, quantized):
        levels, scales = quantized
        return uniform_unquantize(levels, scales, torch.tensor(self.bits))

    def model_size(self):
        """
        Non differentiable model size in MB.
        """
        total = super().model_size()
        subtotal = 0
        for qparam in self._qparams:
            if qparam.other is None:  # if parameter is bound, count only one copy.
                subtotal += self.bits * qparam.param.numel() + 64  # 2 float for the overall scales
        subtotal /= 2**20 * 8  # bits to MegaBytes
        return total + subtotal

    def true_model_size(self):
        """
        Return the true quantized model size, in MB, without extra
        compression.
        """
        return self.model_size().item()

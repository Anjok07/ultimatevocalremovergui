# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Learnt-Stepsize quantizer from [Esser et al. 2019] https://arxiv.org/abs/1902.08153.
"""
from dataclasses import dataclass
import typing as tp

import torch

from .base import BaseQuantizer
from .utils import capture_init, simple_repr


class LSQ(BaseQuantizer):
    """Implements weight only quantization based on [Esser et al. 2019].
    https://arxiv.org/abs/1902.08153
    """
    @dataclass
    class _QuantizedParam(BaseQuantizer._QuantizedParam):
        scale: torch.nn.Parameter

    @capture_init
    def __init__(self, model: torch.nn.Module, bits: int = 8, min_size: float = 0.01,
                 float16: bool = False, suffix: str = "_lsq", exclude=[], detect_bound=True):
        assert 0 < bits <= 15
        self.suffix = suffix
        self._optimizer_setup = False
        self.bits = bits

        for name, _ in model.named_parameters():
            if name.endswith(suffix):
                raise RuntimeError("The model already has some noise scales parameters, "
                                   "maybe you used twice a LSQ on the same model?.")

        super().__init__(model, min_size, float16, exclude, detect_bound)

    def _register_param(self, name, param, module, other):
        if other is not None:
            return self.__class__._QuantizedParam(
               name=name, param=param, module=module, scale=other.scale, other=other)
        # we want the initial number of bits to be init_bits.
        scale = 2 * param.data.abs().mean() / (2 ** (self.bits - 1))**0.5
        scale = torch.nn.Parameter(scale)
        module.register_parameter(name + self.suffix, scale)
        return self.__class__._QuantizedParam(
           name=name, param=param, module=module, scale=scale, other=None)

    def clear_optimizer(self, optimizer: torch.optim.Optimizer):
        params = [qp.scale for qp in self._qparams]

        for group in optimizer.param_groups:
            new_params = []
            for q in list(group["params"]):
                matched = False
                for p in params:
                    if p is q:
                        matched = True
                if not matched:
                    new_params.append(q)
            group["params"][:] = new_params

    def setup_optimizer(self, optimizer: torch.optim.Optimizer, **kwargs):
        """
        Setup the optimizer to tune the scale parameter.
        Following [Esser et al. 2019], we use the same LR and weight decay
        as the base optimizer, unless specified otherwise.

        Args:
            optimizer (torch.Optimizer): optimizer to use.
            kwargs (dict): overrides for optimization parameters
        """
        assert not self._optimizer_setup
        self._optimizer_setup = True

        params = [qp.scale for qp in self._qparams]

        for group in optimizer.param_groups:
            for q in list(group["params"]):
                for p in params:
                    if p is q:
                        raise RuntimeError("You should create the optimizer "
                                           "before the quantizer!")

        group = {"params": params}
        group.update(kwargs)
        optimizer.add_param_group(group)

    def no_optimizer(self):
        """
        Call this if you do not want to use an optimizer.
        """
        self._optimizer_setup = True

    def model_size(self, exact=False):
        """
        Differentiable estimate of the model size.
        The size is returned in MB.

        If `exact` is True, then the output is no longer differentiable but
        reflect exactly an achievable size, even without compression,
        i.e.same as returned by `naive_model_size()`.
        """
        total = super().model_size()
        subtotal = 0
        for qparam in self._qparams:
            # only count the first appearance of a Parameter
            if qparam.other is not None:
                continue
            bits = qparam.param.numel() * self.bits
            subtotal += bits
            subtotal += 1 * 32  # param scale

        subtotal /= 2 ** 20 * 8  # bits -> MegaBytes
        return total + subtotal

    def true_model_size(self):
        """
        Naive model size without zlib compression.
        """
        return self.model_size(exact=True).item()

    def _pre_forward_train(self):
        if not self._optimizer_setup:
            raise RuntimeError("You must call `setup_optimizer()` on your optimizer "
                               "before starting training.")
        for qparam in self._qparams:
            scale = qparam.scale
            quant, _ = quantize(qparam.param, scale, self.bits)
            # We bypass the checks by PyTorch on parameters being leafs
            qparam.module._parameters[qparam.name] = quant
        return True

    def _post_forward_train(self):
        for qparam in self._qparams:
            qparam.module._parameters[qparam.name] = qparam.param
        return True

    def _quantize_param(self, qparam: _QuantizedParam) -> tp.Any:
        _, index = quantize(qparam.param, qparam.scale, self.bits)
        assert (index <= (2 ** (self.bits - 1) - 1)).all(), index.max()
        assert (index >= (-2 ** (self.bits - 1))).all(), index.min()
        return index.detach().short(), qparam.scale.detach()

    def _unquantize_param(self, qparam: _QuantizedParam, quantized: tp.Any) -> torch.Tensor:
        index, scale = quantized
        return index.float() * scale

    def _bit_pack_param(self, qparam, quantized, pack_fn):
        levels, scale = quantized
        packed = pack_fn(levels + 2 ** (self.bits - 1))
        return (packed, scale)

    def _bit_unpack_param(self, qparam, packed, unpack_fn):
        """Unpack bitpacked representation. Should be overriden
        """
        packed_levels, scale = packed
        levels = unpack_fn(
            packed_levels, qparam.param.numel()).to(qparam.param.device).view_as(qparam.param)
        levels -= 2 ** (self.bits - 1)
        return (levels, scale)

    def detach(self):
        super().detach()
        for qparam in self._qparams:
            delattr(qparam.module, qparam.name + self.suffix)

    def __repr__(self):
        return simple_repr(self)


def roundpass(x):
    return (x.round() - x).detach() + x


def gradscale(x, scale):
    return (x - x * scale).detach() + x * scale


def quantize(tensor, scale, bits):
    low = - 2 ** (bits - 1)
    high = 2 ** (bits - 1) - 1
    scale = gradscale(scale, 1 / (tensor.numel() * high)**0.5)

    index = tensor / scale
    index = index.clamp(low, high)
    index = roundpass(index)
    return index * scale, index

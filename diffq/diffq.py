# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Differentiable quantizer based on scaled noise injection.
"""
from dataclasses import dataclass
import math
import typing as tp

import torch

from .base import BaseQuantizer
from .uniform import uniform_quantize, uniform_unquantize
from .utils import simple_repr


class DiffQuantizer(BaseQuantizer):
    @dataclass
    class _QuantizedParam(BaseQuantizer._QuantizedParam):
        logit: torch.nn.Parameter

    def __init__(self, model: torch.nn.Module, min_size: float = 0.01, float16: bool = False,
                 group_size: int = 1, min_bits: float = 2, max_bits: float = 15,
                 param="bits", noise="gaussian",
                 init_bits: float = 8, extra_bits: float = 0, suffix: str = "_diffq",
                 exclude: tp.List[str] = [], detect_bound: bool = True):
        """
        Differentiable quantizer based on scaled noise injection.
        For every parameter `p` in the model, this introduces a number of bits parameter
        `b` with the same dimensions (when group_size = 1).
        Before each forward, `p` is replaced by `p + U`
        with U uniform iid noise with range [-d/2, d/2], with `d` the uniform quantization
        step for `b` bits.
        This noise approximates the quantization noise in a differentiable manner, both
        with respect to the unquantized parameter `p` and the number of bits `b`.

        At eveluation (as detected with `model.eval()`), the model is replaced
        by its true quantized version, and restored when going back to training.

        When doing actual quantization (for serialization, or evaluation),
        the number of bits is rounded to the nearest integer, and needs to be stored along.
        This will cost a few bits per dimension. To reduce this cost, one can use `group_size`,
        which will use a single noise level for multiple weight entries.

        You can use the `DiffQuantizer.model_size` method to get a differentiable estimate of the
        model size in MB. You can then use this estimate as a penalty in your training loss.

        Args:
            model (torch.nn.Module): model to quantize
            min_size (float): minimum size in MB of a parameter to be quantized.
            float16 (bool): if a layer is smaller than min_size, should we still do float16?
            group_size (int): weight entries are groupped together to reduce the number
                of noise scales to store. This should divide the size of all parameters
                bigger than min_size.
            min_bits (float): minimal number of bits.
            max_bits (float): maximal number of bits.
            init_bits (float): initial number of bits.
            extra_bits (float): extra bits to add for actual quantization (before roundoff).
            suffix (str): suffix used for the name of the extra noise scale parameters.
            exclude (list[str]): list of patterns used to match parameters to exclude.
                For instance `['bias']` to exclude all bias terms.
            detect_bound (bool): if True, will detect bound parameters and reuse
                the same quantized tensor for both, as well as the same number of bits.

        ..Warning::
            You must call `model.training()` and `model.eval()` for `DiffQuantizer` work properly.

        """
        self.group_size = group_size
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.init_bits = init_bits
        self.extra_bits = extra_bits
        self.suffix = suffix
        self.param = param
        self.noise = noise
        assert noise in ["gaussian", "uniform"]
        self._optimizer_setup = False

        self._min_noise = 1 / (2 ** self.max_bits - 1)
        self._max_noise = 1 / (2 ** self.min_bits - 1)

        assert group_size >= 0
        assert min_bits < init_bits < max_bits, \
               "init_bits must be between min_bits and max_bits excluded3"

        for name, _ in model.named_parameters():
            if name.endswith(suffix):
                raise RuntimeError("The model already has some noise scales parameters, "
                                   "maybe you used twice a DiffQuantizer on the same model?.")

        super().__init__(model, min_size, float16, exclude, detect_bound)

    def _get_bits(self, logit: torch.Tensor):
        if self.param == "noise":
            return torch.log2(1 + 1 / self._get_noise_scale(logit))
        else:
            t = torch.sigmoid(logit)
            return self.max_bits * t + (1 - t) * self.min_bits

    def _get_noise_scale(self, logit: torch.Tensor):
        if self.param == "noise":
            t = torch.sigmoid(logit)
            return torch.exp(t * math.log(self._min_noise) + (1 - t) * math.log(self._max_noise))
        else:
            return 1 / (2 ** self._get_bits(logit) - 1)

    def _register_param(self, name, param, module, other):
        if other is not None:
            return self.__class__._QuantizedParam(
               name=name, param=param, module=module, logit=other.logit, other=other)
        assert self.group_size == 0 or param.numel() % self.group_size == 0
        # we want the initial number of bits to be init_bits.
        if self.param == "noise":
            noise_scale = 1 / (2 ** self.init_bits - 1)
            t = (math.log(noise_scale) - math.log(self._max_noise)) / (
                math.log(self._min_noise) - math.log(self._max_noise))
        else:
            t = (self.init_bits - self.min_bits) / (self.max_bits - self.min_bits)
        assert 0 < t < 1
        logit = torch.logit(torch.tensor(float(t)))
        assert abs(self._get_bits(logit) - self.init_bits) < 1e-5
        if self.group_size > 0:
            nparam = param.numel() // self.group_size
        else:
            nparam = 1
        logit = torch.nn.Parameter(
            torch.full(
                (nparam,),
                logit,
                device=param.device))
        module.register_parameter(name + self.suffix, logit)
        return self.__class__._QuantizedParam(
           name=name, param=param, module=module, logit=logit, other=None)

    def clear_optimizer(self, optimizer: torch.optim.Optimizer):
        params = [qp.logit for qp in self._qparams]

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

    def setup_optimizer(self, optimizer: torch.optim.Optimizer,
                        lr: float = 1e-3, **kwargs):
        """
        Setup the optimizer to tune the number of bits. In particular, this will deactivate
        weight decay for the bits parameters.

        Args:
            optimizer (torch.Optimizer): optimizer to use.
            lr (float): specific learning rate for the bits parameters. 1e-3
                is perfect for Adam.,w
            kwargs (dict): overrides for other optimization parameters for the bits.
        """
        assert not self._optimizer_setup
        self._optimizer_setup = True

        params = [qp.logit for qp in self._qparams]

        for group in optimizer.param_groups:
            for q in list(group["params"]):
                for p in params:
                    if p is q:
                        raise RuntimeError("You should create the optimizer "
                                           "before the quantizer!")

        group = {"params": params, "lr": lr, "weight_decay": 0}
        group.update(kwargs)
        optimizer.add_param_group(group)

    def no_optimizer(self):
        """
        Call this if you do not want to use an optimizer.
        """
        self._optimizer_setup = True

    def check_unused(self):
        for qparam in self._qparams:
            if qparam.other is not None:
                continue
            grad = qparam.param.grad
            if grad is None or (grad == 0).all():
                if qparam.logit.grad is not None:
                    qparam.logit.grad.data.zero_()

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
            bits = self.extra_bits + self._get_bits(qparam.logit)
            if exact:
                bits = bits.round().clamp(1, 15)
            if self.group_size == 0:
                group_size = qparam.param.numel()
            else:
                group_size = self.group_size
            subtotal += group_size * bits.sum()
            subtotal += 2 * 32  # param scale

            # Number of bits to represent each number of bits
            bits_bits = math.ceil(math.log2(1 + (bits.max().round().item() - self.min_bits)))
            subtotal += 8  # 8 bits for bits_bits
            subtotal += bits_bits * bits.numel()

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
            if qparam.other is not None:
                noisy = qparam.other.module._parameters[qparam.other.name]
            else:
                bits = self._get_bits(qparam.logit)[:, None]
                if self.group_size == 0:
                    p_flat = qparam.param.view(-1)
                else:
                    p_flat = qparam.param.view(-1, self.group_size)
                scale = p_flat.max() - p_flat.min()
                unit = 1 / (2**bits - 1)
                if self.noise == "uniform":
                    noise_source = (torch.rand_like(p_flat) - 0.5)
                elif self.noise == "gaussian":
                    noise_source = torch.randn_like(p_flat) / 2
                noise = scale * unit * noise_source
                noisy = p_flat + noise
            # We bypass the checks by PyTorch on parameters being leafs
            qparam.module._parameters[qparam.name] = noisy.view_as(qparam.param)
        return True

    def _post_forward_train(self):
        for qparam in self._qparams:
            qparam.module._parameters[qparam.name] = qparam.param
        return True

    def _quantize_param(self, qparam: _QuantizedParam) -> tp.Any:
        bits = self.extra_bits + self._get_bits(qparam.logit)
        bits = bits.round().clamp(1, 15)[:, None].byte()
        if self.group_size == 0:
            p = qparam.param.data.view(-1)
        else:
            p = qparam.param.data.view(-1, self.group_size)
        levels, scales = uniform_quantize(p, bits)
        return levels, scales, bits

    def _unquantize_param(self, qparam: _QuantizedParam, quantized: tp.Any) -> torch.Tensor:
        levels, param_scale, bits = quantized
        return uniform_unquantize(levels, param_scale, bits).view_as(qparam.param.data)

    def detach(self):
        super().detach()
        for qparam in self._qparams:
            delattr(qparam.module, qparam.name + self.suffix)

    def __repr__(self):
        return simple_repr(self)

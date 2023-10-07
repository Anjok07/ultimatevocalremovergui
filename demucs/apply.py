# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Code to apply a model to a mix. It will handle chunking with overlaps and
inteprolation between chunks, as well as the "shift trick".
"""
from concurrent.futures import ThreadPoolExecutor
import random
import typing as tp
from multiprocessing import Process,Queue,Pipe

import torch as th
from torch import nn
from torch.nn import functional as F
import tqdm
import tkinter as tk

from .demucs import Demucs
from .hdemucs import HDemucs
from .utils import center_trim, DummyPoolExecutor

Model = tp.Union[Demucs, HDemucs]

progress_bar_num = 0

class BagOfModels(nn.Module):
    def __init__(self, models: tp.List[Model],
                 weights: tp.Optional[tp.List[tp.List[float]]] = None,
                 segment: tp.Optional[float] = None):
        """
        Represents a bag of models with specific weights.
        You should call `apply_model` rather than calling directly the forward here for
        optimal performance.

        Args:
            models (list[nn.Module]): list of Demucs/HDemucs models.
            weights (list[list[float]]): list of weights. If None, assumed to
                be all ones, otherwise it should be a list of N list (N number of models),
                each containing S floats (S number of sources).
            segment (None or float): overrides the `segment` attribute of each model
                (this is performed inplace, be careful if you reuse the models passed).
        """
        
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                other.segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1. for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")

class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out

def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)

def apply_model(model, 
                mix, 
                shifts=1, 
                split=True, 
                overlap=0.25, 
                transition_power=1., 
                static_shifts=1, 
                set_progress_bar=None, 
                device=None, 
                progress=False, 
                num_workers=0, 
                pool=None): 
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    
    global fut_length
    global bag_num
    global prog_bar
    
    if device is None:
        device = mix.device
    else:
        device = th.device(device)
    if pool is None:
        if num_workers > 0 and device.type == 'cpu':
            pool = ThreadPoolExecutor(num_workers)
        else:
            pool = DummyPoolExecutor()
            
    kwargs = {
        'shifts': shifts,
        'split': split,
        'overlap': overlap,
        'transition_power': transition_power,
        'progress': progress,
        'device': device,
        'pool': pool,
        'set_progress_bar': set_progress_bar,
        'static_shifts': static_shifts,
    }
    
    if isinstance(model, BagOfModels):
        # Special treatment for bag of model.
        # We explicitely apply multiple times `apply_model` so that the random shifts
        # are different for each model.

        estimates = 0
        totals = [0] * len(model.sources)
        bag_num = len(model.models)
        fut_length = 0
        prog_bar = 0
        current_model = 0 #(bag_num + 1)
        for sub_model, weight in zip(model.models, model.weights):
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)
            fut_length += fut_length
            current_model += 1
            out = apply_model(sub_model, mix, **kwargs)
            sub_model.to(original_model_device)
            for k, inst_weight in enumerate(weight):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out

        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates

    model.to(device)
    model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    
    if shifts:
        kwargs['shifts'] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted, **kwargs)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        return out
    elif split:
        kwargs['split'] = False
        out = th.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = th.zeros(length, device=mix.device)
        segment = int(model.samplerate * model.segment)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = float(format(stride / model.samplerate, ".2f"))
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment // 2 + 1, device=device),
                         th.arange(segment - segment // 2, 0, -1, device=device)])
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max())**transition_power
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            future = pool.submit(apply_model, model, chunk, **kwargs)
            futures.append((future, offset))
            offset += segment
        if progress:
            futures = tqdm.tqdm(futures, unit_scale=scale, ncols=120, unit='seconds')
        for future, offset in futures:
            if set_progress_bar:
                fut_length = (len(futures) * bag_num * static_shifts)
                prog_bar += 1
                set_progress_bar(0.1, (0.8/fut_length*prog_bar))
            chunk_out = future.result()
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment] += weight[:chunk_length].to(mix.device)
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    else:
        if hasattr(model, 'valid_length'):
            valid_length = model.valid_length(length)
        else:
            valid_length = length
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length).to(device)
        with th.no_grad():
            out = model(padded_mix)
        return center_trim(out, length)
    
def demucs_segments(demucs_segment, demucs_model):
    
    if demucs_segment == 'Default':
        segment = None
        if isinstance(demucs_model, BagOfModels):
            if segment is not None:
                for sub in demucs_model.models:
                    sub.segment = segment
        else:
            if segment is not None:
                sub.segment = segment
    else:
        try:
            segment = int(demucs_segment)
            if isinstance(demucs_model, BagOfModels):
                if segment is not None:
                    for sub in demucs_model.models:
                        sub.segment = segment
            else:
                if segment is not None:
                    sub.segment = segment
        except:
            segment = None
            if isinstance(demucs_model, BagOfModels):
                if segment is not None:
                    for sub in demucs_model.models:
                        sub.segment = segment
            else:
                if segment is not None:
                    sub.segment = segment
                    
    return demucs_model
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from contextlib import contextmanager
import math
import os
import tempfile
import typing as tp

import errno
import functools
import hashlib
import inspect
import io
import os
import random
import socket
import tempfile
import warnings
import zlib
import tkinter as tk

from diffq import UniformQuantizer, DiffQuantizer
import torch as th
import tqdm
from torch import distributed
from torch.nn import functional as F

import torch

def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.

    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.

    Parameters:
        tensor (torch.Tensor): The tensor to be trimmed.
        reference (Union[torch.Tensor, int]): The reference tensor or length to trim to.

    Returns:
        torch.Tensor: The trimmed tensor.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def pull_metric(history: tp.List[dict], name: str):
    """
    Retrieve a specific metric from a list of dictionaries.

    Parameters:
        history (List[dict]): The list of dictionaries containing the metrics.
        name (str): The name of the metric to retrieve.

    Returns:
        List: The list of retrieved metrics.
    """
    out = []
    for metrics in history:
        metric = metrics
        for part in name.split("."):
            metric = metric[part]
        out.append(metric)
    return out


def EMA(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


def sizeof_fmt(num: float, suffix: str = 'B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def average_metric(metric, count=1.):
    """
    Average `metric` which should be a float across all hosts. `count` should be
    the weight for this particular host (i.e. number of examples).

    Parameters:
        metric (float): The metric to be averaged.
        count (float): The weight for this particular host (default is 1.0).

    Returns:
        float: The average metric across all hosts.
    """
    metric = th.tensor([count, count * metric], dtype=th.float32, device='cuda')
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Return a port number that is most likely free.
    This could suffer from a race condition although
    it should be quite rare.

    Parameters:
        host (str): The host name or IP address (default is '').
        low (int): The lower bound of the port number range (default is 20000).
        high (int): The upper bound of the port number range (default is 40000).

    Returns:
        int: A port number that is most likely free.
    """
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        return port


def sizeof_fmt(num, suffix='B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933

    Parameters:
        num (int): The number of bytes.
        suffix (str): The suffix for the size (default is 'B').

    Returns:
        str: The human-readable size.
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def human_seconds(seconds, display='.2f'):
    """
    Given `seconds` seconds, return human readable duration.

    Parameters:
        seconds (float): Number of seconds.
        display (str): Format specifier for the output value.

    Returns:
        str: Human readable duration.
    """
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']
    last = names.pop(0)
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:
            break
        value /= ratio
        last = name
    return f"{format(value, display)} {last}"


class TensorChunk:
    """
    This class represents a chunk of a tensor.
    """
    def __init__(self, tensor, offset=0, length=None):
        """
        Initialize a new TensorChunk object.

        Parameters:
            tensor: The tensor object.
            offset (int): The starting position of the chunk.
            length (int, optional): The length of the chunk.
        """
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        """
        Get the shape of the chunk.

        Returns:
            list: The shape of the chunk.
        """
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        """
        Pad the chunk to a target length.

        Parameters:
            target_length (int): The target length of the padded chunk.

        Returns:
            tensor: The padded chunk.
        """
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
    """
    Convert a tensor or TensorChunk object to a TensorChunk object.

    Parameters:
        tensor_or_chunk: The input tensor or TensorChunk object.

    Returns:
        TensorChunk: The converted TensorChunk object.
    """
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model_v1(model, mix, shifts=None, split=False, progress=False, set_progress_bar=None):
    """
    Apply model to a given mixture.

    Args:
        model: The model to be applied.
        mix: The input mixture.
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
        set_progress_bar: A callback function to update the progress bar.

    Returns:
        The output of the model applied to the input mixture.
    """

    channels, length = mix.size()
    device = mix.device
    progress_value = 0

    if split:
        out = th.zeros(4, channels, length, device=device)
        shift = model.samplerate * 10
        offsets = range(0, length, shift)
        scale = 10
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        for offset in offsets:
            chunk = mix[..., offset:offset + shift]
            if set_progress_bar:
                progress_value += 1
                set_progress_bar(0.1, (0.8/len(offsets)*progress_value))
                chunk_out = apply_model_v1(model, chunk, shifts=shifts, set_progress_bar=set_progress_bar)
            else:
                chunk_out = apply_model_v1(model, chunk, shifts=shifts)
            out[..., offset:offset + shift] = chunk_out
            offset += shift
        return out
    elif shifts:
        max_shift = int(model.samplerate / 2)
        mix = F.pad(mix, (max_shift, max_shift))
        offsets = list(range(max_shift))
        random.shuffle(offsets)
        out = 0
        for offset in offsets[:shifts]:
            shifted = mix[..., offset:offset + length + max_shift]
            if set_progress_bar:
                shifted_out = apply_model_v1(model, shifted, set_progress_bar=set_progress_bar)
            else:
                shifted_out = apply_model_v1(model, shifted)
            out += shifted_out[..., max_shift - offset:max_shift - offset + length]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        delta = valid_length - length
        padded = F.pad(mix, (delta // 2, delta - delta // 2))
        with th.no_grad():
            out = model(padded.unsqueeze(0))[0]
        return center_trim(out, mix)


def apply_model_v2(model, mix, shifts=None, split=False,
                overlap=0.25, transition_power=1., progress=False, set_progress_bar=None): 
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
    """

    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    device = mix.device
    channels, length = mix.shape
    progress_value = 0

    if split:
        out = th.zeros(len(model.sources), channels, length, device=device)
        sum_weight = th.zeros(length, device=device)
        segment = model.segment_length
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / model.samplerate
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment // 2 + 1),
                         th.arange(segment - segment // 2, 0, -1)]).to(device)
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max())**transition_power
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            if set_progress_bar:
                progress_value += 1
                set_progress_bar(0.1, (0.8/len(offsets)*progress_value))
                chunk_out = apply_model_v2(model, chunk, shifts=shifts, set_progress_bar=set_progress_bar)
            else:
                chunk_out = apply_model_v2(model, chunk, shifts=shifts)
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset:offset + segment] += weight[:chunk_length]
            offset += segment
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    elif shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)

            if set_progress_bar:
                progress_value += 1
                shifted_out = apply_model_v2(model, shifted, set_progress_bar=set_progress_bar)
            else:
                shifted_out = apply_model_v2(model, shifted)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        with th.no_grad():
            out = model(padded_mix.unsqueeze(0))[0]
        return center_trim(out, length)


@contextmanager
def temp_filenames(count, delete=True):
    """
    Context manager that generates temporary file names.

    This context manager generates a specified number of temporary file names
    using the tempfile.NamedTemporaryFile() function. The generated file names
    are stored in a list and yielded. If the delete flag is set to True, the
    generated files will be deleted upon exiting the context.

    Parameters:
        count (int): The number of temporary file names to generate.
        delete (bool, optional): Flag to control whether to delete the generated files.
            Defaults to True.

    Yields:
        list: A list of generated temporary file names.
    """
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def get_quantizer(model, args, optimizer=None):
    """
    Get a quantizer object based on the provided arguments.

    This function returns a quantizer object based on the provided model and
    arguments. If the 'diffq' flag is set in the args, it creates a DiffQuantizer
    object with the given parameters. If the 'qat' flag is set, it creates a
    UniformQuantizer object with the specified number of bits.

    Parameters:
        model: The model to be quantized.
        args: The arguments containing flags and parameters for quantization.
        optimizer: The optimizer used for quantization. Defaults to None.

    Returns:
        quantizer: The quantizer object based on the provided arguments.
    """
    quantizer = None
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.q_min_size, group_size=8)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.q_min_size)
    return quantizer


def load_model(path, strict=False):
    """
    Load a model from a given path.

    This function loads a model from the specified path. It deserializes the
    model and its associated state using torch.load(). It then creates an instance
    of the model class using the loaded arguments and keyword arguments. If the
    'strict' flag is set to True, it raises a warning and drops any keyword
    arguments that are not found in the signature of the model class.

    Parameters:
        path (str): The path to the serialized model.
        strict (bool, optional): Flag to control strict parameter checking.
            Defaults to False.

    Returns:
        model: The loaded model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        package = th.load(load_from, 'cpu')

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]
    training_args = package["training_args"]
    quantizer = get_quantizer(model, training_args)

    set_state(model, quantizer, state)
    return model


def get_state(model, quantizer):
    """
    Get the current state of the model.

    This function retrieves the current state of the model. If the quantizer is None, it uses the 'state_dict'
    method of the model to obtain the state and converts the data to CPU. If the quantizer is not None, it
    calls the 'get_quantized_state' method of the quantizer and compresses the data using zlib.

    Parameters:
        model: The model to get the state from.
        quantizer: The quantizer used for quantization, or None if not applicable.

    Returns:
        dict: The current state of the model.
    """
    if quantizer is None:
        state = {k: p.data.to('cpu') for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        buf = io.BytesIO()
        th.save(state, buf)
        state = {'compressed': zlib.compress(buf.getvalue())}
    return state


def set_state(model, quantizer, state):
    """
    Set the state of the model.

    This function restores the state of the model. If the quantizer is None, it calls the 'load_state_dict'
    method of the model with the provided state. If the quantizer is not None, it decompresses the data from
    the 'compressed' key of the state, loads the state using 'th.load', and restores the quantized state
    using the 'restore_quantized_state' method of the quantizer.

    Parameters:
        model: The model to set the state for.
        quantizer: The quantizer used for quantization, or None if not applicable.
        state: The state dictionary to restore.

    Returns:
        dict: The restored state.
    """
    if quantizer is None:
        model.load_state_dict(state)
    else:
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = th.load(buf, "cpu")
        quantizer.restore_quantized_state(state)

    return state


def save_state(state, path):
    """
    Save the state to a file.

    This function saves the provided state to the specified path. It calculates the SHA256 hash of the
    state data, appends it to the file name, and saves the state as a binary file.

    Parameters:
        state: The state dictionary to save.
        path: The path to save the state to.
    """
    buf = io.BytesIO()
    th.save(state, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


def save_model(model, quantizer, training_args, path):
    """
    Save the model, state, and training arguments to a file.

    This function saves the provided model, its initialization arguments, the model's state, and the training
    arguments to the specified path using 'th.save'.

    Parameters:
        model: The model to save.
        quantizer: The quantizer used for quantization, or None if not applicable.
        training_args: The training arguments to save.
        path: The path to save the model to.
    """
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    state = get_state(model, quantizer)

    save_to = path
    package = {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': training_args,
    }
    th.save(package, save_to)


def capture_init(init):
    """
    Decorator to capture the initialization arguments of a class.

    This decorator captures the arguments and keyword arguments passed to the __init__ method of a class
    and stores them in the '_init_args_kwargs' attribute of the instance.

    Parameters:
        init: The __init__ method of the class to decorate.

    Returns:
        function: The decorated __init__ method.
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


class DummyPoolExecutor:
    """A dummy implementation of the concurrent.futures.Executor interface."""
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        """Initialize a new DummyPoolExecutor instance with an optional number of workers."""
        pass

    def submit(self, func, *args, **kwargs):
        """Submit a task to be executed by creating a new DummyResult instance with the provided function and arguments."""
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        """Enter a context and return self."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Exit a context."""
        return

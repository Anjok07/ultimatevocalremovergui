# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import errno
import functools
import gzip
import os
import random
import socket
import tempfile
import warnings
from contextlib import contextmanager

import torch as th
import tqdm
from torch import distributed
from torch.nn import functional as F


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def average_metric(metric, count=1.):
    """
    Average `metric` which should be a float across all hosts. `count` should be
    the weight for this particular host (i.e. number of examples).
    """
    metric = th.tensor([count, count * metric], dtype=th.float32, device='cuda')
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Return a port number that is most likely free.
    This could suffer from a race condition although
    it should be quite rare.
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
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def human_seconds(seconds, display='.2f'):
    """
    Given `seconds` seconds, return human readable duration.
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


def apply_model_v1(model, mix, shifts=None, split=False, progress=False):
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
    channels, length = mix.size()
    device = mix.device
    if split:
        out = th.zeros(4, channels, length, device=device)
        shift = model.samplerate * 10
        offsets = range(0, length, shift)
        scale = 10
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        for offset in offsets:
            chunk = mix[..., offset:offset + shift]
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
            shifted_out = apply_model_v1(model, shifted)
            out += shifted_out[..., max_shift - offset:max_shift - offset + length]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        print('valid_length: ', valid_length)
        delta = valid_length - length
        padded = F.pad(mix, (delta // 2, delta - delta // 2))
        with th.no_grad():
            out = model(padded.unsqueeze(0))[0]
        return center_trim(out, mix)


@contextmanager
def temp_filenames(count, delete=True, **kwargs):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def load_model(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        if str(path).endswith(".gz"):
            load_from = gzip.open(path, "rb")
        klass, args, kwargs, state = th.load(load_from, 'cpu')
    model = klass(*args, **kwargs)
    model.load_state_dict(state)
    return model


def save_model(model, path):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__
    state = {k: p.data.to('cpu') for k, p in model.state_dict().items()}
    save_to = path
    if str(path).endswith(".gz"):
        save_to = gzip.open(path, "wb", compresslevel=5)
    th.save((klass, args, kwargs, state), save_to)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__

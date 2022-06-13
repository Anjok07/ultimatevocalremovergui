# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Distributed training utilities.
"""
import logging
import pickle

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel

from dora import distrib as dora_distrib

logger = logging.getLogger(__name__)
rank = 0
world_size = 1


def init():
    global rank, world_size
    if not torch.distributed.is_initialized():
        dora_distrib.init()
    rank = dora_distrib.rank()
    world_size = dora_distrib.world_size()


def average(metrics, count=1.):
    if isinstance(metrics, dict):
        keys, values = zip(*sorted(metrics.items()))
        values = average(values, count)
        return dict(zip(keys, values))
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def wrap(model):
    if world_size == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            # find_unused_parameters=True,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())


def barrier():
    if world_size > 1:
        torch.distributed.barrier()


def share(obj=None, src=0):
    if world_size == 1:
        return obj
    size = torch.empty(1, device='cuda', dtype=torch.long)
    if rank == src:
        dump = pickle.dumps(obj)
        size[0] = len(dump)
    torch.distributed.broadcast(size, src=src)
    # size variable is now set to the length of pickled obj in all processes

    if rank == src:
        buffer = torch.from_numpy(np.frombuffer(dump, dtype=np.uint8).copy()).cuda()
    else:
        buffer = torch.empty(size[0].item(), device='cuda', dtype=torch.uint8)
    torch.distributed.broadcast(buffer, src=src)
    # buffer variable is now set to pickled obj in all processes

    if rank != src:
        obj = pickle.loads(buffer.cpu().numpy().tobytes())
    logger.debug(f"Shared object of size {len(buffer)}")
    return obj


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    """
    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.
    """
    if world_size == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

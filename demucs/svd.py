# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Ways to make the model stronger."""
import random
import torch


def power_iteration(m, niters=1, bs=1):
    """This is the power method. batch size is used to try multiple starting point in parallel."""
    assert m.dim() == 2
    assert m.shape[0] == m.shape[1]
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype)

    for _ in range(niters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True)
        b = n / (1e-10 + norm)

    return norm.mean()


# We need a shared RNG to make sure all the distributed worker will skip the penalty together,
# as otherwise we wouldn't get any speed up.
penalty_rng = random.Random(1234)


def svd_penalty(model, min_size=0.1, dim=1, niters=2, powm=False, convtr=True,
                proba=1, conv_only=False, exact=False, bs=1):
    """
    Penalty on the largest singular value for a layer.
    Args:
        - model: model to penalize
        - min_size: minimum size in MB of a layer to penalize.
        - dim: projection dimension for the svd_lowrank. Higher is better but slower.
        - niters: number of iterations in the algorithm used by svd_lowrank.
        - powm: use power method instead of lowrank SVD, my own experience
            is that it is both slower and less stable.
        - convtr: when True, differentiate between Conv and Transposed Conv.
            this is kept for compatibility with older experiments.
        - proba: probability to apply the penalty.
        - conv_only: only apply to conv and conv transposed, not LSTM
            (might not be reliable for other models than Demucs).
        - exact: use exact SVD (slow but useful at validation).
        - bs: batch_size for power method.
    """
    total = 0
    if penalty_rng.random() > proba:
        return 0.

    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if p.numel() / 2**18 < min_size:
                continue
            if convtr:
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    if p.dim() in [3, 4]:
                        p = p.transpose(0, 1).contiguous()
            if p.dim() == 3:
                p = p.view(len(p), -1)
            elif p.dim() == 4:
                p = p.view(len(p), -1)
            elif p.dim() == 1:
                continue
            elif conv_only:
                continue
            assert p.dim() == 2, (name, p.shape)
            if exact:
                estimate = torch.svd(p, compute_uv=False)[1].pow(2).max()
            elif powm:
                a, b = p.shape
                if a < b:
                    n = p.mm(p.t())
                else:
                    n = p.t().mm(p)
                estimate = power_iteration(n, niters, bs)
            else:
                estimate = torch.svd_lowrank(p, dim, niters)[1][0].pow(2)
            total += estimate
    return total / proba

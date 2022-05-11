# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa
"""
This package implements different quantization strategies:

- `diffq.uniform.UniformQuantizer`: classic uniform quantization over n bits.
- `diffq.diffq.DiffQuantizer`: differentiable quantizer based on scaled noise injection.

Also, do check `diffq.base.BaseQuantizer` for the common methods of all Quantizers.
"""

from .uniform import UniformQuantizer
from .diffq import DiffQuantizer

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Optional, List


def simple_repr(obj, attrs: Optional[List[str]] = None, overrides={}):
    """
    Return a simple representation string for `obj`.
    If `attrs` is not None, it should be a list of attributes to include.
    """
    params = inspect.signature(obj.__class__).parameters
    attrs_repr = []
    if attrs is None:
        attrs = params.keys()
    for attr in attrs:
        display = False
        if attr in overrides:
            value = overrides[attr]
        elif hasattr(obj, attr):
            value = getattr(obj, attr)
        else:
            continue
        if attr in params:
            param = params[attr]
            if param.default is inspect._empty or value != param.default:
                display = True
        else:
            display = True

        if display:
            attrs_repr.append(f"{attr}={value}")
    return f"{obj.__class__.__name__}({','.join(attrs_repr)})"

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
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


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    signature = inspect.signature(init)

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        actual_kwargs = dict(bound.arguments)
        del actual_kwargs['self']
        actual_kwargs.update(bound.kwargs)
        self._init_kwargs = actual_kwargs
        init(self, *args, **kwargs)

    return __init__

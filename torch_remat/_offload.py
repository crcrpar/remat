# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Synchronous CPU storage for tensors retained by saved-tensor hooks."""

from __future__ import annotations

import contextlib
from typing import cast

import torch
from torch_remat._api import saved_tensors_hooks


def _pack(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
    detached = tensor.detach()
    return tensor.device, detached.to("cpu") if tensor.is_cuda else detached


def _unpack(payload: object) -> torch.Tensor:
    device, tensor = cast(tuple[torch.device, torch.Tensor], payload)
    return tensor.to(device)


def cpu_offload() -> contextlib.AbstractContextManager[None]:
    """Offload retained CUDA tensors to CPU until replay or backward loads them.

    Use this inside a checkpoint body around selected ``recompute=False`` regions.
    Other regions keep their normal recompute behavior. Transfers are synchronous;
    callers needing overlap can supply custom :func:`saved_tensors_hooks` instead.

    CPU tensors pass through without a copy. Entering this context outside a
    checkpoint follows ordinary PyTorch saved-tensor-hook semantics.
    """
    return saved_tensors_hooks(_pack, _unpack)

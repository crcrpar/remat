# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU storage for tensors retained by saved-tensor hooks."""

from __future__ import annotations

import contextlib
from typing import cast

import torch
from torch_remat._api import saved_tensors_hooks
from torch_remat._offload_async import CpuOffloader


def _pack(tensor: torch.Tensor) -> tuple[torch.device, torch.Tensor]:
    detached = tensor.detach()
    return tensor.device, detached.to("cpu") if tensor.is_cuda else detached


def _unpack(payload: object) -> torch.Tensor:
    device, tensor = cast(tuple[torch.device, torch.Tensor], payload)
    return tensor.to(device)


def cpu_offload(
    *, offloader: CpuOffloader | None = None
) -> contextlib.AbstractContextManager[None]:
    """Offload retained CUDA tensors to CPU until replay or backward loads them.

    Use this inside a checkpoint body around selected ``recompute=False`` regions.
    Other regions keep their normal recompute behavior. Transfers are synchronous
    unless an eager :class:`CpuOffloader` is supplied for bounded asynchronous
    transfers and explicit prefetching.

    CPU tensors pass through without a copy. Entering this context outside a
    checkpoint follows ordinary PyTorch saved-tensor-hook semantics.
    """
    if torch.compiler.is_compiling():
        if offloader is not None:
            raise RuntimeError(
                "CpuOffloader is eager-only; use compiled offloading config"
            )
        from torch_remat._compile import in_compiled_checkpoint

        if not in_compiled_checkpoint():
            raise RuntimeError(
                "compiled cpu_offload must be inside a torch_remat.checkpoint body"
            )
        from torch._functorch import config

        if not getattr(config, "enable_activation_offloading", False):
            raise RuntimeError(
                "compiled cpu_offload requires PyTorch's experimental "
                "enable_activation_offloading config during compilation"
            )
        from torch.fx.traceback import annotate

        return annotate({"should_offload": True})
    if offloader is not None:
        return saved_tensors_hooks(offloader.pack, offloader.unpack)
    return saved_tensors_hooks(_pack, _unpack)

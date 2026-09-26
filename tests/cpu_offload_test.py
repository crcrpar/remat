# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc

import pytest
import torch
import torch_remat as remat


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_offload_retains_save_region_without_recompute(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            nonlocal calls
            calls += 1
            saved = x.square()
            ctx.save_for_backward(saved)
            return saved.sum()

        @staticmethod
        def backward(ctx, grad):
            (saved,) = ctx.saved_tensors
            return 2 * saved.sqrt() * grad

    def body(x):
        with remat.cpu_offload():
            return remat.region(Square.apply, "square", recompute=False)(x)

    x = torch.ones(2**20, device=device, requires_grad=True)
    y = remat.checkpoint(region_name="block")(body)(x)
    if device == "cuda":
        torch.cuda.synchronize()
        offloaded_bytes = torch.cuda.memory_allocated()
    y.backward()
    assert calls == 1
    assert torch.equal(x.grad, torch.full_like(x, 2))

    if device == "cuda":
        del y
        x.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        # A resident SAVE region keeps its internal 4 MiB tensor on CUDA.
        resident = remat.checkpoint(region_name="block")(
            lambda t: remat.region(Square.apply, "square", recompute=False)(t)
        )(x)
        torch.cuda.synchronize()
        assert (
            torch.cuda.memory_allocated() - offloaded_bytes
            >= x.numel() * x.element_size()
        )
        resident.backward()
        assert calls == 2
        assert torch.equal(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_offload_restores_saved_output_for_replay(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    def producer(t):
        nonlocal calls
        calls += 1
        return t.square()

    def body(t):
        with remat.cpu_offload():
            y = remat.region(producer, "producer", recompute=False)(t)
        remat.recompute_needs_tensor(y)
        return torch.sin(y)

    x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    remat.checkpoint(region_name="block")(body)(x).sum().backward()

    assert calls == 1
    assert torch.allclose(x.grad, 2 * x.detach() * x.detach().square().cos())

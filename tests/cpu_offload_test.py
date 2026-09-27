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


def test_compiled_cpu_offload_selects_only_annotated_region() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from torch._functorch import config

    if not hasattr(config, "enable_activation_offloading"):
        pytest.skip("compiled activation offloading is unavailable")

    def block(x: torch.Tensor, offload: bool) -> torch.Tensor:
        resident = remat.region(torch.exp, "resident", recompute=False)(x)
        if offload:
            with remat.cpu_offload():
                selected = remat.region(torch.sin, "selected", recompute=False)(x)
        else:
            selected = remat.region(torch.sin, "selected", recompute=False)(x)
        return (
            resident * remat.region(torch.cos, "recomputed", recompute=True)(selected)
        ).sum()

    def run(offload: bool) -> tuple[int, torch.Tensor]:
        torch._dynamo.reset()
        torch.manual_seed(0)
        with config.patch(enable_activation_offloading=True):
            compiled = torch.compile(
                lambda x: remat.checkpoint()(lambda t: block(t, offload))(x),
                backend="aot_eager",
                fullgraph=True,
            )
            x = torch.randn(2**18, device="cuda", requires_grad=True)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            y = compiled(x)
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated() - before
            y.backward()
            assert x.grad is not None
            return retained, x.grad

    resident_bytes, resident_grad = run(False)
    offloaded_bytes, offloaded_grad = run(True)
    assert torch.equal(offloaded_grad, resident_grad)
    assert resident_bytes - offloaded_bytes >= 2**18 * 4


def test_compiled_cpu_offload_requires_native_feature() -> None:
    from torch._functorch import config

    if not hasattr(config, "enable_activation_offloading"):
        pytest.skip("compiled activation offloading is unavailable")

    def block(x: torch.Tensor) -> torch.Tensor:
        with remat.cpu_offload():
            return remat.region(torch.sin, "saved", recompute=False)(x)

    with config.patch(enable_activation_offloading=False):
        compiled = torch.compile(
            lambda x: remat.checkpoint()(block)(x),
            backend="aot_eager",
            fullgraph=True,
        )
        with pytest.raises(Exception, match="enable_activation_offloading"):
            compiled(torch.ones(4))


def test_async_cpu_offloader_budget_and_prefetch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    size = 2**18 * 4
    offloader = remat.CpuOffloader(max_host_bytes=2 * size, max_prefetch_bytes=2 * size)
    queued: list[int] = []

    def body(t: torch.Tensor) -> torch.Tensor:
        if remat.is_recomputing():
            queued.append(offloader.prefetch())
        with remat.cpu_offload(offloader=offloader):
            saved = remat.region(torch.exp, "saved", recompute=False)(t)
        remat.recompute_needs_tensor(saved)
        return torch.sin(saved)

    x = torch.randn(2**18, device="cuda", requires_grad=True)
    y = remat.checkpoint()(body)(x)
    assert 0 < offloader.host_bytes <= 2 * size
    y.sum().backward()
    assert queued and queued[0] > 0
    assert x.grad is not None
    torch.testing.assert_close(x.grad, x.detach().exp() * x.detach().exp().cos())
    del y
    gc.collect()
    torch.cuda.synchronize()
    assert offloader.host_bytes == 0

    too_small = remat.CpuOffloader(max_host_bytes=size - 1)
    with pytest.raises(MemoryError, match="pinned bytes"):
        remat.checkpoint()(lambda t: _bounded_exp(t, too_small))(x)


def test_async_cpu_offloader_minimum_activation_size() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    threshold = 4096
    offloader = remat.CpuOffloader(
        max_host_bytes=threshold, min_offload_bytes=threshold
    )
    small = torch.ones(1023, device="cuda")
    large = torch.ones(1024, device="cuda")
    small_payload = offloader.pack(small)
    assert isinstance(small_payload, torch.Tensor) and small_payload.is_cuda
    assert offloader.host_bytes == 0
    large_payload = offloader.pack(large)
    assert not isinstance(large_payload, torch.Tensor)
    assert offloader.host_bytes == threshold
    assert torch.equal(offloader.unpack(small_payload), small)
    assert torch.equal(offloader.unpack(large_payload), large)
    with pytest.raises(ValueError, match="minimum offload"):
        remat.CpuOffloader(max_host_bytes=threshold, min_offload_bytes=-1)


def _bounded_exp(t: torch.Tensor, offloader: remat.CpuOffloader) -> torch.Tensor:
    with remat.cpu_offload(offloader=offloader):
        return remat.region(torch.exp, "saved", recompute=False)(t)


def test_async_cpu_offloader_keeps_budget_until_transfer_finishes() -> None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA stream delay is unavailable")

    x = torch.ones(2**20, device="cuda")
    size = x.numel() * x.element_size()
    offloader = remat.CpuOffloader(max_host_bytes=size)
    torch.cuda._sleep(500_000_000)
    payload = offloader.pack(x)
    restored = offloader.unpack(payload)
    del payload
    gc.collect()
    assert offloader.host_bytes == size
    with pytest.raises(MemoryError, match="pinned bytes"):
        offloader.pack(x)
    torch.cuda.synchronize()
    assert offloader.host_bytes == 0
    assert torch.equal(restored, x)

    odd = torch.ones(129, device="cuda")
    with pytest.raises(MemoryError, match="pinned bytes"):
        remat.CpuOffloader(max_host_bytes=odd.numel() * odd.element_size()).pack(odd)


def test_async_cpu_offloader_preserves_saved_tensor_layout() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    offloader = remat.CpuOffloader(max_host_bytes=2 * 16 * 8 * 4)
    for tensor in (
        torch.randn(8, 16, device="cuda").T,
        torch.randn(16, 16, device="cuda")[:, :8],
        torch.randn(16, 1, device="cuda").expand(16, 8),
    ):
        expected = tensor.detach().to("cpu").to(tensor.device)
        payload = offloader.pack(tensor)
        actual = offloader.unpack(payload)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()
        del payload


def test_async_cpu_offloader_reuses_budget_across_streams_and_shapes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    sizes = (129, 1024, 2**18)
    reserved = sum(1 << (n * 4 - 1).bit_length() for n in sizes)
    offloader = remat.CpuOffloader(max_host_bytes=reserved)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(3):
        with torch.cuda.stream(producer):
            originals = [torch.randn(n, device="cuda").sin() for n in sizes]
            payloads = [offloader.pack(t) for t in originals]
        assert offloader.host_bytes == reserved
        with torch.cuda.stream(consumer):
            restored = [offloader.unpack(payload) for payload in payloads]
        consumer.synchronize()
        for actual, expected in zip(restored, originals):
            torch.testing.assert_close(actual, expected)
        del originals, payloads, restored
        gc.collect()
        torch.cuda.synchronize()
        assert offloader.host_bytes == 0


def test_prefetched_tensor_survives_a_different_consumer_stream() -> None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA stream delay is unavailable")

    original = torch.ones(2**20, device="cuda")
    size = original.numel() * original.element_size()
    offloader = remat.CpuOffloader(max_host_bytes=size, max_prefetch_bytes=size)
    payload = offloader.pack(original)
    assert offloader.prefetch() == size
    torch.cuda.synchronize()

    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        restored = offloader.unpack(payload)
        pointer = restored.data_ptr()
        torch.cuda._sleep(500_000_000)
        result = restored.sum()
    del restored, payload
    gc.collect()
    replacement = torch.zeros_like(original)
    assert replacement.data_ptr() != pointer
    consumer.synchronize()
    assert result.item() == original.numel()


def test_host_budget_waits_for_every_reload_stream() -> None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA stream delay is unavailable")

    original = torch.ones(2**20, device="cuda")
    size = original.numel() * original.element_size()
    offloader = remat.CpuOffloader(max_host_bytes=size)
    payload = offloader.pack(original)
    torch.cuda.synchronize()

    slow, fast = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(slow):
        torch.cuda._sleep(500_000_000)
        first = offloader.unpack(payload)
    with torch.cuda.stream(fast):
        second = offloader.unpack(payload)
    fast.synchronize()
    del payload
    gc.collect()
    assert offloader.host_bytes == size
    with pytest.raises(MemoryError, match="pinned bytes"):
        offloader.pack(original)
    slow.synchronize()
    assert offloader.host_bytes == 0
    assert torch.equal(first, original)
    assert torch.equal(second, original)


def test_prefetch_budget_waits_for_abandoned_copy() -> None:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
        pytest.skip("CUDA stream delay is unavailable")

    original = torch.ones(2**20, device="cuda")
    size = original.numel() * original.element_size()
    offloader = remat.CpuOffloader(max_host_bytes=2 * size, max_prefetch_bytes=size)
    first = offloader.pack(original)
    with torch.cuda.stream(offloader._stream(original.device)):
        torch.cuda._sleep(500_000_000)
    assert offloader.prefetch() == size
    del first
    gc.collect()
    second = offloader.pack(original)
    assert offloader.prefetch() == 0
    torch.cuda.synchronize()
    assert offloader.prefetch() == size
    torch.testing.assert_close(offloader.unpack(second), original)

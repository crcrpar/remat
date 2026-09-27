# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded, stream-aware eager activation transfers."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field

import torch

from torch_remat._region import _state


@dataclass
class _Transfer:
    event: torch.cuda.Event
    reloads: list[torch.cuda.Event] = field(default_factory=list)
    prefetched_bytes: int = 0

    def complete(self) -> bool:
        return self.event.query() and all(event.query() for event in self.reloads)


@dataclass(eq=False)
class _Payload:
    host: torch.Tensor
    device: torch.device
    transfer: _Transfer
    region: object
    gpu: torch.Tensor | None = None
    ready: torch.cuda.Event | None = None


class CpuOffloader:
    """Bound pinned host storage and optionally prefetch saved CUDA tensors.

    Use ``with cpu_offload(offloader=offloader)`` around an eager SAVE region.
    ``prefetch()`` queues reloads for the current checkpoint during recompute;
    call it before the selected tensors are needed to overlap transfer with work.
    The host and prefetch limits apply across all contexts sharing this object.
    ``min_offload_bytes`` keeps smaller CUDA tensors on GPU; its default of zero
    offloads every selected CUDA tensor.
    """

    def __init__(
        self,
        *,
        max_host_bytes: int,
        max_prefetch_bytes: int = 0,
        min_offload_bytes: int = 0,
    ) -> None:
        if max_host_bytes <= 0 or max_prefetch_bytes < 0 or min_offload_bytes < 0:
            raise ValueError(
                "max_host_bytes must be positive; prefetch and minimum offload "
                "byte limits must be non-negative"
            )
        self.max_host_bytes = max_host_bytes
        self.max_prefetch_bytes = max_prefetch_bytes
        self.min_offload_bytes = min_offload_bytes
        self._host_bytes = 0
        self._prefetch_bytes = 0
        self._pending: list[tuple[int, _Transfer]] = []
        self._payloads: list[weakref.ReferenceType[_Payload]] = []
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._lock = threading.RLock()

    def _reap(self) -> None:
        remaining = []
        for size, transfer in self._pending:
            if transfer.complete():
                self._host_bytes -= size
                self._prefetch_bytes -= transfer.prefetched_bytes
            else:
                remaining.append((size, transfer))
        self._pending = remaining

    def _release(self, size: int, transfer: _Transfer) -> None:
        with self._lock:
            if transfer.complete():
                self._host_bytes -= size
                self._prefetch_bytes -= transfer.prefetched_bytes
            else:
                self._pending.append((size, transfer))

    @property
    def host_bytes(self) -> int:
        """Rounded pinned bytes reserved by live payloads or in-flight copies."""
        with self._lock:
            self._reap()
            return self._host_bytes

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        with self._lock:
            if device not in self._streams:
                self._streams[device] = torch.cuda.Stream(device=device)
            return self._streams[device]

    def pack(self, tensor: torch.Tensor) -> object:
        size = tensor.numel() * tensor.element_size()
        if not tensor.is_cuda or size < self.min_offload_bytes:
            return tensor.detach()
        # PyTorch's pinned allocator caches power-of-two blocks. Account for
        # the block it may own, rather than only the tensor's logical bytes.
        reserved = 1 << (size - 1).bit_length() if size else 0
        with self._lock:
            self._reap()
            if self._host_bytes + reserved > self.max_host_bytes:
                raise MemoryError(
                    f"CPU activation offload needs {reserved} more pinned bytes; "
                    f"{self._host_bytes} of {self.max_host_bytes} are already reserved"
                )
            self._host_bytes += reserved
        event: torch.cuda.Event | None = None
        stream: torch.cuda.Stream | None = None
        try:
            host = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=True,
                memory_format=torch.preserve_format,
            )
            stream = self._stream(tensor.device)
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            tensor.record_stream(stream)
            with torch.cuda.stream(stream):
                host.copy_(tensor.detach(), non_blocking=True)
                event = stream.record_event()
            region_state = _state.get()
            payload = _Payload(
                host,
                tensor.device,
                _Transfer(event),
                region_state.region_state if region_state is not None else None,
            )
            weakref.finalize(payload, self._release, reserved, payload.transfer)
            with self._lock:
                if len(self._payloads) >= 256 and len(self._payloads) % 256 == 0:
                    self._payloads = [
                        ref for ref in self._payloads if ref() is not None
                    ]
                self._payloads.append(weakref.ref(payload))
            return payload
        except BaseException:
            if event is None:
                if stream is not None:
                    stream.synchronize()
                with self._lock:
                    self._host_bytes -= reserved
            else:
                self._release(reserved, _Transfer(event))
            raise

    def unpack(self, packed: object) -> torch.Tensor:
        if isinstance(packed, torch.Tensor):
            return packed
        assert isinstance(packed, _Payload)
        with self._lock:
            stream = torch.cuda.current_stream(packed.device)
            if packed.gpu is not None:
                assert packed.ready is not None
                stream.wait_event(packed.ready)
                gpu = packed.gpu
                gpu.record_stream(stream)
                packed.gpu = None
                self._prefetch_bytes -= packed.transfer.prefetched_bytes
                packed.transfer.prefetched_bytes = 0
                return gpu
            stream.wait_event(packed.transfer.event)
            # PyTorch's nonblocking CPU/CUDA copy records pinned storage with its
            # host allocator, so the buffer cannot be reused before H2D completes.
            gpu = packed.host.to(packed.device, non_blocking=True)
            packed.transfer.reloads.append(stream.record_event())
            return gpu

    def prefetch(self) -> int:
        """Queue bounded reloads for the active checkpoint, newest saves first.

        Returns the newly queued byte count. Reloads wait on their offload
        events; unpack waits on each reload event without synchronizing the host.
        """
        active = _state.get()
        region = active.region_state if active is not None else None
        queued = 0
        with self._lock:
            self._reap()
            self._payloads = [ref for ref in self._payloads if ref() is not None]
            for ref in reversed(self._payloads):
                payload = ref()
                if (
                    payload is None
                    or payload.region is not region
                    or payload.gpu is not None
                ):
                    continue
                size = payload.host.numel() * payload.host.element_size()
                if self._prefetch_bytes + size > self.max_prefetch_bytes:
                    continue
                stream = self._stream(payload.device)
                current = torch.cuda.current_stream(payload.device)
                gpu = torch.empty_like(payload.host, device=payload.device)
                stream.wait_stream(current)
                stream.wait_event(payload.transfer.event)
                gpu.record_stream(stream)
                try:
                    with torch.cuda.stream(stream):
                        gpu.copy_(payload.host, non_blocking=True)
                        ready = stream.record_event()
                except BaseException:
                    stream.synchronize()
                    raise
                payload.gpu = gpu
                payload.ready = ready
                payload.transfer.reloads.append(ready)
                payload.transfer.prefetched_bytes = size
                self._prefetch_bytes += size
                queued += size
        return queued

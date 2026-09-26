# Saved-tensor hooks and offloading

`torch_remat` exposes saved-tensor hooks for transforming tensors that remain
live across forward and replay/backward. CPU offloading is the main use case,
but the interface can also support compression, logging, or custom storage.

## Built-in synchronous CPU offload

Use `cpu_offload()` inside a checkpoint body around SAVE regions whose
activations are expensive to recompute but costly to retain on CUDA:

```python
def block(x):
    with remat.cpu_offload():
        y = remat.region(expensive_op, "expensive", recompute=False)(x)
    remat.recompute_needs_tensor(y)
    return torch.relu(y)

output = remat.checkpoint(region_name="block")(block)(x)
```

Only the selected SAVE region's retained tensors move to CPU. The checkpoint
input and ordinary RECOMPUTE regions keep their existing behavior. The copy is
synchronous in each direction; use custom hooks below when transfer overlap or
prefetching is needed. Host memory must hold the offloaded tensors until their
last use.

The region policy chooses *whether* to recreate an activation. The offload
context chooses *where* retained activations live. An expensive call that would
normally recompute can therefore be changed to `recompute=False` and placed
inside `cpu_offload()` without changing the policy of other calls in the
checkpoint. The context uses PyTorch saved-tensor hooks, so it also works
outside a checkpoint for ordinary autograd saves.

| Policy for a call | After-forward storage | Work before backward uses it |
| --- | --- | --- |
| `recompute=True` | No saved intermediates from that call | Rerun the call |
| `recompute=False` | Retained tensors on their original device | Load resident tensors |
| `recompute=False` inside `cpu_offload()` | Retained CUDA tensors on CPU | Copy them back to CUDA |

Place the context *inside* the checkpoint body if checkpoint inputs should stay
on CUDA. Enclosing the checkpoint call itself also applies ordinary saved-tensor
hooks to its input anchors. A saved output needed by a later recomputed call may
be packed after the context exits; remat still remembers the offload policy
active where that output was produced.

## Choosing a policy

CPU offload is a way to reduce retained CUDA memory, not an established
training-throughput improvement over recomputation. No end-to-end training
comparison has demonstrated a speedup for this API. Simple single-region
probes favored recomputation; they do not predict a compute-heavy workload.
Synchronous offload can help throughput only when avoided recomputation costs
more than both transfers and their synchronization, or when saved CUDA memory
enables a more efficient batch size. Compare end-to-end tokens per second under
the same model, sequence length, batch size, and parallelism before changing a
training policy for speed. If offload permits a larger batch, compare the best
feasible configurations as a separate capacity-driven result.

Count all retained tensors across in-flight checkpoint regions against host
memory. The forward copy needs both the original CUDA tensor and its CPU copy
until the transfer finishes. Backward similarly allocates a CUDA copy before
the CPU payload can be released. Offload therefore lowers memory *between*
forward and backward, while instantaneous peak memory can still include
temporary copies. The built-in context does not prefetch or overlap transfers;
the custom hook interface below is the place for a workload-specific transfer
schedule.

Saved-tensor hooks are eager-only in `torch_remat`; `torch.compile` does not
support this offload context inside compiled checkpoints. Retained regions with
random or external state still need appropriate `recompute_state_hooks`, just
as they do without offloading.

## Basic hook pair

A pack hook replaces a saved tensor with an opaque payload. The matching unpack
hook recreates the tensor:

```python
from dataclasses import dataclass

import torch
import torch_remat as remat


@dataclass
class CpuTensor:
    tensor: torch.Tensor
    device: torch.device


def pack(tensor: torch.Tensor) -> CpuTensor:
    return CpuTensor(tensor=tensor.detach().to("cpu"), device=tensor.device)


def unpack(saved: CpuTensor) -> torch.Tensor:
    return saved.tensor.to(saved.device)


with remat.saved_tensors_hooks(pack, unpack):
    output = remat.checkpoint(region_name="layers.0")(block)(x)
```

This synchronous example demonstrates the contract, not a high-performance
transfer schedule. A production offloader generally uses pinned memory,
dedicated streams, events, bounded buffers, and prefetching.

The unpack hook is bound to each packed payload. It can run after the context
manager has exited.

## Which tensors are packed

Within `remat.saved_tensors_hooks`, hooks apply to:

- tensors produced internally by a SAVE region and saved for backward; and
- SAVE-region outputs persisted because later replay needs them.

An enclosing `torch.autograd.graph.saved_tensors_hooks` pair is inherited when
there is no explicit remat hook pair, so monitoring and offloading hooks still
observe tensors remat keeps.

Checkpoint inputs are different: replay always needs them, and applying a
destructive pack policy while the forward body is still reading them requires
care. To opt checkpoint inputs into the policy, pass the pair directly to
`checkpoint`:

```python
checkpointed = remat.checkpoint(
    region_name="layers.0",
    saved_tensors_hooks=(pack, unpack),
)(block)
output = checkpointed(x)
```

The checkpoint-level pack hook runs at checkpoint entry. It must not
synchronously invalidate storage that the body still reads; defer any storage
release until the forward is finished.

Hook precedence, from highest to lowest, is:

1. the pair passed with `checkpoint(saved_tensors_hooks=...)`;
2. an enclosing `remat.saved_tensors_hooks` pair;
3. an enclosing PyTorch saved-tensor hook pair inherited by remat; and
4. remat's default resident saves.

The explicit checkpoint pair also applies to checkpoint inputs. The enclosing
pairs retain their SAVE-only behavior.

## Inspecting a pack

Call `current_saved_tensor_info()` from a pack hook to learn why the tensor is
being retained:

```python
def pack(tensor):
    info = remat.current_saved_tensor_info()
    print(info.kind, info.context)
    return tensor
```

`info.kind` is a `SavedTensorKind`:

- `CHECKPOINT_INPUT`: an input retained to anchor replay;
- `BACKWARD`: an ordinary saved-for-backward tensor from a SAVE region; or
- `SAVE_OUTPUT`: a SAVE output retained for later replay.

The accessor raises outside a pack-hook invocation.

## Deferred SAVE-output packs

SAVE outputs are retained on demand. The pack can therefore run when a later
consumer claims the output, after the producing region's hook context has
exited. Remat still uses the hook pair active at the producer, but ambient
state read directly by `pack` may already have changed.

Use `capture_context` to snapshot producer-time state:

```python
current_bucket = None


def capture_context():
    return current_bucket


def pack(tensor):
    info = remat.current_saved_tensor_info()
    bucket = info.context
    return offload_to_bucket(tensor, bucket)


with remat.saved_tensors_hooks(
    pack,
    unpack_from_bucket,
    capture_context=capture_context,
):
    y = remat.region(producer, "producer", recompute=False)(x)
```

`capture_context` runs where the tensor is produced. Its opaque return value is
exposed as `SavedTensorInfo.context` whenever the corresponding pack hook runs,
including a deferred pack triggered by a downstream consumer.

## Unpack timing and order

SAVE outputs needed as replay inputs are unpacked during checkpoint replay,
before ordinary backward begins. If an offloader needs to prefetch or prepare
before any unpack, do that at the start of the checkpoint body when
`is_recomputing()` is true.

Do not assume unpack calls occur in reverse pack order. Persisted SAVE outputs
are loaded in forward order so replay can consume them, while ordinary
saved-for-backward values are loaded when their autograd nodes run.

## Lifetime strategy

A natural large-model strategy is to manage tensors at checkpoint granularity:

1. pack SAVE activations and selected checkpoint inputs after their final
   forward use;
2. prefetch the next checkpoint's payloads before its replay needs them; and
3. release host and device buffers after their final unpack.

Account for temporary copies when sizing buffers. A transfer that allocates a
new device tensor before releasing the old one can briefly increase peak
memory even when the steady-state policy saves memory.

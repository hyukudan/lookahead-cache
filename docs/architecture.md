# Architecture

## Components and status

Strix Lookahead supplies proposal-side components with a C interface. Strix Halo
is the first device target; HIP and CUDA use the same pair-operation contract.

| Component | API | Current scope |
| --- | --- | --- |
| Generic vector cache | `cache.h` | Tested host storage with LRU eviction and reader/writer leases. |
| Static pair operator | `pair.h` | Tested CPU reference, rank-256 FP16 inputs, FP32 bias cache and greedy selection. |
| Device pair operator | `device.h` | HIP and CUDA implementations compile; physical GPU execution is not yet qualified. |
| Certified candidate selector | `selection.h` | Optional tested CPU reference with immutable snapshots and exact fallback. |
| Engine adapter | Integration contract | Designed, not wired into the inference engine. |

The generic cache and the pair operator have separate storage and lifetime
mechanisms. The pair operator does not call the host LRU cache on its device path.
The engine retains model execution, verification, sampling policy, KV/recurrent
state and commit/rollback.

## Static pair computation

For a fixed ordered mapping `g(j)` from candidate row to global token ID, the
operator computes:

```text
bias[u, j] = dot(head[g(j)], embedding[u])
score[j]   = base_logits[j] + bias[u, j]
```

With the selected-head layout, `head[j]` already follows that ordered mapping.
Both layouts use rank 256 and IEEE binary16 bit patterns. The predecessor `u` is
always a global token ID. The cache stores complete FP32 bias vectors; contextual
base logits are never cached under that key.

Each of 32 logical lanes sums eight consecutive products in order. XOR reduction
then uses distances 16, 8, 4, 2 and 1. Products and additions round separately to
FP32. Greedy selection prefers the lower global token ID on equal scores and
ignores NaNs. The CPU reference requires round-to-nearest and gradual underflow.
Compiler settings preserve the declared arithmetic.

The pair cache uses a deterministic direct map from predecessor to slot. A
collision replaces that slot's vector. Capacity zero executes the same operation
without caching and provides a comparison or rollback configuration. Payload
storage is `capacity * vocabulary_size * sizeof(float)`; payload budgets exclude
metadata and allocator overhead.

## Device execution

A step enqueues exactly two kernels on its instance's bound stream:

1. The pass reads the predecessor and cache tag. A miss computes and stores the
   complete bias while forming partial winners. A hit reads consecutive cached
   rows with consecutive lanes. Both paths add the current base logits and supply
   the predecessor embedding when requested for confidence calculation.
2. The finish reduces partial winners, publishes the cache tag after the pass has
   completed, and writes the result status and selection.

Capacity zero retains the two-kernel execution shape. The enqueue path performs
no allocation, host synchronization or device-to-host copy. Creation validates
device token IDs through a one-time transfer. Reset and destruction may wait for
the bound stream.

One instance owns its workspace and belongs to one device and stream. External
serialization and stream ordering protect publication and reuse. Captured graphs
must execute on that stream and must not outlive the instance or borrowed buffers.
Strix Halo's shared memory still requires these ordering and lifetime rules.

## Identity and ownership

The caller supplies opaque 32-byte identities for weights, numerical profile and
ordered vocabulary. The library does not verify those identities against model
files. Pair-model arrays and IDs are borrowed and immutable for the instance's
lifetime. Changing weights, order, dimensions or mapping requires a new pair
instance; reset only invalidates entries. Device pointers and host pointers are
distinct API requirements.

The generic host cache instead accepts vectors computed by its caller. Readers
pin committed slots through individual leases; writers reserve unpublished slots
and commit or abort explicitly. Reset and destruction refuse while leases remain.
This contract supports reference integrations and lifetime tests independently of
the fixed pair operator.

## Optional certified selection

The CPU selector owns copies of the vocabulary, base logits, candidate indices and
bias values. It derives bounds from all excluded rows. A successful strict
certificate proves the same winner as full selection; an inconclusive certificate
requires the full fallback. This API rejects nonfinite inputs and overflow, unlike
the pair operator's explicit infinity/NaN selection semantics.

Snapshot preparation scans the full vocabulary and allocates storage. The
certificate's `O(K)` lookup is useful only when those preparation costs are reused
or integrated into an existing producer. It is not connected to the pair kernels
and does not currently run on a GPU.

See [validation](validation.md) for recorded evidence. Host component timings do
not establish GPU performance, full-model acceleration or acceptance gains.

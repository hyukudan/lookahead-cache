# Integrate the optimization library

LookaheadCache provides reusable computation and storage components. The caller
owns model loading, tensor layout conversion, contextual logits, draft policy,
target verification, attention/KV state, rollback and sampling. Integration is
specific to the application's numerical and execution contracts.

## External reference adapter

An optional HIP engine integration is tracked separately in
[engine PR #54](https://github.com/hyukudan/strix-halo-engine-lab/pull/54).
It links an installed LookaheadCache package and keeps the engine source and
model assets in that engine's project. The
[adapter guide at the reviewed revision](https://github.com/hyukudan/strix-halo-engine-lab/blob/e3dd04342d6101ce03075218757c3d4c8f1150e7/docs/models/deepseek-v4.1/hip-lookahead.md)
describes the build opt-in, original/uncached/cached runtime modes, memory
admission, native checks and qualification commands.

That proposal was compiled against library revision
`d44244e098c72fb22a09c06c05a54de023037d73` (0.4.1). Its host checks and HIP build
do not establish native GPU or complete-model correctness/performance. Use it as
a concrete integration reference, then qualify the intended hardware and workload.
Records for 0.4.1 and earlier use the former name StrixLookahead.

## CMake integration

For an installed package:

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyApplication LANGUAGES C CXX)
find_package(LookaheadCache 0.4.2 CONFIG REQUIRED COMPONENTS cache)
add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE LookaheadCache::cache)
```

Select the `hip` or `cuda` package component and link `LookaheadCache::hip` or
`LookaheadCache::cuda` when calling that device API. Those targets transitively
include the host target and corresponding runtime dependencies. They exist only
if that backend was built. A C/C++ consumer of the C ABI need not compile its own
HIP/CUDA translation unit unless the application itself defines device code.

For a source checkout placed at `third_party/lookahead-cache`:

```cmake
set(SL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(SL_ENABLE_HIP ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/lookahead-cache)
target_link_libraries(my_application PRIVATE LookaheadCache::hip)
```

The parent project sets its HIP compiler, SDK and architecture as in the
[build guide](usage.md#hip-build-for-strix-halo). For a CPU integration omit
`SL_ENABLE_HIP` and link `LookaheadCache::cache`; for CUDA select
`SL_ENABLE_CUDA` and its target. This repository supplies the library and examples,
not a prewritten adapter for arbitrary inference engines.

## What the pair operator computes

For predecessor `p` and ordered candidate row `j`, the global-head layout uses:

```text
bias_p[j] = dot256(head[token_ids[j]], embeddings[p])
score[j]  = current_base_logits[j] + bias_p[j]
```

The cached object is the complete `bias_p` FP32 vector. Base logits and the
winning token are recomputed for each step. A cache hit still produces the
predecessor embedding through optional `markov_out[256]`, so a caller can retain
its live confidence computation. It does not certify the caller's confidence,
verification or sampling algorithm.

[`sl_pair_config`](../include/strix_lookahead/pair.h) requires:

- A fixed **rank of 256** and row-major IEEE binary16 bit patterns, supplied as
  `uint16_t` arrays for `head` and `embeddings`.
- `embedding_rows` rows in the embedding table. Each predecessor is a global
  row ID within that table.
- An ordered, unique candidate mapping of `vocabulary_size` global IDs.
  A null `token_ids` pointer means IDs `0..V-1`.
- `SL_PAIR_HEAD_GLOBAL` for a head indexed by global ID, or
  `SL_PAIR_HEAD_SELECTED` for `V` head rows already arranged in candidate order.

The CPU pair reference and device implementations use a **direct-map** cache.
Each predecessor maps to one slot through `sl_pair_slot_for`; a collision replaces
that entry. `capacity_entries=0` executes the same selector without caching and
is the appropriate library baseline. The generic host cache below uses LRU.

## Numerical compatibility

The rank-256 reduction has 32 logical lanes. Lane `l` accumulates terms
`8*l..8*l+7` in order, followed by the XOR reduction distances `16,8,4,2,1`.
Multiplications and additions round separately to FP32. The supplied builds
disable contraction/fast-math and preserve gradual underflow for these kernels.

Greedy ties choose the **lowest global token ID**, independently of candidate
order. NaN scores are ignored; an all-NaN step returns `SL_INVALID_STATE` with an
invalid token. All-negative-infinity scores still have a valid tie winner. An
all-NaN step can still publish its computed bias, so failure of selection does
not imply an empty cache.

The CPU path requires round-to-nearest and gradual underflow; it rejects other
rounding modes and x86 FTZ/DAZ without changing cache state. On other host
architectures the caller ensures gradual underflow. A fused, reassociated,
TF32 or differently quantized existing head is not automatically equivalent to
this contract. Compare the proposal results and any consumed diagnostic arrays
against the existing path before using it in a model pipeline.

## Identity, invalidation and memory

`sl_identity` carries caller-supplied identities for weights, numerical profile
and **ordered** vocabulary. The library does not hash model files or detect
mutated external tensors. The pair operator borrows immutable model/mapping
arrays for its entire lifetime. Create a new instance when weights, layout,
dimensions or candidate IDs/order change, even when the vocabulary size stays
the same. Keep new instances' identity metadata consistent with their inputs.

`sl_pair_reset` invalidates entries while retaining the existing pointers and
dimensions; it does not authorize mutating those inputs. Device reset similarly
invalidates without rebinding. A change in contextual base logits alone does
not invalidate the static pair bias.

Payload storage is `capacity × V × sizeof(float)` per instance. For example,
16 entries and 129,280 candidates require **8,273,920 bytes**, about 7.89 MiB,
before tags, mapping and workspace. Set `payload_budget_bytes` explicitly for
your memory policy; zero means no additional payload cap. The cap excludes
metadata and does not reserve the application's other tensors. Multiply the
budget by the number of resident contexts/instances when planning memory.

## GPU lifetime, ordering and results

[`device.h`](../include/strix_lookahead/device.h) is the authoritative lifecycle
contract. The important sequence is:

1. Allocate model, mapping, predecessor, base and result buffers on the chosen
   device. Create the instance on that current device and bind it to one native
   stream. Creation validates mapping IDs with a device-to-host copy.
2. Serialize calls on the instance, keep the same current device and use stream
   events to order producers/consumers on other streams. Distinct streams need
   distinct instances.
3. Enqueue with device pointers. A contiguous base vector uses `split_count=V`
   and `base_tail=null`; the split form represents a head segment followed by a
   tail segment. Optional outputs are FP32 `markov_out[256]` and `bias_out[V]`.
4. Treat `SL_OK` from enqueue as launch acceptance. Wait/query for completion,
   check the native runtime for errors, and inspect **`out->status`** before
   consuming `out->selection`. `sl_*_pair_query` returns `SL_BUSY` while pending.
5. Keep all borrowed arrays, buffers and the instance alive until work completes.
   Reset/destruction can synchronize. Check destroy's result against the header:
   a device error destroys the handle, while specified precheck failures leave
   it alive.

Enqueue does not allocate, synchronize or copy results to the host. It uses two
kernel launches and fixed workspace. Creation/reset/destruction must occur
outside capture; captured graphs run on the bound stream and must not outlive
the instance. Buffers may not overlap, except that the predecessor can alias
`out->selection.token` for device token chaining.

A native enqueue failure poisons the instance until a successful reset.
An enqueue failure during capture requires ending and discarding that capture;
reset cannot repair a partially captured graph. A runtime/device failure may
also require recovery outside the library.

## Generic host cache and certified selector

The host cache stores caller-computed FP32 vectors. On `SL_MISS`, reserve a
writer, fill **every value**, commit, acquire a reader and release it after use.
Abort an unfinished producer. Readers pin committed entries; reserve evicts the
least-recently-used unpinned entry. Filling/fully pinned situations return
`SL_BUSY`. All calls require external serialization. A lease copy is the same
ticket, not a second independently releasable lease. Reset/destroy refuse while
leases remain active; changed dimensions require a new cache.

The optional CPU selector copies vocabulary, base and bias snapshots. Base and
bias must come from the **same vocabulary handle**. Snapshot preparation reads
the full vocabulary; only `sl_selection_certify` is O(K). On `SL_MISS`, call
`sl_selection_full`; an invalid/overflow status must be handled as an error.
Finite inputs, compatible dimensions and the prescribed floating-point
environment are required. This can save work only if snapshot preparation is
reused or incorporated into an existing producer. There is no GPU selector in
this version.

Use the [benchmark guide](benchmarks.md) to separate operator correctness,
component timings and complete application measurements.

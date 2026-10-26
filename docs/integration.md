# Integration

## Status and engine boundary

Version 0.3 implements a private HIP adapter against engine commit
`683e69e91eb2203addf016e2cfdead8217e65fa2`. The [engine integration guide](engine-integration.md)
covers its pinned patch, build options and three-variant qualification runner.
Physical-device and real-model qualification remain pending. The contracts below
also apply when embedding the standalone library in another caller.

The adapter connects a DeepSeek proposal head with global FP16 pair matrices,
rank 256, ordered candidate IDs, separate core/extra base-logit arrays and a
confidence head consuming the predecessor embedding. It preserves the engine's
declared arithmetic, global-ID tie-breaking and confidence inputs. The engine
retains target verification and state rollback. Qualification compares against
the existing fused selector as well as the new operator with caching disabled.
The standalone host benchmark compares the same CPU reference with capacity zero;
it is not a measurement of that engine.

## CPU pair reference

1. Initialize `sl_pair_config`. Set candidate count, global embedding-row count,
   cache capacity, borrowed FP16 arrays and optional ordered global IDs.
2. Select `SL_PAIR_HEAD_GLOBAL` for a head indexed by global ID, or
   `SL_PAIR_HEAD_SELECTED` for a head already packed in candidate order.
3. Create the instance, then call `sl_pair_step` with the current FP32 base logits
   and global predecessor ID. Optional diagnostic buffers expose the complete
   bias and converted predecessor embedding on both hits and misses.
4. Keep arrays and mapping immutable. Destroy the instance when finished; create
   a new instance for a changed model or vocabulary.

The CPU call is synchronous and externally serialized. Unsupported floating-point
execution modes fail before changing cache state. Invalid predecessors return
`SL_INVALID_ARGUMENT`; all-NaN scores return `SL_INVALID_STATE` with an invalid
token, while still completing the bias fill. Capacity zero is the uncached path.

## HIP and CUDA pair path

Create the corresponding backend instance with device model pointers and a native
stream. Initialization may allocate, transfer IDs for validation and synchronize.
The current device and the stream binding must remain correct for every call.

`enqueue_step` accepts a device predecessor pointer and either one contiguous base
vector or two logical segments: `base_head[0..split_count)` followed by
`base_tail[0..V-split_count)`. This accommodates the engine's core and extra logits
without packing them on every step. Candidate IDs must describe that combined
order. Global-head layout uses the existing global pair matrix directly.

The two queued kernels compute/cache/select, then finish and publish. Enqueue
does not allocate, synchronize or read results back to the host. `SL_OK` from
enqueue means the launches were accepted; inspect the device result status after
completion and check native asynchronous errors. A failed enqueue poisons the
instance until a successful reset. Reset cannot guarantee recovery from a failed
device context.

Argument-validation failures leave the instance usable. A failed capture must be
ended and discarded; reset cannot repair a partially captured step. Stop graph
replays while an instance is poisoned: native graph launch bypasses the enqueue
function's host-side guard. The integration owns that external replay lifecycle.

Request `markov_out` when the confidence calculation needs the predecessor
embedding. A cache hit must still populate it. The next predecessor may point to
the previous output's token field, allowing device-side chaining without a host
readback. Other input/output overlaps are unsupported.

The caller orders producers and consumers with events. Borrowed buffers remain
valid until all consuming work completes. Instances cannot be shared across
streams. Enqueue can participate in graph capture; create, reset, query and destroy
are outside capture. Replayed graphs must use the bound stream and retain their
instance and buffers until their final execution finishes.

## Vocabulary changes and request contexts

The inspected engine's code default is the full vocabulary. Its production profile
selects a reduced vocabulary automatically: a nominal 32k core plus a language
pack, or a 48k fallback, with prompt/context IDs included. Actual row counts depend
on the deployed lists and deduplication. Only previously unseen IDs grow the list;
calling the vocabulary-update method does not necessarily change the mapping.

Accepted proposals already belong to the current vocabulary. In a speculative
window, the target correction can therefore introduce at most one new ID. Plain
decode periods can accumulate more additions before the next draft. The inspected
serving path assigns whole request contexts to ranks, and each context retains its
own ordered vocabulary. A separate vocabulary-sharded path divides candidate rows
between ranks and can move the partition boundary when the list grows.

The current pair API fixes the mapping and count at creation. **Growth requires a
new instance and loses the previous entries.** Reset does not rebind pointers or
extend the mapping. A context switch may select another matching instance; it must
not reuse an instance created for a different order. Reusing one instance across
contexts is valid only when weights, numerical identity and mapping are identical.

In the inspected engine, global pair weights and embedding arrays live with the
drafter. Core logits use persistent scratch storage. Extra IDs and logits belong
to request contexts; their buffers can be swapped or grown. An adapter must follow
those lifetimes rather than retaining an address after replacement.

The private adapter adds predecessor/context/epoch traces, actual hit/miss counts
and resident payload receipts. No native workload trace or cache speed result has
yet been collected here. Before enabling caching by default, record predecessors,
context identity, mapping changes, hits/misses and whole-cycle times.
Include invalidation costs in the comparison. An append-only extension design is
a possible later improvement, not a current capability.

## Generic host-cache integration

For caller-computed vectors, use `cache.h` independently: acquire by predecessor;
on a miss reserve, fill every element and commit; release each read lease after
consumption. Abort failed fills. A reservation may immediately evict an unpinned
victim, and abort does not restore it. Busy capacity can fall back to uncached work.

Do not reuse a live lease's output object or complete copied leases independently.
Acquire/reserve overwrite their output on success and clear it on failure;
release/commit/abort clear only on success. Generic-cache reset can replace identity
when idle, but dimension changes require a new cache. Its host pointers provide no
asynchronous device-readiness contract.

HIP/CUDA compilation is recorded separately from physical GPU qualification.
Native model execution, graph replay, failure handling and end-to-end performance
remain required integration gates. The first engine campaign isolates a single
HIP node with greedy decoding and fixed windows; it does not qualify adaptive
controller policies or two-node execution.

## Installed CMake consumer

The installed C ABI can be linked from a project enabling only C and C++. GPU
consumers still need the corresponding runtime SDK. The CUDA package dependency
requires version 12.8 or newer, matching the library's stream-ownership API.

After building, exercise installation and an independent C11 consumer with:

```sh
cmake -DSL_BUILD_DIR="$(pwd)/build-cpu" -P tests/install_consumer.cmake
```

For a non-default SDK, pass `SL_SDK_PREFIX_PATH` for HIP or
`SL_CUDATOOLKIT_ROOT` for CUDA. The consumer links backend symbols without requiring
a GPU language in its own project; this is a packaging check, not device execution.

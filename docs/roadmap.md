# Roadmap

## Current foundation

The generic host LRU cache, rank-256 CPU pair reference, direct-mapped bias cache,
greedy selector, lifecycle tests and synthetic component benchmark are implemented.
An optional CPU selector supplies immutable snapshots, an exact candidate
certificate and a full-selection fallback.

HIP and CUDA pair implementations compile. Both use a two-kernel step, device
cache lookup, coalesced hit reads, explicit publication and a capacity-zero
comparison mode. They are experimental until executed and qualified on physical
GPUs. Version 0.3 adds a private pinned HIP engine adapter and a three-variant
qualification runner. Native model execution is still pending; see the
[engine integration guide](engine-integration.md). Earlier component evidence is
recorded in [validation](validation.md).

The [0.2.1 review](review-0.2.1.md) closes an independent-SSE-rounding bug, removes
unnecessary CPU arithmetic, reduces GPU winner-reduction resources and validates
benchmark evidence and installed consumers. None of these changes closes the
physical-device or model-equivalence milestones below.

## First native milestone: Strix Halo

Qualify the HIP path on `gfx1151`: compare cached and uncached bias bits, selected
scores/tokens and confidence embeddings. Exercise hits, misses, direct-map
collisions, reduced/reordered vocabularies, invalid predecessors, reset, asynchronous
errors and graph replay. Measure both cold and reused workloads.

Run the implemented opt-in adapter at the existing head-selection boundary.
Its first campaign compares the original head, capacity zero and caching in one
HIP binary, with fixed windows, exact output IDs and the engine's native
row/state gates. Extend qualification to controller behavior and two-node
execution separately. Measure whole-cycle latency, acceptance,
request-tail latency, memory footprint and token throughput, with the same model
identity and workload. Host component ratios do not establish any of these gains.

## Mapping-aware reuse, after trace evidence

Collect the implemented context/epoch/hit traces to measure how often predecessors
recur and how often candidate mappings grow or change. The current fixed-mapping
API requires a new instance after growth; its
real hit rate must include that cost. Preserve per-request context ownership and
distinguish request placement from vocabulary sharding.

If traces show useful reuse lost to append-only growth, investigate preallocated
candidate capacity and a per-entry valid-prefix count. A hit could retain the old
prefix and compute only the new tail. This requires an explicit extension contract,
prefix validation, safe pointer rebinding and device tests before implementation
can be considered qualified. Arbitrary global-row validity tags or full-vocabulary
fills have different memory, access and miss costs and need separate evidence.

## RTX qualification

Execute the CUDA implementation on the available RTX model with the same numerical
and lifecycle fixtures. Check graph behavior, cold miss overhead and sustained
performance. Measure whether cache memory displaces useful resident or streamed
weights. Share semantics and tests while tuning launch choices with device evidence.

## Candidate reduction and contextual proposals

Measure whether the optional certificate succeeds often enough to repay preparation
and metadata costs. Move it into a device producer only after demonstrating that
its full-vocabulary scans can be reused or fused usefully. Inconclusive cases must
retain the exact full fallback.

For a learned contextual residual, first measure candidate coverage and collect
appropriate training/evaluation data. Every corrector configuration needs its own
proposal identity, acceptance calibration and measured cost. The existing engine
controller should compare proposal modes, candidate counts and block lengths;
the target verifier remains authoritative.

Compilation, operator correctness, model equivalence and full-workload improvement
are separate milestones. Advancement requires evidence for the relevant gate.

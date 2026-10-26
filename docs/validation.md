# Validation

## Integration evidence, version 0.3.0, 2026-10-26

Version 0.3.0 adds the private HIP engine adapter, its reversible pinned patch,
the native model-test gate, the three-variant campaign and trace replay. The
numerical library implementation is unchanged from 0.2.1. The full build record
is [engine-integration-build-2026-10-26.json](../benchmarks/results/engine-integration-build-2026-10-26.json).

| Gate | Result |
| --- | --- |
| Release host library and C ABI | Built with GCC/G++ 13.3.0 |
| Host CTest suite | Eight targets passed: the four native host contracts and four Python tool contracts |
| Campaign collector | Eleven test groups passed using explicitly synthetic executables; these are not model runs |
| Trace replay | Six test groups passed, including context/epoch changes and inconsistent hit flags |
| Patch installer | Six test groups passed; the final five-file patch also applied and reversed on the exact engine base, leaving it clean |
| HIP adapter enabled/disabled | Both translation-unit modes compiled for `gfx1151` with strict floating-point flags and warnings as errors |
| Engine application and native test | Batch application, HIP model-test executable and host drafter-test executable compiled and linked |
| Engine host drafter tests | Passed, including simulation, driver and controller contracts |
| Native HIP attempt | Initialization stopped at `hipMalloc: no ROCm-capable device is detected`, exit 2; no GPU/model qualification passed |
| Model campaign, Strix Halo throughput, CUDA engine adapter, two-node performance | Not run / not established |

The engine patch is bound to commit
`683e69e91eb2203addf016e2cfdead8217e65fa2`. The build uses the supported 0.2.1
installed HIP package ABI; the 0.3.0 numerical library has identical source.
The private package and engine source hashes are recorded without claiming a
source-to-binary attestation. No changes were published to the public engine.

The collector requires a successful real-weight native fixture receipt before
running any comparison. That fixture compares proposal tokens, confidence,
Markov embeddings, base logits and stage outputs bitwise on its cases. It also
checks deterministic cold/warm reuse, reordered or enlarged vocabularies, reset
and context recreation. Its hidden-state inputs are synthetic; this does not
replace the subsequent real-prompt target/state gate.

The three engine variants use the same executable and files. Exact generated IDs
and kept-row/final-state digests must agree before timing; FNV64 digests are not
collision-free numerical proofs. Timing excludes detailed traces and model load,
but includes vocabulary growth during decode. The guide documents the separate
setup counters and the truncated-layer diagnostic label.

Follow the [engine integration procedure](engine-integration.md) on the physical
device. Native tests which were skipped, receipts which are missing, or an adapter
which performs no proposal work cannot qualify a speed measurement.

## Previous component evidence, version 0.2.1, 2026-10-26

| Gate | Result |
| --- | --- |
| CMake Release host build, GCC/G++ 13.3.0 | Passed |
| Host CTest suite | Five targets passed: cache, pair arithmetic, certified selection, C ABI and evidence tools |
| AddressSanitizer and UndefinedBehaviorSanitizer | All 26 host scenario groups, C smoke and example passed |
| Release-mode checks (`-DNDEBUG`) and conversion warnings promoted to errors | Passed in sanitizer build |
| C11 compilation of the public headers | Passed |
| Evidence-collector regressions | Seven groups passed, including malformed results, nonfinite JSON, unavailable hardware and changed executables |
| Installed CMake package, independent C11/C++ consumer | Host, HIP and CUDA packages configured, linked and ran null-handle validation successfully |
| HIP library, qualification executable and benchmark | Compiled with ROCm 10.1.0 / Clang 24.0.0 for `gfx1151` |
| CUDA library, qualification executable and benchmark | Compiled with CUDA 13.4.92 for `sm_86`, `sm_89`, `sm_120` |
| Kernel resource comparison against version 0.2 | `gfx1151` compiler occupancy estimate unchanged, no scratch/spills; `sm_120` no stack/spills; shared storage 128 bytes on both |
| Native GPU execution | Pending: this environment has no supported GPU |
| Full-model or competing-system comparison | Not performed |

CMake 4.4.4 was installed in an isolated toolchain directory for these builds.
LeakSanitizer was disabled because the environment restricts the process inspection
it needs. Address and undefined-behavior sanitizers completed with leak detection
disabled; no leak-check result is claimed.

The isolated HIP runtime needed its SDK library directories on `LD_LIBRARY_PATH`.
Once the loader was configured, HIP reported that no ROCm-capable device was
detected. CUDA reported an insufficient driver for the installed runtime. Native
tests and benchmarks then returned the documented unavailable status (77).

The public headers compile as C11 and link to the C++20 implementation. Checks
remain active when `NDEBUG` is defined.

## Host coverage

The generic cache has eight scenario groups covering byte preservation, NaN payload
storage, unpublished fills, commit/abort, independent reader tickets, pinned LRU,
stale/foreign leases, identity reset, statistics and checked allocation dimensions.

The pair operator has nine groups covering analytical dot products, global and
selected head layouts, global-ID ties, NaNs/infinities, binary16 subnormals,
butterfly-order cancellation, cached diagnostic outputs, collisions, resets,
invalid inputs and floating-point environment changes. An independent full
butterfly oracle traverses all 65,536 binary16 encodings in the embeddings and
checks exact finite outputs; NaN arithmetic outputs are compared by class. The
MXCSR regression rejects independently changed SSE rounding before state mutation
and demonstrates a fixture where accepting it would change the winning token.

The certified selector has nine groups, including 5,832 exhaustive small cases
and 3,000 generated finite-binary32 cases with overflow handling. They compare
certified results with a separate full-scan oracle and exercise ownership,
mapping identity, conservative bounds, signed zero, subnormals, nonfinite inputs,
rounding mode and FTZ/DAZ rejection. The tie-aware certificate resolves 2,245 of the
5,832 exhaustive cases, versus 1,707 before the review. It resolves 1,047 of the
3,000 generated binary32 cases, versus 948 before; an actual-overflow case remains
an error. These are correctness fixtures, not an estimate of real-workload hit rate.

Reproduce the sanitizer gate with:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 make \
  BUILD_DIR=build-sanitize \
  CXXFLAGS='-O1 -g -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -DNDEBUG' \
  CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -DNDEBUG' \
  LDFLAGS='-fsanitize=address,undefined' test example
```

The Makefile appends the required C++20 and strict floating-point flags even when
custom `CXXFLAGS` are supplied.

## Device qualification prepared

The same qualification source builds separately against HIP and CUDA. It compares
the device path with the CPU oracle using 513 candidates and 1,024 embedding rows,
exercising three blocks and a ragged tail. It covers full bias/embedding/score
equivalence, split input buffers, cache hits/misses/collisions, reset, global-ID
ties, invalid predecessors, all-NaN and all-negative-infinity results, in-place token
chaining, graph capture/replay and candidate mapping validation. A separate zero-head
fixture checks that a minimum-positive-FP32 score survives both miss and hit paths.

Additional analytic fixtures cover candidate counts 1, 31, 32, 33, 255, 256, 257,
513 and 131,073, including ragged subgroups and work beyond the grid cap. Lifecycle
calls during capture must be rejected without invalidating the capture or live
instance. Wrong-device and foreign-stream checks run when two devices are present.
These tests compile for both backends but await physical-device execution.

A recognized missing device or driver yields exit 77. Unexpected initialization
errors fail. A skipped test is not a passing GPU numerical or performance result.

`tools/qualify_device.py` runs native correctness first and only starts the
benchmark after a successful result. It preserves unavailable/failed status in
the saved record and the process exit code. Schema 2 additionally rejects missing
or inconsistent benchmark fields and preserves invalid JSON as failed evidence.
The review records are
[HIP](../benchmarks/results/hip-review-qualification-2026-10-26.json) and
[CUDA](../benchmarks/results/cuda-review-qualification-2026-10-26.json).

## Performance evidence

The [current CPU record](../benchmarks/results/cpu-129280-review-2026-10-26.json) contains
source and executable hashes, compiler flags, hardware, input seed, timing data and
output checks. Hashing a working tree and executable is not a build attestation.
It is a single-thread synthetic operator experiment on an AMD EPYC host. See
[benchmark interpretation](benchmarks.md) for the results and their limits.

Before a hardware speed claim, run native qualification and compare cached and
uncached execution at the same engine revision, weights, numerical profile,
vocabulary policy, context and concurrency. Include vocabulary changes, failures,
request switches and verifier rejection/rollback paths. Record whole-cycle latency,
draft time, cache hit rate, fill cost, memory footprint and request-tail latency.
Keep prefill, sustained decode and component timing separate.

# Validation

## Recorded evidence, version 0.2.1, 2026-10-26

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

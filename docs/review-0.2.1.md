# Review 0.2.1

Date: 2026-10-26. Reviewed baseline: private repository commit `7417c3f`.

This revision preserves the public C ABI, fixed-vocabulary identity, direct-map
policy, exact greedy selection and two-kernel device step. It addresses numerical
validation, unnecessary work, measurement integrity and package integration. It
does not claim native GPU qualification or full-engine integration.

## Numerical environment: a real correctness fix

The CPU pair reference checked `fegetround()` and the SSE FTZ/DAZ bits. On x86,
an integrator can independently change the SSE rounding bits in MXCSR, so checking
the other controls did not establish the required round-to-nearest environment.
The certified selector had the same gap: its underflow probes use exact operations
and could not distinguish those rounding modes.

Both implementations now check the SSE rounding, FTZ and DAZ bits, in addition to
the existing environment checks. Unsupported arithmetic returns
`SL_INVALID_STATE` before any pair-cache or statistics change.

The new pair regression fails against the old implementation. Its finite fixture
has a 0.75-ULP score advantage: accepting round-down would erase the advantage and
change the winning global token. Selection tests also exercise all three
non-nearest SSE modes in both snapshot preparation and evaluation.

## CPU work removed without changing the result

The original reference computed all 32 lane outputs at each of five butterfly
stages: 160 additions after the lane-local products. Only lane zero is observable.
At a stage of distance `d`, its remaining ancestors are the lanes below `d`; their
second operand is `lane + d`. Retaining only those ancestors performs
`16 + 8 + 4 + 2 + 1 = 31` additions with exactly the same operands and order.

A cache hit also skips the 256 FP16-to-FP32 embedding conversions when no caller
requests the embedding output. Hits requesting that output still receive it, so
confidence-head integration retains its original input.

An independent oracle performs the complete butterfly and converts finite half
values through powers of two. It traverses all 65,536 binary16 encodings in the
embeddings. Finite bias, score and embedding results match bit for bit; arithmetic
NaNs are compared by class because their result payload is not a portable promise.

### Paired CPU measurement

The before/after experiment uses the same unchanged benchmark source, compiler
flags and generated inputs. Three paired runs alternate version order, pinned to
CPU 0 of an AMD EPYC 9V74 host. Each version uses 32,768 candidates, rank 256,
12 steps per repeat, three repeats and 16 cache slots.

| Synthetic case | Old/new time, uncached | Old/new time, cached |
| --- | ---: | ---: |
| Warm | 0.989x | 1.029x |
| Cold | 1.101x | 1.142x |
| Mixed | 1.110x | 1.098x |
| Collision | 1.053x | 1.068x |
| Sum of all four cases and three paired runs | 1.062x | 1.100x |

A ratio above one means the reviewed implementation took less time. Variation
between cases is visible, including a roughly one-percent reversal in one row.
No confidence interval was estimated. This is evidence of a modest improvement
in the synthetic CPU reference, not a full-model throughput or Strix measurement.

The [raw paired record](../benchmarks/results/cpu-review-2026-10-26.json) retains
source/binary digests, flags, hardware, affinity, execution order and every result.
A separate [full-vocabulary run](../benchmarks/results/cpu-129280-review-2026-10-26.json)
checks the revised implementation at 129,280 candidates using the schema-2
collector. All cached/uncached outputs are equivalent in both experiments.

## GPU reduction resources

The HIP and CUDA implementations now reduce already-computed winner tuples in
logical groups of 32 lanes, publish eight group winners and combine them in the
first group. Every participating group is complete, including lanes without a
candidate at a vocabulary boundary. This replaces the full shared-memory tree.

| Resource per kernel block | Before | After |
| --- | ---: | ---: |
| Explicit shared-memory winner storage | 4,096 bytes | 128 bytes |
| Block barriers in winner reduction | 9 | 1 |
| Kernels per proposal step | 2 | 2 |

The dot-product operations are unchanged. The reduction selects among completed
scores; unique global token IDs make its tie-breaking independent of the tree.
It does not sum or otherwise modify scores.

A separate [compiler-resource comparison](../benchmarks/results/gpu-compiler-resources-review-2026-10-26.json)
checks the register-pressure tradeoff against the original source with matching
flags. On `gfx1151`, `pair_pass` remains at 34 SGPR and 29 VGPR; `pair_finish`
retains 15 SGPR and increases from 13 to 15 VGPR. Clang reports the same estimated
16 waves/SIMD for both versions and no scratch or spills. On `sm_120`, `pair_pass`
remains at 40 registers and `pair_finish` increases from 25 to 26, with no stack or
spills. Both compilers confirm the shared-memory reduction to 128 bytes. These
are compiler reports, not measurements of hardware occupancy or throughput.

New native fixtures cover counts around logical-group and block boundaries and
131,073 candidates, which requires work beyond the 512-block grid cap. They use
an analytic zero-weight oracle and reversed IDs. Capture lifecycle, invalid
predecessors, argument rejection and optional multi-device ownership checks are
also prepared. Both backends compile these tests; this environment cannot execute
their kernels. Lower shared storage and fewer barriers do not establish a measured
speedup without native timing.

The integration contract now states that failed captures must be discarded and
native graph replay must stop while an instance is poisoned. Replaying a graph
directly bypasses the enqueue function's host guard.

## More exact certificates, with the same fallback

The base snapshot now records the smallest global ID among excluded rows. The
existing rounded upper bound remains an upper bound for every excluded score.
When a candidate score equals that bound, the candidate is still provably the
winner if its ID is smaller than every excluded ID. Preparation already traverses
those rows; lookup retains O(K) work and no allocations.

The finite lower-bound check still prevents hidden negative overflow. Signed zero
is safe because the selected row supplies the returned score bits. `UINT32_MAX`
remains a legal selection ID; a separate `has_tail` flag records tail existence.

| Correctness fixture | Certificates before | Certificates after | Total cases |
| --- | ---: | ---: | ---: |
| Exhaustive small vectors | 1,707 | 2,245 | 5,832 |
| Generated finite-binary32 vectors | 948 | 1,047 | 3,000 |

Every successful certificate agrees with the independent full scan. In the random
fixture, one actual-overflow case remains an error. These counts demonstrate
stronger exact certification on those fixtures, not a production acceptance rate
or measured performance gain. Full-vector preparation costs still require reuse
or integration into a producer.

## Evidence and integration checks

The device collector previously accepted any JSON object after a zero benchmark
exit code. A malformed JSON response could also abort collection before writing a
record. Schema 2 validates scope, backend, baseline, equivalence, counts, timings,
means and ratios. It rejects nonfinite numbers, including exponent overflow in
otherwise-unused fields, and duplicate JSON keys. Failed processes and invalid
payloads are saved with their output and validation errors. Exit 77 remains
unavailable and prevents timing after unavailable correctness.

Both collectors fingerprint source files and executables before and after a run,
and reject changes. They explicitly label source-to-binary binding as
`not_attested`: separately observing both digests is not proof of a build. This
distinction also applies to the older records, which remain preserved as history.

Seven regression groups exercise these failure modes. Installed packages are
also checked by independent C11/C++ consumers for host, HIP and CUDA. All three
configure, link real backend symbols and run null-handle validation without
compiling a GPU translation unit in the consumer. CUDA's exported dependency now
requires Toolkit 12.8 or newer, matching its build-time requirement. Make's C
smoke target depends on all four public headers.

## Validation outcome and next evidence needed

The host CTest suite passes all five targets. AddressSanitizer and
UndefinedBehaviorSanitizer pass 26 host scenario groups, the C ABI smoke and the
example, with `-DNDEBUG` and conversion warnings treated as errors. Leak detection
remains disabled because of the environment's process-inspection restriction.

HIP builds for `gfx1151`; CUDA builds for `sm_86`, `sm_89` and `sm_120`. Fresh
[HIP](../benchmarks/results/hip-review-qualification-2026-10-26.json) and
[CUDA](../benchmarks/results/cuda-review-qualification-2026-10-26.json) records
both report unavailable native execution and contain no GPU benchmark results.

The next hardware gate is unchanged: execute native correctness and graph tests
on Strix, measure the real request trace including vocabulary growth, and wire the
opt-in engine adapter while preserving verification and rollback. The current
fixed mapping still requires a new instance when it changes. Changing cache
associativity, vocabulary-extension behavior or launch tuning without that trace
would add unmeasured tradeoffs, so this review preserves those contracts.

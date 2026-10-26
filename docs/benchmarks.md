# Benchmark scope and results

## What is measured

The CPU executable compares the same rank-256 reference with capacity zero and
with a direct-mapped FP32 bias cache. It uses generated binary16 weights and FP32
base logits, alternates execution order, resets between repeats and verifies every
selected global token, vocabulary row and score bit pattern. Allocation and input
generation are outside the timing; the first miss of each CPU warm sequence is
inside it.

This is a synthetic, single-thread component comparison. It does not run a model
backbone, target verifier, tokenizer, model server, trained contextual corrector or
an independently optimized inference engine.

## Historical CPU run, version 0.2

This record describes the code at commit `7417c3f`, before the 0.2.1 review. Its
source fingerprints are historical and are not fingerprints of the current tree.

Hardware: AMD EPYC 9V74 host, one permitted CPU pinned. Build: GCC 13.3.0, Release
`-O3`, strict FP32 operations. Shape: 129,280 candidates, rank 256, 16 cache slots,
12 steps per repeat and three repeats. Each row below summarizes 36 steps.

| Synthetic sequence | Cache hits | Mean uncached step | Mean cached step | Total-time ratio |
| --- | ---: | ---: | ---: | ---: |
| Repeated predecessor, initial miss included | 33 / 36 | 57.56 ms | 5.18 ms | 11.12x |
| New predecessor each step | 0 / 36 | 56.25 ms | 55.03 ms | 1.02x |
| Mixed repeated and new predecessors | 6 / 36 | 59.11 ms | 48.91 ms | 1.21x |
| Deliberate direct-map collisions | 0 / 36 | 54.08 ms | 54.08 ms | 1.00x |

Every output matched. The approximately two-percent cold difference is not evidence
of a reliable improvement; no uncertainty interval was estimated. The high-reuse
case demonstrates avoided arithmetic, but its 95th-percentile cached step still
takes about 59.74 ms because misses remain expensive. These factors must not be
reported as LLM tokens-per-second gains or Strix Halo measurements.

Raw evidence and source fingerprints are in
[`cpu-129280-2026-10-26.json`](../benchmarks/results/cpu-129280-2026-10-26.json).

Reproduce and save a fresh record:

```sh
python3 tools/run_cpu_benchmark.py \
  --executable build-cpu/sl_pair_benchmark \
  --output benchmarks/results/my-cpu-run.json --pin -- \
  --vocab 129280 --steps 12 --repeats 3 --capacity 16
```

The wrapper refuses to overwrite an earlier record. Remove `--pin` on platforms
without Linux CPU affinity. It uses only the Python standard library.

## Review measurements, version 0.2.1

The [paired revision comparison](../benchmarks/results/cpu-review-2026-10-26.json)
uses the same benchmark source and compiler flags for the old and new CPU
implementation. Three before/after pairs alternate execution order, with one CPU
pinned, 32,768 candidates, 12 steps per repeat and three repeats per mode.
Aggregating all four synthetic modes gives an old-time/new-time ratio of 1.062x
for the uncached path and 1.100x for the cached path. Individual ratios range from
0.989x to 1.142x; no uncertainty interval was estimated. These observations support
a modest component improvement in this environment, not a universal speed factor.

A [fresh full-vocabulary record](../benchmarks/results/cpu-129280-review-2026-10-26.json)
exercises 129,280 candidates with the revised code and schema-2 collector. All
outputs match. Its cached/uncached comparison is distinct from the old/new revision
comparison; do not combine their ratios. The full-vocabulary warm case includes
three misses and 33 hits, just as the historical CPU protocol does.

Schema 2 preserves failed runs and validates each benchmark's declared scope,
equivalence, dimensions, counts and timing consistency. It records source and
executable SHA256 digests before and after execution and fails a run whose observed
inputs change. Source snapshots identify the working tree; they do not establish
that a binary was built from those sources. `build_source_binding: not_attested`
states that limit explicitly. A binary digest identifies the executable measured.

## GPU measurement contract

The HIP and CUDA executables compare the same two kernels with capacity zero and
with caching. GPU events enclose execution; graph upload, reset, deliberate prefill
and result readback are outside the samples. Run order alternates and all resulting
tokens, rows, scores and hit flags are checked after completion.

Unlike the CPU warm case, the GPU warm case deliberately preloads its hot set. The
JSON records this per-case policy. Do not directly compare the CPU and GPU ratios
as if their warmup definitions were identical. The GPU fixture also reuses one base
logit vector; its memory behavior is not a substitute for varying production logits.

`--graph 1` measures graph execution after upload; `--graph 0` measures ordinary
enqueues. Record hardware, SDK, clocks/power policy, memory pressure, context and
concurrency when running on a real target.

After building the backend, collect correctness and timing together:

```sh
python3 tools/qualify_device.py --backend hip --build-dir build-hip \
  --output benchmarks/results/my-halo-run.json -- \
  --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1
```

Use `--backend cuda --build-dir build-cuda` for CUDA. The wrapper refuses to
benchmark after failed or unavailable native correctness and never overwrites an
existing record. An isolated SDK installation may require its runtime library
directories on `LD_LIBRARY_PATH`, as documented by that installation.

The review's HIP and CUDA records both remain `unavailable`, with native-test exit
code 77 and no timing payload. Compiling reduced-memory kernels and linking them
from a C11 consumer do not qualify execution or establish a GPU speedup.

## Why a hit can help

At the recorded full-vocabulary dimensions, an FP16 rank-256 head has 66,191,360
bytes, while one cached FP32 bias vector has 517,120 bytes. Sixteen cached vectors
require 8,273,920 payload bytes (7.89 MiB), excluding small metadata and workspace.
The 128-to-1 ratio describes those logical reads only. Actual memory traffic and
whole-model speed depend on cache hierarchy, hit rate, launches and the rest of the
inference cycle.

Keep capacity zero as the control. Measure misses and mapping invalidations as well
as hits. The current fixed mapping requires a new instance when its candidate list
grows. No measured production reuse or growth trace is available in this repository.

## Certified selection

The optional host selector scores a candidate subset and accepts its winner only
when a validated bound excludes every omitted token. Otherwise, it requests the
full scan. Input preparation performs O(V+K) work and takes immutable snapshots;
certification is O(K). Preparation, candidate extraction and fallback costs must
be included or demonstrably amortized. The exhaustive and random correctness
checks are not performance results, and this selector is not wired into GPU decode.

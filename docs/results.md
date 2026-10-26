# Measured CPU component results

Date: **2026-10-26**. These measurements show how predecessor reuse affects the
standalone **CPU rank-256 pair operator**. The comparison uses the same
implementation and inputs with cache capacity zero as its reference. It does
not compare inference engines or measure a model, Strix Halo, HIP or CUDA.

The [campaign summary](../benchmarks/cpu/2026-10-26-summary.json) contains all
aggregates, source and executable hashes, and a SHA256 inventory of the nine
unaltered [collector records](../benchmarks/cpu/README.md). Every measured
proposal matched the reference token, vocabulary row and FP32 score bits.

## Main findings

For the repeated-predecessor trace, **91.7% of calls hit** and the median ratio
of uncached time to cached time was about **11.4**, including the first miss of
each repeat. The no-reuse and collision traces stayed around a ratio of one,
with individual runs on either side of one. They do not establish a stable
benefit when the bias must be recomputed.

The fixed mixed trace exposes the direct-map cache's dependence on token IDs
and capacity. At 32,768 candidates, changing capacity from 16 to 128 entries
changed its hit rate from **16.7% to 58.3%** and its median time ratio from
**1.205 to 2.415**. Reserved FP32 payload increased from **2 MiB to 16 MiB**.
This result describes that particular mapping and trace; it does not identify
an optimal capacity for a model workload.

## Results

Times are milliseconds per operator call: the median of three process means,
where each process mean includes all 48 calls for that mode and arm. The ratio
is the median of the three **paired process ratios**, so it need not equal the
quotient of the two displayed time medians. A ratio above one means that the
cached operator took less time. The range is the minimum and maximum of three
ratios, **not a confidence interval**. No run was dropped.

| Candidates | Entries | Trace | Hits | Uncached ms/call | Cached ms/call | Median time ratio | Ratio range |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32,768 | 16 | Repeated predecessor | 91.7% | 13.827 | 1.176 | 11.426 | 10.639–11.754 |
| 32,768 | 16 | No reuse | 0% | 13.465 | 13.280 | 1.014 | 1.000–1.020 |
| 32,768 | 16 | Fixed mixed sequence | 16.7% | 13.220 | 10.565 | 1.205 | 1.190–1.258 |
| 32,768 | 16 | Direct-map collision | 0% | 12.809 | 12.794 | 0.998 | 0.979–1.001 |
| 32,768 | 128 | Repeated predecessor | 91.7% | 13.481 | 1.164 | 11.397 | 11.233–12.092 |
| 32,768 | 128 | No reuse | 0% | 13.103 | 13.100 | 0.993 | 0.963–1.027 |
| 32,768 | 128 | Fixed mixed sequence | 58.3% | 13.732 | 5.686 | 2.415 | 2.302–2.471 |
| 32,768 | 128 | Direct-map collision | 0% | 13.268 | 12.969 | 1.023 | 1.001–1.138 |
| 129,280 | 16 | Repeated predecessor | 91.7% | 53.397 | 4.476 | 11.415 | 11.379–11.977 |
| 129,280 | 16 | No reuse | 0% | 53.743 | 52.637 | 1.009 | 0.986–1.047 |
| 129,280 | 16 | Fixed mixed sequence | 16.7% | 53.472 | 43.413 | 1.232 | 1.227–1.278 |
| 129,280 | 16 | Direct-map collision | 0% | 52.485 | 51.732 | 1.015 | 0.984–1.040 |

The cache payload is `entries × candidates × sizeof(float)`: 2 MiB, 16 MiB and
7.890625 MiB for the three configurations above. These are reserved bias-payload
sizes, excluding metadata, input arrays and process memory; they are not RSS
measurements.

## Workload and measurement method

The [benchmark source](../examples/pair_benchmark.cpp) generates finite binary16
head and embedding matrices, descending global candidate IDs and FP32 base
logits from the fixed seed `0x5168724c6f6f6b31`. Logits vary between steps. These
arrays are synthetic, with no model weights, prompts or generated text.

Each configuration ran in **three fresh processes**. Each process measured four
traces, with **12 steps per repeat and four repeats per trace**. Both arms were
reset before every repeat. The arm order alternated uncached/cached and
cached/uncached, twice each. Configuration order rotated between processes;
trace order within a process remained `warm`, `cold`, `mixed`, `collision`.

| Benchmark mode | Twelve predecessors per repeat | Cache behavior |
|---|---|---|
| `warm` | `0` repeated twelve times | One initial miss and eleven hits; no prefill is excluded from timing |
| `cold` | `0,1,2,3,4,5,6,7,8,9,10,11` | Every predecessor is new in the repeat |
| `mixed` | `0,1,0,2,0,1,0,3,0,1,0,4` | Reuse competes with direct-map collisions |
| `collision` | Alternating the first two IDs found to share a slot | Every access replaces the other ID |

For capacity 16, IDs 0 and 1 both map to slot 0. With capacity 128, they map to
slots 0 and 64, so the mixed trace retains more reused entries. The explicitly
colliding pair is `(0,1)` at capacity 16 and `(7,14)` at capacity 128. These
properties follow from the [published slot function](../src/pair.cpp); they
must not be interpreted as an observed predecessor distribution from an LLM.

The timed region contains only synchronous `sl_pair_step` calls. Input
generation, allocation, engine creation and reset are outside it. Both optional
diagnostic outputs are null. In particular, a hit need not reconstruct the
predecessor embedding for a diagnostic output. Each call still combines the
bias with its current base logits and performs argmax. The measurement uses
`std::chrono::steady_clock` and includes the per-call clock overhead.

Before measurement, a small run at 64 candidates, capacity 16, eight steps and
two repeats passed the existing collector's correctness and schema checks.
The nine measured runs then compared **1,728 pairs of returned proposals**,
covering **3,456 timed operator calls**. Token, vocabulary row and score bits
matched in every pair. The timed path does not return or compare the full bias
vector. Existing library contract tests cover that separate contract; this
campaign did not repeat the full test suite.

## Hardware, build and provenance

| Item | Recorded setting |
|---|---|
| Reported CPU model | AMD EPYC 9V74 80-Core Processor |
| Execution | One thread, pinned to permitted logical CPU 0 |
| Platform | Linux x86_64; kernel 6.18.55 |
| Compiler | GCC 13.3.0 |
| Build | CMake Release; CPU backend |
| Matrix arithmetic | Binary16 inputs; separate FP32 multiplies/adds; rank 256 |
| Repetition | Three process runs × four internal repeats × twelve calls, per mode and arm |
| Physical GPU execution | None |
| Model execution | None |

The library's recorded compiler flags were:

```text
-O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math
```

The benchmark executable's own translation unit used
`-O3 -DNDEBUG -std=gnu++20`. Flag strings came from adjacent CMake-generated
`flags.make` files. The collector recorded source and executable SHA256 values
before and after every run, and those values also remained unchanged across
the campaign. This identifies the measured artifacts; it is **not a formal
source-to-binary build attestation**.

This was a shared execution environment. Concurrent workspace builds and benchmarks
were paused during the roughly 80-second campaign, but unrelated host activity,
CPU frequency and thermal state were not controlled. The CPU model string does
not mean the benchmark used 80 cores. Ranges in the table reflect three runs
only; they are too small a sample for strong statistical claims about marginal
differences near one.

## Reproduce the campaign

Build a CPU benchmark from the repository root:

```sh
cmake -S . -B build-research/reproduce-build \
  -DCMAKE_BUILD_TYPE=Release -DSL_BUILD_EXAMPLES=ON -DSL_BUILD_TESTS=OFF \
  -DSL_ENABLE_HIP=OFF -DSL_ENABLE_CUDA=OFF
cmake --build build-research/reproduce-build --target sl_pair_benchmark --parallel
```

The following uses the existing collector, performs the same preflight and
rotates configurations in the same order. Its output names must be new because
the collector preserves existing evidence files. `--pin` selects the first CPU
allowed by the environment; verify the recorded ID on another machine.

```python
from pathlib import Path
import subprocess
import sys

executable = "build-research/reproduce-build/sl_pair_benchmark"
output = Path("build-research/reproduced-records")

def collect(name, vocabulary, capacity, steps, repeats):
    subprocess.run([
        sys.executable, "tools/run_cpu_benchmark.py",
        "--executable", executable, "--output", str(output / (name + ".json")),
        "--pin", "--", "--vocab", str(vocabulary), "--capacity", str(capacity),
        "--steps", str(steps), "--repeats", str(repeats),
    ], check=True)

collect("preflight", 64, 16, 8, 2)
configurations = [(32768, 16), (32768, 128), (129280, 16)]
for run in range(3):
    for vocabulary, capacity in configurations[run:] + configurations[:run]:
        collect(f"v{vocabulary}-c{capacity}-run{run + 1}",
                vocabulary, capacity, 12, 4)
```

To regenerate a table row, divide each record's `baseline_total_ns` and
`cached_total_ns` by 48, then take the median of each list of three process
means. Compute each process ratio as `baseline_total_ns / cached_total_ns`
before taking its median and range. The records retain the benchmark's per-call
nearest-rank p50/p95 aggregates, but the individual per-call durations are not
emitted by this benchmark.

## Interpretation and remaining evidence

These measurements establish that caching can remove substantial work from this
CPU reference operator when the same predecessor and immutable mapping recur.
They also expose the loss of that benefit on misses and direct-map collisions.
A hit still scans the current logits and cached bias, so it is not a complete
replacement for proposal evaluation.

The CPU reference includes software binary16 conversion and a specified FP32
reduction order. Its relative timings cannot estimate the cost of optimized GPU
kernels, an entire draft network, target verification or model generation. No
model token-per-second figure, acceptance improvement, Strix Halo/RTX speedup or
comparison with another runtime follows from these data. Actual predecessor
traces, native device correctness and complete-workload timing remain necessary
for those questions.

Earlier project-owned CPU records used different revisions, repetition counts
or collector versions. They were inspected as historical evidence and are not
pooled with this campaign. This report publishes one coherent current-source
campaign rather than inferring an improvement from measurements taken under
different conditions.

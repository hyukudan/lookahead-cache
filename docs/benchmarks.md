# Benchmarks and evidence

The [release validation record](validation.json) records which checks actually
ran. A successful build, an unavailable native test, a synthetic benchmark and
a real-model measurement establish different things. This repository's default
benchmarks measure the pair component with synthetic inputs.
Records for 0.4.1 and earlier use the former name StrixLookahead.

[Results](results.md) collects the actual measurements and their scope;
[method](method.md) documents the source paper and the reasoning behind the
optimization. The commands below describe how to reproduce component checks
and gather additional evidence.

## CPU component

Build the CPU targets as described in [usage](usage.md), then run:

```sh
python3 tools/run_cpu_benchmark.py \
  --executable build-cpu/sl_pair_benchmark \
  --output results/cpu-pair.json --pin -- \
  --vocab 8192 --steps 24 --repeats 5 --capacity 16
```

The output must be new. `--pin` uses the first CPU permitted to this process on
Linux. Run without competing workloads and record the CPU/power configuration
when comparing machines or revisions.

The single-thread benchmark compares capacity zero with a positive capacity
using the same reference, weights, mapping and base-logit sequence. It alternates
which arm runs first and checks global token, candidate row and score bits at
each step. Four predecessor patterns exercise cold, warm, mixed and collision
behavior. Each repeat resets the caches; CPU `warm` includes its first miss.

Reported times cover each synchronous pair step, including cache lookup,
possible bias computation and greedy selection. Allocation, input generation
and reset occur outside those samples. The record includes total times, p50/p95
step times, hits/misses and the ratio `uncached_time / cached_time`. A ratio above
one means improvement **for that synthetic component workload against its own
uncached reference**. It is not a comparison with an optimized external engine.

## HIP and CUDA

Build the relevant native test and benchmark, then choose one backend:

```sh
python3 tools/qualify_device.py \
  --backend hip --build-dir build-hip \
  --output results/hip-pair.json -- \
  --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1

python3 tools/qualify_device.py \
  --backend cuda --build-dir build-cuda \
  --output results/cuda-pair.json -- \
  --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1
```

The collector runs native correctness before the benchmark. Failure stops the
measurement. An unavailable GPU/driver produces status `unavailable` and exit
77; it is not a successful qualification. Direct CTest runs can report a skipped
device test while the available host tests pass, so inspect the per-test result.

The benchmark uses synthetic FP16 weights and **fixed base logits**, with cold,
warm, mixed and collision patterns. The reference is the same two-kernel device
operator at capacity zero. Cached/uncached order alternates and outputs plus hit
flags are checked after every sample. Device-event intervals exclude reset,
cache warm-up, graph setup and host readback; runtime warm-up occurs outside the
samples. Warm/mixed workloads prefill selected cache entries before measurement.
CPU and GPU warm workloads therefore should not be compared as identical traces.

`--graph 1` measures replay of a captured workload; `--graph 0` uses direct
enqueues. Keep that choice identical when comparing runs. Neither form measures
model loading, contextual-logit production, target verification, sampling or
complete request latency.

## Record interpretation

The CPU/device collectors preserve commands, exit status, raw stdout/stderr,
validated result data and source/executable hashes. Invalid or empty JSON,
nonfinite values, inconsistent baselines and false equivalence fields cause a
failed record. Evidence schema version 2 distinguishes a source snapshot plus
binary SHA from a build attestation: observing both hashes does **not** prove
that the binary was compiled from those exact files.

Keep failures as well as successful runs. To assess application value, measure
how often predecessors recur within one valid binding, how much time is spent
in the original pair head, the cost of misses and invalidation, resident memory
per context, and any new synchronization introduced by the adapter. Include
cold and low-reuse cases. A high cache hit rate alone does not establish a
latency improvement.

## Optional predecessor trace analysis

[`analyze_predecessor_trace.py`](../tools/analyze_predecessor_trace.py) replays
an externally produced native JSONL trace. Each draft record has this shape:

```json
{"event":"draft","context_id":1,"vocabulary_epoch":0,"vocabulary_size":4,"predecessors":[7,7],"cache_hits":[false,true],"capacity":16,"mode":"cached"}
```

`predecessors` are global embedding IDs, not indices into the candidate mapping.
The example is a format illustration, not a measured run. Every context must
start before its first draft with a cold cache. Increase `vocabulary_epoch`
after every reset or change to weights, numerics, mapping or capacity, including
a reorder that leaves the vocabulary size unchanged.

```sh
python3 tools/analyze_predecessor_trace.py \
  --input /absolute/path/to/native-trace.jsonl \
  --output results/trace-analysis.json \
  --capacities 0,4,8,16,32,64
```

Native reported hits must match the exact direct-map replay. Capacity sweeps are
simulations on that trace; no GPU execution, latency or speedup is inferred.
Baseline/fallback records do not fill the simulated cache. Legacy window traces
omit cache identity and require `--format legacy --assume-fixed-vocab`; their
output cannot establish observed cache activity.

## Optional external-application qualification

[`qualify_engine.py`](../tools/qualify_engine.py) is a client for **one specific
external HIP batch CLI and receipt contract**. It is not a universal engine
launcher. The application, adapter, native model test and GGUF assets must be
provided separately. The C library itself does not read `STRIX_LOOKAHEAD_*`
environment variables.

The external application must implement the flags constructed by
`base_command()` in that script, original/uncached/cached selection through
`STRIX_LOOKAHEAD_CAPACITY`, and structured `STRIX_LOOKAHEAD_RUN` receipts validated
by `validate_receipt()`. The separate native test must emit
`STRIX_LOOKAHEAD_MODEL_TEST` with the required schema and
`fixture:"real_weights_synthetic_hidden"`: real weights, synthetic hidden inputs,
bitwise-equal proposal outputs and zero failures. Refer to the script for the
complete field and model-environment contract; merely accepting the environment
variables or emitting arbitrary JSON is insufficient.

For an application that already implements this contract:

```sh
python3 tools/qualify_engine.py \
  --executable /absolute/path/to/compatible-batch-application \
  --native-test /absolute/path/to/compatible-model-test \
  --model /absolute/path/to/target.gguf \
  --draft /absolute/path/to/draft.gguf \
  --prompt-file /absolute/path/to/prompt.ids \
  --concurrency 1,2,4 --layers 40 --generate 128 \
  --max-positions 4096 --window-rows 6 --vocab full \
  --capacity 16 --repeats 3 --trace \
  --output results/new-application-campaign
```

This profile is single-device, fixed-window and greedy. Its deliberate limits
are concurrency 1..7, generation of at least 3 tokens and capacity 1..129;
40 layers is its full-model setting, while 4..39 are labeled truncated
diagnostics. These are **external-profile limits**, not general library limits.

The runner first requires the native proposal gate, then checks original,
uncached and cached application runs with `--check --gather`, comparing exact
output token IDs and retained-row/final-state digests. FNV64 digests are not a
collision-free proof. Timing uses `--spec-only`, rotates/reverses arm order and
still requires the same output IDs. Traces are enabled only for correctness.
The ratio uses the median application-reported `wall_decode_seconds`; process
wall time includes loading and is recorded separately.

The output directory must be new. Logs, configuration, asset/executable hashes
and partial records survive errors/timeouts. Missing receipts, mismatches,
unexpected fallback or no actual adapter work fail the campaign. Runner tests
use explicitly synthetic executable fixtures to check these failure paths;
those tests are not evidence of a GPU or real-model run.

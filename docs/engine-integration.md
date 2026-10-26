# Private HIP engine integration — 0.3

The opt-in adapter is implemented as a private patch to the public
[engine repository](https://github.com/hyukudan/strix-halo-engine-lab), pinned to
commit `683e69e91eb2203addf016e2cfdead8217e65fa2`. It replaces the static pair-head
selection boundary when explicitly enabled. Target execution, confidence inputs,
verification and commit/rollback remain the engine's responsibility.

**Current evidence is implementation, host/tool tests and compilation. Native
GPU execution and real-model equivalence/performance have not been established.**
The commands below prepare and run those gates on a machine with the matching
HIP SDK, GPU and local model assets. They do not download weights or contact an
inference server.

The patched batch application, native HIP drafter test and host drafter test
targets compiled successfully. The host drafter test passed. Attempting the HIP
test stopped at allocation with `no ROCm-capable device is detected` and exit 2;
that is an environment failure, not a GPU correctness result. No model campaign
has run in this environment.

## Scope

| Area | Current contract |
| --- | --- |
| Device | HIP, with Strix Halo `gfx1151` as the first target |
| Engine route | Batch application, `--transport single`, greedy decoding, fixed windows |
| Comparison | Existing head, uncached library, cached library in the same executable |
| Vocabulary | Immutable binding per request; changes rebuild it and discard prior entries |
| Confidence | Receives the predecessor embedding on both hits and misses |
| Errors | Invalid device proposals are discarded; unexpected fallback fails qualification |
| CUDA | Pair library implemented separately; this engine adapter is HIP only |
| Remaining scope | Two-node workloads, adaptive controllers, sampling and other model families |

The vocabulary-sharded two-rank head retains the original path and reports a
fallback. Request placement across ranks is a different mechanism; neither is
qualified by the single-node runner.

## 1. Build and install the HIP package

Use absolute paths for the private repository, an existing public-engine checkout,
a new isolated engine worktree, the package installation prefix and the HIP SDK.
The compiler must be AMD Clang itself, not the `hipcc` wrapper. The package needs
CMake 3.21 or newer; the pinned engine requires **CMake 3.24 or newer**. Use Python
3.10 or newer for the installer and qualification tools.

```sh
SL_SOURCE=/absolute/path/to/strix-lookahead
SL_ENGINE_BASE=/absolute/path/to/existing-engine-checkout
SL_ENGINE=/absolute/path/to/new-private-engine-worktree
SL_PREFIX=/absolute/path/to/strix-lookahead-install
SL_ROCM_PREFIX=/absolute/path/to/rocm-sdk
SL_HIP_COMPILER=/absolute/path/to/amdclang++

cmake -S "$SL_SOURCE" -B "$SL_SOURCE/build-hip" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$SL_HIP_COMPILER" \
  -DCMAKE_PREFIX_PATH="$SL_ROCM_PREFIX" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DSL_ENABLE_HIP=ON -DBUILD_TESTING=ON
cmake --build "$SL_SOURCE/build-hip" --parallel
cmake --install "$SL_SOURCE/build-hip" --prefix "$SL_PREFIX"
```

Before model testing, run the native operator qualification. A result of
`unavailable`/exit 77 means that hardware qualification has not passed.

```sh
python3 "$SL_SOURCE/tools/qualify_device.py" \
  --backend hip --build-dir "$SL_SOURCE/build-hip" \
  --output "$SL_SOURCE/benchmarks/results/halo-operator-run.json" -- \
  --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1
```

The output filename must be new. Configure the SDK's runtime library search path
as required by that installation before running a HIP executable.

## 2. Apply the private patch to the exact engine revision

Create a detached worktree from an existing engine clone containing the pinned
commit. This keeps ordinary development changes separate from qualification.

```sh
git -C "$SL_ENGINE_BASE" worktree add --detach "$SL_ENGINE" \
  683e69e91eb2203addf016e2cfdead8217e65fa2

python3 "$SL_SOURCE/tools/apply_engine_patch.py" \
  --checkout "$SL_ENGINE" --action check
python3 "$SL_SOURCE/tools/apply_engine_patch.py" \
  --checkout "$SL_ENGINE" --action apply
```

The package is in [`integrations/engine_lab`](../integrations/engine_lab):
`manifest.json` binds the base revision, patch and affected file hashes;
`adapter.patch` contains the changes; `ENGINE_LICENSE` preserves the engine's
license; the package README records its provenance.

`check` validates an unpatched checkout. `apply` validates again, applies all
changes, and verifies the resulting contents. A changed revision, staged change
to an adapter file or mismatched file content is rejected. The installer does
not commit, publish, alter remotes or merge the patch into a different revision.

## 3. Build the engine with the adapter available

```sh
cmake -S "$SL_ENGINE" -B "$SL_ENGINE/build-lookahead" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$SL_HIP_COMPILER" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH="$SL_PREFIX;$SL_ROCM_PREFIX" \
  -DSTRIX_HIP_INCLUDE_DIR="$SL_ROCM_PREFIX/include" \
  -DSTRIX_HIP_RUNTIME_LIBRARY="$SL_ROCM_PREFIX/lib/libamdhip64.so" \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_FAMILY_DEEPSEEK41=ON \
  -DSTRIX_ENABLE_LOOKAHEAD=ON -DBUILD_TESTING=ON
cmake --build "$SL_ENGINE/build-lookahead" --parallel \
  --target strix-deepseek41-batch strix-hip-ds41-dspark-tests
```

Adjust the runtime library path if the SDK uses `lib64` or a different layout.
`STRIX_ENABLE_LOOKAHEAD` defaults to `OFF`. Compiling with it enabled makes the
adapter available; the runtime selection remains explicit:

| `STRIX_LOOKAHEAD_CAPACITY` | Head used |
| --- | --- |
| Unset | Existing engine implementation |
| `0` | Library implementation without stored bias vectors |
| Positive integer | Library implementation with that many direct-map entries per context |

An empty or malformed value is an error. Allocation or device errors are not
silently converted into successful cached runs.

The FP32 bias payload is `capacity × candidate_count × 4` bytes per context.
Capacity 16 over 129,280 candidates uses 8,273,920 bytes, approximately 7.89 MiB,
per context, plus mapping/workspace/allocator overhead. The adapter reports actual
resident payload across active and parked contexts; the engine footprint estimate
also reserves mapping, tags and workspace. More capacity consumes memory and can
still lose reuse when a vocabulary binding changes.

The default adapter payload budget is 64 MiB per context. A real vocabulary
growth invalidates that context's entire pair cache; retaining and extending old
vectors is not implemented. This can erase the benefit of repeated predecessors
and must be measured rather than inferred from a warm component benchmark.
The runner permits capacities 1 through 129, the maximum whole-vector capacity
within that budget at the full 129,280-candidate size.

## 4. Qualify the model, then measure three variants

Supply local, compatible target/draft GGUF files and tokenizer-produced prompt
IDs. Each prompt file is a nonempty comma-separated list of decimal IDs. Files
are reused cyclically when concurrency exceeds the number of supplied prompts.
The entire selected model must fit the chosen single-node execution route.

```sh
python3 "$SL_SOURCE/tools/qualify_engine.py" \
  --executable "$SL_ENGINE/build-lookahead/strix-deepseek41-batch" \
  --native-test "$SL_ENGINE/build-lookahead/strix-hip-ds41-dspark-tests" \
  --model /absolute/path/to/target.gguf \
  --draft /absolute/path/to/draft.gguf \
  --prompt-file /absolute/path/to/prompt-a.ids \
  --prompt-file /absolute/path/to/prompt-b.ids \
  --concurrency 1,2,4 --layers 40 --generate 128 \
  --max-positions 4096 --prompt-rows 8 --window-rows 6 \
  --vocab full --capacity 16 --repeats 3 --trace \
  --output /absolute/path/to/new-engine-campaign
```

Start with a smaller concurrency list when establishing memory fit. The runner
accepts concurrency 1 through 7. At eight synchronized decoding requests, the
engine's eight-row step budget leaves no row for an extra proposal, so that case
cannot qualify this adapter and is rejected by the runner. The runner
also accepts a numeric vocabulary-core size. External vocabulary packs are not
included in this initial qualification interface. `--max-positions` must cover
the longest prompt, generated continuation and verification window.

Use all 40 target layers for a full-model campaign. Values 4 through 39 are
accepted only as truncated diagnostic runs and are labeled accordingly in the
record. At least three generated tokens are required so that work remains for
the adapter after the first token produced by prefill.

The campaign uses one executable, fixed model files, prompts, numerical/runtime
settings and greedy sampler for every variant. It first runs the required native
drafter test with the supplied model assets and requires its structured
comparison receipt and successful exit. This checks the proposal computation
itself; target verification alone could hide an incorrect proposal.

It then runs all three engine variants with the
engine's `--check --gather` gate: retained rows' fed tokens, argmax and logit digests,
the generated continuation and final attention state are checked against each
request run alone. The runner requires a passing structured receipt and compares
generated IDs exactly and row/state digests across variants. Digests are not a
collision-free proof of bitwise equality.

Only after those gates pass does it measure `--spec-only` runs. Variant order
rotates across repetitions and then reverses. `--sync-start` completes all prompts
before decode-only measurement; the first generated token per request belongs to
the prompt phase. The runner validates `decode_tokens = concurrency × (generate − 1)`.

`wall_decode_seconds` includes the batch decode step's planning, drafting, target,
control and finish work. Ratios use medians of those complete decode intervals.
Process wall time, including model loading, is recorded separately. Neither
measurement is automatically a two-node throughput or serving-latency result.

Adapter counters are deltas for each concurrency case. `map_rebuilds` includes
initial binding setup and later changes. Setup before the decode interval is
outside `wall_decode_seconds`; a vocabulary change during the decode planner is
inside it. The rebuild count alone therefore does not measure decode-only
invalidation cost.

## Records and trace interpretation

The output directory must not already exist. It contains `record.json`, stdout and
stderr for every run, and optional qualification traces. Failures, timeouts and
partial campaigns retain their records. A missing receipt, ignored opt-in,
unexpected fallback, zero adapter work or token/state discrepancy fails the gate.
Exit 77 is retained as unavailable and produces no successful timing summary.

Model/draft/prompt SHA256 values are computed once, then file metadata is checked
before and after each run. The executable is hashed repeatedly. A
`STRIX_DS41_OVERLAY` file is resolved and fingerprinted automatically. The initial
campaign rejects `STRIX_DS41_TRELLIS` and `STRIX_DS41_TRELLIS_MIX` selections because
their complete external asset inventories are not established. Use `--asset` for
other additional input files referenced through environment settings, and optional
`--model-sha256`/`--draft-sha256` to require known digests. Relevant environment
settings are recorded with credential-like fields redacted. These observations
are explicitly **not a source-to-binary build attestation**.

`--trace` records predecessors, context identity, vocabulary epoch and observed
hits during qualification only. Timing runs disable tracing, including any trace
path inherited from the shell. The trace analyzer replays the direct map from a
cold start and checks reported hits against the actual epoch changes. Its capacity
sweeps are simulations, not timing measurements:

```sh
python3 "$SL_SOURCE/tools/analyze_predecessor_trace.py" \
  --input /absolute/path/to/new-engine-campaign/qualification-00-cached.trace.jsonl \
  --output /absolute/path/to/new-trace-analysis.json \
  --capacities 0,4,8,16,32,64
```

Context resets, mapping growth and reordered IDs must change the epoch. A truncated
or concatenated trace cannot safely establish reuse without that lifecycle context.

## Rollback and remaining gates

Unset `STRIX_LOOKAHEAD_CAPACITY` to select the original head in the same compiled
binary. To remove the source patch from an otherwise unchanged patched checkout:

```sh
python3 "$SL_SOURCE/tools/apply_engine_patch.py" \
  --checkout "$SL_ENGINE" --action reverse
```

Rebuild after a source reversal. The reverse action verifies the patched file
hashes first and restores only the packaged changes.

Required next evidence includes physical `gfx1151` operator execution, model
qualification, measured cold/hit/invalidation costs, memory fit and repeated
workload timings. CUDA engine wiring, two-node execution, dynamic controller
behavior and sampled decoding remain separate work. Passing mock runner tests or
compiling the adapter does not satisfy those gates.

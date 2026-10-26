# LookaheadCache

**A proposal-head cache for exact lookahead/speculative decoding.**

[English](README.md) · [Español](README.es.md) · Version **0.4.2**

LookaheadCache supplies a host cache, a rank-256 pair operator for CPU/HIP/CUDA,
an experimental CPU certified selector, and their tests and measurement tools.
The first backend target is AMD Strix Halo (HIP, `gfx1151`), with other backends
added alongside it. Applications link the library into their own inference engine.
Model loading, execution, verification and sampling belong to that application.

The central optimization reuses a predecessor-dependent FP32 bias vector when
the predecessor repeats under the same weights, arithmetic and ordered
vocabulary. Current base logits are still evaluated on every step. A hit can
avoid recomputing the rank-256 dot products; the benefit depends on reuse, cache
collisions, vocabulary changes and the rest of the application.

The [method note](docs/method.md) explains the reference papers, the implemented
changes and the open research questions. [Recorded results](docs/results.md)
describe the measurements and their limits.

## Backends

AMD Strix Halo (HIP, `gfx1151`) comes first. The library is modular so backends
can be added alongside it. A CUDA backend is present and has been compiled for
`86`, `89` and `120`; native runs remain pending. CUDA engine integration is in
progress. The HIP engine adapter is the draft
[PR #54](https://github.com/hyukudan/strix-halo-engine-lab/pull/54).

## Components

| Component | Purpose | Public API |
| --- | --- | --- |
| Host cache | Caller-produced FP32 vectors, read/write leases and LRU eviction | [`cache.h`](include/strix_lookahead/cache.h) |
| CPU pair reference | FP16 pair weights, prescribed FP32 arithmetic, greedy selection and a direct-map bias cache | [`pair.h`](include/strix_lookahead/pair.h) |
| HIP / CUDA pair operators | Stream-bound asynchronous implementations of the pair contract, including capacity-zero comparison | [`device.h`](include/strix_lookahead/device.h) |
| Experimental CPU selector | Prove a candidate winner from conservative bounds, with full-scan fallback | [`selection.h`](include/strix_lookahead/selection.h) |
| Examples and tools | API usage, synthetic component benchmarks, correctness gates and trace replay | [`examples`](examples), [`tools`](tools) |

The library has a C ABI and a C++20 implementation. The generic host cache and
the pair operator have different eviction policies; choose the API that matches
the data you want to reuse. The selector's snapshot preparation is O(V), so its
O(K) certificate alone is not an end-to-end speedup.

## Build and try the CPU examples

Clone this repository with an account that has access:

```sh
git clone git@github.com:hyukudan/lookahead-cache.git
cd lookahead-cache

cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/sl_cache_example
./build-cpu/sl_pair_example
```

Use a C++20 compiler and CMake 3.21 or newer. Python 3.10+ enables the optional
tool tests and evidence collectors. The examples use small synthetic arrays and
require no model files. See [usage](docs/usage.md) for HIP `gfx1151`, CUDA,
installation and build options.

## Use it in an application

Install a selected build, then link its exported CMake target:

```sh
cmake --install build-cpu --prefix "$PWD/local-install"
```

```cmake
find_package(LookaheadCache 0.4.2 CONFIG REQUIRED COMPONENTS cache)
target_link_libraries(my_application PRIVATE LookaheadCache::cache)
```

Pass the installation prefix in `CMAKE_PREFIX_PATH` when configuring the
application. Builds with GPU support also export `LookaheadCache::hip` or
`LookaheadCache::cuda`. Source integration through `add_subdirectory` is supported.
The [integration guide](docs/integration.md) covers both CMake routes and the
numerical, vocabulary, lifetime, stream and result-handling contracts.

For source compatibility the C headers remain under `strix_lookahead/` and keep the `sl_` prefix.
The old package name `StrixLookahead` still resolves as a deprecated compatibility
package; new consumers should use `LookaheadCache`.

An optional HIP engine adapter is proposed in
[PR #54](https://github.com/hyukudan/strix-halo-engine-lab/pull/54).
The [external adapter section](docs/integration.md#external-reference-adapter)
links its installation and qualification guide; this repository keeps the method,
library and component evidence independent of that engine.

## Measure the component

After building the CPU target:

```sh
python3 tools/run_cpu_benchmark.py \
  --executable build-cpu/sl_pair_benchmark \
  --output results/cpu-pair.json --pin -- \
  --vocab 8192 --steps 24 --repeats 5 --capacity 16
```

Choose a new output filename for each run. `--pin` selects one permitted CPU on
Linux; omit it where affinity is unavailable. This compares the cached operator
with the same CPU reference at capacity zero using synthetic data. It measures
neither full-model inference nor a different inference engine.

For GPU builds, `tools/qualify_device.py` runs native correctness before timing.
See [benchmarks and evidence](docs/benchmarks.md) for the commands, measurement
boundaries and failure semantics. The optional `qualify_engine.py` requires an
external application implementing a specific CLI and receipt contract; that
application and its adapter are not part of this library.

## Validation and scope

The [release validation record](docs/validation.json) distinguishes host checks,
compilation and native device execution. Records for 0.4.1 and earlier use the
former name StrixLookahead. In the recorded 0.4.1 validation, the seven host CTests,
the pair example and package consumers passed. HIP compilation for `gfx1151` and
CUDA compilation for architectures `86`, `89`, `120` completed. Native GPU
tests/benchmarks and model runs have not been performed for that release.

GPU/model equivalence and performance must be established on the intended
hardware and application; no model speedup or superiority over another system
is claimed here. Inspect test skips as well as failures: an unavailable GPU is
not a passed device qualification.

- [Build and API examples](docs/usage.md)
- [Application integration and correctness contract](docs/integration.md)
- [Benchmark methodology and optional tools](docs/benchmarks.md)
- [License](LICENSE), [notice](NOTICE), [provenance](PROVENANCE.json)

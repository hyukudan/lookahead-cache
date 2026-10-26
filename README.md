# StrixLookahead

**A standalone library of reusable lookahead optimizations for LLM inference.**

[English](README.md) · [Español](README.es.md) · Version **0.4.1**

StrixLookahead supplies a host cache, a rank-256 pair operator for CPU/HIP/CUDA,
an experimental CPU certified selector, and their tests and measurement tools.
The main target is AMD Strix Halo; the CUDA backend provides a separate path for
RTX integration. Applications link the library into their own inference engine.
Model loading, execution, verification and sampling belong to that application.

The central optimization reuses a predecessor-dependent FP32 bias vector when
the predecessor repeats under the same weights, arithmetic and ordered
vocabulary. Current base logits are still evaluated on every step. A hit can
avoid recomputing the rank-256 dot products; the benefit depends on reuse, cache
collisions, vocabulary changes and the rest of the application.

The [method note](docs/method.md) explains the reference papers, the implemented
changes and the open research questions. [Recorded results](docs/results.md)
describe the measurements and their limits.

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
git clone git@github.com:hyukudan/strix-lookahead.git
cd strix-lookahead

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
find_package(StrixLookahead 0.4.1 CONFIG REQUIRED COMPONENTS cache)
target_link_libraries(my_application PRIVATE StrixLookahead::cache)
```

Pass the installation prefix in `CMAKE_PREFIX_PATH` when configuring the
application. Builds with GPU support also export `StrixLookahead::hip` or
`StrixLookahead::cuda`. Source integration through `add_subdirectory` is supported.
The [integration guide](docs/integration.md) covers both CMake routes and the
numerical, vocabulary, lifetime, stream and result-handling contracts.

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
compilation and native device execution. For this release, the seven host CTests,
the pair example and package consumers passed. HIP compilation for `gfx1151` and
CUDA compilation for architectures `86`, `89`, `120` completed. Native GPU
tests/benchmarks and model runs have not been performed for this release.

GPU/model equivalence and performance must be established on the intended
hardware and application; no model speedup or superiority over another system
is claimed here. Inspect test skips as well as failures: an unavailable GPU is
not a passed device qualification.

- [Build and API examples](docs/usage.md)
- [Application integration and correctness contract](docs/integration.md)
- [Benchmark methodology and optional tools](docs/benchmarks.md)
- [License](LICENSE), [notice](NOTICE), [provenance](PROVENANCE.json)

# Strix Lookahead

**AMD-first components for speculative LLM inference, designed around Strix Halo.**

[Español](README.es.md)

Version 0.3 adds a private, opt-in HIP engine adapter and a reproducible
three-variant qualification runner to the static rank-256 transition head,
cached execution path and experimental certified selector. The first hardware target is Strix Halo
(`gfx1151`, HIP); a CUDA implementation shares the same numerical and ownership
contract. The inference engine retains target verification, model state and sampling.

## Implemented components

| Component | Implementation | Evidence |
| --- | --- | --- |
| Generic FP32 vector cache | C API, bounded storage, independent leases, deterministic LRU | CPU lifecycle tests |
| Rank-256 pair head | Binary16 weights, FP32 arithmetic, ordered candidate vocabulary, direct-mapped bias cache | CPU equivalence tests and synthetic component measurements |
| HIP pair head | Device-resident cache, coalesced hits, two kernels per step | Compiled for `gfx1151`; native execution pending |
| CUDA pair head | Same contract and two-kernel structure | Compiled for `sm_86`, `sm_89`, `sm_120`; native execution pending |
| Certified sparse selection | Immutable validated inputs, exact upper-bound and tie certificate, full-scan fallback | CPU exhaustive and randomized checks |
| Private HIP engine adapter | Pinned patch, per-request vocabulary bindings, device token chaining and cache receipts | Engine application and host/native test targets compiled; host test passed, GPU/model qualification pending |
| Engine qualification runner | Original head vs uncached adapter vs cached adapter, native correctness gates and exact token comparison | Runner tests use mocks; no real-model speed result yet |

The project is a technical preview. The device kernels are real implementations;
there is no CPU fallback presented as GPU execution. Synthetic CPU measurements
compare the same reference arithmetic with and without caching. They do not measure
full-model inference or establish an advantage over a trained drafting system.

## Private engine integration in 0.3

The engine source is public; its integration changes are distributed **only in
this private repository** as a patch against commit
`683e69e91eb2203addf016e2cfdead8217e65fa2`. The installer checks the exact commit,
patch digest and affected file contents before making local changes. It never
commits or pushes to the public engine repository.

The adapter uses the engine's existing FP16 pair weights, split base logits and
confidence inputs. Each request owns its cache binding; vocabulary changes rebuild
that binding and invalidate old entries. Device statuses and cache hits travel in
the existing draft result readback. The engine continues to own verification,
commit/rollback and model state.

The first qualification route is **one HIP device, greedy decoding and fixed
verification windows**. It measures the existing head, capacity zero, and a positive
cache capacity in the same binary. CUDA pair kernels remain available as a library;
the engine adapter currently uses HIP. Two-node execution, adaptive controller
policies and sampled decoding require separate qualification.

See the [engine integration guide](docs/engine-integration.md) for installation,
patch application, compilation, model qualification, trace analysis and rollback.

## Review changes in 0.2.1

The [technical review](docs/review-0.2.1.md) fixes independent SSE rounding-mode
validation, removes redundant CPU butterfly work, reduces GPU winner-reduction
shared storage and barriers, accepts provably winning certificate ties, and
hardens measurement validation and installed-package checks. The C ABI, cache
mapping and two-kernel execution contract remain compatible.

## Build and test on CPU

Requires C++20, a C11 compiler, and CMake 3.21 or newer. The host library has no
runtime dependencies beyond the standard library.

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/sl_pair_benchmark --vocab 8192 --steps 24 --repeats 3 --capacity 16
```

The Makefile also supports `make test`, `make example`, and `make benchmark`.
Floating-point contraction and fast-math optimizations are disabled deliberately
to preserve the reference numerical profile.

When Python 3 is available, CTest also runs evidence, engine-runner and trace
regressions; patch-installer tests additionally require Git. Python is not a
runtime dependency of the library.

## Build for Strix Halo

Use an installed AMD HIP SDK with `gfx1151` support. Add its installation prefix
to `CMAKE_PREFIX_PATH` if it is outside the compiler's normal search path.

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSL_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build-hip --parallel
ctest --test-dir build-hip --output-on-failure
./build-hip/sl_hip_benchmark --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1
```

The current source uses `hipStreamGetDevice`; use a HIP SDK that provides that API.
The recorded build uses ROCm 10.1.0. CMake's HIP language requires AMD's Clang
compiler, rather than setting `CMAKE_HIP_COMPILER` to the `hipcc` wrapper.

## Build for CUDA

CUDA Toolkit 12.8 or newer is required. The default architectures are 86, 89 and
120; specify the architecture of the target RTX GPU when building for one machine.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/sl_cuda_benchmark --vocab 129280 --steps 24 --repeats 5 --capacity 16 --graph 1
```

Device tests and benchmarks exit with code 77 when a device or compatible driver
is unavailable. A skip is not a successful GPU correctness result.

## Integration

CMake targets are `StrixLookahead::cache`, `StrixLookahead::hip`, and
`StrixLookahead::cuda`. Backend targets exist only when explicitly enabled.

Public C headers:

- [`cache.h`](include/strix_lookahead/cache.h): generic caller-filled vector cache.
- [`pair.h`](include/strix_lookahead/pair.h): numerical reference and model configuration.
- [`device.h`](include/strix_lookahead/device.h): asynchronous HIP/CUDA execution.
- [`selection.h`](include/strix_lookahead/selection.h): certified selector and fallback.

Weights and the ordered candidate mapping remain immutable for each pair instance.
Capacity zero selects the uncached comparison path. Device instances own their
workspace and belong to one stream; enqueue performs no allocation, host readback,
or synchronization. The API supports split base-logit buffers and produces the
predecessor embedding on hits for the engine's confidence calculation.

The certificate is a separate CPU component. Its preparation scans and snapshots
the inputs; those costs must be counted before claiming a benefit from its O(K)
evaluation. A learned contextual corrector is a separate research milestone.

See [architecture](docs/architecture.md), [library integration](docs/integration.md),
[private engine integration](docs/engine-integration.md),
[benchmark interpretation](docs/benchmarks.md), [validation](docs/validation.md),
and [roadmap](docs/roadmap.md).

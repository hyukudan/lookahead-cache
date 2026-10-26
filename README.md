# Strix Lookahead

**AMD-first components for speculative LLM inference, designed around Strix Halo.**

[Español](README.es.md)

Version 0.2 implements a static rank-256 transition head, its cached execution path,
and an experimental certified selector. The first hardware target is Strix Halo
(`gfx1151`, HIP); a CUDA implementation shares the same numerical and ownership
contract. The inference engine retains target verification, model state and sampling.

## Implemented components

| Component | Implementation | Evidence |
| --- | --- | --- |
| Generic FP32 vector cache | C API, bounded storage, independent leases, deterministic LRU | CPU lifecycle tests |
| Rank-256 pair head | Binary16 weights, FP32 arithmetic, ordered candidate vocabulary, direct-mapped bias cache | CPU equivalence tests and synthetic component measurements |
| HIP pair head | Device-resident cache, coalesced hits, two kernels per step | Compiled for `gfx1151`; native execution pending |
| CUDA pair head | Same contract and two-kernel structure | Compiled for `sm_86`, `sm_89`, `sm_120`; native execution pending |
| Certified sparse selection | Immutable validated inputs, strict upper-bound certificate, full-scan fallback | CPU exhaustive and randomized checks |
| Engine integration and full-model comparison | Adapter contract documented | Not yet wired or measured |

The project is a technical preview. The device kernels are real implementations;
there is no CPU fallback presented as GPU execution. Synthetic CPU measurements
compare the same reference arithmetic with and without caching. They do not measure
full-model inference or establish an advantage over a trained drafting system.

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

See [architecture](docs/architecture.md), [integration](docs/integration.md),
[benchmark interpretation](docs/benchmarks.md), [validation](docs/validation.md),
and [roadmap](docs/roadmap.md).

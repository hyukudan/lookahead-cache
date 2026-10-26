# Build and use LookaheadCache

All commands below run from this repository's root. LookaheadCache is a library;
the examples exercise its operators with synthetic arrays. Start with the CPU
reference to learn the API, then build the backend required by the application.

## CPU build

Requirements: CMake 3.21+, a C++20 compiler and a C compiler for the C ABI test.
Python 3.10+ is optional for the library and required for the Python tools.

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
./build-cpu/sl_cache_example
./build-cpu/sl_pair_example
```

[`cache_example.cpp`](../examples/cache_example.cpp) shows how a producer fills
and publishes a host vector, acquires it for reading and releases its lease.
[`pair_example.cpp`](../examples/pair_example.cpp) holds the predecessor and
pair weights constant while changing base logits: its first three calls select
tokens `11`, `7`, `3`, with miss/hit/hit cache results. Resetting then produces
another miss. Every step checks selection and bias bits against capacity zero.
This is a small API fixture, independent of model loaders and GPU runtimes.

The host build exports `LookaheadCache::cache`. This target includes the host
cache, pair reference and certified selector, despite its short target name.
Public declarations are in [`include/strix_lookahead`](../include/strix_lookahead).
For source compatibility the C headers remain under `strix_lookahead/` and keep the `sl_` prefix.

## HIP build for Strix Halo

Use a HIP SDK supporting `gfx1151` and `hipStreamGetDevice`, with its CMake
`hip` package available. Select AMD Clang itself as `CMAKE_HIP_COMPILER`.

```sh
SL_ROCM_PREFIX=/absolute/path/to/rocm-sdk
SL_HIP_COMPILER=/absolute/path/to/amdclang++

cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSL_ENABLE_HIP=ON \
  -DCMAKE_HIP_COMPILER="$SL_HIP_COMPILER" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_PREFIX_PATH="$SL_ROCM_PREFIX"
cmake --build build-hip --parallel
ctest --test-dir build-hip --output-on-failure
```

Set runtime library search paths as required by the selected SDK. This build adds
`LookaheadCache::hip`, `sl_hip_test` and `sl_hip_benchmark`. Compilation alone
does not exercise the device. CTest records an unavailable device test as skipped
through exit 77; use the [qualification collector](benchmarks.md#hip-and-cuda)
to preserve an explicit `unavailable` result.

For source embedding into an application that already selects a HIP SDK,
`SL_HIP_INCLUDE_DIR` and `SL_HIP_RUNTIME_LIBRARY` can reuse its headers/runtime.
Both must be supplied, and this mode requires `SL_ENABLE_INSTALL=OFF` because
it embeds absolute SDK paths. Use the `hip` CMake package for installable builds.

## CUDA build

The backend requires CUDA Toolkit 12.8+ and a compatible host compiler. Choose
`CMAKE_CUDA_ARCHITECTURES` for the actual device; `120` is an example target.

```sh
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DSL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

If multiple toolkits are installed, select `CMAKE_CUDA_COMPILER` and
`CUDAToolkit_ROOT` explicitly. This build adds `LookaheadCache::cuda`,
`sl_cuda_test` and `sl_cuda_benchmark`. HIP and CUDA APIs share the pair contract,
but each requires its own native correctness and performance measurements.

## Install or embed

Install the chosen build into a prefix you control:

```sh
SL_INSTALL_PREFIX="$PWD/local-install"
cmake --install build-cpu --prefix "$SL_INSTALL_PREFIX"
cmake -S /absolute/path/to/application -B application-build \
  -DCMAKE_PREFIX_PATH="$SL_INSTALL_PREFIX"
```

Use `build-hip` or `build-cuda` in the install command for those variants. GPU
package consumers also need the matching SDK dependencies discoverable by CMake.
Link the desired exported target in the application; see the complete
[CMake integration examples](integration.md#cmake-integration).

The library is static and uses a C++ runtime even through its C ABI. A C-only
application must link with a C++ linker; the included
[consumer example](../tests/consumer/CMakeLists.txt) sets `LINKER_LANGUAGE CXX`.
The old package name `StrixLookahead` remains a deprecated compatibility package.
Set `SL_TEST_LEGACY_PACKAGE=ON` when configuring the consumer example, or pass
`-DSL_TEST_LEGACY_PACKAGE=ON` to `tests/install_consumer.cmake`, to test it.

## Build options

| Option | Standalone default | Effect |
| --- | --- | --- |
| `SL_ENABLE_HIP` | `OFF` | Build the HIP pair backend |
| `SL_ENABLE_CUDA` | `OFF` | Build the CUDA pair backend |
| `SL_BUILD_EXAMPLES` | `ON` | Build examples and component benchmarks |
| `SL_BUILD_TESTS` | Value of `BUILD_TESTING` | Build the contract tests |
| `SL_ENABLE_INSTALL` | `ON` | Export/install the library package |

With `add_subdirectory`, examples, tests and installation default to off; the
parent can enable them explicitly. Python tests are registered when an
interpreter is found. Check the CTest list if you need to confirm which tests
were available in a particular build. `make test`, `make test-tools`,
`make example` and `make benchmark` provide a host-only alternative; CMake is
the route for GPU and package integration.

## First API calls

For the CPU pair operator, initialize `sl_pair_config` with
`sl_pair_config_init`, set the immutable weight pointers, dimensions, candidate
mapping and capacity, then call `sl_pair_create`. Each synchronous
`sl_pair_step` consumes the current base logits and predecessor ID and returns
the winning global token, candidate row, FP32 score and hit flag. Check its
`sl_status` before consuming the result. Destroy the instance before releasing
its borrowed arrays.

For the generic cache, initialize `sl_cache_config`, set the vector length and
capacity, then use acquire/reserve/commit/release as shown in the example. That
API stores vectors supplied by the application; it does not calculate a pair
head.

Device calls use device addresses and asynchronous completion. Read
[the integration contract](integration.md) before wiring them into a stream or
graph. Measured release checks and outstanding hardware limitations are recorded
in [validation.json](validation.json).

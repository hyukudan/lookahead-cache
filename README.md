# Strix Lookahead

**AMD-first components for speculative LLM inference, designed around Strix Halo.**

[Español](README.es.md)

Strix Lookahead reduces repeated work in proposal generation while the inference
engine retains responsibility for verifying tokens. The primary hardware target is
Strix Halo (`gfx1151`, HIP); CUDA support for RTX GPUs is the next backend goal.

## Current scope

The initial implementation is a **CPU reference cache for complete FP32 pair-bias
vectors**. It provides a C ABI over C++20, bounded preallocated storage, deterministic
LRU eviction, reader leases, unpublished fills, explicit commit/abort, and statistics.
It stores the values supplied by the engine without transforming them.

| Component | Status |
| --- | --- |
| Host reference cache and C API | Implemented; validation recorded separately |
| HIP device cache and engine adapter | Planned, first hardware milestone |
| CUDA device cache | Planned |
| Trained contextual candidate correction | Research milestone |
| End-to-end speedup | Not measured |

The host cache is a reference implementation with host pointers.

## Build

Requires a C++20 compiler and a C compiler. The reference library has no runtime
dependencies beyond the standard library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Without CMake:

```sh
make test
make example
```

CMake consumers link `StrixLookahead::cache`. The public API is
[`include/strix_lookahead/cache.h`](include/strix_lookahead/cache.h).

## Integration contract

Create a cache for one set of weights, one numerical profile, and one **ordered
vocabulary mapping**. Look up entries by global predecessor token ID. The engine
computes a missing vector using its existing arithmetic, fills the reserved slot,
and commits it. A reader holds its lease until it has finished consuming the data.

All API calls require external serialization. Reset and destruction refuse while
leases remain active. The engine remains responsible for verification correctness.

## Project boundaries

The engine owns model execution, sampling, verification, KV/recurrent state and
commit/rollback. Trellis supplies quantized operators; tierstream manages weight
placement; TensorMeld handles orchestration. Lookahead owns reusable proposal-side
work and the lifetime of its cached data.

Start with [architecture](docs/architecture.md), [integration](docs/integration.md),
[roadmap](docs/roadmap.md), and [validation](docs/validation.md).

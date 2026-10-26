# Validation

## Evidence status

Validated on 2026-10-26 using GCC/G++ 13.3.0 on Linux:

| Check | Result |
| --- | --- |
| `make test` | Eight cache scenarios and the C ABI smoke test passed. |
| `make example` | Passed; one committed vector reused across two rounds. |
| AddressSanitizer and UndefinedBehaviorSanitizer | All tests and the example passed. |
| Release-mode checks (`-DNDEBUG`) and strict conversion warnings (`-Werror`) | Passed in the sanitizer build. |

The C smoke test is compiled as C11 and linked to the C++20 library. The example
reported `commits=1, hits=2, misses=1`. These are correctness checks, not a
performance benchmark.

LeakSanitizer was disabled because the execution environment restricts the process
inspection it requires. The CMake/CTest build is provided but was not executed in
this environment, which has no CMake installation. No GPU or end-to-end performance
result is claimed by the initial implementation.

The additional sanitizer check was run with:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 make \
  BUILD_DIR=build-sanitize \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -DNDEBUG' \
  CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -DNDEBUG' \
  LDFLAGS='-fsanitize=address,undefined' test example
```

## Reference correctness

The test suite covers:

- Byte-preserving round trips, including negative zero and NaN payloads.
- Miss, reserve, commit, hit and release; unpublished and aborted fills.
- Multiple reader tickets, pinned eviction and exhausted capacity.
- Copied stale leases, double completion and wrong-cache handles.
- Deterministic LRU and cumulative statistics.
- Identity changes only when idle; dimensions and budget validation.
- C header compatibility and linking against the C++ implementation.

Run `make test`, or build with CMake and run CTest. The example demonstrates the
lifecycle; it is not a performance benchmark.

## Hardware qualification

Compare uncached and cached execution on the same engine revision, weights,
numerical profile and vocabulary. Check cached vectors, proposal choices and final
tokens separately. Exercise both hits and misses, resets, completion failures and
the engine's rejection/rollback paths.

Measure whole-cycle latency, draft time, cache hit rate, fill cost, memory footprint
and request-tail latency. Include code, structured output and prose, with single
and concurrent requests. Keep prefill and sustained decode metrics distinct.
Report measured workload and hardware conditions with every speed claim; byte-size
ratios are not speedups.

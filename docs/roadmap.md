# Roadmap

## Reference foundation

Establish the host cache, C ABI, lifecycle tests and a minimal example. Use it as a
correctness oracle and for trace-driven sizing. Record coverage and limitations
in the validation document.

## First hardware milestone: Strix Halo

Implement device-resident lookup, FP32 storage and the hit path in HIP for `gfx1151`.
Integrate through a DeepSeek proposal adapter, preserving the engine's arithmetic
and verification. Qualify vocabulary changes, simultaneous readers, failed fills
and event ordering on actual hardware. Measure full-cycle latency and memory cost.

## CUDA portability

Add a CUDA implementation for RTX under the same identity and lifetime contract.
Measure whether cache storage displaces useful streamed weights. Backend availability
must be explicit; unsupported execution must never silently report success.

## Contextual proposal correction

Investigate candidate coverage before training a contextual residual. Treat each
corrector configuration as a distinct proposal identity with its own acceptance
calibration and measured cost. Let the existing engine controller compare modes,
candidate counts and block lengths. Preserve the model verifier as the authority.

Each milestone advances only with recorded evidence. A compiling backend, a passing
operator test and an end-to-end speedup are separate accomplishments.

#ifndef SL_DEVICE_KERNELS_HPP
#define SL_DEVICE_KERNELS_HPP

#include "strix_lookahead/device.h"

#include <cstddef>
#include <cstdint>

// The including backend supplies its runtime and a distinct namespace.
namespace SL_DEVICE_NAMESPACE {

constexpr unsigned block_threads = 256;
constexpr unsigned maximum_blocks = 512;
constexpr unsigned subgroup_threads = 32;
constexpr unsigned block_subgroups = block_threads / subgroup_threads;
static_assert(block_threads % subgroup_threads == 0 && block_subgroups <= subgroup_threads);

struct entry {
    std::uint32_t token;
    std::uint32_t valid;
};

struct winner {
    float score;
    std::uint32_t token;
    std::uint32_t row;
    std::uint32_t valid;
};

struct model_view {
    const std::uint16_t* head;
    const std::uint16_t* embeddings;
    const std::uint32_t* token_ids;
    std::uint32_t vocabulary_size;
    std::uint32_t embedding_rows;
    std::uint32_t head_layout;
    std::uint32_t capacity;
    float* payload;
    entry* entries;
};

__device__ inline std::uint32_t slot_for(std::uint32_t token, std::uint32_t capacity) {
    token ^= token >> 16;
    token *= 0x7feb352du;
    token ^= token >> 15;
    token *= 0x846ca68bu;
    token ^= token >> 16;
    return token % capacity;
}

// Native conversion matches the engine path and avoids per-weight integer
// decoding. Finite half values (including subnormals) convert exactly to FP32;
// NaN payload representation is vendor-specific and is not a model-value claim.
__device__ inline float half_to_float(std::uint16_t value) {
    return __half2float(__ushort_as_half(value));
}

template<class T>
__device__ inline T xor_lane(T value, unsigned distance) {
#if defined(SL_GPU_HIP)
    return __shfl_xor(value, static_cast<int>(distance), 32);
#else
    return __shfl_xor_sync(0xffffffffu, value, static_cast<int>(distance), 32);
#endif
}

__device__ inline winner no_winner() {
    return {__uint_as_float(0xff800000u), 0xffffffffu, 0xffffffffu, 0};
}

__device__ inline winner choose(winner a, winner b) {
    if (!b.valid) return a;
    if (!a.valid || b.score > a.score ||
        (b.score == a.score && b.token < a.token)) return b;
    return a;
}

__device__ inline winner candidate(float score, std::uint32_t token,
                                  std::uint32_t row) {
    return {score, token, row, score == score ? 1u : 0u};
}

// All 32 lanes must participate, including lanes with no candidate in a ragged
// tile. This moves winners, never re-associates score arithmetic: unique global
// IDs make choose's tie-break independent of the reduction tree (also for +/-0).
__device__ inline winner subgroup_winner(winner value) {
#pragma unroll
    for (unsigned distance = subgroup_threads / 2; distance != 0; distance >>= 1) {
        const winner other{xor_lane(value.score, distance),
                           xor_lane(value.token, distance),
                           xor_lane(value.row, distance),
                           xor_lane(value.valid, distance)};
        value = choose(value, other);
    }
    return value;
}

// Only thread zero consumes the returned block winner. One shared-memory
// exchange replaces the full 256-element shared reduction and its nine
// barriers. The second shuffle runs in one complete logical subgroup, which
// also works when HIP's physical wavefront contains two such subgroups.
__device__ inline winner block_winner(winner value, winner* subgroup_partials) {
    const unsigned thread = threadIdx.x;
    const unsigned lane = thread % subgroup_threads;
    value = subgroup_winner(value);
    if (lane == 0) subgroup_partials[thread / subgroup_threads] = value;
    __syncthreads();
    if (thread < subgroup_threads) {
        value = thread < block_subgroups ? subgroup_partials[thread] : no_winner();
        value = subgroup_winner(value);
    }
    return value;
}

__device__ inline std::uint32_t global_id(const model_view& model,
                                         std::uint32_t row) {
    return model.token_ids ? model.token_ids[row] : row;
}

__device__ inline float base_value(const float* head, const float* tail,
                                  std::uint32_t split, std::uint32_t row) {
    return row < split ? head[row] : tail[row - split];
}

// Every launch uses one fixed candidate partition on both branches. On a hit,
// all lanes load consecutive cached rows. On a miss, each 32-lane subgroup
// computes complete rank-256 rows with the reference reduction order.
__global__ void pair_pass(model_view model, const std::uint32_t* predecessor,
                          const float* base_head, const float* base_tail,
                          std::uint32_t split, winner* partials,
                          float* markov_out, float* bias_out) {
    __shared__ winner shared[block_subgroups];
    const unsigned thread = threadIdx.x;
    const std::uint32_t previous = *predecessor;
    winner best = no_winner();
    if (previous < model.embedding_rows) {
        const auto* embedding = model.embeddings +
            static_cast<std::size_t>(previous) * SL_PAIR_RANK;
        if (blockIdx.x == 0 && markov_out != nullptr) {
            markov_out[thread] = half_to_float(embedding[thread]);
        }
        std::uint32_t slot = 0;
        bool hit = false;
        float* cached = nullptr;
        if (model.capacity != 0) {
            slot = slot_for(previous, model.capacity);
            hit = model.entries[slot].valid != 0 && model.entries[slot].token == previous;
            cached = model.payload + static_cast<std::size_t>(slot) * model.vocabulary_size;
        }
        const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * block_threads;
        if (hit) {
            for (std::uint64_t row64 = static_cast<std::uint64_t>(blockIdx.x) * block_threads + thread;
                 row64 < model.vocabulary_size; row64 += stride) {
                const auto row = static_cast<std::uint32_t>(row64);
                const float bias = cached[row];
                if (bias_out != nullptr) bias_out[row] = bias;
                const float score = __fadd_rn(base_value(base_head, base_tail, split, row), bias);
                best = choose(best, candidate(score, global_id(model, row), row));
            }
        } else {
            const unsigned lane = thread & 31u;
            const unsigned subgroup = thread >> 5;
            float embedding_values[8];
#pragma unroll
            for (unsigned k = 0; k < 8; ++k) {
                embedding_values[k] = half_to_float(embedding[8u * lane + k]);
            }
            for (std::uint64_t tile = static_cast<std::uint64_t>(blockIdx.x) * block_threads;
                 tile < model.vocabulary_size; tile += stride) {
                for (unsigned offset = subgroup; offset < block_threads; offset += block_threads / 32) {
                    const std::uint64_t row64 = tile + offset;
                    if (row64 >= model.vocabulary_size) break;
                    const auto row = static_cast<std::uint32_t>(row64);
                    const auto token = global_id(model, row);
                    const auto head_row = model.head_layout == SL_PAIR_HEAD_GLOBAL ? token : row;
                    const auto* weights = model.head + static_cast<std::size_t>(head_row) * SL_PAIR_RANK;
                    float bias = 0.0f;
#pragma unroll
                    for (unsigned k = 0; k < 8; ++k) {
                        const float product = __fmul_rn(half_to_float(weights[8u * lane + k]),
                                                       embedding_values[k]);
                        bias = __fadd_rn(bias, product);
                    }
#pragma unroll
                    for (unsigned distance = 16; distance != 0; distance >>= 1) {
                        bias = __fadd_rn(bias, xor_lane(bias, distance));
                    }
                    if (lane == 0) {
                        if (cached != nullptr) cached[row] = bias;
                        if (bias_out != nullptr) bias_out[row] = bias;
                        const float score = __fadd_rn(base_value(base_head, base_tail, split, row), bias);
                        best = choose(best, candidate(score, token, row));
                    }
                }
            }
        }
    }
    best = block_winner(best, shared);
    if (thread == 0) partials[blockIdx.x] = best;
}

// Stream ordering is the publication barrier: all blocks of pair_pass finish
// before this kernel can publish the tag or overwrite an in-place predecessor.
__global__ void pair_finish(model_view model, const std::uint32_t* predecessor,
                            const winner* partials, std::uint32_t count,
                            sl_device_result* out) {
    __shared__ winner shared[block_subgroups];
    const unsigned thread = threadIdx.x;
    winner best = no_winner();
    for (unsigned i = thread; i < count; i += block_threads) best = choose(best, partials[i]);
    best = block_winner(best, shared);
    if (thread == 0) {
        const std::uint32_t previous = *predecessor;
        sl_device_result result{};
        result.selection = {0xffffffffu, 0xffffffffu, __uint_as_float(0xff800000u), 0};
        if (previous >= model.embedding_rows) {
            result.status = SL_INVALID_ARGUMENT;
        } else {
            if (model.capacity != 0) {
                const auto slot = slot_for(previous, model.capacity);
                result.selection.cache_hit = model.entries[slot].valid != 0 &&
                                             model.entries[slot].token == previous ? 1u : 0u;
                model.entries[slot] = {previous, 1};
            }
            result.status = best.valid ? SL_OK : SL_INVALID_STATE;
            if (best.valid) {
                result.selection.token = best.token;
                result.selection.vocabulary_row = best.row;
                result.selection.score = best.score;
            }
        }
        *out = result;
    }
}

}  // namespace SL_DEVICE_NAMESPACE
#endif

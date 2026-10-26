#include "strix_lookahead/pair.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#if defined(__FAST_MATH__)
#error "The pair reference requires IEEE comparisons and separate FP32 operations."
#endif

namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The pair reference requires IEEE 754 binary32.");
static_assert(FLT_EVAL_METHOD == 0,
              "The pair reference requires operations evaluated in their declared precision.");
constexpr std::uint32_t no_token = std::numeric_limits<std::uint32_t>::max();

bool checked_multiply(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    out = a + b;
    return true;
}

void increment(std::uint64_t& value, std::uint64_t amount = 1) noexcept {
    const auto limit = std::numeric_limits<std::uint64_t>::max();
    value = amount > limit - value ? limit : value + amount;
}

// Conversion is exact for every finite binary16 value, including subnormals.
// NaN sign and payload are expanded to binary32 and signaling NaNs are quieted.
float binary16(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 31U;
    std::uint32_t fraction = h & 1023U;
    std::uint32_t bits = sign;
    if (exponent == 31U) {
        bits |= 0x7f800000U | (fraction << 13U);
        if (fraction != 0U) bits |= 0x00400000U;
    } else if (exponent != 0U) {
        bits |= ((exponent + 112U) << 23U) | (fraction << 13U);
    } else if (fraction != 0U) {
        std::uint32_t normalized_exponent = 113U;
        while ((fraction & 1024U) == 0U) {
            fraction <<= 1U;
            --normalized_exponent;
        }
        bits |= (normalized_exponent << 23U) | ((fraction & 1023U) << 13U);
    }
    return std::bit_cast<float>(bits);
}

// Build this source with FP contraction disabled and without fast-math. The
// project build enforces those options so these scalar operations can remain
// register-resident without introducing fused multiply-adds or reassociation.
float rounded_add(float a, float b) noexcept {
    return a + b;
}

float rounded_multiply(float a, float b) noexcept {
    return a * b;
}

float pair_dot(const std::uint16_t* head, const float* embedding) noexcept {
    float lanes[32];
    for (std::uint32_t lane = 0; lane < 32U; ++lane) {
        float sum = 0.0F;
        for (std::uint32_t k = 0; k < 8U; ++k) {
            const auto index = 8U * lane + k;
            sum = rounded_add(sum, rounded_multiply(binary16(head[index]), embedding[index]));
        }
        lanes[lane] = sum;
    }
    for (std::uint32_t distance = 16U; distance != 0U; distance /= 2U) {
        float next[32];
        for (std::uint32_t lane = 0; lane < 32U; ++lane)
            next[lane] = rounded_add(lanes[lane], lanes[lane ^ distance]);
        std::memcpy(lanes, next, sizeof(lanes));
    }
    return lanes[0];
}

sl_pair_result empty_result() noexcept {
    return {no_token, no_token, -std::numeric_limits<float>::infinity(), 0};
}

bool supported_fp_environment() noexcept {
    if (std::fegetround() != FE_TONEAREST) return false;
#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
    // Reject FTZ/DAZ rather than silently changing the numerical identity.
    if ((_mm_getcsr() & 0x8040U) != 0U) return false;
#endif
    return true;
}

} // namespace

struct sl_pair_engine {
    sl_pair_config config{};
    sl_pair_stats stats{};
    std::unique_ptr<float[]> payload;
    std::unique_ptr<std::uint32_t[]> keys;
    std::unique_ptr<std::uint8_t[]> valid;
};

extern "C" {

void sl_pair_config_init(sl_pair_config* config) {
    if (config == nullptr) return;
    *config = {};
    config->struct_size = static_cast<std::uint32_t>(sizeof(sl_pair_config));
    config->abi_version = SL_ABI_VERSION;
    config->capacity_entries = 16;
    config->head_layout = SL_PAIR_HEAD_GLOBAL;
}

std::uint32_t sl_pair_slot_for(std::uint32_t predecessor, std::uint32_t capacity) {
    if (capacity == 0) return no_token;
    std::uint32_t x = predecessor;
    x ^= x >> 16U;
    x *= 0x7feb352dU;
    x ^= x >> 15U;
    x *= 0x846ca68bU;
    x ^= x >> 16U;
    return x % capacity;
}

sl_status sl_pair_create(const sl_pair_config* config, sl_pair_engine** out) {
    if (out != nullptr) *out = nullptr;
    if (config == nullptr || out == nullptr ||
        config->struct_size != sizeof(sl_pair_config) ||
        config->abi_version != SL_ABI_VERSION ||
        config->reserved[0] != 0 || config->reserved[1] != 0 ||
        config->vocabulary_size == 0 || config->embedding_rows == 0 ||
        config->vocabulary_size > config->embedding_rows ||
        (config->head_layout != SL_PAIR_HEAD_GLOBAL && config->head_layout != SL_PAIR_HEAD_SELECTED) ||
        config->head == nullptr || config->embeddings == nullptr) return SL_INVALID_ARGUMENT;

    std::size_t elements = 0, payload_bytes = 0, metadata_bytes = 0;
    std::size_t model_elements = 0, model_bytes = 0, total_bytes = 0;
    if (!checked_multiply(config->capacity_entries, config->vocabulary_size, elements) ||
        !checked_multiply(elements, sizeof(float), payload_bytes) ||
        !checked_multiply(config->capacity_entries, sizeof(std::uint32_t) + sizeof(std::uint8_t), metadata_bytes) ||
        !checked_add(payload_bytes, metadata_bytes, total_bytes) ||
        !checked_add(total_bytes, sizeof(sl_pair_engine), total_bytes) ||
        !checked_multiply(config->embedding_rows, SL_PAIR_RANK, model_elements) ||
        !checked_multiply(model_elements, sizeof(std::uint16_t), model_bytes)) return SL_OVERFLOW;
    if (config->payload_budget_bytes != 0 && payload_bytes > config->payload_budget_bytes)
        return SL_INVALID_ARGUMENT;

    try {
        if (config->token_ids != nullptr) {
            std::vector<std::uint32_t> sorted(config->token_ids,
                                               config->token_ids + config->vocabulary_size);
            for (const auto id : sorted) if (id >= config->embedding_rows) return SL_INVALID_ARGUMENT;
            std::sort(sorted.begin(), sorted.end());
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return SL_INVALID_ARGUMENT;
        }
        auto engine = std::make_unique<sl_pair_engine>();
        engine->config = *config;
        if (config->capacity_entries != 0) {
            engine->payload.reset(new float[elements]);
            engine->keys.reset(new std::uint32_t[config->capacity_entries]{});
            engine->valid.reset(new std::uint8_t[config->capacity_entries]{});
        }
        engine->stats.payload_bytes = static_cast<std::uint64_t>(payload_bytes);
        engine->stats.capacity_entries = config->capacity_entries;
        engine->stats.vocabulary_size = config->vocabulary_size;
        *out = engine.release();
        return SL_OK;
    } catch (const std::bad_alloc&) {
        return SL_ALLOCATION_FAILED;
    } catch (...) {
        return SL_ALLOCATION_FAILED;
    }
}

sl_status sl_pair_step(sl_pair_engine* engine, std::uint32_t predecessor,
                       const float* base_logits, sl_pair_result* out,
                       float* markov_out, float* bias_out) {
    if (out != nullptr) *out = empty_result();
    if (engine == nullptr || base_logits == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    const auto& c = engine->config;
    if (predecessor >= c.embedding_rows) return SL_INVALID_ARGUMENT;
    if (!supported_fp_environment()) return SL_INVALID_STATE;

    float embedding[SL_PAIR_RANK];
    const auto* encoded_embedding = c.embeddings + static_cast<std::size_t>(predecessor) * SL_PAIR_RANK;
    for (std::uint32_t k = 0; k < SL_PAIR_RANK; ++k) embedding[k] = binary16(encoded_embedding[k]);
    if (markov_out != nullptr) std::memcpy(markov_out, embedding, sizeof(embedding));

    const auto slot = sl_pair_slot_for(predecessor, c.capacity_entries);
    const bool hit = c.capacity_entries != 0 && engine->valid[slot] != 0 && engine->keys[slot] == predecessor;
    float* cached = c.capacity_entries == 0 ? nullptr
        : engine->payload.get() + static_cast<std::size_t>(slot) * c.vocabulary_size;
    increment(engine->stats.steps);
    increment(hit ? engine->stats.hits : engine->stats.misses);
    if (!hit) increment(engine->stats.bias_rows_computed, c.vocabulary_size);

    sl_pair_result result = empty_result();
    result.cache_hit = hit ? 1U : 0U;
    for (std::uint32_t row = 0; row < c.vocabulary_size; ++row) {
        const auto token = c.token_ids == nullptr ? row : c.token_ids[row];
        float bias;
        if (hit) {
            bias = cached[row];
        } else {
            const auto head_row = c.head_layout == SL_PAIR_HEAD_GLOBAL ? token : row;
            bias = pair_dot(c.head + static_cast<std::size_t>(head_row) * SL_PAIR_RANK, embedding);
            if (cached != nullptr) cached[row] = bias;
        }
        if (bias_out != nullptr) bias_out[row] = bias;
        const float score = rounded_add(base_logits[row], bias);
        if (score > result.score || (score == result.score && token < result.token)) {
            result.token = token;
            result.vocabulary_row = row;
            result.score = score;
        }
    }
    // A synchronous call publishes only after every selected row has been filled.
    if (cached != nullptr && !hit) {
        engine->keys[slot] = predecessor;
        engine->valid[slot] = 1;
    }
    *out = result;
    return result.token == no_token ? SL_INVALID_STATE : SL_OK;
}

sl_status sl_pair_reset(sl_pair_engine* engine, const sl_identity* identity) {
    if (engine == nullptr) return SL_INVALID_ARGUMENT;
    if (engine->config.capacity_entries != 0)
        std::memset(engine->valid.get(), 0, engine->config.capacity_entries);
    if (identity != nullptr) engine->config.identity = *identity;
    increment(engine->stats.resets);
    return SL_OK;
}

sl_status sl_pair_get_identity(const sl_pair_engine* engine, sl_identity* out) {
    if (engine == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    *out = engine->config.identity;
    return SL_OK;
}

sl_status sl_pair_get_stats(const sl_pair_engine* engine, sl_pair_stats* out) {
    if (engine == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    *out = engine->stats;
    return SL_OK;
}

sl_status sl_pair_destroy(sl_pair_engine* engine) {
    if (engine == nullptr) return SL_INVALID_ARGUMENT;
    delete engine;
    return SL_OK;
}

} // extern "C"

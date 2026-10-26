#include "strix_lookahead/selection.h"

#include <bit>
#include <cfloat>
#include <cfenv>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#if defined(__FAST_MATH__)
#error "Certified selection requires IEEE arithmetic; disable fast-math."
#endif

namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Certified selection requires IEEE 754 binary32.");
static_assert(FLT_EVAL_METHOD == 0,
              "Certified selection requires direct binary32 expression evaluation.");

struct vocabulary_data {
    std::vector<std::uint32_t> ids;
};

bool finite(float value) noexcept {
    return (std::bit_cast<std::uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

bool supported_arithmetic() noexcept {
    if (std::fegetround() != FE_TONEAREST) return false;
#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
    // fegetround can observe only the x87 control word. Callers can change
    // MXCSR independently, so also reject SSE rounding, FTZ and DAZ modes.
    if ((_mm_getcsr() & 0xe040U) != 0) return false;
#endif
    // Volatile operands require runtime operations. Inspect their bits so the
    // check itself cannot treat a denormal as zero during a comparison.
    volatile float tiny = std::bit_cast<float>(std::uint32_t{1});
    volatile float zero = 0.0F;
    volatile float input_probe = tiny + zero;
    if (std::bit_cast<std::uint32_t>(static_cast<float>(input_probe)) != 1U) return false;
    volatile float normal = std::bit_cast<float>(std::uint32_t{0x00800000U});
    volatile float half = 0.5F;
    volatile float output_probe = normal * half;
    return std::bit_cast<std::uint32_t>(static_cast<float>(output_probe)) == 0x00400000U;
}

float add_fp32(float left, float right) noexcept {
    // Require direct binary32 evaluation above; do not permit double rounding
    // through an excess-precision intermediate before this store.
    volatile float score = left + right;
    return score;
}

bool valid_count(std::size_t count) noexcept {
    return count != 0 && count <= std::numeric_limits<std::uint32_t>::max() &&
           count <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
           count <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t);
}

bool better(float score, std::uint32_t id, const sl_selection_result& best) noexcept {
    return score > best.score || (score == best.score && id < best.token_id);
}

}  // namespace

struct sl_selection_vocabulary {
    std::shared_ptr<const vocabulary_data> data;
};

struct sl_selection_base {
    std::shared_ptr<const vocabulary_data> vocabulary;
    std::vector<float> values;
    std::vector<std::uint32_t> candidates;
    float tail_max = 0.0F;
    float tail_min = 0.0F;
    std::uint32_t tail_min_id = std::numeric_limits<std::uint32_t>::max();
    bool has_tail = false;
};

struct sl_selection_bias {
    std::shared_ptr<const vocabulary_data> vocabulary;
    std::vector<float> values;
    float maximum = 0.0F;
    float minimum = 0.0F;
};

namespace {

sl_status validate_selection(const sl_selection_base* base,
                             const sl_selection_bias* bias,
                             sl_selection_result* out) noexcept {
    if (out != nullptr) *out = {};
    if (base == nullptr || bias == nullptr || out == nullptr ||
        base->vocabulary.get() != bias->vocabulary.get() ||
        base->values.size() != bias->values.size()) return SL_INVALID_ARGUMENT;
    return supported_arithmetic() ? SL_OK : SL_INVALID_STATE;
}

}  // namespace

extern "C" {

sl_status sl_selection_vocabulary_create(const uint32_t* global_ids,
                                         size_t value_count,
                                         sl_selection_vocabulary** out) {
    if (out != nullptr) *out = nullptr;
    if (out == nullptr || !valid_count(value_count)) return SL_INVALID_ARGUMENT;
    try {
        auto data = std::make_shared<vocabulary_data>();
        data->ids.resize(value_count);
        if (global_ids == nullptr) {
            for (std::size_t i = 0; i < value_count; ++i)
                data->ids[i] = static_cast<std::uint32_t>(i);
        } else {
            std::unordered_set<std::uint32_t> seen;
            seen.reserve(value_count);
            for (std::size_t i = 0; i < value_count; ++i) {
                if (!seen.insert(global_ids[i]).second) return SL_INVALID_ARGUMENT;
                data->ids[i] = global_ids[i];
            }
        }
        auto vocabulary = std::make_unique<sl_selection_vocabulary>();
        vocabulary->data = std::move(data);
        *out = vocabulary.release();
        return SL_OK;
    } catch (const std::length_error&) {
        return SL_OVERFLOW;
    } catch (...) {
        return SL_ALLOCATION_FAILED;
    }
}

sl_status sl_selection_vocabulary_destroy(sl_selection_vocabulary* vocabulary) {
    if (vocabulary == nullptr) return SL_INVALID_ARGUMENT;
    delete vocabulary;
    return SL_OK;
}

sl_status sl_selection_base_create(const sl_selection_vocabulary* vocabulary,
                                   const float* values, size_t value_count,
                                   const uint32_t* candidate_indices,
                                   size_t candidate_count,
                                   sl_selection_base** out) {
    if (out != nullptr) *out = nullptr;
    if (vocabulary == nullptr || values == nullptr || out == nullptr ||
        value_count != vocabulary->data->ids.size() || candidate_count > value_count ||
        (candidate_count != 0 && candidate_indices == nullptr)) return SL_INVALID_ARGUMENT;
    if (!supported_arithmetic()) return SL_INVALID_STATE;
    try {
        auto base = std::make_unique<sl_selection_base>();
        base->vocabulary = vocabulary->data;
        base->values.assign(values, values + value_count);
        if (candidate_count != 0)
            base->candidates.assign(candidate_indices, candidate_indices + candidate_count);
        std::vector<unsigned char> included(value_count, 0);
        for (const auto index : base->candidates) {
            if (index >= value_count || included[index] != 0) return SL_INVALID_ARGUMENT;
            included[index] = 1;
        }
        for (std::size_t i = 0; i < value_count; ++i) {
            const float value = base->values[i];
            if (!finite(value)) return SL_INVALID_ARGUMENT;
            if (included[i] != 0) continue;
            if (base->vocabulary->ids[i] < base->tail_min_id)
                base->tail_min_id = base->vocabulary->ids[i];
            if (!base->has_tail) {
                base->tail_max = value;
                base->tail_min = value;
                base->has_tail = true;
            } else {
                if (value > base->tail_max) base->tail_max = value;
                if (value < base->tail_min) base->tail_min = value;
            }
        }
        *out = base.release();
        return SL_OK;
    } catch (const std::length_error&) {
        return SL_OVERFLOW;
    } catch (...) {
        return SL_ALLOCATION_FAILED;
    }
}

sl_status sl_selection_base_destroy(sl_selection_base* base) {
    if (base == nullptr) return SL_INVALID_ARGUMENT;
    delete base;
    return SL_OK;
}

sl_status sl_selection_bias_create(const sl_selection_vocabulary* vocabulary,
                                   const float* values, size_t value_count,
                                   sl_selection_bias** out) {
    if (out != nullptr) *out = nullptr;
    if (vocabulary == nullptr || values == nullptr || out == nullptr ||
        value_count != vocabulary->data->ids.size()) return SL_INVALID_ARGUMENT;
    if (!supported_arithmetic()) return SL_INVALID_STATE;
    try {
        auto bias = std::make_unique<sl_selection_bias>();
        bias->vocabulary = vocabulary->data;
        bias->values.assign(values, values + value_count);
        bias->maximum = bias->values.front();
        bias->minimum = bias->values.front();
        for (const float value : bias->values) {
            if (!finite(value)) return SL_INVALID_ARGUMENT;
            if (value > bias->maximum) bias->maximum = value;
            if (value < bias->minimum) bias->minimum = value;
        }
        *out = bias.release();
        return SL_OK;
    } catch (const std::length_error&) {
        return SL_OVERFLOW;
    } catch (...) {
        return SL_ALLOCATION_FAILED;
    }
}

sl_status sl_selection_bias_destroy(sl_selection_bias* bias) {
    if (bias == nullptr) return SL_INVALID_ARGUMENT;
    delete bias;
    return SL_OK;
}

sl_status sl_selection_certify(const sl_selection_base* base,
                               const sl_selection_bias* bias,
                               sl_selection_result* out) {
    const auto status = validate_selection(base, bias, out);
    if (status != SL_OK) return status;
    if (base->candidates.empty()) return SL_MISS;
    sl_selection_result best{};
    bool first = true;
    for (const auto index : base->candidates) {
        const float score = add_fp32(base->values[index], bias->values[index]);
        if (!finite(score)) return SL_OVERFLOW;
        const auto id = base->vocabulary->ids[index];
        if (first || better(score, id, best)) {
            best = {id, index, score};
            first = false;
        }
    }
    if (base->has_tail) {
        const float upper = add_fp32(base->tail_max, bias->maximum);
        const float lower = add_fp32(base->tail_min, bias->minimum);
        // Equality is sufficient only when every excluded global ID loses the
        // tie. The minimum ID is prepared with the immutable tail, not supplied
        // by a caller; this keeps the certificate exact and its lookup O(K).
        const bool dominates = best.score > upper ||
            (best.score == upper && best.token_id < base->tail_min_id);
        if (!finite(upper) || !finite(lower) || !dominates) return SL_MISS;
    }
    *out = best;
    return SL_OK;
}

sl_status sl_selection_full(const sl_selection_base* base,
                            const sl_selection_bias* bias,
                            sl_selection_result* out) {
    const auto status = validate_selection(base, bias, out);
    if (status != SL_OK) return status;
    sl_selection_result best{};
    for (std::size_t i = 0; i < base->values.size(); ++i) {
        const float score = add_fp32(base->values[i], bias->values[i]);
        if (!finite(score)) return SL_OVERFLOW;
        const auto id = base->vocabulary->ids[i];
        if (i == 0 || better(score, id, best))
            best = {id, static_cast<std::uint32_t>(i), score};
    }
    *out = best;
    return SL_OK;
}

}  // extern "C"

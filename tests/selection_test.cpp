#include "strix_lookahead/selection.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {

struct Failure { int line; };
#define CHECK(condition) do { if (!(condition)) throw Failure{__LINE__}; } while (false)
#define EXPECT(call, wanted) CHECK((call) == (wanted))

using Vocabulary = std::unique_ptr<sl_selection_vocabulary,
                                  decltype(&sl_selection_vocabulary_destroy)>;
using Base = std::unique_ptr<sl_selection_base, decltype(&sl_selection_base_destroy)>;
using Bias = std::unique_ptr<sl_selection_bias, decltype(&sl_selection_bias_destroy)>;

Vocabulary vocabulary(const std::vector<std::uint32_t>& ids) {
    sl_selection_vocabulary* handle = nullptr;
    EXPECT(sl_selection_vocabulary_create(ids.data(), ids.size(), &handle), SL_OK);
    return {handle, sl_selection_vocabulary_destroy};
}

Base base(const Vocabulary& vocab, const std::vector<float>& values,
          const std::vector<std::uint32_t>& candidates) {
    sl_selection_base* handle = nullptr;
    EXPECT(sl_selection_base_create(vocab.get(), values.data(), values.size(),
                                    candidates.data(), candidates.size(), &handle), SL_OK);
    return {handle, sl_selection_base_destroy};
}

Bias bias(const Vocabulary& vocab, const std::vector<float>& values) {
    sl_selection_bias* handle = nullptr;
    EXPECT(sl_selection_bias_create(vocab.get(), values.data(), values.size(), &handle), SL_OK);
    return {handle, sl_selection_bias_destroy};
}

bool finite(float value) {
    return (std::bit_cast<std::uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

bool same(const sl_selection_result& a, const sl_selection_result& b) {
    return a.token_id == b.token_id && a.vocabulary_index == b.vocabulary_index &&
           std::bit_cast<std::uint32_t>(a.score) == std::bit_cast<std::uint32_t>(b.score);
}

void cleared(const sl_selection_result& value) {
    CHECK(value.token_id == 0 && value.vocabulary_index == 0);
    CHECK(std::bit_cast<std::uint32_t>(value.score) == 0);
}

// Independent scan of the original arrays; never uses prepared bounds.
sl_status oracle(const std::vector<std::uint32_t>& ids,
                 const std::vector<float>& values, const std::vector<float>& biases,
                 sl_selection_result& result) {
    result = {};
    for (std::size_t i = 0; i < values.size(); ++i) {
        volatile float sum = values[i] + biases[i];
        const float score = sum;
        if (!finite(score)) { result = {}; return SL_OVERFLOW; }
        if (i == 0 || score > result.score ||
            (score == result.score && ids[i] < result.token_id))
            result = {ids[i], static_cast<std::uint32_t>(i), score};
    }
    return SL_OK;
}

void snapshot_and_certificate() {
    std::vector<std::uint32_t> ids{42, 7, 99};
    std::vector<float> values{100.0F, 1.0F, 2.0F};
    std::vector<float> biases{1.0F, 0.0F, 0.0F};
    std::vector<std::uint32_t> candidates{0};
    auto vocab = vocabulary(ids);
    auto b = base(vocab, values, candidates);
    auto w = bias(vocab, biases);
    sl_selection_result result{};
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 42 && result.vocabulary_index == 0 && result.score == 101.0F);
    const auto original = result;
    ids[0] = 0;
    values[0] = -1000.0F;
    biases[1] = 2000.0F;
    candidates[0] = 2;
    vocab.reset();
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_OK);
    CHECK(same(result, original));
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(same(result, original));
}

void fallback_and_global_ties() {
    auto vocab = vocabulary({90, 4, 12});
    auto b = base(vocab, {10, 9, 0}, {0});
    auto w = bias(vocab, {0, 5, 0});
    sl_selection_result result{99, 99, 99};
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    cleared(result);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 4 && result.vocabulary_index == 1 && result.score == 14.0F);

    b = base(vocab, {1, 1, 1}, {0});
    w = bias(vocab, {0, 0, 0});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 4);
    b = base(vocab, {1, 1, 1}, {2, 0, 1});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 4);
    b = base(vocab, {1, 1, 1}, {});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 4);

    // Half an ULP rounds to an exact tie at 1.0 in binary32.
    b = base(vocab, {1, 1, -1}, {0});
    w = bias(vocab, {0x1p-24F, 0, 0});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 4 && result.score == 1.0F);
}

void signed_zero_and_subnormals() {
    auto vocab = vocabulary({9, 1});
    auto b = base(vocab, {-0.0F, -0.0F}, {0, 1});
    auto w = bias(vocab, {0.0F, -0.0F});
    sl_selection_result certified{}, full{};
    EXPECT(sl_selection_certify(b.get(), w.get(), &certified), SL_OK);
    EXPECT(sl_selection_full(b.get(), w.get(), &full), SL_OK);
    CHECK(same(certified, full) && full.token_id == 1);
    CHECK(std::bit_cast<std::uint32_t>(full.score) == 0x80000000U);
    const float tiny = std::numeric_limits<float>::denorm_min();
    b = base(vocab, {tiny, tiny}, {0, 1});
    w = bias(vocab, {tiny, 0});
    EXPECT(sl_selection_certify(b.get(), w.get(), &certified), SL_OK);
    EXPECT(sl_selection_full(b.get(), w.get(), &full), SL_OK);
    CHECK(same(certified, full) && full.token_id == 9);
    CHECK(std::bit_cast<std::uint32_t>(full.score) == 2U);
}

void invalid_arguments_and_mapping() {
    sl_selection_vocabulary* raw_vocab = nullptr;
    EXPECT(sl_selection_vocabulary_create(nullptr, 0, &raw_vocab), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_vocabulary_create(nullptr, 1, nullptr), SL_INVALID_ARGUMENT);
    const std::uint32_t duplicates[]{4, 4};
    EXPECT(sl_selection_vocabulary_create(duplicates, 2, &raw_vocab), SL_INVALID_ARGUMENT);
    CHECK(raw_vocab == nullptr);
    EXPECT(sl_selection_vocabulary_create(nullptr, 2, &raw_vocab), SL_OK);
    Vocabulary dense{raw_vocab, sl_selection_vocabulary_destroy};
    auto first = vocabulary({7, 3});
    auto second = vocabulary({7, 3});
    auto b = base(first, {2, 1}, {0});
    auto w = bias(second, {0, 0});
    sl_selection_result result{1, 2, 3};
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_INVALID_ARGUMENT);
    cleared(result);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_INVALID_ARGUMENT);
    w = bias(first, {0, 0});
    EXPECT(sl_selection_full(nullptr, w.get(), &result), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_certify(b.get(), nullptr, &result), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_full(b.get(), w.get(), nullptr), SL_INVALID_ARGUMENT);
    sl_selection_base* raw_base = nullptr;
    sl_selection_bias* raw_bias = nullptr;
    const float values[]{1, 2};
    const std::uint32_t duplicate_indices[]{0, 0};
    const std::uint32_t bad_index[]{2};
    EXPECT(sl_selection_base_create(first.get(), values, 2, duplicate_indices, 2, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_base_create(first.get(), values, 2, bad_index, 1, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_base_create(first.get(), values, 2, nullptr, 1, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_base_create(first.get(), values, 2, nullptr, 3, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_base_create(first.get(), values, 1, nullptr, 0, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_base_create(first.get(), nullptr, 2, nullptr, 0, &raw_base), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_bias_create(first.get(), values, 1, &raw_bias), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_bias_create(first.get(), nullptr, 2, &raw_bias), SL_INVALID_ARGUMENT);
    CHECK(raw_base == nullptr && raw_bias == nullptr);
    EXPECT(sl_selection_base_destroy(nullptr), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_bias_destroy(nullptr), SL_INVALID_ARGUMENT);
    EXPECT(sl_selection_vocabulary_destroy(nullptr), SL_INVALID_ARGUMENT);
    auto dense_base = base(dense, {0, 5}, {1});
    auto dense_bias = bias(dense, {0, 0});
    EXPECT(sl_selection_certify(dense_base.get(), dense_bias.get(), &result), SL_OK);
    CHECK(result.token_id == 1 && result.vocabulary_index == 1);
}

void nonfinite_and_overflow() {
    auto vocab = vocabulary({3, 1});
    const std::array<std::uint32_t, 4> invalid_bits{
        0x7f800000U, 0xff800000U, 0x7fc12345U, 0x7fa12345U};
    for (const auto bits : invalid_bits) {
        const float values[]{0, std::bit_cast<float>(bits)};
        sl_selection_base* b = nullptr;
        sl_selection_bias* w = nullptr;
        EXPECT(sl_selection_base_create(vocab.get(), values, 2, nullptr, 0, &b), SL_INVALID_ARGUMENT);
        EXPECT(sl_selection_bias_create(vocab.get(), values, 2, &w), SL_INVALID_ARGUMENT);
        CHECK(b == nullptr && w == nullptr);
    }
    const float largest = std::numeric_limits<float>::max();
    auto b = base(vocab, {largest, 0}, {0});
    auto w = bias(vocab, {largest, 0});
    sl_selection_result result{9, 9, 9};
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_OVERFLOW);
    cleared(result);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OVERFLOW);
    cleared(result);

    // A loose upper bound overflows although both real sums are finite.
    b = base(vocab, {0, largest}, {0});
    w = bias(vocab, {largest, 0});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
    CHECK(result.token_id == 1 && result.score == largest);

    // An excluded negative overflow must not be hidden by a winning candidate.
    b = base(vocab, {0, -largest}, {0});
    w = bias(vocab, {0, -largest});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OVERFLOW);
    cleared(result);
    b = base(vocab, {0, largest}, {0});
    w = bias(vocab, {0, largest});
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_MISS);
    EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OVERFLOW);
}

void unsupported_arithmetic() {
    struct Environment {
        std::fenv_t saved{};
        Environment() { CHECK(std::fegetenv(&saved) == 0); }
        ~Environment() { std::fesetenv(&saved); }
    } environment;
    auto vocab = vocabulary({0, 1});
    auto b = base(vocab, {2, 1}, {0});
    auto w = bias(vocab, {0, 0});
    sl_selection_result result{9, 9, 9};
    for (const int rounding : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        CHECK(std::fesetround(rounding) == 0);
        EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_INVALID_STATE);
        cleared(result);
        EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_INVALID_STATE);
        const float values[]{0, 0};
        sl_selection_bias* raw = nullptr;
        EXPECT(sl_selection_bias_create(vocab.get(), values, 2, &raw), SL_INVALID_STATE);
        CHECK(raw == nullptr);
    }
    CHECK(std::fesetround(FE_TONEAREST) == 0);
#if defined(__SSE__)
    struct Control {
        unsigned saved = _mm_getcsr();
        ~Control() { _mm_setcsr(saved); }
    } control;
    for (const unsigned mode : {0x8000U, 0x0040U}) {
        _mm_setcsr(control.saved | mode);
        EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_INVALID_STATE);
        cleared(result);
        _mm_setcsr(control.saved);
    }
#endif
    EXPECT(sl_selection_certify(b.get(), w.get(), &result), SL_OK);
}

void exhaustive_small_vectors() {
    const std::vector<std::uint32_t> ids{90, 4, 12};
    auto vocab = vocabulary(ids);
    std::size_t accepted = 0;
    std::size_t fallback = 0;
    for (unsigned code = 0; code < 729; ++code) {
        unsigned digits = code;
        std::vector<float> values(3), biases(3);
        for (auto* row : {&values, &biases}) {
            for (auto& value : *row) {
                value = static_cast<float>(static_cast<int>(digits % 3U) - 1);
                digits /= 3U;
            }
        }
        auto w = bias(vocab, biases);
        sl_selection_result expected{};
        EXPECT(oracle(ids, values, biases, expected), SL_OK);
        for (unsigned mask = 0; mask < 8; ++mask) {
            std::vector<std::uint32_t> candidates;
            for (std::uint32_t i = 0; i < 3; ++i)
                if ((mask & (1U << i)) != 0) candidates.push_back(i);
            std::reverse(candidates.begin(), candidates.end());
            auto b = base(vocab, values, candidates);
            sl_selection_result result{};
            EXPECT(sl_selection_full(b.get(), w.get(), &result), SL_OK);
            CHECK(same(result, expected));
            const auto status = sl_selection_certify(b.get(), w.get(), &result);
            if (status == SL_OK) { ++accepted; CHECK(same(result, expected)); }
            else { EXPECT(status, SL_MISS); ++fallback; cleared(result); }
        }
    }
    CHECK(accepted > 0 && fallback > 0);
    std::printf("Exhaustive cases: %zu certified, %zu fullscan required\n", accepted, fallback);
}

void randomized_binary32_vectors() {
    std::mt19937 random(714925U);
    std::size_t accepted = 0;
    std::size_t fallback = 0;
    for (unsigned trial = 0; trial < 3000; ++trial) {
        const std::size_t count = 1U + random() % 24U;
        std::vector<std::uint32_t> ids(count), candidates(count);
        std::vector<float> values(count), biases(count);
        std::iota(candidates.begin(), candidates.end(), 0U);
        for (std::size_t i = 0; i < count; ++i)
            ids[i] = 100U + 3U * static_cast<std::uint32_t>(i);
        std::shuffle(ids.begin(), ids.end(), random);
        std::shuffle(candidates.begin(), candidates.end(), random);
        candidates.resize(random() % (count + 1U));
        for (auto* row : {&values, &biases}) {
            for (auto& value : *row) {
                do { value = std::bit_cast<float>(static_cast<std::uint32_t>(random())); }
                while (!finite(value));
            }
        }
        auto vocab = vocabulary(ids);
        auto b = base(vocab, values, candidates);
        auto w = bias(vocab, biases);
        sl_selection_result expected{}, full{}, certified{};
        const auto expected_status = oracle(ids, values, biases, expected);
        EXPECT(sl_selection_full(b.get(), w.get(), &full), expected_status);
        CHECK(same(full, expected));
        const auto status = sl_selection_certify(b.get(), w.get(), &certified);
        if (status == SL_OK) {
            ++accepted;
            EXPECT(expected_status, SL_OK);
            CHECK(same(certified, expected));
        } else if (status == SL_MISS) {
            ++fallback;
            cleared(certified);
        } else {
            EXPECT(status, SL_OVERFLOW);
            EXPECT(expected_status, SL_OVERFLOW);
            cleared(certified);
        }
    }
    CHECK(accepted > 0 && fallback > 0);
    std::printf("Random binary32 cases: %zu certified, %zu fullscan required\n", accepted, fallback);
}

}  // namespace

int main() {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"snapshot ownership and successful certificate", snapshot_and_certificate},
        {"fallback and global-ID ties", fallback_and_global_ties},
        {"signed zero and subnormal scores", signed_zero_and_subnormals},
        {"invalid inputs and exact mapping identity", invalid_arguments_and_mapping},
        {"nonfinite inputs and score overflow", nonfinite_and_overflow},
        {"unsupported floating-point modes", unsupported_arithmetic},
        {"exhaustive three-token vectors", exhaustive_small_vectors},
        {"random binary32 equivalence", randomized_binary32_vectors},
    };
    for (const auto& test : tests) {
        try {
            test.run();
            std::printf("PASS %s\n", test.name);
        } catch (const Failure& error) {
            std::fprintf(stderr, "FAIL %s at source line %d\n", test.name, error.line);
            return 1;
        } catch (...) {
            std::fprintf(stderr, "FAIL %s: unexpected exception\n", test.name);
            return 1;
        }
    }
    std::puts("8 certified-selection scenarios passed");
    return 0;
}

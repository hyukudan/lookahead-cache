#include "strix_lookahead/pair.h"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace {

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string("line ") + std::to_string(__LINE__) + ": " #condition); } while (false)
#define STATUS(expression, expected) do { const auto actual_status = (expression); \
    if (actual_status != (expected)) throw std::runtime_error(std::string("line ") + \
    std::to_string(__LINE__) + ": " #expression " returned " + sl_status_string(actual_status)); } while (false)

constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();

struct DeleteEngine { void operator()(sl_pair_engine* p) const { (void)sl_pair_destroy(p); } };
using Engine = std::unique_ptr<sl_pair_engine, DeleteEngine>;

Engine create(const sl_pair_config& config) {
    sl_pair_engine* result = nullptr;
    STATUS(sl_pair_create(&config, &result), SL_OK);
    CHECK(result != nullptr);
    return Engine(result);
}

sl_pair_stats stats(sl_pair_engine* engine) {
    sl_pair_stats result{};
    STATUS(sl_pair_get_stats(engine, &result), SL_OK);
    return result;
}

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }

bool equal_stats(const sl_pair_stats& a, const sl_pair_stats& b) {
    return a.steps == b.steps && a.hits == b.hits && a.misses == b.misses &&
        a.bias_rows_computed == b.bias_rows_computed && a.resets == b.resets &&
        a.payload_bytes == b.payload_bytes && a.capacity_entries == b.capacity_entries &&
        a.vocabulary_size == b.vocabulary_size;
}

struct Fixture {
    std::vector<std::uint16_t> head = std::vector<std::uint16_t>(8U * SL_PAIR_RANK, 0x3c00U);
    std::vector<std::uint16_t> embeddings = std::vector<std::uint16_t>(8U * SL_PAIR_RANK, 0x3c00U);
    std::array<std::uint32_t, 3> ids{7, 2, 5};
    sl_pair_config config{};
    Fixture() {
        sl_pair_config_init(&config);
        config.vocabulary_size = 3;
        config.embedding_rows = 8;
        config.capacity_entries = 2;
        config.head = head.data();
        config.embeddings = embeddings.data();
        config.token_ids = ids.data();
        config.identity.weights[0] = 11;
        config.identity.numerics[0] = 22;
        config.identity.vocabulary[0] = 33;
    }
};

void analytical_dot_and_hit_diagnostics() {
    Fixture f;
    auto engine = create(f.config);
    const float base[3] = {0, 0, 0};
    float embedding[SL_PAIR_RANK], first_bias[3], second_bias[3];
    sl_pair_result result{};
    STATUS(sl_pair_step(engine.get(), 4, base, &result, embedding, first_bias), SL_OK);
    CHECK(result.token == 2 && result.vocabulary_row == 1 && result.score == 256.0F && result.cache_hit == 0);
    for (float value : first_bias) CHECK(bits(value) == bits(256.0F));
    for (float value : embedding) CHECK(bits(value) == bits(1.0F));
    std::fill(std::begin(embedding), std::end(embedding), -123.0F);
    STATUS(sl_pair_step(engine.get(), 4, base, &result, embedding, second_bias), SL_OK);
    CHECK(result.token == 2 && result.vocabulary_row == 1 && result.cache_hit == 1);
    CHECK(std::memcmp(first_bias, second_bias, sizeof(first_bias)) == 0);
    for (float value : embedding) CHECK(bits(value) == bits(1.0F));
    const auto s = stats(engine.get());
    CHECK(s.steps == 2 && s.hits == 1 && s.misses == 1 && s.bias_rows_computed == 3);
    CHECK(s.payload_bytes == 2U * 3U * sizeof(float));
    sl_identity identity{};
    STATUS(sl_pair_get_identity(engine.get(), &identity), SL_OK);
    CHECK(std::memcmp(&identity, &f.config.identity, sizeof(identity)) == 0);
}

void global_and_selected_layouts() {
    Fixture f;
    std::fill(f.head.begin(), f.head.end(), 0);
    constexpr std::uint16_t integer_halves[8] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800};
    for (std::uint32_t row = 0; row < 8; ++row) f.head[row * SL_PAIR_RANK] = integer_halves[row];
    std::vector<std::uint16_t> selected(3U * SL_PAIR_RANK);
    for (std::uint32_t row = 0; row < 3; ++row)
        std::memcpy(selected.data() + row * SL_PAIR_RANK, f.head.data() + f.ids[row] * SL_PAIR_RANK,
                    SL_PAIR_RANK * sizeof(std::uint16_t));
    auto global_engine = create(f.config);
    f.config.head_layout = SL_PAIR_HEAD_SELECTED;
    f.config.head = selected.data();
    auto selected_engine = create(f.config);
    const float base[3] = {0, 5, 0};
    sl_pair_result a{}, b{};
    float ba[3], bb[3];
    STATUS(sl_pair_step(global_engine.get(), 0, base, &a, nullptr, ba), SL_OK);
    STATUS(sl_pair_step(selected_engine.get(), 0, base, &b, nullptr, bb), SL_OK);
    CHECK(ba[0] == 8.0F && ba[1] == 3.0F && ba[2] == 6.0F);
    CHECK(std::memcmp(ba, bb, sizeof(ba)) == 0);
    CHECK(a.token == 2 && a.vocabulary_row == 1 && a.score == 8.0F);
    CHECK(a.token == b.token && a.vocabulary_row == b.vocabulary_row && bits(a.score) == bits(b.score));
}

void infinity_nan_and_global_tie_order() {
    Fixture f;
    std::fill(f.head.begin(), f.head.end(), 0);
    auto engine = create(f.config);
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    sl_pair_result result{};
    float base[3] = {-inf, -inf, -inf};
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    CHECK(result.token == 2 && result.vocabulary_row == 1 && result.score == -inf);
    base[0] = inf; base[1] = inf; base[2] = nan;
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    CHECK(result.token == 2 && result.score == inf);
    base[0] = nan; base[1] = -0.0F; base[2] = 0.0F;
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    CHECK(result.token == 2 && result.score == 0.0F);
    base[0] = nan; base[1] = nan; base[2] = nan;
    STATUS(sl_pair_step(engine.get(), 1, base, &result, nullptr, nullptr), SL_INVALID_STATE);
    CHECK(result.token == absent && result.vocabulary_row == absent && result.score == -inf && result.cache_hit == 0);
    STATUS(sl_pair_step(engine.get(), 1, base, &result, nullptr, nullptr), SL_INVALID_STATE);
    CHECK(result.token == absent && result.cache_hit == 1);
}

void binary16_subnormal_and_embedding_conversion() {
    Fixture f;
    std::fill(f.head.begin(), f.head.end(), 0);
    for (std::uint32_t row = 0; row < 8; ++row) f.head[row * SL_PAIR_RANK] = 0x3c00;
    const std::uint16_t halves[] = {0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0x8000};
    const std::uint32_t expected[] = {0x33800000, 0xb3800000, 0x387fc000, 0x38800000, 0x477fe000, 0x80000000};
    std::copy(std::begin(halves), std::end(halves), f.embeddings.begin());
    auto engine = create(f.config);
    const float base[3] = {0, 0, 0};
    sl_pair_result result{};
    float markov[SL_PAIR_RANK], bias[3];
    STATUS(sl_pair_step(engine.get(), 0, base, &result, markov, bias), SL_OK);
    for (std::size_t i = 0; i < std::size(expected); ++i) CHECK(bits(markov[i]) == expected[i]);
    for (float value : bias) CHECK(bits(value) == 0x33800000U);
    CHECK(bits(result.score) == 0x33800000U);
}

void butterfly_order_cancellation() {
    std::array<std::uint16_t, SL_PAIR_RANK> head{}, embedding{};
    head[0] = 0x7bff; head[8] = 0xfbff; head[128] = 0x3c00; head[136] = 0x3c00;
    embedding[0] = 0x7bff; embedding[8] = 0x7bff; embedding[128] = 0x3c00; embedding[136] = 0x3c00;
    sl_pair_config config{};
    sl_pair_config_init(&config);
    config.vocabulary_size = 1;
    config.embedding_rows = 1;
    config.capacity_entries = 1;
    config.head = head.data();
    config.embeddings = embedding.data();
    auto engine = create(config);
    const float base = 0;
    float bias = -1;
    sl_pair_result result{};
    STATUS(sl_pair_step(engine.get(), 0, &base, &result, nullptr, &bias), SL_OK);
    // Distance 16 adds each unit term to a large partial before cancellation.
    // A flat sum in increasing element order would instead leave 2.
    CHECK(bits(bias) == bits(0.0F));
    CHECK(result.token == 0 && bits(result.score) == bits(0.0F));
    STATUS(sl_pair_step(engine.get(), 0, &base, &result, nullptr, &bias), SL_OK);
    CHECK(result.cache_hit == 1 && bits(bias) == bits(0.0F));
}

void collision_reset_and_uncached_path() {
    Fixture f;
    auto engine = create(f.config);
    const float base[3] = {1, 2, 3};
    sl_pair_result result{};
    CHECK(sl_pair_slot_for(0, 2) == 0 && sl_pair_slot_for(1, 2) == 0);
    CHECK(sl_pair_slot_for(123, 0) == absent);
    const std::uint32_t predecessors[] = {0, 0, 1, 0};
    const std::uint32_t expected_hits[] = {0, 1, 0, 0};
    for (std::size_t i = 0; i < std::size(predecessors); ++i) {
        STATUS(sl_pair_step(engine.get(), predecessors[i], base, &result, nullptr, nullptr), SL_OK);
        CHECK(result.cache_hit == expected_hits[i]);
    }
    const auto before = stats(engine.get());
    CHECK(before.steps == 4 && before.hits == 1 && before.misses == 3 && before.bias_rows_computed == 9);
    sl_identity changed = f.config.identity;
    changed.vocabulary[31] = 99;
    STATUS(sl_pair_reset(engine.get(), &changed), SL_OK);
    const auto after = stats(engine.get());
    CHECK(after.resets == before.resets + 1 && after.steps == before.steps && after.hits == before.hits);
    CHECK(after.misses == before.misses && after.bias_rows_computed == before.bias_rows_computed);
    sl_identity identity{};
    STATUS(sl_pair_get_identity(engine.get(), &identity), SL_OK);
    CHECK(std::memcmp(&identity, &changed, sizeof(identity)) == 0);
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    CHECK(result.cache_hit == 0);
    f.config.capacity_entries = 0;
    auto baseline = create(f.config);
    for (unsigned i = 0; i < 2; ++i) {
        STATUS(sl_pair_step(baseline.get(), 0, base, &result, nullptr, nullptr), SL_OK);
        CHECK(result.cache_hit == 0);
    }
    CHECK(stats(baseline.get()).payload_bytes == 0 && stats(baseline.get()).misses == 2);
}

struct FpEnvironment {
    std::fenv_t saved{};
    FpEnvironment() { CHECK(std::fegetenv(&saved) == 0); }
    ~FpEnvironment() { (void)std::fesetenv(&saved); }
};

void invalid_predecessor_and_fp_environment() {
    Fixture f;
    auto engine = create(f.config);
    const float base[3] = {0, 0, 0};
    sl_pair_result result{};
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    const auto before = stats(engine.get());
    std::array<float, SL_PAIR_RANK> markov{};
    std::array<float, 3> bias{};
    markov.fill(-123);
    bias.fill(-456);
    STATUS(sl_pair_step(engine.get(), 8, base, &result, markov.data(), bias.data()), SL_INVALID_ARGUMENT);
    CHECK(result.token == absent && equal_stats(stats(engine.get()), before));
    for (const auto value : markov) CHECK(value == -123);
    for (const auto value : bias) CHECK(value == -456);
    {
        FpEnvironment restore;
        CHECK(std::fesetround(FE_DOWNWARD) == 0);
        STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_INVALID_STATE);
        CHECK(equal_stats(stats(engine.get()), before));
    }
#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
    {
        FpEnvironment restore;
        _mm_setcsr(_mm_getcsr() | 0x8040U);
        STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_INVALID_STATE);
        CHECK(equal_stats(stats(engine.get()), before));
    }
#endif
    STATUS(sl_pair_step(engine.get(), 0, base, &result, nullptr, nullptr), SL_OK);
    CHECK(result.cache_hit == 1 && stats(engine.get()).steps == before.steps + 1);
}

void invalid_configurations() {
    Fixture f;
    sl_pair_engine* result = nullptr;
    STATUS(sl_pair_create(nullptr, &result), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_create(&f.config, nullptr), SL_INVALID_ARGUMENT);
    for (unsigned scenario = 0; scenario < 11; ++scenario) {
        auto c = f.config;
        switch (scenario) {
        case 0: c.struct_size = 0; break;
        case 1: c.abi_version += 1; break;
        case 2: c.vocabulary_size = 0; break;
        case 3: c.embedding_rows = 0; break;
        case 4: c.reserved[1] = 1; break;
        case 5: c.head = nullptr; break;
        case 6: c.embeddings = nullptr; break;
        case 7: c.head_layout = 3; break;
        case 8: c.vocabulary_size = 9; break;
        case 9: c.payload_budget_bytes = 2U * 3U * sizeof(float) - 1U; break;
        case 10: c.struct_size += 8; break;
        default: break;
        }
        STATUS(sl_pair_create(&c, &result), SL_INVALID_ARGUMENT);
        CHECK(result == nullptr);
    }
    auto c = f.config;
    std::array<std::uint32_t, 3> ids{1, 1, 2};
    c.token_ids = ids.data();
    STATUS(sl_pair_create(&c, &result), SL_INVALID_ARGUMENT);
    ids = {0, 8, 1};
    STATUS(sl_pair_create(&c, &result), SL_INVALID_ARGUMENT);
    c = f.config;
    c.vocabulary_size = absent;
    c.embedding_rows = absent;
    c.capacity_entries = absent;
    c.token_ids = nullptr;
    STATUS(sl_pair_create(&c, &result), SL_OVERFLOW);
    c.vocabulary_size = 1U << 30U;
    STATUS(sl_pair_create(&c, &result), SL_OVERFLOW);
    c = f.config;
    c.payload_budget_bytes = 2U * 3U * sizeof(float);
    auto engine = create(c);
    sl_pair_result selection{};
    const float base[3]{};
    STATUS(sl_pair_step(nullptr, 0, base, &selection, nullptr, nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_step(engine.get(), 0, nullptr, &selection, nullptr, nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_step(engine.get(), 0, base, nullptr, nullptr, nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_reset(nullptr, nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_get_stats(engine.get(), nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_get_identity(engine.get(), nullptr), SL_INVALID_ARGUMENT);
    STATUS(sl_pair_destroy(nullptr), SL_INVALID_ARGUMENT);
}

} // namespace

int main() {
    const struct Test { const char* name; void (*run)(); } tests[] = {
        {"analytical dot and hit diagnostics", analytical_dot_and_hit_diagnostics},
        {"global and selected head layouts", global_and_selected_layouts},
        {"infinity, NaNs and global tie order", infinity_nan_and_global_tie_order},
        {"binary16 subnormals and embedding conversion", binary16_subnormal_and_embedding_conversion},
        {"butterfly-order cancellation", butterfly_order_cancellation},
        {"collision, reset and uncached path", collision_reset_and_uncached_path},
        {"invalid predecessor and FP environment", invalid_predecessor_and_fp_environment},
        {"invalid configurations", invalid_configurations},
    };
    unsigned passed = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            ++passed;
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << passed << " pair scenarios passed\n";
    return 0;
}

#include "strix_lookahead/pair.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void check(sl_status status, const char* operation) {
    if (status != SL_OK)
        throw std::runtime_error(std::string(operation) + ": " + sl_status_string(status));
}

struct PairDeleter {
    void operator()(sl_pair_engine* pair) const noexcept {
        // This handle is valid and all synchronous calls have completed.
        (void)sl_pair_destroy(pair);
    }
};
using Pair = std::unique_ptr<sl_pair_engine, PairDeleter>;

Pair create(const sl_pair_config& config) {
    sl_pair_engine* pair = nullptr;
    check(sl_pair_create(&config, &pair), "sl_pair_create");
    return Pair(pair);
}

bool same_bits(float first, float second) {
    return std::bit_cast<std::uint32_t>(first) == std::bit_cast<std::uint32_t>(second);
}

void print_stats(const char* name, const Pair& pair) {
    sl_pair_stats stats{};
    check(sl_pair_get_stats(pair.get(), &stats), "sl_pair_get_stats");
    std::cout << name << ": steps=" << stats.steps << " hits=" << stats.hits
              << " misses=" << stats.misses << " bias_rows_computed=" << stats.bias_rows_computed
              << " resets=" << stats.resets << " payload_bytes=" << stats.payload_bytes << '\n';
}

} // namespace

int main() try {
    constexpr std::uint32_t vocabulary_size = 3;
    constexpr std::uint32_t embedding_rows = 12;
    constexpr std::uint32_t predecessor = 5;
    const std::array<std::uint32_t, vocabulary_size> token_ids{7, 3, 11};

    // Explicit synthetic IEEE binary16 bits: 0x3a00=0.75, 0xb800=-0.5,
    // 0x3c00=1.0. All remaining terms are +0, including the other 255 ranks.
    // Selected head rows and base logits use token_ids order, not global IDs.
    const auto head = [] {
        std::array<std::uint16_t, vocabulary_size * SL_PAIR_RANK> values{};
        values[0] = 0x3a00;
        values[SL_PAIR_RANK] = 0xb800;
        values[2 * SL_PAIR_RANK] = 0x3c00;
        return values;
    }();
    const auto embeddings = [] {
        std::array<std::uint16_t, embedding_rows * SL_PAIR_RANK> values{};
        values[predecessor * SL_PAIR_RANK] = 0x3c00;
        return values;
    }();

    sl_pair_config config{};
    sl_pair_config_init(&config); // Initializes struct_size and ABI version too.
    config.vocabulary_size = vocabulary_size;
    config.embedding_rows = embedding_rows;
    config.capacity_entries = 4;
    config.head_layout = SL_PAIR_HEAD_SELECTED;
    config.head = head.data();
    config.embeddings = embeddings.data();
    config.token_ids = token_ids.data();
    config.payload_budget_bytes = std::uint64_t{4} * vocabulary_size * sizeof(float);
    // Illustrative identities, not file hashes. Integrators supply identities
    // for their weights, numerical contract and ordered vocabulary mapping.
    config.identity.weights[0] = 1;
    config.identity.numerics[0] = 2;
    config.identity.vocabulary[0] = 3;
    // Arrays above stay immutable and outlive both RAII handles below.
    auto cached = create(config);
    config.capacity_entries = 0;
    config.payload_budget_bytes = 0;
    auto uncached = create(config);

    std::cout << "Synthetic CPU rank-256 example; no model or generation benchmark.\n";
    using Logits = std::array<float, vocabulary_size>;
    const auto step = [&](const char* label, const Logits& base,
                          std::uint32_t expected_token, bool expected_hit) {
        sl_pair_result actual{}, reference{};
        Logits bias{}, reference_bias{};
        // Synchronous, serialized calls. Unsupported host rounding/FTZ/DAZ
        // modes return an error, which check() handles before consuming output.
        check(sl_pair_step(cached.get(), predecessor, base.data(), &actual, nullptr, bias.data()),
              "cached sl_pair_step");
        check(sl_pair_step(uncached.get(), predecessor, base.data(), &reference, nullptr,
                           reference_bias.data()), "uncached sl_pair_step");
        if (actual.token != reference.token || actual.vocabulary_row != reference.vocabulary_row
            || !same_bits(actual.score, reference.score))
            throw std::runtime_error("cached and uncached selection differ");
        for (std::size_t row = 0; row < bias.size(); ++row)
            if (!same_bits(bias[row], reference_bias[row]))
                throw std::runtime_error("cached and uncached bias bits differ");
        if (actual.token != expected_token || actual.cache_hit != (expected_hit ? 1U : 0U)
            || reference.cache_hit != 0)
            throw std::runtime_error("unexpected result for the synthetic fixture");
        std::cout << label << ": predecessor=" << predecessor << " token=" << actual.token
                  << " row=" << actual.vocabulary_row << " score=" << actual.score
                  << " cache=" << (actual.cache_hit ? "hit" : "miss")
                  << " selection/bias_equal_bits=true\n";
    };

    // The cached bias stays [0.75, -0.5, 1.0]. Fresh base logits still change
    // the winning global token: cached values are not cached decisions.
    step("first base", Logits{0.0F, 0.0F, 0.0F}, 11, false);
    step("changed base", Logits{1.0F, 0.0F, 0.0F}, 7, true);
    step("changed base again", Logits{0.0F, 2.0F, 0.0F}, 3, true);

    // Reset invalidates entries and retains cumulative statistics. It does
    // not permit changing borrowed arrays: new weights/mapping need a new handle.
    check(sl_pair_reset(cached.get(), nullptr), "sl_pair_reset");
    step("after reset", Logits{0.0F, 2.0F, 0.0F}, 3, false);
    print_stats("cached", cached);
    print_stats("uncached", uncached);
    // An integrating LLM engine still owns target verification and sampling.
    return 0;
} catch (const std::exception& error) {
    std::cerr << "pair example: " << error.what() << '\n';
    return 1;
}

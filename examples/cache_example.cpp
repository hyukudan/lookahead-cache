#include "strix_lookahead/cache.h"

#include <array>
#include <cstdio>
#include <cstring>

// In an engine adapter, this function would compute the exact FP32 bias using
// the engine's established numerical profile and ordered vocabulary mapping.
static void fill_example_bias(sl_write_lease& writer) {
    const std::array<float, 4> values{0.25F, -0.5F, 1.0F, 0.0F};
    std::memcpy(writer.values, values.data(), sizeof(values));
}

static sl_status acquire_or_compute(sl_cache* cache, uint32_t predecessor,
                                    sl_read_lease* result) {
    sl_status status = sl_cache_acquire(cache, predecessor, result);
    if (status != SL_MISS) return status;
    sl_write_lease writer{};
    status = sl_cache_reserve(cache, predecessor, &writer);
    if (status != SL_OK) return status;
    // This synchronous example fills every value before publishing. A failing
    // producer must call sl_cache_abort instead. No GPU work occurs here.
    fill_example_bias(writer);
    status = sl_cache_commit(cache, &writer);
    if (status != SL_OK) {
        (void)sl_cache_abort(cache, &writer);
        return status;
    }
    return sl_cache_acquire(cache, predecessor, result);
}

int main() {
    sl_cache_config config{};
    sl_cache_config_init(&config);
    config.capacity_entries = 2;
    config.vocabulary_size = 4;
    config.payload_budget_bytes = 2 * 4 * sizeof(float);
    // Illustrative identities only. Production adapters supply fingerprints of
    // the actual weights, arithmetic and ordered vocabulary.
    config.identity.weights[0] = 1;
    config.identity.numerics[0] = 2;
    config.identity.vocabulary[0] = 3;

    sl_cache* cache = nullptr;
    sl_status status = sl_cache_create(&config, &cache);
    if (status != SL_OK) {
        std::fprintf(stderr, "create: %s\n", sl_status_string(status));
        return 1;
    }
    for (unsigned round = 0; round < 2; ++round) {
        sl_read_lease reader{};
        status = acquire_or_compute(cache, 12345, &reader);
        if (status != SL_OK) {
            std::fprintf(stderr, "acquire: %s\n", sl_status_string(status));
            (void)sl_cache_destroy(cache);
            return 1;
        }
        std::printf("Round %u: %zu host FP32 values; first value = %.2f\n",
                    round + 1, reader.value_count, static_cast<double>(reader.values[0]));
        // The engine would consume these values without changing its original
        // addition, argmax, target verification or state-commit rules.
        status = sl_cache_release(cache, &reader);
        if (status != SL_OK) {
            std::fprintf(stderr, "release: %s\n", sl_status_string(status));
            (void)sl_cache_destroy(cache);
            return 1;
        }
    }
    sl_cache_stats stats{};
    status = sl_cache_get_stats(cache, &stats);
    if (status == SL_OK) {
        std::printf("commits=%llu, hits=%llu, misses=%llu\n",
                    static_cast<unsigned long long>(stats.commits),
                    static_cast<unsigned long long>(stats.hits),
                    static_cast<unsigned long long>(stats.misses));
    }
    const sl_status destroy_status = sl_cache_destroy(cache);
    return status == SL_OK && destroy_status == SL_OK ? 0 : 1;
}

#include "strix_lookahead/cache.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        throw Failure(#condition); \
    } \
} while (false)

#define EXPECT_STATUS(expression, expected) do { \
    const sl_status actual_status = (expression); \
    if (actual_status != (expected)) { \
        std::fprintf(stderr, "%s:%d: %s: got %s, expected %s\n", \
                     __FILE__, __LINE__, #expression, \
                     sl_status_string(actual_status), sl_status_string(expected)); \
        throw Failure(#expression); \
    } \
} while (false)

using Bits = std::array<std::uint32_t, 4>;
constexpr Bits unusual_bits{0x80000000U, 0x7fc12345U, 0x7fa12345U, 0xff800000U};
constexpr Bits ordinary_bits{0x3f800000U, 0x40000000U, 0xc0400000U, 0x00000001U};

sl_cache_config config(std::uint32_t entries = 2, std::uint32_t readers = 8) {
    sl_cache_config c{};
    sl_cache_config_init(&c);
    c.capacity_entries = entries;
    c.vocabulary_size = 4;
    c.max_read_leases = readers;
    c.identity.weights[0] = 11;
    c.identity.numerics[0] = 22;
    c.identity.vocabulary[0] = 33;
    return c;
}

sl_cache* create(const sl_cache_config& c) {
    sl_cache* result = nullptr;
    EXPECT_STATUS(sl_cache_create(&c, &result), SL_OK);
    CHECK(result != nullptr);
    return result;
}

void put(sl_cache* cache, std::uint32_t predecessor, const Bits& bits = ordinary_bits) {
    sl_write_lease writer{};
    EXPECT_STATUS(sl_cache_reserve(cache, predecessor, &writer), SL_OK);
    CHECK(writer.ticket != 0);
    CHECK(writer.values != nullptr);
    CHECK(writer.value_count == bits.size());
    // Copy object representations, never evaluate the NaN payloads as floats.
    std::memcpy(writer.values, bits.data(), sizeof(bits));
    EXPECT_STATUS(sl_cache_commit(cache, &writer), SL_OK);
}

void check_bytes(const sl_read_lease& reader, const Bits& bits) {
    CHECK(reader.value_count == bits.size());
    CHECK(std::memcmp(reader.values, bits.data(), sizeof(bits)) == 0);
}

sl_cache_stats stats(sl_cache* cache) {
    sl_cache_stats result{};
    EXPECT_STATUS(sl_cache_get_stats(cache, &result), SL_OK);
    return result;
}

void byte_round_trip_and_identity() {
    const auto c = config();
    auto* cache = create(c);
    sl_identity identity{};
    EXPECT_STATUS(sl_cache_get_identity(cache, &identity), SL_OK);
    CHECK(std::memcmp(&identity, &c.identity, sizeof(identity)) == 0);
    sl_read_lease reader{99, nullptr, 99};
    EXPECT_STATUS(sl_cache_acquire(cache, 400000U, &reader), SL_MISS);
    CHECK(reader.ticket == 0 && reader.values == nullptr && reader.value_count == 0);
    // Global token IDs are not indices into the reduced-vocabulary vector.
    put(cache, 400000U, unusual_bits);
    EXPECT_STATUS(sl_cache_acquire(cache, 400000U, &reader), SL_OK);
    check_bytes(reader, unusual_bits);
    EXPECT_STATUS(sl_cache_release(cache, &reader), SL_OK);
    CHECK(reader.ticket == 0 && reader.values == nullptr && reader.value_count == 0);
    const auto s = stats(cache);
    CHECK(s.hits == 1 && s.misses == 1 && s.commits == 1 && s.reservations == 1);
    CHECK(s.ready_entries == 1 && s.active_read_leases == 0);
    CHECK(s.payload_bytes == 2U * 4U * sizeof(float) && s.capacity_entries == 2);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void unpublished_fill_duplicate_and_abort() {
    auto* cache = create(config());
    sl_write_lease writer{}, duplicate{99, nullptr, 99};
    EXPECT_STATUS(sl_cache_reserve(cache, 10, &writer), SL_OK);
    CHECK(stats(cache).filling_entries == 1);
    sl_read_lease reader{99, nullptr, 99};
    EXPECT_STATUS(sl_cache_acquire(cache, 10, &reader), SL_BUSY);
    CHECK(reader.ticket == 0 && reader.values == nullptr && reader.value_count == 0);
    EXPECT_STATUS(sl_cache_reserve(cache, 10, &duplicate), SL_BUSY);
    CHECK(duplicate.ticket == 0 && duplicate.values == nullptr && duplicate.value_count == 0);
    EXPECT_STATUS(sl_cache_reset(cache, nullptr), SL_BUSY);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_BUSY);
    EXPECT_STATUS(sl_cache_abort(cache, &writer), SL_OK);
    CHECK(writer.ticket == 0 && writer.values == nullptr && writer.value_count == 0);
    EXPECT_STATUS(sl_cache_acquire(cache, 10, &reader), SL_MISS);
    put(cache, 10);
    EXPECT_STATUS(sl_cache_reserve(cache, 10, &duplicate), SL_PRESENT);
    CHECK(stats(cache).ready_entries == 1 && stats(cache).filling_entries == 0);
    CHECK(stats(cache).aborts == 1);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void readers_pin_slots_and_have_unique_tickets() {
    auto* cache = create(config(1, 2));
    put(cache, 1, unusual_bits);
    sl_read_lease first{}, second{}, excess{};
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &first), SL_OK);
    const auto copy_of_first = first;
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &second), SL_OK);
    CHECK(first.ticket != second.ticket);
    CHECK(first.values == second.values);
    CHECK(stats(cache).active_read_leases == 2);
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &excess), SL_BUSY);
    sl_write_lease writer{};
    EXPECT_STATUS(sl_cache_reserve(cache, 2, &writer), SL_BUSY);
    EXPECT_STATUS(sl_cache_release(cache, &first), SL_OK);
    auto stale = copy_of_first;
    EXPECT_STATUS(sl_cache_release(cache, &stale), SL_STALE_LEASE);
    CHECK(stats(cache).active_read_leases == 1);
    check_bytes(second, unusual_bits);
    EXPECT_STATUS(sl_cache_reset(cache, nullptr), SL_BUSY);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_BUSY);
    EXPECT_STATUS(sl_cache_release(cache, &second), SL_OK);
    EXPECT_STATUS(sl_cache_release(cache, &second), SL_STALE_LEASE);
    put(cache, 2);
    CHECK(stats(cache).evictions == 1);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void deterministic_lru_and_pinned_victim() {
    auto* cache = create(config(2));
    put(cache, 1, unusual_bits);
    put(cache, 2);
    sl_read_lease reader{};
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_OK);
    EXPECT_STATUS(sl_cache_release(cache, &reader), SL_OK);
    put(cache, 3);
    EXPECT_STATUS(sl_cache_acquire(cache, 2, &reader), SL_MISS);
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_OK);
    // Even an older pinned entry is excluded from the victim set.
    sl_read_lease temporary{};
    EXPECT_STATUS(sl_cache_acquire(cache, 3, &temporary), SL_OK);
    EXPECT_STATUS(sl_cache_release(cache, &temporary), SL_OK);
    sl_write_lease filling{};
    EXPECT_STATUS(sl_cache_reserve(cache, 4, &filling), SL_OK);
    check_bytes(reader, unusual_bits);
    EXPECT_STATUS(sl_cache_acquire(cache, 3, &temporary), SL_MISS);
    sl_write_lease no_slot{};
    EXPECT_STATUS(sl_cache_reserve(cache, 5, &no_slot), SL_BUSY);
    std::memcpy(filling.values, ordinary_bits.data(), sizeof(ordinary_bits));
    EXPECT_STATUS(sl_cache_commit(cache, &filling), SL_OK);
    EXPECT_STATUS(sl_cache_release(cache, &reader), SL_OK);
    CHECK(stats(cache).evictions == 2);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void abort_does_not_restore_evicted_data() {
    auto* cache = create(config(1));
    put(cache, 1);
    sl_write_lease writer{};
    EXPECT_STATUS(sl_cache_reserve(cache, 2, &writer), SL_OK);
    EXPECT_STATUS(sl_cache_abort(cache, &writer), SL_OK);
    sl_read_lease reader{};
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_MISS);
    EXPECT_STATUS(sl_cache_acquire(cache, 2, &reader), SL_MISS);
    const auto s = stats(cache);
    CHECK(s.ready_entries == 0 && s.filling_entries == 0);
    CHECK(s.evictions == 1 && s.aborts == 1);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void stale_and_foreign_handles_do_not_change_live_leases() {
    auto* first_cache = create(config(1));
    auto* other_cache = create(config(1));
    sl_write_lease writer{}, other_writer{};
    EXPECT_STATUS(sl_cache_reserve(first_cache, 1, &writer), SL_OK);
    EXPECT_STATUS(sl_cache_reserve(other_cache, 1, &other_writer), SL_OK);
    auto foreign = writer;
    EXPECT_STATUS(sl_cache_commit(other_cache, &foreign), SL_STALE_LEASE);
    CHECK(stats(other_cache).filling_entries == 1);
    auto altered = writer;
    ++altered.values;
    EXPECT_STATUS(sl_cache_commit(first_cache, &altered), SL_STALE_LEASE);
    altered = writer;
    --altered.value_count;
    EXPECT_STATUS(sl_cache_abort(first_cache, &altered), SL_STALE_LEASE);
    auto stale_writer = writer;
    std::memcpy(writer.values, ordinary_bits.data(), sizeof(ordinary_bits));
    EXPECT_STATUS(sl_cache_commit(first_cache, &writer), SL_OK);
    EXPECT_STATUS(sl_cache_commit(first_cache, &writer), SL_STALE_LEASE);
    EXPECT_STATUS(sl_cache_commit(first_cache, &stale_writer), SL_STALE_LEASE);
    EXPECT_STATUS(sl_cache_abort(other_cache, &other_writer), SL_OK);
    EXPECT_STATUS(sl_cache_abort(other_cache, &other_writer), SL_STALE_LEASE);
    put(other_cache, 1);
    sl_read_lease reader{}, other_reader{};
    EXPECT_STATUS(sl_cache_acquire(first_cache, 1, &reader), SL_OK);
    EXPECT_STATUS(sl_cache_acquire(other_cache, 1, &other_reader), SL_OK);
    EXPECT_STATUS(sl_cache_commit(first_cache, &stale_writer), SL_STALE_LEASE);
    EXPECT_STATUS(sl_cache_abort(first_cache, &stale_writer), SL_STALE_LEASE);
    check_bytes(reader, ordinary_bits);
    auto foreign_reader = reader;
    EXPECT_STATUS(sl_cache_release(other_cache, &foreign_reader), SL_STALE_LEASE);
    auto altered_reader = reader;
    ++altered_reader.values;
    EXPECT_STATUS(sl_cache_release(first_cache, &altered_reader), SL_STALE_LEASE);
    altered_reader = reader;
    --altered_reader.value_count;
    EXPECT_STATUS(sl_cache_release(first_cache, &altered_reader), SL_STALE_LEASE);
    altered_reader = reader;
    altered_reader.ticket = std::numeric_limits<std::uint64_t>::max();
    EXPECT_STATUS(sl_cache_release(first_cache, &altered_reader), SL_STALE_LEASE);
    CHECK(stats(first_cache).active_read_leases == 1);
    CHECK(stats(other_cache).active_read_leases == 1);
    EXPECT_STATUS(sl_cache_release(first_cache, &reader), SL_OK);
    EXPECT_STATUS(sl_cache_release(other_cache, &other_reader), SL_OK);
    // The same storage can be recycled, but an old write ticket must never
    // publish the new producer's unfinished values.
    EXPECT_STATUS(sl_cache_reserve(first_cache, 2, &writer), SL_OK);
    EXPECT_STATUS(sl_cache_commit(first_cache, &stale_writer), SL_STALE_LEASE);
    CHECK(stats(first_cache).filling_entries == 1);
    std::memcpy(writer.values, unusual_bits.data(), sizeof(unusual_bits));
    EXPECT_STATUS(sl_cache_commit(first_cache, &writer), SL_OK);
    EXPECT_STATUS(sl_cache_acquire(first_cache, 2, &reader), SL_OK);
    check_bytes(reader, unusual_bits);
    EXPECT_STATUS(sl_cache_release(first_cache, &reader), SL_OK);
    EXPECT_STATUS(sl_cache_destroy(first_cache), SL_OK);
    EXPECT_STATUS(sl_cache_destroy(other_cache), SL_OK);
}

void reset_changes_identity_and_keeps_counters_without_reusing_tickets() {
    auto c = config(1);
    auto* cache = create(c);
    put(cache, 1);
    sl_read_lease reader{};
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_OK);
    auto old_reader = reader;
    sl_identity replacement = c.identity;
    replacement.vocabulary[31] = 44;
    EXPECT_STATUS(sl_cache_reset(cache, &replacement), SL_BUSY);
    sl_identity identity{};
    EXPECT_STATUS(sl_cache_get_identity(cache, &identity), SL_OK);
    CHECK(std::memcmp(&identity, &c.identity, sizeof(identity)) == 0);
    EXPECT_STATUS(sl_cache_release(cache, &reader), SL_OK);
    const auto before = stats(cache);
    EXPECT_STATUS(sl_cache_reset(cache, &replacement), SL_OK);
    const auto after = stats(cache);
    CHECK(after.hits == before.hits && after.commits == before.commits);
    CHECK(after.misses == before.misses && after.busy == before.busy);
    CHECK(after.reservations == before.reservations && after.aborts == before.aborts);
    CHECK(after.evictions == before.evictions && after.payload_bytes == before.payload_bytes);
    CHECK(after.capacity_entries == before.capacity_entries);
    CHECK(after.resets == before.resets + 1 && after.ready_entries == 0);
    CHECK(after.filling_entries == 0 && after.active_read_leases == 0);
    EXPECT_STATUS(sl_cache_get_identity(cache, &identity), SL_OK);
    CHECK(std::memcmp(&identity, &replacement, sizeof(identity)) == 0);
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_MISS);
    put(cache, 1, unusual_bits);
    EXPECT_STATUS(sl_cache_acquire(cache, 1, &reader), SL_OK);
    CHECK(reader.ticket != old_reader.ticket);
    EXPECT_STATUS(sl_cache_release(cache, &old_reader), SL_STALE_LEASE);
    check_bytes(reader, unusual_bits);
    CHECK(stats(cache).active_read_leases == 1);
    EXPECT_STATUS(sl_cache_release(cache, &reader), SL_OK);
    EXPECT_STATUS(sl_cache_reset(cache, nullptr), SL_OK);
    EXPECT_STATUS(sl_cache_get_identity(cache, &identity), SL_OK);
    CHECK(std::memcmp(&identity, &replacement, sizeof(identity)) == 0);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

void invalid_configurations_and_arguments() {
    sl_cache_config defaults{};
    sl_cache_config_init(&defaults);
    CHECK(defaults.struct_size == sizeof(defaults) && defaults.abi_version == SL_ABI_VERSION);
    CHECK(defaults.capacity_entries == 16 && defaults.max_read_leases == 64);
    CHECK(defaults.vocabulary_size == 0 && defaults.reserved == 0);
    sl_cache* result = nullptr;
    EXPECT_STATUS(sl_cache_create(nullptr, &result), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_create(&defaults, &result), SL_INVALID_ARGUMENT);
    auto c = config();
    EXPECT_STATUS(sl_cache_create(&c, nullptr), SL_INVALID_ARGUMENT);
    for (unsigned scenario = 0; scenario < 8; ++scenario) {
        c = config();
        switch (scenario) {
        case 0: c.struct_size = 0; break;
        case 1: c.struct_size += 8; break;
        case 2: c.abi_version += 1; break;
        case 3: c.capacity_entries = 0; break;
        case 4: c.vocabulary_size = 0; break;
        case 5: c.max_read_leases = 0; break;
        case 6: c.reserved = 1; break;
        case 7: c.payload_budget_bytes = 2U * 4U * sizeof(float) - 1U; break;
        default: break;
        }
        EXPECT_STATUS(sl_cache_create(&c, &result), SL_INVALID_ARGUMENT);
        CHECK(result == nullptr);
    }
    c = config();
    c.capacity_entries = std::numeric_limits<std::uint32_t>::max();
    c.vocabulary_size = std::numeric_limits<std::uint32_t>::max();
    EXPECT_STATUS(sl_cache_create(&c, &result), SL_OVERFLOW);
    CHECK(result == nullptr);
    // On a 64-bit host this payload fits size_t, but its slot metadata does
    // not. Smaller hosts reject the payload multiplication before allocation.
    c.vocabulary_size = 1U << 30U;
    EXPECT_STATUS(sl_cache_create(&c, &result), SL_OVERFLOW);
    CHECK(result == nullptr);
    c = config();
    c.payload_budget_bytes = 2U * 4U * sizeof(float);
    auto* cache = create(c);
    sl_read_lease reader{};
    sl_write_lease writer{};
    sl_cache_stats s{};
    sl_identity identity{};
    EXPECT_STATUS(sl_cache_acquire(nullptr, 0, &reader), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_acquire(cache, 0, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_reserve(nullptr, 0, &writer), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_reserve(cache, 0, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_release(nullptr, &reader), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_release(cache, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_commit(nullptr, &writer), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_abort(cache, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_get_stats(nullptr, &s), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_get_stats(cache, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_get_identity(nullptr, &identity), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_get_identity(cache, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_reset(nullptr, nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_destroy(nullptr), SL_INVALID_ARGUMENT);
    EXPECT_STATUS(sl_cache_destroy(cache), SL_OK);
}

}  // namespace

int main() {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"byte round trip and identity", byte_round_trip_and_identity},
        {"unpublished fill, duplicate and abort", unpublished_fill_duplicate_and_abort},
        {"multiple readers and pinned entries", readers_pin_slots_and_have_unique_tickets},
        {"deterministic LRU and pinned victim", deterministic_lru_and_pinned_victim},
        {"aborted replacement visibility", abort_does_not_restore_evicted_data},
        {"stale and foreign handles", stale_and_foreign_handles_do_not_change_live_leases},
        {"identity reset and ticket lifetime", reset_changes_identity_and_keeps_counters_without_reusing_tickets},
        {"invalid configurations and arguments", invalid_configurations_and_arguments},
    };
    unsigned completed = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            ++completed;
            std::printf("PASS %s\n", test.name);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL %s: %s\n", test.name, error.what());
            return 1;
        }
    }
    std::printf("%u cache scenarios passed\n", completed);
    return 0;
}

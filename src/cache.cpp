#include "strix_lookahead/cache.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The reference backend requires IEEE 754 binary32 float storage.");
constexpr std::uint32_t no_slot = std::numeric_limits<std::uint32_t>::max();

enum class slot_state : std::uint8_t { empty, filling, ready };

struct cache_slot {
    slot_state state = slot_state::empty;
    std::uint32_t predecessor = 0;
    std::uint32_t readers = 0;
    std::uint32_t previous = no_slot;
    std::uint32_t next = no_slot;
    std::uint64_t writer_ticket = 0;
};

struct reader_ticket {
    std::uint64_t ticket = 0;
    std::uint32_t slot = no_slot;
};

bool multiply_size(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool add_size(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (b > std::numeric_limits<std::size_t>::max() - a) return false;
    result = a + b;
    return true;
}

void increment(std::uint64_t& value) noexcept {
    if (value != std::numeric_limits<std::uint64_t>::max()) ++value;
}

}  // namespace

struct sl_cache {
    sl_identity identity{};
    std::uint32_t capacity = 0;
    std::uint32_t vocabulary_size = 0;
    std::uint32_t reader_capacity = 0;
    // Tickets are shared by readers and writers. Zero is never issued, and
    // reset does not rewind this sequence. Exhaustion is an explicit failure.
    std::uint64_t last_ticket = 0;
    std::uint32_t oldest = no_slot;
    std::uint32_t newest = no_slot;
    sl_cache_stats stats{};
    std::unique_ptr<float[]> payload;
    std::unique_ptr<cache_slot[]> slots;
    std::unique_ptr<reader_ticket[]> readers;
};

namespace {

float* slot_values(sl_cache& cache, std::uint32_t slot) noexcept {
    return cache.payload.get() + static_cast<std::size_t>(slot) * cache.vocabulary_size;
}

bool issue_ticket(sl_cache& cache, std::uint64_t& ticket) noexcept {
    if (cache.last_ticket == std::numeric_limits<std::uint64_t>::max()) return false;
    ticket = ++cache.last_ticket;
    return true;
}

sl_status busy(sl_cache& cache) noexcept {
    increment(cache.stats.busy);
    return SL_BUSY;
}

// Only committed slots are linked. Their order records successful acquire
// and commit calls, so there is no timestamp to overflow or renormalize.
void unlink(sl_cache& cache, std::uint32_t index) noexcept {
    auto& slot = cache.slots[index];
    if (slot.previous == no_slot) cache.oldest = slot.next;
    else cache.slots[slot.previous].next = slot.next;
    if (slot.next == no_slot) cache.newest = slot.previous;
    else cache.slots[slot.next].previous = slot.previous;
    slot.previous = no_slot;
    slot.next = no_slot;
}

void append_newest(sl_cache& cache, std::uint32_t index) noexcept {
    auto& slot = cache.slots[index];
    slot.previous = cache.newest;
    slot.next = no_slot;
    if (cache.newest == no_slot) cache.oldest = index;
    else cache.slots[cache.newest].next = index;
    cache.newest = index;
}

void touch(sl_cache& cache, std::uint32_t index) noexcept {
    if (cache.newest == index) return;
    unlink(cache, index);
    append_newest(cache, index);
}

cache_slot* find_writer(sl_cache& cache, const sl_write_lease& lease,
                        std::uint32_t& index) noexcept {
    if (lease.ticket == 0 || lease.value_count != cache.vocabulary_size) return nullptr;
    for (std::uint32_t i = 0; i < cache.capacity; ++i) {
        auto& slot = cache.slots[i];
        if (slot.state == slot_state::filling && slot.writer_ticket == lease.ticket) {
            if (lease.values != slot_values(cache, i)) return nullptr;
            index = i;
            return &slot;
        }
    }
    return nullptr;
}

}  // namespace

extern "C" {

void sl_cache_config_init(sl_cache_config* config) {
    if (config == nullptr) return;
    *config = {};
    config->struct_size = static_cast<std::uint32_t>(sizeof(sl_cache_config));
    config->abi_version = SL_ABI_VERSION;
    config->capacity_entries = 16;
    config->max_read_leases = 64;
}

const char* sl_status_string(sl_status status) {
    switch (status) {
        case SL_OK: return "ok";
        case SL_MISS: return "miss";
        case SL_BUSY: return "busy";
        case SL_PRESENT: return "present";
        case SL_INVALID_ARGUMENT: return "invalid argument";
        case SL_STALE_LEASE: return "stale lease";
        case SL_ALLOCATION_FAILED: return "allocation failed";
        case SL_OVERFLOW: return "overflow";
    }
    return "unknown status";
}

sl_status sl_cache_create(const sl_cache_config* config, sl_cache** out) {
    if (out != nullptr) *out = nullptr;
    if (config == nullptr || out == nullptr ||
        config->struct_size != sizeof(sl_cache_config) ||
        config->abi_version != SL_ABI_VERSION || config->reserved != 0 ||
        config->capacity_entries == 0 || config->vocabulary_size == 0 ||
        config->max_read_leases == 0) return SL_INVALID_ARGUMENT;

    std::size_t elements = 0;
    std::size_t payload_bytes = 0;
    std::size_t slots_bytes = 0;
    std::size_t readers_bytes = 0;
    std::size_t total_bytes = 0;
    if (!multiply_size(config->capacity_entries, config->vocabulary_size, elements) ||
        !multiply_size(elements, sizeof(float), payload_bytes) ||
        !multiply_size(config->capacity_entries, sizeof(cache_slot), slots_bytes) ||
        !multiply_size(config->max_read_leases, sizeof(reader_ticket), readers_bytes) ||
        !add_size(payload_bytes, slots_bytes, total_bytes) ||
        !add_size(total_bytes, readers_bytes, total_bytes) ||
        !add_size(total_bytes, sizeof(sl_cache), total_bytes)) return SL_OVERFLOW;
    // The public budget deliberately covers only payload, not allocator or
    // metadata overhead. All three arrays are allocated before publication.
    if (config->payload_budget_bytes != 0 &&
        payload_bytes > config->payload_budget_bytes) return SL_INVALID_ARGUMENT;

    try {
        auto cache = std::make_unique<sl_cache>();
        cache->identity = config->identity;
        cache->capacity = config->capacity_entries;
        cache->vocabulary_size = config->vocabulary_size;
        cache->reader_capacity = config->max_read_leases;
        cache->payload.reset(new float[elements]);
        cache->slots.reset(new cache_slot[cache->capacity]);
        cache->readers.reset(new reader_ticket[cache->reader_capacity]);
        cache->stats.payload_bytes = static_cast<std::uint64_t>(payload_bytes);
        cache->stats.capacity_entries = cache->capacity;
        *out = cache.release();
        return SL_OK;
    } catch (const std::bad_alloc&) {
        return SL_ALLOCATION_FAILED;
    } catch (...) {
        // No exception is allowed across the C ABI, including failures from
        // a caller-provided replacement allocation function.
        return SL_ALLOCATION_FAILED;
    }
}

sl_status sl_cache_get_identity(const sl_cache* cache, sl_identity* out) {
    if (cache == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    *out = cache->identity;
    return SL_OK;
}

sl_status sl_cache_get_stats(const sl_cache* cache, sl_cache_stats* out) {
    if (cache == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    *out = cache->stats;
    return SL_OK;
}

sl_status sl_cache_acquire(sl_cache* cache, std::uint32_t predecessor, sl_read_lease* out) {
    if (out != nullptr) *out = {};
    if (cache == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    for (std::uint32_t i = 0; i < cache->capacity; ++i) {
        auto& slot = cache->slots[i];
        if (slot.state == slot_state::empty || slot.predecessor != predecessor) continue;
        if (slot.state == slot_state::filling) return busy(*cache);
        reader_ticket* reader = nullptr;
        for (std::uint32_t j = 0; j < cache->reader_capacity; ++j) {
            if (cache->readers[j].ticket == 0) {
                reader = &cache->readers[j];
                break;
            }
        }
        if (reader == nullptr) return busy(*cache);
        std::uint64_t ticket = 0;
        if (!issue_ticket(*cache, ticket)) return SL_OVERFLOW;
        reader->ticket = ticket;
        reader->slot = i;
        ++slot.readers;
        ++cache->stats.active_read_leases;
        increment(cache->stats.hits);
        touch(*cache, i);
        *out = {ticket, slot_values(*cache, i), cache->vocabulary_size};
        return SL_OK;
    }
    increment(cache->stats.misses);
    return SL_MISS;
}

sl_status sl_cache_release(sl_cache* cache, sl_read_lease* lease) {
    if (cache == nullptr || lease == nullptr) return SL_INVALID_ARGUMENT;
    if (lease->ticket == 0 || lease->value_count != cache->vocabulary_size) return SL_STALE_LEASE;
    for (std::uint32_t i = 0; i < cache->reader_capacity; ++i) {
        auto& reader = cache->readers[i];
        if (reader.ticket != lease->ticket) continue;
        if (lease->values != slot_values(*cache, reader.slot)) return SL_STALE_LEASE;
        --cache->slots[reader.slot].readers;
        --cache->stats.active_read_leases;
        reader = {};
        *lease = {};
        return SL_OK;
    }
    return SL_STALE_LEASE;
}

sl_status sl_cache_reserve(sl_cache* cache, std::uint32_t predecessor, sl_write_lease* out) {
    if (out != nullptr) *out = {};
    if (cache == nullptr || out == nullptr) return SL_INVALID_ARGUMENT;
    std::uint32_t selected = no_slot;
    for (std::uint32_t i = 0; i < cache->capacity; ++i) {
        const auto& slot = cache->slots[i];
        if (slot.state == slot_state::empty) {
            if (selected == no_slot) selected = i;
        } else if (slot.predecessor == predecessor) {
            return slot.state == slot_state::ready ? SL_PRESENT : busy(*cache);
        }
    }
    if (selected == no_slot) {
        for (auto i = cache->oldest; i != no_slot; i = cache->slots[i].next) {
            if (cache->slots[i].readers == 0) {
                selected = i;
                break;
            }
        }
    }
    if (selected == no_slot) return busy(*cache);
    std::uint64_t ticket = 0;
    if (!issue_ticket(*cache, ticket)) return SL_OVERFLOW;
    auto& slot = cache->slots[selected];
    if (slot.state == slot_state::ready) {
        unlink(*cache, selected);
        --cache->stats.ready_entries;
        increment(cache->stats.evictions);
    }
    slot = {};
    slot.state = slot_state::filling;
    slot.predecessor = predecessor;
    slot.writer_ticket = ticket;
    ++cache->stats.filling_entries;
    increment(cache->stats.reservations);
    *out = {ticket, slot_values(*cache, selected), cache->vocabulary_size};
    return SL_OK;
}

sl_status sl_cache_commit(sl_cache* cache, sl_write_lease* lease) {
    if (cache == nullptr || lease == nullptr) return SL_INVALID_ARGUMENT;
    std::uint32_t index = no_slot;
    auto* slot = find_writer(*cache, *lease, index);
    if (slot == nullptr) return SL_STALE_LEASE;
    slot->state = slot_state::ready;
    slot->writer_ticket = 0;
    append_newest(*cache, index);
    --cache->stats.filling_entries;
    ++cache->stats.ready_entries;
    increment(cache->stats.commits);
    *lease = {};
    return SL_OK;
}

sl_status sl_cache_abort(sl_cache* cache, sl_write_lease* lease) {
    if (cache == nullptr || lease == nullptr) return SL_INVALID_ARGUMENT;
    std::uint32_t index = no_slot;
    auto* slot = find_writer(*cache, *lease, index);
    if (slot == nullptr) return SL_STALE_LEASE;
    *slot = {};
    --cache->stats.filling_entries;
    increment(cache->stats.aborts);
    *lease = {};
    return SL_OK;
}

sl_status sl_cache_reset(sl_cache* cache, const sl_identity* identity) {
    if (cache == nullptr) return SL_INVALID_ARGUMENT;
    if (cache->stats.active_read_leases != 0 || cache->stats.filling_entries != 0) return busy(*cache);
    for (std::uint32_t i = 0; i < cache->capacity; ++i) cache->slots[i] = {};
    for (std::uint32_t i = 0; i < cache->reader_capacity; ++i) cache->readers[i] = {};
    cache->oldest = no_slot;
    cache->newest = no_slot;
    cache->stats.ready_entries = 0;
    if (identity != nullptr) cache->identity = *identity;
    increment(cache->stats.resets);
    return SL_OK;
}

sl_status sl_cache_destroy(sl_cache* cache) {
    if (cache == nullptr) return SL_INVALID_ARGUMENT;
    if (cache->stats.active_read_leases != 0 || cache->stats.filling_entries != 0) return busy(*cache);
    delete cache;
    return SL_OK;
}

}  // extern "C"

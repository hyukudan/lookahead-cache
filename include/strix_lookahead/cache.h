#ifndef STRIX_LOOKAHEAD_CACHE_H
#define STRIX_LOOKAHEAD_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SL_ABI_VERSION 1u

typedef struct sl_cache sl_cache;

typedef enum sl_status {
    SL_OK = 0,
    SL_MISS = 1,
    SL_BUSY = 2,
    SL_PRESENT = 3,
    SL_INVALID_ARGUMENT = 4,
    SL_STALE_LEASE = 5,
    SL_ALLOCATION_FAILED = 6,
    SL_OVERFLOW = 7,
    SL_DEVICE_ERROR = 8,
    SL_UNSUPPORTED = 9,
    SL_INVALID_STATE = 10
} sl_status;

/* Opaque caller-supplied identities. vocabulary identifies the ORDERED mapping.
 * The library does not compute/check hashes against external model files. */
typedef struct sl_identity {
    uint8_t weights[32];
    uint8_t numerics[32];
    uint8_t vocabulary[32];
} sl_identity;

typedef struct sl_cache_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t capacity_entries;
    uint32_t vocabulary_size;
    uint32_t max_read_leases;
    uint32_t reserved;
    /* 0 means no additional payload budget; metadata is excluded. */
    uint64_t payload_budget_bytes;
    sl_identity identity;
} sl_cache_config;

/* A lease belongs to one cache. Treat its fields as immutable. Copies share
 * one ticket and MUST NOT be released/committed independently. Host pointers
 * only in this reference backend. All calls are externally serialized.
 * Never reuse an acquire/reserve output object while it holds a live lease.
 * Release/commit/abort clear the supplied object only on success; a failure
 * leaves it unchanged. All lease copies become invalid when their cache is
 * destroyed, even if a later cache allocation reuses the same address. */
typedef struct sl_read_lease {
    uint64_t ticket;
    const float* values;
    size_t value_count;
} sl_read_lease;

typedef struct sl_write_lease {
    uint64_t ticket;
    float* values;
    size_t value_count;
} sl_write_lease;

typedef struct sl_cache_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t busy;
    uint64_t reservations;
    uint64_t commits;
    uint64_t aborts;
    uint64_t evictions;
    uint64_t resets;
    uint64_t payload_bytes;
    uint32_t ready_entries;
    uint32_t filling_entries;
    uint32_t active_read_leases;
    uint32_t capacity_entries;
} sl_cache_stats;

/* Defaults: 16 entries, 64 read leases; caller sets vocabulary_size. */
void sl_cache_config_init(sl_cache_config* config);
const char* sl_status_string(sl_status status);
sl_status sl_cache_create(const sl_cache_config* config, sl_cache** out);
sl_status sl_cache_get_identity(const sl_cache* cache, sl_identity* out);
sl_status sl_cache_get_stats(const sl_cache* cache, sl_cache_stats* out);

/* Acquire pins a committed slot until release. A filling key returns BUSY.
 * Success overwrites out. Failure zeroes out when it is non-null. */
sl_status sl_cache_acquire(sl_cache* cache, uint32_t predecessor, sl_read_lease* out);
sl_status sl_cache_release(sl_cache* cache, sl_read_lease* lease);

/* Reserve selects an empty slot or the least-recently-used unpinned slot.
 * A victim is evicted immediately. A committed duplicate returns PRESENT;
 * a filling duplicate or exhausted slots return BUSY. The caller fills ALL
 * values before commit. The library never computes or transforms values.
 * Success overwrites out. Failure zeroes out when it is non-null. */
sl_status sl_cache_reserve(sl_cache* cache, uint32_t predecessor, sl_write_lease* out);
sl_status sl_cache_commit(sl_cache* cache, sl_write_lease* lease);
sl_status sl_cache_abort(sl_cache* cache, sl_write_lease* lease);

/* Reset/destroy refuse while any reader/writer lease is active. Reset keeps
 * cumulative counters and dimensions; non-null identity replaces identity.
 * Use a new cache when dimensions change. A destroyed pointer is invalid. */
sl_status sl_cache_reset(sl_cache* cache, const sl_identity* identity);
sl_status sl_cache_destroy(sl_cache* cache);

#ifdef __cplusplus
}
#endif
#endif

#include "strix_lookahead/cache.h"
#include "strix_lookahead/pair.h"
#include "strix_lookahead/selection.h"
#include "strix_lookahead/device.h"

#include <stdio.h>
#include <string.h>

/* This file must be compiled as C, then linked with the C++ library. */
int main(void) {
    sl_cache_config config;
    sl_cache* cache = NULL;
    sl_write_lease writer = {0};
    sl_read_lease reader = {0};
    const uint32_t bytes[2] = {0x80000000u, 0x7fc01234u};
    int result = 1;

    sl_cache_config_init(&config);
    config.capacity_entries = 1;
    config.vocabulary_size = 2;
    config.max_read_leases = 1;
    config.identity.weights[0] = 1;
    config.identity.numerics[0] = 2;
    config.identity.vocabulary[0] = 3;
    if (sl_cache_create(&config, &cache) != SL_OK) return 1;
    if (sl_cache_acquire(cache, 17, &reader) != SL_MISS) goto cleanup;
    if (sl_cache_reserve(cache, 17, &writer) != SL_OK) goto cleanup;
    memcpy(writer.values, bytes, sizeof(bytes));
    if (sl_cache_commit(cache, &writer) != SL_OK) goto cleanup;
    if (sl_cache_acquire(cache, 17, &reader) != SL_OK) goto cleanup;
    if (reader.value_count != 2 || memcmp(reader.values, bytes, sizeof(bytes)) != 0) goto cleanup;
    if (sl_cache_release(cache, &reader) != SL_OK) goto cleanup;
    result = 0;
cleanup:
    if (reader.ticket != 0 && sl_cache_release(cache, &reader) != SL_OK) result = 1;
    if (writer.ticket != 0 && sl_cache_abort(cache, &writer) != SL_OK) result = 1;
    if (sl_cache_destroy(cache) != SL_OK) result = 1;
    if (result == 0) puts("C ABI smoke passed");
    return result;
}

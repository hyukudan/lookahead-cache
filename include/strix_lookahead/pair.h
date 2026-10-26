#ifndef STRIX_LOOKAHEAD_PAIR_H
#define STRIX_LOOKAHEAD_PAIR_H

#include "strix_lookahead/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SL_PAIR_RANK 256u

typedef struct sl_pair_engine sl_pair_engine;

typedef enum sl_pair_head_layout {
    /* head[embedding_rows][256], indexed by global candidate token ID. */
    SL_PAIR_HEAD_GLOBAL = 0,
    /* head[vocabulary_size][256], in the ordered candidate mapping. */
    SL_PAIR_HEAD_SELECTED = 1
} sl_pair_head_layout;

typedef struct sl_pair_config {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t vocabulary_size;
    uint32_t embedding_rows;
    /* Zero selects the uncached reference path. */
    uint32_t capacity_entries;
    uint32_t head_layout;
    uint32_t reserved[2];
    /* IEEE binary16 bit patterns, row-major; borrowed until destroy. */
    const uint16_t* head;
    const uint16_t* embeddings;
    /* Ordered unique global IDs; null means IDs 0..vocabulary_size-1. */
    const uint32_t* token_ids;
    /* Zero means no extra payload cap; this cap excludes metadata. */
    uint64_t payload_budget_bytes;
    sl_identity identity;
} sl_pair_config;

typedef struct sl_pair_result {
    uint32_t token;
    uint32_t vocabulary_row;
    float score;
    uint32_t cache_hit;
} sl_pair_result;

typedef struct sl_pair_stats {
    uint64_t steps;
    uint64_t hits;
    uint64_t misses;
    uint64_t bias_rows_computed;
    uint64_t resets;
    uint64_t payload_bytes;
    uint32_t capacity_entries;
    uint32_t vocabulary_size;
} sl_pair_stats;

/* CPU reference for an ordered, fixed candidate vocabulary. Calls require
 * external serialization. Inputs are borrowed and immutable for the instance's
 * lifetime. No call after create allocates. This is a greedy proposal component;
 * the inference engine still owns target verification and sampling correctness. */
void sl_pair_config_init(sl_pair_config* config);
sl_status sl_pair_create(const sl_pair_config* config, sl_pair_engine** out);

/* Computes score[j] = base_logits[j] + dot(head[j], embeddings[predecessor]).
 * The rank-256 dot has 32 lanes: lane l sums terms 8*l..8*l+7 in ascending
 * order from +0, then XOR butterfly distances 16,8,4,2,1. Multiplications and
 * additions round separately to FP32. Argmax prefers the lower GLOBAL ID on
 * ties, and ignores NaNs. All -infinity values still have a valid winner.
 *
 * out is required. markov_out[256] and bias_out[vocabulary_size] are optional
 * diagnostic buffers; both are written on hits as well as misses. Buffers must
 * not overlap each other or the borrowed model data. A valid step with all-NaN
 * scores returns SL_INVALID_STATE, token/row UINT32_MAX and score -infinity;
 * its bias computation and cache publication still complete. An out-of-range
 * predecessor returns SL_INVALID_ARGUMENT without touching cache/statistics.
 * Round-to-nearest and gradual underflow are required. A different host rounding
 * mode, or x86 FTZ/DAZ mode, returns SL_INVALID_STATE without changing the cache
 * or statistics. On other architectures the caller ensures gradual underflow.
 * Output is ready when this synchronous reference call returns. */
sl_status sl_pair_step(sl_pair_engine* engine, uint32_t predecessor,
                       const float* base_logits, sl_pair_result* out,
                       float* markov_out, float* bias_out);

/* Invalidate all cache entries without changing dimensions, pointers or IDs.
 * A non-null identity replaces the opaque identity. Cumulative statistics are
 * retained. Construct a new engine when weights or the ordered mapping change. */
sl_status sl_pair_reset(sl_pair_engine* engine, const sl_identity* identity);
sl_status sl_pair_get_identity(const sl_pair_engine* engine, sl_identity* out);
sl_status sl_pair_get_stats(const sl_pair_engine* engine, sl_pair_stats* out);
sl_status sl_pair_destroy(sl_pair_engine* engine);

/* Deterministic direct-map slot shared with device implementations. Capacity
 * zero returns UINT32_MAX. This mapping is not a cryptographic hash. */
uint32_t sl_pair_slot_for(uint32_t predecessor, uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif

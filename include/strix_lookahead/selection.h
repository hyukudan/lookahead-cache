#ifndef STRIX_LOOKAHEAD_SELECTION_H
#define STRIX_LOOKAHEAD_SELECTION_H

#include "strix_lookahead/cache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Optional host reference for certified sparse argmax of FP32 base + bias.
 * This component does not execute on a GPU or compute a learned correction.
 *
 * Preparation copies and validates its inputs. A vocabulary has V unique global
 * token IDs. A base snapshot owns V logits and K unique candidate indices; its
 * tail bounds are derived from ALL excluded logits, never supplied by a caller.
 * A bias snapshot owns V values and their extrema. The original input arrays
 * may be changed or freed after creation without invalidating a certificate.
 *
 * Vocabulary validation is expected O(V); base preparation is O(V + K), bias
 * preparation O(V). Storage is O(V + K). These costs are NOT an inference
 * speedup. A caller must obtain reuse or integrate equivalent summaries into
 * an existing producer before an O(K) certificate can save end-to-end work.
 */
typedef struct sl_selection_vocabulary sl_selection_vocabulary;
typedef struct sl_selection_base sl_selection_base;
typedef struct sl_selection_bias sl_selection_bias;

typedef struct sl_selection_result {
    uint32_t token_id;
    uint32_t vocabulary_index;
    float score;
} sl_selection_result;

/* All output objects must be writable and may not alias inputs. Creation clears
 * *out on failure. IDs are copied; NULL global_ids selects dense IDs [0, V).
 * V must be nonzero and fit uint32_t. Duplicate global IDs are rejected.
 */
sl_status sl_selection_vocabulary_create(const uint32_t* global_ids,
                                         size_t value_count,
                                         sl_selection_vocabulary** out);
sl_status sl_selection_vocabulary_destroy(sl_selection_vocabulary* vocabulary);

/* Base and bias must be created from the SAME vocabulary handle. This gives an
 * exact, constant-time mapping-identity check; separately created handles with
 * identical IDs are conservatively treated as different vocabularies.
 * Snapshots retain shared ownership of the immutable mapping, so the vocabulary
 * handle may be destroyed before its snapshots.
 *
 * Nonfinite inputs are rejected. Candidate indices need not be sorted or be the
 * top K; they must be unique and in range. K=0 is allowed and cannot certify.
 */
sl_status sl_selection_base_create(const sl_selection_vocabulary* vocabulary,
                                   const float* values, size_t value_count,
                                   const uint32_t* candidate_indices,
                                   size_t candidate_count,
                                   sl_selection_base** out);
sl_status sl_selection_base_destroy(sl_selection_base* base);
sl_status sl_selection_bias_create(const sl_selection_vocabulary* vocabulary,
                                   const float* values, size_t value_count,
                                   sl_selection_bias** out);
sl_status sl_selection_bias_destroy(sl_selection_bias* bias);

/* No allocation occurs during certify/full. Immutable snapshots can be read
 * concurrently; destruction must not race with a call using that handle.
 *
 * Scores use one binary32 addition in round-to-nearest, ties-to-even mode.
 * Other rounding modes or flush-to-zero/denormals-are-zero execution are rejected
 * with SL_INVALID_STATE. Build without fast-math and with FLT_EVAL_METHOD == 0.
 * The caller must use a normal
 * nontrapping floating-point environment. Ties select the smallest GLOBAL token
 * ID, independently of vocabulary or candidate order.
 *
 * certify is O(K): SL_OK means its strict upper-bound certificate proves the
 * same winner as full. SL_MISS means run full (including conservative overflow
 * of a loose bound). A finite lower bound also prevents hidden negative score
 * overflow. K=V has no tail and checks every score directly.
 *
 * full is O(V). Actual score overflow returns SL_OVERFLOW, not an infinity
 * winner. Dimension/mapping mismatches return SL_INVALID_ARGUMENT. Output is
 * cleared on every failure, including SL_MISS. There are no caller-supplied
 * bounds, borrowed logit arrays or undocumented trusted-summary escape paths.
 */
sl_status sl_selection_certify(const sl_selection_base* base,
                               const sl_selection_bias* bias,
                               sl_selection_result* out);
sl_status sl_selection_full(const sl_selection_base* base,
                            const sl_selection_bias* bias,
                            sl_selection_result* out);

#ifdef __cplusplus
}
#endif
#endif

#ifndef STRIX_LOOKAHEAD_DEVICE_H
#define STRIX_LOOKAHEAD_DEVICE_H

#include "strix_lookahead/pair.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sl_hip_pair sl_hip_pair;
typedef struct sl_cuda_pair sl_cuda_pair;

/* Written on the device after a step completes. status is an sl_status value.
 * Invalid predecessors produce SL_INVALID_ARGUMENT. All-NaN scores produce
 * SL_INVALID_STATE with UINT32_MAX token/row and negative-infinity score.
 * A completed all-NaN step still publishes its computed bias in the cache. */
typedef struct sl_device_result {
    uint32_t status;
    sl_pair_result selection;
} sl_device_result;

/* Experimental device implementations of the rank-256 greedy contract in
 * pair.h. All model pointers in config are DEVICE addresses, borrowed and
 * immutable until destruction. Creation checks token IDs through a one-time
 * device-to-host copy; it does not copy or own the model weights.
 *
 * Each instance is permanently bound to one native stream and the current
 * device. Calls must be externally serialized, on that same current device.
 * Sharing an instance across streams is unsupported. Different instances may
 * use different streams. Creation/reset/destruction may synchronize their
 * stream; enqueue never allocates, synchronizes, or copies data to the host.
 *
 * Inputs and outputs must remain valid until their queued work completes.
 * The caller orders external producers/consumers with stream events. Creation,
 * reset and destruction are outside graph capture; enqueue uses fixed workspace
 * and two kernel launches. Captured graphs must also run on the bound stream,
 * and must not outlive the instance. Model/mapping changes require a new instance.
 *
 * Native stream is hipStream_t or cudaStream_t converted to void*. A null stream
 * selects that runtime's default stream. Link the corresponding backend library;
 * these entry points are not supplied by the host-only reference library. */
sl_status sl_hip_pair_create(const sl_pair_config* config, void* native_stream,
                            sl_hip_pair** out);
sl_status sl_cuda_pair_create(const sl_pair_config* config, void* native_stream,
                             sl_cuda_pair** out);

/* All buffer arguments are DEVICE addresses. The logical base vector is
 * base_head[0..split_count), followed by base_tail[0..V-split_count).
 * Set split_count=V and base_tail=null for a contiguous vector. Empty segments
 * may have null pointers. predecessor and out are required scalar pointers.
 * markov_out[256] and bias_out[V] are optional diagnostic outputs written on
 * both cache hits and misses. A hit still supplies the predecessor embedding
 * needed by a caller's confidence calculation.
 *
 * Buffers must not overlap the borrowed model or one another, except that
 * predecessor may point to out->selection.token for in-place token chaining.
 * SL_OK means launches were enqueued, not that the device completed them.
 * Inspect out->status after completion and check the native stream for errors.
 * A native enqueue failure poisons the instance until a successful reset;
 * argument validation failures leave it usable. If enqueue fails during graph
 * capture, end and discard that capture: reset cannot repair a partially
 * captured step. Do not launch graphs using a poisoned instance: a native graph
 * replay bypasses this library's host-side guard.
 * Capacity zero executes the same selector without caching for comparison. */
sl_status sl_hip_pair_enqueue_step(sl_hip_pair* engine,
                                  const uint32_t* predecessor,
                                  const float* base_head, const float* base_tail,
                                  uint32_t split_count, sl_device_result* out,
                                  float* markov_out, float* bias_out);
sl_status sl_cuda_pair_enqueue_step(sl_cuda_pair* engine,
                                   const uint32_t* predecessor,
                                   const float* base_head, const float* base_tail,
                                   uint32_t split_count, sl_device_result* out,
                                   float* markov_out, float* bias_out);

/* Query is nonblocking: SL_BUSY means queued work is incomplete. Reset waits
 * for work and invalidates entries without rebinding pointers or identities.
 * Destroy waits before releasing storage; on SL_DEVICE_ERROR the handle is
 * still destroyed. SL_INVALID_ARGUMENT (wrong device/null) or SL_INVALID_STATE
 * (capture active or lifecycle precheck failed) leaves the handle alive.
 * A native asynchronous execution failure may require runtime-level recovery;
 * reset does not promise recovery from a failed device/context. */
sl_status sl_hip_pair_query(sl_hip_pair* engine);
sl_status sl_cuda_pair_query(sl_cuda_pair* engine);
sl_status sl_hip_pair_reset(sl_hip_pair* engine);
sl_status sl_cuda_pair_reset(sl_cuda_pair* engine);
sl_status sl_hip_pair_destroy(sl_hip_pair* engine);
sl_status sl_cuda_pair_destroy(sl_cuda_pair* engine);
int sl_hip_pair_last_native_error(const sl_hip_pair* engine);
int sl_cuda_pair_last_native_error(const sl_cuda_pair* engine);

#ifdef __cplusplus
}
#endif
#endif

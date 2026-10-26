#include <strix_lookahead/cache.h>
#include <strix_lookahead/pair.h>
#include <strix_lookahead/selection.h>
#include <strix_lookahead/device.h>

int main(void) {
    sl_pair_config config;
    sl_pair_config_init(&config);
    if (config.struct_size != sizeof(config)) return 1;
    /* Force backend symbols into the consumer link. Null handles are rejected
     * before runtime work, so this does not require a physical GPU. */
#ifdef SL_CONSUMER_HIP
    if (sl_hip_pair_destroy(NULL) != SL_INVALID_ARGUMENT) return 2;
#endif
#ifdef SL_CONSUMER_CUDA
    if (sl_cuda_pair_destroy(NULL) != SL_INVALID_ARGUMENT) return 3;
#endif
    return 0;
}

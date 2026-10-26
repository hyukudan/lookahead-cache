#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define SL_GPU_HIP 1
#define SL_DEVICE_NAMESPACE sl_hip_device
#define SL_HANDLE sl_hip_pair
#define SL_PUBLIC(name) sl_hip_pair_##name
#define SL_RUNTIME(name) hip##name
#define SL_NATIVE_STREAM hipStream_t
#define SL_NATIVE_ERROR hipError_t
#define SL_NATIVE_CAPTURE hipStreamCaptureStatus
#define SL_NATIVE_CAPTURE_NONE hipStreamCaptureStatusNone
#define SL_NATIVE_SUCCESS hipSuccess
#define SL_NATIVE_NOT_READY hipErrorNotReady
#define SL_NATIVE_DEVICE_TO_HOST hipMemcpyDeviceToHost

#include "device_runtime.hpp"

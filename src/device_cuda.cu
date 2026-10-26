#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define SL_GPU_CUDA 1
#define SL_DEVICE_NAMESPACE sl_cuda_device
#define SL_HANDLE sl_cuda_pair
#define SL_PUBLIC(name) sl_cuda_pair_##name
#define SL_RUNTIME(name) cuda##name
#define SL_NATIVE_STREAM cudaStream_t
#define SL_NATIVE_ERROR cudaError_t
#define SL_NATIVE_CAPTURE cudaStreamCaptureStatus
#define SL_NATIVE_CAPTURE_NONE cudaStreamCaptureStatusNone
#define SL_NATIVE_SUCCESS cudaSuccess
#define SL_NATIVE_NOT_READY cudaErrorNotReady
#define SL_NATIVE_DEVICE_TO_HOST cudaMemcpyDeviceToHost

#include "device_runtime.hpp"

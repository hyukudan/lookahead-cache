#ifndef SL_SUPPORT_DEVICE_RUNTIME_HPP
#define SL_SUPPORT_DEVICE_RUNTIME_HPP

#include "strix_lookahead/device.h"

#if defined(SL_TEST_HIP)
#include <hip/hip_runtime_api.h>
#define SL_GPU_RUNTIME(name) hip##name
#define SL_GPU_PAIR(name) sl_hip_pair_##name
#define SL_GPU_ENGINE sl_hip_pair
#define SL_GPU_SUCCESS hipSuccess
#define SL_GPU_NO_DEVICE hipErrorNoDevice
#define SL_GPU_INSUFFICIENT_DRIVER hipErrorInsufficientDriver
#define SL_GPU_H2D hipMemcpyHostToDevice
#define SL_GPU_D2H hipMemcpyDeviceToHost
#define SL_GPU_STREAM hipStream_t
#define SL_GPU_EVENT hipEvent_t
#define SL_GPU_GRAPH hipGraph_t
#define SL_GPU_GRAPH_EXEC hipGraphExec_t
#define SL_GPU_PROPERTIES hipDeviceProp_t
#define SL_GPU_CAPTURE_MODE hipStreamCaptureModeThreadLocal
#define SL_GPU_LABEL "HIP"
#elif defined(SL_TEST_CUDA)
#include <cuda_runtime_api.h>
#define SL_GPU_RUNTIME(name) cuda##name
#define SL_GPU_PAIR(name) sl_cuda_pair_##name
#define SL_GPU_ENGINE sl_cuda_pair
#define SL_GPU_SUCCESS cudaSuccess
#define SL_GPU_NO_DEVICE cudaErrorNoDevice
#define SL_GPU_INSUFFICIENT_DRIVER cudaErrorInsufficientDriver
#define SL_GPU_H2D cudaMemcpyHostToDevice
#define SL_GPU_D2H cudaMemcpyDeviceToHost
#define SL_GPU_STREAM cudaStream_t
#define SL_GPU_EVENT cudaEvent_t
#define SL_GPU_GRAPH cudaGraph_t
#define SL_GPU_GRAPH_EXEC cudaGraphExec_t
#define SL_GPU_PROPERTIES cudaDeviceProp
#define SL_GPU_CAPTURE_MODE cudaStreamCaptureModeThreadLocal
#define SL_GPU_LABEL "CUDA"
#else
#error "Select SL_TEST_HIP or SL_TEST_CUDA for a real device runtime"
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sl_device_support {

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Error> void check(Error error, const char* operation) {
    if (error != SL_GPU_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + SL_GPU_RUNTIME(GetErrorString)(error));
    }
}

inline void check_status(sl_status status, const char* operation) {
    if (status != SL_OK) throw std::runtime_error(std::string(operation) + ": " + sl_status_string(status));
}

inline bool available(std::string& reason) {
    int count = 0;
    const auto error = SL_GPU_RUNTIME(GetDeviceCount)(&count);
    if (error == SL_GPU_NO_DEVICE || error == SL_GPU_INSUFFICIENT_DRIVER) {
        reason = SL_GPU_RUNTIME(GetErrorString)(error);
        return false;
    }
    check(error, "query visible GPU count");
    if (count == 0) {
        reason = "No compatible GPU is visible";
        return false;
    }
    return true;
}

struct stream {
    SL_GPU_STREAM value{};
    stream() { check(SL_GPU_RUNTIME(StreamCreate)(&value), "create stream"); }
    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;
    ~stream() { if (value) (void)SL_GPU_RUNTIME(StreamDestroy)(value); }
    void wait() const { check(SL_GPU_RUNTIME(StreamSynchronize)(value), "synchronize stream"); }
};

template<class T> struct buffer {
    T* data = nullptr;
    std::size_t count = 0;
    explicit buffer(std::size_t elements) : count(elements) {
        require(elements <= std::numeric_limits<std::size_t>::max() / sizeof(T), "buffer size overflow");
        if (elements != 0) check(SL_GPU_RUNTIME(Malloc)(reinterpret_cast<void**>(&data), elements * sizeof(T)), "allocate buffer");
    }
    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;
    ~buffer() { if (data) (void)SL_GPU_RUNTIME(Free)(data); }
    void upload(const T* source, SL_GPU_STREAM target) {
        if (count != 0) check(SL_GPU_RUNTIME(MemcpyAsync)(data, source, count * sizeof(T), SL_GPU_H2D, target), "upload buffer");
    }
    void download(T* target, SL_GPU_STREAM source) const {
        if (count != 0) check(SL_GPU_RUNTIME(MemcpyAsync)(target, data, count * sizeof(T), SL_GPU_D2H, source), "download buffer");
    }
};

struct engine {
    SL_GPU_ENGINE* value = nullptr;
    engine(const sl_pair_config& config, SL_GPU_STREAM target) {
        check_status(SL_GPU_PAIR(create)(&config, reinterpret_cast<void*>(target), &value), "create pair engine");
    }
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;
    ~engine() { if (value) (void)SL_GPU_PAIR(destroy)(value); }
    void reset() { check_status(SL_GPU_PAIR(reset)(value), "reset pair engine"); }
    void enqueue(const std::uint32_t* predecessor, const float* head, const float* tail,
                 std::uint32_t split, sl_device_result* out, float* markov = nullptr,
                 float* bias = nullptr) {
        check_status(SL_GPU_PAIR(enqueue_step)(value, predecessor, head, tail, split, out, markov, bias), "enqueue pair step");
    }
};

struct event {
    SL_GPU_EVENT value{};
    event() { check(SL_GPU_RUNTIME(EventCreate)(&value), "create event"); }
    event(const event&) = delete;
    event& operator=(const event&) = delete;
    ~event() { if (value) (void)SL_GPU_RUNTIME(EventDestroy)(value); }
};

inline std::uint32_t* token_address(sl_device_result* result) {
    return reinterpret_cast<std::uint32_t*>(reinterpret_cast<unsigned char*>(result) +
        offsetof(sl_device_result, selection) + offsetof(sl_pair_result, token));
}

inline std::string json_quote(const std::string& text) {
    std::string result = "\"";
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c >= 32) result += static_cast<char>(c);
    }
    return result + '"';
}

}  // namespace sl_device_support
#endif

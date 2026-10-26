#ifndef SL_DEVICE_RUNTIME_HPP
#define SL_DEVICE_RUNTIME_HPP

#include "device_kernels.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <vector>

struct SL_HANDLE {
    sl_pair_config config{};
    SL_NATIVE_STREAM stream{};
    SL_DEVICE_NAMESPACE::model_view model{};
    SL_DEVICE_NAMESPACE::winner* partials = nullptr;
    std::uint32_t blocks = 0;
    int device = -1;
    int last_error = 0;
    bool poisoned = false;
};

namespace {

bool checked_product(std::size_t a, std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    result = a * b;
    return true;
}

sl_status device_error(SL_HANDLE* engine, SL_NATIVE_ERROR error) {
    engine->last_error = static_cast<int>(error);
    engine->poisoned = true;
    return SL_DEVICE_ERROR;
}

sl_status current_device(SL_HANDLE* engine) {
    int current = -1;
    const auto error = SL_RUNTIME(GetDevice)(&current);
    if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    return current == engine->device ? SL_OK : SL_INVALID_ARGUMENT;
}

sl_status outside_capture(SL_HANDLE* engine) {
    SL_NATIVE_CAPTURE capture = SL_NATIVE_CAPTURE_NONE;
    const auto error = SL_RUNTIME(StreamIsCapturing)(engine->stream, &capture);
    if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    return capture == SL_NATIVE_CAPTURE_NONE ? SL_OK : SL_INVALID_STATE;
}

// Called only after work has completed or during failed construction before any
// inference launch. Every allocation is attempted even if a preceding free fails.
sl_status free_storage(SL_HANDLE* engine) {
    sl_status status = SL_OK;
    if (engine->model.payload != nullptr) {
        const auto error = SL_RUNTIME(Free)(engine->model.payload);
        if (error != SL_NATIVE_SUCCESS) status = device_error(engine, error);
        engine->model.payload = nullptr;
    }
    if (engine->model.entries != nullptr) {
        const auto error = SL_RUNTIME(Free)(engine->model.entries);
        if (error != SL_NATIVE_SUCCESS) status = device_error(engine, error);
        engine->model.entries = nullptr;
    }
    if (engine->partials != nullptr) {
        const auto error = SL_RUNTIME(Free)(engine->partials);
        if (error != SL_NATIVE_SUCCESS) status = device_error(engine, error);
        engine->partials = nullptr;
    }
    return status;
}

sl_status validate_config(const sl_pair_config* config, std::size_t& payload_bytes,
                          std::size_t& metadata_bytes) {
    if (config == nullptr || config->struct_size != sizeof(sl_pair_config) ||
        config->abi_version != SL_ABI_VERSION || config->reserved[0] != 0 ||
        config->reserved[1] != 0 || config->vocabulary_size == 0 ||
        config->embedding_rows == 0 || config->head == nullptr ||
        config->embeddings == nullptr ||
        (config->head_layout != SL_PAIR_HEAD_GLOBAL && config->head_layout != SL_PAIR_HEAD_SELECTED) ||
        config->vocabulary_size > config->embedding_rows) return SL_INVALID_ARGUMENT;

    std::size_t ignored = 0;
    std::size_t elements = 0;
    if (!checked_product(config->embedding_rows, SL_PAIR_RANK, elements) ||
        !checked_product(elements, sizeof(std::uint16_t), ignored) ||
        !checked_product(config->vocabulary_size, sizeof(std::uint32_t), ignored) ||
        !checked_product(config->vocabulary_size, config->capacity_entries, elements) ||
        !checked_product(elements, sizeof(float), payload_bytes) ||
        !checked_product(config->capacity_entries, sizeof(SL_DEVICE_NAMESPACE::entry), metadata_bytes) ||
        metadata_bytes > std::numeric_limits<std::size_t>::max() - payload_bytes) return SL_OVERFLOW;
    if (config->payload_budget_bytes != 0 && payload_bytes > config->payload_budget_bytes)
        return SL_INVALID_ARGUMENT;
    return SL_OK;
}

}  // namespace

extern "C" {

sl_status SL_PUBLIC(create)(const sl_pair_config* config, void* native_stream,
                            SL_HANDLE** out) {
    if (out != nullptr) *out = nullptr;
    if (out == nullptr) return SL_INVALID_ARGUMENT;
    std::size_t payload_bytes = 0;
    std::size_t metadata_bytes = 0;
    const auto validation = validate_config(config, payload_bytes, metadata_bytes);
    if (validation != SL_OK) return validation;
    try {
        auto engine = std::make_unique<SL_HANDLE>();
        engine->config = *config;
        engine->stream = reinterpret_cast<SL_NATIVE_STREAM>(native_stream);
        auto error = SL_RUNTIME(GetDevice)(&engine->device);
        if (error != SL_NATIVE_SUCCESS) return SL_DEVICE_ERROR;
        int stream_device = -1;
        error = SL_RUNTIME(StreamGetDevice)(engine->stream, &stream_device);
        if (error != SL_NATIVE_SUCCESS) return SL_DEVICE_ERROR;
        if (stream_device != engine->device) return SL_INVALID_ARGUMENT;
        const auto capture = outside_capture(engine.get());
        if (capture != SL_OK) return capture;

        if (config->token_ids != nullptr) {
            std::vector<std::uint32_t> ids(config->vocabulary_size);
            error = SL_RUNTIME(MemcpyAsync)(ids.data(), config->token_ids,
                ids.size() * sizeof(std::uint32_t), SL_NATIVE_DEVICE_TO_HOST, engine->stream);
            if (error != SL_NATIVE_SUCCESS) return SL_DEVICE_ERROR;
            error = SL_RUNTIME(StreamSynchronize)(engine->stream);
            if (error != SL_NATIVE_SUCCESS) return SL_DEVICE_ERROR;
            for (const auto id : ids) if (id >= config->embedding_rows) return SL_INVALID_ARGUMENT;
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) return SL_INVALID_ARGUMENT;
        }

        engine->blocks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            SL_DEVICE_NAMESPACE::maximum_blocks,
            (static_cast<std::uint64_t>(config->vocabulary_size) +
             SL_DEVICE_NAMESPACE::block_threads - 1) / SL_DEVICE_NAMESPACE::block_threads));
        engine->model = {config->head, config->embeddings, config->token_ids,
            config->vocabulary_size, config->embedding_rows, config->head_layout,
            config->capacity_entries, nullptr, nullptr};
        if (payload_bytes != 0) {
            error = SL_RUNTIME(Malloc)(reinterpret_cast<void**>(&engine->model.payload), payload_bytes);
            if (error != SL_NATIVE_SUCCESS) return SL_ALLOCATION_FAILED;
            error = SL_RUNTIME(Malloc)(reinterpret_cast<void**>(&engine->model.entries), metadata_bytes);
            if (error != SL_NATIVE_SUCCESS) {
                free_storage(engine.get());
                return SL_ALLOCATION_FAILED;
            }
        }
        error = SL_RUNTIME(Malloc)(reinterpret_cast<void**>(&engine->partials),
            engine->blocks * sizeof(SL_DEVICE_NAMESPACE::winner));
        if (error != SL_NATIVE_SUCCESS) {
            free_storage(engine.get());
            return SL_ALLOCATION_FAILED;
        }
        if (metadata_bytes != 0) {
            error = SL_RUNTIME(MemsetAsync)(engine->model.entries, 0, metadata_bytes, engine->stream);
            if (error == SL_NATIVE_SUCCESS) error = SL_RUNTIME(StreamSynchronize)(engine->stream);
            if (error != SL_NATIVE_SUCCESS) {
                free_storage(engine.get());
                return SL_DEVICE_ERROR;
            }
        }
        *out = engine.release();
        return SL_OK;
    } catch (const std::bad_alloc&) {
        return SL_ALLOCATION_FAILED;
    } catch (...) {
        return SL_ALLOCATION_FAILED;
    }
}

sl_status SL_PUBLIC(enqueue_step)(SL_HANDLE* engine, const std::uint32_t* predecessor,
                                 const float* base_head, const float* base_tail,
                                 std::uint32_t split_count, sl_device_result* out,
                                 float* markov_out, float* bias_out) {
    if (engine == nullptr || predecessor == nullptr || out == nullptr ||
        split_count > engine->config.vocabulary_size ||
        (split_count != 0 && base_head == nullptr) ||
        (split_count < engine->config.vocabulary_size && base_tail == nullptr)) return SL_INVALID_ARGUMENT;
    if (engine->poisoned) return SL_INVALID_STATE;
    const auto status = current_device(engine);
    if (status != SL_OK) return status;
    SL_DEVICE_NAMESPACE::pair_pass<<<engine->blocks, SL_DEVICE_NAMESPACE::block_threads, 0, engine->stream>>>(
        engine->model, predecessor, base_head, base_tail, split_count, engine->partials,
        markov_out, bias_out);
    auto error = SL_RUNTIME(GetLastError)();
    if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    SL_DEVICE_NAMESPACE::pair_finish<<<1, SL_DEVICE_NAMESPACE::block_threads, 0, engine->stream>>>(
        engine->model, predecessor, engine->partials, engine->blocks, out);
    error = SL_RUNTIME(GetLastError)();
    if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    return SL_OK;
}

sl_status SL_PUBLIC(query)(SL_HANDLE* engine) {
    if (engine == nullptr) return SL_INVALID_ARGUMENT;
    const auto status = current_device(engine);
    if (status != SL_OK) return status;
    const auto capture = outside_capture(engine);
    if (capture != SL_OK) return capture;
    if (engine->poisoned) return SL_INVALID_STATE;
    const auto error = SL_RUNTIME(StreamQuery)(engine->stream);
    if (error == SL_NATIVE_NOT_READY) return SL_BUSY;
    return error == SL_NATIVE_SUCCESS ? SL_OK : device_error(engine, error);
}

sl_status SL_PUBLIC(reset)(SL_HANDLE* engine) {
    if (engine == nullptr) return SL_INVALID_ARGUMENT;
    const auto status = current_device(engine);
    if (status != SL_OK) return status;
    const auto capture = outside_capture(engine);
    if (capture != SL_OK) return capture;
    auto error = SL_RUNTIME(StreamSynchronize)(engine->stream);
    if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    if (engine->config.capacity_entries != 0) {
        error = SL_RUNTIME(MemsetAsync)(engine->model.entries, 0,
            static_cast<std::size_t>(engine->config.capacity_entries) * sizeof(SL_DEVICE_NAMESPACE::entry),
            engine->stream);
        if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
        error = SL_RUNTIME(StreamSynchronize)(engine->stream);
        if (error != SL_NATIVE_SUCCESS) return device_error(engine, error);
    }
    engine->poisoned = false;
    engine->last_error = 0;
    return SL_OK;
}

sl_status SL_PUBLIC(destroy)(SL_HANDLE* engine) {
    if (engine == nullptr) return SL_INVALID_ARGUMENT;
    const auto status = current_device(engine);
    if (status != SL_OK) return status == SL_DEVICE_ERROR ? SL_INVALID_STATE : status;
    const auto capture = outside_capture(engine);
    if (capture != SL_OK) return SL_INVALID_STATE;
    const auto error = SL_RUNTIME(StreamSynchronize)(engine->stream);
    sl_status result = error == SL_NATIVE_SUCCESS ? SL_OK : device_error(engine, error);
    if (free_storage(engine) != SL_OK) result = SL_DEVICE_ERROR;
    delete engine;
    return result;
}

int SL_PUBLIC(last_native_error)(const SL_HANDLE* engine) {
    return engine == nullptr ? -1 : engine->last_error;
}

}  // extern "C"
#endif

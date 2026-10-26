#include "../support/device_runtime.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <unordered_map>

using namespace sl_device_support;

namespace {

struct options {
    std::uint32_t vocab = 129280;
    std::uint32_t steps = 64;
    std::uint32_t repeats = 7;
    std::uint32_t capacity = 16;
    bool graph = true;
};

std::uint32_t number(const char* input) {
    std::uint32_t value = 0;
    const std::string text(input);
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), value);
    require(conversion.ec == std::errc{} && conversion.ptr == text.data() + text.size(), "invalid unsigned integer option");
    return value;
}

options parse(int argc, char** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        require(i + 1 < argc, "option requires a value");
        const auto value = number(argv[++i]);
        if (key == "--vocab") result.vocab = value;
        else if (key == "--steps") result.steps = value;
        else if (key == "--repeats") result.repeats = value;
        else if (key == "--capacity") result.capacity = value;
        else if (key == "--graph") { require(value <= 1, "--graph must be 0 or 1"); result.graph = value != 0; }
        else throw std::runtime_error("unknown option: " + key);
    }
    require(result.vocab != 0 && result.steps != 0 && result.repeats != 0 && result.capacity != 0,
            "vocab, steps, repeats and capacity must be positive");
    require(result.steps < UINT32_MAX, "steps is too large");
    return result;
}

struct graph {
    SL_GPU_GRAPH value{};
    SL_GPU_GRAPH_EXEC executable{};
    graph(engine& target, SL_GPU_STREAM queue, const std::uint32_t* predecessors,
          const float* base, sl_device_result* results, const options& settings) {
        if (!settings.graph) return;
        check(SL_GPU_RUNTIME(StreamBeginCapture)(queue, SL_GPU_CAPTURE_MODE), "begin benchmark capture");
        for (std::uint32_t i = 0; i < settings.steps; ++i)
            target.enqueue(predecessors + i, base, nullptr, settings.vocab, results + i);
        check(SL_GPU_RUNTIME(StreamEndCapture)(queue, &value), "end benchmark capture");
        check(SL_GPU_RUNTIME(GraphInstantiateWithFlags)(&executable, value, 0), "instantiate benchmark graph");
        check(SL_GPU_RUNTIME(GraphUpload)(executable, queue), "upload benchmark graph");
        check(SL_GPU_RUNTIME(StreamSynchronize)(queue), "finish graph upload");
    }
    graph(const graph&) = delete;
    graph& operator=(const graph&) = delete;
    ~graph() {
        if (executable) (void)SL_GPU_RUNTIME(GraphExecDestroy)(executable);
        if (value) (void)SL_GPU_RUNTIME(GraphDestroy)(value);
    }
};

struct pattern {
    std::string name;
    std::vector<std::uint32_t> predecessors;
    std::vector<std::uint32_t> warmup;
};

std::vector<pattern> patterns(const options& settings, std::uint32_t rows) {
    std::vector<pattern> result;
    pattern cold{"cold", std::vector<std::uint32_t>(settings.steps), {}};
    std::iota(cold.predecessors.begin(), cold.predecessors.end(), 0u);
    result.push_back(std::move(cold));
    std::vector<std::uint32_t> hot, occupied;
    const auto hot_count = std::min<std::uint32_t>(8, settings.capacity);
    for (std::uint32_t token = 0; token < rows && hot.size() < hot_count; ++token) {
        const auto slot = sl_pair_slot_for(token, settings.capacity);
        if (std::find(occupied.begin(), occupied.end(), slot) == occupied.end()) {
            occupied.push_back(slot);
            hot.push_back(token);
        }
    }
    require(!hot.empty(), "cannot construct a warm workload");
    pattern warm{"warm", std::vector<std::uint32_t>(settings.steps), hot};
    pattern mixed{"mixed", std::vector<std::uint32_t>(settings.steps), hot};
    for (std::uint32_t i = 0; i < settings.steps; ++i) {
        warm.predecessors[i] = hot[i % hot.size()];
        mixed.predecessors[i] = i % 10u == 9u ? (i + 1u) % rows : hot[i % hot.size()];
    }
    result.push_back(std::move(warm));
    result.push_back(std::move(mixed));
    std::unordered_map<std::uint32_t, std::uint32_t> seen;
    for (std::uint32_t token = 0; token < rows; ++token) {
        const auto slot = sl_pair_slot_for(token, settings.capacity);
        const auto inserted = seen.emplace(slot, token);
        if (!inserted.second) {
            pattern collision{"collision", std::vector<std::uint32_t>(settings.steps), {}};
            for (std::uint32_t i = 0; i < settings.steps; ++i)
                collision.predecessors[i] = i % 2u ? token : inserted.first->second;
            result.push_back(std::move(collision));
            break;
        }
    }
    return result;
}

std::vector<std::uint32_t> expected_hits(const pattern& workload, std::uint32_t capacity) {
    std::unordered_map<std::uint32_t, std::uint32_t> entries;
    for (const auto token : workload.warmup) entries[sl_pair_slot_for(token, capacity)] = token;
    std::vector<std::uint32_t> result;
    for (const auto token : workload.predecessors) {
        const auto slot = sl_pair_slot_for(token, capacity);
        const auto current = entries.find(slot);
        result.push_back(current != entries.end() && current->second == token ? 1u : 0u);
        entries[slot] = token;
    }
    return result;
}

struct measurement {
    std::string name;
    std::vector<float> uncached_ms, cached_ms;
    std::uint64_t hits = 0;
    std::size_t warmup_steps = 0;
};

float run_once(engine& target, const graph& captured, SL_GPU_STREAM queue,
               const buffer<std::uint32_t>& predecessors, const buffer<std::uint32_t>& warmup,
               const buffer<float>& base, buffer<sl_device_result>& results,
               buffer<sl_device_result>& warmup_result, const options& settings,
               std::vector<sl_device_result>& host_results) {
    target.reset();
    for (std::size_t i = 0; i < warmup.count; ++i)
        target.enqueue(warmup.data + i, base.data, nullptr, settings.vocab, warmup_result.data);
    event begin, end;
    check(SL_GPU_RUNTIME(EventRecord)(begin.value, queue), "record begin event");
    if (captured.executable) check(SL_GPU_RUNTIME(GraphLaunch)(captured.executable, queue), "launch benchmark graph");
    else for (std::uint32_t i = 0; i < settings.steps; ++i)
        target.enqueue(predecessors.data + i, base.data, nullptr, settings.vocab, results.data + i);
    check(SL_GPU_RUNTIME(EventRecord)(end.value, queue), "record end event");
    check(SL_GPU_RUNTIME(EventSynchronize)(end.value), "wait for benchmark event");
    float elapsed = 0;
    check(SL_GPU_RUNTIME(EventElapsedTime)(&elapsed, begin.value, end.value), "elapsed device time");
    results.download(host_results.data(), queue);
    check(SL_GPU_RUNTIME(StreamSynchronize)(queue), "read benchmark results");
    return elapsed;
}

measurement measure(const pattern& workload, const options& settings, SL_GPU_STREAM queue,
                    const sl_pair_config& device_config, const buffer<float>& base) {
    auto uncached_config = device_config;
    uncached_config.capacity_entries = 0;
    engine uncached(uncached_config, queue), cached(device_config, queue);
    buffer<std::uint32_t> predecessors(workload.predecessors.size()), warmup(workload.warmup.size());
    buffer<sl_device_result> baseline_results(settings.steps), cached_results(settings.steps), warmup_result(1);
    predecessors.upload(workload.predecessors.data(), queue);
    warmup.upload(workload.warmup.data(), queue);
    check(SL_GPU_RUNTIME(StreamSynchronize)(queue), "prepare benchmark inputs");
    graph baseline_graph(uncached, queue, predecessors.data, base.data, baseline_results.data, settings);
    graph cached_graph(cached, queue, predecessors.data, base.data, cached_results.data, settings);
    const auto hit_reference = expected_hits(workload, settings.capacity);
    measurement result{workload.name, {}, {}, 0, workload.warmup.size()};
    std::vector<sl_device_result> baseline(settings.steps), optimized(settings.steps);
    // Exercise both complete launch paths once outside the samples. run_once
    // resets before each later sample, so cold samples still start with no tags.
    (void)run_once(uncached, baseline_graph, queue, predecessors, warmup, base,
                   baseline_results, warmup_result, settings, baseline);
    (void)run_once(cached, cached_graph, queue, predecessors, warmup, base,
                   cached_results, warmup_result, settings, optimized);
    for (std::uint32_t repetition = 0; repetition < settings.repeats; ++repetition) {
        float baseline_ms = 0, cached_ms = 0;
        const auto run_baseline = [&] {
            baseline_ms = run_once(uncached, baseline_graph, queue, predecessors, warmup, base,
                                   baseline_results, warmup_result, settings, baseline);
        };
        const auto run_cached = [&] {
            cached_ms = run_once(cached, cached_graph, queue, predecessors, warmup, base,
                                 cached_results, warmup_result, settings, optimized);
        };
        if (repetition % 2u == 0) { run_baseline(); run_cached(); }
        else { run_cached(); run_baseline(); }
        for (std::uint32_t i = 0; i < settings.steps; ++i) {
            require(baseline[i].status == SL_OK && optimized[i].status == SL_OK, "benchmark device result failed");
            require(baseline[i].selection.token == optimized[i].selection.token &&
                    baseline[i].selection.vocabulary_row == optimized[i].selection.vocabulary_row &&
                    std::bit_cast<std::uint32_t>(baseline[i].selection.score) ==
                    std::bit_cast<std::uint32_t>(optimized[i].selection.score), "cached and uncached results differ");
            require(baseline[i].selection.cache_hit == 0 && optimized[i].selection.cache_hit == hit_reference[i],
                    "measured hit flags differ from direct-map trace");
            result.hits += optimized[i].selection.cache_hit;
        }
        result.uncached_ms.push_back(baseline_ms);
        result.cached_ms.push_back(cached_ms);
    }
    return result;
}

void array(const std::vector<float>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Device pair-cache microbenchmark: --vocab N --steps N --repeats N --capacity N --graph 0|1\n"
                     "Defaults: 129280, 64, 7, 16, 1. Emits JSON; exit77 when no compatible GPU is visible.\n";
        return 0;
    }
    try {
        const auto settings = parse(argc, argv);
        std::string reason;
        if (!available(reason)) {
            std::cout << "{\"status\":\"skipped\",\"backend\":" << json_quote(SL_GPU_LABEL)
                      << ",\"reason\":" << json_quote(reason) << "}\n";
            return 77;
        }
        int current = -1;
        check(SL_GPU_RUNTIME(GetDevice)(&current), "get device");
        SL_GPU_PROPERTIES properties{};
        check(SL_GPU_RUNTIME(GetDeviceProperties)(&properties, current), "get device properties");
        const auto rows = std::max<std::uint32_t>(64, std::max(settings.vocab, settings.steps + 1u));
        const auto head_count = static_cast<std::size_t>(settings.vocab) * SL_PAIR_RANK;
        const auto embedding_count = static_cast<std::size_t>(rows) * SL_PAIR_RANK;
        std::vector<std::uint16_t> weights(head_count), embeddings(embedding_count);
        std::vector<float> host_base(settings.vocab);
        // Deterministic finite binary16 data with varied mantissas and signs;
        // these are synthetic component inputs, not trained model weights.
        const auto fill_half = [](std::vector<std::uint16_t>& target, std::uint32_t state) {
            for (auto& value : target) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                const auto sign = (state >> 16) & 0x8000u;
                const auto exponent = 5u + ((state >> 10) % 14u);
                value = static_cast<std::uint16_t>(sign | (exponent << 10) | (state & 1023u));
            }
        };
        fill_half(weights, 0x243f6a88u);
        fill_half(embeddings, 0x85a308d3u);
        for (std::size_t i = 0; i < host_base.size(); ++i) host_base[i] = static_cast<float>(static_cast<int>(i % 31u) - 15) * 0.125f;
        stream queue;
        buffer<std::uint16_t> device_weights(weights.size()), device_embeddings(embeddings.size());
        buffer<float> base(host_base.size());
        device_weights.upload(weights.data(), queue.value);
        device_embeddings.upload(embeddings.data(), queue.value);
        base.upload(host_base.data(), queue.value);
        queue.wait();
        sl_pair_config config{};
        sl_pair_config_init(&config);
        config.vocabulary_size = settings.vocab;
        config.embedding_rows = rows;
        config.capacity_entries = settings.capacity;
        config.head_layout = SL_PAIR_HEAD_SELECTED;
        config.head = device_weights.data;
        config.embeddings = device_embeddings.data;
        const auto workloads = patterns(settings, rows);
        std::vector<measurement> measurements;
        for (const auto& workload : workloads) measurements.push_back(measure(workload, settings, queue.value, config, base));
        std::cout << std::setprecision(9)
                  << "{\"status\":\"ok\",\"measurement\":\"pair_cache_component_only\",\"synthetic_inputs\":true,\"fixed_base_logits\":true,\"end_to_end_inference\":false,\"baseline\":\"same_two_kernel_capacity_zero\",\"runtime_warmup_repetitions\":1,\"weights_seed\":\"0x243f6a88\",\"embeddings_seed\":\"0x85a308d3\",\"backend\":" << json_quote(SL_GPU_LABEL)
                  << ",\"hardware_name\":" << json_quote(properties.name)
                  << ",\"wave_size\":" << properties.warpSize
                  << ",\"vocab\":" << settings.vocab << ",\"embedding_rows\":" << rows
                  << ",\"rank\":256,\"capacity\":" << settings.capacity
                  << ",\"steps\":" << settings.steps << ",\"repeats\":" << settings.repeats
                  << ",\"graph\":" << (settings.graph ? "true" : "false")
                  << ",\"order\":\"alternating\",\"timing\":\"device_events_excluding_reset_warmup_and_readback\",\"cases\":[";
        for (std::size_t i = 0; i < measurements.size(); ++i) {
            if (i) std::cout << ',';
            const auto& value = measurements[i];
            const double uncached_mean = std::accumulate(value.uncached_ms.begin(), value.uncached_ms.end(), 0.0) / settings.repeats;
            const double cached_mean = std::accumulate(value.cached_ms.begin(), value.cached_ms.end(), 0.0) / settings.repeats;
            const auto measured_steps = static_cast<std::uint64_t>(settings.steps) * settings.repeats;
            std::cout << "{\"pattern\":" << json_quote(value.name) << ",\"equivalent\":true,\"hits\":" << value.hits
                      << ",\"warmup_prefill_steps\":" << value.warmup_steps
                      << ",\"warm_cache_prefilled\":" << (value.warmup_steps ? "true" : "false")
                      << ",\"initial_hotset_fill_excluded\":" << (value.warmup_steps ? "true" : "false")
                      << ",\"hit_rate\":" << static_cast<double>(value.hits) / static_cast<double>(measured_steps)
                      << ",\"uncached_ms_mean\":" << uncached_mean << ",\"cached_ms_mean\":" << cached_mean
                      << ",\"component_speedup\":";
            if (cached_mean > 0) std::cout << uncached_mean / cached_mean;
            else std::cout << "null";
            std::cout << ",\"uncached_ms\":"; array(value.uncached_ms);
            std::cout << ",\"cached_ms\":"; array(value.cached_ms);
            std::cout << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"status\":\"error\",\"backend\":" << json_quote(SL_GPU_LABEL)
                  << ",\"reason\":" << json_quote(error.what()) << "}\n";
        return 1;
    }
}

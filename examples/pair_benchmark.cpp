#include "strix_lookahead/pair.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t data_seed = 0x5168724c6f6f6b31ULL;
constexpr std::uint64_t memory_limit = 512ULL * 1024ULL * 1024ULL;

struct Options {
    std::uint32_t vocab = 8192;
    std::uint32_t steps = 24;
    std::uint32_t repeats = 3;
    std::uint32_t capacity = 16;
};

std::uint32_t number(std::string_view input) {
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || input.empty())
        throw std::invalid_argument("expected an unsigned decimal integer");
    return value;
}

Options options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key = argv[i];
        if (key != "--vocab" && key != "--steps" && key != "--repeats" && key != "--capacity")
            throw std::invalid_argument("unknown option: " + std::string(key));
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(key));
        const auto value = number(argv[++i]);
        if (key == "--vocab") out.vocab = value;
        else if (key == "--steps") out.steps = value;
        else if (key == "--repeats") out.repeats = value;
        else out.capacity = value;
    }
    if (out.vocab == 0 || out.vocab > 262144U || out.steps == 0 || out.steps > 4096U ||
        out.repeats == 0 || out.repeats > 31U || out.capacity == 0 || out.capacity > 4096U)
        throw std::invalid_argument("limits: vocab 1..262144, steps 1..4096, repeats 1..31, capacity 1..4096");
    const auto rows = static_cast<std::uint64_t>(out.vocab) * out.steps * out.repeats;
    if (rows > 50000000ULL)
        throw std::invalid_argument("vocab * steps * repeats must not exceed 50000000");
    return out;
}

struct Random {
    std::uint64_t state = data_seed;
    std::uint32_t next() {
        state ^= state >> 12U;
        state ^= state << 25U;
        state ^= state >> 27U;
        return static_cast<std::uint32_t>((state * 0x2545f4914f6cdd1dULL) >> 32U);
    }
    std::uint16_t half() {
        const auto bits = next();
        // Finite, exactly representable binary16 inputs with varied signs.
        return static_cast<std::uint16_t>((bits & 0x83ffU) | ((13U + ((bits >> 11U) & 3U)) << 10U));
    }
};

void require_status(sl_status status, const char* operation) {
    if (status != SL_OK)
        throw std::runtime_error(std::string(operation) + ": " + sl_status_string(status));
}

struct DeleteEngine {
    void operator()(sl_pair_engine* engine) const { (void)sl_pair_destroy(engine); }
};
using Engine = std::unique_ptr<sl_pair_engine, DeleteEngine>;

Engine create(const sl_pair_config& config) {
    sl_pair_engine* raw = nullptr;
    require_status(sl_pair_create(&config, &raw), "create");
    return Engine(raw);
}

sl_pair_stats stats(sl_pair_engine* engine) {
    sl_pair_stats value{};
    require_status(sl_pair_get_stats(engine, &value), "stats");
    return value;
}

std::vector<std::uint32_t> predecessors(std::string_view mode, const Options& o,
                                        std::uint32_t embedding_rows) {
    std::vector<std::uint32_t> out(o.steps);
    if (mode == "collision") {
        const auto absent = std::numeric_limits<std::uint32_t>::max();
        std::vector<std::uint32_t> first(o.capacity, absent);
        std::uint32_t a = absent, b = absent;
        for (std::uint32_t id = 0; id < embedding_rows; ++id) {
            const auto slot = sl_pair_slot_for(id, o.capacity);
            if (first[slot] != absent) { a = first[slot]; b = id; break; }
            first[slot] = id;
        }
        if (b == absent) throw std::runtime_error("unable to construct direct-map collision");
        for (std::uint32_t i = 0; i < o.steps; ++i) out[i] = i % 2U == 0U ? a : b;
    } else {
        for (std::uint32_t i = 0; i < o.steps; ++i) {
            if (mode == "warm") out[i] = 0U;
            else if (mode == "cold") out[i] = i;
            else out[i] = i % 4U == 1U ? 1U : i % 4U == 3U ? 2U + i / 4U : 0U;
        }
    }
    return out;
}

struct Run {
    std::vector<std::uint64_t> samples;
    std::vector<sl_pair_result> outputs;
    std::uint64_t total_ns = 0;
};

void run(sl_pair_engine* engine, const std::vector<std::uint32_t>& keys,
         const std::vector<float>& logits, std::uint32_t vocab, Run& result) {
    using Clock = std::chrono::steady_clock;
    for (std::size_t step = 0; step < keys.size(); ++step) {
        const auto start = Clock::now();
        const auto status = sl_pair_step(engine, keys[step], logits.data() + step * vocab,
                                         &result.outputs[step], nullptr, nullptr);
        const auto end = Clock::now();
        require_status(status, "step");
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        if (elapsed < 0) throw std::runtime_error("steady clock moved backwards");
        const auto ns = static_cast<std::uint64_t>(elapsed);
        result.samples.push_back(ns);
        result.total_ns += ns;
    }
}

std::uint64_t quantile(const std::vector<std::uint64_t>& input, std::size_t percent) {
    auto sorted = input;
    std::sort(sorted.begin(), sorted.end());
    const auto rank = (sorted.size() * percent + 99U) / 100U;
    return sorted[rank - 1U];
}

struct ModeResult {
    std::string mode;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t baseline_ns = 0;
    std::uint64_t cached_ns = 0;
    std::uint64_t baseline_p50 = 0;
    std::uint64_t baseline_p95 = 0;
    std::uint64_t cached_p50 = 0;
    std::uint64_t cached_p95 = 0;
};

void json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (const char raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"') std::cout << "\\\"";
        else if (c == '\\') std::cout << "\\\\";
        else if (c < 0x20U) std::cout << "\\u00" << hex[c >> 4U] << hex[c & 15U];
        else std::cout << raw;
    }
    std::cout << '"';
}

ModeResult benchmark(std::string_view mode, const Options& o, std::uint32_t embedding_rows,
                     const std::vector<float>& logits, sl_pair_engine* baseline,
                     sl_pair_engine* cached) {
    const auto keys = predecessors(mode, o, embedding_rows);
    Run a, b;
    const auto sample_count = static_cast<std::size_t>(o.steps) * o.repeats;
    a.samples.reserve(sample_count);
    b.samples.reserve(sample_count);
    a.outputs.resize(o.steps);
    b.outputs.resize(o.steps);
    const auto before = stats(cached);
    for (std::uint32_t repeat = 0; repeat < o.repeats; ++repeat) {
        require_status(sl_pair_reset(baseline, nullptr), "reset baseline");
        require_status(sl_pair_reset(cached, nullptr), "reset cached");
        if (repeat % 2U == 0U) {
            run(baseline, keys, logits, o.vocab, a);
            run(cached, keys, logits, o.vocab, b);
        } else {
            run(cached, keys, logits, o.vocab, b);
            run(baseline, keys, logits, o.vocab, a);
        }
        for (std::size_t step = 0; step < keys.size(); ++step) {
            const auto& reference = a.outputs[step];
            const auto& candidate = b.outputs[step];
            if (reference.token != candidate.token || reference.vocabulary_row != candidate.vocabulary_row ||
                std::bit_cast<std::uint32_t>(reference.score) != std::bit_cast<std::uint32_t>(candidate.score))
                throw std::runtime_error("output mismatch in " + std::string(mode) + " at step " + std::to_string(step));
        }
    }
    const auto after = stats(cached);
    ModeResult out;
    out.mode = mode;
    out.hits = after.hits - before.hits;
    out.misses = after.misses - before.misses;
    out.baseline_ns = a.total_ns;
    out.cached_ns = b.total_ns;
    out.baseline_p50 = quantile(a.samples, 50);
    out.baseline_p95 = quantile(a.samples, 95);
    out.cached_p50 = quantile(b.samples, 50);
    out.cached_p95 = quantile(b.samples, 95);
    return out;
}

void emit(const Options& o, std::uint32_t embedding_rows, std::uint64_t allocated,
          const std::vector<ModeResult>& results) {
    std::cout << std::setprecision(9)
              << "{\n  \"benchmark\": \"cpu-reference-pair-component\",\n"
              << "  \"baseline\": \"same_cpu_reference_capacity_zero\",\n"
              << "  \"optimized_engine_comparison\": false,\n"
              << "  \"compiler_version\": ";
#if defined(__VERSION__)
    json_string(__VERSION__);
#elif defined(_MSC_VER)
    json_string("MSVC");
#else
    json_string("not reported by compiler");
#endif
    std::cout << ",\n"
              << "  \"synthetic_inputs\": true,\n  \"single_thread\": true,\n"
              << "  \"device_measurement\": false,\n  \"end_to_end_inference\": false,\n"
              << "  \"seed\": \"0x5168724c6f6f6b31\",\n"
              << "  \"vocabulary_size\": " << o.vocab << ",\n"
              << "  \"embedding_rows\": " << embedding_rows << ",\n"
              << "  \"rank\": " << SL_PAIR_RANK << ",\n"
              << "  \"capacity_entries\": " << o.capacity << ",\n"
              << "  \"steps_per_repeat\": " << o.steps << ",\n"
              << "  \"repeats\": " << o.repeats << ",\n"
              << "  \"estimated_input_and_payload_bytes\": " << allocated << ",\n"
              << "  \"warm_includes_initial_miss\": true,\n"
              << "  \"reset_between_repeats\": true,\n"
              << "  \"alternating_run_order\": true,\n"
              << "  \"quantile_method\": \"nearest_rank\",\n"
              << "  \"all_outputs_equivalent\": true,\n  \"modes\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "    {\"mode\": \"" << r.mode << "\", \"hits\": " << r.hits
                  << ", \"misses\": " << r.misses
                  << ", \"baseline_total_ns\": " << r.baseline_ns
                  << ", \"cached_total_ns\": " << r.cached_ns
                  << ", \"baseline_p50_step_ns\": " << r.baseline_p50
                  << ", \"baseline_p95_step_ns\": " << r.baseline_p95
                  << ", \"cached_p50_step_ns\": " << r.cached_p50
                  << ", \"cached_p95_step_ns\": " << r.cached_p95
                  << ", \"component_speed_ratio\": ";
        if (r.cached_ns == 0) std::cout << "null";
        else std::cout << static_cast<double>(r.baseline_ns) / static_cast<double>(r.cached_ns);
        std::cout << ", \"all_outputs_equivalent\": true}" << (i + 1U == results.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: sl_pair_benchmark [--vocab N] [--steps N] [--repeats N] [--capacity N]\n"
                     "Synthetic single-thread CPU component timing; JSON output. No GPU or full-model inference.\n";
        return 0;
    }
    try {
        const auto o = options(argc, argv);
        const auto embedding_rows = std::max({o.vocab, o.steps + 1U, o.capacity + 1U, 4U});
        const std::uint64_t input_bytes =
            (static_cast<std::uint64_t>(o.vocab) + embedding_rows) * SL_PAIR_RANK * sizeof(std::uint16_t) +
            static_cast<std::uint64_t>(o.vocab) * (o.steps * sizeof(float) + sizeof(std::uint32_t));
        const std::uint64_t payload_bytes = static_cast<std::uint64_t>(o.vocab) * o.capacity * sizeof(float);
        const auto allocated = input_bytes + payload_bytes;
        if (allocated > memory_limit) throw std::invalid_argument("input and cache payload exceed the 512 MiB benchmark limit");
        Random random;
        std::vector<std::uint16_t> head(static_cast<std::size_t>(o.vocab) * SL_PAIR_RANK);
        std::vector<std::uint16_t> embeddings(static_cast<std::size_t>(embedding_rows) * SL_PAIR_RANK);
        std::vector<std::uint32_t> ids(o.vocab);
        std::vector<float> logits(static_cast<std::size_t>(o.steps) * o.vocab);
        for (auto& value : head) value = random.half();
        for (auto& value : embeddings) value = random.half();
        for (std::uint32_t row = 0; row < o.vocab; ++row) ids[row] = o.vocab - 1U - row;
        for (auto& value : logits) value = static_cast<float>(static_cast<int>(random.next() % 20001U) - 10000) / 512.0F;

        sl_pair_config config{};
        sl_pair_config_init(&config);
        config.vocabulary_size = o.vocab;
        config.embedding_rows = embedding_rows;
        config.head_layout = SL_PAIR_HEAD_SELECTED;
        config.head = head.data();
        config.embeddings = embeddings.data();
        config.token_ids = ids.data();
        config.capacity_entries = 0;
        std::memcpy(config.identity.weights, &data_seed, sizeof(data_seed));
        config.identity.numerics[0] = 1;
        config.identity.vocabulary[0] = 1;
        auto baseline = create(config);
        config.capacity_entries = o.capacity;
        config.payload_budget_bytes = payload_bytes;
        auto cached = create(config);
        std::vector<ModeResult> results;
        results.reserve(4);
        for (const std::string_view mode : {"warm", "cold", "mixed", "collision"})
            results.push_back(benchmark(mode, o, embedding_rows, logits, baseline.get(), cached.get()));
        emit(o, embedding_rows, allocated, results);
    } catch (const std::exception& error) {
        std::cerr << "pair benchmark: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

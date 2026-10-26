#include "../support/device_runtime.hpp"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>

using namespace sl_device_support;

namespace {

// Three blocks with a ragged final block exercise the publication/reduction
// boundary; a single-block toy would not qualify the production path.
constexpr std::uint32_t vocabulary = 513;
constexpr std::uint32_t embedding_rows = 1024;
constexpr std::uint16_t values[]{0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0xb400, 0x0001, 0x7bff};

struct fixture {
    stream queue;
    std::vector<std::uint32_t> ids;
    std::vector<std::uint16_t> weights;
    std::vector<std::uint16_t> embeddings;
    std::vector<float> base;
    buffer<std::uint32_t> device_ids{vocabulary};
    buffer<std::uint16_t> device_weights;
    buffer<std::uint16_t> device_embeddings{embedding_rows * SL_PAIR_RANK};
    buffer<float> base_head{vocabulary}, base_tail{vocabulary};
    buffer<float> markov{SL_PAIR_RANK}, bias{vocabulary};
    buffer<std::uint32_t> predecessor{1};
    buffer<sl_device_result> result{1};
    sl_pair_config config{};
    sl_pair_engine* cpu = nullptr;
    std::unique_ptr<engine> gpu;

    explicit fixture(std::uint32_t layout, std::uint32_t capacity = 2, bool zero_weights = false)
        : ids(vocabulary), weights((layout == SL_PAIR_HEAD_GLOBAL ? embedding_rows : vocabulary) * SL_PAIR_RANK),
          embeddings(embedding_rows * SL_PAIR_RANK), base(vocabulary), device_weights(weights.size()) {
        for (std::uint32_t row = 0; row < vocabulary; ++row) {
            ids[row] = (row * 17u + 7u) % embedding_rows;
            base[row] = static_cast<float>(static_cast<int>(row % 9u) - 4) * 0.25f;
        }
        const std::uint32_t head_rows = layout == SL_PAIR_HEAD_GLOBAL ? embedding_rows : vocabulary;
        for (std::uint32_t row = 0; row < head_rows; ++row) {
            const auto global = layout == SL_PAIR_HEAD_GLOBAL ? row : ids[row];
            for (unsigned k = 0; k < SL_PAIR_RANK; ++k)
                weights[static_cast<std::size_t>(row) * SL_PAIR_RANK + k] =
                    zero_weights ? std::uint16_t{0} : values[(global * 13u + k * 5u) % 8u];
        }
        for (std::uint32_t row = 0; row < embedding_rows; ++row)
            for (unsigned k = 0; k < SL_PAIR_RANK; ++k)
                embeddings[static_cast<std::size_t>(row) * SL_PAIR_RANK + k] = values[(row * 7u + k * 3u + 1u) % 8u];
        sl_pair_config_init(&config);
        config.vocabulary_size = vocabulary;
        config.embedding_rows = embedding_rows;
        config.capacity_entries = capacity;
        config.head_layout = layout;
        config.head = weights.data();
        config.embeddings = embeddings.data();
        config.token_ids = ids.data();
        check_status(sl_pair_create(&config, &cpu), "create CPU reference");
        device_ids.upload(ids.data(), queue.value);
        device_weights.upload(weights.data(), queue.value);
        device_embeddings.upload(embeddings.data(), queue.value);
        auto device_config = config;
        device_config.head = device_weights.data;
        device_config.embeddings = device_embeddings.data;
        device_config.token_ids = device_ids.data;
        gpu = std::make_unique<engine>(device_config, queue.value);
    }

    ~fixture() { if (cpu) sl_pair_destroy(cpu); }

    void prepare_base(std::uint32_t split) {
        std::vector<float> head(vocabulary, std::bit_cast<float>(0x7fc00001u));
        std::vector<float> tail(vocabulary, std::bit_cast<float>(0x7fc00002u));
        std::copy(base.begin(), base.begin() + split, head.begin());
        std::copy(base.begin() + split, base.end(), tail.begin());
        base_head.upload(head.data(), queue.value);
        base_tail.upload(tail.data(), queue.value);
        queue.wait();
    }

    void compare(sl_status expected_status, const sl_pair_result& expected,
                 const std::vector<float>& expected_markov,
                 const std::vector<float>& expected_bias) {
        sl_device_result actual{};
        std::vector<float> actual_markov(SL_PAIR_RANK), actual_bias(vocabulary);
        result.download(&actual, queue.value);
        markov.download(actual_markov.data(), queue.value);
        bias.download(actual_bias.data(), queue.value);
        queue.wait();
        require(actual.status == static_cast<std::uint32_t>(expected_status), "device status differs");
        require(actual.selection.token == expected.token, "global token differs");
        require(actual.selection.vocabulary_row == expected.vocabulary_row, "vocabulary row differs");
        require(actual.selection.cache_hit == expected.cache_hit, "cache hit differs");
        require(std::bit_cast<std::uint32_t>(actual.selection.score) == std::bit_cast<std::uint32_t>(expected.score), "score bits differ");
        require(std::memcmp(actual_markov.data(), expected_markov.data(), SL_PAIR_RANK * sizeof(float)) == 0, "embedding bits differ");
        require(std::memcmp(actual_bias.data(), expected_bias.data(), vocabulary * sizeof(float)) == 0, "bias bits differ");
    }

    sl_pair_result run(std::uint32_t previous, std::uint32_t split = vocabulary) {
        prepare_base(split);
        std::vector<float> expected_markov(SL_PAIR_RANK), expected_bias(vocabulary);
        sl_pair_result expected{};
        const auto status = sl_pair_step(cpu, previous, base.data(), &expected,
                                        expected_markov.data(), expected_bias.data());
        require(status == SL_OK || status == SL_INVALID_STATE, "unexpected reference status");
        predecessor.upload(&previous, queue.value);
        gpu->enqueue(predecessor.data, split ? base_head.data : nullptr,
                     split < vocabulary ? base_tail.data : nullptr, split, result.data,
                     markov.data, bias.data);
        compare(status, expected, expected_markov, expected_bias);
        return expected;
    }

    void reset() {
        check_status(sl_pair_reset(cpu, nullptr), "reset CPU reference");
        gpu->reset();
    }

    void chain(bool capture) {
        reset();
        prepare_base(vocabulary);
        sl_device_result seed{};
        seed.selection.token = 5;
        result.upload(&seed, queue.value);
        queue.wait();
        SL_GPU_GRAPH graph{};
        SL_GPU_GRAPH_EXEC executable{};
        if (capture) check(SL_GPU_RUNTIME(StreamBeginCapture)(queue.value, SL_GPU_CAPTURE_MODE), "begin capture");
        for (unsigned i = 0; i < 3; ++i)
            gpu->enqueue(token_address(result.data), base_head.data, nullptr, vocabulary,
                         result.data, markov.data, bias.data);
        if (capture) {
            check(SL_GPU_RUNTIME(StreamEndCapture)(queue.value, &graph), "end capture");
            check(SL_GPU_RUNTIME(GraphInstantiateWithFlags)(&executable, graph, 0), "instantiate graph");
        }
        std::uint32_t previous = 5;
        const unsigned replays = capture ? 2u : 1u;
        for (unsigned replay = 0; replay < replays; ++replay) {
            std::vector<float> expected_markov(SL_PAIR_RANK), expected_bias(vocabulary);
            sl_pair_result expected{};
            sl_status status = SL_OK;
            for (unsigned i = 0; i < 3; ++i) {
                status = sl_pair_step(cpu, previous, base.data(), &expected, expected_markov.data(), expected_bias.data());
                require(status == SL_OK, "chain reference failed");
                previous = expected.token;
            }
            if (capture) check(SL_GPU_RUNTIME(GraphLaunch)(executable, queue.value), "launch graph");
            compare(status, expected, expected_markov, expected_bias);
        }
        if (executable) check(SL_GPU_RUNTIME(GraphExecDestroy)(executable), "destroy graph executable");
        if (graph) check(SL_GPU_RUNTIME(GraphDestroy)(graph), "destroy graph");
    }
};

void test_layout(std::uint32_t layout, std::uint32_t capacity) {
    fixture f(layout, capacity);
    require(f.run(5).cache_hit == 0, "cold step reported a hit");
    require(f.run(5, 13).cache_hit == (capacity ? 1u : 0u), "warm step hit mismatch");
    f.run(5, 0);
    if (capacity) {
        std::uint32_t collision = 0;
        while (collision == 5 || sl_pair_slot_for(collision, capacity) != sl_pair_slot_for(5, capacity)) ++collision;
        f.run(collision);
        require(f.run(5).cache_hit == 0, "collision did not evict");
    }
    f.reset();
    require(f.run(5).cache_hit == 0, "reset did not invalidate");

    // Force exact score ties, including signed zeros, then require global-ID order.
    std::vector<float> local_bias(vocabulary);
    sl_pair_result ignored{};
    check_status(sl_pair_step(f.cpu, 5, f.base.data(), &ignored, nullptr, local_bias.data()), "obtain tie bias");
    for (std::size_t i = 0; i < local_bias.size(); ++i) f.base[i] = -local_bias[i];
    const auto tied = f.run(5);
    require(tied.token == *std::min_element(f.ids.begin(), f.ids.end()), "tie did not prefer smaller global ID");
    std::fill(f.base.begin(), f.base.end(), -std::numeric_limits<float>::infinity());
    const auto negative_infinity = f.run(5);
    require(negative_infinity.token == *std::min_element(f.ids.begin(), f.ids.end()), "all-infinity selection failed");
    std::fill(f.base.begin(), f.base.end(), std::bit_cast<float>(0x7fc12345u));
    f.run(5);

    const std::uint32_t invalid = embedding_rows;
    f.predecessor.upload(&invalid, f.queue.value);
    f.gpu->enqueue(f.predecessor.data, f.base_head.data, nullptr, vocabulary,
                   f.result.data, f.markov.data, f.bias.data);
    sl_device_result invalid_result{};
    f.result.download(&invalid_result, f.queue.value);
    f.queue.wait();
    require(invalid_result.status == SL_INVALID_ARGUMENT && invalid_result.selection.token == UINT32_MAX,
            "invalid predecessor was not reported");
    std::fill(f.base.begin(), f.base.end(), 0.0f);
    f.chain(false);
    f.chain(true);
    check_status(SL_GPU_PAIR(query)(f.gpu->value), "query completed stream");
}

void invalid_mapping() {
    fixture f(SL_PAIR_HEAD_GLOBAL);
    auto bad_ids = f.ids;
    bad_ids[1] = bad_ids[0];
    buffer<std::uint32_t> device_bad(vocabulary);
    device_bad.upload(bad_ids.data(), f.queue.value);
    auto config = f.config;
    config.head = f.device_weights.data;
    config.embeddings = f.device_embeddings.data;
    config.token_ids = device_bad.data;
    SL_GPU_ENGINE* rejected = nullptr;
    require(SL_GPU_PAIR(create)(&config, reinterpret_cast<void*>(f.queue.value), &rejected) == SL_INVALID_ARGUMENT && rejected == nullptr,
            "duplicate candidate IDs were accepted");
    bad_ids[1] = embedding_rows;
    device_bad.upload(bad_ids.data(), f.queue.value);
    require(SL_GPU_PAIR(create)(&config, reinterpret_cast<void*>(f.queue.value), &rejected) == SL_INVALID_ARGUMENT && rejected == nullptr,
            "out-of-range candidate ID was accepted");
}

void subnormal_scores() {
    fixture f(SL_PAIR_HEAD_GLOBAL, 2, true);
    std::fill(f.base.begin(), f.base.end(), 0.0f);
    f.base[17] = std::bit_cast<float>(1u);
    require(f.run(5).token == f.ids[17], "FP32 subnormal score was flushed on a miss");
    require(f.run(5).token == f.ids[17], "FP32 subnormal score was flushed on a hit");
}

}  // namespace

int main() {
    try {
        std::string reason;
        if (!available(reason)) {
            std::printf("SKIP %s device qualification: %s\n", SL_GPU_LABEL, reason.c_str());
            return 77;
        }
        test_layout(SL_PAIR_HEAD_GLOBAL, 2);
        test_layout(SL_PAIR_HEAD_SELECTED, 2);
        test_layout(SL_PAIR_HEAD_GLOBAL, 0);
        invalid_mapping();
        subnormal_scores();
        std::printf("PASS %s device qualification: exact finite bias/embedding/score, cache lifecycle, mapping, NaN/ties, chaining and graph replay\n", SL_GPU_LABEL);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s device qualification: %s\n", SL_GPU_LABEL, error.what());
        return 1;
    }
}

CXX ?= c++
CC ?= cc
AR ?= ar
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O2 -g
override CXXFLAGS += -std=c++20 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
BUILD_DIR ?= build-make

.PHONY: all test test-tools example benchmark clean
all: $(BUILD_DIR)/libstrix_lookahead.a

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/cache.o: src/cache.cpp include/strix_lookahead/cache.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/pair.o: src/pair.cpp include/strix_lookahead/pair.h include/strix_lookahead/cache.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/selection.o: src/selection.cpp include/strix_lookahead/selection.h include/strix_lookahead/cache.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/libstrix_lookahead.a: $(BUILD_DIR)/cache.o $(BUILD_DIR)/pair.o $(BUILD_DIR)/selection.o
	$(AR) rcs $@ $^

$(BUILD_DIR)/sl_cache_test: tests/cache_test.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

$(BUILD_DIR)/sl_pair_test: tests/pair_test.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

$(BUILD_DIR)/sl_selection_test: tests/selection_test.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

$(BUILD_DIR)/c_api_smoke.o: tests/c_api_smoke.c include/strix_lookahead/cache.h include/strix_lookahead/pair.h include/strix_lookahead/selection.h include/strix_lookahead/device.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sl_c_api_smoke: $(BUILD_DIR)/c_api_smoke.o $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/sl_cache_example: examples/cache_example.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

$(BUILD_DIR)/sl_pair_example: examples/pair_example.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

test: $(BUILD_DIR)/sl_cache_test $(BUILD_DIR)/sl_pair_test $(BUILD_DIR)/sl_selection_test $(BUILD_DIR)/sl_c_api_smoke
	$(BUILD_DIR)/sl_cache_test
	$(BUILD_DIR)/sl_pair_test
	$(BUILD_DIR)/sl_selection_test
	$(BUILD_DIR)/sl_c_api_smoke

test-tools:
	python3 tests/evidence_tools_test.py
	python3 tests/engine_qualification_test.py
	python3 tests/predecessor_trace_test.py

example: $(BUILD_DIR)/sl_cache_example $(BUILD_DIR)/sl_pair_example
	$(BUILD_DIR)/sl_cache_example
	$(BUILD_DIR)/sl_pair_example

$(BUILD_DIR)/sl_pair_benchmark: examples/pair_benchmark.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

benchmark: $(BUILD_DIR)/sl_pair_benchmark
	$(BUILD_DIR)/sl_pair_benchmark

clean:
	rm -rf -- $(BUILD_DIR)

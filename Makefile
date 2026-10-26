CXX ?= c++
CC ?= cc
AR ?= ar
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++20 -Wall -Wextra -Wpedantic
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic
BUILD_DIR ?= build-make

.PHONY: all test example clean
all: $(BUILD_DIR)/libstrix_lookahead.a

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/cache.o: src/cache.cpp include/strix_lookahead/cache.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/libstrix_lookahead.a: $(BUILD_DIR)/cache.o
	$(AR) rcs $@ $<

$(BUILD_DIR)/sl_cache_test: tests/cache_test.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

$(BUILD_DIR)/c_api_smoke.o: tests/c_api_smoke.c include/strix_lookahead/cache.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sl_c_api_smoke: $(BUILD_DIR)/c_api_smoke.o $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD_DIR)/sl_cache_example: examples/cache_example.cpp $(BUILD_DIR)/libstrix_lookahead.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD_DIR)/libstrix_lookahead.a $(LDFLAGS) -o $@

test: $(BUILD_DIR)/sl_cache_test $(BUILD_DIR)/sl_c_api_smoke
	$(BUILD_DIR)/sl_cache_test
	$(BUILD_DIR)/sl_c_api_smoke

example: $(BUILD_DIR)/sl_cache_example
	$(BUILD_DIR)/sl_cache_example

clean:
	rm -rf -- $(BUILD_DIR)

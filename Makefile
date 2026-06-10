# N-body galaxy simulator — CPU core.
#
# Targets:
#   make            build tests + bench (release, -O3)
#   make test       build and run the physics test suite
#   make bench      build and run the benchmark sweep (writes bench/results.csv)
#   make clean
#
# No -ffast-math anywhere: it would let the compiler swap the scalar
# baseline's sqrt for the same rsqrt-estimate trick the NEON kernel uses by
# hand, which would both muddy the benchmark comparison and change numerics
# the tests depend on.

CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Isrc
LDFLAGS  :=

BUILD := build
CORE_SRCS := $(wildcard src/nbody/*.cpp)
CORE_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CORE_SRCS))
HDRS := $(wildcard src/nbody/*.h)

.PHONY: all test bench clean

all: $(BUILD)/nbody_tests $(BUILD)/nbody_bench

$(BUILD)/%.o: src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/nbody_tests: src/test_main.cpp $(CORE_OBJS) $(HDRS)
	$(CXX) $(CXXFLAGS) src/test_main.cpp $(CORE_OBJS) -o $@ $(LDFLAGS)

$(BUILD)/nbody_bench: src/bench_main.cpp $(CORE_OBJS) $(HDRS)
	$(CXX) $(CXXFLAGS) src/bench_main.cpp $(CORE_OBJS) -o $@ $(LDFLAGS)

test: $(BUILD)/nbody_tests
	./$(BUILD)/nbody_tests

bench: $(BUILD)/nbody_bench
	./$(BUILD)/nbody_bench

clean:
	rm -rf $(BUILD)

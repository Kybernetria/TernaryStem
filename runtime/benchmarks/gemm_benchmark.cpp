#include "ternarystem/kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

template <class Function>
double median_ms(Function function) {
  std::vector<double> samples;
  function();
  for (int run = 0; run < 11; ++run) {
    const auto start = Clock::now();
    function();
    samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int main(int argc, char** argv) {
  const std::size_t rows = argc > 1 ? std::stoul(argv[1]) : 64;
  const std::size_t outputs = argc > 2 ? std::stoul(argv[2]) : 256;
  const std::size_t inner = argc > 3 ? std::stoul(argv[3]) : 512;
  std::mt19937 random(20250218);
  std::uniform_int_distribution<int> activation_dist(-127, 127), ternary_dist(-1, 1);
  std::vector<std::int8_t> inputs(rows * inner), weights(outputs * inner);
  for (auto& value : inputs) value = static_cast<std::int8_t>(activation_dist(random));
  for (auto& value : weights) value = static_cast<std::int8_t>(ternary_dist(random));
  std::vector<std::uint8_t> packed;
  for (std::size_t out = 0; out < outputs; ++out) {
    auto row = ternarystem::pack_ternary(weights.data() + out * inner, inner);
    packed.insert(packed.end(), row.begin(), row.end());
  }
  std::vector<std::int32_t> output(rows * outputs);
  const auto int8_ms = median_ms([&] {
    ternarystem::int8_gemm_scalar(
        inputs.data(), weights.data(), output.data(), rows, outputs, inner);
  });
  const auto ternary_ms = median_ms([&] {
    ternarystem::ternary_gemm_scalar(
        inputs.data(), packed.data(), output.data(), rows, outputs, inner);
  });
  std::cout << "{\"rows\":" << rows << ",\"outputs\":" << outputs
            << ",\"inner\":" << inner
            << ",\"int8_scalar_median_ms\":" << int8_ms
            << ",\"ternary_scalar_median_ms\":" << ternary_ms;
#ifdef TERNARYSTEM_HAS_AVX2
  if (ternarystem::avx2_available()) {
    const auto int8_avx2_ms = median_ms([&] {
      ternarystem::int8_gemm_avx2(
          inputs.data(), weights.data(), output.data(), rows, outputs, inner);
    });
    const auto ternary_avx2_ms = median_ms([&] {
      ternarystem::ternary_gemm_avx2(
          inputs.data(), packed.data(), output.data(), rows, outputs, inner);
    });
    std::cout << ",\"int8_avx2_median_ms\":" << int8_avx2_ms
              << ",\"ternary_avx2_median_ms\":" << ternary_avx2_ms
              << ",\"ternary_vs_int8_speedup\":" << int8_avx2_ms / ternary_avx2_ms;
  }
#endif
  std::cout << ",\"note\":\"single-thread prototype; optimized-library INT8 comparison still required\"}\n";
}

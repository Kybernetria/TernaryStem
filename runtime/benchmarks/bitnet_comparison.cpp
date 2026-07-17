#include "ternarystem/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

extern "C" void ggml_gemm_i2_i8_s(
    int n, float* output, std::size_t stride, const void* weights,
    const void* activations, int rows, int outputs);

using Clock = std::chrono::steady_clock;

template <class Function>
double median_ms(Function function) {
  std::vector<double> samples;
  function();
  for (int run = 0; run < 11; ++run) {
    const auto start = Clock::now();
    function();
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<std::uint8_t> pack_bitnet(
    const std::vector<std::int8_t>& weights, std::size_t outputs, std::size_t inner) {
  std::vector<std::uint8_t> packed(outputs * inner / 4, 0);
  for (std::size_t out = 0; out < outputs; ++out) {
    for (std::size_t block = 0; block < inner; block += 128) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        auto& byte = packed[out * inner / 4 + block / 4 + lane];
        for (std::size_t quarter = 0; quarter < 4; ++quarter) {
          const std::size_t k = block + quarter * 32 + lane;
          const auto code = static_cast<std::uint8_t>(weights[out * inner + k] + 1);
          byte |= code << (6 - 2 * quarter);
        }
      }
    }
  }
  return packed;
}

int main(int argc, char** argv) {
  const std::size_t rows = argc > 1 ? std::stoul(argv[1]) : 64;
  const std::size_t outputs = argc > 2 ? std::stoul(argv[2]) : 256;
  const std::size_t inner = argc > 3 ? std::stoul(argv[3]) : 512;
  if (inner % 128 != 0) {
    std::cerr << "BitNet I2_S requires K divisible by 128\n";
    return 2;
  }
  std::mt19937 random(20250218);
  std::uniform_int_distribution<int> activation_dist(-127, 127), ternary_dist(-1, 1);
  std::vector<std::int8_t> inputs(rows * inner), weights(outputs * inner);
  for (auto& value : inputs) value = static_cast<std::int8_t>(activation_dist(random));
  for (auto& value : weights) value = static_cast<std::int8_t>(ternary_dist(random));
  const auto bitnet_weights = pack_bitnet(weights, outputs, inner);
  std::vector<std::uint8_t> local_weights;
  for (std::size_t out = 0; out < outputs; ++out) {
    auto packed = ternarystem::pack_ternary(weights.data() + out * inner, inner);
    local_weights.insert(local_weights.end(), packed.begin(), packed.end());
  }
  std::vector<std::int32_t> reference(rows * outputs), local(rows * outputs);
  std::vector<float> bitnet(rows * outputs);
  ternarystem::int8_gemm_scalar(
      inputs.data(), weights.data(), reference.data(), rows, outputs, inner);

  const auto run_bitnet = [&] {
    ggml_gemm_i2_i8_s(static_cast<int>(inner), bitnet.data(), outputs,
                      bitnet_weights.data(), inputs.data(),
                      static_cast<int>(rows), static_cast<int>(outputs));
    // I2_S computes code*x for codes {0,1,2}; ternary value is code-1.
    for (std::size_t row = 0; row < rows; ++row) {
      std::int32_t correction = 0;
      for (std::size_t k = 0; k < inner; ++k) correction += inputs[row * inner + k];
      for (std::size_t out = 0; out < outputs; ++out)
        bitnet[row * outputs + out] -= static_cast<float>(correction);
    }
  };
  run_bitnet();
  for (std::size_t index = 0; index < reference.size(); ++index) {
    if (std::lround(bitnet[index]) != reference[index]) {
      std::cerr << "BitNet adapter disagrees with ternary reference at " << index
                << ": expected " << reference[index] << ", got " << bitnet[index] << "\n";
      return 1;
    }
  }
  const auto bitnet_ms = median_ms(run_bitnet);
#ifdef TERNARYSTEM_HAS_AVX2
  const auto local_ms = median_ms([&] {
    ternarystem::ternary_gemm_avx2(
        inputs.data(), local_weights.data(), local.data(), rows, outputs, inner);
  });
  const auto int8_ms = median_ms([&] {
    ternarystem::int8_gemm_avx2(
        inputs.data(), weights.data(), local.data(), rows, outputs, inner);
  });
  std::cout << "{\"rows\":" << rows << ",\"outputs\":" << outputs
            << ",\"inner\":" << inner << ",\"bitnet_i2s_ms\":" << bitnet_ms
            << ",\"local_avx2_ms\":" << local_ms
            << ",\"int8_avx2_ms\":" << int8_ms
            << ",\"bitnet_vs_local_speedup\":" << local_ms / bitnet_ms
            << ",\"bitnet_vs_int8_speedup\":" << int8_ms / bitnet_ms << "}\n";
#else
  std::cout << "{\"bitnet_i2s_ms\":" << bitnet_ms << "}\n";
#endif
}

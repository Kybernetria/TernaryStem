#include "ternarystem/kernels.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t rows = 3, outputs = 5, inner = 67;
  std::vector<std::int8_t> input(rows * inner), weights(outputs * inner);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<std::int8_t>((i * 17) % 255 - 127);
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<std::int8_t>(i % 3) - 1;
  // Rows must begin at byte boundaries in the version-1 packed layout.
  std::vector<std::uint8_t> packed;
  for (std::size_t out = 0; out < outputs; ++out) {
    auto row = ternarystem::pack_ternary(weights.data() + out * inner, inner);
    packed.insert(packed.end(), row.begin(), row.end());
  }
  std::vector<std::int32_t> expected(rows * outputs), actual(rows * outputs);
  ternarystem::int8_gemm_scalar(input.data(), weights.data(), expected.data(), rows, outputs, inner);
  ternarystem::ternary_gemm_scalar(input.data(), packed.data(), actual.data(), rows, outputs, inner);
  if (actual != expected) {
    std::cerr << "packed scalar GEMM differs from integer reference\n";
    return 1;
  }
#ifdef TERNARYSTEM_HAS_AVX2
  if (ternarystem::avx2_available()) {
    ternarystem::ternary_gemm_avx2(
        input.data(), packed.data(), actual.data(), rows, outputs, inner);
    if (actual != expected) {
      std::cerr << "AVX2 packed GEMM differs from integer reference\n";
      return 1;
    }
    ternarystem::int8_gemm_avx2(
        input.data(), weights.data(), actual.data(), rows, outputs, inner);
    if (actual != expected) {
      std::cerr << "AVX2 INT8 GEMM differs from integer reference\n";
      return 1;
    }
  }
#endif
  return 0;
}

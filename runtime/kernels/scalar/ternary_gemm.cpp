#include "ternarystem/kernels.h"

#include <stdexcept>

namespace ternarystem {
namespace {
inline std::int8_t decode(std::uint8_t bits) {
  return bits == 1 ? 1 : (bits == 2 ? -1 : 0);
}
}  // namespace

std::vector<std::uint8_t> pack_ternary(const std::int8_t* weights, std::size_t count) {
  std::vector<std::uint8_t> packed((count + 3) / 4, 0);
  for (std::size_t index = 0; index < count; ++index) {
    const auto weight = weights[index];
    if (weight < -1 || weight > 1) throw std::invalid_argument("non-ternary weight");
    const std::uint8_t code = weight == 1 ? 1 : (weight == -1 ? 2 : 0);
    packed[index / 4] |= static_cast<std::uint8_t>(code << (2 * (index % 4)));
  }
  return packed;
}

void ternary_gemm_scalar(const std::int8_t* activations,
                         const std::uint8_t* packed_weights,
                         std::int32_t* output,
                         std::size_t rows,
                         std::size_t outputs,
                         std::size_t inner) {
  const std::size_t packed_stride = (inner + 3) / 4;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < outputs; ++out) {
      std::int32_t sum = 0;
      const auto* weights = packed_weights + out * packed_stride;
      for (std::size_t k = 0; k < inner; ++k) {
        const auto code = static_cast<std::uint8_t>((weights[k / 4] >> (2 * (k % 4))) & 3);
        sum += static_cast<std::int32_t>(activations[row * inner + k]) * decode(code);
      }
      output[row * outputs + out] = sum;
    }
  }
}

void int8_gemm_scalar(const std::int8_t* activations,
                      const std::int8_t* weights,
                      std::int32_t* output,
                      std::size_t rows,
                      std::size_t outputs,
                      std::size_t inner) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < outputs; ++out) {
      std::int32_t sum = 0;
      for (std::size_t k = 0; k < inner; ++k)
        sum += static_cast<std::int32_t>(activations[row * inner + k]) * weights[out * inner + k];
      output[row * outputs + out] = sum;
    }
  }
}

}  // namespace ternarystem

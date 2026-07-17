#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ternarystem {

// Encoding: 00=0, 01=+1, 10=-1; four weights per byte, low bits first.
std::vector<std::uint8_t> pack_ternary(const std::int8_t* weights, std::size_t count);
void ternary_gemm_scalar(const std::int8_t* activations,
                         const std::uint8_t* packed_weights,
                         std::int32_t* output,
                         std::size_t rows,
                         std::size_t outputs,
                         std::size_t inner);
void int8_gemm_scalar(const std::int8_t* activations,
                      const std::int8_t* weights,
                      std::int32_t* output,
                      std::size_t rows,
                      std::size_t outputs,
                      std::size_t inner);

#ifdef TERNARYSTEM_HAS_AVX2
bool avx2_available();
void ternary_gemm_avx2(const std::int8_t* activations,
                       const std::uint8_t* packed_weights,
                       std::int32_t* output,
                       std::size_t rows,
                       std::size_t outputs,
                       std::size_t inner);
void int8_gemm_avx2(const std::int8_t* activations,
                    const std::int8_t* weights,
                    std::int32_t* output,
                    std::size_t rows,
                    std::size_t outputs,
                    std::size_t inner);
#endif

}  // namespace ternarystem

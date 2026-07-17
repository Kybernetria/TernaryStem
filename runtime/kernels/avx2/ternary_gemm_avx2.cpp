#include "ternarystem/kernels.h"

#include <cstdint>
#include <immintrin.h>

namespace ternarystem {
namespace {

inline __m256i decode_32(const std::uint8_t* packed) {
  const __m128i packed8 = _mm_loadl_epi64(
      reinterpret_cast<const __m128i*>(packed));
  const __m128i upper4 = _mm_srli_si128(packed8, 4);
  const __m256i lanes = _mm256_set_m128i(upper4, packed8);
  const __m256i low = _mm256_and_si256(lanes, _mm256_set1_epi8(0x0f));
  const __m256i high = _mm256_and_si256(
      _mm256_srli_epi16(lanes, 4), _mm256_set1_epi8(0x0f));
  const __m256i nibbles = _mm256_unpacklo_epi8(low, high);
  const __m256i duplicated = _mm256_unpacklo_epi8(nibbles, nibbles);
  const __m128i first_table = _mm_setr_epi8(
      0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0);
  const __m128i second_table = _mm_setr_epi8(
      0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0);
  const __m256i first = _mm256_shuffle_epi8(
      _mm256_broadcastsi128_si256(first_table), duplicated);
  const __m256i second = _mm256_shuffle_epi8(
      _mm256_broadcastsi128_si256(second_table), duplicated);
  const __m256i odd_mask = _mm256_setr_epi8(
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
  return _mm256_blendv_epi8(first, second, odd_mask);
}

inline std::int32_t horizontal_sum(__m256i values) {
  const __m128i low = _mm256_castsi256_si128(values);
  const __m128i high = _mm256_extracti128_si256(values, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

inline std::int8_t decode(std::uint8_t code) {
  return code == 1 ? 1 : (code == 2 ? -1 : 0);
}

}  // namespace

bool avx2_available() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#else
  return true;
#endif
}

void int8_gemm_avx2(const std::int8_t* activations,
                    const std::int8_t* weights,
                    std::int32_t* output,
                    std::size_t rows,
                    std::size_t outputs,
                    std::size_t inner) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < outputs; ++out) {
      const auto* input = activations + row * inner;
      const auto* weight = weights + out * inner;
      __m256i accumulator = _mm256_setzero_si256();
      std::size_t k = 0;
      for (; k + 32 <= inner; k += 32) {
        const __m256i input8 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input + k));
        const __m256i weight8 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weight + k));
        const __m256i products_low = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(input8)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(weight8)));
        const __m256i products_high = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(input8, 1)),
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight8, 1)));
        accumulator = _mm256_add_epi32(accumulator, products_low);
        accumulator = _mm256_add_epi32(accumulator, products_high);
      }
      std::int32_t sum = horizontal_sum(accumulator);
      for (; k < inner; ++k) {
        sum += static_cast<std::int32_t>(input[k]) * weight[k];
      }
      output[row * outputs + out] = sum;
    }
  }
}

void ternary_gemm_avx2(const std::int8_t* activations,
                       const std::uint8_t* packed_weights,
                       std::int32_t* output,
                       std::size_t rows,
                       std::size_t outputs,
                       std::size_t inner) {
  const std::size_t packed_stride = (inner + 3) / 4;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < outputs; ++out) {
      const auto* input = activations + row * inner;
      const auto* weights = packed_weights + out * packed_stride;
      __m256i accumulator = _mm256_setzero_si256();
      std::size_t k = 0;
      for (; k + 32 <= inner; k += 32) {
        const __m256i weight8 = decode_32(weights + k / 4);
        const __m256i input8 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(input + k));
        const __m256i signed8 = _mm256_sign_epi8(input8, weight8);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i products_low = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(signed8)), ones16);
        const __m256i products_high = _mm256_madd_epi16(
            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(signed8, 1)), ones16);
        accumulator = _mm256_add_epi32(accumulator, products_low);
        accumulator = _mm256_add_epi32(accumulator, products_high);
      }
      std::int32_t sum = horizontal_sum(accumulator);
      for (; k < inner; ++k) {
        const auto code = static_cast<std::uint8_t>(
            (weights[k / 4] >> (2 * (k % 4))) & 3);
        sum += static_cast<std::int32_t>(input[k]) * decode(code);
      }
      output[row * outputs + out] = sum;
    }
  }
}

}  // namespace ternarystem

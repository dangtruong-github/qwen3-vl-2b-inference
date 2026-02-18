#pragma once

#include <immintrin.h>

static inline float add_reduce_mm_256(__m256 vec) {
    // Step 1: Split into two 128-bit halves
    __m128 low  = _mm256_castps256_ps128(vec);          // lower 128 bits
    __m128 high = _mm256_extractf128_ps(vec, 1);        // upper 128 bits

    // Step 2: Add the halves together
    __m128 sum128 = _mm_add_ps(low, high);

    // Step 3: Horizontal add within 128-bit register
    sum128 = _mm_hadd_ps(sum128, sum128);               // pairwise add
    sum128 = _mm_hadd_ps(sum128, sum128);               // final reduction

    // Step 4: Extract scalar result
    return _mm_cvtss_f32(sum128);
}

static inline float add_reduce_m256i(__m256i v) {
    __m128i v128 = _mm_add_epi32(
        _mm256_castsi256_si128(v),
        _mm256_extracti128_si256(v, 1)
    );
    v128 = _mm_hadd_epi32(v128, v128);
    v128 = _mm_hadd_epi32(v128, v128);
    return (float)_mm_cvtsi128_si32(v128);
}

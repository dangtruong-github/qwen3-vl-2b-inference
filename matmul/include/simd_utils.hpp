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

static inline float max_reduce_mm_256(__m256 vec) {
    // Step 1: Split 256-bit into two 128-bit halves and take the max
    // result = [max(low0, high0), max(low1, high1), max(low2, high2), max(low3, high3)]
    __m128 low  = _mm256_castps256_ps128(vec);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    __m128 max128 = _mm_max_ps(low, high);

    // Step 2: Shuffle and max to reduce from 4 elements to 2
    // Compare [x, y, z, w] with [z, w, ?, ?]
    __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));

    // Step 3: Shuffle and max to reduce from 2 elements to 1
    // Compare [x, y, ...] with [y, y, ...]
    __m128 max32 = _mm_max_ps(max64, _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 1, 1, 1)));

    // Step 4: Extract the final scalar
    return _mm_cvtss_f32(max32);
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

static inline int add_reduce_m256i_int(__m256i v) {
    __m128i vlow  = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum   = _mm_add_epi32(vlow, vhigh);

    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2,3,0,1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1,0,3,2)));

    return _mm_cvtsi128_si32(sum);
}

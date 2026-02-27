#pragma once

#include <immintrin.h>
#include <cfloat>
#include <cstddef>
#include <immintrin.h>
#include <float.h>
#include <stddef.h>

// Safe AVX2 exp implementation
static inline __m256 exp256_ps(__m256 x)
{
    // Clamp to prevent overflow
    const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    // Constants
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2   = _mm256_set1_ps(0.6931471805599453f);

    // Compute n = floor(x / ln2)
    __m256 fx = _mm256_mul_ps(x, log2e);
    fx = _mm256_floor_ps(fx);

    // Convert to int
    __m256i emm0 = _mm256_cvttps_epi32(fx);

    // Clamp exponent to valid float range [-126, 127]
    emm0 = _mm256_min_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_max_epi32(emm0, _mm256_set1_epi32(-126));

    // Compute g = x - n*ln2
    __m256 g = _mm256_fnmadd_ps(fx, ln2, x);

    // Polynomial approximation (Cephes)
    const __m256 c1 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 c2 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 c3 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 c4 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 c5 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 c6 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 y = c1;
    y = _mm256_fmadd_ps(y, g, c2);
    y = _mm256_fmadd_ps(y, g, c3);
    y = _mm256_fmadd_ps(y, g, c4);
    y = _mm256_fmadd_ps(y, g, c5);
    y = _mm256_fmadd_ps(y, g, c6);
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.0f));

    // Build 2^n safely
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
}

static inline float avx2_max(const float *arr, size_t N) {
    size_t i = 0;

    // Initialize vector accumulator with -inf
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);

    // Process 8 floats at a time
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    // Horizontal reduction of vmax
    __m128 low  = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 vmax128 = _mm_max_ps(low, high);

    vmax128 = _mm_max_ps(vmax128, _mm_movehl_ps(vmax128, vmax128));
    vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, 0x55));

    float max_val = _mm_cvtss_f32(vmax128);

    // Handle remaining elements
    for (; i < N; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    return max_val;
}

static inline float avx2_max_and_scale(float *arr, size_t N, float scale) {
    size_t i = 0;

    // Initialize vector accumulator with -inf
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
    __m256 v_scale = _mm256_set1_ps(scale);

    // Process 8 floats at a time
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        v = _mm256_mul_ps(v, v_scale);
        vmax = _mm256_max_ps(vmax, v);
        _mm256_storeu_ps(arr + i, v);
    }

    // Horizontal reduction of vmax
    __m128 low  = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 vmax128 = _mm_max_ps(low, high);

    vmax128 = _mm_max_ps(vmax128, _mm_movehl_ps(vmax128, vmax128));
    vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, 0x55));

    float max_val = _mm_cvtss_f32(vmax128);

    // Handle remaining elements
    for (; i < N; i++) {
        arr[i] *= scale;
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    return max_val;
}

static inline float avx2_sum_exp_max(float *arr, size_t T, float max_score) {
    __m256 vsum = _mm256_setzero_ps();
    __m256 vmax = _mm256_set1_ps(max_score);

    int j = 0;
    for (; j + 7 < T; j += 8) {
        __m256 v = _mm256_loadu_ps(arr + j);
        v = _mm256_sub_ps(v, vmax);
        v = exp256_ps(v);
        _mm256_storeu_ps(arr + j, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 s  = _mm_add_ps(lo, hi);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    float sum = _mm_cvtss_f32(s);

    for (; j < T; j++) {
        arr[j] = expf(arr[j] - max_score);
        sum += arr[j];
    }

    return sum;
}

static inline void avx2_max_multiple(const float *arr, size_t N, size_t num_rows, float *results) {
    for (size_t r = 0; r < num_rows; r++) {
        const float *row_ptr = arr + (r * N);
        size_t i = 0;

        // 1. Initialize vector with negative infinity
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);

        // 2. Main SIMD loop (8 floats at a time)
        for (; i + 7 < N; i += 8) {
            __m256 v = _mm256_loadu_ps(row_ptr + i);
            vmax = _mm256_max_ps(vmax, v);
        }

        // 3. Horizontal reduction: Extract max from the 8 lanes
        // Fold 256-bit to 128-bit
        __m128 low  = _mm256_castps256_ps128(vmax);
        __m128 high = _mm256_extractf128_ps(vmax, 1);
        __m128 vmax128 = _mm_max_ps(low, high);

        // Fold 128-bit (4 floats) down to 1
        vmax128 = _mm_max_ps(vmax128, _mm_movehl_ps(vmax128, vmax128)); // compare low 2 with high 2
        vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, 0x1)); // compare adjacent

        float current_max = _mm_cvtss_f32(vmax128);

        // 4. Handle tail elements (remainder of N % 8)
        for (; i < N; i++) {
            if (row_ptr[i] > current_max) {
                current_max = row_ptr[i];
            }
        }

        results[r] = current_max;
    }
}

static inline void avx2_sum_exp_max_multiple(float *arr, size_t N, size_t num_rows, float *max_buffer) {
    const float eps = 1e-6f;
    for (size_t r = 0; r < num_rows; r++) {
        float *row_ptr = arr + (r * N);
        float max_score = max_buffer[r]; // Assuming maxes are already in the buffer

        __m256 vsum = _mm256_setzero_ps();
        __m256 vmax = _mm256_set1_ps(max_score);

        size_t j = 0;
        for (; j + 7 < N; j += 8) {
            __m256 v = _mm256_loadu_ps(row_ptr + j);
            v = _mm256_sub_ps(v, vmax);
            
            // Assuming you have a SIMD exp implementation like exp256_ps
            v = exp256_ps(v); 
            
            _mm256_storeu_ps(row_ptr + j, v);
            vsum = _mm256_add_ps(vsum, v);
        }

        // Horizontal reduction of vsum
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        
        // Shuffle-based reduction (faster than hadd)
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ps(s, _mm_shuffle_ps(s, s, 0x55));
        
        float sum = _mm_cvtss_f32(s);

        // Handle remainder
        for (; j < N; j++) {
            float val = expf(row_ptr[j] - max_score);
            row_ptr[j] = val;
            sum += val;
        }

        // Overwrite the max with the sum in the buffer
        max_buffer[r] = 1.0f / (sum + eps);
    }
}

static inline float add_reduce_mm_256_layer(__m256 vec) {
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

static inline float load_x(const void *ptr, DType::Type dtype, size_t idx) {
    if (dtype == DType::FP32)
        return static_cast<const float*>(ptr)[idx];
    else
        return static_cast<float>(
            static_cast<const half_cpu*>(ptr)[idx]
        );
}

static inline void store_out(void *ptr, DType::Type dtype, size_t idx, float v) {
    if (dtype == DType::FP32)
        static_cast<float*>(ptr)[idx] = v;
    else
        static_cast<half_cpu*>(ptr)[idx] = static_cast<half_cpu>(v);
}
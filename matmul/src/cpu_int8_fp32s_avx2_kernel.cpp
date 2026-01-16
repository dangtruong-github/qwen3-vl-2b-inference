#include "../include/cpu_int8_fp32s_avx2_kernel.hpp"

#if defined(__AVX2__) && defined(__FMA__)
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

void linear_int8_fp32s_transpose(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    // default fallback
    // KERNEL ENSURE group_size >= 16
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const float *a0_ptr = mat_A + i * K;
            const int8_t *b0_ptr = mat_B_in + j * K;
            const float *b_s_ptr = mat_B_scales + j * K / group_size;

            float acc = 0.0f;
            if (mat_bias_in && mat_bias_scale) {
                acc = (float)mat_bias_in[j] * mat_bias_scale[j / group_size];
            }

            size_t k = 0;
            for (; k + group_size <= K; k += group_size) {
                // group_size % 16 == 0
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                for (size_t k_i = k; k_i < k + group_size; k_i += 32) {
                    __m256 a0 = _mm256_loadu_ps(a0_ptr + k_i);
                    __m256 a1 = _mm256_loadu_ps(a0_ptr + k_i + 8);
                    __m256 a2 = _mm256_loadu_ps(a0_ptr + k_i + 16);
                    __m256 a3 = _mm256_loadu_ps(a0_ptr + k_i + 24);
                    
                    // 1. Load 16 int8 values into an xmm register (128-bit)// 1. Load all 32 int8 values into a single 256-bit register
                    __m256i q32 = _mm256_loadu_si256((const __m256i*)(b0_ptr + k_i));

                    // 2. Extract 128-bit chunks (16 bytes each)
                    __m128i low_16 = _mm256_castsi256_si128(q32);
                    __m128i high_16 = _mm256_extracti128_si256(q32, 1);

                    // 3. Convert first 16 bytes (low_16) into 2 float registers (8 floats each)
                    // Low 8 of low_16
                    __m256i i32_0 = _mm256_cvtepi8_epi32(low_16); 
                    __m256 b0 = _mm256_cvtepi32_ps(i32_0);
                    
                    // High 8 of low_16 (shift right by 8 bytes first)
                    __m256i i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(low_16, 8));
                    __m256 b1 = _mm256_cvtepi32_ps(i32_1);

                    // 4. Convert next 16 bytes (high_16) into 2 float registers
                    // Low 8 of high_16
                    __m256i i32_2 = _mm256_cvtepi8_epi32(high_16);
                    __m256 b2 = _mm256_cvtepi32_ps(i32_2);
                    
                    // High 8 of high_16 (shift right by 8 bytes first)
                    __m256i i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(high_16, 8));
                    __m256 b3 = _mm256_cvtepi32_ps(i32_3);

                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a2, b2, c2);
                    c3 = _mm256_fmadd_ps(a3, b3, c3);
                }

                c0 = _mm256_add_ps(c0, c1);
                c2 = _mm256_add_ps(c2, c3);

                acc += add_reduce_mm_256(_mm256_add_ps(c0, c2)) * b_s_ptr[k / group_size];
            }

            // Horizontal sum of all accumulators

            // Remainder loop
            for (; k < K; ++k) {
                float scale = mat_B_scales[(j * K + k) / group_size];
                acc += a0_ptr[k] * ((float)b0_ptr[k] * scale);
            }

            mat_C[i * N + j] = acc;
        }
    }
}

void linear_int8_fp32s_avx2_kernel(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C,                // [M, N] FP32
    size_t M,
    size_t N,
    size_t K,
    bool mat_B_transpose,
    size_t group_size
) {
    if (mat_B_transpose) {
        linear_int8_fp32s_transpose(
            mat_A, mat_B_in, mat_B_scales, mat_bias_in,
            mat_bias_scale, mat_C, M, N, K, group_size
        );
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;

            // -------- GEMM --------
            for (size_t k = 0; k < K; ++k) {
                float a = mat_A[i * K + k];

                // linear index into B (matches quantizer layout)
                size_t b_linear_idx;
                b_linear_idx = k * N + j;

                size_t scale_idx = b_linear_idx / group_size;
                float scale = mat_B_scales[scale_idx];

                float b = (float)mat_B_in[b_linear_idx] * scale;
                acc += a * b;
            }

            mat_C[i * N + j] = acc;
        }
    }

    

    // -------- Bias --------
    if (mat_bias_in && mat_bias_scale) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t bias_scale_idx = j / group_size;
                float bias =
                    (float)mat_bias_in[j] * mat_bias_scale[bias_scale_idx];
                mat_C[i * N + j] += bias;
            }
        }
    }
}
#endif
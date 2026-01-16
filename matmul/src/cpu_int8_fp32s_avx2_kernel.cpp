#include "../include/cpu_int8_fp32s_avx2_kernel.hpp"

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
            
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + group_size <= K; k += group_size) {
                // group_size % 16 == 0
                float scale = b_s_ptr[k / group_size];
                __m256 bs = _mm256_set1_ps(scale);
                for (size_t k_i = k; k_i < k + group_size; k_i += 16) {
                    __m256 a0 = _mm256_loadu_ps(a0_ptr + k_i);
                    __m256 a1 = _mm256_loadu_ps(a0_ptr + k_i + 8);
                    
                    // 1. Load 16 int8 values into an xmm register (128-bit)
                    __m128i q8 = _mm_loadu_si128((const __m128i*)(b0_ptr + k_i));
                    __m256 b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                    __m128i q8_hi = _mm_srli_si128(q8, 8);
                    __m256 b1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8_hi));

                    c0 = _mm256_fmadd_ps(a0, _mm256_mul_ps(b0, bs), c0);
                    c1 = _mm256_fmadd_ps(a1, _mm256_mul_ps(b1, bs), c1);
                }
            }

            for (; k + 16 <= K; k += 16) {
                // Check if we cross a group boundary here. 
                // Assuming group_size >= 32 for this SIMD block.
                float scale = b_s_ptr[k / group_size];
                __m256 bs = _mm256_set1_ps(scale);

                __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                __m256 a1 = _mm256_loadu_ps(a0_ptr + k + 8);
                
                // 1. Load 16 int8 values into an xmm register (128-bit)
                __m128i q8 = _mm_loadu_si128((const __m128i*)(b0_ptr + k));
                __m256 b0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                __m128i q8_hi = _mm_srli_si128(q8, 8);
                __m256 b1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8_hi));

                c0 = _mm256_fmadd_ps(a0, _mm256_mul_ps(b0, bs), c0);
                c1 = _mm256_fmadd_ps(a1, _mm256_mul_ps(b1, bs), c1);
            }

            // Horizontal sum of all accumulators
            float acc = add_reduce_mm_256(_mm256_add_ps(c0, c1));

            // Remainder loop
            for (; k < K; ++k) {
                float scale = mat_B_scales[(j * K + k) / group_size];
                acc += a0_ptr[k] * ((float)b0_ptr[k] * scale);
            }

            // Fusing Bias right here to save memory bandwidth
            if (mat_bias_in && mat_bias_scale) {
                acc += (float)mat_bias_in[j] * mat_bias_scale[j / group_size];
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

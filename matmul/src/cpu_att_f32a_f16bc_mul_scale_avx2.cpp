#include "../include/cpu_wrapper.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void gemm_att_f32a_f16bc_mul_scale_n64(
    const float *mat_A, const half_cpu *mat_B, half_cpu *mat_C,
    const float *scale, size_t M, size_t K
) {
    constexpr size_t TK = 32;
    
    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

        alignas(32) float packed_B[TK << 6];

        for (size_t kk = 0; kk < K; kk += TK) {
            bool end_of_k; 
            size_t k_end = kk + TK; 
            
            if (k_end < K) {
                end_of_k = false;
            } else {
                end_of_k = true;
                k_end = K;
            }
            size_t k_size = k_end - kk;
            const half_cpu *start_b_ptr = mat_B + (kk << 6);
                
            for (size_t k = 0; k < k_size; ++k) {
                size_t j = 0;
                const size_t k6 = (k << 6);
                const half_cpu* b_ptr = start_b_ptr + k6;
                float *packed_B_ptr = packed_B + k6;

                // Process 64 elements by unrolling by 4 (8 elements per register * 4 = 32 per step)
                for (; j + 32 <= 64; j += 32) {
                    // 1. Load 4 chunks of 8 half-floats (128-bit each)
                    __m128i v16_0 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                    __m128i v16_1 = _mm_loadu_si128((const __m128i*)(b_ptr + j + 8));
                    __m128i v16_2 = _mm_loadu_si128((const __m128i*)(b_ptr + j + 16));
                    __m128i v16_3 = _mm_loadu_si128((const __m128i*)(b_ptr + j + 24));

                    // 2. Convert each to 8 single-precision floats (256-bit each)
                    __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                    __m256 v32_1 = _mm256_cvtph_ps(v16_1);
                    __m256 v32_2 = _mm256_cvtph_ps(v16_2);
                    __m256 v32_3 = _mm256_cvtph_ps(v16_3);

                    // 3. Store converted floats (Ensure packed_B_ptr is 32-byte aligned for _mm256_store_ps)
                    _mm256_store_ps(packed_B_ptr + j,      v32_0);
                    _mm256_store_ps(packed_B_ptr + j + 8,  v32_1);
                    _mm256_store_ps(packed_B_ptr + j + 16, v32_2);
                    _mm256_store_ps(packed_B_ptr + j + 24, v32_3);
                }
            }

            #pragma omp for schedule(static)
            for (size_t ii = 0; ii < M; ++ii) {
                const float *a0_ptr = mat_A + ii * K + kk;
                half_cpu *c0_ptr = mat_C + ii * 64;
                
                __m256 c0, c1, c2, c3, c4, c5, c6, c7;
                if (kk == 0) {
                    c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = _mm256_setzero_ps();
                } else {
                    // Load 8 half-floats (128 bits) into a 128-bit register
                    __m128i h0 = _mm_loadu_si128((__m128i*)(c0_ptr));
                    __m128i h1 = _mm_loadu_si128((__m128i*)(c0_ptr + 8));
                    __m128i h2 = _mm_loadu_si128((__m128i*)(c0_ptr + 16));
                    __m128i h3 = _mm_loadu_si128((__m128i*)(c0_ptr + 24));
                    __m128i h4 = _mm_loadu_si128((__m128i*)(c0_ptr + 32));
                    __m128i h5 = _mm_loadu_si128((__m128i*)(c0_ptr + 40));
                    __m128i h6 = _mm_loadu_si128((__m128i*)(c0_ptr + 48));
                    __m128i h7 = _mm_loadu_si128((__m128i*)(c0_ptr + 56));

                    // Convert those 8 halves into 8 singles (256 bits)
                    c0 = _mm256_cvtph_ps(h0);
                    c1 = _mm256_cvtph_ps(h1);
                    c2 = _mm256_cvtph_ps(h2);
                    c3 = _mm256_cvtph_ps(h3);
                    c4 = _mm256_cvtph_ps(h4);
                    c5 = _mm256_cvtph_ps(h5);
                    c6 = _mm256_cvtph_ps(h6);
                    c7 = _mm256_cvtph_ps(h7);
                }

                for (size_t k = 0; k < k_size; ++k) {
                    const float *pb0 = packed_B + (k << 6);
                    __m256 a0 = _mm256_broadcast_ss(a0_ptr + k);

                    c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                    c1 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 8), c1);
                    c2 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 16), c2);
                    c3 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 24), c3);
                    c4 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 32), c4);
                    c5 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 40), c5);
                    c6 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 48), c6);
                    c7 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 56), c7);
                }

                if (end_of_k) {
                    __m256 scale_vec = _mm256_broadcast_ss(scale + ii);

                    c0 = _mm256_mul_ps(c0, scale_vec);
                    c1 = _mm256_mul_ps(c1, scale_vec);
                    c2 = _mm256_mul_ps(c2, scale_vec);
                    c3 = _mm256_mul_ps(c3, scale_vec);
                    c4 = _mm256_mul_ps(c4, scale_vec);
                    c5 = _mm256_mul_ps(c5, scale_vec);
                    c6 = _mm256_mul_ps(c6, scale_vec);
                    c7 = _mm256_mul_ps(c7, scale_vec);
                }
                // Store the 128-bit chunks to your half* pointer
                _mm_storeu_si128((__m128i*)(c0_ptr),      _mm256_cvtps_ph(c0, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 8),  _mm256_cvtps_ph(c1, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 16), _mm256_cvtps_ph(c2, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 24), _mm256_cvtps_ph(c3, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 32), _mm256_cvtps_ph(c4, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 40), _mm256_cvtps_ph(c5, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 48), _mm256_cvtps_ph(c6, _MM_FROUND_CUR_DIRECTION));
                _mm_storeu_si128((__m128i*)(c0_ptr + 56), _mm256_cvtps_ph(c7, _MM_FROUND_CUR_DIRECTION));
            }
        }
    }
}

void att_f32a_f16bc_mul_scale_avx2_kernel(
    const float *mat_A, const half_cpu *mat_B, half_cpu *mat_C,
    const float *scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    if (!mat_B_transpose && N == 64) {
        gemm_att_f32a_f16bc_mul_scale_n64(
            mat_A, mat_B, mat_C, scale, M, K
        );
        return;
    }

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {

        const float* A_row = mat_A + m * K;
        half_cpu* C_row = mat_C + m * N;
        float row_scale = scale ? scale[m] : 1.0f;

        for (size_t n = 0; n < N; ++n) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                float a = A_row[k];

                float b = (!mat_B_transpose)
                    ? (float)mat_B[k * N + n]
                    : (float)mat_B[n * K + k];

                sum += a * b;
            }

            C_row[n] = (half_cpu)(sum * row_scale);
        }
    }
    
}
#endif

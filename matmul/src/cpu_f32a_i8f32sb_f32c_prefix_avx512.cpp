#include "../include/cpu_wrapper.hpp"
#include "../include/cpu_f32a_i8f32sb_f32c.hpp"

#define VERY_LARGE_N 65536
#define K_BLOCK (size_t)6144

#if defined(__AVX512F__) && defined(__AVX512DQ__)
void gemm_m2_lgNK_avx512_prefix(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C,
    size_t N, size_t K, const size_t group_size
) {
    const size_t K2 = (K << 1);
    alignas(32) uint8_t a_q8[K2];
    float a_q8_s[K >> 4];

    for (int kk = 0; kk < K2; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m512 v_max = _mm512_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 8) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            v_max = _mm512_max_ps(v_max, _mm512_abs_ps(f0));
        }
        float max_val = _mm512_reduce_max_ps(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        __m512 invS = _mm512_set1_ps(inv_scale);
        __m512 zp_f = _mm512_set1_ps(128.0f);

        for (int k = kk; k < kk + group_size; k += 16) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            f0 = _mm512_fmadd_ps(f0, invS, zp_f);
            __m512i i0 = _mm512_cvtps_epi32(f0);
            __m128i u0 = _mm512_cvtusepi32_epi8(i0);
            _mm_store_si128((__m128i*)(a_q8 + k), u0);
        }
    }

    const size_t K_g = K / group_size;

    const uint8_t *a1_ptr = a_q8 + K;

    // up
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m512 c0_f = _mm512_setzero_ps();
        __m512 c1_f = _mm512_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict w_ptr = mat_B_in + jjK;
        const float *__restrict s_ptr = mat_B_scales + jjK_g;
        const int *__restrict sum_ptr = sum_int8_B + (jjK_g << 4);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();

            for (size_t k = kk; k < kk + group_size; k += 64) {
                __m512i b_vec = _mm512_loadu_si512((__m512i*)(w_ptr + k));

                __m512i a0_vec = _mm512_loadu_si512((__m512i*)(a_q8 + k));
                __m512i a1_vec = _mm512_loadu_si512((__m512i*)(a1_ptr + k));
                
                c0 = _mm512_dpbusd_epi32(c0, a0_vec, b_vec);
                c1 = _mm512_dpbusd_epi32(c1, a1_vec, b_vec);
            }

            __m512i corr32 = _mm512_loadu_si512((__m512i*)(sum_ptr + (g_off << 4)));

            c0 = _mm512_sub_epi32(c0, corr32);
            c1 = _mm512_sub_epi32(c1, corr32);

            const float scale_val = s_ptr[g_off];
            
            c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_q8_s[g_off] * scale_val), c0_f);
            c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a_q8_s[K_g + g_off] * scale_val), c1_f);
        }

        mat_C[jj] = _mm512_reduce_add_ps(c0_f);
        mat_C[N + jj] = _mm512_reduce_add_ps(c1_f);
    }
}

void gemm_m4_lgNK_avx512_prefix(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C,
    size_t N, size_t K, const size_t group_size
) {
    const size_t K4 = (K << 2);
    alignas(32) uint8_t a_q8[K4];
    float a_q8_s[K >> 4];

    for (int kk = 0; kk < K4; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m512 v_max = _mm512_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 8) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            v_max = _mm512_max_ps(v_max, _mm512_abs_ps(f0));
        }
        float max_val = _mm512_reduce_max_ps(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        __m512 invS = _mm512_set1_ps(inv_scale);
        __m512 zp_f = _mm512_set1_ps(128.0f);

        for (int k = kk; k < kk + group_size; k += 16) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            f0 = _mm512_fmadd_ps(f0, invS, zp_f);
            __m512i i0 = _mm512_cvtps_epi32(f0);
            __m128i u0 = _mm512_cvtusepi32_epi8(i0);
            _mm_store_si128((__m128i*)(a_q8 + k), u0);
        }
    }

    const size_t K_g = K / group_size;

    const uint8_t *a1_ptr = a_q8 + K;
    const uint8_t *a2_ptr = a_q8 + (K << 1);
    const uint8_t *a3_ptr = a_q8 + (K * 3);
    const float *a1_s_ptr = a_q8_s + K_g;
    const float *a2_s_ptr = a_q8_s + (K_g << 1);
    const float *a3_s_ptr = a_q8_s + (K_g * 3);

    // up
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m512 c0_f = _mm512_setzero_ps();
        __m512 c1_f = _mm512_setzero_ps();
        __m512 c2_f = _mm512_setzero_ps();
        __m512 c3_f = _mm512_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict w_ptr = mat_B_in + jjK;
        const float *__restrict s_ptr = mat_B_scales + jjK_g;
        const int *__restrict sum_ptr = sum_int8_B + (jjK_g << 4);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();
            __m512i c2 = _mm512_setzero_si512();
            __m512i c3 = _mm512_setzero_si512();

            for (size_t k = kk; k < kk + group_size; k += 64) {
                __m512i b_vec = _mm512_loadu_si512((__m512i*)(w_ptr + k));

                __m512i a0_vec = _mm512_loadu_si512((__m512i*)(a_q8 + k));
                __m512i a1_vec = _mm512_loadu_si512((__m512i*)(a1_ptr + k));
                __m512i a2_vec = _mm512_loadu_si512((__m512i*)(a2_ptr + k));
                __m512i a3_vec = _mm512_loadu_si512((__m512i*)(a3_ptr + k));
                
                c0 = _mm512_dpbusd_epi32(c0, a0_vec, b_vec);
                c1 = _mm512_dpbusd_epi32(c1, a1_vec, b_vec);
                c2 = _mm512_dpbusd_epi32(c2, a2_vec, b_vec);
                c3 = _mm512_dpbusd_epi32(c3, a3_vec, b_vec);
            }

            __m512i corr32 = _mm512_loadu_si512((__m512i*)(sum_ptr + (g_off << 4)));

            c0 = _mm512_sub_epi32(c0, corr32);
            c1 = _mm512_sub_epi32(c1, corr32);
            c2 = _mm512_sub_epi32(c2, corr32);
            c3 = _mm512_sub_epi32(c3, corr32);

            const float scale_val = s_ptr[g_off];
            
            c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_q8_s[g_off] * scale_val), c0_f);
            c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a1_s_ptr[g_off] * scale_val), c1_f);
            c2_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c2), _mm512_set1_ps(a2_s_ptr[g_off] * scale_val), c2_f);
            c3_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c3), _mm512_set1_ps(a3_s_ptr[g_off] * scale_val), c3_f);
        }

        mat_C[jj] = _mm512_reduce_add_ps(c0_f);
        mat_C[N + jj] = _mm512_reduce_add_ps(c1_f);
        mat_C[(N << 1) + jj] = _mm512_reduce_add_ps(c2_f);
        mat_C[(N * 3) + jj] = _mm512_reduce_add_ps(c3_f);
    }
}

void f32a_i8f32sb_f32c_avx512_prefix_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C, size_t M, size_t N,
    size_t K, size_t group_size, bool add_to_c
) {
    size_t i = 0;

    if (K <= K_BLOCK) {
        if (group_size == 64) {
            for (; i + 4 <= M; i += 4) {
                gemm_m4_lgNK_avx512_prefix(
                    mat_A, mat_B_in, mat_B_scales,
                    sum_int8_B, mat_C, N, K, group_size
                );
                mat_A += (K << 2);
                mat_C += (N << 2);
            }

            if (i + 2 <= M) {
                gemm_m2_lgNK_avx512_prefix(
                    mat_A, mat_B_in, mat_B_scales,
                    sum_int8_B, mat_C, N, K, group_size
                );
                mat_A += (K << 1);
                mat_C += (N << 1);
            }

            if (i < M) {
                gemv_lg_N_K_g64(mat_A, mat_B_in, mat_B_scales, mat_C, N, K);
            }
        } else {
            for (; i + 4 <= M; i += 4) {
                gemm_m4_lgNK_avx512_prefix(
                    mat_A, mat_B_in, mat_B_scales,
                    sum_int8_B, mat_C, N, K, group_size
                );
                mat_A += (K << 2);
                mat_C += (N << 2);
            }

            if (i + 2 <= M) {
                gemm_m2_lgNK_avx512_prefix(
                    mat_A, mat_B_in, mat_B_scales,
                    sum_int8_B, mat_C, N, K, group_size
                );
                mat_A += (K << 1);
                mat_C += (N << 1);
            }

            if (i < M) {
                gemv_lg_N_K(
                    mat_A, mat_B_in, mat_B_scales, mat_C, N, K, group_size
                );
            }
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            gemv_lg_N_K_decode(mat_A, mat_B_in, mat_B_scales, mat_C, N, K, group_size);
            mat_A += K;
            mat_C += N;
        }
    }
    return;    
}
#endif

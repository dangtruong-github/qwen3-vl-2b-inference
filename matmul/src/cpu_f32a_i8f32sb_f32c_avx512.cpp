#include "../include/cpu_wrapper.hpp"

#define VERY_LARGE_N 65536

// #if defined(__AVX512F__) && defined(__AVX512DQ__)
void gemv_lg_N_lg_K_avx512_transpose(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N,
    size_t K, size_t group_size, bool add_to_c
) {
    const size_t K_g = K / group_size;

    alignas(64) uint8_t a_q8[K];
    float a_q8_s[K_g];

    for (int kk = 0; kk < K; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m512 v_max = _mm512_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 16) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            __m512 abs_f0 = _mm512_abs_ps(f0);
            v_max = _mm512_max_ps(v_max, abs_f0);
        }
        float max_val = _mm512_reduce_max_ps(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        __m512 invS = _mm512_set1_ps(inv_scale);
        __m512 zp = _mm512_set1_ps(128.0f);

        for (int k = kk; k < kk + group_size; k += 32) {
            __m512 f0 = _mm512_loadu_ps(mat_A + k);
            __m512 f1 = _mm512_loadu_ps(mat_A + k + 16);

            f0 = _mm512_fmadd_ps(f0, invS, zp);
            f1 = _mm512_fmadd_ps(f1, invS, zp);
            
            __m512i i0 = _mm512_cvtps_epi32(f0);
            __m512i i1 = _mm512_cvtps_epi32(f1);

            __m128i u0 = _mm512_cvtusepi32_epi8(i0);
            __m128i u1 = _mm512_cvtusepi32_epi8(i1);

            _mm_store_si128((__m128i*)(a_q8 + k),      u0);
            _mm_store_si128((__m128i*)(a_q8 + k + 16), u1);
        }
    }
    const __m512i ones8 = _mm512_set1_epi8(1);
    
    if (add_to_c) {
        #pragma omp parallel for schedule(static)
        for (size_t jj = 0; jj < N; jj += 4) {
            __m512 c0_f = _mm512_setzero_ps();
            __m512 c1_f = _mm512_setzero_ps();
            __m512 c2_f = _mm512_setzero_ps();
            __m512 c3_f = _mm512_setzero_ps();

            const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
            const int8_t *__restrict b1_ptr = mat_B_in + (jj + 1) * K;
            const int8_t *__restrict b2_ptr = mat_B_in + (jj + 2) * K;
            const int8_t *__restrict b3_ptr = mat_B_in + (jj + 3) * K;

            const size_t g_off_base = jj * K_g;
            
            for (size_t kk = 0; kk < K; kk += group_size) {
                const size_t a_s_id = kk / group_size;
                const size_t g_off = g_off_base + a_s_id;
                const float *__restrict b_s_ptr = mat_B_scales + g_off;

                __m512i c0 = _mm512_setzero_si512();
                __m512i c1 = _mm512_setzero_si512();
                __m512i c2 = _mm512_setzero_si512();
                __m512i c3 = _mm512_setzero_si512();

                __m512i corr32_0 = _mm512_setzero_si512();
                __m512i corr32_1 = _mm512_setzero_si512();
                __m512i corr32_2 = _mm512_setzero_si512();
                __m512i corr32_3 = _mm512_setzero_si512();

                for (size_t k = kk; k < kk + group_size; k += 64) {
                    __m512i a_vec = _mm512_load_si512((__m512i*)(a_q8 + k));

                    __m512i b0 = _mm512_loadu_si512((__m512i*)(b0_ptr + k));
                    __m512i b1 = _mm512_loadu_si512((__m512i*)(b1_ptr + k));
                    __m512i b2 = _mm512_loadu_si512((__m512i*)(b2_ptr + k));
                    __m512i b3 = _mm512_loadu_si512((__m512i*)(b3_ptr + k));

                    corr32_0 = _mm512_dpbusd_epi32(corr32_0, ones8, b0);
                    corr32_1 = _mm512_dpbusd_epi32(corr32_1, ones8, b1);
                    corr32_2 = _mm512_dpbusd_epi32(corr32_2, ones8, b2);
                    corr32_3 = _mm512_dpbusd_epi32(corr32_3, ones8, b3);

                    c0 = _mm512_dpbusd_epi32(c0, a_vec, b0);
                    c1 = _mm512_dpbusd_epi32(c1, a_vec, b1);
                    c2 = _mm512_dpbusd_epi32(c2, a_vec, b2);
                    c3 = _mm512_dpbusd_epi32(c3, a_vec, b3);
                }

                const float a_s = a_q8_s[a_s_id];

                corr32_0 = _mm512_slli_epi32(corr32_0, 7);
                corr32_1 = _mm512_slli_epi32(corr32_1, 7);
                corr32_2 = _mm512_slli_epi32(corr32_2, 7);
                corr32_3 = _mm512_slli_epi32(corr32_3, 7);
                c0 = _mm512_sub_epi32(c0, corr32_0);
                c1 = _mm512_sub_epi32(c1, corr32_1);
                c2 = _mm512_sub_epi32(c2, corr32_2);
                c3 = _mm512_sub_epi32(c3, corr32_3);
                
                c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_s * b_s_ptr[0]), c0_f);
                c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a_s * b_s_ptr[K_g]), c1_f);
                c2_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c2), _mm512_set1_ps(a_s * b_s_ptr[K_g << 1]), c2_f);
                c3_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c3), _mm512_set1_ps(a_s * b_s_ptr[K_g * 3]), c3_f);
            }
            
            mat_C[jj] += _mm512_reduce_add_ps(c0_f);
            mat_C[jj + 1] += _mm512_reduce_add_ps(c1_f);
            mat_C[jj + 2] += _mm512_reduce_add_ps(c2_f);
            mat_C[jj + 3] += _mm512_reduce_add_ps(c3_f);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t jj = 0; jj < N; jj += 4) {
            __m512 c0_f = _mm512_setzero_ps();
            __m512 c1_f = _mm512_setzero_ps();
            __m512 c2_f = _mm512_setzero_ps();
            __m512 c3_f = _mm512_setzero_ps();

            const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
            const int8_t *__restrict b1_ptr = mat_B_in + (jj + 1) * K;
            const int8_t *__restrict b2_ptr = mat_B_in + (jj + 2) * K;
            const int8_t *__restrict b3_ptr = mat_B_in + (jj + 3) * K;

            const size_t g_off_base = jj * K_g;
            
            for (size_t kk = 0; kk < K; kk += group_size) {
                const size_t a_s_id = kk / group_size;
                const size_t g_off = g_off_base + a_s_id;
                const float *__restrict b_s_ptr = mat_B_scales + g_off;

                __m512i c0 = _mm512_setzero_si512();
                __m512i c1 = _mm512_setzero_si512();
                __m512i c2 = _mm512_setzero_si512();
                __m512i c3 = _mm512_setzero_si512();

                __m512i corr32_0 = _mm512_setzero_si512();
                __m512i corr32_1 = _mm512_setzero_si512();
                __m512i corr32_2 = _mm512_setzero_si512();
                __m512i corr32_3 = _mm512_setzero_si512();

                for (size_t k = kk; k < kk + group_size; k += 64) {
                    __m512i a_vec = _mm512_load_si512((__m512i*)(a_q8 + k));

                    __m512i b0 = _mm512_loadu_si512((__m512i*)(b0_ptr + k));
                    __m512i b1 = _mm512_loadu_si512((__m512i*)(b1_ptr + k));
                    __m512i b2 = _mm512_loadu_si512((__m512i*)(b2_ptr + k));
                    __m512i b3 = _mm512_loadu_si512((__m512i*)(b3_ptr + k));

                    corr32_0 = _mm512_dpbusd_epi32(corr32_0, ones8, b0);
                    corr32_1 = _mm512_dpbusd_epi32(corr32_1, ones8, b1);
                    corr32_2 = _mm512_dpbusd_epi32(corr32_2, ones8, b2);
                    corr32_3 = _mm512_dpbusd_epi32(corr32_3, ones8, b3);

                    c0 = _mm512_dpbusd_epi32(c0, a_vec, b0);
                    c1 = _mm512_dpbusd_epi32(c1, a_vec, b1);
                    c2 = _mm512_dpbusd_epi32(c2, a_vec, b2);
                    c3 = _mm512_dpbusd_epi32(c3, a_vec, b3);
                }

                const float a_s = a_q8_s[a_s_id];

                corr32_0 = _mm512_slli_epi32(corr32_0, 7);
                corr32_1 = _mm512_slli_epi32(corr32_1, 7);
                corr32_2 = _mm512_slli_epi32(corr32_2, 7);
                corr32_3 = _mm512_slli_epi32(corr32_3, 7);
                c0 = _mm512_sub_epi32(c0, corr32_0);
                c1 = _mm512_sub_epi32(c1, corr32_1);
                c2 = _mm512_sub_epi32(c2, corr32_2);
                c3 = _mm512_sub_epi32(c3, corr32_3);
                
                c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_s * b_s_ptr[0]), c0_f);
                c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a_s * b_s_ptr[K_g]), c1_f);
                c2_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c2), _mm512_set1_ps(a_s * b_s_ptr[K_g << 1]), c2_f);
                c3_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c3), _mm512_set1_ps(a_s * b_s_ptr[K_g * 3]), c3_f);
            }
            
            mat_C[jj] = _mm512_reduce_add_ps(c0_f);
            mat_C[jj + 1] = _mm512_reduce_add_ps(c1_f);
            mat_C[jj + 2] = _mm512_reduce_add_ps(c2_f);
            mat_C[jj + 3] = _mm512_reduce_add_ps(c3_f);
        }
    }
}

void f32a_i8f32sb_f32c_avx512_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N,
    size_t K, size_t group_size, bool add_to_c
) {
    if (M == 1 && N >= 1024 && K >= 1024) {
        gemv_lg_N_lg_K_avx512_transpose(
            mat_A, mat_B_in, mat_B_scales, mat_C,
            N, K, group_size, add_to_c
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
                b_linear_idx = j * K + k;

                size_t scale_idx = b_linear_idx / group_size;
                float scale = mat_B_scales[scale_idx];

                float b = (float)mat_B_in[b_linear_idx] * scale;
                acc += a * b;
            }

            mat_C[i * N + j] = acc;
        }
    }
}
// #endif
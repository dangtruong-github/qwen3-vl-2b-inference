#include "../include/cpu_wrapper.hpp"

#define VERY_LARGE_N 65536

// #if defined(__AVX2__) && defined(__FMA__)
void gemv_lg_N_K(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);

    alignas(64) uint8_t a_q8[K];
    float a_q8_s[K / group_size];

    for (int kk = 0; kk < K; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        __m256 abs_0 = _mm256_set1_ps(-0.0f);
        for (int k = kk; k < kk + 128; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(abs_0, f0);
            v_max = _mm256_max_ps(v_max, abs_f0);
        }
        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 zp_f = _mm256_set1_ps(128.0f);

        // ---------------- quantize ----------------
        for (int k = kk; k < kk + group_size; k += 32) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 f1 = _mm256_loadu_ps(mat_A + k + 8);
            __m256 f2 = _mm256_loadu_ps(mat_A + k + 16);
            __m256 f3 = _mm256_loadu_ps(mat_A + k + 24);

            f0 = _mm256_fmadd_ps(f0, invS, zp_f);
            f1 = _mm256_fmadd_ps(f1, invS, zp_f);
            f2 = _mm256_fmadd_ps(f2, invS, zp_f);
            f3 = _mm256_fmadd_ps(f3, invS, zp_f);

            __m256i i0 = _mm256_cvtps_epi32(f0);
            __m256i i1 = _mm256_cvtps_epi32(f1);
            __m256i i2 = _mm256_cvtps_epi32(f2);
            __m256i i3 = _mm256_cvtps_epi32(f3);

            // int32 -> int16
            __m256i p01 = _mm256_packs_epi32(i0, i1);
            __m256i p23 = _mm256_packs_epi32(i2, i3);

            // fix lane order
            p01 = _mm256_permute4x64_epi64(p01, 0xD8);
            p23 = _mm256_permute4x64_epi64(p23, 0xD8);

            // int16 -> int8 (SIGNED)
            __m256i q8u = _mm256_packus_epi16(p01, p23);
            q8u = _mm256_permute4x64_epi64(q8u, _MM_SHUFFLE(3, 1, 2, 0));

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const size_t K_g = K / group_size;

    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const int8_t *__restrict b0_ptr = mat_B_in + jjK;
        const float *__restrict b_s_ptr = mat_B_scales + (jjK / group_size);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0 = _mm256_setzero_si256();

            __m256i corr32_0 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));

                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                __m256i sum_b0 = _mm256_maddubs_epi16(ones8, b0);
                __m256i prod_0 = _mm256_maddubs_epi16(a_vec, b0);

                corr32_0 = _mm256_add_epi32(corr32_0, _mm256_madd_epi16(sum_b0, ones16));
                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
            }

            corr32_0 = _mm256_slli_epi32(corr32_0, 7);
            c0 = _mm256_sub_epi32(c0, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_s_ptr[g_off]), c0_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
    }
}

void gemv_lg_N_K_g128(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);

    const size_t K_g = K >> 7;

    alignas(64) uint8_t a_q8[K];
    float a_q8_s[K_g];
    
    __m256 zp_f = _mm256_set1_ps(128.0f);

    for (int kk = 0; kk < K; kk += 128) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        __m256 abs_0 = _mm256_set1_ps(-0.0f);
        for (int k = kk; k < kk + 128; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(abs_0, f0);
            v_max = _mm256_max_ps(v_max, abs_f0);
        }
        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk >> 7] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);

        // ---------------- quantize ----------------
        for (int k = kk; k < kk + 128; k += 32) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 f1 = _mm256_loadu_ps(mat_A + k + 8);
            __m256 f2 = _mm256_loadu_ps(mat_A + k + 16);
            __m256 f3 = _mm256_loadu_ps(mat_A + k + 24);

            f0 = _mm256_fmadd_ps(f0, invS, zp_f);
            f1 = _mm256_fmadd_ps(f1, invS, zp_f);
            f2 = _mm256_fmadd_ps(f2, invS, zp_f);
            f3 = _mm256_fmadd_ps(f3, invS, zp_f);

            __m256i i0 = _mm256_cvtps_epi32(f0);
            __m256i i1 = _mm256_cvtps_epi32(f1);
            __m256i i2 = _mm256_cvtps_epi32(f2);
            __m256i i3 = _mm256_cvtps_epi32(f3);

            // int32 -> int16
            __m256i p01 = _mm256_packs_epi32(i0, i1);
            __m256i p23 = _mm256_packs_epi32(i2, i3);

            // fix lane order
            p01 = _mm256_permute4x64_epi64(p01, 0xD8);
            p23 = _mm256_permute4x64_epi64(p23, 0xD8);

            // int16 -> int8 (SIGNED)
            __m256i q8u = _mm256_packus_epi16(p01, p23);
            q8u = _mm256_permute4x64_epi64(q8u, _MM_SHUFFLE(3, 1, 2, 0));

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        const float *__restrict b_s_ptr = mat_B_scales + (jj * K_g);
        
        for (size_t kk = 0; kk < K; kk += 128) {
            const size_t g_off = kk >> 7;
            __m256i c0 = _mm256_setzero_si256();

            __m256i corr32_0 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + 128; k += 32) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));

                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                __m256i sum_b0 = _mm256_maddubs_epi16(ones8, b0);
                __m256i prod_0 = _mm256_maddubs_epi16(a_vec, b0);

                corr32_0 = _mm256_add_epi32(corr32_0, _mm256_madd_epi16(sum_b0, ones16));
                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
            }

            corr32_0 = _mm256_slli_epi32(corr32_0, 7);
            c0 = _mm256_sub_epi32(c0, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_s_ptr[g_off]), c0_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
    }
}

// if defined INT16 x INT16
void f32a_i8f32sb_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    if (M == 1 && N >= 1024 && K >= 1024) {
        if (group_size == 128) {
            gemv_lg_N_K_g128(mat_A, mat_B_in, mat_B_scales, mat_C, N, K);
        } else {
            gemv_lg_N_K(
                mat_A, mat_B_in, mat_B_scales, mat_C, N, K, group_size
            );
        }
        return;
    }
    
    if (group_size == 128) {
        for (size_t i = 0; i < M; ++i) {
            gemv_lg_N_K_g128(
                mat_A + i * K, mat_B_in, mat_B_scales, mat_C + i * N, N, K
            );
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            gemv_lg_N_K(
                mat_A + i * K, mat_B_in, mat_B_scales,
                mat_C + i * N, N, K, group_size
            );
        }
    }

    return;

    const size_t groups_per_row = K / group_size;


    for (size_t i = 0; i < M; ++i) {

        const float *a_row = mat_A + i * K;

        #pragma omp parallel for
        for (size_t j = 0; j < N; ++j) {

            const int8_t *b_row = mat_B_in + j * K;
            const float *b_scale_row = mat_B_scales + j * groups_per_row;

            float acc = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                size_t group = k / group_size;

                float b_dequant =
                    (float)b_row[k] * b_scale_row[group];

                acc += a_row[k] * b_dequant;
            }

            mat_C[i * N + j] = acc;
        }
    }
}
// #endif

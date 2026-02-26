#include "../include/cpu_wrapper.hpp"
#include "../include/cpu_f32a_i8f32sb_f32c.hpp"

#define VERY_LARGE_N 65536

#if defined(__AVX2__) && defined(__FMA__)
void gemm_m4_lgNK_prefix_g128(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C, size_t N, size_t K
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);
    const size_t K_g = K >> 7;
    const size_t K4 = K << 2;
    alignas(32) uint8_t a_q8[K << 2];
    uint8_t *a1_q8_ptr = a_q8 + K;
    uint8_t *a2_q8_ptr = a_q8 + (K << 1);
    uint8_t *a3_q8_ptr = a_q8 + (K * 3);

    float a_q8_s[K >> 5];
    float *a1_q8_s_ptr = a_q8_s + K_g;
    float *a2_q8_s_ptr = a_q8_s + (K_g << 1);
    float *a3_q8_s_ptr = a_q8_s + (K_g * 3);

    for (int kk = 0; kk < K4; kk += 128) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        for (int k = kk; k < kk + 128; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), f0);
            v_max = _mm256_max_ps(v_max, abs_f0);
        }
        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk >> 7] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 zp_f = _mm256_set1_ps(128.0f);

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

    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();
        __m256 c2_f = _mm256_setzero_ps();
        __m256 c3_f = _mm256_setzero_ps();

        const size_t g_off_base = jj * K_g;

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        
        const float *__restrict b_s_ptr_base = mat_B_scales + g_off_base;

        const int *sum_ptr = sum_int8_B + (g_off_base << 3);
        
        for (size_t kk = 0; kk < K; kk += 128) {
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + 128; k += 32) {
                __m256i a0_vec = _mm256_load_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_load_si256((__m256i*)(a1_q8_ptr + k));
                __m256i a2_vec = _mm256_load_si256((__m256i*)(a2_q8_ptr + k));
                __m256i a3_vec = _mm256_load_si256((__m256i*)(a3_q8_ptr + k));

                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                a0_vec = _mm256_maddubs_epi16(a0_vec, b0);
                a1_vec = _mm256_maddubs_epi16(a1_vec, b0);
                a2_vec = _mm256_maddubs_epi16(a2_vec, b0);
                a3_vec = _mm256_maddubs_epi16(a3_vec, b0);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0_vec, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1_vec, ones16));
                c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a2_vec, ones16));
                c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a3_vec, ones16));
            }

            const size_t g_off = kk >> 7;
            const float b_scale = b_s_ptr_base[g_off];
            __m256i corr32_0 = _mm256_load_si256((__m256i*)(sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32_0);
            c1 = _mm256_sub_epi32(c1, corr32_0);
            c2 = _mm256_sub_epi32(c2, corr32_0);
            c3 = _mm256_sub_epi32(c3, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_scale), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_scale), c1_f);
            c2_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), _mm256_set1_ps(a2_q8_s_ptr[g_off] * b_scale), c2_f);
            c3_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), _mm256_set1_ps(a3_q8_s_ptr[g_off] * b_scale), c3_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
        mat_C[N + jj] = add_reduce_mm_256(c1_f);
        mat_C[(N << 1) + jj] = add_reduce_mm_256(c2_f);
        mat_C[3 * N + jj] = add_reduce_mm_256(c3_f);
    }
}

void gemm_m4_lgNK_prefix(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);
    const size_t K_g = K / group_size;
    const size_t K4 = K << 2;
    alignas(32) uint8_t a_q8[K4];
    uint8_t *a1_q8_ptr = a_q8 + K;
    uint8_t *a2_q8_ptr = a_q8 + (K << 1);
    uint8_t *a3_q8_ptr = a_q8 + (K * 3);

    float a_q8_s[K >> 3];
    float *a1_q8_s_ptr = a_q8_s + K_g;
    float *a2_q8_s_ptr = a_q8_s + (K_g << 1);
    float *a3_q8_s_ptr = a_q8_s + (K_g * 3);

    for (int kk = 0; kk < K4; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), f0);
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

    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();
        __m256 c2_f = _mm256_setzero_ps();
        __m256 c3_f = _mm256_setzero_ps();

        const size_t g_off_base = jj * K_g;

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        
        const float *__restrict b_s_ptr_base = mat_B_scales + g_off_base;

        const int *sum_ptr = sum_int8_B + (g_off_base << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a0_vec = _mm256_load_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_load_si256((__m256i*)(a1_q8_ptr + k));
                __m256i a2_vec = _mm256_load_si256((__m256i*)(a2_q8_ptr + k));
                __m256i a3_vec = _mm256_load_si256((__m256i*)(a3_q8_ptr + k));

                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                a0_vec = _mm256_maddubs_epi16(a0_vec, b0);
                a1_vec = _mm256_maddubs_epi16(a1_vec, b0);
                a2_vec = _mm256_maddubs_epi16(a2_vec, b0);
                a3_vec = _mm256_maddubs_epi16(a3_vec, b0);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0_vec, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1_vec, ones16));
                c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a2_vec, ones16));
                c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a3_vec, ones16));
            }

            const size_t g_off = kk / group_size;
            const float b_scale = b_s_ptr_base[g_off];
            __m256i corr32_0 = _mm256_load_si256((__m256i*)(sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32_0);
            c1 = _mm256_sub_epi32(c1, corr32_0);
            c2 = _mm256_sub_epi32(c2, corr32_0);
            c3 = _mm256_sub_epi32(c3, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_scale), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_scale), c1_f);
            c2_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), _mm256_set1_ps(a2_q8_s_ptr[g_off] * b_scale), c2_f);
            c3_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), _mm256_set1_ps(a3_q8_s_ptr[g_off] * b_scale), c3_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
        mat_C[N + jj] = add_reduce_mm_256(c1_f);
        mat_C[(N << 1) + jj] = add_reduce_mm_256(c2_f);
        mat_C[3 * N + jj] = add_reduce_mm_256(c3_f);
    }
}

void gemm_m2_lgNK_prefix_g128(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C, size_t N, size_t K
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);
    const size_t K_g = K >> 7;
    const size_t K2 = K << 1;
    alignas(32) uint8_t a_q8[K2];
    uint8_t *a1_q8_ptr = a_q8 + K;
    float a_q8_s[K >> 6];
    float *a1_q8_s_ptr = a_q8_s + K_g;

    for (int kk = 0; kk < K2; kk += 128) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        for (int k = kk; k < kk + 128; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), f0);
            v_max = _mm256_max_ps(v_max, abs_f0);
        }
        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk >> 7] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 zp_f = _mm256_set1_ps(128.0f);

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

    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();

        const size_t g_off_base = jj * K_g;

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        
        const float *__restrict b_s_ptr_base = mat_B_scales + g_off_base;

        const int *sum_ptr = sum_int8_B + (g_off_base << 3);
        
        for (size_t kk = 0; kk < K; kk += 128) {
            
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + 128; k += 32) {
                __m256i a0_vec = _mm256_load_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_load_si256((__m256i*)(a1_q8_ptr + k));
                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                __m256i prod_0 = _mm256_maddubs_epi16(a0_vec, b0);
                __m256i prod_1 = _mm256_maddubs_epi16(a1_vec, b0);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(prod_1, ones16));
            }

            const size_t g_off = kk >> 7;
            const float b_scale = b_s_ptr_base[g_off];
            __m256i corr32_0 = _mm256_load_si256((__m256i*)(sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32_0);
            c1 = _mm256_sub_epi32(c1, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_scale), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_scale), c1_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
        mat_C[N + jj] = add_reduce_mm_256(c1_f);
    }
}

void gemm_m2_lgNK_prefix(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    // CPUTimer timer("gemv tn1");
    // printf("Shape of gemv: N=%zu, K=%zu\n", N, K);
    const size_t K_g = K / group_size;
    const size_t K2 = K << 1;
    alignas(32) uint8_t a_q8[K2];
    uint8_t *a1_q8_ptr = a_q8 + K;
    float a_q8_s[K >> 4];
    float *a1_q8_s_ptr = a_q8_s + K_g;

    for (int kk = 0; kk < K2; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), f0);
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

    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();

        const size_t g_off_base = jj * K_g;

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        
        const float *__restrict b_s_ptr_base = mat_B_scales + g_off_base;

        const int *sum_ptr = sum_int8_B + (g_off_base << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a0_vec = _mm256_load_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_load_si256((__m256i*)(a1_q8_ptr + k));
                __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                __m256i prod_0 = _mm256_maddubs_epi16(a0_vec, b0);
                __m256i prod_1 = _mm256_maddubs_epi16(a1_vec, b0);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(prod_1, ones16));
            }

            const size_t g_off = kk / group_size;
            const float b_scale = b_s_ptr_base[g_off];
            __m256i corr32_0 = _mm256_load_si256((__m256i*)(sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32_0);
            c1 = _mm256_sub_epi32(c1, corr32_0);
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_scale), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_scale), c1_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
        mat_C[N + jj] = add_reduce_mm_256(c1_f);
    }
}

// if defined INT16 x INT16
void f32a_i8f32sb_f32c_avx2_prefix_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K, size_t group_size
) {
    size_t i = 0;

    if (K <= 4096) {
        if (group_size == 128) {
            for (; i + 4 <= M; i += 4) {
                gemm_m4_lgNK_prefix_g128(mat_A, mat_B_in, mat_B_scales, sum_int8_B, mat_C, N, K);
                mat_A += (K << 2);
                mat_C += (N << 2);
            }

            if (i + 2 <= M) {
                gemm_m2_lgNK_prefix_g128(mat_A, mat_B_in, mat_B_scales, sum_int8_B, mat_C, N, K);
                mat_A += (K << 1);
                mat_C += (N << 1);
            }

            if (i < M) {
                gemv_lg_N_K_g128(mat_A, mat_B_in, mat_B_scales, mat_C, N, K);
            }
        } else {
            for (; i + 4 <= M; i += 4) {
                gemm_m4_lgNK_prefix(mat_A, mat_B_in, mat_B_scales, sum_int8_B, mat_C, N, K, group_size);
                mat_A += (K << 2);
                mat_C += (N << 2);
            }

            if (i + 2 <= M) {
                gemm_m2_lgNK_prefix(
                    mat_A, mat_B_in, mat_B_scales, sum_int8_B, mat_C, N, K, group_size
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

#include "../include/text_layer.hpp"

#if defined(__AVX512F__) && defined(__AVX512DQ__)
void fused_rms_linear_qkv_m2(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *q_ptr, float *k_ptr, float *v_ptr,
    const size_t hidden_size, const size_t kv_dim,
    const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);
    const int *__restrict w_qkv_sum = static_cast<const int*>(w_qkv.sum_int8);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    alignas(64) uint8_t a_q8[hidden_size << 1];
    float a_q8_s[hidden_size >> 4];

    const size_t K_g = hidden_size / group_size;

    // 1. RMS
    for (size_t i_sm = 0; i_sm < 2; ++i_sm) {
        const float *x_now_ptr = x_ptr + i_sm * hidden_size;
        float *t_now_ptr = t_ptr + i_sm * hidden_size;
        uint8_t *a_now_q8_ptr = a_q8 + i_sm * hidden_size;
        float *a_now_q8_s = a_q8_s + i_sm * K_g;

        float ss = 0.0f;
        #pragma omp simd reduction(+:ss)
        for (size_t j = 0; j < hidden_size; ++j) {
            ss += x_now_ptr[j] * x_now_ptr[j];
        }

        float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

        // 2. Normalize + dequantized scale (inplace)
        #pragma omp parallel for schedule(static)
        for (size_t g = 0; g < hidden_size; g += group_size) {
            float s = w_rms_s[g / group_size];
            float combined = s * inv_rms;

            const int8_t *sq = &w_rms_w[g];
            const float *xp = &x_now_ptr[g];
            float *tp = &t_now_ptr[g];

            __m512 combined_v = _mm512_set1_ps(combined);
            __m512 v_max = _mm512_setzero_ps();

            for (size_t k = 0; k + 16 <= group_size; k += 16) {
                __m512 xp0 = _mm512_loadu_ps(xp + k);

                // Load 16 int8 values (unaligned)
                __m128i sq8 = _mm_loadu_si128((const __m128i*)(sq + k));
                __m512i sq32 = _mm512_cvtepi8_epi32(sq8);
                __m512 sqf = _mm512_cvtepi32_ps(sq32);

                xp0 = _mm512_mul_ps(xp0, combined_v);
                xp0 = _mm512_mul_ps(xp0, sqf);

                v_max = _mm512_max_ps(v_max, _mm512_abs_ps(xp0));
                _mm512_storeu_ps(tp + k, xp0);
            }

            float max_val = _mm512_reduce_max_ps(v_max); 
            float scale = max_val / 127.0f;
            float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
            a_now_q8_s[g / group_size] = scale;

            __m512 invS = _mm512_set1_ps(inv_scale);
            __m512 zp_f = _mm512_set1_ps(128.0f);

            for (int k = 0; k < group_size; k += 16) {
                __m512 f0 = _mm512_loadu_ps(tp + k);
                f0 = _mm512_fmadd_ps(f0, invS, zp_f);
                __m512i i0 = _mm512_cvtps_epi32(f0);
                __m128i u0 = _mm512_cvtusepi32_epi8(i0);
                _mm_store_si128((__m128i*)(a_now_q8_ptr + g + k), u0);
            }
        }
    }

    const uint8_t *a1_q8_ptr = a_q8 + hidden_size;
    const float *a1_q8_s_ptr = a_q8_s + K_g;

    const size_t qkv_size = hidden_size + 2 * kv_dim;
    const size_t k_limit = hidden_size + kv_dim;

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < qkv_size; ++jj) {
        __m512 c0_f = _mm512_setzero_ps();
        __m512 c1_f = _mm512_setzero_ps();
    
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict b0_ptr = w_qkv_w + (jj * hidden_size);
        const float *__restrict b_s_ptr = w_qkv_s + jjK_g;
        const int *__restrict b_sum_ptr = w_qkv_sum + (jjK_g << 4);
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();

            for (size_t k = kk; k < kk + group_size; k += 64) {
                __m512i a0_vec = _mm512_loadu_si512((__m512i*)(a_q8 + k));
                __m512i a1_vec = _mm512_loadu_si512((__m512i*)(a1_q8_ptr + k));

                __m512i b_vec = _mm512_loadu_si512((__m512i*)(b0_ptr + k));

                c0 = _mm512_dpbusd_epi32(c0, a0_vec, b_vec);
                c1 = _mm512_dpbusd_epi32(c1, a1_vec, b_vec);
            }

            __m512i corr32 = _mm512_loadu_si512((__m512i*)(b_sum_ptr + (g_off << 4)));

            c0 = _mm512_sub_epi32(c0, corr32);
            c1 = _mm512_sub_epi32(c1, corr32);

            const float b_s = b_s_ptr[g_off];
            
            c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_q8_s[g_off] * b_s), c0_f);
            c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a1_q8_s_ptr[g_off] * b_s), c1_f);
        }

        if (jj < hidden_size) {
            q_ptr[jj] = _mm512_reduce_add_ps(c0_f);
            q_ptr[hidden_size + jj] = _mm512_reduce_add_ps(c1_f);
        } else if (jj < k_limit) {
            const size_t jj_id = jj - hidden_size;
            k_ptr[jj_id] = _mm512_reduce_add_ps(c0_f);
            k_ptr[kv_dim + jj_id] = _mm512_reduce_add_ps(c1_f);
        } else {
            const size_t jj_id = jj - k_limit;
            v_ptr[jj_id] = _mm512_reduce_add_ps(c0_f);
            v_ptr[kv_dim + jj_id] = _mm512_reduce_add_ps(c1_f);
        }
    }
}

void fused_rms_linear_qkv_m4(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *q_ptr, float *k_ptr, float *v_ptr,
    const size_t hidden_size, const size_t kv_dim,
    const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);
    const int *__restrict w_qkv_sum = static_cast<const int*>(w_qkv.sum_int8);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    alignas(64) uint8_t a_q8[hidden_size << 2];
    float a_q8_s[hidden_size >> 3];

    const size_t K_g = hidden_size / group_size;

    for (size_t i_sm = 0; i_sm < 4; ++i_sm) {
        const float *x_now_ptr = x_ptr + i_sm * hidden_size;
        float *t_now_ptr = t_ptr + i_sm * hidden_size;
        uint8_t *a_now_q8_ptr = a_q8 + i_sm * hidden_size;
        float *a_now_q8_s = a_q8_s + i_sm * K_g;

        float ss = 0.0f;
        #pragma omp simd reduction(+:ss)
        for (size_t j = 0; j < hidden_size; ++j) {
            ss += x_now_ptr[j] * x_now_ptr[j];
        }

        float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

        // 2. Normalize + dequantized scale (inplace)
        #pragma omp parallel for schedule(static)
        for (size_t g = 0; g < hidden_size; g += group_size) {
            float s = w_rms_s[g / group_size];
            float combined = s * inv_rms;

            const int8_t *sq = &w_rms_w[g];
            const float *xp = &x_now_ptr[g];
            float *tp = &t_now_ptr[g];

            __m512 combined_v = _mm512_set1_ps(combined);
            __m512 v_max = _mm512_setzero_ps();

            for (size_t k = 0; k + 16 <= group_size; k += 16) {
                __m512 xp0 = _mm512_loadu_ps(xp + k);

                // Load 16 int8 values (unaligned)
                __m128i sq8 = _mm_loadu_si128((const __m128i*)(sq + k));
                __m512i sq32 = _mm512_cvtepi8_epi32(sq8);
                __m512 sqf = _mm512_cvtepi32_ps(sq32);

                xp0 = _mm512_mul_ps(xp0, combined_v);
                xp0 = _mm512_mul_ps(xp0, sqf);

                v_max = _mm512_max_ps(v_max, _mm512_abs_ps(xp0));
                _mm512_storeu_ps(tp + k, xp0);
            }

            float max_val = _mm512_reduce_max_ps(v_max); 
            float scale = max_val / 127.0f;
            float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
            a_now_q8_s[g / group_size] = scale;

            __m512 invS = _mm512_set1_ps(inv_scale);
            __m512 zp_f = _mm512_set1_ps(128.0f);

            for (int k = 0; k < group_size; k += 16) {
                __m512 f0 = _mm512_loadu_ps(tp + k);
                f0 = _mm512_fmadd_ps(f0, invS, zp_f);
                __m512i i0 = _mm512_cvtps_epi32(f0);
                __m128i u0 = _mm512_cvtusepi32_epi8(i0);
                _mm_store_si128((__m128i*)(a_now_q8_ptr + g + k), u0);
            }
        }
    }

    const uint8_t *a1_q8_ptr = a_q8 + hidden_size;
    const uint8_t *a2_q8_ptr = a_q8 + (hidden_size << 1);
    const uint8_t *a3_q8_ptr = a_q8 + (hidden_size * 3);
    const float *a1_q8_s_ptr = a_q8_s + K_g;
    const float *a2_q8_s_ptr = a_q8_s + (K_g << 1);
    const float *a3_q8_s_ptr = a_q8_s + (K_g * 3);

    const size_t qkv_size = hidden_size + 2 * kv_dim;
    const size_t k_limit = hidden_size + kv_dim;
    const size_t q_stride[2] = {hidden_size << 1, hidden_size * 3};
    const size_t kv_stride[2] = {kv_dim << 1, kv_dim * 3};

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < qkv_size; ++jj) {
        __m512 c0_f = _mm512_setzero_ps();
        __m512 c1_f = _mm512_setzero_ps();
        __m512 c2_f = _mm512_setzero_ps();
        __m512 c3_f = _mm512_setzero_ps();
    
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict b0_ptr = w_qkv_w + (jj * hidden_size);
        const float *__restrict b_s_ptr = w_qkv_s + jjK_g;
        const int *__restrict b_sum_ptr = w_qkv_sum + (jjK_g << 4);
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();
            __m512i c2 = _mm512_setzero_si512();
            __m512i c3 = _mm512_setzero_si512();

            for (size_t k = kk; k < kk + group_size; k += 64) {
                __m512i a0_vec = _mm512_loadu_si512((__m512i*)(a_q8 + k));
                __m512i a1_vec = _mm512_loadu_si512((__m512i*)(a1_q8_ptr + k));
                __m512i a2_vec = _mm512_loadu_si512((__m512i*)(a2_q8_ptr + k));
                __m512i a3_vec = _mm512_loadu_si512((__m512i*)(a3_q8_ptr + k));

                __m512i b_vec = _mm512_loadu_si512((__m512i*)(b0_ptr + k));

                c0 = _mm512_dpbusd_epi32(c0, a0_vec, b_vec);
                c1 = _mm512_dpbusd_epi32(c1, a1_vec, b_vec);
                c2 = _mm512_dpbusd_epi32(c2, a2_vec, b_vec);
                c3 = _mm512_dpbusd_epi32(c3, a3_vec, b_vec);
            }

            __m512i corr32 = _mm512_loadu_si512((__m512i*)(b_sum_ptr + (g_off << 4)));

            c0 = _mm512_sub_epi32(c0, corr32);
            c1 = _mm512_sub_epi32(c1, corr32);
            c2 = _mm512_sub_epi32(c2, corr32);
            c3 = _mm512_sub_epi32(c3, corr32);

            const float b_s = b_s_ptr[g_off];
            
            c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_q8_s[g_off] * b_s), c0_f);
            c1_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(a1_q8_s_ptr[g_off] * b_s), c1_f);
            c2_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c2), _mm512_set1_ps(a2_q8_s_ptr[g_off] * b_s), c2_f);
            c3_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c3), _mm512_set1_ps(a3_q8_s_ptr[g_off] * b_s), c3_f);
        }
        
        if (jj < hidden_size) {
            q_ptr[jj] = _mm512_reduce_add_ps(c0_f);
            q_ptr[hidden_size + jj] = _mm512_reduce_add_ps(c1_f);
            q_ptr[q_stride[0] + jj] = _mm512_reduce_add_ps(c2_f);
            q_ptr[q_stride[1] + jj] = _mm512_reduce_add_ps(c3_f);
        } else if (jj < k_limit) {
            const size_t jj_id = jj - hidden_size;
            k_ptr[jj_id] = _mm512_reduce_add_ps(c0_f);
            k_ptr[kv_dim + jj_id] = _mm512_reduce_add_ps(c1_f);
            k_ptr[kv_stride[0] + jj_id] = _mm512_reduce_add_ps(c2_f);
            k_ptr[kv_stride[1] + jj_id] = _mm512_reduce_add_ps(c3_f);
        } else {
            const size_t jj_id = jj - k_limit;
            v_ptr[jj_id] = _mm512_reduce_add_ps(c0_f);
            v_ptr[kv_dim + jj_id] = _mm512_reduce_add_ps(c1_f);
            v_ptr[kv_stride[0] + jj_id] = _mm512_reduce_add_ps(c2_f);
            v_ptr[kv_stride[1] + jj_id] = _mm512_reduce_add_ps(c3_f);
        }
    }
}
#elif defined(__AVX2__) && defined(__FMA__)
void fused_rms_linear_qkv_m2(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *qkv_ptr, const size_t hidden_size,
    const size_t kv_dim, const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);
    const int *__restrict w_qkv_sum = static_cast<const int*>(w_qkv.sum_int8);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    const __m256 abs_0 = _mm256_set1_ps(-0.0f);
    alignas(32) uint8_t a_q8[hidden_size << 1];
    float a_q8_s[hidden_size >> 4];

    const size_t K_g = hidden_size / group_size;

    // 1. RMS
    for (size_t i_sm = 0; i_sm < 2; ++i_sm) {
        const float *x_now_ptr = x_ptr + i_sm * hidden_size;
        float *t_now_ptr = t_ptr + i_sm * hidden_size;
        uint8_t *a_now_q8_ptr = a_q8 + i_sm * hidden_size;
        float *a_now_q8_s = a_q8_s + i_sm * K_g;

        float ss = 0.0f;
        #pragma omp simd reduction(+:ss)
        for (size_t j = 0; j < hidden_size; ++j) {
            ss += x_now_ptr[j] * x_now_ptr[j];
        }

        float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

        // 2. Normalize + dequantized scale (inplace)
        #pragma omp parallel for schedule(static)
        for (size_t g = 0; g < hidden_size; g += group_size) {
            float s = w_rms_s[g / group_size];
            float combined = s * inv_rms;

            const int8_t *sq = &w_rms_w[g];
            const float *xp = &x_now_ptr[g];
            float *tp = &t_now_ptr[g];

            __m256 combined_v = _mm256_set1_ps(combined);
            __m256 v_max = _mm256_setzero_ps();

            for (size_t k = 0; k + 8 <= group_size; k += 8) {
                // -------- first 8 --------
                __m256 xp0 = _mm256_loadu_ps(xp + k);

                __m128i sq8 = _mm_loadl_epi64((const __m128i*)(sq + k));   // load 8 int8
                __m128i sq16 = _mm_cvtepi8_epi16(sq8);                     // 8 int16
                __m256i sq32 = _mm256_cvtepi16_epi32(sq16);                // 8 int32
                __m256 sqf = _mm256_cvtepi32_ps(sq32);                     // 8 float

                xp0 = _mm256_mul_ps(xp0, combined_v);
                xp0 = _mm256_mul_ps(xp0, sqf);

                v_max = _mm256_max_ps(v_max, _mm256_andnot_ps(abs_0, xp0));
                _mm256_storeu_ps(tp + k, xp0);
            }

            float max_val = max_reduce_mm_256(v_max); 
            float scale = max_val / 127.0f;
            float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
            a_now_q8_s[g / group_size] = scale;

            __m256 invS = _mm256_set1_ps(inv_scale);
            __m256 zp_f = _mm256_set1_ps(128.0f);

            // ---------------- quantize ----------------
            for (int k = g; k < g + group_size; k += 32) {
                __m256 f0 = _mm256_loadu_ps(t_now_ptr + k);
                __m256 f1 = _mm256_loadu_ps(t_now_ptr + k + 8);
                __m256 f2 = _mm256_loadu_ps(t_now_ptr + k + 16);
                __m256 f3 = _mm256_loadu_ps(t_now_ptr + k + 24);

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

                _mm256_store_si256((__m256i*)(a_now_q8_ptr + k), q8u);
            }    
        }
    }

    const __m256i ones16 = _mm256_set1_epi16(1);

    const uint8_t *a1_q8_ptr = a_q8 + hidden_size;
    const float *a1_q8_s_ptr = a_q8_s + K_g;

    const size_t qkv_size = hidden_size + 2 * kv_dim;

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < qkv_size; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();
    
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict b0_ptr = w_qkv_w + (jj * hidden_size);
        const float *__restrict b_s_ptr = w_qkv_s + jjK_g;
        const int *__restrict b_sum_ptr = w_qkv_sum + (jjK_g << 3);
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a1_q8_ptr + k));

                __m256i b_vec = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                __m256i prod_0 = _mm256_maddubs_epi16(a0_vec, b_vec);
                __m256i prod_1 = _mm256_maddubs_epi16(a1_vec, b_vec);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(prod_1, ones16));
            }

            __m256i corr32 = _mm256_loadu_si256((__m256i*)(b_sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32);
            c1 = _mm256_sub_epi32(c1, corr32);

            const float b_s = b_s_ptr[g_off];
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_s), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_s), c1_f);
        }
        
        qkv_ptr[jj] = add_reduce_mm_256(c0_f);
        qkv_ptr[qkv_size + jj] = add_reduce_mm_256(c1_f);
    }
}

void fused_rms_linear_qkv_m4(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *qkv_ptr, const size_t hidden_size,
    const size_t kv_dim, const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);
    const int *__restrict w_qkv_sum = static_cast<const int*>(w_qkv.sum_int8);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    const __m256 abs_0 = _mm256_set1_ps(-0.0f);
    alignas(32) uint8_t a_q8[hidden_size << 2];
    float a_q8_s[hidden_size >> 3];

    const size_t K_g = hidden_size / group_size;

    // 1. RMS
    for (size_t i_sm = 0; i_sm < 4; ++i_sm) {
        const float *x_now_ptr = x_ptr + i_sm * hidden_size;
        float *t_now_ptr = t_ptr + i_sm * hidden_size;
        uint8_t *a_now_q8_ptr = a_q8 + i_sm * hidden_size;
        float *a_now_q8_s = a_q8_s + i_sm * K_g;

        float ss = 0.0f;
        #pragma omp simd reduction(+:ss)
        for (size_t j = 0; j < hidden_size; ++j) {
            ss += x_now_ptr[j] * x_now_ptr[j];
        }

        float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

        // 2. Normalize + dequantized scale (inplace)
        #pragma omp parallel for schedule(static)
        for (size_t g = 0; g < hidden_size; g += group_size) {
            float s = w_rms_s[g / group_size];
            float combined = s * inv_rms;

            const int8_t *sq = &w_rms_w[g];
            const float *xp = &x_now_ptr[g];
            float *tp = &t_now_ptr[g];

            __m256 combined_v = _mm256_set1_ps(combined);
            __m256 v_max = _mm256_setzero_ps();

            for (size_t k = 0; k + 8 <= group_size; k += 8) {
                // -------- first 8 --------
                __m256 xp0 = _mm256_loadu_ps(xp + k);

                __m128i sq8 = _mm_loadl_epi64((const __m128i*)(sq + k));   // load 8 int8
                __m128i sq16 = _mm_cvtepi8_epi16(sq8);                     // 8 int16
                __m256i sq32 = _mm256_cvtepi16_epi32(sq16);                // 8 int32
                __m256 sqf = _mm256_cvtepi32_ps(sq32);                     // 8 float

                xp0 = _mm256_mul_ps(xp0, combined_v);
                xp0 = _mm256_mul_ps(xp0, sqf);

                v_max = _mm256_max_ps(v_max, _mm256_andnot_ps(abs_0, xp0));
                _mm256_storeu_ps(tp + k, xp0);
            }

            float max_val = max_reduce_mm_256(v_max); 
            float scale = max_val / 127.0f;
            float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
            a_now_q8_s[g / group_size] = scale;

            __m256 invS = _mm256_set1_ps(inv_scale);
            __m256 zp_f = _mm256_set1_ps(128.0f);

            // ---------------- quantize ----------------
            for (int k = g; k < g + group_size; k += 32) {
                __m256 f0 = _mm256_loadu_ps(t_now_ptr + k);
                __m256 f1 = _mm256_loadu_ps(t_now_ptr + k + 8);
                __m256 f2 = _mm256_loadu_ps(t_now_ptr + k + 16);
                __m256 f3 = _mm256_loadu_ps(t_now_ptr + k + 24);

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

                _mm256_store_si256((__m256i*)(a_now_q8_ptr + k), q8u);
            }    
        }
    }

    const __m256i ones16 = _mm256_set1_epi16(1);

    const uint8_t *a1_q8_ptr = a_q8 + hidden_size;
    const uint8_t *a2_q8_ptr = a_q8 + (hidden_size << 1);
    const uint8_t *a3_q8_ptr = a_q8 + (hidden_size * 3);
    const float *a1_q8_s_ptr = a_q8_s + K_g;
    const float *a2_q8_s_ptr = a_q8_s + (K_g << 1);
    const float *a3_q8_s_ptr = a_q8_s + (K_g * 3);

    const size_t qkv_size = hidden_size + 2 * kv_dim;

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < qkv_size; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();
        __m256 c2_f = _mm256_setzero_ps();
        __m256 c3_f = _mm256_setzero_ps();
    
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict b0_ptr = w_qkv_w + (jj * hidden_size);
        const float *__restrict b_s_ptr = w_qkv_s + jjK_g;
        const int *__restrict b_sum_ptr = w_qkv_sum + (jjK_g << 3);
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a1_q8_ptr + k));
                __m256i a2_vec = _mm256_loadu_si256((__m256i*)(a2_q8_ptr + k));
                __m256i a3_vec = _mm256_loadu_si256((__m256i*)(a3_q8_ptr + k));

                __m256i b_vec = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                a0_vec = _mm256_maddubs_epi16(a0_vec, b_vec);
                a1_vec = _mm256_maddubs_epi16(a1_vec, b_vec);
                a2_vec = _mm256_maddubs_epi16(a2_vec, b_vec);
                a3_vec = _mm256_maddubs_epi16(a3_vec, b_vec);

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0_vec, ones16));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1_vec, ones16));
                c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a2_vec, ones16));
                c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a3_vec, ones16));
            }

            __m256i corr32 = _mm256_loadu_si256((__m256i*)(b_sum_ptr + (g_off << 3)));

            c0 = _mm256_sub_epi32(c0, corr32);
            c1 = _mm256_sub_epi32(c1, corr32);
            c2 = _mm256_sub_epi32(c2, corr32);
            c3 = _mm256_sub_epi32(c3, corr32);

            const float b_s = b_s_ptr[g_off];
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_q8_s[g_off] * b_s), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a1_q8_s_ptr[g_off] * b_s), c1_f);
            c2_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), _mm256_set1_ps(a2_q8_s_ptr[g_off] * b_s), c2_f);
            c3_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), _mm256_set1_ps(a3_q8_s_ptr[g_off] * b_s), c3_f);
        }
        
        qkv_ptr[jj] = add_reduce_mm_256(c0_f);
        qkv_ptr[qkv_size + jj] = add_reduce_mm_256(c1_f);
        qkv_ptr[(qkv_size << 1) + jj] = add_reduce_mm_256(c2_f);
        qkv_ptr[(qkv_size * 3) + jj] = add_reduce_mm_256(c3_f);
    }
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
void fused_rms_linear_qkv_m1(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *q_ptr, float *k_ptr, float *v_ptr,
    const size_t hidden_size, const size_t kv_dim,
    const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    // __m256i a_vec_arr[group_size >> 5];
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256 abs_0 = _mm256_set1_ps(-0.0f);
    alignas(32) uint8_t a_q8[hidden_size];
    float a_q8_s[hidden_size >> 5];

    // 1. RMS
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (size_t j = 0; j < hidden_size; ++j) {
        ss += x_ptr[j] * x_ptr[j];
    }

    float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

    const size_t K_g = hidden_size / group_size;

    // 2. Normalize + dequantized scale (inplace)
    for (size_t g = 0; g < hidden_size; g += group_size) {
        float s = w_rms_s[g / group_size];
        float combined = s * inv_rms;

        const int8_t *sq = &w_rms_w[g];
        const float *xp = &x_ptr[g];
        float *tp = &t_ptr[g];

        __m256 combined_v = _mm256_set1_ps(combined);
        __m256 v_max = _mm256_setzero_ps();

        for (size_t k = 0; k + 8 <= group_size; k += 8) {
            // -------- first 8 --------
            __m256 xp0 = _mm256_loadu_ps(xp + k);

            __m128i sq8 = _mm_loadl_epi64((const __m128i*)(sq + k));   // load 8 int8
            __m128i sq16 = _mm_cvtepi8_epi16(sq8);                     // 8 int16
            __m256i sq32 = _mm256_cvtepi16_epi32(sq16);                // 8 int32
            __m256 sqf = _mm256_cvtepi32_ps(sq32);                     // 8 float

            xp0 = _mm256_mul_ps(xp0, combined_v);
            xp0 = _mm256_mul_ps(xp0, sqf);

            v_max = _mm256_max_ps(v_max, _mm256_andnot_ps(abs_0, xp0));
            _mm256_storeu_ps(tp + k, xp0);
        }

        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[g / group_size] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 zp_f = _mm256_set1_ps(128.0f);

        // ---------------- quantize ----------------
        for (int k = g; k < g + group_size; k += 32) {
            __m256 f0 = _mm256_loadu_ps(t_ptr + k);
            __m256 f1 = _mm256_loadu_ps(t_ptr + k + 8);
            __m256 f2 = _mm256_loadu_ps(t_ptr + k + 16);
            __m256 f3 = _mm256_loadu_ps(t_ptr + k + 24);

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

            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }    
    }

    const size_t qkv_size = hidden_size + 2 * kv_dim;
    const size_t k_limit = hidden_size + kv_dim;

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < qkv_size; ++jj) {
        __m256 c0_f = _mm256_setzero_ps();

        const int8_t *__restrict b0_ptr = w_qkv_w + (jj * hidden_size);
        const float *__restrict b_s_ptr = w_qkv_s + (jj * K_g);
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
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
        
        if (jj < hidden_size) {
            q_ptr[jj] = add_reduce_mm_256(c0_f);
        } else if (jj < k_limit) {
            k_ptr[jj - hidden_size] = add_reduce_mm_256(c0_f);
        } else {
            v_ptr[jj - k_limit] = add_reduce_mm_256(c0_f);
        }
    }
}
#endif

void fused_rms_linear_qkv_dispatch(
    const Tensor *rms_ffn_w, const Tensor *w_attn_qkv, const Tensor *x,
    Tensor *t, Tensor *q, Tensor *k, Tensor *v, size_t M, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
) {
    PtrPair w_qkv = w_attn_qkv->ptr_all({layer_id});

    #if defined(__AVX2__) && defined(__FMA__)
        if (
            !w_attn_qkv->permuted && dtype_w == DType::INT8
            && dtype_s == DType::FP32 && x->dtype == DType::FP32
            && q->dtype == DType::FP32 && text_gq
        ) {
            const PtrPair rms_w = rms_ffn_w->ptr_all({layer_id});

            const float *x_cur_ptr = (const float *)x->ptr();
            float *t_cur_ptr = (float *)t->ptr();
            float *q_cur_ptr = (float *)q->ptr();
            float *k_cur_ptr = (float *)k->ptr();
            float *v_cur_ptr = (float *)v->ptr();

            const size_t qkv_dim = (hidden_size + 2 * kv_dim);

            size_t i = 0;
            for (; i + 4 <= M; i += 4) {
                fused_rms_linear_qkv_m4(
                    rms_w, w_qkv, x_cur_ptr, t_cur_ptr,
                    q_cur_ptr, k_cur_ptr, v_cur_ptr,
                    hidden_size, kv_dim, group_size, rms_norm_eps
                );
                x_cur_ptr += (hidden_size << 2);
                t_cur_ptr += (hidden_size << 2);
                q_cur_ptr += (hidden_size << 2);
                k_cur_ptr += (kv_dim << 2);
                v_cur_ptr += (kv_dim << 2);
            }

            if (i + 2 <= M) {
                fused_rms_linear_qkv_m2(
                    rms_w, w_qkv, x_cur_ptr, t_cur_ptr,
                    q_cur_ptr, k_cur_ptr, v_cur_ptr,
                    hidden_size, kv_dim, group_size, rms_norm_eps
                );
                x_cur_ptr += (hidden_size << 1);
                t_cur_ptr += (hidden_size << 1);
                q_cur_ptr += (hidden_size << 1);
                k_cur_ptr += (kv_dim << 1);
                v_cur_ptr += (kv_dim << 1);
            }

            if (i < M) {
                fused_rms_linear_qkv_m1(
                    rms_w, w_qkv, x_cur_ptr, t_cur_ptr,
                    q_cur_ptr, k_cur_ptr, v_cur_ptr,
                    hidden_size, kv_dim, group_size, rms_norm_eps
                );
            }
        } else {
            rms_norm(x, rms_ffn_w, t, rms_norm_eps, M, layer_id);

            PtrPair w_q = w_qkv;
            PtrPair w_k = w_attn_qkv->ptr_all({layer_id, hidden_size});
            PtrPair w_v = w_attn_qkv->ptr_all({layer_id, hidden_size + kv_dim});
            
            linear(
                t->ptr(), w_q.buf, w_q.scale, w_q.sum_int8, nullptr, nullptr,
                q->ptr(), M, hidden_size, hidden_size, !w_attn_qkv->permuted,
                t->dtype, dtype_w, dtype_s, q->dtype, text_gq, group_size, false
            );
            linear(
                t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, nullptr, nullptr,
                k->ptr(), M, kv_dim, hidden_size, !w_attn_qkv->permuted,
                t->dtype, dtype_w, dtype_s, k->dtype, text_gq, group_size, false
            );
            linear(
                t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, nullptr, nullptr,
                v->ptr(), M, kv_dim, hidden_size, !w_attn_qkv->permuted,
                t->dtype, dtype_w, dtype_s, v->dtype, text_gq, group_size, false
            );
        }
    #else
        rms_norm(x, rms_ffn_w, t, rms_norm_eps, M, layer_id);

        PtrPair w_k = w_attn_qkv->ptr_all({layer_id, hidden_size});
        PtrPair w_v = w_attn_qkv->ptr_all({layer_id, hidden_size + kv_dim});
        
        linear(
            t->ptr(), w_qkv.buf, w_qkv.scale, w_qkv.sum_int8, nullptr, nullptr,
            q->ptr(), M, hidden_size, hidden_size, !w_attn_qkv->permuted,
            t->dtype, dtype_w, dtype_s, q->dtype, text_gq, group_size, false
        );
        linear(
            t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, nullptr, nullptr,
            k->ptr(), M, kv_dim, hidden_size, !w_attn_qkv->permuted,
            t->dtype, dtype_w, dtype_s, k->dtype, text_gq, group_size, false
        );
        linear(
            t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, nullptr, nullptr,
            v->ptr(), M, kv_dim, hidden_size, !w_attn_qkv->permuted,
            t->dtype, dtype_w, dtype_s, v->dtype, text_gq, group_size, false
        );
    #endif
    #ifdef PRINT_LOGITS
        if (!warm_up) {
            const size_t head_dim = qkv->shape[qkv->ndim - 1];
            const size_t num_heads = hidden_size / head_dim;
            const size_t num_kv_heads = kv_dim / head_dim;
            for (size_t i = 0; i < M; ++i) { 
                qkv->printDebug("q", {i});
                qkv->printDebug("k", {i, num_heads});
                qkv->printDebug("v", {i, num_heads + num_kv_heads});
            }
        }
    #endif
}

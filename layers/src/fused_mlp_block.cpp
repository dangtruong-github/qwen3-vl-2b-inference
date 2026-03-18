#include "../include/text_layer.hpp"

void fused_text_mlp_swiglu_m1(
    const PtrPair w_gate, const PtrPair w_up, const float *t_ptr,
    float *gate_ptr, float *up_ptr, const size_t N,
    const size_t K, const size_t group_size
) {
    const int8_t *w_gate_w = static_cast<const int8_t *>(w_gate.buf);
    const float *w_gate_s = static_cast<const float *>(w_gate.scale);
    const int8_t *w_up_w = static_cast<const int8_t *>(w_up.buf);
    const float *w_up_s = static_cast<const float *>(w_up.scale);

    alignas(32) uint8_t a_q8[K];
    float a_q8_s[K >> 5];

    for (int kk = 0; kk < K; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        __m256 abs_0 = _mm256_set1_ps(-0.0f);
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(t_ptr + k);
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

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const size_t K_g = K / group_size;

    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c_up_f = _mm256_setzero_ps();
        __m256 c_gate_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict up_w_ptr = w_up_w + jjK;
        const float *__restrict up_s_ptr = w_up_s + jjK_g;
        const int8_t *__restrict gate_w_ptr = w_gate_w + jjK;
        const float *__restrict gate_s_ptr = w_gate_s + jjK_g;
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c_up = _mm256_setzero_si256();
            __m256i c_gate = _mm256_setzero_si256();

            __m256i corr32_up = _mm256_setzero_si256();
            __m256i corr32_gate = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i a_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));

                __m256i b_up = _mm256_loadu_si256((__m256i*)(up_w_ptr + k));
                __m256i b_gate = _mm256_loadu_si256((__m256i*)(gate_w_ptr + k));

                __m256i sum_b_up = _mm256_maddubs_epi16(ones8, b_up);
                __m256i prod_up = _mm256_maddubs_epi16(a_vec, b_up);

                corr32_up = _mm256_add_epi32(corr32_up, _mm256_madd_epi16(sum_b_up, ones16));
                c_up = _mm256_add_epi32(c_up, _mm256_madd_epi16(prod_up, ones16));

                __m256i sum_b_gate = _mm256_maddubs_epi16(ones8, b_gate);
                __m256i prod_gate = _mm256_maddubs_epi16(a_vec, b_gate);

                corr32_gate = _mm256_add_epi32(corr32_gate, _mm256_madd_epi16(sum_b_gate, ones16));
                c_gate = _mm256_add_epi32(c_gate, _mm256_madd_epi16(prod_gate, ones16));
            }

            corr32_up = _mm256_slli_epi32(corr32_up, 7);
            c_up = _mm256_sub_epi32(c_up, corr32_up);
            corr32_gate = _mm256_slli_epi32(corr32_gate, 7);
            c_gate = _mm256_sub_epi32(c_gate, corr32_gate);
            
            c_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c_up), _mm256_set1_ps(a_q8_s[g_off] * up_s_ptr[g_off]), c_up_f);
            c_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c_gate), _mm256_set1_ps(a_q8_s[g_off] * gate_s_ptr[g_off]), c_gate_f);
        }
        
        const float gate_tmp = add_reduce_mm_256(c_gate_f); 
        const float silu = gate_tmp / (1.0f + expf(-gate_tmp));  // SiLU(x) = x * sigmoid(x)

        const float up_tmp = add_reduce_mm_256(c_up_f);
        gate_ptr[jj] = up_tmp * silu;
    }
}

void fused_text_mlp_swiglu_m2(
    const PtrPair w_gate, const PtrPair w_up, const float *t_ptr,
    float *gate_ptr, float *up_ptr, const size_t N,
    const size_t K, const size_t group_size
) {
    const int8_t *w_gate_w = static_cast<const int8_t *>(w_gate.buf);
    const float *w_gate_s = static_cast<const float *>(w_gate.scale);
    const int *w_gate_sum_int8 = static_cast<const int*>(w_gate.sum_int8);
    const int8_t *w_up_w = static_cast<const int8_t *>(w_up.buf);
    const float *w_up_s = static_cast<const float *>(w_up.scale);
    const int *w_up_sum_int8 = static_cast<const int*>(w_up.sum_int8);

    const size_t K2 = (K << 1);
    alignas(32) uint8_t a_q8[K2];
    float a_q8_s[K >> 4];

    for (int kk = 0; kk < K2; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        __m256 abs_0 = _mm256_set1_ps(-0.0f);
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(t_ptr + k);
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

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const size_t K_g = K / group_size;

    const __m256i ones16 = _mm256_set1_epi16(1);

    // up
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_up_f = _mm256_setzero_ps();
        __m256 c1_up_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict up_w_ptr = w_up_w + jjK;
        const float *__restrict up_s_ptr = w_up_s + jjK_g;
        const int *__restrict up_sum_ptr = w_up_sum_int8 + (jjK_g << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0_up = _mm256_setzero_si256();
            __m256i c1_up = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i b_up = _mm256_loadu_si256((__m256i*)(up_w_ptr + k));

                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));

                __m256i prod0_up = _mm256_maddubs_epi16(a0_vec, b_up);
                
                c0_up = _mm256_add_epi32(c0_up, _mm256_madd_epi16(prod0_up, ones16));

                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a_q8 + K + k));

                __m256i prod1_up = _mm256_maddubs_epi16(a1_vec, b_up);
                
                c1_up = _mm256_add_epi32(c1_up, _mm256_madd_epi16(prod1_up, ones16));
            }

            const size_t g3 = (g_off << 3);
            __m256i corr32_up =_mm256_loadu_si256((__m256i*)(up_sum_ptr + g3));

            c0_up = _mm256_sub_epi32(c0_up, corr32_up);
            c1_up = _mm256_sub_epi32(c1_up, corr32_up);

            const float up_s = up_s_ptr[g_off];
            
            c0_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0_up), _mm256_set1_ps(a_q8_s[g_off] * up_s), c0_up_f);
            c1_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1_up), _mm256_set1_ps(a_q8_s[K_g + g_off] * up_s), c1_up_f);
        }

        up_ptr[jj] = add_reduce_mm_256(c0_up_f);
        up_ptr[N + jj] = add_reduce_mm_256(c1_up_f);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_gate_f = _mm256_setzero_ps();
        __m256 c1_gate_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict gate_w_ptr = w_gate_w + jjK;
        const float *__restrict gate_s_ptr = w_gate_s + jjK_g;
        const int *__restrict gate_sum_ptr = w_gate_sum_int8 + (jjK_g << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0_gate = _mm256_setzero_si256();
            __m256i c1_gate = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i b_gate = _mm256_loadu_si256((__m256i*)(gate_w_ptr + k));

                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));

                __m256i prod0_gate = _mm256_maddubs_epi16(a0_vec, b_gate);
                
                c0_gate = _mm256_add_epi32(c0_gate, _mm256_madd_epi16(prod0_gate, ones16));

                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a_q8 + K + k));

                __m256i prod1_gate = _mm256_maddubs_epi16(a1_vec, b_gate);
                
                c1_gate = _mm256_add_epi32(c1_gate, _mm256_madd_epi16(prod1_gate, ones16));
            }

            const size_t g3 = (g_off << 3);
            __m256i corr32_gate = _mm256_loadu_si256((__m256i*)(gate_sum_ptr + g3));

            c0_gate = _mm256_sub_epi32(c0_gate, corr32_gate);
            c1_gate = _mm256_sub_epi32(c1_gate, corr32_gate);

            const float gate_s = gate_s_ptr[g_off];
            
            c0_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0_gate), _mm256_set1_ps(a_q8_s[g_off] * gate_s), c0_gate_f);
            c1_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1_gate), _mm256_set1_ps(a_q8_s[K_g + g_off] * gate_s), c1_gate_f);
        }
        
        const float gate0_tmp = add_reduce_mm_256(c0_gate_f); 
        const float silu0 = gate0_tmp / (1.0f + expf(-gate0_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[jj] = up_ptr[jj] * silu0;
        
        const float gate1_tmp = add_reduce_mm_256(c1_gate_f); 
        const float silu1 = gate1_tmp / (1.0f + expf(-gate1_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[N + jj] = up_ptr[N + jj] * silu1;
    }
}

void fused_text_mlp_swiglu_m4(
    const PtrPair w_gate, const PtrPair w_up, const float *t_ptr,
    float *gate_ptr, float *up_ptr, const size_t N,
    const size_t K, const size_t group_size
) {
    const int8_t *w_gate_w = static_cast<const int8_t *>(w_gate.buf);
    const float *w_gate_s = static_cast<const float *>(w_gate.scale);
    const int *w_gate_sum_int8 = static_cast<const int*>(w_gate.sum_int8);
    const int8_t *w_up_w = static_cast<const int8_t *>(w_up.buf);
    const float *w_up_s = static_cast<const float *>(w_up.scale);
    const int *w_up_sum_int8 = static_cast<const int*>(w_up.sum_int8);

    const size_t K4 = (K << 2);
    alignas(32) uint8_t a_q8[K4];
    float a_q8_s[K >> 3];
    const size_t K_g = K / group_size;

    const uint8_t *a1_ptr = a_q8 + K;
    const uint8_t *a2_ptr = a_q8 + (K << 1);
    const uint8_t *a3_ptr = a_q8 + K * 3;
    const float *a1_s_ptr = a_q8_s + K_g;
    const float *a2_s_ptr = a_q8_s + (K_g << 1);
    const float *a3_s_ptr = a_q8_s + K_g * 3;

    for (int kk = 0; kk < K4; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        __m256 abs_0 = _mm256_set1_ps(-0.0f);
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(t_ptr + k);
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

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const __m256i ones16 = _mm256_set1_epi16(1);

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_up_f = _mm256_setzero_ps();
        __m256 c1_up_f = _mm256_setzero_ps();
        __m256 c2_up_f = _mm256_setzero_ps();
        __m256 c3_up_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict up_w_ptr = w_up_w + jjK;
        const float *__restrict up_s_ptr = w_up_s + jjK_g;
        const int *__restrict up_sum_ptr = w_up_sum_int8 + (jjK_g << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0_up = _mm256_setzero_si256();
            __m256i c1_up = _mm256_setzero_si256();
            __m256i c2_up = _mm256_setzero_si256();
            __m256i c3_up = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i b_up = _mm256_loadu_si256((__m256i*)(up_w_ptr + k));

                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a1_ptr + k));
                __m256i a2_vec = _mm256_loadu_si256((__m256i*)(a2_ptr + k));
                __m256i a3_vec = _mm256_loadu_si256((__m256i*)(a3_ptr + k));

                a0_vec = _mm256_maddubs_epi16(a0_vec, b_up);
                a1_vec = _mm256_maddubs_epi16(a1_vec, b_up);
                a2_vec = _mm256_maddubs_epi16(a2_vec, b_up);
                a3_vec = _mm256_maddubs_epi16(a3_vec, b_up);
                
                c0_up = _mm256_add_epi32(c0_up, _mm256_madd_epi16(a0_vec, ones16));
                c1_up = _mm256_add_epi32(c1_up, _mm256_madd_epi16(a1_vec, ones16));
                c2_up = _mm256_add_epi32(c2_up, _mm256_madd_epi16(a2_vec, ones16));
                c3_up = _mm256_add_epi32(c3_up, _mm256_madd_epi16(a3_vec, ones16));
            }

            __m256i corr32_up =_mm256_loadu_si256((__m256i*)(up_sum_ptr + (g_off << 3)));

            c0_up = _mm256_sub_epi32(c0_up, corr32_up);
            c1_up = _mm256_sub_epi32(c1_up, corr32_up);
            c2_up = _mm256_sub_epi32(c2_up, corr32_up);
            c3_up = _mm256_sub_epi32(c3_up, corr32_up);

            const float up_s = up_s_ptr[g_off];
            
            c0_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0_up), _mm256_set1_ps(a_q8_s[g_off] * up_s), c0_up_f);
            c1_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1_up), _mm256_set1_ps(a1_s_ptr[g_off] * up_s), c1_up_f);
            c2_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2_up), _mm256_set1_ps(a2_s_ptr[g_off] * up_s), c2_up_f);
            c3_up_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3_up), _mm256_set1_ps(a3_s_ptr[g_off] * up_s), c3_up_f);
        }

        up_ptr[jj] = add_reduce_mm_256(c0_up_f);
        up_ptr[N + jj] = add_reduce_mm_256(c1_up_f);
        up_ptr[(N << 1) + jj] = add_reduce_mm_256(c2_up_f);
        up_ptr[N * 3 + jj] = add_reduce_mm_256(c3_up_f);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m256 c0_gate_f = _mm256_setzero_ps();
        __m256 c1_gate_f = _mm256_setzero_ps();
        __m256 c2_gate_f = _mm256_setzero_ps();
        __m256 c3_gate_f = _mm256_setzero_ps();

        const size_t jjK = jj * K;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict gate_w_ptr = w_gate_w + jjK;
        const float *__restrict gate_s_ptr = w_gate_s + jjK_g;
        const int *__restrict gate_sum_ptr = w_gate_sum_int8 + (jjK_g << 3);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            __m256i c0_gate = _mm256_setzero_si256();
            __m256i c1_gate = _mm256_setzero_si256();
            __m256i c2_gate = _mm256_setzero_si256();
            __m256i c3_gate = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 32) {
                __m256i b_up = _mm256_loadu_si256((__m256i*)(gate_w_ptr + k));

                __m256i a0_vec = _mm256_loadu_si256((__m256i*)(a_q8 + k));
                __m256i a1_vec = _mm256_loadu_si256((__m256i*)(a1_ptr + k));
                __m256i a2_vec = _mm256_loadu_si256((__m256i*)(a2_ptr + k));
                __m256i a3_vec = _mm256_loadu_si256((__m256i*)(a3_ptr + k));

                a0_vec = _mm256_maddubs_epi16(a0_vec, b_up);
                a1_vec = _mm256_maddubs_epi16(a1_vec, b_up);
                a2_vec = _mm256_maddubs_epi16(a2_vec, b_up);
                a3_vec = _mm256_maddubs_epi16(a3_vec, b_up);
                
                c0_gate = _mm256_add_epi32(c0_gate, _mm256_madd_epi16(a0_vec, ones16));
                c1_gate = _mm256_add_epi32(c1_gate, _mm256_madd_epi16(a1_vec, ones16));
                c2_gate = _mm256_add_epi32(c2_gate, _mm256_madd_epi16(a2_vec, ones16));
                c3_gate = _mm256_add_epi32(c3_gate, _mm256_madd_epi16(a3_vec, ones16));
            }

            const size_t g3 = (g_off << 3);
            __m256i corr32_up =_mm256_loadu_si256((__m256i*)(gate_sum_ptr + g3));

            c0_gate = _mm256_sub_epi32(c0_gate, corr32_up);
            c1_gate = _mm256_sub_epi32(c1_gate, corr32_up);
            c2_gate = _mm256_sub_epi32(c2_gate, corr32_up);
            c3_gate = _mm256_sub_epi32(c3_gate, corr32_up);

            const float gate_s = gate_s_ptr[g_off];
            
            c0_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0_gate), _mm256_set1_ps(a_q8_s[g_off] * gate_s), c0_gate_f);
            c1_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1_gate), _mm256_set1_ps(a1_s_ptr[g_off] * gate_s), c1_gate_f);
            c2_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2_gate), _mm256_set1_ps(a2_s_ptr[g_off] * gate_s), c2_gate_f);
            c3_gate_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3_gate), _mm256_set1_ps(a3_s_ptr[g_off] * gate_s), c3_gate_f);
        }
        
        const float gate0_tmp = add_reduce_mm_256(c0_gate_f); 
        const float silu0 = gate0_tmp / (1.0f + expf(-gate0_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[jj] = up_ptr[jj] * silu0;
        
        const float gate1_tmp = add_reduce_mm_256(c1_gate_f); 
        const float silu1 = gate1_tmp / (1.0f + expf(-gate1_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[N + jj] = up_ptr[N + jj] * silu1;
        
        const float gate2_tmp = add_reduce_mm_256(c2_gate_f); 
        const float silu2 = gate2_tmp / (1.0f + expf(-gate2_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[(N << 1) + jj] = up_ptr[(N << 1) + jj] * silu2;
        
        const float gate3_tmp = add_reduce_mm_256(c3_gate_f); 
        const float silu3 = gate3_tmp / (1.0f + expf(-gate3_tmp));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[N * 3 + jj] = up_ptr[N * 3 + jj] * silu3;
        /*
        gate_ptr[jj] = add_reduce_mm_256(c0_gate_f);
        gate_ptr[N + jj] = add_reduce_mm_256(c1_gate_f);
        gate_ptr[(N << 1) + jj] = add_reduce_mm_256(c2_gate_f);
        gate_ptr[N * 3 + jj] = add_reduce_mm_256(c3_gate_f);
        */
    }

    /*
    #pragma omp parallel for simd
    for (size_t i = 0; i < (N << 2); ++i) {
        float x = gate_ptr[i];
        float silu = x / (1.0f + expf(-x));  // SiLU(x) = x * sigmoid(x)
        gate_ptr[i] = silu * up_ptr[i];               // SwiGLU = SiLU(gate) * up
    }
    */
}

void fused_text_rms_mlp_swiglu_dispatch(
    const Tensor *rms_attn_w, const Tensor *w_mlp_gate, const Tensor *w_mlp_up,
    const Tensor *x, Tensor *t, Tensor *gate, Tensor *up, const size_t M,
    const size_t hidden_size, const size_t inter_dim, const DType::Type dtype_w, 
    const DType::Type dtype_s, const bool text_gq, const float eps,
    const size_t group_size, const size_t layer_offset
) {
    if (
        dtype_w == DType::INT8 && dtype_s == DType::FP32 && text_gq && !w_mlp_gate->permuted
        && !w_mlp_gate->permuted && t->dtype == DType::FP32 && gate->dtype == DType::FP32
        && up->dtype == DType::FP32
    ) {
        rms_norm(
            x, rms_attn_w, t, eps, M, layer_offset
        );

        PtrPair w_gate = w_mlp_gate->ptr_all({layer_offset});
        PtrPair w_up = w_mlp_up->ptr_all({layer_offset});

        const float *t_ptr = (const float *)t->ptr();
        float *gate_ptr = (float *)gate->ptr();
        float *up_ptr = (float *)up->ptr();
        
        size_t i = 0;
        for (; i + 4 <= M; i += 4) {
            fused_text_mlp_swiglu_m4(
                w_gate, w_up, t_ptr, gate_ptr,
                up_ptr, inter_dim, hidden_size, group_size
            );
            t_ptr += 4 * hidden_size;
            gate_ptr += 4 * inter_dim;
        }

        if (i + 2 <= M) {
            fused_text_mlp_swiglu_m2(
                w_gate, w_up, t_ptr, gate_ptr,
                up_ptr, inter_dim, hidden_size, group_size
            );
            t_ptr += 2 * hidden_size;
            gate_ptr += 2 * inter_dim;
        }

        if (i < M) {
            fused_text_mlp_swiglu_m1(
                w_gate, w_up, t_ptr, gate_ptr,
                up_ptr, inter_dim, hidden_size, group_size
            );
        }

    } else {
        rms_norm(
            x, rms_attn_w, t, eps, M, layer_offset
        );


        PtrPair w_gate = w_mlp_gate->ptr_all({layer_offset});
        PtrPair w_up = w_mlp_up->ptr_all({layer_offset});
        linear(
            t->ptr(), w_gate.buf, w_gate.scale, w_gate.sum_int8,
            nullptr, nullptr, gate->ptr(), M, inter_dim,
            hidden_size, !w_mlp_gate->permuted, t->dtype,
            dtype_w, dtype_s, gate->dtype,
            text_gq, group_size
        );
        linear(
            t->ptr(), w_up.buf, w_up.scale, w_up.sum_int8, nullptr,
            nullptr, up->ptr(), M, inter_dim,
            hidden_size, !w_mlp_up->permuted, t->dtype,
            dtype_w, dtype_s, up->dtype, text_gq,
            group_size
        );
        
        swiglu(gate, up, M * inter_dim);
    }
}

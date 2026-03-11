#include "../include/text_layer.hpp"

void fused_rms_linear_qkv(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_buf,
    float *t_buf, float *qkv_buf, const size_t prefill_size,
    const size_t hidden_size, const size_t kv_dim, const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);

    const float inv_hs = 1.0f / (float)hidden_size;

    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        
        __m256i a_vec_arr[group_size >> 5];
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256 abs_0 = _mm256_set1_ps(-0.0f);
        // alignas(32) uint8_t a_q8[hidden_size];
        // float a_q8_s[hidden_size >> 5];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < prefill_size; ++i) {
            const float *x_ptr = x_buf + i * hidden_size;
            float *t_ptr = t_buf + i * hidden_size;
            float *qkv_ptr = qkv_buf + i * (hidden_size + (kv_dim << 1));

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
                    a_vec_arr[(k - g) >> 5] = _mm256_permute4x64_epi64(q8u, _MM_SHUFFLE(3, 1, 2, 0));
                }
                
                const size_t g_off = g / group_size;

                size_t jjK, now_jj;
                const int8_t *__restrict b0_ptr;
                float b_s_now;

                for (size_t jj = 0; jj < hidden_size + 2 * kv_dim; ++jj) {
                    jjK = jj * hidden_size;
                    b0_ptr = w_qkv_w + jjK + g;
                    b_s_now = w_qkv_s[jjK / group_size + g_off];

                    __m256 c0_f = _mm256_setzero_ps();
                    __m256i c0 = _mm256_setzero_si256();

                    __m256i corr32 = _mm256_setzero_si256();

                    for (size_t k = 0; k < group_size; k += 32) {
                        __m256i a_vec = a_vec_arr[k >> 5];

                        __m256i b0 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));

                        __m256i sum_b0 = _mm256_maddubs_epi16(ones8, b0);
                        __m256i prod_0 = _mm256_maddubs_epi16(a_vec, b0);

                        corr32 = _mm256_add_epi32(corr32, _mm256_madd_epi16(sum_b0, ones16));
                        c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(prod_0, ones16));
                    }

                    corr32 = _mm256_slli_epi32(corr32, 7);
                    c0 = _mm256_sub_epi32(c0, corr32);
                    
                    c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(scale * b_s_now), c0_f);
                    
                    if (g == 0) {
                        qkv_ptr[jj] = add_reduce_mm_256_layer(c0_f);
                    } else {
                        qkv_ptr[jj] += add_reduce_mm_256_layer(c0_f);
                    }
                }
            }  
        }
    }
}

void fused_rms_linear_qkv_new(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_buf,
    float *t_buf, float *qkv_buf, const size_t prefill_size,
    const size_t hidden_size, const size_t kv_dim, const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);

    const float inv_hs = 1.0f / (float)hidden_size;

    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        
        // __m256i a_vec_arr[group_size >> 5];
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256 abs_0 = _mm256_set1_ps(-0.0f);
        alignas(32) uint8_t a_q8[hidden_size];
        float a_q8_s[hidden_size >> 5];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < prefill_size; ++i) {
            const float *x_ptr = x_buf + i * hidden_size;
            float *t_ptr = t_buf + i * hidden_size;
            float *qkv_ptr = qkv_buf + i * (hidden_size + (kv_dim << 1));

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

            for (size_t jj = 0; jj < hidden_size + 2 * kv_dim; ++jj) {
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
                
                qkv_ptr[jj] = add_reduce_mm_256(c0_f);
            }

        }
    }
}

void fused_rms_linear_qkv_m1(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *qkv_ptr, const size_t hidden_size,
    const size_t kv_dim, const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_qkv_w = static_cast<const int8_t*>(w_qkv.buf);
    const float *__restrict w_qkv_s = static_cast<const float*>(w_qkv.scale);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    // __m256i a_vec_arr[group_size >> 5];
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
    #pragma omp parallel for schedule(static)
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

    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < hidden_size + 2 * kv_dim; ++jj) {
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
        
        qkv_ptr[jj] = add_reduce_mm_256(c0_f);
    }
}

void fused_rms_linear_qkv_dispatch(
    const Tensor *rms_ffn_w, const Tensor *w_attn_qkv, const Tensor *x,
    Tensor *t, Tensor *qkv, size_t M, size_t kv_dim, size_t hidden_size,
    DType::Type dtype_w, DType::Type dtype_s, bool text_gq, size_t group_size,
    const float rms_norm_eps, const size_t layer_id, bool warm_up
) {
    PtrPair w_qkv = w_attn_qkv->ptr_all({layer_id});

    if (
        !w_attn_qkv->permuted && dtype_w == DType::INT8 && dtype_s == DType::FP32
        && x->dtype == DType::FP32 && qkv->dtype == DType::FP32 && text_gq
    ) {
        fused_rms_linear_qkv_new(
            rms_ffn_w->ptr_all({layer_id}), w_qkv,
            (const float *)x->ptr(), (float *)t->ptr(), (float *)qkv->ptr(),
            M, hidden_size, kv_dim, group_size, rms_norm_eps
        );

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
    } else {
        rms_norm(x, rms_ffn_w, t, rms_norm_eps, M, layer_id);

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t i = 0; i < M; ++i) { 
                    t->printDebug("t", {i});
                }
            }
        #endif
        
        linear(
            t->ptr(), w_qkv.buf, w_qkv.scale, w_qkv.sum_int8, nullptr, nullptr,
            qkv->ptr(), M, hidden_size + kv_dim * 2, hidden_size, !w_attn_qkv->permuted,
            t->dtype, dtype_w, dtype_s, qkv->dtype, text_gq, group_size
        );

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
}

#include "../include/text_layer.hpp"

// #if defined(__AVX2__) && defined(__FMA__)
size_t fused_rms_decode(
    const Tensor *w_rms_tensor, const Tensor *w_emb_tensor, float *x_ptr,
    float *logits_ptr, const size_t hidden_size, const size_t vocab_size,
    const size_t group_size, const float eps
) {
    const PtrPair w_rms = w_rms_tensor->ptr_all();
    const PtrPair w_emb = w_emb_tensor->ptr_all();

    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_emb_w = static_cast<const int8_t*>(w_emb.buf);
    const float *__restrict w_emb_s = static_cast<const float*>(w_emb.scale);

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
        float *xp = &x_ptr[g];

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
            _mm256_storeu_ps(xp + k, xp0);
        }

        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[g / group_size] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 zp_f = _mm256_set1_ps(128.0f);

        // ---------------- quantize ----------------
        for (int k = g; k < g + group_size; k += 32) {
            __m256 f0 = _mm256_loadu_ps(x_ptr + k);
            __m256 f1 = _mm256_loadu_ps(x_ptr + k + 8);
            __m256 f2 = _mm256_loadu_ps(x_ptr + k + 16);
            __m256 f3 = _mm256_loadu_ps(x_ptr + k + 24);

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

    constexpr size_t MAX_LOGITS_BATCH = 512;
    
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        alignas(64) int logit_id[MAX_LOGITS_BATCH];
    #else
        alignas(32) int logit_id[MAX_LOGITS_BATCH];
    #endif

    int final_idx;

    for (size_t j_out = 0; j_out < vocab_size; j_out += MAX_LOGITS_BATCH) {
        size_t max_j = std::min(j_out + MAX_LOGITS_BATCH, vocab_size);
        
        #pragma omp parallel for schedule(static)
        for (size_t jj = j_out; jj < max_j; ++jj) {
            __m256 c0_f = _mm256_setzero_ps();

            const int8_t *__restrict b0_ptr = w_emb_w + (jj * hidden_size);
            const float *__restrict b_s_ptr = w_emb_s + (jj * K_g);
            
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
            
            const size_t id_out = jj - j_out;
            if (j_out == 0) {
                logit_id[id_out] = jj;
                logits_ptr[id_out] = add_reduce_mm_256(c0_f);
            } else {
                float val_now = add_reduce_mm_256(c0_f);
                if (logits_ptr[id_out] < val_now) {
                    logits_ptr[id_out] = val_now;
                    logit_id[id_out] = jj;
                }
            }
        }
    }

    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 v_max_vals = _mm512_set1_ps(-FLT_MAX);
        __m512i v_max_idxs = _mm512_setzero_si512();

        // 2. Main Loop (16 elements at a time)
        for (int i = 0; i < MAX_LOGITS_BATCH; i += 16) {
            __m512i v_current_idxs = _mm512_load_epi32((__m512i*)(logit_id + i));
            __m512 v_logits = _mm512_loadu_ps(logits_ptr + i);
            
            // Compare into a 16-bit mask register
            __mmask16 mask = _mm512_cmp_ps_mask(v_logits, v_max_vals, _CMP_GT_OQ);
            
            // Use the mask to update values and indices (only where mask bit is 1)
            v_max_vals = _mm512_mask_mov_ps(v_max_vals, mask, v_logits);
            v_max_idxs = _mm512_mask_mov_epi32(v_max_idxs, mask, v_current_idxs);
        }

        // 3. Horizontal Reduction of the 16 lanes
        float temp_vals[16];
        _mm512_storeu_ps(temp_vals, v_max_vals);
        _mm512_storeu_si512((__m512i*)(logit_id), v_max_idxs);

        float final_max = temp_vals[0];
        final_idx = logit_id[0];
        for (int j = 1; j < 16; ++j) {
            if (temp_vals[j] > final_max) {
                final_max = temp_vals[j];
                final_idx = logit_id[j];
            }
        }
    #else
        __m256 v_max_vals = _mm256_set1_ps(-FLT_MAX);
        __m256i v_max_idxs = _mm256_setzero_si256();
        
        for (int i = 0; i < MAX_LOGITS_BATCH; i += 8) {
            __m256i v_current_idxs = _mm256_load_si256((__m256i*)(logit_id + i));
            __m256 v_logits = _mm256_loadu_ps(logits_ptr + i);
            
            // Compare: result is 0xFFFFFFFF where logits[i] > current_max
            __m256 v_mask = _mm256_cmp_ps(v_logits, v_max_vals, _CMP_GT_OQ);
            
            // Update max values and max indices
            v_max_vals = _mm256_blendv_ps(v_max_vals, v_logits, v_mask);
            v_max_idxs = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(v_max_idxs), 
                _mm256_castsi256_ps(v_current_idxs), 
                v_mask));
        }

        // 3. Horizontal Reduction of the 8 lanes
        float temp_vals[8];
        _mm256_storeu_ps(temp_vals, v_max_vals);
        _mm256_store_si256((__m256i*)(logit_id), v_max_idxs);

        float final_max = temp_vals[0];
        final_idx = logit_id[0];
        for (int j = 1; j < 8; ++j) {
            if (temp_vals[j] > final_max) {
                final_max = temp_vals[j];
                final_idx = logit_id[j];
            }
        }
    #endif

    return (size_t)final_idx;
}
// #endif

size_t fused_rms_decode_dispatch(
    const Tensor *rms_out_w, const Tensor *emb_table,
    Tensor *x, Tensor *logits, const float eps,
    const size_t vocab_size, const size_t hidden_size,
    DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size
) {
    // Final RMSNorm

    size_t token;

    // #if defined(__AVX2__) && defined(__FMA__)
    if (
        x->dtype == DType::FP32 && logits->dtype == DType::FP32
        && dtype_w == DType::INT8 && dtype_s == DType::FP32
        && text_gq
    ) {
        token = fused_rms_decode(
            rms_out_w, emb_table, (float *)x->ptr(),
            (float *)logits->ptr(), hidden_size,
            vocab_size, group_size, eps
        );
    } else {
        rms_norm_inplace(
            (float *)x->ptr(), rms_out_w, eps, 1, 0ll, 1, 0
        );

        // Classifier (LM Head)
        classifier_gemm(
            emb_table, x, logits, vocab_size, hidden_size
        );

        token = greedy_decode((float *)logits->ptr(), vocab_size);
    }
    // #else
    
    // #endif

    return token;
}
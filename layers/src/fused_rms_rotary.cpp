#include "../include/text_layer.hpp"

void fused_rms_rotary_q(
    float *__restrict x /*[batches, hidden]*/,
    const PtrPair scale_ptr /*[hidden]*/,
    const float *cos_row_base, const float *sin_row_base,
    float eps, size_t groups, size_t group_size,
    size_t group_offset, size_t n_heads, size_t head_dim
) {
    const float inv_hs = 1.0f / head_dim;
    // INT8 group-wise scale
    const int8_t *__restrict scale_q =
        static_cast<const int8_t *>(scale_ptr.buf);
    const float *__restrict scale_scales =
        static_cast<const float *>(scale_ptr.scale);

    const int half = head_dim >> 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t g_id = 0; g_id < groups; ++g_id) {
        for (size_t i = 0; i < n_heads; ++i) {
            const float *__restrict cos_row = cos_row_base + g_id * half;
            const float *__restrict sin_row = sin_row_base + g_id * half;
            float *x_ptr = x + g_id * group_offset + i * head_dim;

            // 1. RMS
            float ss = 0.0f;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < head_dim; ++j) {
                ss += x_ptr[j] * x_ptr[j];
            }

            float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

            // 2. Normalize + dequantized scale (inplace)
            for (size_t g = 0; g < head_dim; g += group_size) {
                float s = scale_scales[g / group_size];
                float combined = s * inv_rms;

                const int8_t *sq = &scale_q[g];
                float *xp = &x_ptr[g];

                // Pre-calculate outside the j-loop
                __m256 v_combined = _mm256_set1_ps(combined);
                
                for (size_t j = 0; j + 7 < group_size; j += 8) {
                    __m128i raw_bytes = _mm_loadu_si128((__m128i*)&sq[j]);
                    __m256i v_ints = _mm256_cvtepi8_epi32(raw_bytes);
                    __m256 v_sq_fp32 = _mm256_cvtepi32_ps(v_ints); 

                    // 4. Now you can use it with your other floats
                    __m256 v_xp = _mm256_loadu_ps(&xp[j]);
                    __m256 v_res = _mm256_mul_ps(v_sq_fp32, _mm256_mul_ps(v_combined, v_xp));

                    // Store back to xp
                    _mm256_storeu_ps(&xp[j], v_res);
                }
            }
            
            float *__restrict x1p = x_ptr;
            float *__restrict x2p = x_ptr + half;

            int i_vec_id = 0;
            // --- AVX2 main loop ---
            for (; i_vec_id <= half - 8; i_vec_id += 8) {
                __m256 x1 = _mm256_loadu_ps(x1p + i_vec_id);
                __m256 x2 = _mm256_loadu_ps(x2p + i_vec_id);
                __m256 c  = _mm256_loadu_ps(cos_row + i_vec_id);
                __m256 s  = _mm256_loadu_ps(sin_row + i_vec_id);

                // x1' = x1*c - x2*s
                __m256 y1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));

                // x2' = x1*s + x2*c
                __m256 y2 = _mm256_fmadd_ps(x1, s, _mm256_mul_ps(x2, c));

                _mm256_storeu_ps(x1p + i_vec_id, y1);
                _mm256_storeu_ps(x2p + i_vec_id, y2);
            }
        }
    }
}

void fused_rms_rotary_q_dispatch(
    Tensor *qkv, const Tensor *w_attn_q_norm, const Tensor *cos_tensor,
    const Tensor *sin_tensor, const size_t num_heads, const size_t head_dim,
    const size_t prefill_size, const size_t qkv_stride,
    const size_t layer_offset, const int pos, const float eps
) {
    if (
        qkv->dtype == DType::FP32 && w_attn_q_norm->dtype == DType::INT8
        && w_attn_q_norm->scale_dtype == DType::FP32 && cos_tensor->dtype == DType::FP32
        && sin_tensor->dtype == DType::FP32 && w_attn_q_norm->group_quantized
    ) {
        float *q_ptr = (float *)qkv->ptr();
        PtrPair scale_ptr = w_attn_q_norm->ptr_all({layer_offset});

        const float *__restrict cos_row_base = 
            static_cast<const float *>(cos_tensor->ptr({0, (size_t)pos}));
        const float *__restrict sin_row_base = 
            static_cast<const float *>(sin_tensor->ptr({0, (size_t)pos}));

        fused_rms_rotary_q(
            q_ptr, scale_ptr, cos_row_base, sin_row_base,
            eps, prefill_size, w_attn_q_norm->group_size,
            qkv_stride, num_heads, head_dim
        );
    } else {
        float *q_ptr = (float *)qkv->ptr();
        rms_norm_inplace(
            q_ptr, w_attn_q_norm, eps,
            num_heads, layer_offset, prefill_size, qkv_stride
        );
        apply_rotary(
            qkv, cos_tensor, sin_tensor,
            prefill_size, num_heads, head_dim, pos, qkv_stride
        );
    }
}

void fused_rms_rotary_k(
    float *__restrict x, half_cpu *__restrict k_cache, PtrPair scale_ptr,
    const float *cos_row_base, const float *sin_row_base,
    float eps, size_t groups, size_t group_size, size_t sh_off,
    size_t group_offset, size_t n_heads, size_t head_dim
) {
    const float inv_hs = 1.0f / head_dim;
    // INT8 group-wise scale
    const int8_t *__restrict scale_q =
        static_cast<const int8_t *>(scale_ptr.buf);
    const float *__restrict scale_scales =
        static_cast<const float *>(scale_ptr.scale);

    const int half = head_dim >> 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t g_id = 0; g_id < groups; ++g_id) {
        for (size_t i = 0; i < n_heads; ++i) {
            const float *__restrict cos_row = cos_row_base + g_id * half;
            const float *__restrict sin_row = sin_row_base + g_id * half;
            float *x_ptr = x + g_id * group_offset + i * head_dim;

            // 1. RMS
            float ss = 0.0f;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < head_dim; ++j) {
                ss += x_ptr[j] * x_ptr[j];
            }

            float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

            // 2. Normalize + dequantized scale (inplace)
            for (size_t g = 0; g < head_dim; g += group_size) {
                float s = scale_scales[g / group_size];
                float combined = s * inv_rms;

                const int8_t *sq = &scale_q[g];
                float *xp = &x_ptr[g];

                // Pre-calculate outside the j-loop
                __m256 v_combined = _mm256_set1_ps(combined);
                
                for (size_t j = 0; j + 7 < group_size; j += 8) {
                    __m128i raw_bytes = _mm_loadu_si128((__m128i*)&sq[j]);
                    __m256i v_ints = _mm256_cvtepi8_epi32(raw_bytes);
                    __m256 v_sq_fp32 = _mm256_cvtepi32_ps(v_ints); 

                    // 4. Now you can use it with your other floats
                    __m256 v_xp = _mm256_loadu_ps(&xp[j]);
                    __m256 v_res = _mm256_mul_ps(v_sq_fp32, _mm256_mul_ps(v_combined, v_xp));

                    // Store back to xp
                    _mm256_storeu_ps(&xp[j], v_res);
                }
            }
            
            float *__restrict x1p = x_ptr;
            float *__restrict x2p = x_ptr + half;

            half_cpu *__restrict y1p = k_cache + g_id * head_dim + i * sh_off;
            half_cpu *__restrict y2p = y1p + half;

            for (int i_vec_id = 0; i_vec_id + 8 <= half; i_vec_id += 8) {
                __m256 x1 = _mm256_loadu_ps(x1p + i_vec_id);
                __m256 x2 = _mm256_loadu_ps(x2p + i_vec_id);
                __m256 c  = _mm256_loadu_ps(cos_row + i_vec_id);
                __m256 s  = _mm256_loadu_ps(sin_row + i_vec_id);

                __m256 y1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));
                __m256 y2 = _mm256_fmadd_ps(x1, s, _mm256_mul_ps(x2, c));

                __m128i y1_f16 =
                    _mm256_cvtps_ph(y1, _MM_FROUND_TO_NEAREST_INT);
                __m128i y2_f16 =
                    _mm256_cvtps_ph(y2, _MM_FROUND_TO_NEAREST_INT);

                _mm_storeu_si128((__m128i *)(y1p + i_vec_id), y1_f16);
                _mm_storeu_si128((__m128i *)(y2p + i_vec_id), y2_f16);
            }
        }
    }
}

void fused_rms_rotary_k_dispatch(
    float *k_ptr, char *k_cache_ptr, const Tensor *w_attn_k_norm, const Tensor *cos_tensor,
    const Tensor *sin_tensor, const size_t num_kv_heads, const size_t head_dim,
    const size_t prefill_size, const size_t qkv_stride, const DType::Type k_type,
    const DType::Type cache_type, const size_t layer_offset, const size_t kv_all_off,
    const int pos, const float eps
) {
    if (
        k_type == DType::FP32 && w_attn_k_norm->dtype == DType::INT8
        && w_attn_k_norm->scale_dtype == DType::FP32 && cos_tensor->dtype == DType::FP32
        && sin_tensor->dtype == DType::FP32 && cache_type == DType::FP16
    ) {
        /*
        rms_norm_inplace(
            k_ptr, w_attn_k_norm, eps, num_kv_heads,
            layer_offset, prefill_size, qkv_stride
        );    
        apply_rotary_cache(
            k_ptr, k_cache_ptr, cos_tensor, sin_tensor,
            prefill_size, num_kv_heads, head_dim, pos,
            kv_all_off, cache_type, qkv_stride
        );
        */

        PtrPair scale_ptr = w_attn_k_norm->ptr_all({layer_offset});

        const float *__restrict cos_row_base = 
            static_cast<const float *>(cos_tensor->ptr({0, (size_t)pos}));
        const float *__restrict sin_row_base = 
            static_cast<const float *>(sin_tensor->ptr({0, (size_t)pos}));

        fused_rms_rotary_k(
            k_ptr, (half_cpu *)k_cache_ptr, scale_ptr,
            cos_row_base, sin_row_base, eps, prefill_size,
            w_attn_k_norm->group_size, kv_all_off,
            qkv_stride, num_kv_heads, head_dim
        );
    } else {
        rms_norm_inplace(
            k_ptr, w_attn_k_norm, eps, num_kv_heads,
            layer_offset, prefill_size, qkv_stride
        );    
        apply_rotary_cache(
            k_ptr, k_cache_ptr, cos_tensor, sin_tensor,
            prefill_size, num_kv_heads, head_dim, pos,
            kv_all_off, cache_type, qkv_stride
        );
    }
}

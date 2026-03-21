#include "../include/text_layer.hpp"

void fused_att_decode(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb,  size_t attn_heads, int kv_mul,
    int head_dim, int kv_dim, size_t sh_offset, int pos
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    const size_t att_stride = att->shape[2];
    const size_t seq_len = att->shape[att->ndim - 1];

    const uint16_t *key_cache_fp16 = (const uint16_t *)(key_cache);
    const uint16_t *value_cache_fp16 = (const uint16_t *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

        const uint16_t *__restrict k_ptr =
            key_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

        const float *__restrict q_group_base =
            (const float *)q->ptr({0, h_base});

        float *__restrict att_group_base =
            (float *)att->ptr({0, h_base});

        float *__restrict tb_head =
            tb_base + 1ll * h_base * head_dim;

        const uint16_t *__restrict v_head_base =
            value_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

        gemm_text_qk_att(
            q_group_base, k_ptr, att_group_base,
            inv_sqrt_d, kv_mul, seq_len, head_dim,
            pos + 1, DType::FP32, DType::FP16, DType::FP32
        );

        for (int m = 0; m < kv_mul; ++m) {
            softmax(att_group_base + m * seq_len,
                    (size_t)(pos + 1));
        }

        gemm_text_kv_att(
            att_group_base, v_head_base, tb_head,
            kv_mul, head_dim, seq_len, pos + 1,
            DType::FP32, DType::FP16, DType::FP32
        );
    }
}

void fused_att_prefill(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb,  size_t attn_heads, int kv_mul,
    int head_dim, int kv_dim, size_t sh_offset,
    int pos, const size_t prefill_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    const size_t seq_len = att->shape[att->ndim - 1];

    const uint16_t *value_cache_fp16 = (const uint16_t *)(value_cache);
    const uint16_t *key_cache_fp16 = (const uint16_t *)(key_cache);

    for (size_t b = 0; b < prefill_size; ++b) {

        float *__restrict tb_base =
            (float *)tb->ptr({b});

        const float *__restrict att_base =
            (const float *)att->ptr({b});

        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

            const uint16_t *__restrict k_ptr =
                key_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

            const float *__restrict q_group_base =
                (const float *)q->ptr({b, h_base});

            float *__restrict att_group_base =
                (float *)att->ptr({b, h_base});

            float *__restrict tb_head =
                tb_base + 1ll * h_base * head_dim;

            const uint16_t *__restrict v_head_base =
                value_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

            gemm_text_qk_att(
                q_group_base, k_ptr, att_group_base,
                inv_sqrt_d, kv_mul, seq_len, head_dim,
                pos + b + 1, DType::FP32,
                DType::FP16, DType::FP32
            );

            for (int m = 0; m < kv_mul; ++m) {
                softmax(att_group_base + m * seq_len,
                        (size_t)(pos + b + 1));
            }

            gemm_text_kv_att(
                att_group_base, v_head_base, tb_head,
                kv_mul, head_dim, seq_len, pos + b + 1,
                DType::FP32, DType::FP16, DType::FP32
            );
        }
    }
}

void fused_att_dispatch(
    const char *k_cache_l, const char *v_cache_l, const Tensor *qkv,
    Tensor *att, Tensor *qkv_out, const int num_heads, const int head_dim,
    const int kv_mul, const int kv_dim, const size_t kv_all_off,
    const int pos, const DType::Type kv_dtype, const size_t prefill_size
) {
    if (kv_dtype == DType::FP16) {

        if (prefill_size > 1) {
            fused_att_prefill(
                k_cache_l, v_cache_l, qkv, att,
                qkv_out, num_heads, kv_mul, head_dim,
                kv_dim, kv_all_off, pos, prefill_size
            );
        } else {
            fused_att_decode(
                k_cache_l, v_cache_l, qkv, att,
                qkv_out, num_heads, kv_mul,
                head_dim, kv_dim, kv_all_off, pos
            );
        }
    } else {
        if (prefill_size > 1) {
            attn_scores_all_heads_prefill(
                k_cache_l, qkv, att, num_heads,
                kv_mul, head_dim, kv_dim, kv_all_off,
                pos, prefill_size, kv_dtype
            );
        } else {
            attn_scores_all_heads_decode(
                k_cache_l, qkv, att, num_heads,
                kv_mul, head_dim, kv_dim,
                kv_all_off, pos, kv_dtype
            );
        }

        attn_weighted_sum_all_heads(
            v_cache_l, att, qkv_out, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off,
            pos, prefill_size, kv_dtype
        );
    }
}
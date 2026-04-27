#include "../include/text_layer.hpp"

void fused_att_dispatch(
    const char *k_cache_l, const char *v_cache_l, const float *k_cache_s,
    const float *v_cache_s, const Tensor *q, Tensor *att, Tensor *qkv_out,
    const int num_heads, const int head_dim, const int kv_mul,
    const int kv_dim, const size_t kv_all_off, const int pos,
    const DType::Type key_dtype, const DType::Type value_dtype,
    const size_t group_size, const size_t prefill_size, bool warm_up
) {
    if (prefill_size > 1) {
        attn_scores_all_heads_prefill(
            k_cache_l, k_cache_s, q, att,
            num_heads, kv_mul, head_dim,
            kv_dim, kv_all_off, pos,
            group_size, prefill_size, key_dtype
        );
    } else {
        attn_scores_all_heads_decode(
            k_cache_l, k_cache_s, q, att,
            num_heads, kv_mul, head_dim, kv_dim,
            kv_all_off, pos, group_size, key_dtype
        );
    }

    attn_weighted_sum_all_heads(
        v_cache_l, v_cache_s, att, qkv_out,
        num_heads, kv_mul, head_dim, kv_dim,
        kv_all_off, pos, prefill_size,
        group_size, value_dtype
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < prefill_size; ++i) {
                att->printDebug("att", {i});
                qkv_out->printDebug("qkv_out", {i});
            }
        }
    #endif
}

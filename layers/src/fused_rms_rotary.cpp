#include "../include/text_layer.hpp"

void fused_rms_rotary_q_dispatch(
    Tensor *q, const Tensor *w_attn_q_norm,
    const Tensor *cos_tensor, const Tensor *sin_tensor,
    const size_t num_heads, const size_t head_dim,
    const size_t prefill_size, const size_t layer_offset,
    const int pos, const float eps, bool warm_up
) {
    float *q_ptr = (float *)q->ptr();
    rms_norm_inplace(
        q_ptr, w_attn_q_norm, eps,
        prefill_size * num_heads, layer_offset
    );
    apply_rotary(
        q, cos_tensor, sin_tensor,
        prefill_size, num_heads, head_dim, pos
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < prefill_size; ++i) { 
                qkv->printDebug("q");
            }
        }
    #endif
}

void fused_rms_rotary_k_dispatch(
    Tensor *k, char *k_cache_ptr, float *k_cache_s_ptr,
    const Tensor *key_cache, const Tensor *w_attn_k_norm,
    const Tensor *cos_tensor, const Tensor *sin_tensor,
    const size_t num_kv_heads, const size_t head_dim,
    const size_t prefill_size, const DType::Type k_type,
    const DType::Type cache_type, const size_t layer_offset,
    const size_t kv_all_off, const int pos,
    const float eps, const size_t group_size, bool warm_up
) {
    float *k_ptr = (float *)k->ptr();

    rms_norm_inplace(
        k_ptr, w_attn_k_norm, eps,
        prefill_size * num_kv_heads, layer_offset
    );    
    apply_rotary_cache(
        k_ptr, k_cache_ptr, k_cache_s_ptr,
        cos_tensor, sin_tensor, prefill_size,
        num_kv_heads, head_dim, pos,
        kv_all_off, cache_type, group_size
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            fflush(stdout);
            for (size_t i = 0; i < prefill_size; ++i) {
                for (size_t h_id = 0; h_id < num_kv_heads; ++h_id) {
                    key_cache->printDebug("key_cache", {0, layer_offset, h_id, (size_t)(pos + i)});
                }
            }
        }
    #endif
}

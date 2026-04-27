#include "../include/text_layer.hpp"

void fused_decode_embed_rms_linear_dispatch(
    const Tensor *embed_table, const Tensor *rms_ffn_w,
    const Tensor *w_attn_qkv, Tensor *x, Tensor *t, Tensor *q,
    Tensor *k, Tensor *v, const size_t token_id, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
) { 
    // Embed layer
    embedding_lookup(
        embed_table, x, 0ll, token_id, hidden_size
    );

    fused_rms_linear_qkv_dispatch(
        rms_ffn_w, w_attn_qkv, x, t, q, k, v, 1, kv_dim,
        hidden_size, dtype_w, dtype_s, text_gq,
        group_size, rms_norm_eps, layer_id, warm_up
    );
}
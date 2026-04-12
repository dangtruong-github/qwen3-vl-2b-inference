#include "../include/text_layer.hpp"

void fused_decode_embed_rms_linear_dispatch(
    const Tensor *embed_table, const Tensor *rms_ffn_w,
    const Tensor *w_attn_qkv, Tensor *x, Tensor *t, Tensor *q,
    Tensor *k, Tensor *v, const size_t token_id, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
) {
    PtrPair w_qkv = w_attn_qkv->ptr_all({layer_id});
    const PtrPair rms_w = rms_ffn_w->ptr_all({layer_id});

    const float *x_cur_ptr = (const float *)x->ptr();
    float *t_cur_ptr = (float *)t->ptr();
    float *q_cur_ptr = (float *)q->ptr();
    float *k_cur_ptr = (float *)k->ptr();
    half_cpu *v_cur_ptr = (half_cpu *)v->ptr();
    
    // Embed layer
    embedding_lookup(
        embed_table, x, 0ll, token_id, hidden_size
    );
    
    fused_rms_linear_qkv_m1(
        rms_w, w_qkv, x_cur_ptr, t_cur_ptr,
        q_cur_ptr, k_cur_ptr, v_cur_ptr,
        hidden_size, kv_dim, group_size, rms_norm_eps
    );
}
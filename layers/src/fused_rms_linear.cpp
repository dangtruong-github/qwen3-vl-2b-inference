#include "../include/text_layer.hpp"

void fused_rms_linear_qkv_dispatch(
    const Tensor *rms_ffn_w, const Tensor *w_attn_qkv, const Tensor *x,
    Tensor *t, Tensor *q, Tensor *k, Tensor *v, size_t M, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
) {
    PtrPair w_qkv = w_attn_qkv->ptr_all({layer_id});

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

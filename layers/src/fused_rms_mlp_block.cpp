#include "../include/text_layer.hpp"

void fused_rms_mlp_swiglu_dispatch(
    const Tensor *rms_attn_w, const Tensor *w_mlp_gate, const Tensor *w_mlp_up,
    const Tensor *x, Tensor *t, Tensor *gate, Tensor *up, const size_t M,
    const size_t hidden_size, const size_t inter_dim,
    const DType::Type dtype_w, const DType::Type dtype_s,
    const bool text_gq, const float eps, const size_t group_size,
    const size_t layer_offset, bool warm_up
) {
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
        text_gq, group_size, false
    );
    linear(
        t->ptr(), w_up.buf, w_up.scale, w_up.sum_int8, nullptr,
        nullptr, up->ptr(), M, inter_dim,
        hidden_size, !w_mlp_up->permuted, t->dtype,
        dtype_w, dtype_s, up->dtype, text_gq,
        group_size, false
    );
    
    swiglu(gate, up, M * inter_dim);

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < M; ++i) { 
                gate->printDebug("gate", {i});
            }
        }
    #endif
}

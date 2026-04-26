#include "../include/text_layer.hpp"

size_t fused_rms_decode_dispatch(
    const Tensor *rms_out_w, const Tensor *emb_table,
    Tensor *x, Tensor *logits, const float eps,
    const size_t vocab_size, const size_t hidden_size,
    DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size
) {
    size_t token;
    
    rms_norm_inplace(
        (float *)x->ptr(), rms_out_w, eps, 1, 0ll
    );

    // Classifier (LM Head)
    classifier_gemm(
        emb_table, x, logits, vocab_size, hidden_size
    );

    token = greedy_decode((float *)logits->ptr(), vocab_size);

    return token;
}
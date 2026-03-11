#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <math.h>
#include "../../matmul/module.hpp"
#include "../../utils/module.hpp"
#include "simd_utils.hpp"

// #define DEBUG
#define max(a, b) ((a) > (b) ? (a) : (b))

// ------------------------ Helper functions ------------------------
void embedding_lookup(
    const Tensor *__restrict embedding /*[vocab, hidden]*/, 
    Tensor *__restrict out /*[hidden]*/,
    size_t out_id, size_t token_id, size_t hidden_size
);
void rms_norm(
    const Tensor *__restrict x_tensor /*[hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    Tensor *__restrict out_tensor /*[hidden]*/, 
    float eps, size_t batches, size_t layer_offset
);
void rms_norm_inplace(
    float *x /*[batches, hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    float eps, size_t batches, size_t layer_offset,
    size_t groups, size_t group_offset
);
void classifier_gemm(
    const Tensor *__restrict embedding /*[vocab, hidden]*/,
    const Tensor *__restrict hid_states /*[hidden]*/,
    Tensor *__restrict logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
);
void softmax(float *__restrict x, size_t n);
void add_vector(
    Tensor *__restrict add_to,
    const Tensor *__restrict add_from,
    size_t size_vec = 0
);
void add_vector(
    Tensor *__restrict add_to, const void *__restrict add_from,
    DType::Type add_from_type, size_t size_vec
);
void add_vector(
    void *__restrict add_to, const void *__restrict add_from,
    DType::Type add_from_type, DType::Type add_to_type, size_t size_vec
);
void swiglu(
    Tensor *__restrict gate,  // [d]
    const Tensor *__restrict up,    // [d]
    size_t size_vec
);
void attn_scores_all_heads_prefill(
    const char *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos,
    int prefill_size, DType::Type cache_dtype
);
void attn_scores_all_heads_decode(
    const char *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos,
    DType::Type cache_dtype
);
void attn_weighted_sum_all_heads(
    const char *__restrict value_cache,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos, int prefill_size,
    DType::Type cache_dtype
);
void apply_rotary(
    Tensor *__restrict x,               /* [batch_size, n_heads, head_dim] */
    const Tensor *__restrict cos_table, /* [seq_len, head_dim/2] */
    const Tensor *__restrict sin_table, /* [seq_len, head_dim/2] */
    int batch_size, int n_heads, int head_dim, int pos, size_t stride_x
);
void apply_rotary_cache(
    const float *__restrict in_ptr,
    char *__restrict k_out,
    const Tensor *__restrict cos_table,
    const Tensor *__restrict sin_table,
    int batch_size, int n_heads, int head_dim, int pos,
    size_t sh_off, DType::Type cache_dtype, size_t in_stride
);

// fused kernels
void fused_rms_linear_qkv_dispatch(
    const Tensor *rms_ffn_w, const Tensor *w_attn_qkv, const Tensor *x,
    Tensor *t, Tensor *qkv, size_t M, size_t kv_dim, size_t hidden_size,
    DType::Type dtype_w, DType::Type dtype_s, bool text_gq, size_t group_size,
    const float rms_norm_eps, const size_t layer_id, bool warm_up
);
void fused_text_mlp_swiglu_dispatch(
    const Tensor *w_mlp_gate, const Tensor *w_mlp_up, const Tensor *t,
    Tensor *gate, Tensor *up, const size_t prefill_size, const size_t hidden_size,
    const size_t inter_dim, const DType::Type dtype_w, const DType::Type dtype_s,
    const bool text_gq, const size_t group_size, const size_t layer_offset
);

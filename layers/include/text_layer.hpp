#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <math.h>
#include <assert.h>
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
    float *__restrict x /*[batches, hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    float eps, size_t batches, size_t layer_offset
);
void classifier_gemm(
    const Tensor *__restrict embedding /*[vocab, hidden]*/,
    const Tensor *__restrict hid_states /*[hidden]*/,
    Tensor *__restrict logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
);
void softmax(float *__restrict x, size_t n);
void softmax_with_max(float *__restrict x, float max_val, size_t n);
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
    const float *__restrict key_cache_scale,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos, size_t group_size,
    int prefill_size, DType::Type cache_dtype
);
void attn_scores_all_heads_decode(
    const char *__restrict key_cache,
    const float *__restrict key_cache_scale,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos,
    size_t group_size, DType::Type cache_dtype
);
void attn_weighted_sum_all_heads(
    const char *__restrict value_cache,
    const float *__restrict value_cache_scale,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos, int prefill_size,
    const size_t group_size, DType::Type cache_dtype
);
void apply_rotary(
    Tensor *__restrict x,               /* [batch_size, n_heads, head_dim] */
    const Tensor *__restrict cos_table, /* [seq_len, head_dim/2] */
    const Tensor *__restrict sin_table, /* [seq_len, head_dim/2] */
    int batch_size, int n_heads, int head_dim, int pos
);
void apply_rotary_cache(
    const float *__restrict in_ptr,
    char *__restrict k_out,
    float *__restrict k_s_out,
    const Tensor *__restrict cos_table,
    const Tensor *__restrict sin_table,
    int batch_size, int n_heads, int head_dim,
    int pos, size_t sh_off, DType::Type cache_dtype,
    const size_t group_size
);
void copy_to_v_cache(
    const Tensor *v, char *v_cache_l, float *v_cache_s,
    const DType::Type v_cache_dtype, const size_t prefill_size,
    const size_t head_dim, const size_t num_kv_heads,
    const size_t kv_pos_off, const size_t kv_all_off,
    const size_t kv_pos_scale_off,
    const size_t cache_group_size, bool warm_up
);
size_t greedy_decode(float* logits, int vocab_size);

// fused kernels
void fused_rms_linear_qkv_dispatch(
    const Tensor *rms_ffn_w, const Tensor *w_attn_qkv, const Tensor *x,
    Tensor *t, Tensor *q, Tensor *k, Tensor *v, size_t M, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
);
void fused_decode_embed_rms_linear_dispatch(
    const Tensor *embed_table, const Tensor *rms_ffn_w,
    const Tensor *w_attn_qkv, Tensor *x, Tensor *t, Tensor *q,
    Tensor *k, Tensor *v, const size_t token_id, size_t kv_dim,
    size_t hidden_size, DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size, const float rms_norm_eps,
    const size_t layer_id, bool warm_up
);
void fused_rms_rotary_q_dispatch(
    Tensor *q, const Tensor *w_attn_q_norm,
    const Tensor *cos_tensor, const Tensor *sin_tensor,
    const size_t num_heads, const size_t head_dim,
    const size_t prefill_size, const size_t layer_offset,
    const int pos, const float eps, bool warm_up
);
void fused_rms_rotary_k_dispatch(
    Tensor *k, char *k_cache_ptr, float *k_cache_s_ptr,
    const Tensor *key_cache, const Tensor *w_attn_k_norm,
    const Tensor *cos_tensor, const Tensor *sin_tensor,
    const size_t num_kv_heads, const size_t head_dim,
    const size_t prefill_size, const DType::Type k_type,
    const DType::Type cache_type, const size_t layer_offset,
    const size_t kv_all_off, const int pos,
    const float eps, const size_t group_size, bool warm_up
);
void fused_rms_mlp_swiglu_dispatch(
    const Tensor *rms_attn_w, const Tensor *w_mlp_gate, const Tensor *w_mlp_up,
    const Tensor *x, Tensor *t, Tensor *gate, Tensor *up, const size_t M,
    const size_t hidden_size, const size_t inter_dim,
    const DType::Type dtype_w, const DType::Type dtype_s,
    const bool text_gq, const float eps, const size_t group_size,
    const size_t layer_offset, bool warm_up
);
size_t fused_rms_decode_dispatch(
    const Tensor *rms_out_w, const Tensor *emb_table,
    Tensor *x, Tensor *logits, const float eps,
    const size_t vocab_size, const size_t hidden_size,
    DType::Type dtype_w, DType::Type dtype_s,
    bool text_gq, size_t group_size
);
void fused_att_dispatch(
    const char *k_cache_l, const char *v_cache_l, const float *k_cache_s,
    const float *v_cache_s, const Tensor *q, Tensor *att,
    Tensor *qkv_out, const int num_heads, const int head_dim, const int kv_mul,
    const int kv_dim, const size_t kv_all_off, const int pos,
    const DType::Type key_dtype, const DType::Type value_dtype,
    const size_t group_size, const size_t prefill_size, bool warm_up
);

#if defined(__AVX2__) && defined(__FMA__)
void fused_rms_linear_qkv_m1(
    const PtrPair w_rms, const PtrPair w_qkv, const float *x_ptr,
    float *t_ptr, float *q_ptr, float *k_ptr, half_cpu *v_ptr,
    const size_t hidden_size, const size_t kv_dim,
    const size_t group_size, const float eps
);
#endif

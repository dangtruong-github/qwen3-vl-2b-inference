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
    size_t token_id, size_t hidden_size
);
void rms_norm(
    const Tensor *__restrict x_tensor /*[hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    Tensor *__restrict out_tensor /*[hidden]*/, 
    float eps, size_t batches, size_t layer_offset
);
void rms_norm_inplace(
    Tensor *__restrict x_tensor /*[hidden]*/,
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
void add_vector(
    Tensor *__restrict add_to,
    const Tensor *__restrict add_from,
    size_t size_vec = 0
);
void add_vector(
    Tensor *__restrict add_to,
    const void *__restrict add_from,
    DType::Type add_from_type, size_t size_vec
);
void swiglu(
    Tensor *__restrict gate,  // [d]
    const Tensor *__restrict up,    // [d]
    size_t size_vec
);
void attn_scores_all_heads(
    const float *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos
);
void attn_weighted_sum_all_heads(
    const float *__restrict value_cache,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos
);
void apply_rotary(
    Tensor *__restrict x /*[n_heads*hd]*/,
    const Tensor *__restrict cos_table /*[seq_len*hd/2]*/,
    const Tensor *__restrict sin_table /*[seq_len*hd/2]*/,
    int n_heads, int head_dim, int pos
);
void apply_rotary_cache(
    const Tensor *__restrict in /*[n_heads*hd]*/,
    float *__restrict k_out      /*[n_heads*seq_len*hd]*/,
    const Tensor *__restrict cos_table /*[seq_len*hd/2]*/,
    const Tensor *__restrict sin_table /*[seq_len*hd/2]*/,
    int n_heads, int head_dim, int pos, size_t sh_off
);

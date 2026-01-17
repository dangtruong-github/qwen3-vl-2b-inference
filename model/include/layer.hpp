#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include "../../matmul/module.hpp"
#include "../../utils/module.hpp"

// #define DEBUG
#define max(a, b) ((a) > (b) ? (a) : (b))

// ------------------------ Helper functions ------------------------
void embedding_lookup(
    const Tensor *embedding /*[vocab, hidden]*/, size_t token_id,
    Tensor *out /*[hidden]*/, size_t hidden_size
);
void rms_norm(
    const float *x /*[hidden]*/, const Tensor *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps,
    size_t batches, size_t layer_offset
);
void classifier_gemm(
    const Tensor *embedding /*[vocab, hidden]*/,
    const Tensor *hid_states /*[hidden]*/, Tensor *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
);
void softmax(float *x, size_t n);
void add_vector(Tensor *add_to, const Tensor *add_from, size_t size_vec = 0);
void add_vector(Tensor *add_to, const float *add_from, size_t size_vec = 0);
void swiglu(
    Tensor *gate /*[d]*/, const Tensor *up /*[d]*/, size_t size_vec
);
void attn_scores_all_heads(
    const Tensor *key_cache, const Tensor *q, Tensor *att,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
);
void attn_weighted_sum_all_heads(
    const Tensor *value_cache, const Tensor *att, Tensor *tb,
    size_t layer_offset, int attn_heads, int kv_mul, int head_dim, int kv_dim,
    int seq_len, int pos
);
void apply_rotary(
    float *x /*[n_heads*hd]*/, const Tensor *cos_table /*[seq_len*hd/2]*/,
    const Tensor *sin_table /*[seq_len*hd/2]*/, int n_heads, int head_dim,
    int pos
);

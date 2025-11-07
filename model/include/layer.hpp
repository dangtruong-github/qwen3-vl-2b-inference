#pragma once
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>

// Forward declarations for helper functions
void layer_norm(float* buffer, const float* input, const float* weight, const float* bias,
                int hidden_size, float eps);
float gelu(float x);
int argmax(const float* array, int size);



// ------------------------ Helper functions ------------------------
void embedding_lookup(
    const float *embedding /*[vocab, hidden]*/,
    int token_id, float *out /*[hidden]*/,
    size_t vocab_size, size_t hidden
);
void rms_norm(
    const float *x /*[hidden]*/, const float *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps, size_t hidden_size, size_t layer_offset
);
void linear(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
);
void classifier_gemm(
    const float *embedding /*[vocab, hidden]*/,
    const float *hid_states /*[hidden]*/, float *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
);
void qkv_project(
    const float *x /*[hidden]*/,
    const float *W_qkv /*[(n_q+2*n_kv)*hd, hidden]*/,
    const float *b_qkv /*[(n_q+2*n_kv)*hd]*/,
    float *qkv /*[(n_q+2*n_kv)*hd]*/,
    size_t hidden, size_t n_q, size_t n_kv, size_t layer_offset
);
void swiglu(
    const float *gate /*[d]*/, const float *up /*[d]*/,
    float *out /*[d]*/, size_t size_vec
);
void add_vector(float *add_to, const float *add_from, size_t size_vec);
void attn_scores_all_heads(
    const float *key_cache, const float *q, float *att, size_t loff_one,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
);
void attn_weighted_sum_all_heads(
    const float *value_cache, const float *q, const float *att, float *tb,
    size_t loff, int attn_heads, int kv_mul, int head_dim, int kv_dim,
    int seq_len, int pos
);
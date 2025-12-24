#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "matmul_cpu.hpp"
#include "../../utils/module.hpp"

// #define DEBUG
#define max(a, b) ((a) > (b) ? (a) : (b))

// Forward declarations for helper functions
float gelu(float x);
int argmax(const float* array, int size);



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
void conv_3d(
    const Tensor *conv_w, const Tensor *conv_b, float *in_img, float *out_img,
    long img_h, long VC, long VTP, long VP, long VH
);
void vision_pos_embed(const Tensor *pos_embed_w, float *x_embed, int grid_h, int grid_w, int num_grid_per_side, int VSP, int VH);
void vision_rot_pos_emb(
    float *pos_emb_out_cos, float *pos_emb_out_sin,
    const float *cos_tensor, const float *sin_tensor,
    int grid_h, int grid_w, int merge_size, int head_dim
);
void layer_norm(
    const Tensor *x,           /* [batches, hidden] */
    const Tensor *scale,       /* [layers, hidden] */
    const Tensor *bias,        /* [layers, hidden] */
    Tensor *out,               /* [batches, hidden] */
    float eps, 
    size_t batches, 
    size_t layer_offset
);
void vision_apply_rotary_inplace(
    const float *cos_tensor, // shape (total_tokens, head_dim)
    const float *sin_tensor, // shape (total_tokens, head_dim)
    float *buffer,           // shape (total_tokens, num_heads, head_dim)
    long total_tokens,
    int num_heads,
    int head_dim
);
void tensor_transpose(const float *in, float *out, int dim_0, int dim_1, int dim_2);
void vision_att(
    const float *q, const float *k, const float *v, float *attn_scores,
    float *out, int num_heads, int total_tokens, int head_dim, float scale
);
void gelu_tanh(Tensor *x, size_t x_size);
#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include "../../matmul/module.hpp"
#include "../../utils/module.hpp"
#include "layer.hpp"

// #define DEBUG
#define max(a, b) ((a) > (b) ? (a) : (b))

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
void tensor_transpose(const float *in, float *out, int D0, int D1, int D2);
void vision_att(
    const float *q, const float *k, const float *v, float *attn_scores,
    float *out, int num_heads, int total_tokens, int head_dim, float scale
);
void gelu_tanh(Tensor *x, size_t x_size);

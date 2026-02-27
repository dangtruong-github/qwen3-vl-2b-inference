#pragma once

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <immintrin.h>
#include "../../matmul/module.hpp"
#include "../../utils/module.hpp"
#include "text_layer.hpp"
#include "simd_utils.hpp"

// #define DEBUG
#define max(a, b) ((a) > (b) ? (a) : (b))

void conv_3d(
    const Tensor *__restrict conv_w, const Tensor *__restrict conv_b,
    const float *__restrict in_img, Tensor *__restrict out_img_tensor,
    long img_h, long VC, long VTP, long VP, long VH
);
void vision_pos_embed(
    const Tensor *__restrict pos_embed_w, Tensor *__restrict x_embed,
    int grid_h, int grid_w, int num_grid_per_side, int VSP, int VH
);
void vision_rot_pos_emb(
    Tensor *__restrict pe_cos, Tensor *__restrict pe_sin,
    const Tensor *__restrict cos_total, const Tensor *__restrict sin_total,
    int grid_h, int grid_w, int merge_size, int head_dim
);
void layer_norm(
    const Tensor *__restrict x,           /* [batches, hidden] */
    const Tensor *__restrict scale,       /* [layers, hidden] */
    const Tensor *__restrict bias,        /* [layers, hidden] */
    Tensor *__restrict out,               /* [batches, hidden] */
    float eps, size_t batches, size_t layer_offset
);
void vision_apply_rotary_inplace(
    const Tensor *__restrict cos_total, // shape (T, HD)
    const Tensor *__restrict sin_total, // shape (T, HD)
    Tensor *__restrict tensor_buffer,   // shape (T, NH, HD)
    long total_tokens, int num_heads, int head_dim
);
void tensor_transpose(
    const Tensor *__restrict in,
    Tensor *__restrict out,
    int D0, int D1, int D2
);
void vision_att(
    const Tensor *q_tensor, const Tensor *k_tensor,
    const Tensor *v_tensor, Tensor *attn_scores_tensor, 
    Tensor *out_tensor, int num_heads, int T, int D, size_t max_attn_size, float scale
);
void gelu_tanh(Tensor *x, size_t x_size);

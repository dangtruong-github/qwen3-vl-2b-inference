#pragma once

#include <stdio.h>
#include "../utils/module.hpp"

typedef struct {
    int seq_len;
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int rope_theta;
    float rms_norm_eps;
    int *mrope_section;
    int num_dimensions;

    long long min_pixels;
    long long max_pixels;
    float vision_theta;
    float vision_scale;
    int vision_hidden_size;
    int vision_depth;
    int vision_patch_size;
    int vision_spatial_merge_size;
    int vision_temporal_patch_size;
    int vision_num_heads;
    int vision_intermediate_size;
    int out_hidden_size;
    int image_token_id;
    int vision_start_token_id;
    int vision_end_token_id;
    int video_token_id;
    int vision_num_channels;
    int vision_deep_stack_depth;
    int max_vision_embeddings;
    int *deep_layer;

    int text_bits;
    int group_size;
    int vision_bits;
    int group_quantized;

    int max_prefill_size;
} QwenConfig;

typedef struct {
    // Language Model Weights
    Tensor *token_embedding_table; // [vocab_size, hidden_size]
    Tensor *rms_out_w;         // [hidden_size]

    // Language Model Layer Weights (continuous blocks: [num_hidden_layers, ...])
    Tensor *rms_ffn_w;          // [L, H]
    Tensor *w_mlp_down;            // [L, H, I]
    Tensor *w_mlp_gate;            // [L, I, H]
    Tensor *w_mlp_up;              // [L, I, H]
    Tensor *rms_attn_w; // [L, H]
    Tensor *w_attn_k_norm;         // [L, KVA_Dim]
    Tensor *w_attn_k;         // [L, KVA_Dim, H]
    Tensor *w_attn_o;         // [L, H, H]
    Tensor *w_attn_q_norm;         // [L, Q_Dim]
    Tensor *w_attn_q;         // [L, Q_Dim, H]
    Tensor *w_attn_v;         // [L, KVA_Dim, H]

    // Vision Model Weights (General)
    Tensor *vl_patch_emb_b;
    Tensor *vl_patch_emb_w;
    Tensor *vl_pos_emb_w;
    
    Tensor *vl_attn_proj_b;
    Tensor *vl_attn_proj_w;
    Tensor *vl_attn_qkv_b;
    Tensor *vl_attn_qkv_w;
    Tensor *vl_mlp1_b;
    Tensor *vl_mlp1_w;
    Tensor *vl_mlp2_b;
    Tensor *vl_mlp2_w;
    Tensor *vl_norm1_b;
    Tensor *vl_norm1_w;
    Tensor *vl_norm2_b;
    Tensor *vl_norm2_w;

    Tensor *vl_d_mlp1_b;
    Tensor *vl_d_mlp1_w;
    Tensor *vl_d_mlp2_b;
    Tensor *vl_d_mlp2_w;
    Tensor *vl_d_norm_b;
    Tensor *vl_d_norm_w;

    Tensor *vl_merge_mlp1_b;
    Tensor *vl_merge_mlp1_w;
    Tensor *vl_merge_mlp2_b;
    Tensor *vl_merge_mlp2_w;
    Tensor *vl_merge_norm_b;
    Tensor *vl_merge_norm_w;
} QwenWeight;

typedef struct {
    // ---- Hidden / intermediate buffers ----
    Tensor *x;            // current hidden state [hidden_size]
    Tensor *t;            // normalized hidden before attention [hidden_size]

    // ---- Attention projections ----
    Tensor *q;            // query [num_attention_heads * head_dim]
    Tensor *k;            // query [num_attention_heads * head_dim]
    Tensor *v;            // query [num_attention_heads * head_dim]

    Tensor *att;          // attention scores (temporary buffer) [num_attention_heads * max_position_embeddings]
    Tensor *qkv_out;      // attention output before projection [hidden_size]

    // ---- MLP intermediate ----
    Tensor *gate;         // gate projection [intermediate_size]
    Tensor *up;           // up projection [intermediate_size]

    // ---- Rotary embeddings ----
    Tensor *cos_tensor;   // cached cosines for rotary embedding [max_position_embeddings * head_dim/2]
    Tensor *sin_tensor;   // cached sines for rotary embedding [max_position_embeddings * head_dim/2]

    // ---- Output ----
    Tensor *logits;       // final logits [vocab_size]

    // ---- KV cache (for autoregressive decoding) ----
    Tensor *key_cache;    // [num_hidden_layers, num_key_value_heads, max_position_embeddings, head_dim]
    Tensor *value_cache;  // same shape

    // vision
    Tensor *vision_x;
    Tensor *vision_t;
    Tensor *vision_q;
    Tensor *vision_k;

    Tensor *vision_cos_tensor;
    Tensor *vision_sin_tensor;

    Tensor *vision_pe_cos;
    Tensor *vision_pe_sin;
    
    Tensor *vision_mlp_out;
    Tensor *vision_deep_stack;
    Tensor *vision_attn_scores;

    int vision_embed_tokens;
    int cur_img_token_id;
} QwenRunState;

#pragma once

#include <stdio.h>

// Helper to check for allocation errors
#define CHECK_ALLOC(ptr, size) if (!ptr) { \
    fprintf(stderr, "Error: failed to allocate memory for %ld bytes\n", (long)(size)); \
    return; \
}

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
} QwenConfig;

typedef struct {
    // Language Model Weights
    const float *token_embedding_table; // [vocab_size, hidden_size]
    const float *rms_out_w;         // [hidden_size]

    // Language Model Layer Weights (continuous blocks: [num_hidden_layers, ...])
    const float *rms_ffn_w;          // [L, H]
    const float *w_mlp_down;            // [L, H, I]
    const float *w_mlp_gate;            // [L, I, H]
    const float *w_mlp_up;              // [L, I, H]
    const float *rms_attn_w; // [L, H]
    const float *w_attn_k_norm;         // [L, KVA_Dim]
    const float *w_attn_k;         // [L, KVA_Dim, H]
    const float *w_attn_o;         // [L, H, H]
    const float *w_attn_q_norm;         // [L, Q_Dim]
    const float *w_attn_q;         // [L, Q_Dim, H]
    const float *w_attn_v;         // [L, KVA_Dim, H]

    // Vision Model Weights (General)
    const float *vl_patch_emb_b;
    const float *vl_patch_emb_w;
    const float *vl_pos_emb_w;
    
    const float *vl_attn_proj_b;
    const float *vl_attn_proj_w;
    const float *vl_attn_qkv_b;
    const float *vl_attn_qkv_w;
    const float *vl_mlp1_b;
    const float *vl_mlp1_w;
    const float *vl_mlp2_b;
    const float *vl_mlp2_w;
    const float *vl_norm1_b;
    const float *vl_norm1_w;
    const float *vl_norm2_b;
    const float *vl_norm2_w;

    const float *vl_d_mlp1_b;
    const float *vl_d_mlp1_w;
    const float *vl_d_mlp2_b;
    const float *vl_d_mlp2_w;
    const float *vl_d_norm_b;
    const float *vl_d_norm_w;

    const float *vl_merge_mlp1_b;
    const float *vl_merge_mlp1_w;
    const float *vl_merge_mlp2_b;
    const float *vl_merge_mlp2_w;
    const float *vl_merge_norm_b;
    const float *vl_merge_norm_w;
} QwenWeight;

typedef struct {
    // ---- Hidden / intermediate buffers ----
    float *x;            // current hidden state [hidden_size]
    float *t;            // normalized hidden before attention [hidden_size]

    // ---- Attention projections ----
    float *q;            // query [num_attention_heads * head_dim]
    float *k;            // key   [num_key_value_heads * head_dim]
    float *v;            // value [num_key_value_heads * head_dim]

    float *att;          // attention scores (temporary buffer) [num_attention_heads * max_position_embeddings]
    float *qkv_out;      // attention output before projection [hidden_size]

    // ---- MLP intermediate ----
    float *gate;         // gate projection [intermediate_size]
    float *up;           // up projection [intermediate_size]

    // ---- Rotary embeddings ----
    float *cos_tensor;   // cached cosines for rotary embedding [max_position_embeddings * head_dim/2]
    float *sin_tensor;   // cached sines for rotary embedding [max_position_embeddings * head_dim/2]

    // ---- Output ----
    float *logits;       // final logits [vocab_size]

    // ---- KV cache (for autoregressive decoding) ----
    float *key_cache;    // [num_hidden_layers, num_key_value_heads, max_position_embeddings, head_dim]
    float *value_cache;  // same shape

    // vision
    float *vision_x;
    float *vision_t;
    float *vision_q;
    float *vision_k;

    float *vision_cos_tensor;
    float *vision_sin_tensor;

    float *vision_pe_cos;
    float *vision_pe_sin;
    
    float *vision_mlp_out;

    float *vision_deep_stack;
    
    float *vision_attn_scores;

    int vision_embed_tokens;
    int cur_img_token_id;
} QwenRunState;

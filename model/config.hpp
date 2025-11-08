#pragma once

#include <stdio.h>

// Helper to check for allocation errors
#define CHECK_ALLOC(ptr, size) if (!ptr) { \
    fprintf(stderr, "Error: failed to allocate memory for %ld bytes\n", (long)(size)); \
    return; \
}

typedef struct {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    int rope_theta;
    float rms_norm_eps;
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
} QwenConfig;

typedef struct {
    // Language Model Weights
    float* token_embedding_table; // [vocab_size, hidden_size]
    float* rms_out_w;         // [hidden_size]

    // Language Model Layer Weights (continuous blocks: [num_hidden_layers, ...])
    float* rms_ffn_w;          // [L, H]
    float* w_mlp_down;            // [L, H, I]
    float* w_mlp_gate;            // [L, I, H]
    float* w_mlp_up;              // [L, I, H]
    float* rms_attn_w; // [L, H]
    float* w_attn_k_norm;         // [L, KVA_Dim]
    float* w_attn_k;         // [L, KVA_Dim, H]
    float* w_attn_o;         // [L, H, H]
    float* w_attn_q_norm;         // [L, Q_Dim]
    float* w_attn_q;         // [L, Q_Dim, H]
    float* w_attn_v;         // [L, KVA_Dim, H]

    // Vision Model Weights (General)
    float* visual_attn_qkv_bias;        // [QKV_Dim]
    float* visual_attn_qkv_weight;      // [QKV_Dim, VH]
    float* visual_attn_proj_bias;       // [VH]
    float* visual_attn_proj_weight;     // [VH, VH]
    float* visual_class_embedding;      // [VH]
    float* visual_conv1_weight;         // [VH, 3, VP, VP]
    float* visual_ln_post_bias;         // [VH]
    float* visual_ln_post_weight;       // [VH]
    float* visual_ln_pre_bias;          // [VH]
    float* visual_ln_pre_weight;        // [VH]
    float* visual_patch_embed_proj_bias; // [VH]
    float* visual_patch_embed_proj_weight; // [VH, 3, VP, VP]
    float* visual_positional_embedding; // [257, VH]

    // Vision ResBlocks Weights (continuous blocks: [vision_depth, ...])
    float* visual_resblocks_attn_in_proj_bias;  // [VD, QKV_Dim]
    float* visual_resblocks_attn_in_proj_weight; // [VD, QKV_Dim, VH]
    float* visual_resblocks_attn_out_proj_bias; // [VD, VH]
    float* visual_resblocks_attn_out_proj_weight; // [VD, VH, VH]
    float* visual_resblocks_ln_1_bias;          // [VD, VH]
    float* visual_resblocks_ln_1_weight;        // [VD, VH]
    float* visual_resblocks_ln_2_bias;          // [VD, VH]
    float* visual_resblocks_ln_2_weight;        // [VD, VH]
    float* visual_resblocks_mlp_c_fc_bias;      // [VD, VI]
    float* visual_resblocks_mlp_c_fc_weight;    // [VD, VI, VH]
    float* visual_resblocks_mlp_c_proj_bias;    // [VD, VH]
    float* visual_resblocks_mlp_c_proj_weight;  // [VD, VH, VI]

    // Vision Deepstack Merger Weights (continuous blocks: [3, ...])
    float* visual_deepstack_merger_list_linear_fc1_bias;  // [3, VI]
    float* visual_deepstack_merger_list_linear_fc1_weight; // [3, VI, VI]
    float* visual_deepstack_merger_list_linear_fc2_bias;  // [3, OH]
    float* visual_deepstack_merger_list_linear_fc2_weight; // [3, OH, VI]
    float* visual_deepstack_merger_list_norm_bias;        // [3, VI]
    float* visual_deepstack_merger_list_rms_out_w;      // [3, VI]

    // Final Merger Weights
    float* visual_merger_linear_fc1_bias;  // [VI]
    float* visual_merger_linear_fc1_weight; // [VI, VI]
    float* visual_merger_linear_fc2_bias;  // [OH]
    float* visual_merger_linear_fc2_weight; // [OH, VI]
    float* visual_merger_norm_bias;        // [VI]
    float* visual_merger_rms_out_w;      // [VI]
} QwenWeight;

typedef struct {
    // ---- Hidden / intermediate buffers ----
    float* x;            // current hidden state [hidden_size]
    float* t;            // normalized hidden before attention [hidden_size]

    // ---- Attention projections ----
    float* q;            // query [num_attention_heads * head_dim]
    float* k;            // key   [num_key_value_heads * head_dim]
    float* v;            // value [num_key_value_heads * head_dim]

    float* att;          // attention scores (temporary buffer) [num_attention_heads * max_position_embeddings]
    float* qkv_out;      // attention output before projection [hidden_size]
    float* attn_out;     // after output projection [hidden_size]

    // ---- MLP intermediate ----
    float* gate;         // gate projection [intermediate_size]
    float* up;           // up projection [intermediate_size]
    float* gate_up;      // after SwiGLU [intermediate_size]
    float* down;         // down projection [hidden_size]

    // ---- Rotary embeddings ----
    float* cos_tensor;   // cached cosines for rotary embedding [max_position_embeddings * head_dim/2]
    float* sin_tensor;   // cached sines for rotary embedding [max_position_embeddings * head_dim/2]

    // ---- Output ----
    float* logits;       // final logits [vocab_size]

    // ---- KV cache (for autoregressive decoding) ----
    float* key_cache;    // [num_hidden_layers, num_key_value_heads, max_position_embeddings, head_dim]
    float* value_cache;  // same shape
    
    bool vision_embed_true;
} QwenRunState;

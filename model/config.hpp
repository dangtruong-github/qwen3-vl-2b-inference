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
    float* embed_tokens_weight; // [vocab_size, hidden_size]
    float* norm_weight;         // [hidden_size]
    float* lm_head_weight;      // [vocab_size, hidden_size]

    // Language Model Layer Weights (continuous blocks: [num_hidden_layers, ...])
    float* input_layernorm_weight;          // [L, H]
    float* mlp_down_proj_weight;            // [L, H, I]
    float* mlp_gate_proj_weight;            // [L, I, H]
    float* mlp_up_proj_weight;              // [L, I, H]
    float* post_attention_layernorm_weight; // [L, H]
    float* self_attn_k_norm_weight;         // [L, KVA_Dim]
    float* self_attn_k_proj_weight;         // [L, KVA_Dim, H]
    float* self_attn_o_proj_weight;         // [L, H, H]
    float* self_attn_q_norm_weight;         // [L, Q_Dim]
    float* self_attn_q_proj_weight;         // [L, Q_Dim, H]
    float* self_attn_v_proj_weight;         // [L, KVA_Dim, H]

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
    float* visual_deepstack_merger_list_norm_weight;      // [3, VI]

    // Final Merger Weights
    float* visual_merger_linear_fc1_bias;  // [VI]
    float* visual_merger_linear_fc1_weight; // [VI, VI]
    float* visual_merger_linear_fc2_bias;  // [OH]
    float* visual_merger_linear_fc2_weight; // [OH, VI]
    float* visual_merger_norm_bias;        // [VI]
    float* visual_merger_norm_weight;      // [VI]
} QwenWeight;

typedef struct {
    int b;
} QwenRunState;
#include "../include/forward.hpp"

void forward_img(QwenConfig *config, QwenRunState *state, QwenWeight *weight, float *img_data, int img_h, int img_w, int grid_h, int grid_w) {
    if (img_data == nullptr) {
        return;
    }

    printf("img_h=%d, img_w=%d, grid_h=%d, grid_w=%d\n", img_h, img_w, grid_h, grid_w);
    printf("vision_num_channels=%d, vision_temporal_patch_size=%d, vision_patch_size=%d\n", config->vision_num_channels, config->vision_temporal_patch_size, config->vision_patch_size);
    fflush(stdout);

    long VC = config->vision_num_channels;
    long VTP = config->vision_temporal_patch_size;
    long VP = config->vision_patch_size;
    long VH = config->vision_hidden_size;
    long VSP = config->vision_spatial_merge_size;
    long total_tokens = grid_h * grid_w;
    long VNH = config->vision_num_heads;
    long VHD = VH / VNH;  // vision_head_dim
    long VI = config->vision_intermediate_size;
    long OH = config->out_hidden_size;
    float vision_scale = config->vision_scale;
    long d_tokens = total_tokens / (VSP * VSP);

    
    conv_3d(weight->vl_patch_emb_w, weight->vl_patch_emb_b, img_data, state->vision_x, img_h, VC, VTP, VP, VH);

    free(img_data);

    long num_grid_per_side = sqrt(config->max_vision_embeddings);

    printf("num_grid_per_side=%d\n", num_grid_per_side);

    vision_pos_embed(weight->vl_pos_emb_w, state->vision_t, grid_h, grid_w, num_grid_per_side, VSP, VH);

    add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

    vision_rot_pos_emb(
        state->vision_pe_cos, state->vision_pe_sin, state->vision_cos_tensor, state->vision_sin_tensor,
        grid_h, grid_w, config->vision_spatial_merge_size, VHD
    );

    printf("Finish preprocessing forward_img\n");
    fflush(stdout);

    for (size_t l = 0; l < config->vision_depth; l++) {
        for (int i = 0; i < total_tokens; i++) {
            layer_norm(
                state->vision_x + 1ll * i * VH, weight->vl_norm1_w,
                weight->vl_norm1_b, state->vision_t + 1ll * i * VH,
                config->rms_norm_eps, VH, 1ll * l
            );
        }

        const size_t qkv_layer_off_w = 1ll * 3 * VH * VH;
        const size_t qkv_each_off_w = 1ll * VH * VH;
        const float *w_q = weight->vl_attn_qkv_w + 1ll * l * qkv_layer_off_w;
        const float *w_k = weight->vl_attn_qkv_w + 1ll * l * qkv_layer_off_w + 1ll * qkv_each_off_w;
        const float *w_v = weight->vl_attn_qkv_w + 1ll * l * qkv_layer_off_w + 1ll * 2 * qkv_each_off_w;
        const float *b_q = weight->vl_attn_qkv_b + 1ll * l * 3 * VH;
        const float *b_k = weight->vl_attn_qkv_b + 1ll * l * 3 * VH + 1ll * VH;
        const float *b_v = weight->vl_attn_qkv_b + 1ll * l * 3 * VH + 1ll * 2 * VH;
        
        // use vl_v as temporary buffer for vision_q
        linear(
            state->vision_t, w_q, b_q, state->vision_mlp_out,
            total_tokens, VH, VH, true
        );
        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );
        tensor_transpose(
            state->vision_mlp_out, state->vision_q, total_tokens, VNH, VHD
        );

        // use vl_v as temporary buffer for vision_k
        linear(
            state->vision_t, w_k, b_k, state->vision_mlp_out,
            total_tokens, VH, VH, true
        );
        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );
        tensor_transpose(
            state->vision_mlp_out, state->vision_k, total_tokens, VNH, VHD
        );

        // swap vision_t and vl_v
        linear(
            state->vision_t, w_v, b_v, state->vision_mlp_out,
            total_tokens, VH, VH, true
        );
        tensor_transpose(
            state->vision_mlp_out, state->vision_t, total_tokens, VNH, VHD
        );
        
        vision_att(
            state->vision_q, state->vision_k, state->vision_t, 
            state->vision_attn_scores, state->vision_mlp_out, 
            VNH, total_tokens, VHD, vision_scale
        );
        
        // swap back vision_t and vl_v
        tensor_transpose(
            state->vision_mlp_out, state->vision_t, VNH, total_tokens, VHD
        );

        const float *w_attn_proj_ptr = weight->vl_attn_proj_w + 1ll * l * VH * VH;
        const float *b_attn_proj_ptr = weight->vl_attn_proj_b + 1ll * l * VH;

        // use vision_q as temporary buffer here
        linear(
            state->vision_t, w_attn_proj_ptr, b_attn_proj_ptr,
            state->vision_q, total_tokens, VH, VH, true
        );

        add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);
        
        for (int i = 0; i < total_tokens; i++) {
            layer_norm(
                state->vision_x + 1ll * i * VH, weight->vl_norm2_w,
                weight->vl_norm2_b, state->vision_t + 1ll * i * VH,
                config->rms_norm_eps, VH, 1ll * l
            );
        }
        
        const float *w_mlp1_ptr = weight->vl_mlp1_w + 1ll * l * VI * VH;
        const float *b_mlp1_ptr = weight->vl_mlp1_b + 1ll * l * VI;
        linear(
            state->vision_t, w_mlp1_ptr, b_mlp1_ptr, state->vision_mlp_out,
            total_tokens, VI, VH, true
        );

        gelu_tanh(state->vision_mlp_out, 1ll * total_tokens * VI);
        
        const float *w_mlp2_ptr = weight->vl_mlp2_w + 1ll * l * VH * VI;
        const float *b_mlp2_ptr = weight->vl_mlp2_b + 1ll * l * VH;
        linear(
            state->vision_mlp_out, w_mlp2_ptr, b_mlp2_ptr, state->vision_t,
            total_tokens, VH, VI, true
        );

        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

        if (config->deep_layer[l] > 0) {
            int d_stride = config->deep_layer[l] - 1;
            for (int i = 0; i < d_tokens; i++) {
                layer_norm(
                    state->vision_x + 1ll * i * VI,
                    weight->vl_d_norm_w,
                    weight->vl_d_norm_b,
                    state->vision_t + 1ll * i * VI,
                    config->rms_norm_eps, 
                    VI, 
                    1ll * d_stride
                );
            }

            const float *w_mlp1_d_ptr = weight->vl_d_mlp1_w + 1ll * d_stride * VI * VI;
            const float *b_mlp1_d_ptr = weight->vl_d_mlp1_b + 1ll * d_stride * VI;
            linear(
                state->vision_t, w_mlp1_d_ptr, b_mlp1_d_ptr,
                state->vision_mlp_out, d_tokens, VI, VI, true
            );
        
            gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

            const float *w_mlp2_d_ptr = weight->vl_d_mlp2_w + 1ll * d_stride * OH * VI;
            const float *b_mlp2_d_ptr = weight->vl_d_mlp2_b + 1ll * d_stride * OH;
            linear(
                state->vision_mlp_out, w_mlp2_d_ptr, b_mlp2_d_ptr,
                state->vision_deep_stack + 1ll * d_stride * d_tokens * OH,
                d_tokens, OH, VI, true
            );
        }

        printf("Finish layer %ld\n", l);
        fflush(stdout);
    }

    for (int i = 0; i < total_tokens; i++) {
        layer_norm(
            state->vision_x + 1ll * i * VH, weight->vl_merge_norm_w,
            weight->vl_merge_norm_b, state->vision_t + 1ll * i * VH,
            config->rms_norm_eps, VH, 0
        );
    }

    linear(
        state->vision_t, weight->vl_merge_mlp1_w, weight->vl_merge_mlp1_b,
        state->vision_mlp_out, d_tokens, VI, VI, true
    );

    gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

    linear(
        state->vision_mlp_out, weight->vl_merge_mlp2_w, weight->vl_merge_mlp2_b,
        state->vision_x, d_tokens, OH, VI, true
    );
    
    state->vision_embed_tokens = d_tokens;
    state->cur_img_token_id = 0;
}

float *forward_text(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id, int pos) {
    long hidden_size = config->hidden_size;

    long head_dim = hidden_size / config->num_attention_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;

    // Cache dimensions: [num_layers][seq_len][kv_dim]
    long long cache_layer_size = 1ll * seq_len * kv_dim; // Size of one layer's cache
    long long loff_one = cache_layer_size;               // Offset between layers
    long long loff_pos = kv_dim;                         // Offset between positions
    
    // Embed layer
    if (token_id != config->image_token_id && token_id != config->video_token_id) {
        embedding_lookup(
            weight->token_embedding_table, token_id, state->x,
            config->vocab_size, hidden_size
        );
    } else {
        const float *src = state->vision_x + 1ll * hidden_size * state->cur_img_token_id;
        memcpy(state->x, src, 1ll * hidden_size * sizeof(float));
        state->cur_img_token_id += 1;
    }

    for (int l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            state->x, weight->rms_ffn_w, state->t, config->rms_norm_eps, hidden_size, 1ll * l
        );

        // QKV Projections
        long long loff = 1ll * l * loff_one;  // kv cache layer offset (layer_cache_start)

        const float *w_q = weight->w_attn_q + 1ll * l * hidden_size * hidden_size;
        const float *w_k = weight->w_attn_k + 1ll * l * hidden_size * kv_dim;
        const float *w_v = weight->w_attn_v + 1ll * l * hidden_size * kv_dim;
        
        linear(
            state->t, w_q, nullptr, state->q, 1, hidden_size, hidden_size, true
        );
        linear(
            state->t, w_k, nullptr, state->k, 1, kv_dim, hidden_size, true
        );
        linear(
            state->t, w_v, nullptr, state->v, 1, kv_dim, hidden_size, true
        );

        // QK RMSNorm
        for (int h = 0; h < config->num_attention_heads; h++) {
            rms_norm(
                state->q + h * head_dim,
                weight->w_attn_q_norm,
                state->q + h * head_dim,
                config->rms_norm_eps, head_dim, 1ll * l
            );
            if (h < config->num_key_value_heads) { 
                rms_norm(
                    state->k + h * head_dim,
                    weight->w_attn_k_norm,
                    state->k + h * head_dim,
                    config->rms_norm_eps, head_dim, 1ll * l
                );
            }
        }

        // Apply Rotary Position Embeddings
        apply_rotary(
            state->q, state->cos_tensor, state->sin_tensor,
            config->num_attention_heads, head_dim, pos
        );
        apply_rotary(
            state->k, state->cos_tensor, state->sin_tensor,
            config->num_key_value_heads, head_dim, pos
        );

        // Store k, v in cache - FIXED VERSION
        long long loff_cache = loff + pos * kv_dim;  // Position offset within layer
        memcpy(state->key_cache + loff_cache, state->k, kv_dim * sizeof(float));
        memcpy(state->value_cache + loff_cache, state->v, kv_dim * sizeof(float));

        // Multi-head attention
        int kv_mul = config->num_attention_heads / config->num_key_value_heads;  // integer multiplier for GQA

        // Compute attention scores
        attn_scores_all_heads(
            state->key_cache, state->q, state->att,
            1ll * l, // layer_offset
            config->num_attention_heads, kv_mul, head_dim,
            kv_dim, seq_len, pos
        );

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            state->value_cache, state->q, state->att, state->qkv_out, 
            loff, // layer_cache_start
            config->num_attention_heads, kv_mul, head_dim, 
            kv_dim, seq_len, pos
        );

        // Output projection: using state->t for attn_out
        const float *w_out_proj = weight->w_attn_o + 1ll * l * hidden_size * hidden_size;
        linear(
            state->qkv_out, w_out_proj, nullptr, state->t, 1,
            hidden_size, hidden_size, true
        );

        // Residual connection 1
        add_vector(
            state->x, state->t, hidden_size
        );

        // Post-attention RMSNorm
        rms_norm(
            state->x, weight->rms_attn_w, state->t,
            config->rms_norm_eps, hidden_size, 1ll * l
        );

        // MLP: Gate and Up projections
        const float *w_gate = weight->w_mlp_gate + 1ll * l * config->intermediate_size * hidden_size;
        const float *w_up = weight->w_mlp_up + 1ll * l * config->intermediate_size * hidden_size;
        linear(
            state->t, w_gate, nullptr, state->gate, 1,
            config->intermediate_size, hidden_size, true
        );
        linear(
            state->t, w_up, nullptr, state->up, 1,
            config->intermediate_size, hidden_size, true
        );
        
        // SwiGLU activation
        swiglu(
            state->gate, state->up, config->intermediate_size
        );

        // MLP: Down projection: using state->t for down
        const float *w_down = weight->w_mlp_down + 1ll * l * config->intermediate_size * hidden_size;
        linear(
            state->gate, w_down, nullptr, state->t, 1,
            hidden_size, config->intermediate_size, true
        );

        // Residual connection 2
        add_vector(
            state->x, state->t, hidden_size
        );
    }

    // Final RMSNorm
    rms_norm(
        state->x, weight->rms_out_w, state->x, config->rms_norm_eps,
        hidden_size, 0ll
    );

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, hidden_size
    );

    return state->logits;
}

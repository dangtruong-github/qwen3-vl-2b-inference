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

    conv_3d(
        weight->vl_patch_emb_w, weight->vl_patch_emb_b, img_data,
        (float *)state->vision_x->buf, img_h, VC, VTP, VP, VH
    );

    free(img_data);

    long num_grid_per_side = sqrt(config->max_vision_embeddings);

    printf("num_grid_per_side=%d\n", num_grid_per_side);

    vision_pos_embed(
        weight->vl_pos_emb_w, (float *)state->vision_t->buf, grid_h, grid_w,
        num_grid_per_side, VSP, VH
    );

    add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

    vision_rot_pos_emb(
        (float *)state->vision_pe_cos->buf,
        (float *)state->vision_pe_sin->buf,
        (const float *)state->vision_cos_tensor->buf,
        (const float *)state->vision_sin_tensor->buf,
        grid_h, grid_w, config->vision_spatial_merge_size, VHD
    );

    printf("Finish preprocessing forward_img\n");
    fflush(stdout);

    for (size_t l = 0; l < config->vision_depth; l++) {
        for (int i = 0; i < total_tokens; i++) {
            layer_norm(
                (const float *)state->vision_x->buf + 1ll * i * VH,
                weight->vl_norm1_w,
                weight->vl_norm1_b,
                (float *)state->vision_t->buf + 1ll * i * VH,
                config->rms_norm_eps, VH, 1ll * l
            );
        }

        const size_t qkv_layer_off_w = 1ll * 3 * VH * VH;
        const size_t qkv_each_off_w = 1ll * VH * VH;
        const float *w_q = (const float *)weight->vl_attn_qkv_w->buf + 1ll * l * qkv_layer_off_w;
        const float *w_k = (const float *)weight->vl_attn_qkv_w->buf + 1ll * l * qkv_layer_off_w + 1ll * qkv_each_off_w;
        const float *w_v = (const float *)weight->vl_attn_qkv_w->buf + 1ll * l * qkv_layer_off_w + 1ll * 2 * qkv_each_off_w;
        const float *b_q = (const float *)weight->vl_attn_qkv_b->buf + 1ll * l * 3 * VH;
        const float *b_k = (const float *)weight->vl_attn_qkv_b->buf + 1ll * l * 3 * VH + 1ll * VH;
        const float *b_v = (const float *)weight->vl_attn_qkv_b->buf + 1ll * l * 3 * VH + 1ll * 2 * VH;
        
        // use vl_v as temporary buffer for vision_q
        linear(
            (const float *)state->vision_t->buf, w_q, b_q,
            (float *)state->vision_mlp_out->buf,
            total_tokens, VH, VH, true
        );

        vision_apply_rotary_inplace(
            (const float *)state->vision_pe_cos->buf,
            (const float *)state->vision_pe_sin->buf,
            (float *)state->vision_mlp_out->buf, total_tokens, VNH, VHD
        );
        tensor_transpose(
            (const float *)state->vision_mlp_out->buf,
            (float *)state->vision_q->buf, total_tokens, VNH, VHD
        );

        // use vl_v as temporary buffer for vision_k
        linear(
            (const float *)state->vision_t->buf, w_k, b_k,
            (float *)state->vision_mlp_out->buf,
            total_tokens, VH, VH, true
        );
        vision_apply_rotary_inplace(
            (const float *)state->vision_pe_cos->buf,
            (const float *)state->vision_pe_sin->buf,
            (float *)state->vision_mlp_out->buf, total_tokens, VNH, VHD
        );
        tensor_transpose(
            (const float *)state->vision_mlp_out->buf,
            (float *)state->vision_k->buf, total_tokens, VNH, VHD
        );

        // swap vision_t and vl_v
        linear(
            (const float *)state->vision_t->buf, w_v, b_v,
            (float *)state->vision_mlp_out->buf,
            total_tokens, VH, VH, true
        );
        tensor_transpose(
            (const float *)state->vision_mlp_out->buf,
            (float *)state->vision_t->buf, total_tokens, VNH, VHD
        );
        
        vision_att(
            (const float *)state->vision_q->buf,
            (const float *)state->vision_k->buf,
            (const float *)state->vision_t->buf, 
            (float *)state->vision_attn_scores->buf,
            (float *)state->vision_mlp_out->buf, 
            VNH, total_tokens, VHD, vision_scale
        );
        
        // swap back vision_t and vl_v
        tensor_transpose(
            (float *)state->vision_mlp_out->buf,
            (float *)state->vision_t->buf, VNH, total_tokens, VHD
        );

        const float *w_attn_proj_ptr = (const float *)weight->vl_attn_proj_w->buf + 1ll * l * VH * VH;
        const float *b_attn_proj_ptr = (const float *)weight->vl_attn_proj_b->buf + 1ll * l * VH;

        // use vision_q as temporary buffer here
        linear(
            (const float *)state->vision_t->buf, w_attn_proj_ptr,
            b_attn_proj_ptr,
            (float *)state->vision_q->buf, total_tokens, VH, VH, true
        );

        add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);
        
        for (int i = 0; i < total_tokens; i++) {
            layer_norm(
                (const float *)state->vision_x->buf + 1ll * i * VH,
                weight->vl_norm2_w,
                weight->vl_norm2_b,
                (float *)state->vision_t->buf + 1ll * i * VH,
                config->rms_norm_eps, VH, 1ll * l
            );
        }
        
        const float *w_mlp1_ptr = (const float *)weight->vl_mlp1_w->buf + 1ll * l * VI * VH;
        const float *b_mlp1_ptr = (const float *)weight->vl_mlp1_b->buf + 1ll * l * VI;
        linear(
            (const float *)state->vision_t->buf, w_mlp1_ptr, b_mlp1_ptr,
            (float *)state->vision_mlp_out->buf,
            total_tokens, VI, VH, true
        );

        gelu_tanh((float *)state->vision_mlp_out->buf, 1ll * total_tokens * VI);
        
        const float *w_mlp2_ptr = (const float *)weight->vl_mlp2_w->buf + 1ll * l * VH * VI;
        const float *b_mlp2_ptr = (const float *)weight->vl_mlp2_b->buf + 1ll * l * VH;
        linear(
            (const float *)state->vision_mlp_out->buf, w_mlp2_ptr, b_mlp2_ptr,
            (float *)state->vision_t->buf,
            total_tokens, VH, VI, true
        );

        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

        if (config->deep_layer[l] > 0) {
            int d_stride = config->deep_layer[l] - 1;
            for (int i = 0; i < d_tokens; i++) {
                layer_norm(
                    (float *)state->vision_x->buf + 1ll * i * VI,
                    weight->vl_d_norm_w,
                    weight->vl_d_norm_b,
                    (float *)state->vision_t->buf + 1ll * i * VI,
                    config->rms_norm_eps, 
                    VI, 
                    1ll * d_stride
                );
            }

            const float *w_mlp1_d_ptr = (const float *)weight->vl_d_mlp1_w->buf + 1ll * d_stride * VI * VI;
            const float *b_mlp1_d_ptr = (const float *)weight->vl_d_mlp1_b->buf + 1ll * d_stride * VI;
            linear(
                (const float *)state->vision_t->buf, w_mlp1_d_ptr, b_mlp1_d_ptr,
                (float *)state->vision_mlp_out->buf, d_tokens, VI, VI, true
            );
        
            gelu_tanh((float *)state->vision_mlp_out->buf, 1ll * d_tokens * VI);

            const float *w_mlp2_d_ptr = (const float *)weight->vl_d_mlp2_w->buf + 1ll * d_stride * OH * VI;
            const float *b_mlp2_d_ptr = (const float *)weight->vl_d_mlp2_b->buf + 1ll * d_stride * OH;
            linear(
                (const float *)state->vision_mlp_out->buf, w_mlp2_d_ptr,
                b_mlp2_d_ptr,
                (float *)state->vision_deep_stack->buf + 1ll * d_stride * d_tokens * OH,
                d_tokens, OH, VI, true
            );
        }

        printf("Finish layer %ld\n", l);
        fflush(stdout);
    }

    for (int i = 0; i < total_tokens; i++) {
        layer_norm(
            (const float *)state->vision_x->buf + 1ll * i * VH,
            weight->vl_merge_norm_w,
            weight->vl_merge_norm_b,
            (float *)state->vision_t->buf + 1ll * i * VH,
            config->rms_norm_eps, VH, 0
        );
    }

    linear(
        (const float *)state->vision_t->buf,
        (const float *)weight->vl_merge_mlp1_w->buf,
        (const float *)weight->vl_merge_mlp1_b->buf,
        (float *)state->vision_mlp_out->buf, d_tokens, VI, VI, true
    );

    gelu_tanh((float *)state->vision_mlp_out->buf, 1ll * d_tokens * VI);

    linear(
        (const float *)state->vision_mlp_out->buf,
        (const float *)weight->vl_merge_mlp2_w->buf,
        (const float *)weight->vl_merge_mlp2_b->buf,
        (float *)state->vision_x->buf, d_tokens, OH, VI, true
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
        const float *src = (const float *)state->vision_x->buf + 1ll * hidden_size * state->cur_img_token_id;
        memcpy(state->x->buf, src, 1ll * hidden_size * sizeof(float));
        state->cur_img_token_id += 1;
    }

    for (int l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            (const float *)state->x->buf, weight->rms_ffn_w,
            (float *)state->t->buf,
            config->rms_norm_eps, hidden_size, 1ll * l
        );

        // QKV Projections
        long long loff = 1ll * l * loff_one;  // kv cache layer offset (layer_cache_start)

        const float *w_q = (const float *)weight->w_attn_q->buf + 1ll * l * hidden_size * hidden_size;
        const float *w_k = (const float *)weight->w_attn_k->buf + 1ll * l * hidden_size * kv_dim;
        const float *w_v = (const float *)weight->w_attn_v->buf + 1ll * l * hidden_size * kv_dim;

        // Store k, v in cache - FIXED VERSION
        long long loff_cache = loff + pos * kv_dim;  // Position offset within layer
        float *k_cur_cache_ptr = (float *)state->key_cache->buf + loff_cache;
        float *v_cur_cache_ptr = (float *)state->value_cache->buf + loff_cache;
        
        linear(
            (const float *)state->t->buf, w_q, nullptr, (float *)state->q->buf,
            1, hidden_size, hidden_size, true
        );
        linear(
            (const float *)state->t->buf, w_k, nullptr, k_cur_cache_ptr,
            1, kv_dim, hidden_size, true
        );
        linear(
            (const float *)state->t->buf, w_v, nullptr, v_cur_cache_ptr,
            1, kv_dim, hidden_size, true
        );

        // QK RMSNorm
        for (int h = 0; h < config->num_attention_heads; h++) {
            rms_norm(
                (const float *)state->q->buf + h * head_dim,
                weight->w_attn_q_norm,
                (float *)state->q->buf + h * head_dim,
                config->rms_norm_eps, head_dim, 1ll * l
            );
            if (h < config->num_key_value_heads) { 
                rms_norm(
                    k_cur_cache_ptr + h * head_dim,
                    weight->w_attn_k_norm,
                    k_cur_cache_ptr + h * head_dim,
                    config->rms_norm_eps, head_dim, 1ll * l
                );
            }
        }

        // Apply Rotary Position Embeddings
        apply_rotary(
            (float *)state->q->buf, state->cos_tensor, state->sin_tensor,
            config->num_attention_heads, head_dim, pos
        );
        apply_rotary(
            k_cur_cache_ptr, state->cos_tensor, state->sin_tensor,
            config->num_key_value_heads, head_dim, pos
        );

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
            state->value_cache, state->att, state->qkv_out, 
            loff, // layer_cache_start
            config->num_attention_heads, kv_mul, head_dim, 
            kv_dim, seq_len, pos
        );

        // Output projection: using state->t for attn_out
        const float *w_out_proj = (const float *)weight->w_attn_o->buf + 1ll * l * hidden_size * hidden_size;
        linear(
            (const float *)state->qkv_out->buf, w_out_proj, nullptr,
            (float *)state->t->buf, 1, hidden_size, hidden_size, true
        );

        // Residual connection 1
        add_vector(
            state->x, state->t, hidden_size
        );

        // Post-attention RMSNorm
        rms_norm(
            (const float *)state->x->buf, weight->rms_attn_w,
            (float *)state->t->buf,
            config->rms_norm_eps, hidden_size, 1ll * l
        );

        // MLP: Gate and Up projections
        const float *w_gate = (const float *)weight->w_mlp_gate->buf + 1ll * l * config->intermediate_size * hidden_size;
        const float *w_up = (const float *)weight->w_mlp_up->buf + 1ll * l * config->intermediate_size * hidden_size;
        linear(
            (const float *)state->t->buf, w_gate, nullptr,
            (float *)state->gate->buf, 1,
            config->intermediate_size, hidden_size, true
        );
        linear(
            (const float *)state->t->buf, w_up, nullptr,
            (float *)state->up->buf, 1,
            config->intermediate_size, hidden_size, true
        );
        
        // SwiGLU activation
        swiglu(
            state->gate, state->up, config->intermediate_size
        );

        // MLP: Down projection: using state->t for down
        const float *w_down = (const float *)weight->w_mlp_down->buf + 1ll * l * config->intermediate_size * hidden_size;
        linear(
            (const float *)state->gate->buf, w_down, nullptr,
            (float *)state->t->buf, 1,
            hidden_size, config->intermediate_size, true
        );

        // Residual connection 2
        add_vector(
            state->x, state->t, hidden_size
        );
    }

    // Final RMSNorm
    rms_norm(
        (const float *)state->x->buf, weight->rms_out_w,
        (float *)state->x->buf, config->rms_norm_eps,
        hidden_size, 0ll
    );

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, hidden_size
    );

    // exit(1);
    return (float *)state->logits->buf;
}

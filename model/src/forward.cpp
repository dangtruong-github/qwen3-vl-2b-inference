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
        (float *)state->vision_x->ptr(), img_h, VC, VTP, VP, VH
    );

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif

    free(img_data);

    long VNP = config->max_vision_embeddings;
    long num_grid_per_side = sqrt(VNP);

    printf("num_grid_per_side=%d\n", num_grid_per_side);

    vision_pos_embed(
        weight->vl_pos_emb_w, (float *)state->vision_t->ptr(), grid_h, grid_w,
        num_grid_per_side, VSP, VH
    );

    #ifdef PRINT_LOGITS
        state->vision_t->printDebug("vision_t");
    #endif

    add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif

    vision_rot_pos_emb(
        (float *)state->vision_pe_cos->ptr(),
        (float *)state->vision_pe_sin->ptr(),
        (const float *)state->vision_cos_tensor->ptr(),
        (const float *)state->vision_sin_tensor->ptr(),
        grid_h, grid_w, config->vision_spatial_merge_size, VHD
    );

    #ifdef PRINT_LOGITS
        state->vision_pe_cos->printDebug("vision_pe_cos");
        state->vision_pe_sin->printDebug("vision_pe_sin");
    #endif

    printf("Finish preprocessing forward_img\n");
    fflush(stdout);

    for (size_t l = 0; l < config->vision_depth; l++) {
        layer_norm(
            state->vision_x, weight->vl_norm1_w,
            weight->vl_norm1_b, state->vision_t,
            config->rms_norm_eps, total_tokens, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        const float *w_q = (const float *)weight->vl_attn_qkv_w->ptr({l});
        const float *w_k = (const float *)weight->vl_attn_qkv_w->ptr({l, 1});
        const float *w_v = (const float *)weight->vl_attn_qkv_w->ptr({l, 2});
        const float *b_q = (const float *)weight->vl_attn_qkv_b->ptr({l});
        const float *b_k = (const float *)weight->vl_attn_qkv_b->ptr({l, 1});
        const float *b_v = (const float *)weight->vl_attn_qkv_b->ptr({l, 2});
        
        // use vl_v as temporary buffer for vision_q
        linear(
            (const float *)state->vision_t->ptr(), w_q, b_q,
            (float *)state->vision_mlp_out->ptr(),
            total_tokens, VH, VH, true
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        vision_apply_rotary_inplace(
            (const float *)state->vision_pe_cos->ptr(),
            (const float *)state->vision_pe_sin->ptr(),
            (float *)state->vision_mlp_out->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_q->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_q->printDebug("vision_q");
        #endif

        // use vl_v as temporary buffer for vision_k
        linear(
            (const float *)state->vision_t->ptr(), w_k, b_k,
            (float *)state->vision_mlp_out->ptr(),
            total_tokens, VH, VH, true
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        vision_apply_rotary_inplace(
            (const float *)state->vision_pe_cos->ptr(),
            (const float *)state->vision_pe_sin->ptr(),
            (float *)state->vision_mlp_out->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_k->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_k->printDebug("vision_k");
        #endif

        // swap vision_t and vl_v
        linear(
            (const float *)state->vision_t->ptr(), w_v, b_v,
            (float *)state->vision_mlp_out->ptr(),
            total_tokens, VH, VH, true
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_t->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif
        
        vision_att(
            (const float *)state->vision_q->ptr(),
            (const float *)state->vision_k->ptr(),
            (const float *)state->vision_t->ptr(), 
            (float *)state->vision_attn_scores->ptr(),
            (float *)state->vision_mlp_out->ptr(), 
            VNH, total_tokens, VHD, vision_scale
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif
        
        // swap back vision_t and vl_v
        tensor_transpose(
            (float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_t->ptr(), VNH, total_tokens, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        const float *w_attn_proj_ptr = (const float *)weight->vl_attn_proj_w->ptr({l});
        const float *b_attn_proj_ptr = (const float *)weight->vl_attn_proj_b->ptr({l});

        // use vision_q as temporary buffer here
        linear(
            (const float *)state->vision_t->ptr(), w_attn_proj_ptr,
            b_attn_proj_ptr,
            (float *)state->vision_q->ptr(), total_tokens, VH, VH, true
        );

        #ifdef PRINT_LOGITS
            state->vision_q->printDebug("vision_q");
        #endif

        add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);

        #ifdef PRINT_LOGITS
            state->vision_x->printDebug("vision_x");
        #endif

        layer_norm(
            state->vision_x, weight->vl_norm2_w,
            weight->vl_norm2_b, state->vision_t,
            config->rms_norm_eps, total_tokens, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif
        
        const float *w_mlp1_ptr = (const float *)weight->vl_mlp1_w->ptr({l});;
        const float *b_mlp1_ptr = (const float *)weight->vl_mlp1_b->ptr({l});
        linear(
            (const float *)state->vision_t->ptr(), w_mlp1_ptr, b_mlp1_ptr,
            (float *)state->vision_mlp_out->ptr(),
            total_tokens, VI, VH, true
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        gelu_tanh(state->vision_mlp_out, 1ll * total_tokens * VI);

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif
        
        const float *w_mlp2_ptr = (const float *)weight->vl_mlp2_w->ptr({l});
        const float *b_mlp2_ptr = (const float *)weight->vl_mlp2_b->ptr({l});;
        linear(
            (const float *)state->vision_mlp_out->ptr(), w_mlp2_ptr, b_mlp2_ptr,
            (float *)state->vision_t->ptr(),
            total_tokens, VH, VI, true
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

        #ifdef PRINT_LOGITS
            state->vision_x->printDebug("vision_x");
        #endif

        if (config->deep_layer[l] > 0) {
            size_t d_stride = config->deep_layer[l] - 1;
            layer_norm(
                state->vision_x, weight->vl_d_norm_w,
                weight->vl_d_norm_b, state->vision_t,
                config->rms_norm_eps, d_tokens, 1ll * d_stride
            );

            #ifdef PRINT_LOGITS
                state->vision_t->printDebug("vision_t");
            #endif

            const float *w_mlp1_d_ptr = (const float *)weight->vl_d_mlp1_w->ptr({d_stride});
            const float *b_mlp1_d_ptr = (const float *)weight->vl_d_mlp1_b->ptr({d_stride});
            linear(
                (const float *)state->vision_t->ptr(), w_mlp1_d_ptr, b_mlp1_d_ptr,
                (float *)state->vision_mlp_out->ptr(), d_tokens, VI, VI, true
            );

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif
        
            gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif

            const float *w_mlp2_d_ptr = (const float *)weight->vl_d_mlp2_w->ptr({d_stride});
            const float *b_mlp2_d_ptr = (const float *)weight->vl_d_mlp2_b->ptr({d_stride});
            linear(
                (const float *)state->vision_mlp_out->ptr(), w_mlp2_d_ptr,
                b_mlp2_d_ptr,
                (float *)state->vision_deep_stack->ptr({d_stride}),
                d_tokens, OH, VI, true
            );

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif
        }

        printf("Finish layer %ld\n", l);
        fflush(stdout);
    }

    layer_norm(
        state->vision_x, weight->vl_merge_norm_w,
        weight->vl_merge_norm_b, state->vision_t,
        config->rms_norm_eps, total_tokens, 0
    );

    #ifdef PRINT_LOGITS
        state->vision_t->printDebug("vision_t");
    #endif

    linear(
        (const float *)state->vision_t->ptr(),
        (const float *)weight->vl_merge_mlp1_w->ptr(),
        (const float *)weight->vl_merge_mlp1_b->ptr(),
        (float *)state->vision_mlp_out->ptr(), d_tokens, VI, VI, true
    );

    #ifdef PRINT_LOGITS
        state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

    gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

    #ifdef PRINT_LOGITS
        state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

    linear(
        (const float *)state->vision_mlp_out->ptr(),
        (const float *)weight->vl_merge_mlp2_w->ptr(),
        (const float *)weight->vl_merge_mlp2_b->ptr(),
        (float *)state->vision_x->ptr(), d_tokens, OH, VI, true
    );

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif
    
    state->vision_embed_tokens = d_tokens;
    state->cur_img_token_id = 0;
}

float *forward_text(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id, size_t pos) {
    long hidden_size = config->hidden_size;

    long head_dim = hidden_size / config->num_attention_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;

    bool img_token_true = (token_id == config->image_token_id) || (token_id == config->video_token_id);
    
    // Embed layer
    if (!img_token_true) {
        embedding_lookup(
            weight->token_embedding_table, token_id, state->x, hidden_size
        );
    } else {
        const float *src = (const float *)state->vision_x->ptr() + 1ll * hidden_size * state->cur_img_token_id;
        memcpy(state->x->ptr(), src, 1ll * hidden_size * sizeof(float));
    }

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            (const float *)state->x->ptr(), weight->rms_ffn_w,
            (float *)state->t->ptr(), config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        // QKV Projections
        const float *w_q = (const float *)weight->w_attn_q->ptr({l});
        const float *w_k = (const float *)weight->w_attn_k->ptr({l});
        const float *w_v = (const float *)weight->w_attn_v->ptr({l});

        float *k_cache_ptr = (float *)state->key_cache->ptr({0, l, pos});
        float *v_cache_ptr = (float *)state->value_cache->ptr({0, l, pos});
        
        linear(
            (const float *)state->t->ptr(), w_q, nullptr,
            (float *)state->q->ptr(),
            1, hidden_size, hidden_size, true
        );
        linear(
            (const float *)state->t->ptr(), w_k, nullptr, k_cache_ptr,
            1, kv_dim, hidden_size, true
        );
        linear(
            (const float *)state->t->ptr(), w_v, nullptr, v_cache_ptr,
            1, kv_dim, hidden_size, true
        );

        // QK RMSNorm
        rms_norm(
            (const float *)state->q->ptr(), weight->w_attn_q_norm,
            (float *)state->q->ptr(), config->rms_norm_eps,
            BATCH_SIZE * config->num_attention_heads, 1ll * l
        );
        rms_norm(
            k_cache_ptr, weight->w_attn_k_norm, k_cache_ptr,
            config->rms_norm_eps, BATCH_SIZE * config->num_key_value_heads,
            1ll * l
        );

        // Apply Rotary Position Embeddings
        apply_rotary(
            (float *)state->q->ptr(), state->cos_tensor, state->sin_tensor,
            config->num_attention_heads, head_dim, pos
        );
        apply_rotary(
            k_cache_ptr, state->cos_tensor, state->sin_tensor,
            config->num_key_value_heads, head_dim, pos
        );

        // Multi-head attention
        int kv_mul = config->num_attention_heads / config->num_key_value_heads;  // integer multiplier for GQA

        // Compute attention scores
        attn_scores_all_heads(
            state->key_cache, state->q, state->att,
            1ll * l, config->num_attention_heads, kv_mul, head_dim,
            kv_dim, seq_len, pos
        );

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            state->value_cache, state->att, state->qkv_out, 
            1ll * l, config->num_attention_heads, kv_mul, head_dim, 
            kv_dim, seq_len, pos
        );

        // Output projection: using state->t for attn_out
        const float *w_out_proj = (const float *)weight->w_attn_o->ptr({l});
        linear(
            (const float *)state->qkv_out->ptr(), w_out_proj, nullptr,
            (float *)state->t->ptr(), 1, hidden_size, hidden_size, true
        );

        // Residual connection 1
        add_vector(state->x, state->t, 1ll * BATCH_SIZE * hidden_size);

        // Post-attention RMSNorm
        rms_norm(
            (const float *)state->x->ptr(), weight->rms_attn_w,
            (float *)state->t->ptr(), config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        // MLP: Gate and Up projections
        const float *w_gate = (const float *)weight->w_mlp_gate->ptr({l});
        const float *w_up = (const float *)weight->w_mlp_up->ptr({l});
        linear(
            (const float *)state->t->ptr(), w_gate, nullptr,
            (float *)state->gate->ptr(), 1,
            config->intermediate_size, hidden_size, true
        );
        linear(
            (const float *)state->t->ptr(), w_up, nullptr,
            (float *)state->up->ptr(), 1,
            config->intermediate_size, hidden_size, true
        );
        
        // SwiGLU activation
        swiglu(
            state->gate, state->up, config->intermediate_size
        );

        // MLP: Down projection: using state->t for down
        const float *w_down = (const float *)weight->w_mlp_down->ptr({l});
        linear(
            (const float *)state->gate->ptr(), w_down, nullptr,
            (float *)state->t->ptr(), 1,
            hidden_size, config->intermediate_size, true
        );

        // Residual connection 2
        add_vector(state->x, state->t, 1ll * BATCH_SIZE * hidden_size);

        if (l < config->vision_deep_stack_depth && img_token_true) {
            const float *deep_ptr = (const float *)state->vision_deep_stack->ptr({l, (size_t)state->cur_img_token_id});
            add_vector(state->x, deep_ptr, 1ll * BATCH_SIZE * hidden_size);
        }
    }

    // Final RMSNorm
    rms_norm(
        (const float *)state->x->ptr(), weight->rms_out_w,
        (float *)state->x->ptr(), config->rms_norm_eps, BATCH_SIZE, 0ll
    );

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, hidden_size
    );


    if (img_token_true) {
        state->cur_img_token_id += 1;
    }

    #ifdef CPU_TIME
        exit(1);
    #endif
    
    return (float *)state->logits->ptr();
}

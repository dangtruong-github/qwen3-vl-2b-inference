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

    const DType::Type dtype_weight = weight->vl_patch_emb_w->dtype;
    const DType::Type dtype_scale = weight->vl_patch_emb_w->scale_dtype;
    const size_t vision_group_size = weight->vl_patch_emb_w->group_size;
    const bool g_false = vision_group_size <= 0;

    printf("grid_h=%d, grid_w=%d, total_tokens=%zu\n", grid_h, grid_w, total_tokens);

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

    printf("num_grid_per_side=%ld\n", num_grid_per_side);

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

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif

        const void *w_q = weight->vl_attn_qkv_w->ptr({l});
        const void *w_k = weight->vl_attn_qkv_w->ptr({l, 1});
        const void *w_v = weight->vl_attn_qkv_w->ptr({l, 2});
        const void *b_q = weight->vl_attn_qkv_b->ptr({l});
        const void *b_k = weight->vl_attn_qkv_b->ptr({l, 1});
        const void *b_v = weight->vl_attn_qkv_b->ptr({l, 2});
        const void *w_q_s = g_false ? nullptr : weight->vl_attn_qkv_w->ptr({l}, true);
        const void *w_k_s = g_false ? nullptr : weight->vl_attn_qkv_w->ptr({l, 1}, true);
        const void *w_v_s = g_false ? nullptr : weight->vl_attn_qkv_w->ptr({l, 2}, true);
        const void *b_q_s = g_false ? nullptr : weight->vl_attn_qkv_b->ptr({l}, true);
        const void *b_k_s = g_false ? nullptr : weight->vl_attn_qkv_b->ptr({l, 1}, true);
        const void *b_v_s = g_false ? nullptr : weight->vl_attn_qkv_b->ptr({l, 2}, true);
        
        // use vl_v as temporary buffer for vision_q
        linear(
            (const float *)state->vision_t->ptr(), w_q, w_q_s, b_q, b_q_s,
            (float *)state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
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

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_q->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_q->printDebug("vision_q");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_q->printDebug("vision_q");
        #endif

        // use vl_v as temporary buffer for vision_k
        linear(
            (const float *)state->vision_t->ptr(), w_k, w_k_s, b_k, b_k_s,
            (float *)state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
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

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_k->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_k->printDebug("vision_k");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_k->printDebug("vision_k");
        #endif

        // swap vision_t and vl_v
        linear(
            (const float *)state->vision_t->ptr(), w_v, w_v_s, b_v, b_v_s,
            (float *)state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            (const float *)state->vision_mlp_out->ptr(),
            (float *)state->vision_t->ptr(), total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        #ifdef PRINT_LOGITS_2
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

        #ifdef PRINT_LOGITS_2
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

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif

        const void *w_attn_proj_ptr = weight->vl_attn_proj_w->ptr({l});
        const void *b_attn_proj_ptr = weight->vl_attn_proj_b->ptr({l});
        const void *w_attn_proj_s = g_false ? nullptr : weight->vl_attn_proj_w->ptr({l}, true);
        const void *b_attn_proj_s = g_false ? nullptr : weight->vl_attn_proj_b->ptr({l}, true);

        // use vision_q as temporary buffer here
        linear(
            (const float *)state->vision_t->ptr(),
            w_attn_proj_ptr, w_attn_proj_s,
            b_attn_proj_ptr, b_attn_proj_s,
            (float *)state->vision_q->ptr(), total_tokens, VH, VH,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_q->printDebug("vision_q");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_q->printDebug("vision_q");
        #endif

        add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);

        #ifdef PRINT_LOGITS
            state->vision_x->printDebug("vision_x");
        #endif

        #ifdef PRINT_LOGITS_2
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

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif
        
        const void *w_mlp1_ptr = weight->vl_mlp1_w->ptr({l});
        const void *b_mlp1_ptr = weight->vl_mlp1_b->ptr({l});
        const void *w_mlp1_s = g_false ? nullptr : weight->vl_mlp1_w->ptr({l}, true);
        const void *b_mlp1_s = g_false ? nullptr : weight->vl_mlp1_b->ptr({l}, true);
        linear(
            (const float *)state->vision_t->ptr(),
            w_mlp1_ptr, w_mlp1_s, b_mlp1_ptr, b_mlp1_s,
            (float *)state->vision_mlp_out->ptr(), total_tokens, VI, VH,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        gelu_tanh(state->vision_mlp_out, 1ll * total_tokens * VI);

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif
        
        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif
        
        const void *w_mlp2_ptr = weight->vl_mlp2_w->ptr({l});
        const void *b_mlp2_ptr = weight->vl_mlp2_b->ptr({l});
        const void *w_mlp2_s = g_false ? nullptr : weight->vl_mlp2_w->ptr({l}, true);
        const void *b_mlp2_s = g_false ? nullptr : weight->vl_mlp2_b->ptr({l}, true);
        linear(
            (const float *)state->vision_mlp_out->ptr(),
            w_mlp2_ptr, w_mlp2_s, b_mlp2_ptr, b_mlp2_s,
            (float *)state->vision_t->ptr(), total_tokens, VH, VI,
            true, dtype_weight, dtype_scale, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif

        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

        #ifdef PRINT_LOGITS
            state->vision_x->printDebug("vision_x");
        #endif

        #ifdef PRINT_LOGITS_2
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

            #ifdef PRINT_LOGITS_2
                state->vision_t->printDebug("vision_t");
            #endif

            const void *w_mlp1_d_ptr = weight->vl_d_mlp1_w->ptr({d_stride});
            const void *b_mlp1_d_ptr = weight->vl_d_mlp1_b->ptr({d_stride});
            const void *w_mlp1_d_s = g_false ? nullptr : weight->vl_d_mlp1_w->ptr({d_stride}, true);
            const void *b_mlp1_d_s = g_false ? nullptr : weight->vl_d_mlp1_b->ptr({d_stride}, true);
            linear(
                (const float *)state->vision_t->ptr(),
                w_mlp1_d_ptr, w_mlp1_d_s, b_mlp1_d_ptr, b_mlp1_d_s,
                (float *)state->vision_mlp_out->ptr(), d_tokens, VI, VI,
                true, dtype_weight, dtype_scale, vision_group_size
            );

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif

            #ifdef PRINT_LOGITS_2
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif
        
            gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif

            #ifdef PRINT_LOGITS_2
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif

            const void *w_mlp2_d_ptr = weight->vl_d_mlp2_w->ptr({d_stride});
            const void *b_mlp2_d_ptr = weight->vl_d_mlp2_b->ptr({d_stride});
            const void *w_mlp2_d_s = g_false ? nullptr : weight->vl_d_mlp2_w->ptr({d_stride}, true);
            const void *b_mlp2_d_s = g_false ? nullptr : weight->vl_d_mlp2_b->ptr({d_stride}, true);
            linear(
                (const float *)state->vision_mlp_out->ptr(),
                w_mlp2_d_ptr, w_mlp2_d_s, b_mlp2_d_ptr, b_mlp2_d_s,
                (float *)state->vision_deep_stack->ptr({d_stride}), d_tokens,
                OH, VI, true, dtype_weight, dtype_scale, vision_group_size
            );

            #ifdef PRINT_LOGITS
                state->vision_mlp_out->printDebug("vision_mlp_out");
            #endif

            #ifdef PRINT_LOGITS_2
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
        weight->vl_merge_mlp1_w->ptr(),
        g_false ? nullptr : weight->vl_merge_mlp1_w->ptr({}, true),
        weight->vl_merge_mlp1_b->ptr(),
        g_false ? nullptr : weight->vl_merge_mlp1_b->ptr({}, true),
        (float *)state->vision_mlp_out->ptr(),
        d_tokens, VI, VI, true, dtype_weight, dtype_scale, vision_group_size
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
        weight->vl_merge_mlp2_w->ptr(),
        g_false ? nullptr : weight->vl_merge_mlp2_w->ptr({}, true),
        weight->vl_merge_mlp2_b->ptr(),
        g_false ? nullptr : weight->vl_merge_mlp2_b->ptr({}, true),
        (float *)state->vision_x->ptr(),
        d_tokens, OH, VI, true, dtype_weight, dtype_scale, vision_group_size
    );

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif

    /*
    const float *print_ptr = (const float *)state->vision_x->ptr();
    printf("Shape print: (%zu, %zu)\n", d_tokens, OH);
    for (size_t i = 0; i < d_tokens; ++i) {
        for (size_t j = 0; j < OH; ++j) {
            printf("%.2f ", print_ptr[i * OH + j]);
        }
        printf("\n");
    }
    fflush(stdout);
    exit(1);
    */
    
    state->vision_embed_tokens = d_tokens;
    state->cur_img_token_id = 0;
}

float *forward_text(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id, size_t pos) {
    long hidden_size = config->hidden_size;

    long head_dim = hidden_size / config->num_attention_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const bool g_false = text_group_size <= 0;

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

    #ifdef PRINT_LOGITS
        state->x->printDebug("x");
    #endif

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            (const float *)state->x->ptr(), weight->rms_ffn_w,
            (float *)state->t->ptr(), config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // QKV Projections
        // printf("BEFORE PTR\n");
        // fflush(stdout);
        const void *w_q = weight->w_attn_q->ptr({l});
        const void *w_k = weight->w_attn_k->ptr({l});
        const void *w_v = weight->w_attn_v->ptr({l});
        const void *w_q_s = g_false ? nullptr : weight->w_attn_q->ptr({l}, true);
        const void *w_k_s = g_false ? nullptr : weight->w_attn_k->ptr({l}, true);
        const void *w_v_s = g_false ? nullptr : weight->w_attn_v->ptr({l}, true);

        float *k_cache_ptr = (float *)state->key_cache->ptr({0, l, pos});
        float *v_cache_ptr = (float *)state->value_cache->ptr({0, l, pos});

        // printf("AFTER PTR\n");
        // fflush(stdout);
        
        fflush(stdout);
        linear(
            (const float *)state->t->ptr(), w_q, w_q_s, nullptr, nullptr,
            (float *)state->q->ptr(), 1, hidden_size, hidden_size, true,
            dtype_weight, dtype_scale, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        linear(
            (const float *)state->t->ptr(), w_k, w_k_s, nullptr, nullptr,
            k_cache_ptr, 1, kv_dim, hidden_size, true, dtype_weight,
            dtype_scale, text_group_size
        );
        linear(
            (const float *)state->t->ptr(), w_v, w_v_s, nullptr, nullptr, v_cache_ptr, 1, kv_dim, hidden_size, true, dtype_weight,
            dtype_scale, text_group_size
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

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        // Apply Rotary Position Embeddings
        apply_rotary(
            (float *)state->q->ptr(), state->cos_tensor, state->sin_tensor,
            config->num_attention_heads, head_dim, pos
        );
        apply_rotary(
            k_cache_ptr, state->cos_tensor, state->sin_tensor,
            config->num_key_value_heads, head_dim, pos
        );

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        // Multi-head attention
        int kv_mul = config->num_attention_heads / config->num_key_value_heads;  // integer multiplier for GQA

        // Compute attention scores
        attn_scores_all_heads(
            state->key_cache, state->q, state->att,
            1ll * l, config->num_attention_heads, kv_mul, head_dim,
            kv_dim, seq_len, pos
        );

        #ifdef PRINT_LOGITS
            state->att->printDebug("att");
        #endif

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            state->value_cache, state->att, state->qkv_out, 
            1ll * l, config->num_attention_heads, kv_mul, head_dim, 
            kv_dim, seq_len, pos
        );

        #ifdef PRINT_LOGITS
            state->qkv_out->printDebug("qkv_out");
        #endif

        // Output projection: using state->t for attn_out
        const void *w_out_proj = weight->w_attn_o->ptr({l});
        const void *w_out_proj_s = g_false ? nullptr : weight->w_attn_o->ptr({l}, true);
        linear(
            (const float *)state->qkv_out->ptr(),
            w_out_proj, w_out_proj_s, nullptr, nullptr,
            (float *)state->t->ptr(), 1, hidden_size,
            hidden_size, true, dtype_weight, dtype_scale, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // Residual connection 1
        add_vector(state->x, state->t, 1ll * BATCH_SIZE * hidden_size);

        #ifdef PRINT_LOGITS
            state->x->printDebug("x");
        #endif

        // Post-attention RMSNorm
        rms_norm(
            (const float *)state->x->ptr(), weight->rms_attn_w,
            (float *)state->t->ptr(), config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // MLP: Gate and Up projections
        const void *w_gate = weight->w_mlp_gate->ptr({l});
        const void *w_up = weight->w_mlp_up->ptr({l});
        const void *w_gate_s = g_false ? nullptr : weight->w_mlp_gate->ptr({l}, true);
        const void *w_up_s = g_false ? nullptr : weight->w_mlp_up->ptr({l}, true);
        linear(
            (const float *)state->t->ptr(),
            w_gate, w_gate_s, nullptr, nullptr,
            (float *)state->gate->ptr(), 1,
            config->intermediate_size, hidden_size, true,
            dtype_weight, dtype_scale, text_group_size
        );
        linear(
            (const float *)state->t->ptr(),
            w_up, w_up_s, nullptr, nullptr,
            (float *)state->up->ptr(), 1,
            config->intermediate_size, hidden_size, true,
            dtype_weight, dtype_scale, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->gate->printDebug("gate");
            state->up->printDebug("up");
        #endif
        
        // SwiGLU activation
        swiglu(
            state->gate, state->up, config->intermediate_size
        );

        #ifdef PRINT_LOGITS
            state->gate->printDebug("gate");
        #endif

        // MLP: Down projection: using state->t for down
        const void *w_down = weight->w_mlp_down->ptr({l});
        const void *w_down_s = g_false ? nullptr : weight->w_mlp_down->ptr({l}, true);
        linear(
            (const float *)state->gate->ptr(),
            w_down, w_down_s, nullptr, nullptr,
            (float *)state->t->ptr(), 1, hidden_size,
            config->intermediate_size, true, dtype_weight,
            dtype_scale, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // Residual connection 2
        add_vector(state->x, state->t, 1ll * BATCH_SIZE * hidden_size);

        #ifdef PRINT_LOGITS
            state->x->printDebug("x");
        #endif

        if (l < config->vision_deep_stack_depth && img_token_true) {
            const float *deep_ptr = (const float *)state->vision_deep_stack->ptr({l, (size_t)state->cur_img_token_id});
            add_vector(state->x, deep_ptr, 1ll * BATCH_SIZE * hidden_size);

            #ifdef PRINT_LOGITS
                state->x->printDebug("x");
            #endif
        }
        if (pos < 50) {
            /*
            if (l == 0) {
                const float *print_ptr = (const float *)state->x->ptr();
                printf("pos = %d\n", pos);
                for (int i = 0; i < config->hidden_size; i++) {
                    printf("%.2f ", print_ptr[i]);
                }
                printf("\n");
                fflush(stdout);
            }
            */
        } else {
            // exit(1);
        }
    }

    // Final RMSNorm
    rms_norm(
        (const float *)state->x->ptr(), weight->rms_out_w,
        (float *)state->x->ptr(), config->rms_norm_eps, BATCH_SIZE, 0ll
    );

    #ifdef PRINT_LOGITS
        state->x->printDebug("x");
    #endif

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, hidden_size
    );

    #ifdef PRINT_LOGITS
        state->logits->printDebug("logits");
    #endif

    if (img_token_true) {
        state->cur_img_token_id += 1;
    }

    #ifdef CPU_TIME
        if (pos > 100) exit(1);
    #endif
    
    return (float *)state->logits->ptr();
}

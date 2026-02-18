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
    const bool vision_gq = false;

    printf("grid_h=%d, grid_w=%d, total_tokens=%zu\n", grid_h, grid_w, total_tokens);

    conv_3d(
        weight->vl_patch_emb_w, weight->vl_patch_emb_b, img_data,
        state->vision_x, img_h, VC, VTP, VP, VH
    );

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif

    free(img_data);

    long VNP = config->max_vision_embeddings;
    long num_grid_per_side = sqrt(VNP);

    printf("num_grid_per_side=%ld\n", num_grid_per_side);

    vision_pos_embed(
        weight->vl_pos_emb_w, state->vision_t,
        grid_h, grid_w, num_grid_per_side, VSP, VH
    );

    #ifdef PRINT_LOGITS
        state->vision_t->printDebug("vision_t");
    #endif

    add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif

    vision_rot_pos_emb(
        state->vision_pe_cos, state->vision_pe_sin,
        state->vision_cos_tensor, state->vision_sin_tensor,
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

        PtrPair w_q = weight->vl_attn_qkv_w->ptr_all({l});
        PtrPair w_k = weight->vl_attn_qkv_w->ptr_all({l, 1});
        PtrPair w_v = weight->vl_attn_qkv_w->ptr_all({l, 2});
        PtrPair b_q = weight->vl_attn_qkv_b->ptr_all({l});
        PtrPair b_k = weight->vl_attn_qkv_b->ptr_all({l, 1});
        PtrPair b_v = weight->vl_attn_qkv_b->ptr_all({l, 2});
        
        // use vl_v as temporary buffer for vision_q
        linear(
            state->vision_t->ptr(), w_q.buf, w_q.scale, b_q.buf, b_q.scale,
            state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_q, total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_q->printDebug("vision_q");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_q->printDebug("vision_q");
        #endif

        // use vl_v as temporary buffer for vision_k
        linear(
            state->vision_t->ptr(), w_k.buf, w_k.scale, b_k.buf, b_k.scale,
            state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_k, total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_k->printDebug("vision_k");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_k->printDebug("vision_k");
        #endif

        // swap vision_t and vl_v
        linear(
            state->vision_t->ptr(), w_v.buf, w_v.scale, b_v.buf, b_v.scale,
            state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #ifdef PRINT_LOGITS
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_mlp_out->printDebug("vision_mlp_out");
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_t, total_tokens, VNH, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif
        
        vision_att(
            state->vision_q, state->vision_k, state->vision_t,
            state->vision_attn_scores, state->vision_mlp_out, 
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
            state->vision_mlp_out, state->vision_t, VNH, total_tokens, VHD
        );

        #ifdef PRINT_LOGITS
            state->vision_t->printDebug("vision_t");
        #endif

        #ifdef PRINT_LOGITS_2
            state->vision_t->printDebug("vision_t");
        #endif

        PtrPair w_attn_proj_ptr, b_attn_proj_ptr;
        w_attn_proj_ptr = weight->vl_attn_proj_w->ptr_all({l});
        b_attn_proj_ptr = weight->vl_attn_proj_b->ptr_all({l});

        // use vision_q as temporary buffer here
        linear(
            state->vision_t->ptr(), w_attn_proj_ptr.buf, w_attn_proj_ptr.scale,
            b_attn_proj_ptr.buf, b_attn_proj_ptr.scale, state->vision_q->ptr(),
            total_tokens, VH, VH, !weight->vl_attn_proj_w->permuted,
            state->vision_t->dtype, dtype_weight, dtype_scale,
            state->vision_q->dtype, vision_gq, vision_group_size
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
        
        PtrPair w_mlp1_ptr = weight->vl_mlp1_w->ptr_all({l});
        PtrPair b_mlp1_ptr = weight->vl_mlp1_b->ptr_all({l});
        linear(
            state->vision_t->ptr(), w_mlp1_ptr.buf, w_mlp1_ptr.scale,
            b_mlp1_ptr.buf, b_mlp1_ptr.scale, state->vision_mlp_out->ptr(),
            total_tokens, VI, VH, !weight->vl_mlp1_w->permuted,
            state->vision_t->dtype, dtype_weight, dtype_scale,
            state->vision_mlp_out->dtype, vision_gq, vision_group_size
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
        
        PtrPair w_mlp2_ptr = weight->vl_mlp2_w->ptr_all({l});
        PtrPair b_mlp2_ptr = weight->vl_mlp2_b->ptr_all({l});
        linear(
            state->vision_mlp_out->ptr(), w_mlp2_ptr.buf, w_mlp2_ptr.scale,
            b_mlp2_ptr.buf, b_mlp2_ptr.scale, state->vision_t->ptr(),
            total_tokens, VH, VI, !weight->vl_mlp2_w->permuted,
            state->vision_mlp_out->dtype, dtype_weight, dtype_scale,
            state->vision_t->dtype, vision_gq, vision_group_size
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
            
            PtrPair w_mlp1_d_ptr = weight->vl_d_mlp1_w->ptr_all({d_stride});
            PtrPair b_mlp1_d_ptr = weight->vl_d_mlp1_b->ptr_all({d_stride});
            linear(
                state->vision_t->ptr(), w_mlp1_d_ptr.buf, w_mlp1_d_ptr.scale,
                b_mlp1_d_ptr.buf, b_mlp1_d_ptr.scale,
                state->vision_mlp_out->ptr(), d_tokens, VI, VI,
                !weight->vl_d_mlp1_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size
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

            PtrPair w_mlp2_d_ptr = weight->vl_d_mlp2_w->ptr_all({d_stride});
            PtrPair b_mlp2_d_ptr = weight->vl_d_mlp2_b->ptr_all({d_stride});
            linear(
                state->vision_mlp_out->ptr(), w_mlp2_d_ptr.buf,
                w_mlp2_d_ptr.scale, b_mlp2_d_ptr.buf, b_mlp2_d_ptr.scale,
                state->vision_deep_stack->ptr({d_stride}), d_tokens, OH, VI,
                !weight->vl_d_mlp2_w->permuted, state->vision_mlp_out->dtype,
                dtype_weight, dtype_scale, state->vision_deep_stack->dtype,
                vision_gq, vision_group_size
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

    PtrPair vl_merge_mlp1_w_ptr = weight->vl_merge_mlp1_w->ptr_all();
    PtrPair vl_merge_mlp1_b_ptr = weight->vl_merge_mlp1_b->ptr_all();
    linear(
        state->vision_t->ptr(), vl_merge_mlp1_w_ptr.buf,
        vl_merge_mlp1_w_ptr.scale, vl_merge_mlp1_b_ptr.buf,
        vl_merge_mlp1_b_ptr.scale, state->vision_mlp_out->ptr(), d_tokens, VI,
        VI, !weight->vl_merge_mlp1_w->permuted, state->vision_t->dtype,
        dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
        vision_gq, vision_group_size
    );

    #ifdef PRINT_LOGITS
        state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

    gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

    #ifdef PRINT_LOGITS
        state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

    PtrPair vl_merge_mlp2_w_ptr = weight->vl_merge_mlp2_w->ptr_all();
    PtrPair vl_merge_mlp2_b_ptr = weight->vl_merge_mlp2_b->ptr_all();
    linear(
        state->vision_mlp_out->ptr(), vl_merge_mlp2_w_ptr.buf,
        vl_merge_mlp2_w_ptr.scale, vl_merge_mlp2_b_ptr.buf,
        vl_merge_mlp2_b_ptr.scale, state->vision_x->ptr(), d_tokens, OH, VI,
        !weight->vl_merge_mlp2_w->permuted, state->vision_mlp_out->dtype,
        dtype_weight, dtype_scale, state->vision_x->dtype,
        vision_gq, vision_group_size
    );

    #ifdef PRINT_LOGITS
        state->vision_x->printDebug("vision_x");
    #endif
    
    state->vision_embed_tokens = d_tokens;
    state->cur_img_token_id = 0;
}

float *forward_text(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id, size_t pos) {
    long hidden_size = config->hidden_size;
    long num_heads = config->num_attention_heads;
    long num_kv_heads = config->num_key_value_heads;
    long head_dim = hidden_size / num_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;
    int kv_mul = num_heads / num_kv_heads;

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const bool text_gq = config->group_quantized ? true : false;

    bool img_token_true = (token_id == config->image_token_id) || (token_id == config->video_token_id);
    
    // Embed layer
    if (!img_token_true) {
        embedding_lookup(
            weight->token_embedding_table, state->x, token_id, hidden_size
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
            state->x, weight->rms_ffn_w, state->t,
            config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // QKV Projections
        // printf("BEFORE PTR\n");
        // fflush(stdout);
        PtrPair w_q = weight->w_attn_q->ptr_all({l});
        PtrPair w_k = weight->w_attn_k->ptr_all({l});
        PtrPair w_v = weight->w_attn_v->ptr_all({l});

        // printf("AFTER PTR\n");
        // fflush(stdout);
        
        linear(
            state->t->ptr(), w_q.buf, w_q.scale, nullptr, nullptr,
            state->q->ptr(), 1, hidden_size, hidden_size,
            !weight->w_attn_q->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->q->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        linear(
            state->t->ptr(), w_k.buf, w_k.scale, nullptr, nullptr,
            state->k->ptr(), 1, kv_dim, hidden_size,
            !weight->w_attn_k->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->k->dtype, text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_v.buf, w_v.scale, nullptr, nullptr,
            state->v->ptr(), 1, kv_dim, hidden_size,
            !weight->w_attn_v->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->v->dtype, text_gq, text_group_size
        );

        // QK RMSNorm
        rms_norm_inplace(
            state->q, weight->w_attn_q_norm, config->rms_norm_eps,
            BATCH_SIZE * num_heads, 1ll * l
        );
        rms_norm_inplace(
            state->k, weight->w_attn_k_norm, config->rms_norm_eps,
            BATCH_SIZE * num_kv_heads, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        // Apply Rotary Position Embeddings
        apply_rotary(
            state->q, state->cos_tensor, state->sin_tensor,
            num_heads, head_dim, pos
        );

        const float *k_cache_l = (const float *)state->key_cache->ptr({0, l});
        float *k_cache_ptr = (float *)k_cache_l + kv_pos_off;
        apply_rotary_cache(
            state->k, k_cache_ptr, state->cos_tensor, state->sin_tensor,
            num_kv_heads, head_dim, pos, kv_all_off
        );

        const float *v_cache_l = (const float *)state->value_cache->ptr({0, l});
        float *v_cache_ptr = (float *)v_cache_l + kv_pos_off;

        for (int h = 0; h < num_kv_heads; h++) {
            memcpy(v_cache_ptr + h * kv_all_off, (const float *)state->v->ptr() + h*head_dim, head_dim*sizeof(float));
        }

        #ifdef PRINT_LOGITS
            state->q->printDebug("q");
        #endif

        // Multi-head attention

        // Compute attention scores
        attn_scores_all_heads(
            k_cache_l, state->q, state->att, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos
        );

        #ifdef PRINT_LOGITS
            state->att->printDebug("att");
        #endif

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            v_cache_l, state->att, state->qkv_out, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos
        );

        #ifdef PRINT_LOGITS
            state->qkv_out->printDebug("qkv_out");
        #endif

        // Output projection: using state->t for attn_out
        PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
        linear(
            state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale, nullptr,
            nullptr, state->t->ptr(), 1, hidden_size, hidden_size,
            !weight->w_attn_o->permuted, state->qkv_out->dtype, dtype_weight,
            dtype_scale, state->t->dtype, text_gq, text_group_size
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
            state->x, weight->rms_attn_w, state->t,
            config->rms_norm_eps, BATCH_SIZE, 1ll * l
        );

        #ifdef PRINT_LOGITS
            state->t->printDebug("t");
        #endif

        // MLP: Gate and Up projections
        PtrPair w_gate = weight->w_mlp_gate->ptr_all({l});
        PtrPair w_up = weight->w_mlp_up->ptr_all({l});
        linear(
            state->t->ptr(), w_gate.buf, w_gate.scale, nullptr, nullptr,
            state->gate->ptr(), 1, config->intermediate_size, hidden_size,
            !weight->w_mlp_gate->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->gate->dtype, text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_up.buf, w_up.scale, nullptr, nullptr,
            state->up->ptr(), 1, config->intermediate_size, hidden_size,
            !weight->w_mlp_up->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->up->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            state->gate->printDebug("gate");
            state->up->printDebug("up");
        #endif
        
        // SwiGLU activation
        swiglu(state->gate, state->up, config->intermediate_size);

        #ifdef PRINT_LOGITS
            state->gate->printDebug("gate");
        #endif

        // MLP: Down projection: using state->t for down
        PtrPair w_down = weight->w_mlp_down->ptr_all({l});
        linear(
            state->gate->ptr(), w_down.buf, w_down.scale, nullptr, nullptr,
            state->t->ptr(), 1, hidden_size, config->intermediate_size,!weight->w_mlp_down->permuted, state->gate->dtype, dtype_weight,
            dtype_scale, state->t->dtype, text_gq, text_group_size
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
            const void *deep_ptr = state->vision_deep_stack->ptr({l, (size_t)state->cur_img_token_id});
            add_vector(state->x, deep_ptr, state->vision_deep_stack->dtype, 1ll * BATCH_SIZE * hidden_size);

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
    rms_norm_inplace(
        state->x, weight->rms_out_w, config->rms_norm_eps, BATCH_SIZE, 0ll
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

    #if defined(CPU_TIME) || defined(CPU_TIME_FP16)
        if (pos > 10) exit(1);
    #endif
    
    return (float *)state->logits->ptr();
}

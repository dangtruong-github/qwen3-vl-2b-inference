#include "../include/forward.hpp"

void forward_img(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    float *img_data, int img_h, int img_w, int grid_h, int grid_w, const bool warm_up
) {
    if (img_data == nullptr) {
        return;
    }

    // ... [Variable initializations remain the same] ...
    long VC = config->vision_num_channels;
    long VTP = config->vision_temporal_patch_size;
    long VP = config->vision_patch_size;
    long VH = config->vision_hidden_size;
    long VSP = config->vision_spatial_merge_size;
    long total_tokens = grid_h * grid_w;
    long VNH = config->vision_num_heads;
    long VHD = VH / VNH;
    long VI = config->vision_intermediate_size;
    long OH = config->out_hidden_size;
    float vision_scale = config->vision_scale;
    long d_tokens = total_tokens / (VSP * VSP);

    const DType::Type dtype_weight = weight->vl_patch_emb_w->dtype;
    const DType::Type dtype_scale = weight->vl_patch_emb_w->scale_dtype;
    const size_t vision_group_size = weight->vl_patch_emb_w->group_size;
    const bool vision_gq = false;

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("conv_3d", warm_up);
        #endif
        conv_3d(
            weight->vl_patch_emb_w, weight->vl_patch_emb_b, img_data,
            state->vision_x, img_h, VC, VTP, VP, VH
        );
    }

    free(img_data);

    long VNP = config->max_vision_embeddings;
    long num_grid_per_side = sqrt(VNP);

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("vision_pos_embed", warm_up);
        #endif
        vision_pos_embed(
            weight->vl_pos_emb_w, state->vision_t,
            grid_h, grid_w, num_grid_per_side, VSP, VH
        );
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("add_vector_pos", warm_up);
        #endif
        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("vision_rot_pos_emb", warm_up);
        #endif
        vision_rot_pos_emb(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_cos_tensor, state->vision_sin_tensor,
            grid_h, grid_w, config->vision_spatial_merge_size, VHD
        );
    }

    for (size_t l = 0; l < config->vision_depth; l++) {
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("layer_norm_1", warm_up);
            #endif
            layer_norm(
                state->vision_x, weight->vl_norm1_w,
                weight->vl_norm1_b, state->vision_t,
                config->rms_norm_eps, total_tokens, 1ll * l
            );
        }

        PtrPair w_q = weight->vl_attn_qkv_w->ptr_all({l});
        PtrPair w_k = weight->vl_attn_qkv_w->ptr_all({l, 1});
        PtrPair w_v = weight->vl_attn_qkv_w->ptr_all({l, 2});
        PtrPair b_q = weight->vl_attn_qkv_b->ptr_all({l});
        PtrPair b_k = weight->vl_attn_qkv_b->ptr_all({l, 1});
        PtrPair b_v = weight->vl_attn_qkv_b->ptr_all({l, 2});
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("linear_q", warm_up);
            #endif
            linear(
                state->vision_t->ptr(), w_q.buf, w_q.scale, w_q.sum_int8,
                b_q.buf, b_q.scale, state->vision_mlp_out->ptr(), total_tokens, VH,
                VH, !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size, false
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("rope_q", warm_up);
            #endif
            vision_apply_rotary_inplace(
                state->vision_pe_cos, state->vision_pe_sin,
                state->vision_mlp_out, total_tokens, VNH, VHD
            );
            tensor_transpose(
                state->vision_mlp_out, state->vision_q, total_tokens, VNH, VHD
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("linear_k", warm_up);
            #endif
            linear(
                state->vision_t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, b_k.buf,
                b_k.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
                !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size, false
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("rope_k", warm_up);
            #endif
            vision_apply_rotary_inplace(
                state->vision_pe_cos, state->vision_pe_sin,
                state->vision_mlp_out, total_tokens, VNH, VHD
            );
            tensor_transpose(
                state->vision_mlp_out, state->vision_k, total_tokens, VNH, VHD
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("linear_v", warm_up);
            #endif
            linear(
                state->vision_t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, b_v.buf,
                b_v.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
                !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size, false
            );
            tensor_transpose(
                state->vision_mlp_out, state->vision_t, total_tokens, VNH, VHD
            );
        }
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("vision_attention", warm_up);
            #endif
            vision_att(
                state->vision_q, state->vision_k, state->vision_t,
                state->vision_attn_scores, state->vision_mlp_out, 
                VNH, total_tokens, VHD, config->max_vision_attention_size, vision_scale
            );
            tensor_transpose(
                state->vision_mlp_out, state->vision_t, VNH, total_tokens, VHD
            );
        }

        PtrPair w_attn_proj_ptr = weight->vl_attn_proj_w->ptr_all({l});
        PtrPair b_attn_proj_ptr = weight->vl_attn_proj_b->ptr_all({l});

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("linear_attn_proj", warm_up);
            #endif
            linear(
                state->vision_t->ptr(), w_attn_proj_ptr.buf, w_attn_proj_ptr.scale,
                w_attn_proj_ptr.sum_int8, b_attn_proj_ptr.buf,
                b_attn_proj_ptr.scale, state->vision_q->ptr(),
                total_tokens, VH, VH, !weight->vl_attn_proj_w->permuted,
                state->vision_t->dtype, dtype_weight, dtype_scale,
                state->vision_q->dtype, vision_gq, vision_group_size, false
            );
            add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("layer_norm_2", warm_up);
            #endif
            layer_norm(
                state->vision_x, weight->vl_norm2_w,
                weight->vl_norm2_b, state->vision_t,
                config->rms_norm_eps, total_tokens, 1ll * l
            );
        }
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("mlp_block", warm_up);
            #endif
            PtrPair w_mlp1_ptr = weight->vl_mlp1_w->ptr_all({l});
            PtrPair b_mlp1_ptr = weight->vl_mlp1_b->ptr_all({l});
            linear(
                state->vision_t->ptr(), w_mlp1_ptr.buf, w_mlp1_ptr.scale,
                w_mlp1_ptr.sum_int8, b_mlp1_ptr.buf, b_mlp1_ptr.scale,
                state->vision_mlp_out->ptr(), total_tokens, VI, VH,
                !weight->vl_mlp1_w->permuted, state->vision_t->dtype, dtype_weight,
                dtype_scale, state->vision_mlp_out->dtype, vision_gq,
                vision_group_size, false
            );

            gelu_tanh(state->vision_mlp_out, 1ll * total_tokens * VI);
            
            PtrPair w_mlp2_ptr = weight->vl_mlp2_w->ptr_all({l});
            PtrPair b_mlp2_ptr = weight->vl_mlp2_b->ptr_all({l});
            linear(
                state->vision_mlp_out->ptr(), w_mlp2_ptr.buf, w_mlp2_ptr.scale,
                w_mlp2_ptr.sum_int8, b_mlp2_ptr.buf, b_mlp2_ptr.scale,
                state->vision_t->ptr(), total_tokens, VH, VI,
                !weight->vl_mlp2_w->permuted, state->vision_mlp_out->dtype,
                dtype_weight, dtype_scale, state->vision_t->dtype, vision_gq,
                vision_group_size, false
            );

            add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);
        }

        if (config->deep_layer[l] > 0) {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("deep_stack_block", warm_up);
            #endif
            size_t d_stride = config->deep_layer[l] - 1;
            layer_norm(
                state->vision_x, weight->vl_d_norm_w,
                weight->vl_d_norm_b, state->vision_t,
                config->rms_norm_eps, d_tokens, 1ll * d_stride
            );
            
            PtrPair w_mlp1_d_ptr = weight->vl_d_mlp1_w->ptr_all({d_stride});
            PtrPair b_mlp1_d_ptr = weight->vl_d_mlp1_b->ptr_all({d_stride});
            linear(
                state->vision_t->ptr(), w_mlp1_d_ptr.buf, w_mlp1_d_ptr.scale,
                w_mlp1_d_ptr.sum_int8, b_mlp1_d_ptr.buf, b_mlp1_d_ptr.scale,
                state->vision_mlp_out->ptr(), d_tokens, VI, VI,
                !weight->vl_d_mlp1_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size, false
            );
        
            gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

            PtrPair w_mlp2_d_ptr = weight->vl_d_mlp2_w->ptr_all({d_stride});
            PtrPair b_mlp2_d_ptr = weight->vl_d_mlp2_b->ptr_all({d_stride});
            linear(
                state->vision_mlp_out->ptr(), w_mlp2_d_ptr.buf,
                w_mlp2_d_ptr.scale, w_mlp2_d_ptr.sum_int8, b_mlp2_d_ptr.buf,
                b_mlp2_d_ptr.scale, state->vision_deep_stack->ptr({d_stride}),
                d_tokens, OH, VI, !weight->vl_d_mlp2_w->permuted,
                state->vision_mlp_out->dtype, dtype_weight, dtype_scale,
                state->vision_deep_stack->dtype, vision_gq,
                vision_group_size, false
            );
        }
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("final_merge_block", warm_up);
        #endif
        layer_norm(
            state->vision_x, weight->vl_merge_norm_w,
            weight->vl_merge_norm_b, state->vision_t,
            config->rms_norm_eps, total_tokens, 0
        );

        PtrPair vl_merge_mlp1_w_ptr = weight->vl_merge_mlp1_w->ptr_all();
        PtrPair vl_merge_mlp1_b_ptr = weight->vl_merge_mlp1_b->ptr_all();
        linear(
            state->vision_t->ptr(), vl_merge_mlp1_w_ptr.buf,
            vl_merge_mlp1_w_ptr.scale, vl_merge_mlp1_w_ptr.sum_int8,
            vl_merge_mlp1_b_ptr.buf, vl_merge_mlp1_b_ptr.scale,
            state->vision_mlp_out->ptr(), d_tokens, VI, VI,
            !weight->vl_merge_mlp1_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size, false
        );

        gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

        PtrPair vl_merge_mlp2_w_ptr = weight->vl_merge_mlp2_w->ptr_all();
        PtrPair vl_merge_mlp2_b_ptr = weight->vl_merge_mlp2_b->ptr_all();
        linear(
            state->vision_mlp_out->ptr(), vl_merge_mlp2_w_ptr.buf,
            vl_merge_mlp2_w_ptr.scale, vl_merge_mlp2_w_ptr.sum_int8,
            vl_merge_mlp2_b_ptr.buf, vl_merge_mlp2_b_ptr.scale,
            state->vision_x->ptr(), d_tokens, OH, VI,
            !weight->vl_merge_mlp2_w->permuted, state->vision_mlp_out->dtype,
            dtype_weight, dtype_scale, state->vision_x->dtype,
            vision_gq, vision_group_size, false
        );
    }
    
    state->vision_embed_tokens = d_tokens;
    state->cur_img_token_id = 0;
}

void forward_text_prefill(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    int *token_list, const size_t prefill_size, size_t pos, bool warm_up
) {
    long hidden_size = config->hidden_size;
    long num_heads = config->num_attention_heads;
    long num_kv_heads = config->num_key_value_heads;
    long head_dim = hidden_size / num_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;
    int kv_mul = num_heads / num_kv_heads;

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const size_t cache_group_size = config->cache_group_size;
    const bool text_gq = config->group_quantized ? true : false;

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;
    const size_t kv_pos_off_bytes = kv_pos_off * state->key_cache->get_dtype_size();
    const size_t kv_pos_scale_off = kv_pos_off / text_group_size;

    int num_img_tokens = 0;
    size_t first_img_token = 0;
    size_t first_img_token_id = 0;

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("text_embedding_lookup", warm_up);
        #endif
        // Embed layer

        for (size_t i = 0; i < prefill_size; ++i) { 
            int token_id = token_list[i];
            if ((token_id == config->image_token_id) || (token_id == config->video_token_id)) {
                if (num_img_tokens == 0) {
                    first_img_token = i;
                    first_img_token_id = state->cur_img_token_id;
                }
                num_img_tokens++;
            } else {
                embedding_lookup(
                    weight->token_embedding_table,
                    state->x, i, token_id, hidden_size
                );
            }
        }

        if (num_img_tokens > 0) {
            const float *src = (const float *)state->vision_x->ptr() + 1ll * hidden_size * first_img_token_id;
            memcpy(
                state->x->ptr({first_img_token}), src,
                1ll * num_img_tokens * hidden_size * sizeof(float)
            );
            state->cur_img_token_id += num_img_tokens;
        }

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t i = 0; i < prefill_size; ++i) { 
                    state->x->printDebug("x", {i});
                }
            }
        #endif
    }

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_rms_linear_qkv", warm_up);
            #endif
            fused_rms_linear_qkv_dispatch(
                weight->rms_ffn_w, weight->w_attn_qkv, state->x,
                state->t, state->q, state->k, state->v, prefill_size,
                kv_dim, hidden_size, dtype_weight, dtype_scale, text_gq,
                text_group_size, config->rms_norm_eps, 1ll * l, warm_up
            );
        }

        PtrPair k_cache_l_pair = state->key_cache->ptr_all({0, l});
        PtrPair v_cache_l_pair = state->value_cache->ptr_all({0, l});
        const char *k_cache_l = (const char *)(k_cache_l_pair.buf);
        const char *v_cache_l = (const char *)(v_cache_l_pair.buf);
        float *k_cache_s = (float *)(k_cache_l_pair.scale);
        float *v_cache_s = (float *)(v_cache_l_pair.scale);

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_qk_norm", warm_up);
            #endif
            fused_rms_rotary_q_dispatch(
                state->q, weight->w_attn_q_norm, state->cos_tensor,
                state->sin_tensor, num_heads, head_dim, prefill_size,
                1ll * l, pos, config->rms_norm_eps, warm_up
            );
            char *k_cache_ptr = (char *)(k_cache_l + kv_pos_off_bytes);
            float *k_cache_s_ptr = state->key_cache->dtype == DType::INT8
                ? (k_cache_s + kv_pos_scale_off) : nullptr;
            fused_rms_rotary_k_dispatch(
                state->k, k_cache_ptr, k_cache_s_ptr, state->key_cache,
                weight->w_attn_k_norm, state->cos_tensor,
                state->sin_tensor, num_kv_heads, head_dim, prefill_size,
                state->k->dtype, state->key_cache->dtype, 1ll * l,
                kv_all_off, pos, config->rms_norm_eps,
                cache_group_size, warm_up
            );
        }

        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_v_cache_update", warm_up);
            #endif

            copy_to_v_cache(
                state->v, (char *)v_cache_l, v_cache_s,
                state->value_cache->dtype, 1, head_dim,
                num_kv_heads, kv_pos_off, kv_all_off,
                kv_pos_scale_off, cache_group_size, warm_up
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_kv_cache_update", warm_up);
            #endif

            copy_to_v_cache(
                state->v, (char *)v_cache_l, v_cache_s,
                state->value_cache->dtype, prefill_size,
                head_dim, num_kv_heads, kv_pos_off, kv_all_off,
                kv_pos_scale_off, cache_group_size, warm_up
            );
        }

        if (l == config->num_hidden_layers - 1) return;

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_attention_mechanism", warm_up);
            #endif

            fused_att_dispatch(
                k_cache_l, v_cache_l, k_cache_s, v_cache_s,
                state->q, state->att, state->qkv_out,
                num_heads, head_dim, kv_mul, kv_dim,
                kv_all_off, pos, state->key_cache->dtype,
                state->value_cache->dtype, cache_group_size,
                prefill_size, warm_up
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_attn_out_linear", warm_up);
            #endif
            PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
            linear(
                state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale,
                w_out_proj.sum_int8, nullptr, nullptr, state->x->ptr(), prefill_size,
                hidden_size, hidden_size, !weight->w_attn_o->permuted,
                state->qkv_out->dtype, dtype_weight, dtype_scale,
                state->t->dtype, text_gq, text_group_size, true
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->t->printDebug("t", {i});
                    }
                }
            #endif

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->x->printDebug("x", {i});
                    }
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_mlp_block", warm_up);
            #endif

            fused_rms_mlp_swiglu_dispatch(
                weight->rms_attn_w, weight->w_mlp_gate, weight->w_mlp_up,
                state->x, state->t, state->gate, state->up, prefill_size,
                hidden_size, config->intermediate_size, weight->w_mlp_gate->dtype,
                weight->w_mlp_gate->scale_dtype, text_gq, config->rms_norm_eps,
                text_group_size, 1ll * l, warm_up
            );

            PtrPair w_down = weight->w_mlp_down->ptr_all({l});
            linear(
                state->gate->ptr(), w_down.buf, w_down.scale, w_down.sum_int8, 
                nullptr, nullptr, state->x->ptr(), prefill_size, hidden_size,
                config->intermediate_size, !weight->w_mlp_down->permuted,
                state->gate->dtype, dtype_weight, dtype_scale, state->t->dtype,
                text_gq, text_group_size, true
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->t->printDebug("t", {i});
                    }
                }
            #endif

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->x->printDebug("x", {i});
                    }
                }
            #endif
        }

        if (l < config->vision_deep_stack_depth) {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_vision_deep_stack_add", warm_up);
            #endif
            if (num_img_tokens > 0) {
                const void *deep_ptr = state->vision_deep_stack->ptr({l, first_img_token_id});
                void *x_ptr = state->x->ptr({first_img_token});

                add_vector(
                    x_ptr, deep_ptr, state->vision_deep_stack->dtype, state->x->dtype, num_img_tokens * hidden_size
                );
            }
        }
    }
}

size_t forward_text_decode(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    int token_id, size_t pos, bool warm_up
) {
    long hidden_size = config->hidden_size;
    long num_heads = config->num_attention_heads;
    long num_kv_heads = config->num_key_value_heads;
    long head_dim = hidden_size / num_heads;
    long seq_len = config->seq_len;
    long kv_dim = config->num_key_value_heads * head_dim;
    int kv_mul = num_heads / num_kv_heads;

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const size_t cache_group_size = config->cache_group_size;
    const bool text_gq = config->group_quantized ? true : false;

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;
    const size_t kv_pos_off_bytes = kv_pos_off * state->key_cache->get_dtype_size();
    const size_t kv_pos_scale_off = kv_pos_off / text_group_size;

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        if (l == 0) {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_embed_rms_linear_qkv", warm_up);
            #endif

            fused_decode_embed_rms_linear_dispatch(
                weight->token_embedding_table, weight->rms_ffn_w,
                weight->w_attn_qkv, state->x, state->t, state->q, state->k,
                state->v, token_id, kv_dim, hidden_size, dtype_weight, dtype_scale, text_gq, text_group_size,
                config->rms_norm_eps, 1ll * l, warm_up
            );
        } else {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_rms_linear_qkv", warm_up);
            #endif
            
            fused_rms_linear_qkv_dispatch(
                weight->rms_ffn_w, weight->w_attn_qkv, state->x,
                state->t, state->q, state->k, state->v, 1, kv_dim,
                hidden_size, dtype_weight, dtype_scale, text_gq,
                text_group_size, config->rms_norm_eps, 1ll * l, warm_up
            );
        }

        PtrPair k_cache_l_pair = state->key_cache->ptr_all({0, l});
        PtrPair v_cache_l_pair = state->value_cache->ptr_all({0, l});
        const char *k_cache_l = (const char *)(k_cache_l_pair.buf);
        const char *v_cache_l = (const char *)(v_cache_l_pair.buf);
        float *k_cache_s = (float *)(k_cache_l_pair.scale);
        float *v_cache_s = (float *)(v_cache_l_pair.scale);
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_qk_norm", warm_up);
            #endif
            fused_rms_rotary_q_dispatch(
                state->q, weight->w_attn_q_norm, state->cos_tensor,
                state->sin_tensor, num_heads, head_dim, 1,
                1ll * l, pos, config->rms_norm_eps, warm_up
            );
            char *k_cache_ptr = (char *)(k_cache_l + kv_pos_off_bytes);
            float *k_cache_s_ptr = state->key_cache->dtype == DType::INT8
                ? (k_cache_s + kv_pos_scale_off) : nullptr;
            fused_rms_rotary_k_dispatch(
                state->k, k_cache_ptr, k_cache_s_ptr, state->key_cache,
                weight->w_attn_k_norm, state->cos_tensor, state->sin_tensor,
                num_kv_heads, head_dim, 1, state->k->dtype,
                state->key_cache->dtype, 1ll * l, kv_all_off, pos,
                config->rms_norm_eps, cache_group_size, warm_up
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_v_cache_update", warm_up);
            #endif

            copy_to_v_cache(
                state->v, (char *)v_cache_l, v_cache_s,
                state->value_cache->dtype, 1, head_dim,
                num_kv_heads, kv_pos_off, kv_all_off,
                kv_pos_scale_off, cache_group_size, warm_up
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_attention_mechanism", warm_up);
            #endif

            fused_att_dispatch(
                k_cache_l, v_cache_l, k_cache_s, v_cache_s,
                state->q, state->att, state->qkv_out,
                num_heads, head_dim, kv_mul, kv_dim,
                kv_all_off, pos, state->key_cache->dtype,
                state->value_cache->dtype,
                cache_group_size, 1, warm_up
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_attn_out_linear", warm_up);
            #endif
            PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
            linear(
                state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale,
                nullptr, nullptr, nullptr, state->x->ptr(), 1,
                hidden_size, hidden_size, !weight->w_attn_o->permuted,
                state->qkv_out->dtype, dtype_weight, dtype_scale,
                state->x->dtype, text_gq, text_group_size, true
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->x->printDebug("x");
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_mlp_block", warm_up);
            #endif
            fused_rms_mlp_swiglu_dispatch(
                weight->rms_attn_w, weight->w_mlp_gate, weight->w_mlp_up,
                state->x, state->t, state->gate, state->up, 1,
                hidden_size, config->intermediate_size, weight->w_mlp_gate->dtype,
                weight->w_mlp_gate->scale_dtype, text_gq, config->rms_norm_eps,
                text_group_size, 1ll * l, warm_up
            );

            PtrPair w_down = weight->w_mlp_down->ptr_all({l});
            linear(
                state->gate->ptr(), w_down.buf, w_down.scale, nullptr, 
                nullptr, nullptr, state->x->ptr(), 1, hidden_size,
                config->intermediate_size, !weight->w_mlp_down->permuted,
                state->gate->dtype, dtype_weight, dtype_scale, state->x->dtype,
                text_gq, text_group_size, true
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->x->printDebug("x");
                }
            #endif
        }
    }

    size_t token;
    {        
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("decode_final_head", warm_up);
        #endif

        token = fused_rms_decode_dispatch(
            weight->rms_out_w, weight->token_embedding_table,
            state->x, state->logits, config->rms_norm_eps,
            config->vocab_size, hidden_size, weight->rms_out_w->dtype,
            weight->rms_out_w->scale_dtype, text_gq, text_group_size
        );
    }

    return token;
}

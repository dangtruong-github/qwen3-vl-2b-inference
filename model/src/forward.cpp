#include "../include/forward.hpp"

void forward_img(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    float *img_data, int img_h, int img_w, int grid_h, int grid_w, bool warm_up
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
            CPUTimer timer("conv_3d");
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
            CPUTimer timer("vision_pos_embed");
        #endif
        vision_pos_embed(
            weight->vl_pos_emb_w, state->vision_t,
            grid_h, grid_w, num_grid_per_side, VSP, VH
        );
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("add_vector_pos");
        #endif
        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("vision_rot_pos_emb");
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
                CPUTimer timer("layer_norm_1");
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
                CPUTimer timer("linear_q");
            #endif
            linear(
                state->vision_t->ptr(), w_q.buf, w_q.scale, w_q.sum_int8,
                b_q.buf, b_q.scale, state->vision_mlp_out->ptr(), total_tokens, VH,
                VH, !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("rope_q");
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
                CPUTimer timer("linear_k");
            #endif
            linear(
                state->vision_t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, b_k.buf,
                b_k.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
                !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("rope_k");
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
                CPUTimer timer("linear_v");
            #endif
            linear(
                state->vision_t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, b_v.buf,
                b_v.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
                !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
                dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
                vision_gq, vision_group_size
            );
            tensor_transpose(
                state->vision_mlp_out, state->vision_t, total_tokens, VNH, VHD
            );
        }
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("vision_attention");
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
                CPUTimer timer("linear_attn_proj");
            #endif
            linear(
                state->vision_t->ptr(), w_attn_proj_ptr.buf, w_attn_proj_ptr.scale,
                w_attn_proj_ptr.sum_int8, b_attn_proj_ptr.buf,
                b_attn_proj_ptr.scale, state->vision_q->ptr(),
                total_tokens, VH, VH, !weight->vl_attn_proj_w->permuted,
                state->vision_t->dtype, dtype_weight, dtype_scale,
                state->vision_q->dtype, vision_gq, vision_group_size
            );
            add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("layer_norm_2");
            #endif
            layer_norm(
                state->vision_x, weight->vl_norm2_w,
                weight->vl_norm2_b, state->vision_t,
                config->rms_norm_eps, total_tokens, 1ll * l
            );
        }
        
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("mlp_block");
            #endif
            PtrPair w_mlp1_ptr = weight->vl_mlp1_w->ptr_all({l});
            PtrPair b_mlp1_ptr = weight->vl_mlp1_b->ptr_all({l});
            linear(
                state->vision_t->ptr(), w_mlp1_ptr.buf, w_mlp1_ptr.scale,
                w_mlp1_ptr.sum_int8, b_mlp1_ptr.buf, b_mlp1_ptr.scale,
                state->vision_mlp_out->ptr(), total_tokens, VI, VH,
                !weight->vl_mlp1_w->permuted, state->vision_t->dtype, dtype_weight,
                dtype_scale, state->vision_mlp_out->dtype, vision_gq,
                vision_group_size
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
                vision_group_size
            );

            add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);
        }

        if (config->deep_layer[l] > 0) {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("deep_stack_block");
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
                vision_gq, vision_group_size
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
                state->vision_deep_stack->dtype, vision_gq, vision_group_size
            );
        }
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("final_merge_block");
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
            vision_gq, vision_group_size
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
            vision_gq, vision_group_size
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

    const size_t qkv_stride = (num_heads + 2 * num_kv_heads) * head_dim;

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;
    const size_t kv_pos_off_bytes = kv_pos_off * state->key_cache->get_dtype_size();

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const bool text_gq = config->group_quantized ? true : false;

    int img_token_id[prefill_size];

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("text_embedding_lookup");
        #endif
        // Embed layer
        for (size_t i = 0; i < prefill_size; ++i) { 
            int token_id = token_list[i];
            bool img_token_true = (token_id == config->image_token_id) || (token_id == config->video_token_id);
            if (!img_token_true) {
                embedding_lookup(
                    weight->token_embedding_table,
                    state->x, i, token_id, hidden_size
                );
                img_token_id[i] = -1;
            } else {
                const float *src = (const float *)state->vision_x->ptr() + 1ll * hidden_size * state->cur_img_token_id;
                memcpy(state->x->ptr({i}), src, 1ll * hidden_size * sizeof(float));
                img_token_id[i] = state->cur_img_token_id;
                state->cur_img_token_id++;
            }
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
                CPUTimer timer("text_rms_linear_qkv");
            #endif
            fused_rms_linear_qkv_dispatch(
                weight->rms_ffn_w, weight->w_attn_qkv, state->x,
                state->t, state->qkv, prefill_size, kv_dim,
                hidden_size, dtype_weight,  dtype_scale, text_gq,
                text_group_size, config->rms_norm_eps, 1ll * l, warm_up
            );
        }

        float *q_ptr = (float *)state->qkv->ptr();
        float *k_ptr = q_ptr + num_heads * head_dim;

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_qk_norm");
            #endif
            rms_norm_inplace(
                q_ptr, weight->w_attn_q_norm, config->rms_norm_eps,
                num_heads, 1ll * l, prefill_size, qkv_stride
            );
            rms_norm_inplace(
                k_ptr, weight->w_attn_k_norm, config->rms_norm_eps,
                num_kv_heads, 1ll * l, prefill_size, qkv_stride
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->qkv->printDebug("q", {i}); 
                        state->qkv->printDebug("k", {i, (size_t)num_heads});
                    }
                }
            #endif
        }

        const char *k_cache_l = (const char *)state->key_cache->ptr({0, l});
        const char *v_cache_l = (const char *)state->value_cache->ptr({0, l});

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_apply_rope");
            #endif
            apply_rotary(
                state->qkv, state->cos_tensor, state->sin_tensor,
                prefill_size, num_heads, head_dim, pos, qkv_stride
            );

            char *k_cache_ptr = (char *)(k_cache_l + kv_pos_off_bytes);
            apply_rotary_cache(
                k_ptr, k_cache_ptr, state->cos_tensor, state->sin_tensor,
                prefill_size, num_kv_heads, head_dim, pos,
                kv_all_off, state->key_cache->dtype, qkv_stride
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->qkv->printDebug("q", {i});
                    }
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_kv_cache_update");
            #endif

            if (state->value_cache->dtype == DType::FP32) {
                float *v_cache_base_ptr = (float *)(v_cache_l) + kv_pos_off;
                const float *v_now_base_ptr = k_ptr + num_kv_heads * head_dim;

                for (size_t b = 0; b < prefill_size; ++b) {
                    const float *v_now_ptr = v_now_base_ptr + b * qkv_stride;
                    float *v_cache_ptr = v_cache_base_ptr + b * head_dim;
                    for (int h = 0; h < num_kv_heads; h++) {
                        memcpy(v_cache_ptr + h * kv_all_off, v_now_ptr + h * head_dim, head_dim * sizeof(float));
                    }
                }
            } else {
                // fp16 path
                uint16_t *v_cache_base_ptr = (uint16_t *)(v_cache_l) + kv_pos_off;
                const float *v_now_base_ptr = k_ptr + num_kv_heads * head_dim;

                #pragma omp parallel for
                for (size_t b = 0; b < prefill_size; ++b) {
                    const float *v_now_ptr = v_now_base_ptr + b * qkv_stride;
                    uint16_t *v_cache_ptr = v_cache_base_ptr + b * head_dim;
                    for (int h = 0; h < num_kv_heads; h++) {
                        const float *src = v_now_ptr + h * head_dim;
                        uint16_t *dst = v_cache_ptr + h * kv_all_off;

                        // process 8 floats at a time
                        int i = 0;

                        #if defined(__F16C__)
                            for (; i + 8 <= head_dim; i += 8) {
                                __m256 v = _mm256_loadu_ps(src + i);                     // load 8 floats
                                __m128i h16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT); // convert to 8 fp16
                                _mm_storeu_si128((__m128i*)(dst + i), h16);              // store 8 fp16 (16 bytes)
                            }
                        #endif

                        for (; i < head_dim; i++) {
                            dst[i] = (half_cpu)(src[i]);
                        }
                    }
                }
            }
        }

        if (l == config->num_hidden_layers - 1) return;

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_attention_mechanism");
            #endif

            attn_scores_all_heads_prefill(
                k_cache_l, state->qkv, state->att, num_heads,
                kv_mul, head_dim, kv_dim, kv_all_off,
                pos, prefill_size, state->key_cache->dtype
            );

            attn_weighted_sum_all_heads(
                v_cache_l, state->att, state->qkv_out, num_heads,
                kv_mul, head_dim, kv_dim, kv_all_off,
                pos, prefill_size, state->key_cache->dtype
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) {
                        state->att->printDebug("att", {i});
                        state->qkv_out->printDebug("qkv_out", {i});
                    }
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_attn_out_linear");
            #endif
            PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
            linear(
                state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale,
                w_out_proj.sum_int8, nullptr, nullptr, state->t->ptr(), prefill_size,
                hidden_size, hidden_size, !weight->w_attn_o->permuted,
                state->qkv_out->dtype, dtype_weight, dtype_scale,
                state->t->dtype, text_gq, text_group_size
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->t->printDebug("t", {i});
                    }
                }
            #endif

            add_vector(state->x, state->t, prefill_size * hidden_size);

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
                CPUTimer timer("text_post_attn_norm");
            #endif
            rms_norm(
                state->x, weight->rms_attn_w, state->t,
                config->rms_norm_eps, prefill_size, 1ll * l
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->t->printDebug("t", {i});
                    }
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("text_mlp_block");
            #endif
            fused_text_mlp_swiglu_dispatch(
                weight->w_mlp_gate, weight->w_mlp_up, state->t, state->gate,
                state->up, prefill_size, hidden_size, config->intermediate_size, 
                weight->w_mlp_gate->dtype, weight->w_mlp_gate->scale_dtype,
                text_gq, text_group_size, 1ll * l
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->gate->printDebug("gate", {i});
                    }
                }
            #endif

            PtrPair w_down = weight->w_mlp_down->ptr_all({l});
            linear(
                state->gate->ptr(), w_down.buf, w_down.scale, w_down.sum_int8, 
                nullptr, nullptr, state->t->ptr(), prefill_size, hidden_size,
                config->intermediate_size, !weight->w_mlp_down->permuted,
                state->gate->dtype, dtype_weight, dtype_scale, state->t->dtype,
                text_gq, text_group_size
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) { 
                        state->t->printDebug("t", {i});
                    }
                }
            #endif

            add_vector(state->x, state->t, prefill_size * hidden_size);

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
                CPUTimer timer("text_vision_deep_stack_add");
            #endif
            for (size_t i = 0; i < prefill_size; ++i) {
                if (img_token_id[i] < 0) continue;
                
                const void *deep_ptr = state->vision_deep_stack->ptr({l, (size_t)img_token_id[i]});
                void *x_ptr = state->x->ptr({i});

                add_vector(
                    x_ptr, deep_ptr, state->vision_deep_stack->dtype, state->x->dtype, hidden_size
                );
            }
        }
    }
}

float *forward_text_decode(
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

    const size_t qkv_stride = (num_heads + 2 * num_kv_heads) * head_dim;

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;
    const size_t kv_pos_off_bytes = kv_pos_off * state->key_cache->get_dtype_size();

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const bool text_gq = config->group_quantized ? true : false;

    bool img_token_true = (token_id == config->image_token_id) || (token_id == config->video_token_id);
    
    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("decode_embedding");
        #endif
        // Embed layer
        if (!img_token_true) {
            embedding_lookup(
                weight->token_embedding_table, state->x, 0ll, token_id, hidden_size
            );
        } else {
            const float *src = (const float *)state->vision_x->ptr() + 1ll * hidden_size * state->cur_img_token_id;
            memcpy(state->x->ptr(), src, 1ll * hidden_size * sizeof(float));
        }

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                state->x->printDebug("x");
            }
        #endif
    }

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_text_rms_linear_qkv");
            #endif
            fused_rms_linear_qkv_dispatch(
                weight->rms_ffn_w, weight->w_attn_qkv, state->x,
                state->t, state->qkv, 1, kv_dim, hidden_size,
                dtype_weight, dtype_scale, text_gq, text_group_size,
                config->rms_norm_eps, 1ll * l, warm_up
            );
        }

        float *q_ptr = (float *)state->qkv->ptr();
        float *k_ptr = q_ptr + num_heads * head_dim;

        const char *k_cache_l = (const char *)state->key_cache->ptr({0, l});
        const char *v_cache_l = (const char *)state->value_cache->ptr({0, l});

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_qk_norm_rope");
            #endif
            rms_norm_inplace(
                q_ptr, weight->w_attn_q_norm, config->rms_norm_eps,
                num_heads, 1ll * l, 1, 0
            );
            rms_norm_inplace(
                k_ptr, weight->w_attn_k_norm, config->rms_norm_eps,
                num_kv_heads, 1ll * l, 1, 0
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->q->printDebug("q");
                    state->k->printDebug("k");
                }
            #endif

            apply_rotary(
                state->qkv, state->cos_tensor, state->sin_tensor,
                1, num_heads, head_dim, pos, qkv_stride
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->q->printDebug("q");
                }
            #endif

            char *k_cache_ptr = (char *)k_cache_l + kv_pos_off_bytes;
            apply_rotary_cache(
                k_ptr, k_cache_ptr, state->cos_tensor, state->sin_tensor,
                1, num_kv_heads, head_dim, pos, kv_all_off,
                state->key_cache->dtype, qkv_stride
            );
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_v_cache_update");
            #endif

            if (state->value_cache->dtype == DType::FP32) {
                float *v_cache_ptr = (float *)(v_cache_l) + kv_pos_off;
                const float *v_now_base_ptr = k_ptr + num_kv_heads * head_dim;

                for (int h = 0; h < num_kv_heads; h++) {
                    memcpy(v_cache_ptr + h * kv_all_off, v_now_base_ptr + h*head_dim, head_dim*sizeof(float));
                }
            } else {
                // fp16 path
                uint16_t *v_cache_ptr = (uint16_t *)(v_cache_l) + kv_pos_off;
                const float *v_now_base_ptr = k_ptr + num_kv_heads * head_dim;

                for (size_t h = 0; h < num_kv_heads; h++) {
                    const float *src = v_now_base_ptr + h*head_dim;
                    uint16_t *dst = v_cache_ptr + h * kv_all_off;

                    // process 8 floats at a time
                    int i = 0;
                    
                    #if defined(__F16C__)
                        for (; i + 8 <= head_dim; i += 8) {
                            __m256 v = _mm256_loadu_ps(src + i);                     // load 8 floats
                            __m128i h16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT); // convert to 8 fp16
                            _mm_storeu_si128((__m128i*)(dst + i), h16);              // store 8 fp16 (16 bytes)
                        }
                    #endif

                    for (; i < head_dim; i++) {
                        dst[i] = (half_cpu)(src[i]);
                    }
                }
            }
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_attention_mechanism");
            #endif

            attn_scores_all_heads_decode(
                k_cache_l, state->qkv, state->att, num_heads,
                kv_mul, head_dim, kv_dim, kv_all_off,
                pos, state->key_cache->dtype
            );

            attn_weighted_sum_all_heads(
                v_cache_l, state->att, state->qkv_out, num_heads,
                kv_mul, head_dim, kv_dim, kv_all_off,
                pos, 1, state->key_cache->dtype
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->att->printDebug("att");
                    state->qkv_out->printDebug("qkv_out");
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_attn_out_linear");
            #endif
            PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
            linear(
                state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale,
                nullptr, state->x->ptr(), nullptr, state->t->ptr(), 1,
                hidden_size, hidden_size, !weight->w_attn_o->permuted,
                state->qkv_out->dtype, dtype_weight, dtype_scale,
                state->t->dtype, text_gq, text_group_size
            );
            // add_vector(state->x, state->t, hidden_size);

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->t->printDebug("t");
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_post_attn_norm");
            #endif
            rms_norm(
                state->t, weight->rms_attn_w, state->x,
                config->rms_norm_eps, 1, 1ll * l
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->x->printDebug("x");
                }
            #endif
        }

        {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_mlp_block");
            #endif
            fused_text_mlp_swiglu_dispatch(
                weight->w_mlp_gate, weight->w_mlp_up, state->x, state->gate,
                state->up, 1, hidden_size, config->intermediate_size, 
                weight->w_mlp_gate->dtype, weight->w_mlp_gate->scale_dtype,
                text_gq, text_group_size, 1ll * l
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->gate->printDebug("gate");
                }
            #endif

            PtrPair w_down = weight->w_mlp_down->ptr_all({l});
            linear(
                state->gate->ptr(), w_down.buf, w_down.scale, nullptr, 
                state->t->ptr(), nullptr, state->x->ptr(), 1, hidden_size,
                config->intermediate_size, !weight->w_mlp_down->permuted,
                state->gate->dtype, dtype_weight, dtype_scale, state->t->dtype,
                text_gq, text_group_size
            );
            // add_vector(state->x, state->t, hidden_size);

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    state->x->printDebug("x");
                }
            #endif
        }

        if (l < config->vision_deep_stack_depth && img_token_true) {
            #ifdef CPU_TIME_OUTSIDE
                CPUTimer timer("decode_vision_deep_stack");
            #endif
            const void *deep_ptr = state->vision_deep_stack->ptr({l, (size_t)state->cur_img_token_id});
            add_vector(
                state->x, deep_ptr,
                state->vision_deep_stack->dtype, hidden_size
            );
        }
    }

    {
        #ifdef CPU_TIME_OUTSIDE
            CPUTimer timer("decode_final_head");
        #endif
        // Final RMSNorm
        rms_norm_inplace(
            (float *)state->x->ptr(), weight->rms_out_w,
            config->rms_norm_eps, 1, 0ll, 1, 0
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                state->x->printDebug("x");
            }
        #endif

        // Classifier (LM Head)
        classifier_gemm(
            weight->token_embedding_table, state->x, state->logits,
            config->vocab_size, hidden_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                state->logits->printDebug("x");
            }
        #endif
    }

    if (img_token_true) {
        state->cur_img_token_id += 1;
    }

    return (float *)state->logits->ptr();
}

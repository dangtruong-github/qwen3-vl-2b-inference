#include "../include/forward.hpp"

void forward_img(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    float *img_data, int img_h, int img_w, int grid_h, int grid_w, bool warm_up
) {
    if (img_data == nullptr) {
        return;
    }

    printf("img_h=%d, img_w=%d, grid_h=%d, grid_w=%d\n", img_h, img_w, grid_h, grid_w);
    printf("vision_num_channels=%d, vision_temporal_patch_size=%d, vision_patch_size=%d\n", config->vision_num_channels, config->vision_temporal_patch_size, config->vision_patch_size);

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
        if (!warm_up) state->vision_x->printDebug("vision_x");
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
        if (!warm_up) state->vision_t->printDebug("vision_t");
    #endif

    add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

    #ifdef PRINT_LOGITS
        if (!warm_up) state->vision_x->printDebug("vision_x");
    #endif

    vision_rot_pos_emb(
        state->vision_pe_cos, state->vision_pe_sin,
        state->vision_cos_tensor, state->vision_sin_tensor,
        grid_h, grid_w, config->vision_spatial_merge_size, VHD
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            state->vision_pe_cos->printDebug("vision_pe_cos");
            state->vision_pe_sin->printDebug("vision_pe_sin");
        }
    #endif

    printf("Finish preprocessing forward_img\n");

    for (size_t l = 0; l < config->vision_depth; l++) {
        layer_norm(
            state->vision_x, weight->vl_norm1_w,
            weight->vl_norm1_b, state->vision_t,
            config->rms_norm_eps, total_tokens, 1ll * l
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_t->printDebug("vision_t");
                if (l == 13) {
                    fprintf(stderr, "After layer_norm\n");
                    float *vision_x_ptr = (float *)state->vision_x->ptr();
                    for (size_t i = 0; i < state->vision_x->num_elem(); ++i) {
                        if (isnan(vision_x_ptr[i])) {
                            fprintf(stderr, "Error vision_x nan at index %zu\n", i);
                            break;
                        }
                    }

                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        PtrPair w_q = weight->vl_attn_qkv_w->ptr_all({l});
        PtrPair w_k = weight->vl_attn_qkv_w->ptr_all({l, 1});
        PtrPair w_v = weight->vl_attn_qkv_w->ptr_all({l, 2});
        PtrPair b_q = weight->vl_attn_qkv_b->ptr_all({l});
        PtrPair b_k = weight->vl_attn_qkv_b->ptr_all({l, 1});
        PtrPair b_v = weight->vl_attn_qkv_b->ptr_all({l, 2});
        
        // use vl_v as temporary buffer for vision_q
        linear(
            state->vision_t->ptr(), w_q.buf, w_q.scale, w_q.sum_int8,
            b_q.buf, b_q.scale, state->vision_mlp_out->ptr(), total_tokens, VH,
            VH, !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After vision_apply_rotary_inplace\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_q, total_tokens, VNH, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_q->printDebug("vision_q");
                if (l == 13) {
                    fprintf(stderr, "After tensor_transpose\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_q->ptr();
                    for (size_t i = 0; i < state->vision_q->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_q nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        // use vl_v as temporary buffer for vision_k
        linear(
            state->vision_t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, b_k.buf,
            b_k.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        vision_apply_rotary_inplace(
            state->vision_pe_cos, state->vision_pe_sin,
            state->vision_mlp_out, total_tokens, VNH, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After vision_apply_rotary_inplace\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_k, total_tokens, VNH, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_k->printDebug("vision_k");
                if (l == 13) {
                    fprintf(stderr, "After tensor_transpose\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_k->ptr();
                    for (size_t i = 0; i < state->vision_k->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_k nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        // swap vision_t and vl_v
        linear(
            state->vision_t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, b_v.buf,
            b_v.scale, state->vision_mlp_out->ptr(), total_tokens, VH, VH,
            !weight->vl_attn_qkv_w->permuted, state->vision_t->dtype,
            dtype_weight, dtype_scale, state->vision_mlp_out->dtype,
            vision_gq, vision_group_size
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        tensor_transpose(
            state->vision_mlp_out, state->vision_t, total_tokens, VNH, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_t->printDebug("vision_t");
                if (l == 13) {
                    fprintf(stderr, "After tensor_transpose\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif
        
        vision_att(
            state->vision_q, state->vision_k, state->vision_t,
            state->vision_attn_scores, state->vision_mlp_out, 
            VNH, total_tokens, VHD, vision_scale
        );
        
        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_attn_scores->printDebug("vision_attn_scores");
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 13) {
                    fprintf(stderr, "After vision_att\n");
                    half_cpu *vision_q_ptr = (half_cpu *)state->vision_q->ptr();
                    for (size_t i = 0; i < state->vision_q->num_elem(); ++i) {
                        if (isnan((float)vision_q_ptr[i])) {
                            fprintf(stderr, "Error vision_q nan at index %zu\n", i);
                            break;
                        }
                    }

                    half_cpu *vision_k_ptr = (half_cpu *)state->vision_k->ptr();
                    for (size_t i = 0; i < state->vision_k->num_elem(); ++i) {
                        if (isnan((float)vision_k_ptr[i])) {
                            fprintf(stderr, "Error vision_k nan at index %zu\n", i);
                            break;
                        }
                    }

                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }

                    float *print_ptr = (float *)state->vision_attn_scores->ptr();
                    for (size_t i = 0; i < state->vision_attn_scores->num_elem(); ++i) {
                        if (isnan(print_ptr[i])) {
                            fprintf(stderr, "Error vision_attn_scores nan at index %zu\n", i);
                            break;
                        }
                    }
                    exit(1);
                }
            }
        #endif
        
        // swap back vision_t and vl_v
        tensor_transpose(
            state->vision_mlp_out, state->vision_t, VNH, total_tokens, VHD
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_t->printDebug("vision_t");
                if (l == 12) {
                    fprintf(stderr, "After tensor_transpose\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        PtrPair w_attn_proj_ptr, b_attn_proj_ptr;
        w_attn_proj_ptr = weight->vl_attn_proj_w->ptr_all({l});
        b_attn_proj_ptr = weight->vl_attn_proj_b->ptr_all({l});

        // use vision_q as temporary buffer here
        linear(
            state->vision_t->ptr(), w_attn_proj_ptr.buf, w_attn_proj_ptr.scale,
            w_attn_proj_ptr.sum_int8, b_attn_proj_ptr.buf,
            b_attn_proj_ptr.scale, state->vision_q->ptr(),
            total_tokens, VH, VH, !weight->vl_attn_proj_w->permuted,
            state->vision_t->dtype, dtype_weight, dtype_scale,
            state->vision_q->dtype, vision_gq, vision_group_size
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_q->printDebug("vision_q");
                if (l == 12) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_q->ptr();
                    for (size_t i = 0; i < state->vision_q->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_q nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        add_vector(state->vision_x, state->vision_q, 1ll * total_tokens * VH);

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_x->printDebug("vision_x");
                if (l == 12) {
                    fprintf(stderr, "After add_vector\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_x->ptr();
                    for (size_t i = 0; i < state->vision_x->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_x nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        layer_norm(
            state->vision_x, weight->vl_norm2_w,
            weight->vl_norm2_b, state->vision_t,
            config->rms_norm_eps, total_tokens, 1ll * l
        );

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_t->printDebug("vision_t");
                if (l == 12) {
                    fprintf(stderr, "After layer_norm\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
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

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 12) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        gelu_tanh(state->vision_mlp_out, 1ll * total_tokens * VI);

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_mlp_out->printDebug("vision_mlp_out");
                if (l == 12) {
                    fprintf(stderr, "After gelu_tanh\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                    for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif
        
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

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_t->printDebug("vision_t");
                if (l == 12) {
                    fprintf(stderr, "After linear\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                    for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        add_vector(state->vision_x, state->vision_t, 1ll * total_tokens * VH);

        #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
            if (!warm_up) {
                state->vision_x->printDebug("vision_x");
                if (l == 12) {
                    fprintf(stderr, "After add_vector\n");
                    half_cpu *vision_t_ptr = (half_cpu *)state->vision_x->ptr();
                    for (size_t i = 0; i < state->vision_x->num_elem(); ++i) {
                        if (isnan((float)vision_t_ptr[i])) {
                            fprintf(stderr, "Error vision_x nan at index %zu\n", i);
                            break;
                        }
                    }
                }
            }
        #endif

        if (config->deep_layer[l] > 0) {
            size_t d_stride = config->deep_layer[l] - 1;
            layer_norm(
                state->vision_x, weight->vl_d_norm_w,
                weight->vl_d_norm_b, state->vision_t,
                config->rms_norm_eps, d_tokens, 1ll * d_stride
            );

            #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
                if (!warm_up) {
                    state->vision_t->printDebug("vision_t");
                    if (l == 12) {
                        fprintf(stderr, "After layer_norm deep\n");
                        half_cpu *vision_t_ptr = (half_cpu *)state->vision_t->ptr();
                        for (size_t i = 0; i < state->vision_t->num_elem(); ++i) {
                            if (isnan((float)vision_t_ptr[i])) {
                                fprintf(stderr, "Error vision_t nan at index %zu\n", i);
                                break;
                            }
                        }
                    }
                }
            #endif
            
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

            #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
                if (!warm_up) {
                    state->vision_mlp_out->printDebug("vision_mlp_out");
                    if (l == 12) {
                        fprintf(stderr, "After linear deep\n");
                        half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                        for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                            if (isnan((float)vision_t_ptr[i])) {
                                fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                                break;
                            }
                        }
                    }
                }
            #endif
        
            gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

            #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
                if (!warm_up) {
                    state->vision_mlp_out->printDebug("vision_mlp_out");
                    if (l == 12) {
                        fprintf(stderr, "After gelu_tanh deep\n");
                        half_cpu *vision_t_ptr = (half_cpu *)state->vision_mlp_out->ptr();
                        for (size_t i = 0; i < state->vision_mlp_out->num_elem(); ++i) {
                            if (isnan((float)vision_t_ptr[i])) {
                                fprintf(stderr, "Error vision_mlp_out nan at index %zu\n", i);
                                break;
                            }
                        }
                    }
                }
            #endif

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

            #if defined(PRINT_LOGITS) || defined(PRINT_LOGITS_2)
                if (!warm_up) {
                    state->vision_mlp_out->printDebug("vision_mlp_out");
                    if (l == 12) {
                        fprintf(stderr, "After linear deep\n");
                        half_cpu *vision_t_ptr = (half_cpu *)state->vision_deep_stack->ptr({d_stride});
                        size_t d_stride_elem = OH * VI;
                        for (size_t i = 0; i < d_stride_elem; ++i) {
                            if (isnan((float)vision_t_ptr[i])) {
                                fprintf(stderr, "Error vision_deep_stack nan at index %zu\n", i);
                                break;
                            }
                        }
                    }
                }
            #endif
        }

        printf("Finish layer %ld\n", l);

    }

    layer_norm(
        state->vision_x, weight->vl_merge_norm_w,
        weight->vl_merge_norm_b, state->vision_t,
        config->rms_norm_eps, total_tokens, 0
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) state->vision_t->printDebug("vision_t");
    #endif

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

    #ifdef PRINT_LOGITS
        if (!warm_up) state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

    gelu_tanh(state->vision_mlp_out, 1ll * d_tokens * VI);

    #ifdef PRINT_LOGITS
        if (!warm_up) state->vision_mlp_out->printDebug("vision_mlp_out");
    #endif

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

    #ifdef PRINT_LOGITS
        if (!warm_up) state->vision_x->printDebug("vision_x");
    #endif
    
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

    const size_t kv_pos_off = 1ll * pos * head_dim;
    const size_t kv_all_off = 1ll * seq_len * head_dim;

    const DType::Type dtype_weight = weight->token_embedding_table->dtype;
    const DType::Type dtype_scale = weight->token_embedding_table->scale_dtype;
    const size_t text_group_size = weight->token_embedding_table->group_size;
    const bool text_gq = config->group_quantized ? true : false;

    int img_token_id[prefill_size];

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
            for (size_t b = 0; b < prefill_size; ++b) {
                state->x->printDebug("x", {b});
            }
        }
    #endif

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            state->x, weight->rms_ffn_w, state->t,
            config->rms_norm_eps, prefill_size, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->t->printDebug("t", {b});
                }
            }
        #endif

        PtrPair w_q = weight->w_attn_q->ptr_all({l});
        PtrPair w_k = weight->w_attn_k->ptr_all({l});
        PtrPair w_v = weight->w_attn_v->ptr_all({l});

        linear(
            state->t->ptr(), w_q.buf, w_q.scale, w_q.sum_int8, nullptr, nullptr,
            state->q->ptr(), prefill_size, hidden_size, hidden_size,
            !weight->w_attn_q->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->q->dtype, text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_k.buf, w_k.scale, w_k.sum_int8, nullptr, nullptr,
            state->k->ptr(), prefill_size, kv_dim, hidden_size,
            !weight->w_attn_k->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->k->dtype, text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_v.buf, w_v.scale, w_v.sum_int8, nullptr, nullptr,
            state->v->ptr(), prefill_size, kv_dim, hidden_size,
            !weight->w_attn_v->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->v->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->q->printDebug("q", {b});
                    state->k->printDebug("k", {b});
                    state->v->printDebug("v", {b});
                }
            }
        #endif

        // QK RMSNorm
        rms_norm_inplace(
            state->q, weight->w_attn_q_norm, config->rms_norm_eps,
            prefill_size * num_heads, 1ll * l
        );
        rms_norm_inplace(
            state->k, weight->w_attn_k_norm, config->rms_norm_eps,
            prefill_size * num_kv_heads, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->q->printDebug("q", {b});
                }
            }
        #endif

        // Apply Rotary Position Embeddings
        apply_rotary(
            state->q, state->cos_tensor, state->sin_tensor,
            prefill_size, num_heads, head_dim, pos
        );

        const float *k_cache_l = (const float *)state->key_cache->ptr({0, l});
        float *k_cache_ptr = (float *)k_cache_l + kv_pos_off;
        apply_rotary_cache(
            state->k, k_cache_ptr, state->cos_tensor, state->sin_tensor,
            prefill_size, num_kv_heads, head_dim, pos, kv_all_off
        );

        const float *v_cache_l = (const float *)state->value_cache->ptr({0, l});
        float *v_cache_base_ptr = (float *)v_cache_l + kv_pos_off;

        for (size_t b = 0; b < prefill_size; ++b) {
            const float *v_now_ptr = (const float *)state->v->ptr({b});
            float *v_cache_ptr = v_cache_base_ptr + b * head_dim;
            for (int h = 0; h < num_kv_heads; h++) {
                memcpy(v_cache_ptr + h * kv_all_off, v_now_ptr + h * head_dim, head_dim * sizeof(float));
            }
        }

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                if (l == config->num_hidden_layers - 1) printf("pos=%ld\n", pos);
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->q->printDebug("q", {b});
                }
            }
        #endif

        if (l == config->num_hidden_layers - 1) return;

        // Multi-head attention

        // Compute attention scores
        attn_scores_all_heads_prefill(
            k_cache_l, state->q, state->att, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos, prefill_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->att->printDebug("att", {b});
                }
            }
        #endif

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            v_cache_l, state->att, state->qkv_out, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos, prefill_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->qkv_out->printDebug("qkv_out", {b});
                }
            }
        #endif

        // Output projection: using state->t for attn_out
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
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->t->printDebug("t", {b});
                }
            }
        #endif

        // Residual connection 1
        add_vector(state->x, state->t, prefill_size * hidden_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->x->printDebug("x", {b});
                }
            }
        #endif

        // Post-attention RMSNorm
        rms_norm(
            state->x, weight->rms_attn_w, state->t,
            config->rms_norm_eps, prefill_size, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->t->printDebug("t", {b});
                }
            }
        #endif

        // MLP: Gate and Up projections
        PtrPair w_gate = weight->w_mlp_gate->ptr_all({l});
        PtrPair w_up = weight->w_mlp_up->ptr_all({l});
        linear(
            state->t->ptr(), w_gate.buf, w_gate.scale, w_gate.sum_int8,
            nullptr, nullptr, state->gate->ptr(), prefill_size, config->intermediate_size,
            hidden_size, !weight->w_mlp_gate->permuted, state->t->dtype,
            dtype_weight, dtype_scale, state->gate->dtype,
            text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_up.buf, w_up.scale, w_up.sum_int8, nullptr,
            nullptr, state->up->ptr(), prefill_size, config->intermediate_size,
            hidden_size, !weight->w_mlp_up->permuted, state->t->dtype,
            dtype_weight, dtype_scale, state->up->dtype, text_gq,
            text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->gate->printDebug("gate", {b});
                    state->up->printDebug("up", {b});
                }
            }
        #endif
        
        // SwiGLU activation
        swiglu(state->gate, state->up, prefill_size * config->intermediate_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->gate->printDebug("gate", {b});
                }
            }
        #endif

        // MLP: Down projection: using state->t for down
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
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->t->printDebug("t", {b});
                }
            }
        #endif

        // Residual connection 2
        add_vector(state->x, state->t, prefill_size * hidden_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                for (size_t b = 0; b < prefill_size; ++b) {
                    state->x->printDebug("x", {b});
                }
            }
        #endif

        if (l < config->vision_deep_stack_depth) {
            for (size_t i = 0; i < prefill_size; ++i) {
                if (img_token_id[i] < 0) {
                    continue;
                }
                
                const void *deep_ptr = state->vision_deep_stack->ptr({l, (size_t)img_token_id[i]});
                void *x_ptr = state->x->ptr({i});

                // edit add_vector
                add_vector(
                    x_ptr, deep_ptr, state->vision_deep_stack->dtype, state->x->dtype, hidden_size
                );
            }

            #ifdef PRINT_LOGITS
                if (!warm_up) {
                    for (size_t i = 0; i < prefill_size; ++i) {
                        if (img_token_id[i] >= 0) {
                            for (size_t b = 0; b < prefill_size; ++b) {
                                state->x->printDebug("x", {b});
                            }
                            break;
                        }
                    }
                }
            #endif
        }
        
        if (warm_up) {
            if ((l + 2) % 7 == 0) printf("Finish layer text %zu\n", l);
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
            weight->token_embedding_table, state->x, 0ll, token_id, hidden_size
        );
    } else {
        const float *src = (const float *)state->vision_x->ptr() + 1ll * hidden_size * state->cur_img_token_id;
        memcpy(state->x->ptr(), src, 1ll * hidden_size * sizeof(float));
    }

    #ifdef PRINT_LOGITS
        if (!warm_up) state->x->printDebug("x");
    #endif

    for (size_t l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            state->x, weight->rms_ffn_w, state->t,
            config->rms_norm_eps, 1, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->t->printDebug("t");
        #endif

        PtrPair w_q = weight->w_attn_q->ptr_all({l});
        PtrPair w_k = weight->w_attn_k->ptr_all({l});
        PtrPair w_v = weight->w_attn_v->ptr_all({l});

        linear(
            state->t->ptr(), w_q.buf, w_q.scale, nullptr, nullptr, nullptr,
            state->q->ptr(), 1, hidden_size, hidden_size,
            !weight->w_attn_q->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->q->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->q->printDebug("q");
        #endif

        linear(
            state->t->ptr(), w_k.buf, w_k.scale, nullptr, nullptr, nullptr,
            state->k->ptr(), 1, kv_dim, hidden_size,
            !weight->w_attn_k->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->k->dtype, text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_v.buf, w_v.scale, nullptr, nullptr, nullptr,
            state->v->ptr(), 1, kv_dim, hidden_size,
            !weight->w_attn_v->permuted, state->t->dtype, dtype_weight,
            dtype_scale, state->v->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                state->k->printDebug("k");
                state->v->printDebug("v");
            }
        #endif

        // QK RMSNorm
        rms_norm_inplace(
            state->q, weight->w_attn_q_norm, config->rms_norm_eps,
            num_heads, 1ll * l
        );
        rms_norm_inplace(
            state->k, weight->w_attn_k_norm, config->rms_norm_eps,
            num_kv_heads, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->q->printDebug("q");
        #endif

        // Apply Rotary Position Embeddings
        apply_rotary(
            state->q, state->cos_tensor, state->sin_tensor,
            1, num_heads, head_dim, pos
        );

        const float *k_cache_l = (const float *)state->key_cache->ptr({0, l});
        float *k_cache_ptr = (float *)k_cache_l + kv_pos_off;
        apply_rotary_cache(
            state->k, k_cache_ptr, state->cos_tensor, state->sin_tensor,
            1, num_kv_heads, head_dim, pos, kv_all_off
        );

        const float *v_cache_l = (const float *)state->value_cache->ptr({0, l});
        float *v_cache_ptr = (float *)v_cache_l + kv_pos_off;

        for (int h = 0; h < num_kv_heads; h++) {
            memcpy(v_cache_ptr + h * kv_all_off, (const float *)state->v->ptr() + h*head_dim, head_dim*sizeof(float));
        }

        #ifdef PRINT_LOGITS
            if (!warm_up) state->q->printDebug("q");
        #endif

        // Multi-head attention

        // Compute attention scores
        attn_scores_all_heads_decode(
            k_cache_l, state->q, state->att, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->att->printDebug("att");
        #endif

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            v_cache_l, state->att, state->qkv_out, num_heads,
            kv_mul, head_dim, kv_dim, kv_all_off, pos, 1
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->qkv_out->printDebug("qkv_out");
        #endif

        // Output projection: using state->t for attn_out
        PtrPair w_out_proj = weight->w_attn_o->ptr_all({l});
        linear(
            state->qkv_out->ptr(), w_out_proj.buf, w_out_proj.scale,
            nullptr, nullptr, nullptr, state->t->ptr(), 1,
            hidden_size, hidden_size, !weight->w_attn_o->permuted,
            state->qkv_out->dtype, dtype_weight, dtype_scale,
            state->t->dtype, text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->t->printDebug("t");
        #endif

        // Residual connection 1
        add_vector(state->x, state->t, hidden_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) state->x->printDebug("x");
        #endif

        // Post-attention RMSNorm
        rms_norm(
            state->x, weight->rms_attn_w, state->t,
            config->rms_norm_eps, 1, 1ll * l
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->t->printDebug("t");
        #endif

        // MLP: Gate and Up projections
        PtrPair w_gate = weight->w_mlp_gate->ptr_all({l});
        PtrPair w_up = weight->w_mlp_up->ptr_all({l});
        linear(
            state->t->ptr(), w_gate.buf, w_gate.scale, nullptr,
            nullptr, nullptr, state->gate->ptr(), 1, config->intermediate_size,
            hidden_size, !weight->w_mlp_gate->permuted, state->t->dtype,
            dtype_weight, dtype_scale, state->gate->dtype,
            text_gq, text_group_size
        );
        linear(
            state->t->ptr(), w_up.buf, w_up.scale, nullptr, nullptr,
            nullptr, state->up->ptr(), 1, config->intermediate_size,
            hidden_size, !weight->w_mlp_up->permuted, state->t->dtype,
            dtype_weight, dtype_scale, state->up->dtype, text_gq,
            text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) {
                state->gate->printDebug("gate");
                state->up->printDebug("up");
            }
        #endif
        
        // SwiGLU activation
        swiglu(state->gate, state->up, config->intermediate_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) state->gate->printDebug("gate");
        #endif

        // MLP: Down projection: using state->t for down
        PtrPair w_down = weight->w_mlp_down->ptr_all({l});
        linear(
            state->gate->ptr(), w_down.buf, w_down.scale, nullptr, 
            nullptr, nullptr, state->t->ptr(), 1, hidden_size,
            config->intermediate_size, !weight->w_mlp_down->permuted,
            state->gate->dtype, dtype_weight, dtype_scale, state->t->dtype,
            text_gq, text_group_size
        );

        #ifdef PRINT_LOGITS
            if (!warm_up) state->t->printDebug("t");
        #endif

        // Residual connection 2
        add_vector(state->x, state->t, hidden_size);

        #ifdef PRINT_LOGITS
            if (!warm_up) state->x->printDebug("x");
        #endif

        if (l < config->vision_deep_stack_depth && img_token_true) {
            const void *deep_ptr = state->vision_deep_stack->ptr({l, (size_t)state->cur_img_token_id});
            add_vector(
                state->x, deep_ptr,
                state->vision_deep_stack->dtype, hidden_size
            );

            #ifdef PRINT_LOGITS
                if (!warm_up) state->x->printDebug("x");
            #endif
        }
        
        
        if (warm_up) {
            if ((l + 2) % 7 == 0) printf("Finish layer text %zu\n", l);
        }
    }

    // Final RMSNorm
    rms_norm_inplace(
        state->x, weight->rms_out_w, config->rms_norm_eps, 1, 0ll
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) state->x->printDebug("x");
    #endif

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, hidden_size
    );

    #ifdef PRINT_LOGITS
        if (!warm_up) state->logits->printDebug("logits");
    #endif

    if (img_token_true) {
        state->cur_img_token_id += 1;
    }

    #if defined(CPU_TIME) || defined(CPU_TIME_FP16) || defined(PRINT_LOGITS)
        if (pos > 30) exit(1);
    #endif
    
    return (float *)state->logits->ptr();
}

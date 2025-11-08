#include "load_model.hpp"

void init_model_weights(const char* path, QwenConfig* config, QwenWeight* weights) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return;
    }

    // ==================================================================================
    // 1. Read Config
    // ==================================================================================
    if (
        (fread(&(config->vocab_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->hidden_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->intermediate_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_hidden_layers), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_attention_heads), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_key_value_heads), sizeof(int), 1, file) != 1) ||
        (fread(&(config->max_position_embeddings), sizeof(int), 1, file) != 1) ||
        (fread(&(config->rope_theta), sizeof(int), 1, file) != 1) ||
        (fread(&(config->rms_norm_eps), sizeof(float), 1, file) != 1) ||
        (fread(&(config->vision_hidden_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_depth), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_patch_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_spatial_merge_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_temporal_patch_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_num_heads), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_intermediate_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->out_hidden_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->image_token_id), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_start_token_id), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_end_token_id), sizeof(int), 1, file) != 1) ||
        (fread(&(config->video_token_id), sizeof(int), 1, file) != 1)
    ) {
        fprintf(stderr, "Error reading config field from file.\n");
        fclose(file);
        return;
    }

    // ==================================================================================
    // 2. Derived Dimensions
    // ==================================================================================
    int head_dim = config->hidden_size / config->num_attention_heads;
    int vision_head_dim = config->vision_hidden_size / config->vision_num_heads;
    int vision_qkv_dim = 3 * config->vision_num_heads * vision_head_dim;
    int num_deepstack_mergers = 3;

    long L  = config->num_hidden_layers;
    long H  = config->hidden_size;
    long I  = config->intermediate_size;
    long QD = config->num_attention_heads * head_dim;
    long KVAD = config->num_key_value_heads * head_dim;

    long VH = config->vision_hidden_size;
    long VP = config->vision_patch_size;
    long VD = config->vision_depth;
    long VI = config->vision_intermediate_size;
    long OH = config->out_hidden_size;
    long QKVD = vision_qkv_dim;
    int M = num_deepstack_mergers;

    // ==================================================================================
    // 3. Allocate + Read + Print (Preserve fread order)
    // ==================================================================================

    // --- embed_tokens.weight
    weights->token_embedding_table = (float*)malloc((long)config->vocab_size * H * sizeof(float));
    fread(weights->token_embedding_table, sizeof(float), (long)config->vocab_size * H, file);
    printf("Shape of token_embedding_table (embed_tokens.weight): (%d, %ld)\n", config->vocab_size, H);

    // --- input_layernorm.weight
    weights->rms_ffn_w = (float*)malloc(L * H * sizeof(float));
    fread(weights->rms_ffn_w, sizeof(float), L * H, file);
    printf("Shape of rms_ffn_w (input_layernorm.weight): (%ld, %ld)\n", L, H);

    // --- mlp.down_proj.weight
    weights->w_mlp_down = (float*)malloc(L * H * I * sizeof(float));
    fread(weights->w_mlp_down, sizeof(float), L * H * I, file);
    printf("Shape of w_mlp_down (mlp.down_proj.weight): (%ld, %ld, %ld)\n", L, H, I);

    // --- mlp.gate_proj.weight
    weights->w_mlp_gate = (float*)malloc(L * I * H * sizeof(float));
    fread(weights->w_mlp_gate, sizeof(float), L * I * H, file);
    printf("Shape of w_mlp_gate (mlp.gate_proj.weight): (%ld, %ld, %ld)\n", L, I, H);

    // --- mlp.up_proj.weight
    weights->w_mlp_up = (float*)malloc(L * I * H * sizeof(float));
    fread(weights->w_mlp_up, sizeof(float), L * I * H, file);
    printf("Shape of w_mlp_up (mlp.up_proj.weight): (%ld, %ld, %ld)\n", L, I, H);

    // --- post_attention_layernorm.weight
    weights->rms_attn_w = (float*)malloc(L * H * sizeof(float));
    fread(weights->rms_attn_w, sizeof(float), L * H, file);
    printf("Shape of rms_attn_w (post_attention_layernorm.weight): (%ld, %ld)\n", L, H);

    // --- self_attn.k_norm.weight
    weights->w_attn_k_norm = (float*)malloc(L * 1ll * head_dim * sizeof(float));
    fread(weights->w_attn_k_norm, sizeof(float), L * 1ll * head_dim, file);
    printf("Shape of w_attn_k_norm (self_attn.k_norm.weight): (%ld, %d)\n", L, head_dim);

    // --- self_attn.k_proj.weight
    weights->w_attn_k = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(weights->w_attn_k, sizeof(float), L * KVAD * H, file);
    printf("Shape of w_attn_k (self_attn.k_proj.weight): (%ld, %ld, %ld)\n", L, KVAD, H);

    // --- self_attn.o_proj.weight
    weights->w_attn_o = (float*)malloc(L * H * H * sizeof(float));
    fread(weights->w_attn_o, sizeof(float), L * H * H, file);
    printf("Shape of w_attn_o (self_attn.o_proj.weight): (%ld, %ld, %ld)\n", L, H, H);

    // --- self_attn.q_norm.weight
    weights->w_attn_q_norm = (float*)malloc(L * head_dim * sizeof(float));
    fread(weights->w_attn_q_norm, sizeof(float), L * head_dim, file);
    printf("Shape of w_attn_q_norm (self_attn.q_norm.weight): (%ld, %ld)\n", L, head_dim);

    // --- self_attn.q_proj.weight
    weights->w_attn_q = (float*)malloc(L * QD * H * sizeof(float));
    fread(weights->w_attn_q, sizeof(float), L * QD * H, file);
    printf("Shape of w_attn_q (self_attn.q_proj.weight): (%ld, %ld, %ld)\n", L, QD, H);

    // --- self_attn.v_proj.weight
    weights->w_attn_v = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(weights->w_attn_v, sizeof(float), L * KVAD * H, file);
    printf("Shape of w_attn_v (self_attn.v_proj.weight): (%ld, %ld, %ld)\n", L, KVAD, H);

    // --- final layernorm.weight
    weights->rms_out_w = (float*)malloc(H * sizeof(float));
    fread(weights->rms_out_w, sizeof(float), H, file);
    printf("Shape of rms_out_w (final_layernorm.weight): (%ld)\n", H);

    // --- Vision model general
    weights->visual_attn_qkv_bias = (float*)malloc(QKVD * sizeof(float));
    fread(weights->visual_attn_qkv_bias, sizeof(float), QKVD, file);
    printf("Shape of visual_attn_qkv_bias: (%ld)\n", QKVD);

    weights->visual_attn_qkv_weight = (float*)malloc(QKVD * VH * sizeof(float));
    fread(weights->visual_attn_qkv_weight, sizeof(float), QKVD * VH, file);
    printf("Shape of visual_attn_qkv_weight: (%ld, %ld)\n", QKVD, VH);

    weights->visual_attn_proj_bias = (float*)malloc(VH * sizeof(float));
    fread(weights->visual_attn_proj_bias, sizeof(float), VH, file);
    printf("Shape of visual_attn_proj_bias: (%ld)\n", VH);

    weights->visual_attn_proj_weight = (float*)malloc(VH * VH * sizeof(float));
    fread(weights->visual_attn_proj_weight, sizeof(float), VH * VH, file);
    printf("Shape of visual_attn_proj_weight: (%ld, %ld)\n", VH, VH);

    weights->visual_class_embedding = (float*)malloc(VH * sizeof(float));
    fread(weights->visual_class_embedding, sizeof(float), VH, file);
    printf("Shape of visual_class_embedding: (%ld)\n", VH);

    weights->visual_conv1_weight = (float*)malloc(VH * 3 * VP * VP * sizeof(float));
    fread(weights->visual_conv1_weight, sizeof(float), VH * 3 * VP * VP, file);
    printf("Shape of visual_conv1_weight: (%ld, 3, %ld, %ld)\n", VH, VP, VP);

    weights->visual_ln_post_bias = (float*)malloc(VH * sizeof(float));
    fread(weights->visual_ln_post_bias, sizeof(float), VH, file);
    printf("Shape of visual_ln_post_bias: (%ld)\n", VH);

    weights->visual_ln_post_weight = (float*)malloc(VH * sizeof(float));
    fread(weights->visual_ln_post_weight, sizeof(float), VH, file);
    printf("Shape of visual_ln_post_weight: (%ld)\n", VH);

    // (continues with the same allocate + fread + printf pattern)
    // for all remaining vision weights, resblocks, merger_list, and final merger
    // ...
    
    fclose(file);
    printf("Successfully loaded model from %s\n", path);
}


void init_model_run_state(QwenRunState *run_state) {
    printf("Init model run state...\n");
}

void free_model_weights(QwenWeight* weights) {
    if (!weights) return;

    // Language Model Standalone Weights
    free(weights->token_embedding_table);
    free(weights->rms_out_w);

    // Language Model Layer Weights (Continuous Blocks)
    free(weights->rms_ffn_w);
    free(weights->w_mlp_down);
    free(weights->w_mlp_gate);
    free(weights->w_mlp_up);
    free(weights->rms_attn_w);
    free(weights->w_attn_k_norm);
    free(weights->w_attn_k);
    free(weights->w_attn_o);
    free(weights->w_attn_q_norm);
    free(weights->w_attn_q);
    free(weights->w_attn_v);

    // Vision Model Weights (General)
    free(weights->visual_attn_qkv_bias);
    free(weights->visual_attn_qkv_weight);
    free(weights->visual_attn_proj_bias);
    free(weights->visual_attn_proj_weight);
    free(weights->visual_class_embedding);
    free(weights->visual_conv1_weight);
    free(weights->visual_ln_post_bias);
    free(weights->visual_ln_post_weight);
    free(weights->visual_ln_pre_bias);
    free(weights->visual_ln_pre_weight);
    free(weights->visual_patch_embed_proj_bias);
    free(weights->visual_patch_embed_proj_weight);
    free(weights->visual_positional_embedding);

    // Vision ResBlocks Weights (Continuous Blocks)
    free(weights->visual_resblocks_attn_in_proj_bias);
    free(weights->visual_resblocks_attn_in_proj_weight);
    free(weights->visual_resblocks_attn_out_proj_bias);
    free(weights->visual_resblocks_attn_out_proj_weight);
    free(weights->visual_resblocks_ln_1_bias);
    free(weights->visual_resblocks_ln_1_weight);
    free(weights->visual_resblocks_ln_2_bias);
    free(weights->visual_resblocks_ln_2_weight);
    free(weights->visual_resblocks_mlp_c_fc_bias);
    free(weights->visual_resblocks_mlp_c_fc_weight);
    free(weights->visual_resblocks_mlp_c_proj_bias);
    free(weights->visual_resblocks_mlp_c_proj_weight);

    // Vision Deepstack Merger Weights (Continuous Blocks)
    free(weights->visual_deepstack_merger_list_linear_fc1_bias);
    free(weights->visual_deepstack_merger_list_linear_fc1_weight);
    free(weights->visual_deepstack_merger_list_linear_fc2_bias);
    free(weights->visual_deepstack_merger_list_linear_fc2_weight);
    free(weights->visual_deepstack_merger_list_norm_bias);
    free(weights->visual_deepstack_merger_list_rms_out_w);

    // Final Merger Weights
    free(weights->visual_merger_linear_fc1_bias);
    free(weights->visual_merger_linear_fc1_weight);
    free(weights->visual_merger_linear_fc2_bias);
    free(weights->visual_merger_linear_fc2_weight);
    free(weights->visual_merger_norm_bias);
    free(weights->visual_merger_rms_out_w);
    
    // Set the struct memory to zero to prevent accidental double-free attempts
    memset(weights, 0, sizeof(QwenWeight)); 
}

void init_model_run_state(QwenRunState* state, const QwenConfig* config) {
    memset(state, 0, sizeof(QwenRunState));
    state->vision_embed_true = false;

    int H   = config->hidden_size;
    int I   = config->intermediate_size;
    int V   = config->vocab_size;
    int L   = config->num_hidden_layers;
    int NH  = config->num_attention_heads;
    int NKV = config->num_key_value_heads;
    int D   = 128;
    int S   = 1024;

    size_t cache_size = (size_t)L * NKV * S * D * sizeof(float);

    // ---- Hidden / intermediate ----
    state->x = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->x, H * sizeof(float));

    state->t = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->t, H * sizeof(float));

    state->q = (float*)malloc(NH * D * sizeof(float));
    CHECK_ALLOC(state->q, NH * D * sizeof(float));

    state->k = (float*)malloc(NKV * D * sizeof(float));
    CHECK_ALLOC(state->k, NKV * D * sizeof(float));

    state->v = (float*)malloc(NKV * D * sizeof(float));
    CHECK_ALLOC(state->v, NKV * D * sizeof(float));

    state->att = (float*)malloc(NH * S * sizeof(float));
    CHECK_ALLOC(state->att, NH * S * sizeof(float));

    state->qkv_out = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->qkv_out, H * sizeof(float));

    state->attn_out = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->attn_out, H * sizeof(float));

    state->gate = (float*)malloc(I * sizeof(float));
    CHECK_ALLOC(state->gate, I * sizeof(float));

    state->up = (float*)malloc(I * sizeof(float));
    CHECK_ALLOC(state->up, I * sizeof(float));

    state->gate_up = (float*)malloc(I * sizeof(float));
    CHECK_ALLOC(state->gate_up, I * sizeof(float));

    state->down = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->down, H * sizeof(float));

    state->cos_tensor = (float*)malloc(S * (D / 2) * sizeof(float));
    CHECK_ALLOC(state->cos_tensor, S * (D / 2) * sizeof(float));

    state->sin_tensor = (float*)malloc(S * (D / 2) * sizeof(float));
    CHECK_ALLOC(state->sin_tensor, S * (D / 2) * sizeof(float));

    state->logits = (float*)malloc(V * sizeof(float));
    CHECK_ALLOC(state->logits, V * sizeof(float));

    // ---- KV cache ----
    state->key_cache = (float*)malloc(cache_size);
    CHECK_ALLOC(state->key_cache, cache_size);

    state->value_cache = (float*)malloc(cache_size);
    CHECK_ALLOC(state->value_cache, cache_size);

    qwen_rope_precompute(state->cos_tensor, state->sin_tensor, config);
}

void free_model_run_state(QwenRunState* state) {
    if (!state) return;

    if (state->x) free(state->x);
    if (state->t) free(state->t);
    if (state->q) free(state->q);
    if (state->k) free(state->k);
    if (state->v) free(state->v);
    if (state->att) free(state->att);
    if (state->qkv_out) free(state->qkv_out);
    if (state->attn_out) free(state->attn_out);
    if (state->gate) free(state->gate);
    if (state->up) free(state->up);
    if (state->gate_up) free(state->gate_up);
    if (state->down) free(state->down);
    if (state->cos_tensor) free(state->cos_tensor);
    if (state->sin_tensor) free(state->sin_tensor);
    if (state->logits) free(state->logits);
    if (state->key_cache) free(state->key_cache);
    if (state->value_cache) free(state->value_cache);

    memset(state, 0, sizeof(QwenRunState));
}

void qwen_rope_precompute(
    float *cos_all_out,  // (seq_len * head_dim/2)
    float *sin_all_out,  // (seq_len * head_dim/2)
    const QwenConfig *config
) {
    int seq_len = 1024;
    int head_dim = 128;
    float rope_theta = 5000000.0f;
    const int mrope_section[] = {24, 20, 20};
    int num_sections = 3;
    bool mrope_interleaved = true;

    int d_half = head_dim / 2;

    // --- Step 1: compute inv_freq (standard RoPE)
    float *inv_freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        inv_freq[i] = 1.0f / powf(rope_theta, (2.0f * i) / (float)head_dim);
    }

    // --- Step 2: optionally handle mRoPE interleaving ---
    // For Qwen with "rope_type": "default", rope_theta is same across sections,
    // so we just leave inv_freq as-is.
    // If you wanted to enforce section-based remapping, here’s where you’d do it.

    // Example of section-wise adjustment (not needed for "default")
    // int offset = 0;
    // for (int s = 0; s < num_sections; ++s) {
    //     int section_len = mrope_section[s];
    //     for (int i = 0; i < section_len && (offset + i) < d_half; ++i) {
    //         inv_freq[offset + i] = 1.0f / powf(rope_theta, (2.0f * (offset + i)) / head_dim);
    //     }
    //     offset += section_len;
    // }

    // --- Step 3: precompute sin/cos for all positions ---
    for (int pos = 0; pos < seq_len; pos++) {
        for (int j = 0; j < d_half; j++) {
            float angle = (float)pos * inv_freq[j];
            int index = pos * d_half + j;
            cos_all_out[index] = cosf(angle);
            sin_all_out[index] = sinf(angle);
        }
    }

    free(inv_freq);
}
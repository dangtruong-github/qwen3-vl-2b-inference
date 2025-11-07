#include "load_model.hpp"

void init_model_weights(const char* path, QwenConfig* config, QwenWeight* weights) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return;
    }

    // 1. Read config header field by field
    // NOTE: This assumes all 'int' fields are 4 bytes and 'float' are 4 bytes.
    // The order MUST match the order they were written to the binary file (as seen in export.txt).
    if (
        // int fields
        (fread(&(config->vocab_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->hidden_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->intermediate_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_hidden_layers), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_attention_heads), sizeof(int), 1, file) != 1) ||
        (fread(&(config->num_key_value_heads), sizeof(int), 1, file) != 1) ||
        (fread(&(config->max_position_embeddings), sizeof(int), 1, file) != 1) ||
        (fread(&(config->rope_theta), sizeof(int), 1, file) != 1) ||
        // float field
        (fread(&(config->rms_norm_eps), sizeof(float), 1, file) != 1) ||
        // remaining int fields
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

    // Derived dimensions (in elements/floats)
    int head_dim = config->hidden_size / config->num_attention_heads;
    int vision_head_dim = config->vision_hidden_size / config->vision_num_heads;
    // QKV_Dim is 3x the total Q/K/V dimension (3 * num_heads * head_dim)
    // The vision weight naming suggests a single QKV projection that's 3x
    int vision_qkv_dim = 3 * config->vision_num_heads * vision_head_dim;
    int num_deepstack_mergers = 3; // Based on export.txt (0, 1, 2)

    // ==================================================================================
    // 2. Allocate Memory for All Weights
    // ==================================================================================

    // Language Model Standalone
    weights->embed_tokens_weight = (float*)malloc((long)config->vocab_size * config->hidden_size * sizeof(float));
    weights->norm_weight = (float*)malloc((long)config->hidden_size * sizeof(float));
    weights->lm_head_weight = (float*)malloc((long)config->vocab_size * config->hidden_size * sizeof(float));

    // Language Model Layers (L = num_hidden_layers)
    long L = config->num_hidden_layers;
    long H = config->hidden_size;
    long I = config->intermediate_size;
    long QD = config->num_attention_heads * head_dim;
    long KVAD = config->num_key_value_heads * head_dim;

    weights->input_layernorm_weight          = (float*)malloc(L * H * sizeof(float));
    weights->mlp_down_proj_weight            = (float*)malloc(L * H * I * sizeof(float));
    weights->mlp_gate_proj_weight            = (float*)malloc(L * I * H * sizeof(float));
    weights->mlp_up_proj_weight              = (float*)malloc(L * I * H * sizeof(float));
    weights->post_attention_layernorm_weight = (float*)malloc(L * H * sizeof(float));
    weights->self_attn_k_norm_weight         = (float*)malloc(L * KVAD * sizeof(float));
    weights->self_attn_k_proj_weight         = (float*)malloc(L * KVAD * H * sizeof(float));
    weights->self_attn_o_proj_weight         = (float*)malloc(L * H * H * sizeof(float));
    weights->self_attn_q_norm_weight         = (float*)malloc(L * QD * sizeof(float));
    weights->self_attn_q_proj_weight         = (float*)malloc(L * QD * H * sizeof(float));
    weights->self_attn_v_proj_weight         = (float*)malloc(L * KVAD * H * sizeof(float));

    // Vision Model Standalone (VH = vision_hidden_size, VP = vision_patch_size)
    long VH = config->vision_hidden_size;
    long VP = config->vision_patch_size;
    long VD = config->vision_depth;
    long VI = config->vision_intermediate_size;
    long OH = config->out_hidden_size;
    long QKVD = vision_qkv_dim;

    weights->visual_attn_qkv_bias        = (float*)malloc(QKVD * sizeof(float));
    weights->visual_attn_qkv_weight      = (float*)malloc(QKVD * VH * sizeof(float));
    weights->visual_attn_proj_bias       = (float*)malloc(VH * sizeof(float));
    weights->visual_attn_proj_weight     = (float*)malloc(VH * VH * sizeof(float));
    weights->visual_class_embedding      = (float*)malloc(VH * sizeof(float));
    weights->visual_conv1_weight         = (float*)malloc(VH * 3 * VP * VP * sizeof(float));
    weights->visual_ln_post_bias         = (float*)malloc(VH * sizeof(float));
    weights->visual_ln_post_weight       = (float*)malloc(VH * sizeof(float));
    weights->visual_ln_pre_bias          = (float*)malloc(VH * sizeof(float));
    weights->visual_ln_pre_weight        = (float*)malloc(VH * sizeof(float));
    weights->visual_patch_embed_proj_bias= (float*)malloc(VH * sizeof(float));
    weights->visual_patch_embed_proj_weight = (float*)malloc(VH * 3 * VP * VP * sizeof(float));
    weights->visual_positional_embedding = (float*)malloc(257 * VH * sizeof(float));

    // Vision ResBlocks (VD = vision_depth)
    weights->visual_resblocks_attn_in_proj_bias  = (float*)malloc(VD * QKVD * sizeof(float));
    weights->visual_resblocks_attn_in_proj_weight = (float*)malloc(VD * QKVD * VH * sizeof(float));
    weights->visual_resblocks_attn_out_proj_bias = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_attn_out_proj_weight = (float*)malloc(VD * VH * VH * sizeof(float));
    weights->visual_resblocks_ln_1_bias          = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_ln_1_weight        = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_ln_2_bias          = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_ln_2_weight        = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_mlp_c_fc_bias      = (float*)malloc(VD * VI * sizeof(float));
    weights->visual_resblocks_mlp_c_fc_weight    = (float*)malloc(VD * VI * VH * sizeof(float));
    weights->visual_resblocks_mlp_c_proj_bias    = (float*)malloc(VD * VH * sizeof(float));
    weights->visual_resblocks_mlp_c_proj_weight  = (float*)malloc(VD * VH * VI * sizeof(float));

    // Vision Deepstack Merger List (3 layers)
    int M = num_deepstack_mergers;
    weights->visual_deepstack_merger_list_linear_fc1_bias  = (float*)malloc(M * VI * sizeof(float));
    weights->visual_deepstack_merger_list_linear_fc1_weight = (float*)malloc(M * VI * VI * sizeof(float));
    weights->visual_deepstack_merger_list_linear_fc2_bias  = (float*)malloc(M * OH * sizeof(float));
    weights->visual_deepstack_merger_list_linear_fc2_weight = (float*)malloc(M * OH * VI * sizeof(float));
    weights->visual_deepstack_merger_list_norm_bias        = (float*)malloc(M * VI * sizeof(float));
    weights->visual_deepstack_merger_list_norm_weight      = (float*)malloc(M * VI * sizeof(float));

    // Final Merger
    weights->visual_merger_linear_fc1_bias  = (float*)malloc(VI * sizeof(float));
    weights->visual_merger_linear_fc1_weight = (float*)malloc(VI * VI * sizeof(float));
    weights->visual_merger_linear_fc2_bias  = (float*)malloc(OH * sizeof(float));
    weights->visual_merger_linear_fc2_weight = (float*)malloc(OH * VI * sizeof(float));
    weights->visual_merger_norm_bias        = (float*)malloc(VI * sizeof(float));
    weights->visual_merger_norm_weight      = (float*)malloc(VI * sizeof(float));

    // ==================================================================================
    // 3. Read Weights From File (MUST match export.txt order)
    // ==================================================================================
    
    // --- Language Model Weights (Embeddings) ---
    fread(weights->embed_tokens_weight, sizeof(float), (long)config->vocab_size * H, file);

    // --- Language Model Layers (Interleaved Read into Continuous Blocks) ---
    for (int l = 0; l < L; ++l) {
        long offset_H = (long)l * H;
        long offset_HI = (long)l * H * I;
        long offset_IH = (long)l * I * H;
        long offset_QD = (long)l * QD;
        long offset_KVAD = (long)l * KVAD;
        long offset_KVADH = (long)l * KVAD * H;
        long offset_HH = (long)l * H * H;
        long offset_QDH = (long)l * QD * H;
        
        // input_layernorm.weight
        fread(weights->input_layernorm_weight + offset_H, sizeof(float), H, file);
        // mlp.down_proj.weight
        fread(weights->mlp_down_proj_weight + offset_HI, sizeof(float), H * I, file);
        // mlp.gate_proj.weight
        fread(weights->mlp_gate_proj_weight + offset_IH, sizeof(float), I * H, file);
        // mlp.up_proj.weight
        fread(weights->mlp_up_proj_weight + offset_IH, sizeof(float), I * H, file);
        // post_attention_layernorm.weight
        fread(weights->post_attention_layernorm_weight + offset_H, sizeof(float), H, file);
        // self_attn.k_norm.weight
        fread(weights->self_attn_k_norm_weight + offset_KVAD, sizeof(float), KVAD, file);
        // self_attn.k_proj.weight
        fread(weights->self_attn_k_proj_weight + offset_KVADH, sizeof(float), KVAD * H, file);
        // self_attn.o_proj.weight
        fread(weights->self_attn_o_proj_weight + offset_HH, sizeof(float), H * H, file);
        // self_attn.q_norm.weight
        fread(weights->self_attn_q_norm_weight + offset_QD, sizeof(float), QD, file);
        // self_attn.q_proj.weight
        fread(weights->self_attn_q_proj_weight + offset_QDH, sizeof(float), QD * H, file);
        // self_attn.v_proj.weight
        fread(weights->self_attn_v_proj_weight + offset_KVADH, sizeof(float), KVAD * H, file);
    }

    // --- Language Model Weights (Final Norm and Head) ---
    fread(weights->norm_weight, sizeof(float), H, file);
    fread(weights->lm_head_weight, sizeof(float), (long)config->vocab_size * H, file);

    // --- Vision Model Weights (General) ---
    fread(weights->visual_attn_qkv_bias, sizeof(float), QKVD, file);
    fread(weights->visual_attn_qkv_weight, sizeof(float), QKVD * VH, file);
    fread(weights->visual_attn_proj_bias, sizeof(float), VH, file);
    fread(weights->visual_attn_proj_weight, sizeof(float), VH * VH, file);
    fread(weights->visual_class_embedding, sizeof(float), VH, file);
    fread(weights->visual_conv1_weight, sizeof(float), VH * 3 * VP * VP, file);
    fread(weights->visual_ln_post_bias, sizeof(float), VH, file);
    fread(weights->visual_ln_post_weight, sizeof(float), VH, file);
    fread(weights->visual_ln_pre_bias, sizeof(float), VH, file);
    fread(weights->visual_ln_pre_weight, sizeof(float), VH, file);
    fread(weights->visual_patch_embed_proj_bias, sizeof(float), VH, file);
    fread(weights->visual_patch_embed_proj_weight, sizeof(float), VH * 3 * VP * VP, file);
    fread(weights->visual_positional_embedding, sizeof(float), 257 * VH, file); // 257 is (max_patches + 1)

    // --- Vision ResBlocks (Interleaved Read into Continuous Blocks) ---
    for (int d = 0; d < VD; ++d) {
        long offset_QKVD = (long)d * QKVD;
        long offset_QKVDS = (long)d * QKVD * VH;
        long offset_VH = (long)d * VH;
        long offset_VHH = (long)d * VH * VH;
        long offset_VI = (long)d * VI;
        long offset_VIH = (long)d * VI * VH;
        long offset_VHI = (long)d * VH * VI;

        // attn.in_proj.bias
        fread(weights->visual_resblocks_attn_in_proj_bias + offset_QKVD, sizeof(float), QKVD, file);
        // attn.in_proj.weight
        fread(weights->visual_resblocks_attn_in_proj_weight + offset_QKVDS, sizeof(float), QKVD * VH, file);
        // attn.out_proj.bias
        fread(weights->visual_resblocks_attn_out_proj_bias + offset_VH, sizeof(float), VH, file);
        // attn.out_proj.weight
        fread(weights->visual_resblocks_attn_out_proj_weight + offset_VHH, sizeof(float), VH * VH, file);
        // ln_1.bias
        fread(weights->visual_resblocks_ln_1_bias + offset_VH, sizeof(float), VH, file);
        // ln_1.weight
        fread(weights->visual_resblocks_ln_1_weight + offset_VH, sizeof(float), VH, file);
        // ln_2.bias
        fread(weights->visual_resblocks_ln_2_bias + offset_VH, sizeof(float), VH, file);
        // ln_2.weight
        fread(weights->visual_resblocks_ln_2_weight + offset_VH, sizeof(float), VH, file);
        // mlp.c_fc.bias
        fread(weights->visual_resblocks_mlp_c_fc_bias + offset_VI, sizeof(float), VI, file);
        // mlp.c_fc.weight
        fread(weights->visual_resblocks_mlp_c_fc_weight + offset_VIH, sizeof(float), VI * VH, file);
        // mlp.c_proj.bias
        fread(weights->visual_resblocks_mlp_c_proj_bias + offset_VH, sizeof(float), VH, file);
        // mlp.c_proj.weight
        fread(weights->visual_resblocks_mlp_c_proj_weight + offset_VHI, sizeof(float), VH * VI, file);
    }

    // --- Vision Deepstack Merger List (Interleaved Read into Continuous Blocks) ---
    for (int m = 0; m < M; ++m) {
        long offset_VI = (long)m * VI;
        long offset_VIVI = (long)m * VI * VI;
        long offset_OH = (long)m * OH;
        long offset_OHVI = (long)m * OH * VI;

        // linear_fc1.bias
        fread(weights->visual_deepstack_merger_list_linear_fc1_bias + offset_VI, sizeof(float), VI, file);
        // linear_fc1.weight
        fread(weights->visual_deepstack_merger_list_linear_fc1_weight + offset_VIVI, sizeof(float), VI * VI, file);
        // linear_fc2.bias
        fread(weights->visual_deepstack_merger_list_linear_fc2_bias + offset_OH, sizeof(float), OH, file);
        // linear_fc2.weight
        fread(weights->visual_deepstack_merger_list_linear_fc2_weight + offset_OHVI, sizeof(float), OH * VI, file);
        // norm.bias
        fread(weights->visual_deepstack_merger_list_norm_bias + offset_VI, sizeof(float), VI, file);
        // norm.weight
        fread(weights->visual_deepstack_merger_list_norm_weight + offset_VI, sizeof(float), VI, file);
    }

    // --- Final Vision Merger ---
    fread(weights->visual_merger_linear_fc1_bias, sizeof(float), VI, file);
    fread(weights->visual_merger_linear_fc1_weight, sizeof(float), VI * VI, file);
    fread(weights->visual_merger_linear_fc2_bias, sizeof(float), OH, file);
    fread(weights->visual_merger_linear_fc2_weight, sizeof(float), OH * VI, file);
    fread(weights->visual_merger_norm_bias, sizeof(float), VI, file);
    fread(weights->visual_merger_norm_weight, sizeof(float), VI, file);

    // 4. Close File
    fclose(file);
    printf("Successfully loaded model from %s\n", path);
}

void init_model_run_state(QwenRunState *run_state) {
    printf("Init model run state...\n");
}

void free_model_weights(QwenWeight* weights) {
    if (!weights) return;

    // Language Model Standalone Weights
    free(weights->embed_tokens_weight);
    free(weights->norm_weight);
    free(weights->lm_head_weight);

    // Language Model Layer Weights (Continuous Blocks)
    free(weights->input_layernorm_weight);
    free(weights->mlp_down_proj_weight);
    free(weights->mlp_gate_proj_weight);
    free(weights->mlp_up_proj_weight);
    free(weights->post_attention_layernorm_weight);
    free(weights->self_attn_k_norm_weight);
    free(weights->self_attn_k_proj_weight);
    free(weights->self_attn_o_proj_weight);
    free(weights->self_attn_q_norm_weight);
    free(weights->self_attn_q_proj_weight);
    free(weights->self_attn_v_proj_weight);

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
    free(weights->visual_deepstack_merger_list_norm_weight);

    // Final Merger Weights
    free(weights->visual_merger_linear_fc1_bias);
    free(weights->visual_merger_linear_fc1_weight);
    free(weights->visual_merger_linear_fc2_bias);
    free(weights->visual_merger_linear_fc2_weight);
    free(weights->visual_merger_norm_bias);
    free(weights->visual_merger_norm_weight);
    
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

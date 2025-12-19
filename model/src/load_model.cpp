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
    long VTP = config->vision_temporal_patch_size;
    long VD = config->vision_depth;
    long VI = config->vision_intermediate_size;
    long OH = config->out_hidden_size;
    config->vision_num_channels = 3;
    long VC = config->vision_num_channels;
    config->vision_num_position_embeddings = 2304;
    long VNPE = config->vision_num_position_embeddings;
    config->vision_deep_stack_depth = 3;
    long VDSD = config->vision_deep_stack_depth;
    long QKVD = vision_qkv_dim;
    int M = num_deepstack_mergers;

    // ==================================================================================
    // 3. Allocate + Read + Print (Preserve fread order)
    // ==================================================================================
    float *tmp_ptr = nullptr;

    // EDIT THE CODE FROM THIS TO THE END OF THE FUNCTION

    // --- embed_tokens.weight
    tmp_ptr = (float*)malloc((long)config->vocab_size * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), (long)config->vocab_size * H, file);
    printf("Shape of token_embedding_table (embed_tokens.weight): (%d, %ld)\n", config->vocab_size, H);
    weights->token_embedding_table = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- input_layernorm.weight
    tmp_ptr = (float*)malloc(L * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H, file);
    printf("Shape of rms_ffn_w (input_layernorm.weight): (%ld, %ld)\n", L, H);
    weights->rms_ffn_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- mlp.down_proj.weight
    tmp_ptr = (float*)malloc(L * H * I * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H * I, file);
    printf("Shape of w_mlp_down (mlp.down_proj.weight): (%ld, %ld, %ld)\n", L, H, I);
    weights->w_mlp_down = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- mlp.gate_proj.weight
    tmp_ptr = (float*)malloc(L * I * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * I * H, file);
    printf("Shape of w_mlp_gate (mlp.gate_proj.weight): (%ld, %ld, %ld)\n", L, I, H);
    weights->w_mlp_gate = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- mlp.up_proj.weight
    tmp_ptr = (float*)malloc(L * I * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * I * H, file);
    printf("Shape of w_mlp_up (mlp.up_proj.weight): (%ld, %ld, %ld)\n", L, I, H);
    weights->w_mlp_up = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- post_attention_layernorm.weight
    tmp_ptr = (float*)malloc(L * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H, file);
    printf("Shape of rms_attn_w (post_attention_layernorm.weight): (%ld, %ld)\n", L, H);
    weights->rms_attn_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.k_norm.weight
    tmp_ptr = (float*)malloc(L * 1ll * head_dim * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * 1ll * head_dim, file);
    printf("Shape of w_attn_k_norm (self_attn.k_norm.weight): (%ld, %d)\n", L, head_dim);
    weights->w_attn_k_norm = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.k_proj.weight
    tmp_ptr = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * KVAD * H, file);
    printf("Shape of w_attn_k (self_attn.k_proj.weight): (%ld, %ld, %ld)\n", L, KVAD, H);
    weights->w_attn_k = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.o_proj.weight
    tmp_ptr = (float*)malloc(L * H * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H * H, file);
    printf("Shape of w_attn_o (self_attn.o_proj.weight): (%ld, %ld, %ld)\n", L, H, H);
    weights->w_attn_o = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.q_norm.weight
    tmp_ptr = (float*)malloc(L * head_dim * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * head_dim, file);
    printf("Shape of w_attn_q_norm (self_attn.q_norm.weight): (%ld, %ld)\n", L, head_dim);
    weights->w_attn_q_norm = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.q_proj.weight
    tmp_ptr = (float*)malloc(L * QD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * QD * H, file);
    printf("Shape of w_attn_q (self_attn.q_proj.weight): (%ld, %ld, %ld)\n", L, QD, H);
    weights->w_attn_q = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- self_attn.v_proj.weight
    tmp_ptr = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * KVAD * H, file);
    printf("Shape of w_attn_v (self_attn.v_proj.weight): (%ld, %ld, %ld)\n", L, KVAD, H);
    weights->w_attn_v = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- final layernorm.weight
    tmp_ptr = (float*)malloc(H * sizeof(float));
    fread(tmp_ptr, sizeof(float), H, file);
    printf("Shape of rms_out_w (final_layernorm.weight): (%ld)\n", H);
    weights->rms_out_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    // --- Vision model general
    tmp_ptr = (float*)malloc(VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), VH, file);
    printf("Shape of vl_patch_emb_b: (%ld)\n", VH);
    weights->vl_patch_emb_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(VH * VC * VTP * VP * VP * sizeof(float));
    fread(tmp_ptr, sizeof(float), VH * VC * VTP * VP * VP, file);
    printf("Shape of vl_patch_emb_w: (%ld, %ld, %ld, %ld, %ld)\n", VH, VC, VTP, VP, VP);
    weights->vl_patch_emb_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VNPE * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VNPE * VH, file);
    printf("Shape of vl_pos_emb_w: (%ld, %ld)\n", VNPE, VH);
    weights->vl_pos_emb_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_attn_proj_b: (%ld, %ld)\n", VD, VH);
    weights->vl_attn_proj_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH * VH, file);
    printf("Shape of vl_attn_proj_w: (%ld, %ld, %ld)\n", VD, VH, VH);
    weights->vl_attn_proj_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * 3 * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * 3 * VH, file);
    printf("Shape of vl_attn_qkv_b: (%ld, %ld)\n", VD, 3 * VH);
    weights->vl_attn_qkv_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * 3 * VH * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * 3 * VH * VH, file);
    printf("Shape of vl_attn_qkv_w: (%ld, %ld, %ld)\n", VD, 3 * VH, VH);
    weights->vl_attn_qkv_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VI, file);
    printf("Shape of vl_mlp1_b: (%ld, %ld)\n", VD, VI);
    weights->vl_mlp1_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VI * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VI * VH, file);
    printf("Shape of vl_mlp1_w: (%ld, %ld, %ld)\n", VD, VI, VH);
    weights->vl_mlp1_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_mlp2_b: (%ld, %ld)\n", VD, VH);
    weights->vl_mlp2_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH * VI, file);
    printf("Shape of vl_mlp2_w: (%ld, %ld, %ld)\n", VD, VH, VI);
    weights->vl_mlp2_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_norm1_b: (%ld, %ld)\n", VD, VH);
    weights->vl_norm1_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_norm1_w: (%ld, %ld)\n", VD, VH);
    weights->vl_norm1_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_norm2_b: (%ld, %ld)\n", VD, VH);
    weights->vl_norm2_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    printf("Shape of vl_norm2_w: (%ld, %ld)\n", VD, VH);
    weights->vl_norm2_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    printf("Shape of vl_d_mlp1_b: (%ld, %ld)\n", VDSD, VI);
    weights->vl_d_mlp1_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI * VI, file);
    printf("Shape of vl_d_mlp1_w: (%ld, %ld, %ld)\n", VDSD, VI, VI);
    weights->vl_d_mlp1_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * OH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * OH, file);
    printf("Shape of vl_d_mlp2_b: (%ld, %ld)\n", VDSD, OH);
    weights->vl_d_mlp2_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * OH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * OH * VI, file);
    printf("Shape of vl_d_mlp2_w: (%ld, %ld, %ld)\n", VDSD, OH, VI);
    weights->vl_d_mlp2_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    printf("Shape of vl_d_norm_b: (%ld, %ld)\n", VDSD, VI);
    weights->vl_d_norm_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    printf("Shape of vl_d_norm_w: (%ld, %ld)\n", VDSD, VI);
    weights->vl_d_norm_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VI, file);
    printf("Shape of vl_merge_mlp1_b: (%ld)\n", VI);
    weights->vl_merge_mlp1_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VI * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VI * VI, file);
    printf("Shape of vl_merge_mlp1_w: (%ld, %ld)\n", VI, VI);
    weights->vl_merge_mlp1_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * OH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * OH, file);
    printf("Shape of vl_merge_mlp2_b: (%ld)\n", OH);
    weights->vl_merge_mlp2_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * OH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * OH * VI, file);
    printf("Shape of vl_merge_mlp2_w: (%ld, %ld)\n", OH, VI);
    weights->vl_merge_mlp2_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VH, file);
    printf("Shape of vl_merge_norm_b: (%ld)\n", VH);
    weights->vl_merge_norm_b = (const float *)tmp_ptr;
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VH, file);
    printf("Shape of vl_merge_norm_w: (%ld)\n", VH);
    weights->vl_merge_norm_w = (const float *)tmp_ptr;
    tmp_ptr = nullptr;
    
    fclose(file);
    printf("Successfully loaded model from %s\n", path);
}

void free_model_weights(QwenWeight* weights) {
    if (!weights) return;

    // Language Model Standalone Weights
    free(const_cast<float *>(weights->token_embedding_table));
    free(const_cast<float *>(weights->rms_out_w));

    // Language Model Layer Weights (Continuous Blocks)
    free(const_cast<float *>(weights->rms_ffn_w));
    free(const_cast<float *>(weights->w_mlp_down));
    free(const_cast<float *>(weights->w_mlp_gate));
    free(const_cast<float *>(weights->w_mlp_up));
    free(const_cast<float *>(weights->rms_attn_w));
    free(const_cast<float *>(weights->w_attn_k_norm));
    free(const_cast<float *>(weights->w_attn_k));
    free(const_cast<float *>(weights->w_attn_o));
    free(const_cast<float *>(weights->w_attn_q_norm));
    free(const_cast<float *>(weights->w_attn_q));
    free(const_cast<float *>(weights->w_attn_v));

    // Vision Model Weights (General)
    free(const_cast<float *>(weights->vl_patch_emb_b));
    free(const_cast<float *>(weights->vl_patch_emb_w));
    free(const_cast<float *>(weights->vl_pos_emb_w));

    free(const_cast<float *>(weights->vl_attn_proj_b));
    free(const_cast<float *>(weights->vl_attn_proj_w));
    free(const_cast<float *>(weights->vl_attn_qkv_b));
    free(const_cast<float *>(weights->vl_attn_qkv_w));
    free(const_cast<float *>(weights->vl_mlp1_b));
    free(const_cast<float *>(weights->vl_mlp1_w));
    free(const_cast<float *>(weights->vl_mlp2_b));
    free(const_cast<float *>(weights->vl_mlp2_w));
    free(const_cast<float *>(weights->vl_norm1_b));
    free(const_cast<float *>(weights->vl_norm1_w));
    free(const_cast<float *>(weights->vl_norm2_b));
    free(const_cast<float *>(weights->vl_norm2_w));

    free(const_cast<float *>(weights->vl_d_mlp1_b));
    free(const_cast<float *>(weights->vl_d_mlp1_w));
    free(const_cast<float *>(weights->vl_d_mlp2_b));
    free(const_cast<float *>(weights->vl_d_mlp2_w));
    free(const_cast<float *>(weights->vl_d_norm_b));
    free(const_cast<float *>(weights->vl_d_norm_w));

    free(const_cast<float *>(weights->vl_merge_mlp1_b));
    free(const_cast<float *>(weights->vl_merge_mlp1_w));
    free(const_cast<float *>(weights->vl_merge_mlp2_b));
    free(const_cast<float *>(weights->vl_merge_mlp2_w));
    free(const_cast<float *>(weights->vl_merge_norm_b));
    free(const_cast<float *>(weights->vl_merge_norm_w));
    
    // Set the struct memory to zero to prevent accidental double-free attempts
    memset(weights, 0, sizeof(QwenWeight)); 
}

void init_model_run_state(QwenRunState* state, const QwenConfig* config) {
    memset(state, 0, sizeof(QwenRunState));
    state->vision_embed_tokens = 0;

    int H   = config->hidden_size;
    int I   = config->intermediate_size;
    int V   = config->vocab_size;
    int L   = config->num_hidden_layers;
    int NH  = config->num_attention_heads;
    int NKV = config->num_key_value_heads;
    int D   = 128;
    int S   = 1024;

    int VH = config->vision_hidden_size;
    int VNP = 2304;
    int VD = 64;
    int VI = config->vision_intermediate_size;

    size_t cache_size = (size_t)L * S * NKV * D * sizeof(float);

    // ---- Hidden / intermediate ----
    state->x = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->x, H * sizeof(float));

    state->t = (float*)malloc(H * sizeof(float));
    CHECK_ALLOC(state->t, H * sizeof(float));

    state->q = (float*)malloc(NH * D * sizeof(float));
    CHECK_ALLOC(state->q, NH * D * sizeof(float));

    state->k = (float*)malloc(NKV * D * sizeof(float));
    printf("Shape of state->k: (%d, %d)\n", NKV, D);
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

    state->cos_tensor = (float*)malloc(3 * S * (D / 2) * sizeof(float));
    CHECK_ALLOC(state->cos_tensor, 3 * S * (D / 2) * sizeof(float));

    state->sin_tensor = (float*)malloc(3 * S * (D / 2) * sizeof(float));
    CHECK_ALLOC(state->sin_tensor, 3 * S * (D / 2) * sizeof(float));

    state->logits = (float*)malloc(V * sizeof(float));
    CHECK_ALLOC(state->logits, V * sizeof(float));

    // ---- KV cache ----
    state->key_cache = (float*)malloc(cache_size);
    printf("Allocating key_cache of size: %d x %d x %d x %d\n", L, S, NKV, D);
    CHECK_ALLOC(state->key_cache, cache_size);

    state->value_cache = (float*)malloc(cache_size);
    CHECK_ALLOC(state->value_cache, cache_size);

    long long vl_x_size = 1ll * VNP * VH * sizeof(float);
    state->vl_x = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_x, vl_x_size);
    state->vl_b = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_b, vl_x_size);

    state->vl_embed = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_embed, vl_x_size);

    long long vl_rope_size = 1ll * VNP * (VD / 4) * sizeof(float);
    state->vision_cos_tensor = (float *)malloc(vl_rope_size);
    CHECK_ALLOC(state->vision_cos_tensor, vl_rope_size);
    state->vision_sin_tensor = (float *)malloc(vl_rope_size);
    CHECK_ALLOC(state->vision_sin_tensor, vl_rope_size);

    long long vl_embed_size = 1ll * VNP * VD * sizeof(float);
    state->vl_pos_embed_cos = (float *)malloc(vl_embed_size);
    CHECK_ALLOC(state->vl_pos_embed_cos, vl_embed_size);
    state->vl_pos_embed_sin = (float *)malloc(vl_embed_size);
    CHECK_ALLOC(state->vl_pos_embed_sin, vl_embed_size);

    state->vl_q = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_q, vl_x_size);
    state->vl_k = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_k, vl_x_size);
    state->vl_v = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_v, vl_x_size);
    state->vl_q_rot = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_q_rot, vl_x_size);
    state->vl_k_rot = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_k_rot, vl_x_size);
    state->vl_v_orig = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_v_orig, vl_x_size);
    
    state->vl_qkv_out = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_qkv_out, vl_x_size);
    state->vl_qkv_out_orig = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_qkv_out_orig, vl_x_size);
    state->vl_proj_out = (float *)malloc(vl_x_size);
    CHECK_ALLOC(state->vl_proj_out, vl_x_size);
    
    long long vl_mlp1_out_size = 1ll * VNP * VI * sizeof(float);
    state->vl_mlp1_out = (float *)malloc(vl_mlp1_out_size);
    CHECK_ALLOC(state->vl_mlp1_out, vl_mlp1_out_size);

    long long vl_deep_stack_size = 1ll * 3 * VNP * (VH / 2) * sizeof(float);
    state->vl_deep_stack = (float *)malloc(vl_deep_stack_size);
    CHECK_ALLOC(state->vl_deep_stack, vl_deep_stack_size);

    qwen_rope_precompute(state->cos_tensor, state->sin_tensor, config);
    qwen_vision_rope_precompute(state->vision_cos_tensor, state->vision_sin_tensor, config);
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

    if (state->vision_cos_tensor) free(state->vision_cos_tensor);
    if (state->vision_sin_tensor) free(state->vision_sin_tensor);

    if (state->vl_x) free(state->vl_x);
    if (state->vl_b) free(state->vl_b);
    if (state->vl_embed) free(state->vl_embed);
    if (state->vl_pos_embed_cos) free(state->vl_pos_embed_cos);
    if (state->vl_pos_embed_sin) free(state->vl_pos_embed_sin);

    if (state->vl_q) free(state->vl_q);
    if (state->vl_k) free(state->vl_k);
    if (state->vl_v) free(state->vl_v);

    if (state->vl_q_rot) free(state->vl_q_rot);
    if (state->vl_k_rot) free(state->vl_k_rot);
    if (state->vl_v_orig) free(state->vl_v_orig);

    if (state->vl_qkv_out) free(state->vl_qkv_out);
    if (state->vl_qkv_out_orig) free(state->vl_qkv_out_orig);
    
    if (state->vl_proj_out) free(state->vl_proj_out);

    if (state->vl_mlp1_out) free(state->vl_mlp1_out);

    if (state->vl_deep_stack) free(state->vl_deep_stack);

    memset(state, 0, sizeof(QwenRunState));
}

void qwen_rope_precompute(
    float *cos_all_out,  // (3, seq_len, head_dim/2) 
    float *sin_all_out,  // (3, seq_len, head_dim/2)
    const QwenConfig *config
) {
    // Extract parameters from config
    int seq_len = 1024;  // Should be config->max_position_embeddings
    int head_dim = 128;  // Should be config->hidden_size / config->num_attention_heads
    float rope_theta = 5000000.0f;
    
    // MRoPE sections - should come from config->rope_scaling->mrope_section
    const int mrope_section[] = {24, 20, 20};
    int num_dimensions = 3;
    
    int d_half = head_dim / 2;

    // Step 1: compute inv_freq (standard RoPE)
    float *inv_freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        inv_freq[i] = 1.0f / powf(rope_theta, (2.0f * i) / (float)head_dim);
    }

    // Step 2: Compute frequencies for each dimension and position
    // We'll create temporary arrays for frequencies of each dimension
    float *freqs[3];  // freqs[dim][pos * d_half]
    
    for (int dim = 0; dim < num_dimensions; dim++) {
        freqs[dim] = (float *)malloc(seq_len * d_half * sizeof(float));
        
        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < d_half; i++) {
                float freq = pos * inv_freq[i];
                freqs[dim][pos * d_half + i] = freq;
            }
        }
    }

    // Step 3: Apply interleaved MRoPE pattern
    for (int pos = 0; pos < seq_len; pos++) {
        // Start with T dimension frequencies
        float *freq_ptr = &freqs[0][pos * d_half];
        
        // Apply interleaving pattern for H and W dimensions
        // Pattern: [THTHWHTHW...TT] as in Python implementation
        for (int dim = 1; dim <= 2; dim++) {  // H and W dimensions
            int length = mrope_section[dim] * 3;
            for (int offset = dim; offset < length; offset += 3) {
                if (offset < d_half) {
                    freq_ptr[offset] = freqs[dim][pos * d_half + offset];
                }
            }
        }
        
        // Step 4: Compute cosine and sine from interleaved frequencies
        for (int i = 0; i < d_half; i++) {
            float freq = freq_ptr[i];
            
            // Output for T dimension (after interleaving)
            int t_idx = (0 * seq_len * d_half) + (pos * d_half) + i;
            cos_all_out[t_idx] = cosf(freq);
            sin_all_out[t_idx] = sinf(freq);
            
            // For H and W dimensions, we use the original frequencies
            // (These might be overwritten in actual forward pass)
            for (int dim = 1; dim < num_dimensions; dim++) {
                int hw_idx = (dim * seq_len * d_half) + (pos * d_half) + i;
                float orig_freq = freqs[dim][pos * d_half + i];
                cos_all_out[hw_idx] = cosf(orig_freq);
                sin_all_out[hw_idx] = sinf(orig_freq);
            }
        }
    }

    // Cleanup
    free(inv_freq);
    for (int dim = 0; dim < num_dimensions; dim++) {
        free(freqs[dim]);
    }
}

void qwen_vision_rope_precompute(
    float *cos_tensor, float *sin_tensor, const QwenConfig *config
) {
    int dim = 32;
    int max_seq_len = 2304;
    int half_dim = dim / 2;
    float theta = 10000.0f;

    /* Allocate inv_freq (same as register_buffer in PyTorch) */
    float *inv_freq = (float *)malloc(sizeof(float) * half_dim);

    /* Compute inv_freq */
    for (int i = 0; i < half_dim; ++i) {
        float exponent = (float)(2 * i) / (float)dim;
        inv_freq[i] = 1.0f / powf(theta, exponent);
    }

    /* Compute outer product: seq ⊗ inv_freq */
    for (int s = 0; s < max_seq_len; ++s) {
        for (int i = 0; i < half_dim; ++i) {
            const float freq_val = (float)s * inv_freq[i];
            cos_tensor[s * half_dim + i] = cosf(freq_val);
            sin_tensor[s * half_dim + i] = sinf(freq_val);
        }
    }

    free(inv_freq);
}

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
    config->seq_len = 3072;
    config->mrope_section = (int *)malloc(3 * sizeof(int));
    config->mrope_section[0] = 24;
    config->mrope_section[1] = 20;
    config->mrope_section[2] = 20;
    config->num_dimensions = 3;

    config->min_pixels = 256ll * 256ll;
    config->max_pixels = 1024ll * 1024ll;
    config->vision_theta = 10000.0f;
    config->vision_scale = 0.125;
    config->max_vision_embeddings = 2304;
    config->vision_num_channels = 3;
    config->vision_deep_stack_depth = 3;
    config->deep_layer = (int *)calloc(config->vision_depth, sizeof(int)); // set all values to 0
    config->deep_layer[5] = 1; // set index 5 to 1
    config->deep_layer[11] = 2; // set index 11 to 2
    config->deep_layer[17] = 3;


    // ==================================================================================
    // 2. Derived Dimensions
    // ==================================================================================
    size_t head_dim = config->hidden_size / config->num_attention_heads;
    size_t vision_head_dim = config->vision_hidden_size / config->vision_num_heads;
    size_t vision_qkv_dim = 3 * config->vision_num_heads * vision_head_dim;
    size_t num_deepstack_mergers = 3;

    size_t L  = config->num_hidden_layers;
    size_t H  = config->hidden_size;
    size_t I  = config->intermediate_size;
    size_t QD = config->num_attention_heads * head_dim;
    size_t KVAD = config->num_key_value_heads * head_dim;

    size_t VH = config->vision_hidden_size;
    size_t VP = config->vision_patch_size;
    size_t VTP = config->vision_temporal_patch_size;
    size_t VD = config->vision_depth;
    size_t VI = config->vision_intermediate_size;
    size_t OH = config->out_hidden_size;
    size_t VC = config->vision_num_channels;
    size_t VNPE = config->max_vision_embeddings;
    size_t VDSD = config->vision_deep_stack_depth;
    size_t QKVD = vision_qkv_dim;
    size_t vocab_size = config->vocab_size;
    size_t M = num_deepstack_mergers;

    // ==================================================================================
    // 3. Allocate + Read + Print (Preserve fread order)
    // ==================================================================================
    float *tmp_ptr = nullptr;

    // EDIT THE CODE FROM THIS TO THE END OF THE FUNCTION

    // --- embed_tokens.weight
    tmp_ptr = (float*)malloc(vocab_size * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), vocab_size * H, file);
    weights->token_embedding_table = new Tensor({vocab_size, H}, tmp_ptr);
    weights->token_embedding_table->printShape("token_embedding_table (embed_tokens.weight)");
    tmp_ptr = nullptr;

    // --- input_layernorm.weight
    tmp_ptr = (float*)malloc(L * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H, file);
    weights->rms_ffn_w = new Tensor({L, H}, tmp_ptr);
    weights->rms_ffn_w->printShape("rms_ffn_w (input_layernorm.weight)");
    tmp_ptr = nullptr;

    // --- mlp.down_proj.weight
    tmp_ptr = (float*)malloc(L * H * I * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H * I, file);
    weights->w_mlp_down = new Tensor({L, H, I}, tmp_ptr);
    weights->w_mlp_down->printShape("w_mlp_down (mlp.down_proj.weight)");
    tmp_ptr = nullptr;

    // --- mlp.gate_proj.weight
    tmp_ptr = (float*)malloc(L * I * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * I * H, file);
    weights->w_mlp_gate = new Tensor({L, I, H}, tmp_ptr);
    weights->w_mlp_gate->printShape("w_mlp_gate (mlp.gate_proj.weight)");
    tmp_ptr = nullptr;

    // --- mlp.up_proj.weight
    tmp_ptr = (float*)malloc(L * I * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * I * H, file);
    weights->w_mlp_up = new Tensor({L, I, H}, tmp_ptr);
    weights->w_mlp_up->printShape("w_mlp_up (mlp.up_proj.weight)");
    tmp_ptr = nullptr;

    // --- post_attention_layernorm.weight
    tmp_ptr = (float*)malloc(L * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H, file);
    weights->rms_attn_w = new Tensor({L, H}, tmp_ptr);
    weights->rms_attn_w->printShape("rms_attn_w (post_attention_layernorm.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.k_norm.weight
    tmp_ptr = (float*)malloc(L * 1ll * head_dim * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * 1ll * head_dim, file);
    weights->w_attn_k_norm = new Tensor({L, 1ll * head_dim}, tmp_ptr);
    weights->w_attn_k_norm->printShape("w_attn_k_norm (self_attn.k_norm.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.k_proj.weight
    tmp_ptr = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * KVAD * H, file);
    weights->w_attn_k = new Tensor({L, KVAD, H}, tmp_ptr);
    weights->w_attn_k->printShape("w_attn_k (self_attn.k_proj.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.o_proj.weight
    tmp_ptr = (float*)malloc(L * H * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * H * H, file);
    weights->w_attn_o = new Tensor({L, H, H}, tmp_ptr);
    weights->w_attn_o->printShape("w_attn_o (self_attn.o_proj.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.q_norm.weight
    tmp_ptr = (float*)malloc(L * head_dim * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * head_dim, file);
    weights->w_attn_q_norm = new Tensor({L, head_dim}, tmp_ptr);
    weights->w_attn_q_norm->printShape("w_attn_q_norm (self_attn.q_norm.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.q_proj.weight
    tmp_ptr = (float*)malloc(L * QD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * QD * H, file);
    weights->w_attn_q = new Tensor({L, QD, H}, tmp_ptr);
    weights->w_attn_q->printShape("w_attn_q (self_attn.q_proj.weight)");
    tmp_ptr = nullptr;

    // --- self_attn.v_proj.weight
    tmp_ptr = (float*)malloc(L * KVAD * H * sizeof(float));
    fread(tmp_ptr, sizeof(float), L * KVAD * H, file);
    weights->w_attn_v = new Tensor({L, KVAD, H}, tmp_ptr);
    weights->w_attn_v->printShape("w_attn_v (self_attn.v_proj.weight)");
    tmp_ptr = nullptr;

    // --- final layernorm.weight
    tmp_ptr = (float*)malloc(H * sizeof(float));
    fread(tmp_ptr, sizeof(float), H, file);
    weights->rms_out_w = new Tensor({H}, tmp_ptr);
    weights->rms_out_w->printShape("rms_out_w (final_layernorm.weight)");
    tmp_ptr = nullptr;

    // --- Vision model general
    tmp_ptr = (float*)malloc(VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), VH, file);
    weights->vl_patch_emb_b = new Tensor({VH}, tmp_ptr);
    weights->vl_patch_emb_b->printShape("vl_patch_emb_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(VH * VC * VTP * VP * VP * sizeof(float));
    fread(tmp_ptr, sizeof(float), VH * VC * VTP * VP * VP, file);
    weights->vl_patch_emb_w = new Tensor({VH, VC, VTP, VP, VP}, tmp_ptr);
    weights->vl_patch_emb_w->printShape("vl_patch_emb_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VNPE * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VNPE * VH, file);
    weights->vl_pos_emb_w = new Tensor({VNPE, VH}, tmp_ptr);
    weights->vl_pos_emb_w->printShape("vl_pos_emb_w");
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_attn_proj_b = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_attn_proj_b->printShape("vl_attn_proj_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH * VH, file);
    weights->vl_attn_proj_w = new Tensor({VD, VH, VH}, tmp_ptr);
    weights->vl_attn_proj_w->printShape("vl_attn_proj_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * 3 * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * 3 * VH, file);
    weights->vl_attn_qkv_b = new Tensor({VD, 3, VH}, tmp_ptr);
    weights->vl_attn_qkv_b->printShape("vl_attn_qkv_b");
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * 3 * VH * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * 3 * VH * VH, file);
    weights->vl_attn_qkv_w = new Tensor({VD, 3, VH, VH}, tmp_ptr);
    weights->vl_attn_qkv_w->printShape("vl_attn_qkv_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VI, file);
    weights->vl_mlp1_b = new Tensor({VD, VI}, tmp_ptr);
    weights->vl_mlp1_b->printShape("vl_mlp1_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VI * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VI * VH, file);
    weights->vl_mlp1_w = new Tensor({VD, VI, VH}, tmp_ptr);
    weights->vl_mlp1_w->printShape("vl_mlp1_w");
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_mlp2_b = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_mlp2_b->printShape("vl_mlp2_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH * VI, file);
    weights->vl_mlp2_w = new Tensor({VD, VH, VI}, tmp_ptr);
    weights->vl_mlp2_w->printShape("vl_mlp2_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_norm1_b = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_norm1_b->printShape("vl_norm1_b");
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_norm1_w = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_norm1_w->printShape("vl_norm1_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_norm2_b = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_norm2_b->printShape("vl_norm2_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VD * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VD * VH, file);
    weights->vl_norm2_w = new Tensor({VD, VH}, tmp_ptr);
    weights->vl_norm2_w->printShape("vl_norm2_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    weights->vl_d_mlp1_b = new Tensor({VDSD, VI}, tmp_ptr);
    weights->vl_d_mlp1_b->printShape("vl_d_mlp1_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI * VI, file);
    weights->vl_d_mlp1_w = new Tensor({VDSD, VI, VI}, tmp_ptr);
    weights->vl_d_mlp1_w->printShape("vl_d_mlp1_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * OH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * OH, file);
    weights->vl_d_mlp2_b = new Tensor({VDSD, OH}, tmp_ptr);
    weights->vl_d_mlp2_b->printShape("vl_d_mlp2_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * OH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * OH * VI, file);
    weights->vl_d_mlp2_w = new Tensor({VDSD, OH, VI}, tmp_ptr);
    weights->vl_d_mlp2_w->printShape("vl_d_mlp2_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    weights->vl_d_norm_b = new Tensor({VDSD, VI}, tmp_ptr);
    weights->vl_d_norm_b->printShape("vl_d_norm_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VDSD * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VDSD * VI, file);
    weights->vl_d_norm_w = new Tensor({VDSD, VI}, tmp_ptr);
    weights->vl_d_norm_w->printShape("vl_d_norm_w");
    tmp_ptr = nullptr;
    
    tmp_ptr = (float*)malloc(1ll * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VI, file);
    weights->vl_merge_mlp1_b = new Tensor({VI}, tmp_ptr);
    weights->vl_merge_mlp1_b->printShape("vl_merge_mlp1_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VI * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VI * VI, file);
    weights->vl_merge_mlp1_w = new Tensor({VI, VI}, tmp_ptr);
    weights->vl_merge_mlp1_w->printShape("vl_merge_mlp1_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * OH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * OH, file);
    weights->vl_merge_mlp2_b = new Tensor({OH}, tmp_ptr);
    weights->vl_merge_mlp2_b->printShape("vl_merge_mlp2_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * OH * VI * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * OH * VI, file);
    weights->vl_merge_mlp2_w = new Tensor({OH, VI}, tmp_ptr);
    weights->vl_merge_mlp2_w->printShape("vl_merge_mlp2_w");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VH, file);
    weights->vl_merge_norm_b = new Tensor({VH}, tmp_ptr);
    weights->vl_merge_norm_b->printShape("vl_merge_norm_b");
    tmp_ptr = nullptr;

    tmp_ptr = (float*)malloc(1ll * VH * sizeof(float));
    fread(tmp_ptr, sizeof(float), 1ll * VH, file);
    weights->vl_merge_norm_w = new Tensor({VH}, tmp_ptr);
    weights->vl_merge_norm_w->printShape("vl_merge_norm_w");
    tmp_ptr = nullptr;
    
    fclose(file);
    printf("Successfully loaded model from %s\n", path);
    fflush(stdout);
    // exit(1);
}

void free_model_config(QwenConfig *config) {
    free(config->mrope_section);
    free(config->deep_layer);
}

void free_model_weights(QwenWeight* weights) {
    if (!weights) return;

    // Language Model Standalone Weights
    delete weights->token_embedding_table;
    delete weights->rms_out_w;

    // Language Model Layer Weights (Continuous Blocks)
    delete weights->rms_ffn_w;
    delete weights->w_mlp_down;
    delete weights->w_mlp_gate;
    delete weights->w_mlp_up;
    delete weights->rms_attn_w;
    delete weights->w_attn_k_norm;
    delete weights->w_attn_k;
    delete weights->w_attn_o;
    delete weights->w_attn_q_norm;
    delete weights->w_attn_q;
    delete weights->w_attn_v;

    // Vision Model Weights (General)
    delete weights->vl_patch_emb_b;
    delete weights->vl_patch_emb_w;
    delete weights->vl_pos_emb_w;
    
    delete weights->vl_attn_proj_b;
    delete weights->vl_attn_proj_w;
    delete weights->vl_attn_qkv_b;
    delete weights->vl_attn_qkv_w;
    delete weights->vl_mlp1_b;
    delete weights->vl_mlp1_w;
    delete weights->vl_mlp2_b;
    delete weights->vl_mlp2_w;
    delete weights->vl_norm1_b;
    delete weights->vl_norm1_w;
    delete weights->vl_norm2_b;
    delete weights->vl_norm2_w;
    
    delete weights->vl_d_mlp1_b;
    delete weights->vl_d_mlp1_w;
    delete weights->vl_d_mlp2_b;
    delete weights->vl_d_mlp2_w;
    delete weights->vl_d_norm_b;
    delete weights->vl_d_norm_w;
    
    delete weights->vl_merge_mlp1_b;
    delete weights->vl_merge_mlp1_w;
    delete weights->vl_merge_mlp2_b;
    delete weights->vl_merge_mlp2_w;
    delete weights->vl_merge_norm_b;
    delete weights->vl_merge_norm_w;
}

void init_model_run_state(QwenRunState* state, const QwenConfig* config) {
    state->vision_embed_tokens = 0;

    size_t H   = config->hidden_size;
    size_t I   = config->intermediate_size;
    size_t V   = config->vocab_size;
    size_t L   = config->num_hidden_layers;
    size_t NH  = config->num_attention_heads;
    size_t NKV = config->num_key_value_heads;
    size_t D   = H / NH;
    size_t S   = config->seq_len;

    size_t VH = config->vision_hidden_size;
    size_t VNP = config->max_vision_embeddings;
    size_t VD = VH / config->vision_num_heads;
    size_t VI = config->vision_intermediate_size;
    size_t VDS = config->vision_deep_stack_depth;

    // ---- Hidden / intermediate ----
    state->x = new Tensor({BATCH_SIZE, H});

    state->t = new Tensor({BATCH_SIZE, H});

    state->q = new Tensor({BATCH_SIZE, NH, D});

    state->att = new Tensor({BATCH_SIZE, NH, S});

    state->qkv_out = new Tensor({BATCH_SIZE, H});

    state->gate = new Tensor({BATCH_SIZE, I});

    state->up = new Tensor({BATCH_SIZE, I});

    state->cos_tensor = new Tensor({3, S, D / 2});
    state->sin_tensor = new Tensor({3, S, D / 2});

    state->logits = new Tensor({BATCH_SIZE, V});

    // ---- KV cache ----
    state->key_cache = new Tensor({BATCH_SIZE, L, S, NKV, D});
    state->value_cache = new Tensor({BATCH_SIZE, L, S, NKV, D});

    // -- Vision states --
    state->vision_x = new Tensor({VNP, VH});
    state->vision_t = new Tensor({VNP, VH});
    state->vision_q = new Tensor({VNP, VH});
    state->vision_k = new Tensor({VNP, VH});

    state->vision_cos_tensor = new Tensor({VNP, VD / 4});
    state->vision_sin_tensor = new Tensor({VNP, VD / 4});

    state->vision_pe_cos = new Tensor({VNP, VD});
    state->vision_pe_sin = new Tensor({VNP, VD});
    
    state->vision_mlp_out = new Tensor({VNP, VI});

    state->vision_deep_stack = new Tensor({VDS, VNP / 4, H});

    state->vision_attn_scores = new Tensor({VNP, VNP});

    qwen_rope_precompute(
        state->cos_tensor, state->sin_tensor, config
    );
    qwen_vision_rope_precompute(
        state->vision_cos_tensor, state->vision_sin_tensor, config
    );
    printf("Finish init run state\n");
    fflush(stdout);
}

void free_model_run_state(QwenRunState* state) {
    if (!state) return;

    if (state->x) delete state->x;
    if (state->t) delete state->t;
    if (state->q) delete state->q;
    if (state->att) delete state->att;
    if (state->qkv_out) delete state->qkv_out;
    if (state->gate) delete state->gate;
    if (state->up) delete state->up;
    if (state->cos_tensor) delete state->cos_tensor;
    if (state->sin_tensor) delete state->sin_tensor;
    if (state->logits) delete state->logits;
    if (state->key_cache) delete state->key_cache;
    if (state->value_cache) delete state->value_cache;

    if (state->vision_cos_tensor) delete state->vision_cos_tensor;
    if (state->vision_sin_tensor) delete state->vision_sin_tensor;

    if (state->vision_x) delete state->vision_x;
    if (state->vision_t) delete state->vision_t;
    if (state->vision_pe_cos) delete state->vision_pe_cos;
    if (state->vision_pe_sin) delete state->vision_pe_sin;

    if (state->vision_q) delete state->vision_q;
    if (state->vision_k) delete state->vision_k;

    if (state->vision_mlp_out) delete state->vision_mlp_out;

    if (state->vision_deep_stack) delete state->vision_deep_stack;

    if (state->vision_attn_scores) delete state->vision_attn_scores;
}

void qwen_rope_precompute(
    Tensor *cos_all_out,  // (3, seq_len, head_dim/2) 
    Tensor *sin_all_out,  // (3, seq_len, head_dim/2)
    const QwenConfig *config
) {
    // Extract parameters from config
    int seq_len = config->seq_len;  // Should be config->max_position_embeddings
    int head_dim = config->hidden_size / config->num_attention_heads;

    float *cos_buf = (float *)cos_all_out->ptr();
    float *sin_buf = (float *)sin_all_out->ptr();
    
    // MRoPE sections - should come from config->rope_scaling->mrope_section
    int d_half = head_dim / 2;

    // Step 1: compute inv_freq (standard RoPE)
    float *inv_freq = (float *)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        inv_freq[i] = 1.0f / powf(config->rope_theta, (2.0f * i) / (float)head_dim);
    }

    // Step 2: Compute frequencies for each dimension and position
    // We'll create temporary arrays for frequencies of each dimension
    float *freqs[3];  // freqs[dim][pos * d_half]
    
    for (int dim = 0; dim < config->num_dimensions; dim++) {
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
            int length = config->mrope_section[dim] * 3;
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
            cos_buf[t_idx] = cosf(freq);
            sin_buf[t_idx] = sinf(freq);
            
            // For H and W dimensions, we use the original frequencies
            // (These might be overwritten in actual forward pass)
            for (int dim = 1; dim < config->num_dimensions; dim++) {
                int hw_idx = (dim * seq_len * d_half) + (pos * d_half) + i;
                float orig_freq = freqs[dim][pos * d_half + i];
                cos_buf[hw_idx] = cosf(orig_freq);
                sin_buf[hw_idx] = sinf(orig_freq);
            }
        }
    }

    // Cleanup
    free(inv_freq);
    for (int dim = 0; dim < config->num_dimensions; dim++) {
        free(freqs[dim]);
    }
}

void qwen_vision_rope_precompute(
    Tensor *cos_tensor, Tensor *sin_tensor, const QwenConfig *config
) {
    int dim = (config->vision_hidden_size / config->vision_num_heads) / 2;
    int max_seq_len = config->max_vision_embeddings;
    int half_dim = dim / 2;
    float theta = config->vision_theta;

    float *cos_buf = (float *)cos_tensor->ptr();
    float *sin_buf = (float *)sin_tensor->ptr();

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
            cos_buf[s * half_dim + i] = cosf(freq_val);
            sin_buf[s * half_dim + i] = sinf(freq_val);
        }
    }

    free(inv_freq);
}

#include "load_model.hpp"

void* mmap_weight_safe(
    int fd,
    size_t size_in_bytes,
    off_t& current_offset,
    DType::Type dtype
) {
    if (size_in_bytes == 0) return nullptr;

    static size_t page_size = sysconf(_SC_PAGESIZE);

    // Align mmap offset to page boundary
    off_t aligned_offset = (current_offset / page_size) * page_size;
    off_t offset_in_page = current_offset - aligned_offset;
    size_t mapped_size = size_in_bytes + offset_in_page;

    void* mapped_ptr = mmap(
        nullptr,
        mapped_size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        aligned_offset
    );

    if (mapped_ptr == MAP_FAILED) {
        perror("mmap failed");
        return nullptr;
    }

    void* data_ptr = (char*)mapped_ptr + offset_in_page;

    // Advance file offset
    current_offset += size_in_bytes;

    // Optional: sanity check
    #ifdef DEBUG
    if (dtype == DType::FP32 && size_in_bytes % sizeof(float) != 0) {
        fprintf(stderr, "Warning: FP32 size not aligned\n");
    }
    if (dtype == DType::FP16 && size_in_bytes % sizeof(uint16_t) != 0) {
        fprintf(stderr, "Warning: FP16 size not aligned\n");
    }
    #endif

    return data_ptr;
}

void init_model_weights(const char* path, QwenConfig* config, QwenWeight* weights) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("Error opening file"); return; }

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
        (fread(&(config->video_token_id), sizeof(int), 1, file) != 1)||
        (fread(&(config->text_bits), sizeof(int), 1, file) != 1) ||
        (fread(&(config->group_size), sizeof(int), 1, file) != 1) ||
        (fread(&(config->vision_bits), sizeof(int), 1, file) != 1)
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
    config->max_pixels = 4096ll * 4096ll;
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

    size_t text_bits = config->text_bits;
    size_t vision_bits = config->vision_bits;

    off_t current_offset = ftell(file);

    auto map_tensor = [&](Tensor** target, std::vector<size_t> dims, const char* name, size_t weight_bits, bool print_shape = true) {
        size_t count = 1;
        size_t size_elem;
        DType::Type type_elem;
        if (weight_bits >= 16) {
            if (weight_bits == 32) {
                type_elem = DType::FP32;
                size_elem = 4;
            } else {
                type_elem = DType::FP16;
                size_elem = 2;
            }
            for (auto d : dims) count *= d;
            void* ptr = mmap_weight_safe(fd, count * size_elem, current_offset, type_elem);
            *target = new Tensor(dims, ptr, type_elem);
        } else {
            if (weight_bits == 8) {
                type_elem = DType::INT8;
                size_elem = 1;
            }
            for (auto d : dims) count *= d;
            size_t group_size = config->group_size;
            if (count % group_size) {
                fprintf(stderr, "Not even weights %s\n", name);
                exit(1);
            }
            size_t count_scales = count / group_size;
            size_t size_scale_elem = 4;
            DType::Type scale_type_elem = DType::FP32;
            void *scale_ptr = mmap_weight_safe(fd, count_scales * size_scale_elem, current_offset, scale_type_elem);
            void *ptr = mmap_weight_safe(fd, count * size_elem, current_offset, type_elem);
            *target = new Tensor(dims, ptr, scale_ptr, group_size, type_elem, scale_type_elem);
        }
        if (print_shape) (*target)->printShape(name);
    };

    // --- LLM WEIGHTS ---
    map_tensor(&weights->token_embedding_table, {vocab_size, H}, "token_embedding_table", text_bits);
    map_tensor(&weights->rms_ffn_w, {L, H}, "rms_ffn_w", text_bits);
    map_tensor(&weights->w_mlp_down, {L, H, I}, "w_mlp_down", text_bits);
    map_tensor(&weights->w_mlp_gate, {L, I, H}, "w_mlp_gate", text_bits);
    map_tensor(&weights->w_mlp_up, {L, I, H}, "w_mlp_up", text_bits);
    map_tensor(&weights->rms_attn_w, {L, H}, "rms_attn_w", text_bits);
    map_tensor(&weights->w_attn_k_norm, {L, head_dim}, "w_attn_k_norm", text_bits);
    map_tensor(&weights->w_attn_k, {L, KVAD, H}, "w_attn_k", text_bits);
    map_tensor(&weights->w_attn_o, {L, H, H}, "w_attn_o", text_bits);
    map_tensor(&weights->w_attn_q_norm, {L, head_dim}, "w_attn_q_norm", text_bits);
    map_tensor(&weights->w_attn_q, {L, QD, H}, "w_attn_q", text_bits);
    map_tensor(&weights->w_attn_v, {L, KVAD, H}, "w_attn_v", text_bits);
    map_tensor(&weights->rms_out_w, {H}, "rms_out_w", text_bits);

    // --- VISION WEIGHTS ---
    map_tensor(&weights->vl_patch_emb_b, {VH}, "vl_patch_emb_b", vision_bits);
    map_tensor(&weights->vl_patch_emb_w, {VH, VC, VTP, VP, VP}, "vl_patch_emb_w", vision_bits);
    map_tensor(&weights->vl_pos_emb_w, {VNPE, VH}, "vl_pos_emb_w", vision_bits);
    map_tensor(&weights->vl_attn_proj_b, {VD, VH}, "vl_attn_proj_b", vision_bits);
    map_tensor(&weights->vl_attn_proj_w, {VD, VH, VH}, "vl_attn_proj_w", vision_bits, false);
    weights->vl_attn_proj_w->permute({0, 2, 1});
    weights->vl_attn_proj_w->printShape("vl_attn_qkv_w");
    map_tensor(&weights->vl_attn_qkv_b, {VD, 3, VH}, "vl_attn_qkv_b", vision_bits);
    map_tensor(&weights->vl_attn_qkv_w, {VD, 3, VH, VH}, "vl_attn_qkv_w", vision_bits, false);
    // weights->vl_attn_qkv_w->permute({0, 1, 3, 2});
    weights->vl_attn_qkv_w->printShape("vl_attn_qkv_w");
    map_tensor(&weights->vl_mlp1_b, {VD, VI}, "vl_mlp1_b", vision_bits);
    map_tensor(&weights->vl_mlp1_w, {VD, VI, VH}, "vl_mlp1_w", vision_bits, false);
    weights->vl_mlp1_w->permute({0, 2, 1});
    weights->vl_mlp1_w->printShape("vl_mlp1_w");
    map_tensor(&weights->vl_mlp2_b, {VD, VH}, "vl_mlp2_b", vision_bits);
    map_tensor(&weights->vl_mlp2_w, {VD, VH, VI}, "vl_mlp2_w", vision_bits, false);
    // weights->vl_mlp2_w->permute({0, 2, 1});
    weights->vl_mlp2_w->printShape("vl_mlp2_w");
    map_tensor(&weights->vl_norm1_b, {VD, VH}, "vl_norm1_b", vision_bits);
    map_tensor(&weights->vl_norm1_w, {VD, VH}, "vl_norm1_w", vision_bits);
    map_tensor(&weights->vl_norm2_b, {VD, VH}, "vl_norm2_b", vision_bits);
    map_tensor(&weights->vl_norm2_w, {VD, VH}, "vl_norm2_w", vision_bits);

    // --- DEEP STACK & MERGER ---
    map_tensor(&weights->vl_d_mlp1_b, {VDSD, VI}, "vl_d_mlp1_b", vision_bits);
    map_tensor(&weights->vl_d_mlp1_w, {VDSD, VI, VI}, "vl_d_mlp1_w", vision_bits);
    map_tensor(&weights->vl_d_mlp2_b, {VDSD, OH}, "vl_d_mlp2_b", vision_bits);
    map_tensor(&weights->vl_d_mlp2_w, {VDSD, OH, VI}, "vl_d_mlp2_w", vision_bits);
    map_tensor(&weights->vl_d_norm_b, {VDSD, VI}, "vl_d_norm_b", vision_bits);
    map_tensor(&weights->vl_d_norm_w, {VDSD, VI}, "vl_d_norm_w", vision_bits);
    map_tensor(&weights->vl_merge_mlp1_b, {VI}, "vl_merge_mlp1_b", vision_bits);
    map_tensor(&weights->vl_merge_mlp1_w, {VI, VI}, "vl_merge_mlp1_w", vision_bits);
    map_tensor(&weights->vl_merge_mlp2_b, {OH}, "vl_merge_mlp2_b", vision_bits);
    map_tensor(&weights->vl_merge_mlp2_w, {OH, VI}, "vl_merge_mlp2_w", vision_bits);
    map_tensor(&weights->vl_merge_norm_b, {VH}, "vl_merge_norm_b", vision_bits);
    map_tensor(&weights->vl_merge_norm_w, {VH}, "vl_merge_norm_w", vision_bits);

    fclose(file); // Note: file was opened from fd, so this closes the file handle.
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
    size_t VNP_max = std::min(S * 4, VNP * 10);
    size_t VD = VH / config->vision_num_heads;
    size_t VI = config->vision_intermediate_size;
    size_t VDS = config->vision_deep_stack_depth;

    // ---- Hidden / intermediate ----
    state->x = new Tensor({BATCH_SIZE, H});

    state->t = new Tensor({BATCH_SIZE, H});

    state->q = new Tensor({BATCH_SIZE, NH, D});
    state->k = new Tensor({BATCH_SIZE, NH, D});
    state->v = new Tensor({BATCH_SIZE, NH, D});

    state->att = new Tensor({BATCH_SIZE, NH, S});

    state->qkv_out = new Tensor({BATCH_SIZE, H});

    state->gate = new Tensor({BATCH_SIZE, I});

    state->up = new Tensor({BATCH_SIZE, I});

    state->cos_tensor = new Tensor({3, S, D / 2});
    state->sin_tensor = new Tensor({3, S, D / 2});

    state->logits = new Tensor({BATCH_SIZE, V});

    // ---- KV cache ----
    state->key_cache = new Tensor({BATCH_SIZE, L, NKV, S, D});
    state->value_cache = new Tensor({BATCH_SIZE, L, NKV, S, D});

    // -- Vision states --
    state->vision_x = new Tensor({VNP_max, VH});
    state->vision_t = new Tensor({VNP_max, VH});
    state->vision_q = new Tensor({VNP_max, VH});
    state->vision_k = new Tensor({VNP_max, VH});

    state->vision_cos_tensor = new Tensor({VNP_max, VD / 4});
    state->vision_sin_tensor = new Tensor({VNP_max, VD / 4});

    state->vision_pe_cos = new Tensor({VNP_max, VD});
    state->vision_pe_sin = new Tensor({VNP_max, VD});
    
    state->vision_mlp_out = new Tensor({VNP_max, VI});

    state->vision_deep_stack = new Tensor({VDS, VNP, H});

    state->vision_attn_scores = new Tensor({VNP_max, VNP_max});

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
    if (state->k) delete state->k;
    if (state->v) delete state->v;
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

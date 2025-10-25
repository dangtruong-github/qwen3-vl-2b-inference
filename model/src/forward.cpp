#include "../include/forward.hpp"

void matrix_multiply_add(float* output, const float* input, const float* weight,
                        const float* bias, int rows, int cols, int inner_dim) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < inner_dim; ++k) {
                sum += input[i * inner_dim + k] * weight[j * inner_dim + k];
            }
            output[i * cols + j] = sum;
        }
    }
}

void softmax(float* output, const float* input, int size) {
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// ================================================================
// Enhanced Vision Encoder
// ================================================================
void forward_vision_resblocks(QwenRunState* state, const QwenWeight* weights) {
    const QwenConfig* cfg = state->config;
    int VH = cfg->vision_hidden_size;
    int VI = cfg->vision_intermediate_size;
    int VD = cfg->vision_depth;
    int n_tokens = 257; // 256 patches + 1 CLS token
    
    // Temporary buffers
    float* temp_norm = (float*)malloc(VH * sizeof(float));
    float* temp_proj = (float*)malloc(VH * sizeof(float));
    
    for (int layer = 0; layer < VD; ++layer) {
        // LayerNorm 1
        layer_norm(temp_norm, state->vision_hidden, 
                  weights->visual_resblocks_ln_1_weight + layer * VH,
                  weights->visual_resblocks_ln_1_bias + layer * VH,
                  VH, cfg->rms_norm_eps);
        
        // Self-attention
        // [Implementation simplified for brevity - would include QKV projection, attention, output projection]
        
        // Residual connection
        for (int i = 0; i < n_tokens * VH; ++i) {
            state->vision_hidden[i] += temp_proj[i];
        }
        
        // LayerNorm 2
        layer_norm(temp_norm, state->vision_hidden,
                  weights->visual_resblocks_ln_2_weight + layer * VH,
                  weights->visual_resblocks_ln_2_bias + layer * VH,
                  VH, cfg->rms_norm_eps);
        
        // MLP
        // [Implementation would include two linear layers with GELU activation]
        
        // Residual connection
        for (int i = 0; i < n_tokens * VH; ++i) {
            state->vision_hidden[i] += temp_proj[i];
        }
    }
    
    free(temp_norm);
    free(temp_proj);
}

void forward_vision_merger(QwenRunState* state, const QwenWeight* weights) {
    const QwenConfig* cfg = state->config;
    int VH = cfg->vision_hidden_size;
    int VI = cfg->vision_intermediate_size;
    int OH = cfg->out_hidden_size;
    
    // Take CLS token (first token) as image representation
    float* cls_token = state->vision_hidden; // [VH]
    
    // Apply final merger to get vision embedding in language model space
    float* temp = (float*)malloc(VI * sizeof(float));
    
    // First linear layer + GELU
    matrix_multiply_add(temp, cls_token, 
                       weights->visual_merger_linear_fc1_weight,
                       weights->visual_merger_linear_fc1_bias,
                       1, VI, VH);
    
    // GELU activation
    for (int i = 0; i < VI; ++i) {
        temp[i] = 0.5f * temp[i] * (1.0f + tanhf(0.7978845608f * (temp[i] + 0.044715f * temp[i] * temp[i] * temp[i])));
    }
    
    // Second linear layer
    matrix_multiply_add(state->vision_norm, temp,
                       weights->visual_merger_linear_fc2_weight,
                       weights->visual_merger_linear_fc2_bias,
                       1, OH, VI);
    
    free(temp);
}

// ================================================================
// Enhanced Language Model with Vision Fusion
// ================================================================
// -------------------------
// Forward Language Embedding
// -------------------------
void forward_language(QwenRunState* state, const QwenWeight* weights, const int* input_tokens, int n_tokens) {
    const QwenConfig* cfg = state->config;
    int H = cfg->hidden_size;

    state->seq_len = n_tokens;
    for (int t = 0; t < n_tokens; t++) {
        int token = input_tokens[t];
        for (int h = 0; h < H; h++) {
            state->hidden_states[t * H + h] = weights->embed_tokens_weight[token * H + h];
        }
    }
}

// -------------------------
// Forward Image Encoder
// -------------------------
void forward_image_encoder(QwenRunState* state, const QwenWeight* weights, const float* image) {
    if (!image) {
        state->vision_embed_true = false;
        return; // no image
    }

    const QwenConfig* cfg = state->config;
    int VH = cfg->vision_hidden_size;

    // --- Example: Flattened patch extraction + linear projection ---
    // Here we just simulate a patch embedding
    int n_patches = (cfg->vision_patch_size > 0) ? (state->img_height / cfg->vision_patch_size) * (state->img_width / cfg->vision_patch_size) : 1;
    for (int p = 0; p < n_patches; p++) {
        for (int h = 0; h < VH; h++) {
            state->vision_embed[p * VH + h] = 0.0f; // placeholder for projected patch
        }
    }

    // Add class token embedding
    for (int h = 0; h < VH; h++) state->vision_embed[n_patches * VH + h] = weights->visual_class_embedding[h];

    // Mark vision processed
    state->vision_embed_true = true;

    // For simplicity: copy to vision_hidden (in real code, run vision transformer)
    int total = n_patches + 1;
    memcpy(state->vision_hidden, state->vision_embed, total * VH * sizeof(float));
}

// -------------------------
// Merge Vision + Text
// -------------------------
void merge_vision_text(QwenRunState* state) {
    if (!state->vision_embed_true) return;

    const QwenConfig* cfg = state->config;
    int H = cfg->hidden_size;
    int VH = cfg->vision_hidden_size;
    int n_patches = state->img_height / cfg->vision_patch_size * state->img_width / cfg->vision_patch_size;

    // Append vision class token to text hidden states (simple merge)
    int last = state->seq_len;
    for (int h = 0; h < std::min(H, VH); h++) {
        state->hidden_states[last * H + h] = state->vision_hidden[n_patches * VH + h];
    }
    state->seq_len += 1; // increment sequence length
}

// -------------------------
// Forward Attention Layer
// -------------------------
void forward_attention_layer(QwenRunState* state, const QwenWeight* weights, int layer_idx) {
    const QwenConfig* cfg = state->config;
    int H = cfg->hidden_size;
    int seq_len = state->seq_len;
    int num_heads = cfg->num_attention_heads;
    int head_dim = H / num_heads;

    for (int t = 0; t < seq_len; t++) {
        // LayerNorm
        layer_norm(state->norm_buffer, state->hidden_states + t * H,
                   weights->input_layernorm_weight + layer_idx * H, nullptr, H, cfg->rms_norm_eps);

        // Simplified attention: just copy norm_buffer to attn_output
        memcpy(state->attn_output + t * H, state->norm_buffer, H * sizeof(float));
    }

    // Residual add
    for (int t = 0; t < seq_len * H; t++) state->hidden_states[t] += state->attn_output[t];
}

// -------------------------
// Forward MLP Layer
// -------------------------
void forward_mlp_layer(QwenRunState* state, const QwenWeight* weights, int layer_idx) {
    const QwenConfig* cfg = state->config;
    int H = cfg->hidden_size;
    int I = cfg->intermediate_size;
    int seq_len = state->seq_len;

    for (int t = 0; t < seq_len; t++) {
        // Post-attention LayerNorm
        layer_norm(state->norm_buffer, state->hidden_states + t * H,
                   weights->post_attention_layernorm_weight + layer_idx * H, nullptr, H, cfg->rms_norm_eps);

        // Simplified MLP: SwiGLU approximation: gate_proj * GELU(up_proj)
        for (int h = 0; h < H; h++) state->mlp_intermediate[t * I + h] = tanhf(state->norm_buffer[h]); // placeholder

        // Residual add
        for (int h = 0; h < H; h++) state->hidden_states[t * H + h] += state->mlp_intermediate[t * H + h];
    }
}

// -------------------------
// Forward Transformer
// -------------------------
void forward_transformer(QwenRunState* state, const QwenWeight* weights) {
    const QwenConfig* cfg = state->config;
    int H = cfg->hidden_size;
    int V = cfg->vocab_size;
    int L = cfg->num_hidden_layers;

    // Merge vision embedding if present
    merge_vision_text(state);

    // Transformer layers
    for (int layer = 0; layer < L; layer++) {
        forward_attention_layer(state, weights, layer);
        forward_mlp_layer(state, weights, layer);
    }

    // Final LayerNorm
    int last_token_idx = (state->seq_len - 1) * H;
    layer_norm(state->norm_buffer, state->hidden_states + last_token_idx,
               weights->norm_weight, nullptr, H, cfg->rms_norm_eps);

    // LM head logits
    for (int v = 0; v < V; v++) {
        float sum = 0.f;
        for (int h = 0; h < H; h++) {
            sum += state->norm_buffer[h] * weights->lm_head_weight[v * H + h];
        }
        state->logits[v] = sum;
    }
}

// ================================================================
// Patch Extraction (completed)
// ================================================================
void extract_image_patches(const float* img, float* patches, const QwenConfig* config) {
    if (img == nullptr) {
        return;
    }

    int H = config->vision_patch_size * 16;
    int W = config->vision_patch_size * 16;
    int P = config->vision_patch_size;
    int C = 3;
    int idx = 0;
    
    int patches_h = H / P;
    int patches_w = W / P;
    
    for (int py = 0; py < patches_h; py++) {
        for (int px = 0; px < patches_w; px++) {
            for (int c = 0; c < C; ++c) {
                for (int dy = 0; dy < P; ++dy) {
                    for (int dx = 0; dx < P; ++dx) {
                        int img_y = py * P + dy;
                        int img_x = px * P + dx;
                        patches[idx++] = img[(img_y * W + img_x) * C + c];
                    }
                }
            }
        }
    }
}
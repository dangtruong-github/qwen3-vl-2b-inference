#include "../include/forward.hpp"

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

float *forward_llm(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id) {
    // Embed layer
    embedding_lookup(
        weight->embed_tokens_weight, token_id, state->hidden_states,
        config->vocab_size, config->hidden_size
    );

    for (int i = 0; i < config->num_hidden_layers; i++) {

    }

    // Norm attention
    rms_norm(
        state->hidden_states, weight->norm_weight, state->hidden_states, config->rms_norm_eps, config->hidden_size
    );
    classifier_gemm(weight->embed_tokens_weight, state->hidden_states, state->logits, config->vocab_size, config->hidden_size);

    // Final layer

    return state->logits;
}

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
void forward_image_encoder(QwenRunState* state, const QwenWeight* weights, const float* image, int img_h, int img_w, int grid_h, int grid_w) {
    if (!image) {
        state->vision_embed_true = false;
        return; // no image
    }

    return;
}

// -------------------------
// Merge Vision + Text
// -------------------------
void merge_vision_text(QwenRunState* state) {
    if (!state->vision_embed_true) return;

    return;
}

float *forward_llm(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token_id, int pos) {
    // Embed layer
    embedding_lookup(
        weight->token_embedding_table, token_id, state->x,
        config->vocab_size, config->hidden_size
    );

    // **FIXED:** Define dimensions clearly
    int head_dim = config->hidden_size / config->num_attention_heads;
    int seq_len = 1024;
    int kv_dim = config->num_key_value_heads * head_dim;

    // Cache dimensions: [num_layers][seq_len][kv_dim]
    long long cache_layer_size = 1ll * seq_len * kv_dim; // Size of one layer's cache
    long long loff_one = cache_layer_size;               // Offset between layers
    long long loff_pos = kv_dim;                         // Offset between positions

    #ifdef DEBUG
        printf("Dimensions: head_dim=%d, kv_dim=%d, seq_len=%d\n", head_dim, kv_dim, seq_len);
        printf("Cache: loff_one=%lld, loff_pos=%lld\n", loff_one, loff_pos);
        printf("finish embedding_lookup\n");
        fflush(stdout);
        printf("state->x: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", state->x[i]);
        }
        printf("\n");
    #endif

    // print out the cos_table and sin_table for debugging
    #ifdef DEBUG
        printf("cos_table (first 5 values): ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", state->cos_tensor[i]);
        }
        printf("\n");

        printf("sin_table (first 5 values): ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", state->sin_tensor[i]);
        }
        printf("\n");
    #endif

    for (int l = 0; l < config->num_hidden_layers; l++) {
        // Pre-attention RMSNorm
        rms_norm(
            state->x, weight->rms_ffn_w, state->t, config->rms_norm_eps, config->hidden_size, 1ll * l
        );

        #ifdef DEBUG
            printf("finish rms_norm layer %d\n", l);
            fflush(stdout);
            printf("state->t (CHECKED): ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->t[i]);
            }
            printf("\n");
        #endif

        // QKV Projections
        long long loff = 1ll * l * loff_one;  // kv cache layer offset (layer_cache_start)

        const float *w_q = weight->w_attn_q + 1ll * l * config->hidden_size * 2048ll;
        const float *w_k = weight->w_attn_k + 1ll * l * config->hidden_size * 1024ll;
        const float *w_v = weight->w_attn_v + 1ll * l * config->hidden_size * 1024ll;
        
        linear(
            state->t, w_q, nullptr, state->q, 1, 2048, config->hidden_size, true
        );
        linear(
            state->t, w_k, nullptr, state->k, 1, 1024, config->hidden_size, true
        );
        linear(
            state->t, w_v, nullptr, state->v, 1, 1024, config->hidden_size, true
        );

        #ifdef DEBUG
            printf("finish linear qkv layer %d\n", l);
            fflush(stdout);
            printf("state->q (CHECKED): ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->q[i]);
            }
            printf("\n");
            printf("state->k (CHECKED): ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->k[i]);
            }
            printf("\n");
            printf("state->v (CHECKED): ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->v[i]);
            }
            printf("\n");
        #endif

        // QK RMSNorm
        for (int h = 0; h < config->num_attention_heads; h++) {
            rms_norm(
                state->q + h * head_dim,
                weight->w_attn_q_norm,
                state->q + h * head_dim,
                config->rms_norm_eps, head_dim, 1ll * l
            );
            if (h < config->num_key_value_heads) { 
                rms_norm(
                    state->k + h * head_dim,
                    weight->w_attn_k_norm,
                    state->k + h * head_dim,
                    config->rms_norm_eps, head_dim, 1ll * l
                );
            }
        }

        #ifdef DEBUG
            printf("finish norm qk layer %d\n", l);
            fflush(stdout);
            printf("state->q: ");
            for (int i = 0; i < 2048; i++) {
                printf("%.6f ", state->q[i]);
            }
            printf("\n");
            printf("state->k: ");
            for (int i = 0; i < 1024; i++) {
                printf("%.6f ", state->k[i]);
            }
            printf("\n");
        #endif

        // Apply Rotary Position Embeddings
        apply_rotary(
            state->q, state->cos_tensor, state->sin_tensor,
            config->num_attention_heads, head_dim, pos
        );
        apply_rotary(
            state->k, state->cos_tensor, state->sin_tensor,
            config->num_key_value_heads, head_dim, pos
        );

        #ifdef DEBUG
            printf("finish apply rotary layer %d\n", l);
            fflush(stdout);
            printf("state->q: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->q[i]);
            }
            printf("\n");
            printf("state->k: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->k[i]);
            }
            printf("\n");

            if (pos == 1) exit(1);
        #endif

        // Store k, v in cache - FIXED VERSION
        long long loff_cache = loff + pos * kv_dim;  // Position offset within layer
        memcpy(state->key_cache + loff_cache, state->k, kv_dim * sizeof(float));
        memcpy(state->value_cache + loff_cache, state->v, kv_dim * sizeof(float));

        #ifdef DEBUG
            printf("KV cache stored at layer %d, pos %d, offset %lld\n", l, pos, loff_cache);
            printf("Stored k (first 3): %.6f %.6f %.6f\n", 
                   state->key_cache[loff_cache],
                   state->key_cache[loff_cache + 1],
                   state->key_cache[loff_cache + 2]);
            printf("Original k (first 3): %.6f %.6f %.6f\n",
                   state->k[0], state->k[1], state->k[2]);
        #endif

        // Multi-head attention
        int kv_mul = config->num_attention_heads / config->num_key_value_heads;  // integer multiplier for GQA

        // Compute attention scores
        attn_scores_all_heads(
            state->key_cache, state->q, state->att,
            1ll * l, // layer_offset
            config->num_attention_heads, kv_mul, head_dim,
            kv_dim, seq_len, pos
        );

        #ifdef DEBUG
            printf("finish attn_scores_all_heads layer %d\n", l);
            fflush(stdout);
            printf("state->att (first head, first 5 scores): ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->att[i]);
            }
            printf("\n");
        #endif

        // Compute weighted sum of values
        attn_weighted_sum_all_heads(
            state->value_cache, state->q, state->att, state->qkv_out, 
            loff, // layer_cache_start
            config->num_attention_heads, kv_mul, head_dim, 
            kv_dim, seq_len, pos
        );

        #ifdef DEBUG
            printf("finish attn_weighted_sum_all_heads layer %d\n", l);
            fflush(stdout);
            printf("state->qkv_out: ");
            for (int i = 0; i < 129; i++) {
                printf("%.6f ", state->qkv_out[i]);
            }
            printf("\n");
        #endif

        // Output projection
        const float *w_out_proj = weight->w_attn_o + 1ll * l * config->hidden_size * config->hidden_size;
        linear(
            state->qkv_out, w_out_proj, nullptr, state->attn_out, 1,
            config->hidden_size, config->hidden_size, true
        );

        #ifdef DEBUG
            printf("finish w_out_proj layer %d\n", l);
            fflush(stdout);
            printf("state->attn_out: ");
            for (int i = 0; i < 129; i++) {
                printf("%.6f ", state->attn_out[i]);
            }
            printf("\n");
            if (pos == 1) exit(1);
        #endif

        // Residual connection 1
        add_vector(
            state->x, state->attn_out, config->hidden_size
        );

        #ifdef DEBUG
            printf("finish attn residual layer %d\n", l);
            fflush(stdout);
            printf("state->x: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->x[i]);
            }
            printf("\n");
        #endif

        // Post-attention RMSNorm
        rms_norm(
            state->x, weight->rms_attn_w, state->t,
            config->rms_norm_eps, config->hidden_size, 1ll * l
        );

        #ifdef DEBUG
            printf("finish attn rms_norm layer %d\n", l);
            fflush(stdout);
            printf("state->t: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->t[i]);
            }
            printf("\n");
        #endif

        // MLP: Gate and Up projections
        const float *w_gate = weight->w_mlp_gate + 1ll * l * config->intermediate_size * config->hidden_size;
        const float *w_up = weight->w_mlp_up + 1ll * l * config->intermediate_size * config->hidden_size;
        linear(
            state->t, w_gate, nullptr, state->gate, 1,
            config->intermediate_size, config->hidden_size, true
        );
        linear(
            state->t, w_up, nullptr, state->up, 1,
            config->intermediate_size, config->hidden_size, true
        );

        #ifdef DEBUG
            printf("finish gate up layer %d\n", l);
            fflush(stdout);
            printf("state->gate: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->gate[i]);
            }
            printf("\n");
            printf("state->up: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->up[i]);
            }
            printf("\n");
        #endif

        // SwiGLU activation
        swiglu(
            state->gate, state->up, state->gate_up, config->intermediate_size
        );

        #ifdef DEBUG
            printf("finish swiglu gate up layer %d\n", l);
            fflush(stdout);
            printf("state->gate_up: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->gate_up[i]);
            }
            printf("\n");
        #endif

        // MLP: Down projection
        const float *w_down = weight->w_mlp_down + 1ll * l * config->intermediate_size * config->hidden_size;
        linear(
            state->gate_up, w_down, nullptr, state->down, 1,
            config->hidden_size, config->intermediate_size, true
        );

        #ifdef DEBUG
            printf("finish down layer %d\n", l);
            fflush(stdout);
            printf("state->down: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->down[i]);
            }
            printf("\n");
        #endif

        // Residual connection 2
        add_vector(
            state->x, state->down, config->hidden_size
        );

        #ifdef DEBUG
            printf("finish residual end layer %d\n", l);
            fflush(stdout);
            printf("state->x: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", state->x[i]);
            }
            printf("\n");
        #endif
    }

    // Final RMSNorm
    rms_norm(
        state->x, weight->rms_out_w, state->x, config->rms_norm_eps,
        config->hidden_size, 0ll
    );

    #ifdef DEBUG
        printf("finish rms_norm final, pos %d\n", pos);
        fflush(stdout);
        printf("state->x: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", state->x[i]);
        }
        printf("\n");
    #endif

    // Classifier (LM Head)
    classifier_gemm(
        weight->token_embedding_table, state->x, state->logits,
        config->vocab_size, config->hidden_size
    );

    #ifdef DEBUG
        printf("finish classifier_gemm\n");
        fflush(stdout);
        printf("state->logits (first 5): ");
        for (int i = 0; i < 5; i++) {
            printf("%.6f ", state->logits[i]);
        }
        printf("\n");

        if (pos > 2) exit(1);
    #endif

    return state->logits;
}

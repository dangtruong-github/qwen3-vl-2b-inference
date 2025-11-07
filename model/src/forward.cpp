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
        weight->embed_tokens_weight, token_id, state->x,
        config->vocab_size, config->hidden_size
    );

    long long loff_pos = 1ll * config->num_key_value_heads * 128;
    long long loff_one = 1ll * 1024 * loff_pos;

    // printf("finish embedding_lookup\n");
    // fflush(stdout);

    for (int l = 0; l < config->num_hidden_layers; l++) {
        rms_norm(
            state->x, weight->input_layernorm_weight, state->t, config->rms_norm_eps, config->hidden_size, 1ll * l
        );

        // printf("finish rms_norm\n");
        // fflush(stdout);

        // key and value point to the kv cache
        long long loff = 1ll * l * loff_one;  // kv cache layer offset

        const float *w_q = weight->self_attn_q_proj_weight + 1ll * l * config->hidden_size * 2048ll;
        const float *w_k = weight->self_attn_k_proj_weight + 1ll * l * config->hidden_size * 1024ll;
        const float *w_v = weight->self_attn_v_proj_weight + 1ll * l * config->hidden_size * 1024ll;
        linear(
            state->t, w_q, nullptr, state->q, 1, 2048, config->hidden_size, true
        );
        linear(
            state->t, w_k, nullptr, state->k, 1, 1024, config->hidden_size, true
        );
        linear(
            state->t, w_v, nullptr, state->v, 1, 1024, config->hidden_size, true
        );

        // printf("finish linear qkv\n");
        // fflush(stdout);

        rms_norm(
            state->q, weight->self_attn_q_norm_weight, state->q,
            config->rms_norm_eps, 2048, 1ll * l
        );
        rms_norm(
            state->k, weight->self_attn_k_norm_weight, state->k,
            config->rms_norm_eps, 1024, 1ll * l
        );

        // printf("finish norm qk\n");
        // fflush(stdout);

        // attention (blank)
        // ApplyRotary(state->q, state->cos_tensor, state->sin_tensor, config->num_attention_heads, 128, pos);
        // ApplyRotary(state->k, state->cos_tensor, state->sin_tensor, config->num_key_value_heads, 128, pos);

        // Store k, v in cache
        long long loff_cache = 1ll * loff + 1ll * pos * loff_pos;
        memcpy(state->key_cache + loff_cache, state->k, loff_pos * sizeof(float));
        memcpy(state->value_cache + loff_cache, state->v, loff_pos * sizeof(float));

        // printf("finish memcpy cache\n");
        // fflush(stdout);

        // multihead attention
        int kv_mul = config->num_attention_heads / config->num_key_value_heads;  // integer multiplier for GQA

        attn_scores_all_heads(
            state->key_cache, state->q, state->att, loff_one, 1ll * l,
            config->num_attention_heads, kv_mul, 128, loff_pos, 1024, pos
        );

        // printf("finish attn_scores_all_heads\n");
        // fflush(stdout);

        attn_weighted_sum_all_heads(
            state->value_cache, state->q, state->att, state->qkv_out, loff, config->num_attention_heads, kv_mul, 128, loff_pos, 1024, pos
        );

        // printf("finish attn_weighted_sum_all_heads\n");
        // fflush(stdout);

        // output projection
        const float *w_out_proj = weight->self_attn_o_proj_weight + 1ll * l * config->hidden_size * config->hidden_size;
        linear(
            state->qkv_out, w_out_proj, nullptr, state->attn_out, 1,
            config->hidden_size, config->hidden_size, true
        );

        // printf("finish w_out_proj\n");
        // fflush(stdout);

        // residual connection
        add_vector(
            state->x, state->attn_out, config->hidden_size
        ); // equals residual add

        // printf("finish attn residual\n");
        //  fflush(stdout);

        // attn_norm
        rms_norm(
            state->x, weight->post_attention_layernorm_weight, state->x,
            config->rms_norm_eps, config->hidden_size, 1ll * l
        );

        // printf("finish attn rms_norm\n");
        // fflush(stdout);

        // gate up
        const float *w_gate = weight->mlp_gate_proj_weight + 1ll * l * config->intermediate_size * config->hidden_size;
        const float *w_up = weight->mlp_gate_proj_weight + 1ll * l * config->intermediate_size * config->hidden_size;
        linear(
            state->x, w_gate, nullptr, state->gate, 1,
            config->hidden_size, config->intermediate_size, true
        );
        linear(
            state->x, w_up, nullptr, state->up, 1,
            config->hidden_size, config->intermediate_size, true
        );

        // printf("finish gate up\n");
        // fflush(stdout);

        // SwiGLU
        swiglu(
            state->gate, state->up, state->gate_up, config->intermediate_size
        );

        // printf("finish swiglu gate up\n");
        // fflush(stdout);

        // down
        const float *w_down = weight->mlp_gate_proj_weight + 1ll * l * config->intermediate_size * config->hidden_size;
        linear(
            state->gate_up, w_down, nullptr, state->down, 1,
            config->intermediate_size, config->hidden_size, true
        );

        // printf("finish down\n");
        // fflush(stdout);

        // residual connection
        add_vector(
            state->x, state->down, config->hidden_size
        );  // equals residual add

        // printf("finish residual end layer %d\n", l);
        // fflush(stdout);
    }

    rms_norm(
        state->x, weight->norm_weight, state->x, config->rms_norm_eps,
        config->hidden_size, 0ll
    );

    // printf("finish rms_norm final\n");
    // fflush(stdout);

    classifier_gemm(
        weight->embed_tokens_weight, state->x, state->logits,
        config->vocab_size, config->hidden_size
    );

    // printf("finish classifier_gemm\n");
    // fflush(stdout);

    return state->logits;
}

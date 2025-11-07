#include "../include/layer.hpp"

void embedding_lookup(
    const float *embedding /*[vocab, hidden]*/,
    int token_id, float *out /*[hidden]*/,
    size_t vocab_size, size_t hidden_size
) {
    memcpy(out, embedding + token_id * hidden_size, hidden_size * sizeof(float));
}

void rms_norm(
    const float *x /*[hidden]*/, const float *scale /*[layers, hidden]*/,
    float *out /*[hidden]*/, float eps, size_t hidden_size, size_t layer_offset
) {
    const float *scale_buf = scale + 1ll * layer_offset * hidden_size;
    // calculate sum of squares
    double ss = 0.0;
    for (size_t j = 0; j < hidden_size; j++) {
        ss += x[j] * x[j];
    }
    // printf("Finish get ss\n");
    // fflush(stdout);
    ss /= hidden_size;
    ss += eps;
    ss = 1.0 / sqrt(ss);
    // normalize and scale
    for (size_t j = 0; j < hidden_size; j++) {
        out[j] = scale_buf[j] * (ss * x[j]);
    }
    // printf("Finish out\n");
    // fflush(stdout);
}

void linear(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    // A is M x K
    // B is K x N (or N x K if transposed)
    // C is M x N
    // mat_bias is a vector of length N, or NULL

    // 1. Initialize C and apply Bias
    // C[i][j] = mat_bias[j] (if mat_bias is not NULL) or 0.0f
    if (mat_bias != nullptr) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                // Initialize C[i][j] with the bias value mat_bias[j]
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    } else {
        // If no bias, initialize C to zero
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }

    // 2. Perform Matrix Multiplication: C = A * B + C_initial (where C_initial is the bias)
    // We can use the i-j-k loop order and perform the accumulation directly into mat_C.

    if (!mat_B_transpose) {
        // B is K x N. B[k][j] = mat_B[k * N + j]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B and C
                // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                float sum = mat_C[i * N + j];
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // C[i][j] += A[i][k] * B[k][j]
                    sum += mat_A[i * K + k] * mat_B[k * N + j];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
    else {
        // B is N x K. B^T[k][j] = B[j][k] = mat_B[j * K + k]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B^T and C
                // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                float sum = mat_C[i * N + j];
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // C[i][j] += A[i][k] * B[j][k]
                    sum += mat_A[i * K + k] * mat_B[j * K + k];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
}

void classifier_gemm(
    const float *embedding /*[vocab, hidden]*/,
    const float *hid_states /*[hidden]*/, float *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
) {
    linear(
        hid_states, embedding, nullptr, logits,
        1, vocab_size, hidden_size, true
    );
}

void qkv_project(
    const float *x /*[hidden]*/,
    const float *W_qkv /*[(n_q+2*n_kv)*hd, hidden]*/,
    const float *b_qkv /*[(n_q+2*n_kv)*hd]*/,
    float *qkv /*[(n_q+2*n_kv)*hd]*/,
    size_t hidden, size_t n_q, size_t n_kv, size_t layer_offset
) {
    const size_t nq_kv_hd = 1ll * n_q + 2ll * n_kv;

    const float *w_qkv_ptr = W_qkv + 1ll * layer_offset * nq_kv_hd * hidden;
    const float *b_qkv_ptr = b_qkv + 1ll * layer_offset * nq_kv_hd;

    linear(x, w_qkv_ptr, b_qkv_ptr, qkv, 1, nq_kv_hd, hidden, true);
}

void add_vector(float *add_to, const float *add_from, size_t size_vec) {
    for (size_t i = 0; i < size_vec; i++) {
        add_to[i] += add_from[i];
    }
}

// Corrected SwiGLU implementation
void swiglu(
    const float *gate /*[d]*/, const float *up /*[d]*/,
    float *out /*[d]*/, size_t size_vec
) {
    // This constant is used in the sigmoid calculation
    const float alpha = 1.702f;

    for (size_t i = 0; i < size_vec; i++) {
        float gate_val = gate[i];
        float up_val = up[i];

        // --- FIX 2: Corrected SiLU Calculation ---
        // silu(x) = x * σ(αx), where σ(x) is the logistic sigmoid.
        // The 'alpha' constant was missing in your original code.
        gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));

        // Element-wise multiply with the modified 'up' tensor
        out[i] = gate_val * (up_val + 1.0f);
    }
}

void softmax(float *x, size_t size) {
    // find max value (for numerical stability)
    double max_val = x[0];
    for (size_t i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        // printf("x[%d] before: %.6f\n", i, x[i]);
        x[i] = expf(x[i] - max_val);
        // printf("x[%d] after: %.6f\n", i, x[i]);
        sum += x[i];
    }
    // printf("sum: %.6f\n", sum);
    // printf("max_val: %.6f\n", max_val);
    // normalize
    for (size_t i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Attention scores for 1 head: att[0..pos] = q·k_t / sqrt(hd) (+ mask)
void attn_scores_all_heads(
    const float *key_cache, const float *q, float *att, size_t loff_one,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
) {
    long long loff = 1ll * layer_offset * loff_one;

    for (int h = 0; h < attn_heads; h++) {
        // get the query vector for this head
        const float *q_head = q + h * head_dim;

        // attention scores for this head
        float *att_head = att + h * seq_len;

        // Get the key cache for this head group (GQA)
        int kv_head_idx = h / kv_mul;

        // Compute attention scores
        for (int t = 0; t <= pos; t++) {
            // Calculate the correct key position
            const float *k = key_cache + loff + t * kv_dim + kv_head_idx * head_dim;
            double score = 0.0;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k[i];
            }
            score /= sqrtf(head_dim);
            att_head[t] = score;
        }

        // softmax the scores to get attention weights
        softmax(att_head, (size_t)pos + 1);
    }
}

void attn_weighted_sum_all_heads(
    const float *value_cache, const float *q, const float *att, float *tb,
    size_t loff, int attn_heads, int kv_mul, int head_dim, int kv_dim,
    int seq_len, int pos
) {
    // For each head, compute attention scores and weighted sum
    for (int h = 0; h < attn_heads; h++) {
        // get the query vector for this head
        const float *q_head = q + h * head_dim;

        // attention scores for this head
        const float *att_head = att + h * seq_len;

        // Get the key cache for this head group (GQA)
        int kv_head_idx = h / kv_mul;

        // Create mask if needed
        // Get the value cache for this head group (GQA)

        // weighted sum of the values
        float *tb_head = tb + h * head_dim;

        memset(tb_head, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *v = (value_cache + loff) + t * kv_dim + kv_head_idx * head_dim;
            float a = att_head[t];
            for (int i = 0; i < head_dim; i++) {
                tb_head[i] += a * v[i];
            }
        }
    }
}
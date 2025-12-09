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

void swiglu(
    const float *gate,  // [d]
    const float *up,    // [d]
    float *out,         // [d]
    size_t size_vec
) {
    for (size_t i = 0; i < size_vec; i++) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));  // SiLU(x) = x * sigmoid(x)
        out[i] = silu * up[i];               // SwiGLU = SiLU(gate) * up
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

// **FIXED** attn_scores_all_heads function:
// Changed signature to remove loff_one and calculate offset internally.
void attn_scores_all_heads(
    const float *key_cache, const float *q, float *att,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
) {
    // Calculate the start of the current layer in the cache
    long long layer_cache_start = 1ll * layer_offset * seq_len * kv_dim;
    const float *key_cache_layer = key_cache + 1ll * layer_cache_start;

    for (int h = 0; h < attn_heads; h++) {
        const float *q_head = q + h * head_dim;
        float *att_head = att + h * seq_len;
        int kv_head_idx = h / kv_mul;

        // Compute attention scores
        for (int t = 0; t <= pos; t++) {
            // Get the key for this timestep
            const float *k_head = key_cache_layer + 1ll * t * kv_dim + 1ll * kv_head_idx * head_dim;

            #ifdef DEBUG
                printf("Inside Layer %zu, Head %d, Time %d\n", layer_offset, h, t);
                printf("k_head: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", k_head[i]);
                }
                printf("\n");
            #endif
            
            double score = 0.0;
            for (int i = 0; i < head_dim; i++) {
                score += (double)q_head[i] * (double)k_head[i];
            }
            score /= sqrtf((float)head_dim);
            att_head[t] = (float)score;
        }

        #ifdef DEBUG
            printf("Attention scores for Layer %zu, Head %d up to pos %d:\n", layer_offset, h, pos);
            for (int t = 0; t <= pos; t++) {
                printf("%.6f ", att_head[t]);
            }
            printf("\n");
        #endif

        // Softmax over the valid scores (0 to pos)
        // This implicitly handles the causal mask.
        softmax(att_head, (size_t)pos + 1);
    }
}

// Fix attn_weighted_sum_all_heads function:
// This function was already correct. 'loff' passed from forward_text
// corresponds to the layer_cache_start.
void attn_weighted_sum_all_heads(
    const float *value_cache, const float *q, const float *att, float *tb,
    size_t loff, int attn_heads, int kv_mul, int head_dim, int kv_dim,
    int seq_len, int pos
) {
    // Initialize output to zero
    memset(tb, 0, attn_heads * head_dim * sizeof(float));

    for (int h = 0; h < attn_heads; h++) {
        const float *att_head = att + h * seq_len;
        int kv_head_idx = h / kv_mul;
        float *tb_head = tb + h * head_dim;

        for (int t = 0; t <= pos; t++) {
            // FIX: Correct value cache indexing
            const float *v = value_cache + loff + t * kv_dim + kv_head_idx * head_dim;

            #ifdef DEBUG
                printf("V cache Inside Layer %zu, Head %d, Time %d\n", loff, h, t);
                printf("v_head: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", v[i]);
                }
                printf("\n");
            #endif

            float a = att_head[t];
            for (int i = 0; i < head_dim; i++) {
                tb_head[i] += a * v[i];
            }
        }

        #ifdef DEBUG
            printf("Weighted sum output for Layer offset %zu, Head %d up to pos %d:\n", loff, h, pos);
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", tb_head[i]);
            }
            printf("\n");
        #endif
    }
}

void apply_rotary(
    float *x /*[n_heads*hd]*/, const float *cos_table /*[seq_len*hd/2]*/,
    const float *sin_table /*[seq_len*hd/2]*/, int n_heads, int head_dim,
    int pos
) {
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            // --- THIS IS THE KEY CHANGE ---
            // Calculate the index to look up the cos/sin values for the given position.
            // The tables are logically 2D [pos, i], so the 1D index is pos * (width) + i.
            int rope_idx = pos * half + i;
            float c = cos_table[rope_idx];
            float s = sin_table[rope_idx];

            // --- The rotation logic remains the same ---
            // Indexing for the input tensor: head h, dim i
            int x_idx1 = h * head_dim + i;         // first half
            int x_idx2 = h * head_dim + half + i;  // second half

            float x1 = x[x_idx1];
            float x2 = x[x_idx2];

            // Apply the 2D rotation
            float o1 = x1 * c - x2 * s;
            float o2 = x2 * c + x1 * s;

            // Write the results back
            x[x_idx1] = o1;
            x[x_idx2] = o2;
        }
    }
}

void conv_3d(
    const float *conv_w, const float *conv_b, float *in_img, float *out_img,
    long img_h, long VC, long VTP, long VP, long VH
) {
    // Total size of the VTP * VP * VP plane
    const long PLANE_SIZE = VTP * VP * VP;
    // Total size of the VC * VTP * VP * VP feature block
    const long FEATURE_BLOCK_SIZE = VC * PLANE_SIZE;

    // --- Outer loop: Iterate over the 'batch' dimension (img_h) ---
    for (long i = 0; i < img_h; ++i) {
        // Pointer to the current feature block in the input
        const float *input_block_ptr = in_img + i * FEATURE_BLOCK_SIZE;

        // --- Second loop: Iterate over the output channels (VH) ---
        for (long h = 0; h < VH; ++h) {
            float accumulator = 0.0f;

            // Pointer to the current kernel slice (VC * VTP * VP * VP) for this output channel
            const float *kernel_slice_ptr = conv_w + h * FEATURE_BLOCK_SIZE;

            // --- Innermost loop: Perform the dot product (contraction) ---
            // This loop iterates over the flattened feature block (VC * VTP * VP * VP)
            for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                accumulator += input_block_ptr[c] * kernel_slice_ptr[c];
            }

            // Add the bias and store the result
            // Output index: i * VH + h
            out_img[i * VH + h] = accumulator + conv_b[h];
        }
    }
}

void vision_pos_embed(const float *pos_embed_w, float *x_embed, int grid_h, int grid_w, int num_grid_per_side, int VSP, int VH) {

    if (grid_h <= 0 || grid_w <= 0 || num_grid_per_side <= 0) {
        return;
    }

    size_t total_elements = (size_t)grid_h * (size_t)grid_w;
    
    // --- Allocate Temporary Buffers (equivalent to Python's intermediate lists/tensors) ---
    // The final result tensors would also be total_elements * 4, but cannot be returned.
    
    // Float arrays
    float *h_idxs = (float *)malloc(grid_h * sizeof(float));
    float *w_idxs = (float *)malloc(grid_w * sizeof(float));
    float *dh = (float *)malloc(grid_h * sizeof(float));
    float *dw = (float *)malloc(grid_w * sizeof(float));
    
    // Long/int arrays
    long *h_idxs_floor = (long *)malloc(grid_h * sizeof(long));
    long *w_idxs_floor = (long *)malloc(grid_w * sizeof(long));
    long *h_idxs_ceil = (long *)malloc(grid_h * sizeof(long));
    long *w_idxs_ceil = (long *)malloc(grid_w * sizeof(long));
    long *base_h = (long *)malloc(grid_h * sizeof(long));
    long *base_h_ceil = (long *)malloc(grid_h * sizeof(long));

    // Array of pointers for the 4 final index and weight results (allocated here but effectively lost)
    // This is the memory for the FINAL idx_tensor and weight_tensor
    long *idx_list_mem[4];
    float *weight_list_mem[4];
    
    for (int i = 0; i < 4; ++i) {
        idx_list_mem[i] = (long *)malloc(total_elements * sizeof(long));
        weight_list_mem[i] = (float *)malloc(total_elements * sizeof(float));
        // Check for malloc failure in a real application here...
    }

    // --- Check for malloc failure on ALL buffers (omitted for brevity) ---
    // If any malloc failed, free everything successfully allocated and return.
    
    // --- 1. Calculate the normalized floating-point indices (h_idxs, w_idxs) ---
    float max_idx = (float)(num_grid_per_side - 1);
    
    if (grid_h > 1) {
        float h_step = max_idx / (float)(grid_h - 1);
        for (int i = 0; i < grid_h; ++i) h_idxs[i] = (float)i * h_step;
    } else if (grid_h == 1) {
        h_idxs[0] = 0.0f;
    }
    
    if (grid_w > 1) {
        float w_step = max_idx / (float)(grid_w - 1);
        for (int i = 0; i < grid_w; ++i) w_idxs[i] = (float)i * w_step;
    } else if (grid_w == 1) {
        w_idxs[0] = 0.0f;
    }

    // --- 2. Calculate floor, ceil, and delta (dh, dw) ---
    long max_clip = (long)num_grid_per_side - 1;

    for (int i = 0; i < grid_h; ++i) {
        h_idxs_floor[i] = (long)floorf(h_idxs[i]);
        long h_ceil_temp = h_idxs_floor[i] + 1;
        h_idxs_ceil[i] = (h_ceil_temp < max_clip) ? h_ceil_temp : max_clip;
        dh[i] = h_idxs[i] - (float)h_idxs_floor[i];
        base_h[i] = h_idxs_floor[i] * num_grid_per_side;
        base_h_ceil[i] = h_idxs_ceil[i] * num_grid_per_side;
    }

    for (int i = 0; i < grid_w; ++i) {
        w_idxs_floor[i] = (long)floorf(w_idxs[i]);
        long w_ceil_temp = w_idxs_floor[i] + 1;
        w_idxs_ceil[i] = (w_ceil_temp < max_clip) ? w_ceil_temp : max_clip;
        dw[i] = w_idxs[i] - (float)w_idxs_floor[i];
    }
    
    // --- 3. Flatten and calculate indices/weights into the *allocated* result arrays ---
    size_t linear_idx = 0;
    
    for (int h_idx = 0; h_idx < grid_h; ++h_idx) {
        float one_minus_dh = 1.0f - dh[h_idx];
        float current_dh = dh[h_idx];
        
        long current_base_h = base_h[h_idx];
        long current_base_h_ceil = base_h_ceil[h_idx];

        for (int w_idx = 0; w_idx < grid_w; ++w_idx) {
            float one_minus_dw = 1.0f - dw[w_idx];
            float current_dw = dw[w_idx];
            long current_w_floor = w_idxs_floor[w_idx];
            long current_w_ceil = w_idxs_ceil[w_idx];
            
            // Indices (written to the malloc'd result arrays)
            idx_list_mem[0][linear_idx] = current_base_h + current_w_floor;
            idx_list_mem[1][linear_idx] = current_base_h + current_w_ceil;
            idx_list_mem[2][linear_idx] = current_base_h_ceil + current_w_floor;
            idx_list_mem[3][linear_idx] = current_base_h_ceil + current_w_ceil;

            // Weights (written to the malloc'd result arrays)
            weight_list_mem[0][linear_idx] = one_minus_dh * one_minus_dw; 
            weight_list_mem[1][linear_idx] = one_minus_dh * current_dw;   
            weight_list_mem[2][linear_idx] = current_dh * one_minus_dw;   
            weight_list_mem[3][linear_idx] = current_dh * current_dw;     

            linear_idx++;
        }
    }

    // --- 4. Clean up dynamically allocated TEMPORARY memory ---
    free(h_idxs); free(w_idxs);
    free(h_idxs_floor); free(w_idxs_floor);
    free(h_idxs_ceil); free(w_idxs_ceil);
    free(dh); free(dw);
    free(base_h); free(base_h_ceil);

    float *pos_embed_ptr = (float *)malloc(1ll * total_elements * VH * sizeof(float));

    for (size_t i = 0; i < total_elements; ++i) {
        float *pos_embed_ptr_cur = pos_embed_ptr + (i * VH);

        for (size_t j = 0; j < VH; j++) {
            pos_embed_ptr_cur[j] = 0.0f;
        }

        for (int k = 0; k < 4; ++k) {

            long source_index = idx_list_mem[k][i];
            const float *pos_embed_w_idx_start = pos_embed_w + (source_index * VH);

            float weight = weight_list_mem[k][i];

            for (int j = 0; j < VH; ++j) {
                pos_embed_ptr_cur[j] += pos_embed_w_idx_start[j] * weight;
            }
        }   
    }

    for (int i = 0; i < 4; ++i) {
        free(idx_list_mem[i]);
        free(weight_list_mem[i]);
    }

    int grid_t = 1;

    // Tensors dimensions after view and permute (6D structure):
    // (T, H_grid, W_grid, H_intra, W_intra, D)
    int H_grid = grid_h / VSP;
    int W_grid = grid_w / VSP;
    int H_intra = VSP;
    int W_intra = VSP;

    // The total size of the final tensor
    size_t total_elements_with_t = (size_t)grid_t * grid_h * grid_w * VH;
    size_t token_size = (size_t)VH;
    size_t patch_size_d = (size_t)H_intra * W_intra * VH; // VSP * VSP * D
    size_t hw_size_d = (size_t)grid_h * grid_w * VH;                // H * W * D (size of the input)

    // --- C-Style Implementation of Repeat, View, Permute, and Flatten ---
    
    // pos_embed.repeat(t, 1) is achieved implicitly in the outer loop:
    for (int it = 0; it < grid_t; ++it) {
        // Loop over the new 6D shape dimensions: T, H_grid, W_grid, H_intra, W_intra
        for (int ih_grid = 0; ih_grid < H_grid; ++ih_grid) {
            for (int iw_grid = 0; iw_grid < W_grid; ++iw_grid) {
                for (int ih_intra = 0; ih_intra < H_intra; ++ih_intra) {
                    for (int iw_intra = 0; iw_intra < W_intra; ++iw_intra) {
                        
                        // --- 1. Calculate source index (Original H*W layout) ---
                        // The source index is based on the original (H, W) coordinates.
                        // H_original = H_grid * VSP + H_intra
                        // W_original = W_grid * VSP + W_intra
                        
                        int h_original = ih_grid * VSP + ih_intra;
                        int w_original = iw_grid * VSP + iw_intra;
                        
                        // Source index offset for the start of the (H, W) token in pos_embed_in
                        size_t source_token_offset = (size_t)(h_original * grid_w + w_original) * token_size;
                        const float *source_ptr = pos_embed_ptr + source_token_offset;

                        // --- 2. Calculate destination index (Flattened 5D layout) ---
                        // The Python code results in a flattened tensor:
                        // Index = (T * H_grid * W_grid * H_intra * W_intra) * D
                        
                        // This corresponds to the permutation: (T, H_grid, W_grid, H_intra, W_intra, D)
                        size_t dest_flat_idx = 0;
                        
                        // Contribution of T
                        dest_flat_idx += (size_t)it * (H_grid * W_grid * patch_size_d);
                        
                        // Contribution of H_grid
                        dest_flat_idx += (size_t)ih_grid * (W_grid * patch_size_d);
                        
                        // Contribution of W_grid
                        dest_flat_idx += (size_t)iw_grid * (patch_size_d);
                        
                        // Contribution of H_intra
                        dest_flat_idx += (size_t)ih_intra * (W_intra * VH);
                        
                        // Contribution of W_intra
                        dest_flat_idx += (size_t)iw_intra * (VH);

                        float *dest_ptr = x_embed + dest_flat_idx;
                        
                        // --- 3. Copy the token (D elements) ---
                        // Copy the D-dimensional token vector from source to destination
                        memcpy(dest_ptr, source_ptr, token_size * sizeof(float));
                    }
                }
            }
        }
    }

    free(pos_embed_ptr);
}
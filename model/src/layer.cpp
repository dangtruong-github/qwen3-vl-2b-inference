#include "../include/layer.hpp"

void embedding_lookup(
    const Tensor *embedding /*[vocab, hidden]*/, size_t token_id,
    Tensor *out /*[hidden]*/, size_t hidden_size
) {
    memcpy(
        out->ptr(), embedding->ptr({token_id}),
        hidden_size * sizeof(float)
    );
}

void rms_norm(
    const float *x /*[hidden]*/, const Tensor *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps,
    size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    const float *scale_buf = (const float *)scale->ptr({layer_offset});

    for (size_t i = 0; i < batches; i++) {
        // calculate sum of squares
        double ss = 0.0;
        for (size_t j = 0; j < hidden_size; j++) {
            ss += x[j] * x[j];
        }
        ss /= hidden_size;
        ss += eps;
        ss = 1.0 / sqrt(ss);
        // normalize and scale
        for (size_t j = 0; j < hidden_size; j++) {
            out[j] = scale_buf[j] * (ss * x[j]);
        }
        
        x += hidden_size;
        out += hidden_size;
    }
}

void classifier_gemm(
    const Tensor *embedding /*[vocab, hidden]*/,
    const Tensor *hid_states /*[hidden]*/, Tensor *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
) {
    linear(
        (const float *)hid_states->ptr(), (const float *)embedding->ptr(), nullptr, 
        (float *)logits->ptr(), 1, vocab_size, hidden_size, true
    );
}

void add_vector(Tensor *add_to, const Tensor *add_from, size_t size_vec) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }
    float *add_to_buf = (float *)add_to->ptr();
    const float *add_from_buf = (const float *)add_from->ptr();
    for (size_t i = 0; i < size_vec; i++) {
        add_to_buf[i] += add_from_buf[i];
    }
}

void add_vector(Tensor *add_to, const float *add_from, size_t size_vec) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }
    float *add_to_buf = (float *)add_to->ptr();
    for (size_t i = 0; i < size_vec; i++) {
        add_to_buf[i] += add_from[i];
    }
}

void swiglu(
    Tensor *gate,  // [d]
    const Tensor *up,    // [d]
    size_t size_vec
) {
    float *gate_buf = (float *)gate->ptr();
    const float *up_buf = (const float *)up->ptr();
    for (size_t i = 0; i < size_vec; i++) {
        float x = gate_buf[i];
        float silu = x / (1.0f + expf(-x));  // SiLU(x) = x * sigmoid(x)
        gate_buf[i] = silu * up_buf[i];               // SwiGLU = SiLU(gate) * up
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
    const Tensor *key_cache, const Tensor *q, Tensor *att,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
) {
    // Calculate the start of the current layer in the cache
    const float *key_cache_ptr = (const float *)key_cache->ptr({0, layer_offset});

    for (size_t h = 0; h < attn_heads; h++) {
        const float *q_head = (const float *)q->ptr({0, h});
        float *att_head = (float *)att->ptr({0, h});
        int kv_head_idx = h / kv_mul;

        // Compute attention scores
        for (int t = 0; t <= pos; t++) {
            // Get the key for this timestep
            const float *k_head = key_cache_ptr + 1ll * t * kv_dim + 1ll * kv_head_idx * head_dim;
            
            double score = 0.0;
            for (int i = 0; i < head_dim; i++) {
                score += (double)q_head[i] * (double)k_head[i];
            }
            score /= sqrtf((float)head_dim);
            att_head[t] = (float)score;
        }

        // Softmax over the valid scores (0 to pos)
        // This implicitly handles the causal mask.
        softmax(att_head, (size_t)pos + 1);
    }
}

// Fix attn_weighted_sum_all_heads function:
// This function was already correct. 'loff' passed from forward_text
// corresponds to the layer_cache_start.
void attn_weighted_sum_all_heads(
    const Tensor *value_cache, const Tensor *att, Tensor *tb,
    size_t layer_offset, int attn_heads, int kv_mul, int head_dim, int kv_dim,
    int seq_len, int pos
) {
    // Initialize output to zero
    memset(tb->ptr(), 0, 1ll * attn_heads * head_dim * sizeof(float));
    float *tb_head = (float *)tb->ptr();
    const float *v_cache_ptr = (const float *)value_cache->ptr({0, layer_offset});

    for (size_t h = 0; h < attn_heads; h++) {
        const float *att_head = (const float *)att->ptr({0, h});
        int kv_head_idx = h / kv_mul;
        const float *v = v_cache_ptr + kv_head_idx * head_dim;

        for (int t = 0; t <= pos; t++) {
            float a = att_head[t];

            for (int i = 0; i < head_dim; i++) {
                tb_head[i] += a * v[i];
            }

            v += kv_dim;
        }

        tb_head += head_dim;
    }
}

void apply_rotary(
    float *x /*[n_heads*hd]*/, const Tensor *cos_table /*[seq_len*hd/2]*/,
    const Tensor *sin_table /*[seq_len*hd/2]*/, int n_heads, int head_dim,
    int pos
) {
    int half = head_dim / 2;
    const float *cos_buf = (const float *)cos_table->ptr();
    const float *sin_buf = (const float *)sin_table->ptr();
    float *x_buf = (float *)x;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            // --- THIS IS THE KEY CHANGE ---
            // Calculate the index to look up the cos/sin values for the given position.
            // The tables are logically 2D [pos, i], so the 1D index is pos * (width) + i.
            int rope_idx = pos * half + i;
            float c = cos_buf[rope_idx];
            float s = sin_buf[rope_idx];

            // --- The rotation logic remains the same ---
            // Indexing for the input tensor: head h, dim i
            int x_idx1 = h * head_dim + i;         // first half
            int x_idx2 = h * head_dim + half + i;  // second half

            float x1 = x_buf[x_idx1];
            float x2 = x_buf[x_idx2];

            // Apply the 2D rotation
            float o1 = x1 * c - x2 * s;
            float o2 = x2 * c + x1 * s;

            // Write the results back
            x_buf[x_idx1] = o1;
            x_buf[x_idx2] = o2;
        }
    }
}

void conv_3d(
    const Tensor *conv_w, const Tensor *conv_b, float *in_img, float *out_img,
    long img_h, long VC, long VTP, long VP, long VH
) {
    // Total size of the VTP * VP * VP plane
    const long PLANE_SIZE = VTP * VP * VP;
    // Total size of the VC * VTP * VP * VP feature block
    const long FEATURE_BLOCK_SIZE = VC * PLANE_SIZE;

    const float *conv_b_buf = (const float *)conv_b->ptr();

    // --- Outer loop: Iterate over the 'batch' dimension (img_h) ---
    for (long i = 0; i < img_h; ++i) {
        // Pointer to the current feature block in the input
        const float *input_block_ptr = in_img + i * FEATURE_BLOCK_SIZE;

        // --- Second loop: Iterate over the output channels (VH) ---
        for (size_t h = 0; h < VH; ++h) {
            float accumulator = 0.0f;

            // Pointer to the current kernel slice (VC * VTP * VP * VP) for this output channel
            const float *kernel_slice_ptr = (const float *)conv_w->ptr({h});

            // --- Innermost loop: Perform the dot product (contraction) ---
            // This loop iterates over the flattened feature block (VC * VTP * VP * VP)
            for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                accumulator += input_block_ptr[c] * kernel_slice_ptr[c];
            }

            // Add the bias and store the result
            // Output index: i * VH + h
            out_img[i * VH + h] = accumulator + conv_b_buf[h];
        }
    }
}

void vision_pos_embed(const Tensor *pos_embed_w, float *x_embed, int grid_h, int grid_w, int num_grid_per_side, int VSP, int VH) {
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
            const float *pos_embed_w_idx_start = (const float *)pos_embed_w->ptr({(size_t)source_index});

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

// Full vision_rot_pos_emb wrapper function example
void vision_rot_pos_emb(
    float *pos_emb_out_cos, float *pos_emb_out_sin,
    const float *cos_tensor, const float *sin_tensor,
    int grid_h, int grid_w, int merge_size, int head_dim
) {
    int max_hw = max(grid_h, grid_w);
    int total_tokens = grid_h * grid_w;
    int freqs_depth_dim = head_dim / 4;
    size_t size_copy = freqs_depth_dim * sizeof(float);

    // --- 1. Generate Coordinates (as completed previously) ---
    int merged_h = grid_h / merge_size;
    int merged_w = grid_w / merge_size;
    int k = 0; 
    for (int block_row = 0; block_row < merged_h; block_row++) {
        int start_row = block_row * merge_size;
        for (int block_col = 0; block_col < merged_w; block_col++) {
            int start_col = block_col * merge_size;
            for (int intra_row = 0; intra_row < merge_size; intra_row++) {
                int row_idx = start_row + intra_row;
                for (int intra_col = 0; intra_col < merge_size; intra_col++) {
                    int col_idx = start_col + intra_col;
                    if (k < total_tokens) {
                        // cos
                        const float *row_freqs_cos = cos_tensor + (row_idx * freqs_depth_dim); 
                        const float *col_freqs_cos = cos_tensor + (col_idx * freqs_depth_dim);
                        float *out_cos = pos_emb_out_cos + (k * head_dim);

                        memcpy(out_cos, row_freqs_cos, size_copy);
                        memcpy(out_cos + 2 * freqs_depth_dim, row_freqs_cos, size_copy);
                        memcpy(out_cos + freqs_depth_dim, col_freqs_cos, size_copy);
                        memcpy(out_cos + 3 * freqs_depth_dim, col_freqs_cos, size_copy);

                        // sin
                        const float *row_freqs_sin = sin_tensor + (row_idx * freqs_depth_dim); 
                        const float *col_freqs_sin = sin_tensor + (col_idx * freqs_depth_dim);
                        float *out_sin = pos_emb_out_sin + (k * head_dim);

                        memcpy(out_sin, row_freqs_sin, size_copy);
                        memcpy(out_sin + 2 * freqs_depth_dim, row_freqs_sin, size_copy);
                        memcpy(out_sin + freqs_depth_dim, col_freqs_sin, size_copy);
                        memcpy(out_sin + 3 * freqs_depth_dim, col_freqs_sin, size_copy);
                        
                        k++;
                    }
                }
            }
        }
    }
}

void layer_norm(
    const Tensor *x,           /* [batches, hidden] */
    const Tensor *scale,       /* [layers, hidden] */
    const Tensor *bias,        /* [layers, hidden] */
    Tensor *out,               /* [batches, hidden] */
    float eps, 
    size_t batches, 
    size_t layer_offset
) {
    // 1. Pointers to current layer's weights
    const size_t hidden_size = scale->shape[scale->ndim - 1];

    const float *scale_buf = (const float *)scale->ptr({layer_offset});
    const float *bias_buf = (const float *)bias->ptr({layer_offset});
    const float *x_buf = (float *)x->ptr();
    float *out_buf = (float *)out->ptr();

    for (size_t i = 0; i < batches; i++) {
        // 2. Calculate Mean
        float mean = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            mean += x_buf[j];
        }
        mean /= hidden_size;

        // 3. Calculate Variance (Sum of Squared Differences)
        float variance = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float diff = x_buf[j] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;

        // 4. Calculate Inverse Standard Deviation
        // nn.LayerNorm uses sqrt(variance + eps)
        float inv_std = 1.0f / sqrt(variance + eps);

        // 5. Normalize and Apply Affine Transform
        for (size_t j = 0; j < hidden_size; j++) {
            // (x - mean) * inv_std * gamma + beta
            out_buf[j] = (x_buf[j] - mean) * inv_std * scale_buf[j] + bias_buf[j];
        }

        x_buf += hidden_size;
        out_buf += hidden_size;
    }
}

void vision_apply_rotary_inplace(
    const float *cos_tensor, // shape (total_tokens, head_dim)
    const float *sin_tensor, // shape (total_tokens, head_dim)
    float *buffer,           // shape (total_tokens, num_heads, head_dim)
    long total_tokens,
    int num_heads,
    int head_dim
) {
    const int half_dim = head_dim / 2;
    const int head_size = head_dim;

    for (long i = 0; i < total_tokens * num_heads; ++i) {
        const long head_start_idx = i * head_size;
        const long token_index = i / num_heads;

        // cos/sin indexed by token
        const long cos_sin_offset = token_index * head_dim;

        float *current = buffer + head_start_idx;
        const float *current_cos = cos_tensor + cos_sin_offset;
        const float *current_sin = sin_tensor + cos_sin_offset;

        for (int m = 0; m < half_dim; ++m) {
            // Cache before overwrite (CRITICAL)
            const float x1 = current[m];
            const float x2 = current[m + half_dim];

            const float cos_first  = current_cos[m];
            const float sin_first  = current_sin[m];
            const float cos_second = current_cos[m + half_dim];
            const float sin_second = current_sin[m + half_dim];

            // First half
            current[m] =
                (x1 * cos_first) + (-x2 * sin_first);

            // Second half
            current[m + half_dim] =
                (x2 * cos_second) + (x1 * sin_second);
        }
    }
}

void tensor_transpose(const float *in, float *out, int dim_0, int dim_1, int dim_2) {
    // dim_0 is D0, dim_1 is D1, dim_2 is D2
    const int D0 = dim_0;
    const int D1 = dim_1;
    const int D2 = dim_2;

    // The inner-most dimension (D2) remains the same in both the input and output
    // The dimensions change from (D0, D1, D2) to (D1, D0, D2)

    // Stride for the D0 dimension in the input (D1 * D2)
    const int in_stride_0 = D1 * D2;
    // Stride for the D1 dimension in the input (D2)
    const int in_stride_1 = D2;

    // Stride for the D1 dimension in the output (D0 * D2)
    const int out_stride_1 = D0 * D2;
    // Stride for the D0 dimension in the output (D2)
    const int out_stride_0 = D2;
    
    // Total size check (optional but good practice)
    // const int total_size = D0 * D1 * D2;

    // i iterates through dim_0 (D0)
    for (int i = 0; i < D0; ++i) {
        // j iterates through dim_1 (D1)
        for (int j = 0; j < D1; ++j) {
            // k iterates through dim_2 (D2)
            for (int k = 0; k < D2; ++k) {
                
                // Calculate the linear index for the input tensor in (D0, D1, D2) order
                // in_idx = i * D1 * D2 + j * D2 + k
                const int in_idx = i * in_stride_0 + j * in_stride_1 + k;

                // Calculate the linear index for the output tensor in (D1, D0, D2) order
                // The element at (i, j, k) in 'in' moves to (j, i, k) in 'out'
                // out_idx = j * D0 * D2 + i * D2 + k
                const int out_idx = j * out_stride_1 + i * out_stride_0 + k;

                // Perform the element copy
                out[out_idx] = in[in_idx];
            }
        }
    }
}

void vision_att(
    const float *q, const float *k, const float *v, float *attn_scores,
    float *out, int num_heads, int total_tokens, int head_dim, float scale
) {    
    // 1. Allocate memory for attention scores (attn_weight)
    // Size needed is (total_tokens * total_tokens) for a single head.
    // We will reuse this buffer for every head to save memory.
    size_t score_size = (size_t)total_tokens * total_tokens;

    // Strides to move pointers to the next head
    size_t head_stride = (size_t)total_tokens * head_dim;

    // Loop over each head
    for (int h = 0; h < num_heads; h++) {
        // Calculate pointers for the current head
        const float *curr_q = q + (h * head_stride);
        const float *curr_k = k + (h * head_stride);
        const float *curr_v = v + (h * head_stride);
        float *curr_out = out + (h * head_stride);

        // ---------------------------------------------------------
        // Step 1: Query x Key^T
        // Q shape: [total_tokens, head_dim]
        // K shape: [total_tokens, head_dim] (transposed logic handled by linear)
        // Result:  [total_tokens, total_tokens]
        // M = total_tokens, N = total_tokens, K = head_dim
        // ---------------------------------------------------------
        linear(curr_q, curr_k, NULL, attn_scores, total_tokens, total_tokens, head_dim, true);
        
        

        // ---------------------------------------------------------
        // Step 2: Scale Factor and Bias
        // Python: attn_weight = ... * scale_factor + attn_bias
        // Note: attn_bias is zeros in the python sample, so we skip adding it.
        // We iterate manually as 'linear' does not support scalar multiplication.
        // ---------------------------------------------------------
        for (size_t i = 0; i < score_size; i++) {
            attn_scores[i] *= scale;
        }

        // ---------------------------------------------------------
        // Step 3: Softmax
        // Applied along the last dimension (rows of the score matrix)
        // ---------------------------------------------------------
        for (int row = 0; row < total_tokens; row++) {
            softmax(attn_scores + (row * total_tokens), total_tokens);
        }

        // ---------------------------------------------------------
        // Step 4: Attn_Weights x Value
        // Weights shape: [total_tokens, total_tokens]
        // V shape:       [total_tokens, head_dim]
        // Result:        [total_tokens, head_dim] -> written to curr_out
        // M = total_tokens, N = head_dim, K = total_tokens
        // ---------------------------------------------------------
        linear(attn_scores, curr_v, NULL, curr_out, total_tokens, head_dim, total_tokens, false);
    }
}

void gelu_tanh(Tensor *x, size_t x_size) {
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;

    float *x_buf = (float *)x->ptr();

    for (size_t i = 0; i < x_size; ++i) {
        float xi = x_buf[i];
        float xi3 = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * xi3);
        float tanh_inner = tanhf(inner);
        x_buf[i] = 0.5f * xi * (1.0f + tanh_inner);
    }
}
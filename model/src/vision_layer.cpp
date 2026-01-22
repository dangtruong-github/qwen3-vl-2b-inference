#include "../include/vision_layer.hpp"

void conv_3d(
    const Tensor *__restrict conv_w, const Tensor *__restrict conv_b,
    const float *__restrict in_img, Tensor *__restrict out_img_tensor,
    long img_h, long VC, long VTP, long VP, long VH
) {
    // Total size of the VTP * VP * VP plane
    const long PLANE_SIZE = VTP * VP * VP;
    // Total size of the VC * VTP * VP * VP feature block
    const long FEATURE_BLOCK_SIZE = VC * PLANE_SIZE;

    PtrPair conv_w_ptr = conv_w->ptr_all();
    PtrPair conv_b_ptr = conv_b->ptr_all();

    float *__restrict out_img = (float *)out_img_tensor->ptr();

    if (conv_w->dtype == DType::FP32) {
        const float *__restrict conv_b_buf = (const float *)(conv_b_ptr.buf);

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
    } else if (conv_w->dtype == DType::FP16) {
        // Cast bias to fp16 type
        const half_cpu *__restrict conv_b_buf = static_cast<const half_cpu*>(conv_b_ptr.buf);

        // Parallelize across the batch (img_h) and output channels (VH)
        #pragma omp parallel for collapse(2)
        for (long i = 0; i < img_h; ++i) {
            for (size_t h = 0; h < VH; ++h) {
                float accumulator = 0.0f;

                const float *input_block_ptr = in_img + i * FEATURE_BLOCK_SIZE;
                // Weights for this specific output channel
                const half_cpu *kernel_slice_ptr = static_cast<const half_cpu*>(conv_w->ptr({h}));

                // Vectorized dot product (Contraction)
                #pragma omp simd reduction(+:accumulator)
                for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                    // F16C expands kernel_slice_ptr[c] to fp32 for the FMA
                    accumulator += input_block_ptr[c] * static_cast<float>(kernel_slice_ptr[c]);
                }

                // Expand fp16 bias to fp32 and store result
                out_img[i * VH + h] = accumulator + static_cast<float>(conv_b_buf[h]);
            }
        }
    } else {
        // Quantized Weights and Scales
        const int8_t *w_q = static_cast<const int8_t*>(conv_w_ptr.buf);
        const float *w_scales = static_cast<const float*>(conv_w_ptr.scale);
        
        // Quantized Bias and Scales
        const int8_t *b_q = static_cast<const int8_t*>(conv_b_ptr.buf);
        const float *b_scales = static_cast<const float*>(conv_b_ptr.scale);

        const size_t group_size = conv_w->group_size;
        const size_t b_group_size = conv_b->group_size;

        #pragma omp parallel for collapse(2)
        for (long i = 0; i < img_h; ++i) {
            for (size_t h = 0; h < VH; ++h) {
                float accumulator = 0.0f;

                const float *input_block_ptr = in_img + i * FEATURE_BLOCK_SIZE;
                
                // Offset weights to the h-th output channel
                // Each channel has FEATURE_BLOCK_SIZE weights
                const int8_t *kernel_q_slice = w_q + (h * FEATURE_BLOCK_SIZE);
                // Each channel has (FEATURE_BLOCK_SIZE / group_size) scales
                const float *kernel_scales_slice = w_scales + (h * FEATURE_BLOCK_SIZE / group_size);

                // Inner dot product
                for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                    float scale = kernel_scales_slice[c / group_size];
                    float weight = static_cast<float>(kernel_q_slice[c]) * scale;
                    accumulator += input_block_ptr[c] * weight;
                }

                // Dequantize bias: bias[h] = b_q[h] * b_scale[h / b_group_size]
                float bias = static_cast<float>(b_q[h]) * b_scales[h / b_group_size];
                
                out_img[i * VH + h] = accumulator + bias;
            }
        }
    }
}

void vision_pos_embed(
    const Tensor *__restrict pos_embed_w, Tensor *__restrict x_tensor,
    int grid_h, int grid_w, int num_grid_per_side, int VSP, int VH
) {
    if (grid_h <= 0 || grid_w <= 0 || num_grid_per_side <= 0) {
        return;
    }

    float *__restrict x_embed = (float *)x_tensor->ptr();

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

    if (pos_embed_w->dtype == DType::FP32) {
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
    } else if (pos_embed_w->dtype == DType::FP16) {
        for (size_t i = 0; i < total_elements; ++i) {
            float *pos_embed_ptr_cur = pos_embed_ptr + (i * VH);

            // Initialize accumulator to zero
            for (size_t j = 0; j < VH; j++) {
                pos_embed_ptr_cur[j] = 0.0f;
            }

            for (int k = 0; k < 4; ++k) {
                long source_index = idx_list_mem[k][i];
                
                // CHANGE: Cast to half_cpu instead of float
                const half_cpu *pos_embed_w_idx_start = static_cast<const half_cpu*>(pos_embed_w->ptr({(size_t)source_index}));
                float weight = weight_list_mem[k][i];

                // Optimized with SIMD + F16C
                #pragma omp simd
                for (int j = 0; j < VH; ++j) {
                    // Conversion from fp16 to fp32 happens here via F16C
                    pos_embed_ptr_cur[j] += static_cast<float>(pos_embed_w_idx_start[j]) * weight;
                }
            }   
        }
    } else {
        PtrPair pe_start_ptr;
        for (size_t i = 0; i < total_elements; ++i) {
            float *pos_embed_ptr_cur = pos_embed_ptr + (i * VH);

            // Initialize accumulator to zero
            for (size_t j = 0; j < VH; j++) {
                pos_embed_ptr_cur[j] = 0.0f;
            }

            for (int k = 0; k < 4; ++k) {
                long source_index = idx_list_mem[k][i];
                
                // CHANGE: Cast to half_cpu instead of float
                pe_start_ptr = pos_embed_w->ptr_all({(size_t)source_index});
                const int8_t *pos_embed_w_idx_start = static_cast<const int8_t*>(pe_start_ptr.buf);
                const float *pos_embed_w_idx_start_scale = static_cast<const float*>(pe_start_ptr.scale);
                float weight = weight_list_mem[k][i];

                // Optimized with SIMD + F16C
                #pragma omp simd
                for (int j = 0; j < VH; ++j) {
                    // Conversion from fp16 to fp32 happens here via F16C
                    pos_embed_ptr_cur[j] += static_cast<float>(pos_embed_w_idx_start[j]) * pos_embed_w_idx_start_scale[j] * weight;
                }
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
    Tensor *__restrict pe_cos, Tensor *__restrict pe_sin,
    const Tensor *__restrict cos_total, const Tensor *__restrict sin_total,
    int grid_h, int grid_w, int merge_size, int head_dim
) {
    float *pos_emb_out_cos = (float *)pe_cos->ptr();
    float *pos_emb_out_sin = (float *)pe_sin->ptr();
    const float *cos_tensor = (const float *)cos_total->ptr();
    const float *sin_tensor = (const float *)sin_total->ptr();

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
    const Tensor *__restrict x,           /* [batches, hidden] */
    const Tensor *__restrict scale,       /* [layers, hidden] */
    const Tensor *__restrict bias,        /* [layers, hidden] */
    Tensor *__restrict out,               /* [batches, hidden] */
    float eps, size_t batches, size_t layer_offset
) {
    // 1. Pointers to current layer's weights
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    const float *x_buf = (float *)x->ptr();
    float *out_buf = (float *)out->ptr();

    if (scale->dtype == DType::FP32) {
        const float *scale_buf = (const float *)scale->ptr({layer_offset});
        const float *bias_buf = (const float *)bias->ptr({layer_offset});

        for (size_t i = 0; i < batches; i++) {
            // 2. Calculate Mean
            float mean = 0.0f;
            #pragma omp simd reduction(+:mean)
            for (size_t j = 0; j < hidden_size; j++) {
                mean += x_buf[j];
            }
            mean /= hidden_size;

            // 3. Calculate Variance (Sum of Squared Differences)
            float variance = 0.0f;
            #pragma omp simd reduction(+:variance)
            for (size_t j = 0; j < hidden_size; j++) {
                float diff = x_buf[j] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;

            // 4. Calculate Inverse Standard Deviation
            // nn.LayerNorm uses sqrt(variance + eps)
            float inv_std = 1.0f / sqrt(variance + eps);

            // 5. Normalize and Apply Affine Transform
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                // (x - mean) * inv_std * gamma + beta
                out_buf[j] = (x_buf[j] - mean) * inv_std * scale_buf[j] + bias_buf[j];
            }

            x_buf += hidden_size;
            out_buf += hidden_size;
        }
    } else if (scale->dtype == DType::FP16) {
        const half_cpu *scale_buf = static_cast<const half_cpu*>(scale->ptr({layer_offset}));
        const half_cpu *bias_buf = static_cast<const half_cpu*>(bias->ptr({layer_offset}));

        const float inv_hs = 1.0f / (float)(hidden_size);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < batches; i++) {
            const float *x_buf_ptr = x_buf + i * hidden_size;
            float *out_buf_ptr = out_buf + i * hidden_size;

            // 1. Calculate Mean (Vectorized Reduction)
            float mean = 0.0f;
            #pragma omp simd reduction(+:mean)
            for (size_t j = 0; j < hidden_size; j++) {
                mean += x_buf_ptr[j];
            }
            mean *= inv_hs;

            // 2. Calculate Variance
            float variance = 0.0f;
            #pragma omp simd reduction(+:variance)
            for (size_t j = 0; j < hidden_size; j++) {
                float diff = x_buf_ptr[j] - mean;
                variance += diff * diff;
            }
            variance *= inv_hs;

            // 3. Normalization Factor
            float inv_std = 1.0f / sqrtf(variance + eps);

            // 4. Normalize and Apply Affine Transform (Mixed Precision)
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                // scale_buf and bias_buf are fp16, converted via F16C during load
                float gamma = static_cast<float>(scale_buf[j]);
                float beta  = static_cast<float>(bias_buf[j]);
                
                out_buf_ptr[j] = (x_buf_ptr[j] - mean) * inv_std * gamma + beta;
            }
        }
    } else {
        PtrPair scale_ptr = scale->ptr_all({layer_offset});
        PtrPair bias_ptr = bias->ptr_all({layer_offset});
        // Quantized scale (gamma) and its FP32 scales
        const int8_t *scale_q = static_cast<const int8_t*>(scale_ptr.buf);
        const float *scale_scales = static_cast<const float*>(scale_ptr.scale);
        
        // Quantized bias (beta) and its FP32 scales
        const int8_t *bias_q = static_cast<const int8_t*>(bias_ptr.buf);
        const float *bias_scales = static_cast<const float*>(bias_ptr.scale);

        const size_t group_size = scale->group_size;
        const size_t b_group_size = bias->group_size;

        for (size_t i = 0; i < batches; i++) {
            // 1. Calculate Mean
            float mean = 0.0f;
            #pragma omp simd reduction(+:mean)
            for (size_t j = 0; j < hidden_size; j++) {
                mean += x_buf[j];
            }
            mean /= hidden_size;

            // 2. Calculate Variance
            float variance = 0.0f;
            #pragma omp simd reduction(+:variance)
            for (size_t j = 0; j < hidden_size; j++) {
                float diff = x_buf[j] - mean;
                variance += diff * diff;
            }
            variance /= hidden_size;

            // 3. Normalization Factor
            float inv_std = 1.0f / sqrtf(variance + eps);

            // 4. Normalize and Apply Dequantized Affine Transform
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                // Dequantize gamma (scale)
                float gamma = static_cast<float>(scale_q[j]) * scale_scales[j / group_size];
                // Dequantize beta (bias)
                float beta = static_cast<float>(bias_q[j]) * bias_scales[j / b_group_size];
                
                out_buf[j] = (x_buf[j] - mean) * inv_std * gamma + beta;
            }

            x_buf += hidden_size;
            out_buf += hidden_size;
        }
    }
}

void vision_apply_rotary_inplace(
    const Tensor *__restrict cos_total, // shape (T, HD)
    const Tensor *__restrict sin_total, // shape (T, HD)
    Tensor *__restrict tensor_buffer,   // shape (T, NH, HD)
    long total_tokens, int num_heads, int head_dim
) {
    const float *__restrict cos_tensor = (const float *)cos_total->ptr();
    const float *__restrict sin_tensor = (const float *)sin_total->ptr();
    float *__restrict buffer = (float *)tensor_buffer->ptr();
    const int half_dim = head_dim / 2;

    // Parallelize across tokens - this is usually the largest dimension
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total_tokens; ++t) {
        
        // Pre-calculate the cos/sin row for this specific token
        const float *c_ptr = cos_tensor + (t * head_dim);
        const float *s_ptr = sin_tensor + (t * head_dim);

        for (int h = 0; h < num_heads; ++h) {
            // Calculate the start of the current head's data
            float *current_x1 = buffer + (t * num_heads * head_dim) + (h * head_dim);
            float *current_x2 = current_x1 + half_dim;

            // Use pragma simd to tell the compiler to use AVX2/AVX-512 automatically
            #pragma omp simd
            for (int m = 0; m < half_dim; ++m) {
                float x1_val = current_x1[m];
                float x2_val = current_x2[m];
                
                float c1 = c_ptr[m];
                float s1 = s_ptr[m];
                float c2 = c_ptr[m + half_dim];
                float s2 = s_ptr[m + half_dim];

                // Rotation math
                current_x1[m] = (x1_val * c1) - (x2_val * s1);
                current_x2[m] = (x2_val * c2) + (x1_val * s2);
            }
        }
    }
}

void tensor_transpose(
    const Tensor *__restrict in_tensor,
    Tensor *__restrict out_tensor,
    int D0, int D1, int D2
) {
    const float *__restrict in  = (const float *)in_tensor->ptr();
    float *__restrict out = (float *)out_tensor->ptr();

    const int in_stride_0  = D1 * D2;
    const int out_stride_1 = D0 * D2;
    const size_t block_size = (size_t)D2 * sizeof(float);

    // Tile sizes (tuneable)
    // Rule of thumb: Ti * Tj * D2 * sizeof(float) ~ L1 or L2 size
    const int Ti = 16;
    const int Tj = 16;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < D0; ii += Ti) {
        for (int jj = 0; jj < D1; jj += Tj) {

            const int i_end = (ii + Ti < D0) ? ii + Ti : D0;
            const int j_end = (jj + Tj < D1) ? jj + Tj : D1;

            for (int i = ii; i < i_end; ++i) {
                const float *in_i = in + i * in_stride_0;
                for (int j = jj; j < j_end; ++j) {

                    const float *src = in_i + j * D2;
                    float *dst = out + (j * out_stride_1) + (i * D2);

                    memcpy(dst, src, block_size);
                }
            }
        }
    }
}

static inline float avx2_max(const float *arr, size_t N) {
    size_t i = 0;

    // Initialize vector accumulator with -inf
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);

    // Process 8 floats at a time
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    // Horizontal reduction of vmax
    __m128 low  = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 vmax128 = _mm_max_ps(low, high);

    vmax128 = _mm_max_ps(vmax128, _mm_movehl_ps(vmax128, vmax128));
    vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, 0x55));

    float max_val = _mm_cvtss_f32(vmax128);

    // Handle remaining elements
    for (; i < N; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    return max_val;
}

static inline float avx2_max_and_scale(float *arr, size_t N, float scale) {
    size_t i = 0;

    // Initialize vector accumulator with -inf
    __m256 vmax = _mm256_set1_ps(-FLT_MAX);
    __m256 v_scale = _mm256_set1_ps(scale);

    // Process 8 floats at a time
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        v = _mm256_mul_ps(v, v_scale);
        vmax = _mm256_max_ps(vmax, v);
        _mm256_storeu_ps(arr + i, v);
    }

    // Horizontal reduction of vmax
    __m128 low  = _mm256_castps256_ps128(vmax);
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 vmax128 = _mm_max_ps(low, high);

    vmax128 = _mm_max_ps(vmax128, _mm_movehl_ps(vmax128, vmax128));
    vmax128 = _mm_max_ps(vmax128, _mm_shuffle_ps(vmax128, vmax128, 0x55));

    float max_val = _mm_cvtss_f32(vmax128);

    // Handle remaining elements
    for (; i < N; i++) {
        arr[i] *= scale;
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }

    return max_val;
}

static inline __m256 exp256_ps(__m256 x) {
    // Clamp
    const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    // Constants
    const __m256 ln2      = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2  = _mm256_set1_ps(1.44269504089f);

    // n = floor(x / ln2)
    __m256 fx = _mm256_mul_ps(x, inv_ln2);
    fx = _mm256_floor_ps(fx);

    __m256i emm0 = _mm256_cvttps_epi32(fx);

    // g = x - n * ln2
    __m256 g = _mm256_fnmadd_ps(fx, ln2, x);

    // Polynomial approximation of exp(g)
    // exp(g) ≈ 1 + g + g²/2 + g³/6 + g⁴/24
    __m256 y = _mm256_set1_ps(1.0f);
    y = _mm256_fmadd_ps(g, y, _mm256_set1_ps(1.0f));

    __m256 g2 = _mm256_mul_ps(g, g);
    y = _mm256_fmadd_ps(g2, _mm256_set1_ps(0.5f), y);

    __m256 g3 = _mm256_mul_ps(g2, g);
    y = _mm256_fmadd_ps(g3, _mm256_set1_ps(1.0f / 6.0f), y);

    __m256 g4 = _mm256_mul_ps(g3, g);
    y = _mm256_fmadd_ps(g4, _mm256_set1_ps(1.0f / 24.0f), y);

    // Build 2^n
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
}

static inline float avx2_sum_exp_max(float *arr, size_t T, float max_score) {
    __m256 vsum = _mm256_setzero_ps();
    __m256 vmax = _mm256_set1_ps(max_score);

    int j = 0;
    for (; j + 7 < T; j += 8) {
        __m256 v = _mm256_loadu_ps(arr + j);
        v = _mm256_sub_ps(v, vmax);
        v = exp256_ps(v);
        _mm256_storeu_ps(arr + j, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 s  = _mm_add_ps(lo, hi);

    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);

    float sum = _mm_cvtss_f32(s);

    for (; j < T; j++) {
        arr[j] = expf(arr[j] - max_score);
        sum += arr[j];
    }

    return sum;
}

void vision_att(
    const float *q, const float *k, const float *v, float *attn_scores, 
    float *out, int num_heads, int T, int D, float scale
) {
    #ifdef CPU_TIME_GEMM
        CPUTimer timer("gemm_att");
        printf("Shape of gemm att w/ precision FP32: T=%zu, D=%zu\n", T, D);
    #endif

    size_t head_stride = (size_t)T * D;

    for (int h = 0; h < num_heads; h++) {
        const float *qh = q + h * head_stride;
        const float *kh = k + h * head_stride;
        const float *vh = v + h * head_stride;
        float *oh = out + h * head_stride;

        for (int i = 0; i < T; i++) {
            const float *qi = qh + i * D;

            gemm_att(qi, kh, attn_scores, scale, T, D, true);
            float max_score = avx2_max(attn_scores, T);
            // float max_score = avx2_max_and_scale(attn_scores, T, scale);
            float sum = avx2_sum_exp_max(attn_scores, T, max_score);
            gemm_att(attn_scores, vh, oh + i * D, 1.0f / sum, D, T, false);
        }
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

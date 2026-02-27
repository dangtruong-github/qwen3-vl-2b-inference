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

    DType::Type out_img_type = out_img_tensor->dtype;

    if (out_img_type == DType::FP32) {
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
    } else if (out_img_type == DType::FP16) {
        half_cpu *__restrict out_img =
            static_cast<half_cpu *>(out_img_tensor->ptr());

        /* ================= FP32 weights ================= */
        if (conv_w->dtype == DType::FP32) {
            const float *__restrict conv_b_buf =
                static_cast<const float *>(conv_b_ptr.buf);
            const float *__restrict w_buf =
                static_cast<const float *>(conv_w_ptr.buf);

            #pragma omp parallel for collapse(2)
            for (long i = 0; i < img_h; ++i) {
                for (size_t h = 0; h < VH; ++h) {
                    float acc = 0.0f;

                    const float *input_block_ptr =
                        in_img + i * FEATURE_BLOCK_SIZE;
                    const float *kernel_slice_ptr =
                        w_buf + h * FEATURE_BLOCK_SIZE;

                    #pragma omp simd reduction(+:acc)
                    for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                        acc += input_block_ptr[c] * kernel_slice_ptr[c];
                    }

                    acc += conv_b_buf[h];
                    out_img[i * VH + h] = static_cast<half_cpu>(acc);
                }
            }

        /* ================= FP16 weights ================= */
        } else if (conv_w->dtype == DType::FP16) {
            const half_cpu *__restrict conv_b_buf =
                static_cast<const half_cpu *>(conv_b_ptr.buf);
            const half_cpu *__restrict w_buf =
                static_cast<const half_cpu *>(conv_w_ptr.buf);

            #pragma omp parallel for collapse(2)
            for (long i = 0; i < img_h; ++i) {
                for (size_t h = 0; h < VH; ++h) {
                    float acc = 0.0f;

                    const float *input_block_ptr =
                        in_img + i * FEATURE_BLOCK_SIZE;
                    const half_cpu *kernel_slice_ptr =
                        w_buf + h * FEATURE_BLOCK_SIZE;

                    #pragma omp simd reduction(+:acc)
                    for (long c = 0; c < FEATURE_BLOCK_SIZE; ++c) {
                        acc += input_block_ptr[c] *
                            static_cast<float>(kernel_slice_ptr[c]);
                    }

                    acc += static_cast<float>(conv_b_buf[h]);
                    out_img[i * VH + h] = static_cast<half_cpu>(acc);
                }
            }

        /* ================= INT8 weights ================= */
        } else {
            const int8_t *__restrict w_q =
                static_cast<const int8_t *>(conv_w_ptr.buf);
            const float *__restrict w_scales =
                static_cast<const float *>(conv_w_ptr.scale);

            const int8_t *__restrict b_q =
                static_cast<const int8_t *>(conv_b_ptr.buf);
            const float *__restrict b_scales =
                static_cast<const float *>(conv_b_ptr.scale);

            const size_t group_size = conv_w->group_size;
            const size_t b_group_size = conv_b->group_size;

            #pragma omp parallel for collapse(2)
            for (long i = 0; i < img_h; ++i) {
                for (size_t h = 0; h < VH; ++h) {
                    float acc = 0.0f;

                    const float *input_block_ptr =
                        in_img + i * FEATURE_BLOCK_SIZE;

                    const int8_t *kernel_q_slice =
                        w_q + h * FEATURE_BLOCK_SIZE;
                    const float *kernel_scales_slice =
                        w_scales + (h * FEATURE_BLOCK_SIZE / group_size);

                    // scale-hoisted loop
                    for (long g = 0; g < FEATURE_BLOCK_SIZE; g += group_size) {
                        float scale = kernel_scales_slice[g / group_size];
                        for (long c = g; c < g + group_size; ++c) {
                            acc += input_block_ptr[c] *
                                (static_cast<float>(kernel_q_slice[c]) * scale);
                        }
                    }

                    float bias =
                        static_cast<float>(b_q[h]) *
                        b_scales[h / b_group_size];

                    out_img[i * VH + h] =
                        static_cast<half_cpu>(acc + bias);
                }
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

    bool x_is_fp16 = (x_tensor->dtype == DType::FP16);

    float *x_embed_fp32 = nullptr;
    half_cpu *x_embed_fp16 = nullptr;

    if (x_is_fp16) {
        x_embed_fp16 = static_cast<half_cpu *>(x_tensor->ptr());
    } else {
        x_embed_fp32 = static_cast<float *>(x_tensor->ptr());
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

                        if (!x_is_fp16) {
                            float *dest_ptr = x_embed_fp32 + dest_flat_idx;
                            memcpy(dest_ptr, source_ptr, token_size * sizeof(float));
                        } else {
                            half_cpu *dest_ptr = x_embed_fp16 + dest_flat_idx;

                            #pragma omp simd
                            for (int j = 0; j < VH; ++j) {
                                dest_ptr[j] = static_cast<half_cpu>(source_ptr[j]);
                            }
                        }
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
    // check if pe_cos->dtype == pe_sin->dtype == cos_total->dtype == sin_total->dtype
    int max_hw = max(grid_h, grid_w);
    int total_tokens = grid_h * grid_w;
    int freqs_depth_dim = head_dim / 4;

    // --- 1. Generate Coordinates (as completed previously) ---
    int merged_h = grid_h / merge_size;
    int merged_w = grid_w / merge_size;

    if (pe_cos->dtype == DType::FP16) {
        half_cpu *pos_emb_out_cos = (half_cpu *)pe_cos->ptr();
        half_cpu *pos_emb_out_sin = (half_cpu *)pe_sin->ptr();
        const half_cpu *cos_tensor = (const half_cpu *)cos_total->ptr();
        const half_cpu *sin_tensor = (const half_cpu *)sin_total->ptr();

        size_t size_copy = freqs_depth_dim * sizeof(half_cpu);

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
                            const half_cpu *row_freqs_cos =
                                cos_tensor + row_idx * freqs_depth_dim;
                            const half_cpu *col_freqs_cos =
                                cos_tensor + col_idx * freqs_depth_dim;
                            half_cpu *out_cos =
                                pos_emb_out_cos + k * head_dim;

                            memcpy(out_cos, row_freqs_cos, size_copy);
                            memcpy(out_cos + freqs_depth_dim,
                                col_freqs_cos, size_copy);
                            memcpy(out_cos + 2 * freqs_depth_dim,
                                row_freqs_cos, size_copy);
                            memcpy(out_cos + 3 * freqs_depth_dim,
                                col_freqs_cos, size_copy);

                            // sin
                            const half_cpu *row_freqs_sin =
                                sin_tensor + row_idx * freqs_depth_dim;
                            const half_cpu *col_freqs_sin =
                                sin_tensor + col_idx * freqs_depth_dim;
                            half_cpu *out_sin =
                                pos_emb_out_sin + k * head_dim;

                            memcpy(out_sin, row_freqs_sin, size_copy);
                            memcpy(out_sin + freqs_depth_dim,
                                col_freqs_sin, size_copy);
                            memcpy(out_sin + 2 * freqs_depth_dim,
                                row_freqs_sin, size_copy);
                            memcpy(out_sin + 3 * freqs_depth_dim,
                                col_freqs_sin, size_copy);

                            k++;
                        }
                    }
                }
            }
        }

        return;
    } else {
        float *pos_emb_out_cos = (float *)pe_cos->ptr();
        float *pos_emb_out_sin = (float *)pe_sin->ptr();
        const float *cos_tensor = (const float *)cos_total->ptr();
        const float *sin_tensor = (const float *)sin_total->ptr();

        size_t size_copy = freqs_depth_dim * sizeof(float);
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
}

void layer_norm(
    const Tensor *__restrict x,           /* [batches, hidden] */
    const Tensor *__restrict scale,       /* [layers, hidden] */
    const Tensor *__restrict bias,        /* [layers, hidden] */
    Tensor *__restrict out,               /* [batches, hidden] */
    float eps, size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];

    const void *x_buf = x->ptr();
    void *out_buf = out->ptr();

    const DType::Type x_dtype   = x->dtype;
    const DType::Type out_dtype = out->dtype;
    const DType::Type p_dtype   = scale->dtype;

    const float inv_hs = 1.0f / hidden_size;

    /* ===== Load parameter pointers ===== */
    const float *scale_f32 = nullptr;
    const float *bias_f32  = nullptr;

    const half_cpu *scale_f16 = nullptr;
    const half_cpu *bias_f16  = nullptr;

    const int8_t *scale_q = nullptr;
    const int8_t *bias_q  = nullptr;
    const float *scale_scales = nullptr;
    const float *bias_scales  = nullptr;

    size_t scale_gs = 0, bias_gs = 0;

    if (p_dtype == DType::FP32) {
        scale_f32 = static_cast<const float*>(scale->ptr({layer_offset}));
        bias_f32  = static_cast<const float*>(bias->ptr({layer_offset}));
    } else if (p_dtype == DType::FP16) {
        scale_f16 = static_cast<const half_cpu*>(scale->ptr({layer_offset}));
        bias_f16  = static_cast<const half_cpu*>(bias->ptr({layer_offset}));
    } else { // INT8 + FP32 scales
        PtrPair s = scale->ptr_all({layer_offset});
        PtrPair b = bias->ptr_all({layer_offset});

        scale_q = static_cast<const int8_t*>(s.buf);
        bias_q  = static_cast<const int8_t*>(b.buf);

        scale_scales = static_cast<const float*>(s.scale);
        bias_scales  = static_cast<const float*>(b.scale);

        scale_gs = scale->group_size;
        bias_gs  = bias->group_size;
    }

    /* ===== Main kernel ===== */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < batches; i++) {
        const size_t base = i * hidden_size;

        /* ---- mean ---- */
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (size_t j = 0; j < hidden_size; j++) {
            mean += load_x(x_buf, x_dtype, base + j);
        }
        mean *= inv_hs;

        /* ---- variance ---- */
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (size_t j = 0; j < hidden_size; j++) {
            float v = load_x(x_buf, x_dtype, base + j) - mean;
            var += v * v;
        }
        var *= inv_hs;

        const float inv_std = 1.0f / sqrtf(var + eps);

        /* ---- normalize + affine ---- */
        #pragma omp simd
        for (size_t j = 0; j < hidden_size; j++) {
            float xval = load_x(x_buf, x_dtype, base + j);

            float gamma, beta;
            if (p_dtype == DType::FP32) {
                gamma = scale_f32[j];
                beta  = bias_f32[j];
            } else if (p_dtype == DType::FP16) {
                gamma = static_cast<float>(scale_f16[j]);
                beta  = static_cast<float>(bias_f16[j]);
            } else { // INT8
                gamma = static_cast<float>(scale_q[j]) *
                        scale_scales[j / scale_gs];
                beta  = static_cast<float>(bias_q[j]) *
                        bias_scales[j / bias_gs];
            }

            float y = (xval - mean) * inv_std * gamma + beta;
            store_out(out_buf, out_dtype, base + j, y);
        }
    }
}

void vision_apply_rotary_inplace(
    const Tensor *__restrict cos_total, // (T, HD)
    const Tensor *__restrict sin_total, // (T, HD)
    Tensor *__restrict tensor_buffer,   // (T, NH, HD)
    long total_tokens, int num_heads, int head_dim
) {
    const void *__restrict cos_buf = cos_total->ptr();
    const void *__restrict sin_buf = sin_total->ptr();
    void *__restrict buf = tensor_buffer->ptr();

    const DType::Type cs_dtype  = cos_total->dtype;
    const DType::Type x_dtype   = tensor_buffer->dtype;

    const int half_dim = head_dim >> 1;
    const size_t head_stride  = head_dim;
    const size_t token_stride = (size_t)num_heads * head_dim;

    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total_tokens; ++t) {
        const size_t cs_base = (size_t)t * head_dim;
        const size_t x_base  = (size_t)t * token_stride;

        for (int h = 0; h < num_heads; ++h) {
            const size_t head_base = x_base + (size_t)h * head_stride;

            #pragma omp simd
            for (int m = 0; m < half_dim; ++m) {
                const size_t i1 = head_base + m;
                const size_t i2 = head_base + m + half_dim;

                // Load x
                float x1 = load_x(buf, x_dtype, i1);
                float x2 = load_x(buf, x_dtype, i2);

                // Load cos/sin (same dtype for both)
                float c1 = load_x(cos_buf, cs_dtype, cs_base + m);
                float s1 = load_x(sin_buf, cs_dtype, cs_base + m);
                float c2 = load_x(cos_buf, cs_dtype, cs_base + m + half_dim);
                float s2 = load_x(sin_buf, cs_dtype, cs_base + m + half_dim);

                // Rotary
                float y1 = (x1 * c1) - (x2 * s1);
                float y2 = (x2 * c2) + (x1 * s2);

                // Store back
                store_out(buf, x_dtype, i1, y1);
                store_out(buf, x_dtype, i2, y2);
            }
        }
    }
}

void tensor_transpose(
    const Tensor *__restrict in_tensor,
    Tensor *__restrict out_tensor,
    int D0, int D1, int D2
) {
    if (in_tensor->dtype != out_tensor->dtype) {
        fprintf(stderr, "Wrong dtype b/w in_tensor %s and out_tensor %s\n", dtypeToStr(in_tensor->dtype), dtypeToStr(out_tensor->dtype));
        exit(1);
    }
    const void *__restrict in  = in_tensor->ptr();
    void *__restrict out = out_tensor->ptr();

    const size_t elem_size = in_tensor->get_dtype_size(); // 2 for FP16, 4 for FP32
    const size_t block_size = (size_t)D2 * elem_size;

    const int in_stride_0  = D1 * D2;
    const int out_stride_1 = D0 * D2;

    const int Ti = 16;
    const int Tj = 16;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < D0; ii += Ti) {
        for (int jj = 0; jj < D1; jj += Tj) {

            const int i_end = (ii + Ti < D0) ? ii + Ti : D0;
            const int j_end = (jj + Tj < D1) ? jj + Tj : D1;

            for (int i = ii; i < i_end; ++i) {
                const char *in_i = (const char *)in + (size_t)i * in_stride_0 * elem_size;

                for (int j = jj; j < j_end; ++j) {
                    const void *src = in_i + (size_t)j * D2 * elem_size;
                    void *dst = (char *)out
                        + ((size_t)j * out_stride_1 + (size_t)i * D2) * elem_size;

                    memcpy(dst, src, block_size);
                }
            }
        }
    }
}

void vision_att(
    const Tensor *q_tensor, const Tensor *k_tensor,
    const Tensor *v_tensor, Tensor *attn_scores_tensor, 
    Tensor *out_tensor, int num_heads, int T, int D, size_t max_attn_size, float scale
) {
    #ifdef CPU_TIME_GEMM
        CPUTimer timer("gemm_att");
        printf("Shape of gemm att w/ precision FP32: T=%zu, D=%zu\n", T, D);
    #endif

    size_t head_stride = (size_t)T * D;
    DType::Type q_type = q_tensor->dtype;
    DType::Type k_type = k_tensor->dtype;
    DType::Type v_type = v_tensor->dtype;
    DType::Type att_s_type = attn_scores_tensor->dtype;
    DType::Type out_type = out_tensor->dtype;

    const char *q = (const char *)q_tensor->ptr();
    const char *k = (const char *)k_tensor->ptr();
    const char *v = (const char *)v_tensor->ptr();
    char *out = (char *)out_tensor->ptr();

    const size_t q_size = q_tensor->get_dtype_size();
    const size_t k_size = k_tensor->get_dtype_size();
    const size_t v_size = v_tensor->get_dtype_size();
    const size_t out_size = out_tensor->get_dtype_size();

    float *attn_scores = (float *)attn_scores_tensor->ptr();

    float max_scores[max_attn_size];  

    for (int h = 0; h < num_heads; ++h) {
        const char *qh = q + 1ll * h * head_stride * q_size;
        const char *kh = k + 1ll * h * head_stride * k_size;
        const char *vh = v + 1ll * h * head_stride * v_size;
        char *oh = out + 1ll * h * head_stride * out_size;

        for (size_t i = 0; i < T; i += max_attn_size) {
            const size_t cur_attn_size = std::min(T - i, max_attn_size);

            const char *qi = qh + 1ll * i * D * q_size;
            char *oi = oh + 1ll * i * D * out_size;

            gemm_att(qi, kh, attn_scores, scale, cur_attn_size, T, D, true, q_type, k_type, att_s_type);
            avx2_max_multiple(attn_scores, T, cur_attn_size, max_scores);
            // float max_score = avx2_max_and_scale(attn_scores, T, scale);
            avx2_sum_exp_max_multiple(attn_scores, T, cur_attn_size, max_scores);
            gemm_att_multiple_scale(attn_scores, vh, oi, max_scores, cur_attn_size, D, T, false, att_s_type, v_type, out_type);

            /*
            for (size_t i_sm = 0; i_sm < cur_attn_size; ++i_sm) {
                char *oi_sm = oi + 1ll * i_sm * D * out_size;

                if (out_type == DType::FP16) {
                    half_cpu *oi_cpu = (half_cpu *)oi_sm;

                    for (size_t d = 0; d < D; ++d) {
                        oi_cpu[d] *= max_scores[i_sm];
                    }
                } else {
                    float *oi_cpu = (float *)oi_sm;

                    for (size_t d = 0; d < D; ++d) {
                        oi_cpu[d] *= max_scores[i_sm];
                    }
                }
            }
            */
        }
    }
}

void gelu_tanh(Tensor *x, size_t x_size) {
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;

    switch (x->dtype) {

    case DType::FP32: {
        float *x_buf = (float *)x->ptr();

        for (size_t i = 0; i < x_size; ++i) {
            float xi = x_buf[i];
            float xi3 = xi * xi * xi;
            float inner = sqrt_2_over_pi * (xi + coeff * xi3);
            float tanh_inner = tanhf(inner);
            x_buf[i] = 0.5f * xi * (1.0f + tanh_inner);
        }
        break;
    }

    case DType::FP16: {
        half_cpu *x_buf = (half_cpu *)x->ptr();

        for (size_t i = 0; i < x_size; ++i) {
            // FP16 → FP32
            float xi = (float)x_buf[i];

            float xi3 = xi * xi * xi;
            float inner = sqrt_2_over_pi * (xi + coeff * xi3);
            float tanh_inner = tanhf(inner);

            float y = 0.5f * xi * (1.0f + tanh_inner);

            // FP32 → FP16
            x_buf[i] = (half_cpu)y;
        }
        break;
    }

    default:
        fprintf(stderr, "gelu_tanh: unsupported dtype of x: %s\n", dtypeToStr(x->dtype));
        exit(1);
    }
}

#include "../include/text_layer.hpp"

void embedding_lookup(
    const Tensor *__restrict embedding /*[vocab, hidden]*/, 
    Tensor *__restrict out /*[hidden]*/,
    size_t out_id, size_t token_id, size_t hidden_size
) {
    PtrPair emb_ptr = embedding->ptr_all({token_id});
    float *out_ptr = static_cast<float*>(out->ptr({out_id}));

    if (embedding->dtype == DType::FP32) {
        memcpy(out_ptr, emb_ptr.buf, hidden_size * sizeof(float));
    } else if (embedding->dtype == DType::FP16) {
        // Get typed pointers
        const half_cpu *__restrict src = static_cast<const half_cpu*>(emb_ptr.buf);

        // Loop-based conversion (memcpy cannot be used here)
        #pragma omp simd
        for (size_t i = 0; i < hidden_size; ++i) {
            out_ptr[i] = static_cast<float>(src[i]);
        }
    } else {
        const int8_t *__restrict src_q = static_cast<const int8_t*>(emb_ptr.buf);
        const float *__restrict scales = static_cast<const float*>(emb_ptr.scale);

        if (embedding->group_quantized) {
            size_t group_size = embedding->group_size; 
            size_t groups = hidden_size / group_size;

            #pragma omp parallel for
            for (size_t g = 0; g < groups; ++g) {
                float scale = scales[g];
                size_t base = g * group_size;

                for (size_t i = 0; i < group_size; ++i) {
                    size_t idx = base + i;
                    out_ptr[idx] = (float)src_q[idx] * scale;
                }
            }
        } else {
            float scale_token = scales[0];

            #pragma omp simd
            for (size_t i = 0; i < hidden_size; ++i) {
                out_ptr[i] = (float)src_q[i] * scale_token;
            }
        }
    }
}

void rms_norm(
    const Tensor *__restrict x_tensor /*[hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    Tensor *__restrict out_tensor /*[hidden]*/, 
    float eps, size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    const float inv_hs = 1.0f / (float)hidden_size;

    PtrPair scale_ptr = scale->ptr_all({layer_offset});

    const float *x = (const float *)(x_tensor->ptr());
    float *out = (float *)(out_tensor->ptr());
    
    if (scale->dtype == DType::FP32) {
        const float *__restrict scale_buf = (const float *)(scale_ptr.buf);

        for (size_t i = 0; i < batches; i++) {
            // calculate sum of squares
            double ss = 0.0;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < hidden_size; j++) {
                ss += x[j] * x[j];
            }
            ss /= hidden_size;
            ss += eps;
            ss = 1.0 / sqrt(ss);
            // normalize and scale
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                out[j] = scale_buf[j] * (ss * x[j]);
            }
            
            x += hidden_size;
            out += hidden_size;
        }
    } else if (scale->dtype == DType::FP16) {
        const half_cpu *__restrict scale_buf = static_cast<const half_cpu*>(scale_ptr.buf);

        for (size_t i = 0; i < batches; i++) {
            // calculate sum of squares
            double ss = 0.0;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < hidden_size; j++) {
                ss += x[j] * x[j];
            }
            ss /= hidden_size;
            ss += eps;
            ss = 1.0 / sqrt(ss);
            // normalize and scale
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                out[j] = static_cast<float>(scale_buf[j]) * (ss * x[j]);
            }
            
            x += hidden_size;
            out += hidden_size;
        }
    } else {
        const int8_t *__restrict scale_q = static_cast<const int8_t*>(scale_ptr.buf);
        const float *__restrict scale_scales = static_cast<const float*>(scale_ptr.scale);

        if (scale->group_quantized) {
            const size_t group_size = scale->group_size;

            #pragma omp parallel for
            for (size_t i = 0; i < batches; i++) {
                const float *x_ptr = x + i * hidden_size;
                float *out_ptr = out + i * hidden_size;

                // 1. Calculate sum of squares (standard RMS logic)
                float ss = 0.0f;
                #pragma omp simd reduction(+:ss)
                for (size_t j = 0; j < hidden_size; ++j) {
                    ss += x_ptr[j] * x_ptr[j];
                }
                
                float inv_rms = static_cast<float>(1.0 / sqrt(ss * inv_hs + eps));

                // 2. Normalize and apply dequantized weights
                // out[j] = (x[j] * inv_rms) * (scale_q[j] * scale_scales[j / group_size])
                for (size_t g = 0; g < hidden_size; g += group_size) {
                    // Load the scale once for the entire tile
                    float scale = scale_scales[g / group_size];
                    float combined_factor = scale * inv_rms;

                    const int8_t* s_q = &scale_q[g];
                    const float* x_p = &x_ptr[g];
                    float* o_p = &out_ptr[g];

                    #pragma omp simd
                    for (size_t j = 0; j < group_size; ++j) {
                        o_p[j] = static_cast<float>(s_q[j]) * (combined_factor * x_p[j]);
                    }
                }
            }
        } else {
            const float scale_base = scale_scales[0];

            #pragma omp parallel for
            for (size_t i = 0; i < batches; i++) {
                const float *x_ptr = x + i * hidden_size;
                float *out_ptr = out + i * hidden_size;

                // 1. RMS
                float ss = 0.0f;
                #pragma omp simd reduction(+:ss)
                for (size_t j = 0; j < hidden_size; ++j) {
                    ss += x_ptr[j] * x_ptr[j];
                }

                float inv_rms = static_cast<float>(1.0 / sqrt(ss * inv_hs + eps));

                // 2. One scale per channel vector
                float combined = scale_base * inv_rms;

                #pragma omp simd
                for (size_t j = 0; j < hidden_size; ++j) {
                    out_ptr[j] = x_ptr[j] * combined * static_cast<float>(scale_q[j]);
                }
            }
        }
    }
}

void rms_norm_inplace(
    float *__restrict x /*[batches, hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    float eps, size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    const float inv_hs = 1.0f / (float)hidden_size;

    PtrPair scale_ptr = scale->ptr_all({layer_offset});

    if (scale->dtype == DType::FP32) {
        const float *__restrict scale_buf = (const float *)(scale_ptr.buf);

        for (size_t i = 0; i < batches; ++i) {
            // 1. RMS
            double ss = 0.0;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < hidden_size; ++j) {
                ss += x[j] * x[j];
            }

            float inv_rms = (float)(1.0 / sqrt(ss * inv_hs + eps));

            // 2. Normalize + scale (inplace)
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; ++j) {
                x[j] = scale_buf[j] * (inv_rms * x[j]);
            }

            x += hidden_size;
        }

    } else if (scale->dtype == DType::FP16) {
        const half_cpu *__restrict scale_buf =
            static_cast<const half_cpu *>(scale_ptr.buf);

        for (size_t i = 0; i < batches; ++i) {
            // 1. RMS
            double ss = 0.0;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < hidden_size; ++j) {
                ss += x[j] * x[j];
            }

            float inv_rms = (float)(1.0 / sqrt(ss * inv_hs + eps));

            // 2. Normalize + scale (inplace)
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; ++j) {
                x[j] = static_cast<float>(scale_buf[j]) * (inv_rms * x[j]);
            }

            x += hidden_size;
        }

    } else {
        // INT8 group-wise scale
        const int8_t *__restrict scale_q =
            static_cast<const int8_t *>(scale_ptr.buf);
        const float *__restrict scale_scales =
            static_cast<const float *>(scale_ptr.scale);
        
        if (scale->group_quantized) {
            const size_t group_size = scale->group_size;

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < batches; ++i) {
                float *x_ptr = x + i * hidden_size;

                // 1. RMS
                float ss = 0.0f;
                #pragma omp simd reduction(+:ss)
                for (size_t j = 0; j < hidden_size; ++j) {
                    ss += x_ptr[j] * x_ptr[j];
                }

                float inv_rms =
                    1.0f / sqrtf(ss * inv_hs + eps);

                // 2. Normalize + dequantized scale (inplace)
                for (size_t g = 0; g < hidden_size; g += group_size) {
                    float s = scale_scales[g / group_size];
                    float combined = s * inv_rms;

                    const int8_t *sq = &scale_q[g];
                    float *xp = &x_ptr[g];

                    #pragma omp simd
                    for (size_t j = 0; j < group_size; ++j) {
                        xp[j] = static_cast<float>(sq[j]) *
                                (combined * xp[j]);
                    }
                }
            }
        } else {
            const float scale_base = scale_scales[layer_offset];

            #pragma omp parallel for
            for (size_t i = 0; i < batches; i++) {
                float *x_ptr = x + i * hidden_size;
                const int8_t *s_q = scale_q + i * hidden_size;

                // 1. RMS
                float ss = 0.0f;
                #pragma omp simd reduction(+:ss)
                for (size_t j = 0; j < hidden_size; ++j) {
                    ss += x_ptr[j] * x_ptr[j];
                }

                float inv_rms = static_cast<float>(1.0 / sqrt(ss * inv_hs + eps));

                // 2. One scale per channel vector
                float combined = scale_base * inv_rms;

                #pragma omp simd
                for (size_t j = 0; j < hidden_size; ++j) {
                    x_ptr[j] *= combined * static_cast<float>(s_q[j]);
                }
            }
        }
    }
}

void classifier_gemm(
    const Tensor *__restrict embedding /*[vocab, hidden]*/,
    const Tensor *__restrict hid_states /*[hidden]*/,
    Tensor *__restrict logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
) {
    PtrPair emb_ptr = embedding->ptr_all();
    linear(
        hid_states->ptr(), emb_ptr.buf, emb_ptr.scale, nullptr,
        nullptr, nullptr, logits->ptr(), 1, vocab_size, hidden_size, true,
        hid_states->dtype, embedding->dtype, embedding->scale_dtype,
        logits->dtype, embedding->group_quantized,
        embedding->group_size, false
    );
}

void add_vector(
    Tensor *__restrict add_to,
    const Tensor *__restrict add_from,
    size_t size_vec
) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }

    add_vector(
        add_to->ptr(),
        add_from->ptr(),
        add_from->dtype,
        add_to->dtype,
        size_vec
    );
}

void add_vector(
    Tensor *__restrict add_to, const void *__restrict add_from,
    DType::Type add_from_type, size_t size_vec
) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }

    add_vector(
        add_to->ptr(),
        add_from,
        add_from_type,
        add_to->dtype,
        size_vec
    );
}

void add_vector(
    void *__restrict add_to, const void *__restrict add_from,
    DType::Type add_from_type, DType::Type add_to_type, size_t size_vec
) {
    if (size_vec == 0) return;

    if (add_to_type == DType::FP32 && add_from_type == DType::FP32) {
        float *__restrict to = (float *)add_to;
        const float *__restrict from = (const float *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += from[i];
        }
    } else if (add_to_type == DType::FP32 && add_from_type == DType::FP16) {
        float *__restrict to = (float *)add_to;
        const half_cpu *__restrict from = (const half_cpu *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += (float)from[i];
        }
    } else if (add_to_type == DType::FP16 && add_from_type == DType::FP16) {
        half_cpu *__restrict to = (half_cpu *)add_to;
        const half_cpu *__restrict from = (const half_cpu *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + (float)from[i];
            to[i] = (half_cpu)acc;
        }
    } else if (add_to_type == DType::FP16 && add_from_type == DType::FP32) {
        half_cpu *__restrict to = (half_cpu *)add_to;
        const float *__restrict from = (const float *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + from[i];
            to[i] = (half_cpu)acc;
        }
    }
}

void swiglu(
    Tensor *__restrict gate,  // [d]
    const Tensor *__restrict up,    // [d]
    size_t size_vec
) {
    float *__restrict gate_buf = (float *)gate->ptr();
    const float *__restrict up_buf = (const float *)up->ptr();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size_vec; ++i) {
        float x = gate_buf[i];
        float silu = x / (1.0f + expf(-x));  // SiLU(x) = x * sigmoid(x)
        gate_buf[i] = silu * up_buf[i];               // SwiGLU = SiLU(gate) * up
    }
}

void softmax(float *__restrict x, size_t n) {
    if (n == 0) return;

    // -------------------------
    // 1. max reduction
    // -------------------------
    float max_val = x[0];

    #pragma omp simd reduction(max:max_val)
    for (size_t i = 0; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // -------------------------
    // 2. exp + sum
    // -------------------------
    float sum = 0.0f;

    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // -------------------------
    // 3. normalize
    // -------------------------
    float inv_sum = 1.0f / sum;

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void softmax_with_max(float *__restrict x, float max_val, size_t n) {
    if (n == 0) return;

    // -------------------------
    // 2. exp + sum
    // -------------------------
    float sum = 0.0f;

    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // -------------------------
    // 3. normalize
    // -------------------------
    float inv_sum = 1.0f / sum;

    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void attn_scores_all_heads_prefill(
    const char *__restrict key_cache,
    const float *__restrict key_cache_scale,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos, size_t group_size,
    int prefill_size, DType::Type cache_dtype
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    const size_t att_stride = att->shape[2];

    // =========================================================
    // ======================== FP16 PATH ======================
    // =========================================================
    if (cache_dtype == DType::FP16) {
        const uint16_t *key_cache_fp16 = (const uint16_t *)(key_cache);

        for (size_t b = 0; b < prefill_size; ++b) {

            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

                const uint16_t *__restrict k_ptr =
                    key_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

                const float *__restrict q_group_base =
                    (const float *)q->ptr({b, h_base});

                float *__restrict att_group_base =
                    (float *)att->ptr({b, h_base});

                gemm_text_qk_att(
                    q_group_base, k_ptr, nullptr, att_group_base,
                    inv_sqrt_d, nullptr, kv_mul, att_stride,
                    head_dim, pos + b + 1, group_size, DType::FP32,
                    DType::FP16, DType::NONETYPE, DType::FP32
                );

                for (int m = 0; m < kv_mul; ++m)
                    softmax(att_group_base + m * att_stride,
                            (size_t)(pos + b + 1));
            }
        }
    }

    // =========================================================
    // ======================== FP32 PATH ======================
    // =========================================================
    else if (cache_dtype == DType::FP32) {
        const float *key_cache_fp32 = (const float *)(key_cache);

        // Outer loop: iterate through groups of Q heads that share one K head
        for (size_t b = 0; b < prefill_size; ++b) {
            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
                // K pointer remains the same for all kv_mul Q-heads
                const float *__restrict k_ptr = key_cache_fp32 + 1ll * (h_base / kv_mul) * sh_offset;

                // Base pointers for the first Q head and first Attention head in this group
                const float *__restrict q_group_base = (const float *)q->ptr({b, h_base});
                float       *__restrict att_group_base = (float *)att->ptr({b, h_base});

                gemm_text_qk_att(
                    q_group_base, k_ptr, nullptr, att_group_base,
                    inv_sqrt_d, nullptr, kv_mul, att_stride,
                    head_dim, pos + b + 1, group_size, DType::FP32,
                    DType::FP32, DType::NONETYPE, DType::FP32
                );

                // Final Softmax for each head in the group
                for (int m = 0; m < kv_mul; ++m) {
                    softmax(att_group_base + m * att_stride, (size_t)(pos + b + 1));
                }
            } 
        }
    } 
    // =========================================================
    // ======================== INT8 PATH ======================
    // =========================================================
    else {
        const int8_t *key_cache_int8 = (const int8_t *)(key_cache);

        // Outer loop: iterate through groups of Q heads that share one K head
        for (size_t b = 0; b < prefill_size; ++b) {
            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
                // K pointer remains the same for all kv_mul Q-heads
                size_t key_cache_offset = 1ll * (h_base / kv_mul) * sh_offset;
                const int8_t *__restrict k_ptr = key_cache_int8 + key_cache_offset;
                const float *__restrict k_ptr_s = key_cache_scale + key_cache_offset / group_size;


                // Base pointers for the first Q head and first Attention head in this group
                const float *__restrict q_group_base = (const float *)q->ptr({b, h_base});
                float       *__restrict att_group_base = (float *)att->ptr({b, h_base});

                gemm_text_qk_att(
                    q_group_base, k_ptr, k_ptr_s, att_group_base,
                    inv_sqrt_d, nullptr, kv_mul, att_stride,
                    head_dim, pos + b + 1, group_size, DType::FP32,
                    DType::INT8, DType::FP32, DType::FP32
                );

                // Final Softmax for each head in the group
                for (int m = 0; m < kv_mul; ++m) {
                    softmax(att_group_base + m * att_stride, (size_t)(pos + b + 1));
                }
            } 
        }
    }
}

void attn_scores_all_heads_decode(
    const char *__restrict key_cache,
    const float *__restrict key_cache_scale,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos,
    size_t group_size, DType::Type cache_dtype
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    const size_t att_stride = att->shape[2];

    // =========================================================
    // ======================= FP16 PATH =======================
    // =========================================================
    if (cache_dtype == DType::FP16) {

        const uint16_t *key_cache_fp16 = (const uint16_t *)(key_cache);

        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

            const uint16_t *__restrict k_ptr =
                key_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

            const float *__restrict q_group_base =
                (const float *)q->ptr({0, h_base});

            float *__restrict att_group_base =
                (float *)att->ptr({0, h_base});

            gemm_text_qk_att(
                q_group_base, k_ptr, nullptr, att_group_base,
                inv_sqrt_d, nullptr, kv_mul, att_stride, head_dim,
                pos + 1, group_size, DType::FP32,
                DType::FP16, DType::NONETYPE, DType::FP32
            );

            for (int m = 0; m < kv_mul; ++m)
                softmax(att_group_base + m * att_stride, (size_t)(pos + 1));
        }
    }

    // =========================================================
    // ======================= FP32 PATH =======================
    // =========================================================
    else if (cache_dtype == DType::FP32) {

        // Your original FP32 implementation unchanged
        // (keep exactly what you already wrote)
        // Only difference:
        const float *key_cache_fp32 = (const float *)(key_cache);
        
        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

            const float *__restrict k_ptr =
                key_cache_fp32 + 1ll * (h_base / kv_mul) * sh_offset;

            const float *__restrict q_group_base =
                (const float *)q->ptr({0, h_base});

            float *__restrict att_group_base =
                (float *)att->ptr({0, h_base});

            gemm_text_qk_att(
                q_group_base, k_ptr, nullptr, att_group_base,
                inv_sqrt_d, nullptr, kv_mul, att_stride, head_dim,
                pos + 1, group_size, DType::FP32,
                DType::FP32, DType::NONETYPE, DType::FP32
            );

            // Final Softmax for each head in the group
            for (int m = 0; m < kv_mul; ++m) {
                softmax(att_group_base + m * att_stride, (size_t)(pos + 1));
            }
        }
    }
    // =========================================================
    // ======================== INT8 PATH ======================
    // =========================================================
    else {
        const int8_t *key_cache_int8 = (const int8_t *)(key_cache);

        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
            // K pointer remains the same for all kv_mul Q-heads
            size_t key_cache_offset = 1ll * (h_base / kv_mul) * sh_offset;
            const int8_t *__restrict k_ptr = key_cache_int8 + key_cache_offset;
            const float *__restrict k_ptr_s = key_cache_scale + key_cache_offset / group_size;


            // Base pointers for the first Q head and first Attention head in this group
            const float *__restrict q_group_base =
                (const float *)q->ptr({0, h_base});
            float *__restrict att_group_base =
                (float *)att->ptr({0, h_base});

            gemm_text_qk_att(
                q_group_base, k_ptr, k_ptr_s, att_group_base,
                inv_sqrt_d, nullptr, kv_mul, att_stride,
                head_dim, pos + 1, group_size, DType::FP32,
                DType::INT8, DType::FP32, DType::FP32
            );

            // Final Softmax for each head in the group
            for (int m = 0; m < kv_mul; ++m) {
                softmax(att_group_base + m * att_stride, (size_t)(pos + 1));
            }
        }
    }
}

void attn_weighted_sum_all_heads(
    const char *__restrict value_cache,
    const float *__restrict value_cache_scale,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos, int prefill_size,
    const size_t group_size, DType::Type cache_dtype
) {
    const size_t seq_len = att->shape[att->ndim - 1];

    // =========================================================
    // ======================== FP16 PATH ======================
    // =========================================================
    if (cache_dtype == DType::FP16) {
        const uint16_t *value_cache_fp16 = (const uint16_t *)(value_cache);

        for (size_t b = 0; b < prefill_size; ++b) {

            float *__restrict tb_base =
                (float *)tb->ptr({b});

            const float *__restrict att_base =
                (const float *)att->ptr({b});

            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

                float *__restrict tb_head =
                    tb_base + 1ll * h_base * head_dim;

                const float *__restrict att_head =
                    att_base + 1ll * h_base * seq_len;

                const uint16_t *__restrict v_head_base =
                    value_cache_fp16 + 1ll * (h_base / kv_mul) * sh_offset;

                gemm_text_kv_att(
                    att_head, v_head_base, nullptr, tb_head,
                    kv_mul, head_dim, seq_len, pos + b + 1,
                    group_size, DType::FP32, DType::FP16,
                    DType::NONETYPE, DType::FP32
                );
            }
        }
    }

    // =========================================================
    // ======================== FP32 PATH ======================
    // =========================================================
    else if (cache_dtype == DType::FP32) {
        const float *value_cache_fp32 = (const float *)(value_cache);

        for (size_t b = 0; b < prefill_size; ++b) {

            float *__restrict tb_base =
                (float *)tb->ptr({b});

            const float *__restrict att_base =
                (const float *)att->ptr({b});

            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

                float *__restrict tb_head =
                    tb_base + 1ll * h_base * head_dim;

                const float *__restrict att_head =
                    att_base + 1ll * h_base * seq_len;

                const float *__restrict v_head_base =
                    value_cache_fp32 + 1ll * (h_base / kv_mul) * sh_offset;
                
                gemm_text_kv_att(
                    att_head, v_head_base, nullptr, tb_head,
                    kv_mul, head_dim, seq_len, pos + b + 1,
                    group_size, DType::FP32, DType::FP32,
                    DType::NONETYPE, DType::FP32
                );            
            }
        }
    }
    
    // =========================================================
    // ======================== INT8 PATH ======================
    // =========================================================
    else {
        const int8_t *value_cache_int8 = (const int8_t *)(value_cache);

        for (size_t b = 0; b < prefill_size; ++b) {

            float *__restrict tb_base =
                (float *)tb->ptr({b});

            const float *__restrict att_base =
                (const float *)att->ptr({b});

            for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {

                float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;
                const float *__restrict att_head = att_base + 1ll * h_base * seq_len;

                const size_t v_offset = 1ll * (h_base / kv_mul) * sh_offset;
                const int8_t *__restrict v_head_base = value_cache_int8 + v_offset;
                const float *__restrict v_head_base_s = value_cache_scale + (v_offset / group_size);
                
                gemm_text_kv_att(
                    att_head, v_head_base, v_head_base_s,
                    tb_head, kv_mul, head_dim, seq_len,
                    pos + b + 1, group_size, DType::FP32,
                    DType::INT8, DType::FP32, DType::FP32
                );            
            }
        }
    }
}

void apply_rotary(
    Tensor *__restrict x,               /* [batch_size, n_heads, head_dim] */
    const Tensor *__restrict cos_table, /* [seq_len, head_dim/2] */
    const Tensor *__restrict sin_table, /* [seq_len, head_dim/2] */
    int batch_size, int n_heads, int head_dim, int pos
) {
    const int half = head_dim >> 1;

    const size_t stride_x = n_heads * head_dim;

    const float *__restrict cos_row_base = (const float *)cos_table->ptr({0, (size_t)pos});
    const float *__restrict sin_row_base = (const float *)sin_table->ptr({0, (size_t)pos});
    float *__restrict x_buf = (float *)x->ptr();

    constexpr int VEC = 8;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {

            float *__restrict x_base = x_buf + (b * stride_x) + (h * head_dim);
            float *__restrict x1p = x_base;
            float *__restrict x2p = x_base + half;

            const float *__restrict cos_row = cos_row_base + b * half;
            const float *__restrict sin_row = sin_row_base + b * half;

            int i = 0;

            // Same structure: process 8 elements at a time
            for (; i <= half - VEC; i += VEC) {
                for (int k = 0; k < VEC; k++) {
                    float x1 = x1p[i + k];
                    float x2 = x2p[i + k];
                    float c  = cos_row[i + k];
                    float s  = sin_row[i + k];

                    float y1 = x1 * c - x2 * s;
                    float y2 = x1 * s + x2 * c;

                    x1p[i + k] = y1;
                    x2p[i + k] = y2;
                }
            }
        }
    }
}

void apply_rotary_cache(
    const float *__restrict in_ptr,
    char *__restrict k_out,
    float *__restrict k_s_out,
    const Tensor *__restrict cos_table,
    const Tensor *__restrict sin_table,
    int batch_size, int n_heads, int head_dim,
    int pos, size_t sh_off, DType::Type cache_dtype,
    const size_t group_size
) {
    const int half = head_dim >> 1;
    const size_t in_stride = n_heads * head_dim;

    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();

    const float *__restrict cos_row_base = cos_buf + pos * half;
    const float *__restrict sin_row_base = sin_buf + pos * half;

    // ================= FP16 =================
    if (cache_dtype == DType::FP16) {
        half_cpu *k_out_fp16 = (half_cpu *)(k_out);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < n_heads; h++) {

                const float *x1p = in_ptr + b * in_stride + h * head_dim;
                const float *x2p = x1p + half;

                half_cpu *y1p = k_out_fp16 + b * head_dim + h * sh_off;
                half_cpu *y2p = y1p + half;

                const float *cos_row = cos_row_base + b * half;
                const float *sin_row = sin_row_base + b * half;

                for (int i = 0; i < half; i++) {
                    float x1 = x1p[i];
                    float x2 = x2p[i];
                    float c = cos_row[i];
                    float s = sin_row[i];

                    float y1 = x1 * c - x2 * s;
                    float y2 = x1 * s + x2 * c;

                    y1p[i] = (half_cpu)y1;
                    y2p[i] = (half_cpu)y2;
                }
            }
        }
    }

    // ================= INT8 =================
    else if (cache_dtype == DType::INT8) {
        int8_t *k_out_i8 = (int8_t *)(k_out);

        const int groups_per_half = half / group_size;
        assert(group_size % 32 == 0);
        assert(half % group_size == 0);

        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < n_heads; h++) {

                const float *x1p = in_ptr + b * in_stride + h * head_dim;
                const float *x2p = x1p + half;

                const size_t cache_offset = b * head_dim + h * sh_off;

                int8_t *y1p = k_out_i8 + cache_offset;
                int8_t *y2p = y1p + half;

                const float *cos_row = cos_row_base + b * half;
                const float *sin_row = sin_row_base + b * half;

                float *y1s = k_s_out + cache_offset / group_size;
                float *y2s = y1s + groups_per_half;

                std::vector<float> y_tmp(head_dim);

                // compute rotary
                for (int i = 0; i < half; i++) {
                    float x1 = x1p[i];
                    float x2 = x2p[i];
                    float c = cos_row[i];
                    float s = sin_row[i];

                    y_tmp[i]        = x1 * c - x2 * s;
                    y_tmp[half + i] = x1 * s + x2 * c;
                }

                // quantize per group
                for (int ii = 0; ii < head_dim; ii += group_size) {

                    float max_val = 0.0f;
                    for (int i = ii; i < ii + group_size; i++) {
                        float v = fabsf(y_tmp[i]);
                        if (v > max_val) max_val = v;
                    }

                    float scale = max_val / 127.0f;
                    float invS  = (max_val > 0) ? (127.0f / max_val) : 0.0f;

                    y1s[ii / group_size] = scale;

                    for (int i = ii; i < ii + group_size; i++) {
                        float v = y_tmp[i] * invS;
                        int q = (int)roundf(v);

                        if (q > 127) q = 127;
                        if (q < -127) q = -127;

                        y1p[i] = (int8_t)q;
                    }
                }
            }
        }
    }

    // ================= FP32 =================
    else {
        float *k_out_fp32 = (float *)(k_out);

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < n_heads; h++) {

                const float *x1p = in_ptr + b * in_stride + h * head_dim;
                const float *x2p = x1p + half;

                float *y1p = k_out_fp32 + b * head_dim + h * sh_off;
                float *y2p = y1p + half;

                const float *cos_row = cos_row_base + b * half;
                const float *sin_row = sin_row_base + b * half;

                for (int i = 0; i < half; i++) {
                    float x1 = x1p[i];
                    float x2 = x2p[i];
                    float c = cos_row[i];
                    float s = sin_row[i];

                    y1p[i] = x1 * c - x2 * s;
                    y2p[i] = x1 * s + x2 * c;
                }
            }
        }
    }
}

void copy_to_v_cache(
    const Tensor *v, char *v_cache_l, float *v_cache_s,
    const DType::Type v_cache_dtype, const size_t prefill_size,
    const size_t head_dim, const size_t num_kv_heads,
    const size_t kv_pos_off, const size_t kv_all_off,
    const size_t kv_pos_scale_off,
    const size_t cache_group_size, bool warm_up
) {
    const size_t v_stride = num_kv_heads * head_dim;

    if (v->dtype == DType::FP32) {
        float *v_ptr = (float *)v->ptr();

        // -------- FP32 --------
        if (v_cache_dtype == DType::FP32) {
            float *dst_base = (float *)(v_cache_l) + kv_pos_off;

            for (size_t b = 0; b < prefill_size; ++b) {
                const float *src = v_ptr + b * v_stride;
                float *dst = dst_base + b * head_dim;

                for (size_t h = 0; h < num_kv_heads; h++) {
                    memcpy(dst + h * kv_all_off,
                           src + h * head_dim,
                           head_dim * sizeof(float));
                }
            }
        }

        // -------- FP16 --------
        else if (v_cache_dtype == DType::FP16) {
            uint16_t *dst_base = (uint16_t *)(v_cache_l) + kv_pos_off;

            #pragma omp parallel for collapse(2)
            for (size_t b = 0; b < prefill_size; ++b) {
                for (size_t h = 0; h < num_kv_heads; h++) {

                    const float *src = v_ptr + b * v_stride + h * head_dim;
                    uint16_t *dst = dst_base + b * head_dim + h * kv_all_off;

                    for (size_t i = 0; i < head_dim; i++) {
                        dst[i] = (half_cpu)(src[i]);
                    }
                }
            }
        }

        // -------- INT8 --------
        else {
            int8_t *dst_base = (int8_t *)(v_cache_l) + kv_pos_off;
            float  *scale_base = v_cache_s + kv_pos_scale_off;

            for (size_t b = 0; b < prefill_size; ++b) {
                const float *src_base = v_ptr + b * v_stride;

                int8_t *dst_b = dst_base + b * head_dim;
                float  *scale_b = scale_base + (b * head_dim) / cache_group_size;

                for (size_t h = 0; h < num_kv_heads; h++) {

                    const float *src = src_base + h * head_dim;
                    int8_t *dst = dst_b + h * kv_all_off;
                    float *scale_dst = scale_b + h * kv_all_off / cache_group_size;

                    for (size_t g = 0; g < head_dim; g += cache_group_size) {

                        float max_val = 0.0f;
                        for (size_t i = g; i < g + cache_group_size; i++) {
                            float v = fabsf(src[i]);
                            if (v > max_val) max_val = v;
                        }

                        float scale = max_val / 127.0f;
                        float invS  = (max_val > 0) ? (127.0f / max_val) : 0.0f;

                        scale_dst[g / cache_group_size] = scale;

                        for (size_t i = g; i < g + cache_group_size; i++) {
                            float v = src[i] * invS;
                            int q = (int)roundf(v);

                            if (q > 127) q = 127;
                            if (q < -127) q = -127;

                            dst[i] = (int8_t)q;
                        }
                    }
                }
            }
        }
    }

    // -------- input FP16 --------
    else {
        half_cpu *v_ptr = (half_cpu *)v->ptr();

        if (v_cache_dtype == DType::FP16) {
            half_cpu *dst = (half_cpu *)(v_cache_l) + kv_pos_off;

            #pragma omp parallel for collapse(2)
            for (size_t b = 0; b < prefill_size; ++b) {
                for (size_t h = 0; h < num_kv_heads; ++h) {

                    memcpy(
                        dst + b * head_dim + h * kv_all_off,
                        v_ptr + b * v_stride + h * head_dim,
                        head_dim * sizeof(half_cpu)
                    );
                }
            }
        }
    }
}

size_t greedy_decode(float* logits, int vocab_size) {
    float max_val = -FLT_MAX;
    size_t max_idx = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

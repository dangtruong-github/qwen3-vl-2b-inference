#include "../include/text_layer.hpp"

void embedding_lookup(
    const Tensor *__restrict embedding /*[vocab, hidden]*/, 
    Tensor *__restrict out /*[hidden]*/,
    size_t token_id, size_t hidden_size
) {
    PtrPair emb_ptr = embedding->ptr_all({token_id});

    if (embedding->dtype == DType::FP32) {
        memcpy(out->ptr(), emb_ptr.buf, hidden_size * sizeof(float));
    } else if (embedding->dtype == DType::FP16) {
        // Get typed pointers
        const half_cpu *__restrict src = static_cast<const half_cpu*>(emb_ptr.buf);
        float *__restrict dst = static_cast<float*>(out->ptr());

        // Loop-based conversion (memcpy cannot be used here)
        #pragma omp simd
        for (size_t i = 0; i < hidden_size; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        const int8_t *__restrict src_q = static_cast<const int8_t*>(emb_ptr.buf);
        const float *__restrict scales = static_cast<const float*>(emb_ptr.scale);
        float *__restrict dst = static_cast<float*>(out->ptr());

        if (embedding->group_quantized) {
            size_t group_size = embedding->group_size; 
            size_t groups = hidden_size / group_size;

            #pragma omp parallel for
            for (size_t g = 0; g < groups; ++g) {
                float scale = scales[g];
                size_t base = g * group_size;

                for (size_t i = 0; i < group_size; ++i) {
                    size_t idx = base + i;
                    dst[idx] = (float)src_q[idx] * scale;
                }
            }
        } else {
            float scale_token = scales[0];

            #pragma omp simd
            for (size_t i = 0; i < hidden_size; ++i) {
                dst[i] = (float)src_q[i] * scale_token;
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
    Tensor *__restrict x_tensor /*[batches, hidden]*/,
    const Tensor *__restrict scale /*[hidden]*/,
    float eps, size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    const float inv_hs = 1.0f / (float)hidden_size;

    PtrPair scale_ptr = scale->ptr_all({layer_offset});

    float *x = (float *)(x_tensor->ptr());

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

            #pragma omp parallel for
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
        hid_states->ptr(), emb_ptr.buf, emb_ptr.scale, nullptr, nullptr,
        logits->ptr(), 1, vocab_size, hidden_size, true, hid_states->dtype, embedding->dtype, embedding->scale_dtype, logits->dtype,
        embedding->group_quantized, embedding->group_size
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

    const DType::Type to_dtype   = add_to->dtype;
    const DType::Type from_dtype = add_from->dtype;

    if (to_dtype == DType::FP32 && from_dtype == DType::FP32) {
        float *__restrict to   = (float *)add_to->ptr();
        const float *__restrict from = (const float *)add_from->ptr();

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += from[i];
        }
        return;
    } else if (to_dtype == DType::FP32 && from_dtype == DType::FP16) {
        float *__restrict to = (float *)add_to->ptr();
        const half_cpu *__restrict from = (const half_cpu *)add_from->ptr();

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += (float)from[i];
        }
        return;
    } else if (to_dtype == DType::FP16 && from_dtype == DType::FP16) {
        half_cpu *__restrict to = (half_cpu *)add_to->ptr();
        const half_cpu *__restrict from = (const half_cpu *)add_from->ptr();

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + (float)from[i];
            to[i] = (half_cpu)acc;
        }
        return;
    } else if (to_dtype == DType::FP16 && from_dtype == DType::FP32) {
        half_cpu *__restrict to = (half_cpu *)add_to->ptr();
        const float *__restrict from = (const float *)add_from->ptr();

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + from[i];  // FP32 accumulate
            to[i] = (half_cpu)acc;               // cast back
        }
        return;
    }
}

void add_vector(
    Tensor *__restrict add_to,
    const void *__restrict add_from,
    DType::Type add_from_type, size_t size_vec
) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }
    const DType::Type to_dtype = add_to->dtype;

    if (to_dtype == DType::FP32 && add_from_type == DType::FP32) {
        float *__restrict to   = (float *)add_to->ptr();
        const float *__restrict from = (const float *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += from[i];
        }
        return;
    } else if (to_dtype == DType::FP32 && add_from_type == DType::FP16) {
        float *__restrict to = (float *)add_to->ptr();
        const half_cpu *__restrict from = (const half_cpu *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            to[i] += (float)from[i];
        }
        return;
    } else if (to_dtype == DType::FP16 && add_from_type == DType::FP16) {
        half_cpu *__restrict to = (half_cpu *)add_to->ptr();
        const half_cpu *__restrict from = (const half_cpu *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + (float)from[i];
            to[i] = (half_cpu)acc;
        }
        return;
    } else if (to_dtype == DType::FP16 && add_from_type == DType::FP32) {
        half_cpu *__restrict to = (half_cpu *)add_to->ptr();
        const float *__restrict from = (const float *)add_from;

        #pragma omp parallel for simd
        for (size_t i = 0; i < size_vec; ++i) {
            float acc = (float)to[i] + from[i];  // FP32 accumulate
            to[i] = (half_cpu)acc;               // cast back
        }
        return;
    }
}

void swiglu(
    Tensor *__restrict gate,  // [d]
    const Tensor *__restrict up,    // [d]
    size_t size_vec
) {
    float *__restrict gate_buf = (float *)gate->ptr();
    const float *__restrict up_buf = (const float *)up->ptr();

    #pragma omp simd
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

void attn_scores_all_heads(
    const float *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    for (size_t h = 0; h < attn_heads; h++) {
        const float *__restrict q_head  = (const float *)q->ptr({0, h});
        float       *__restrict att_head = (float *)att->ptr({0, h});

        const float *__restrict k_ptr = key_cache + (size_t)(h / kv_mul) * sh_offset;

        // Parallel over sequence positions
        #pragma omp parallel for
        for (int t = 0; t <= pos; ++t) {
            const float *k_head = k_ptr + (size_t)t * head_dim;

            float score = 0.0f;

            // dot product q_head · k_head
            #pragma omp simd reduction(+:score)
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }

            att_head[t] = score * inv_sqrt_d;
        }

        // Causal softmax over [0, pos]
        softmax(att_head, (size_t)pos + 1);
    }
}

void attn_weighted_sum_all_heads(
    const float *__restrict value_cache,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos
) {
    float *__restrict tb_base = (float *)tb->ptr();
    const float *__restrict att_base = (const float *)att->ptr();
    const size_t seq_len = att->shape[att->ndim - 1];

    #pragma omp parallel for
    for (int h = 0; h < attn_heads; h++) {
        float *__restrict tb_head = tb_base + (size_t)h * head_dim;
        const float *__restrict att_head = att_base + (size_t)h * seq_len;

        const int kv_head_idx = h / kv_mul;
        const float *__restrict v_head_base = value_cache + (size_t)kv_head_idx * sh_offset;

        // initialize output
        for (int i = 0; i < head_dim; i++) {
            tb_head[i] = 0.0f;
        }

        for (int t = 0; t <= pos; ++t) {
            float a = att_head[t];
            const float *v = v_head_base + (size_t)t * head_dim;

            for (int i = 0; i < head_dim; i++) {
                tb_head[i] += a * v[i];
            }
        }
    }
}

void apply_rotary(
    Tensor *__restrict x,
    const Tensor *__restrict cos_table,
    const Tensor *__restrict sin_table,
    int n_heads, int head_dim, int pos
) {
    const int half = head_dim >> 1;

    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();
    float *__restrict x_buf = (float *)x->ptr();

    const float *__restrict cos_row = cos_buf + (size_t)pos * half;
    const float *__restrict sin_row = sin_buf + (size_t)pos * half;

    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        float *__restrict x1p = x_buf + (size_t)h * head_dim;
        float *__restrict x2p = x1p + half;

        for (int i = 0; i < half; i++) {
            float c = cos_row[i];
            float s = sin_row[i];
            float x1 = x1p[i];
            float x2 = x2p[i];

            x1p[i] = x1 * c - x2 * s;
            x2p[i] = x1 * s + x2 * c;
        }
    }
}

void apply_rotary_cache(
    const Tensor *__restrict in,
    float *__restrict k_out,
    const Tensor *__restrict cos_table,
    const Tensor *__restrict sin_table,
    int n_heads, int head_dim, int pos, size_t sh_off
) {
    const int half = head_dim >> 1;

    const float *__restrict in_ptr  = (const float *)in->ptr();
    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();

    const float *__restrict cos_row = cos_buf + (size_t)pos * half;
    const float *__restrict sin_row = sin_buf + (size_t)pos * half;

    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        const float *__restrict x1p = in_ptr + (size_t)h * head_dim;
        const float *__restrict x2p = x1p + half;

        float *__restrict y1p = k_out + (size_t)h * sh_off;
        float *__restrict y2p = y1p + half;

        for (int i = 0; i < half; i++) {
            float c = cos_row[i];
            float s = sin_row[i];
            float x1 = x1p[i];
            float x2 = x2p[i];

            y1p[i] = x1 * c - x2 * s;
            y2p[i] = x1 * s + x2 * c;
        }
    }
}

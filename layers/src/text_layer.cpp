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
            float scale_token = scales[token_id];

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
            const float scale_base = scale_scales[layer_offset];

            #pragma omp parallel for
            for (size_t i = 0; i < batches; i++) {
                const float *x_ptr = x + i * hidden_size;
                float *out_ptr = out + i * hidden_size;
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
                    out_ptr[j] = x_ptr[j] * combined * static_cast<float>(s_q[j]);
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

#if defined(__AVX2__) && defined(__FMA__)
void softmax(float *__restrict x, size_t n) {
    if (n == 0) return;

    size_t i = 0;

    // -------------------------
    // 1. max reduction
    // -------------------------
    __m256 vmax = _mm256_set1_ps(-INFINITY);

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);

    float max_val = tmp[0];
    for (int k = 1; k < 8; k++) {
        max_val = fmaxf(max_val, tmp[k]);
    }

    for (; i < n; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    // ---------- ATTENTION SAFETY ----------
    // All values are -INF or NaN
    if (!isfinite(max_val)) {
        memset(x, 0, n * sizeof(float));
        return;
    }

    // -------------------------
    // 2. exp(x - max) + sum
    // -------------------------
    __m256 vsum = _mm256_setzero_ps();
    __m256 v_max = _mm256_set1_ps(max_val);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);

        // Mask out -INF lanes BEFORE subtraction
        __m256 mask = _mm256_cmp_ps(v, _mm256_set1_ps(-INFINITY), _CMP_GT_OQ);

        v = _mm256_sub_ps(v, v_max);
        v = exp256_ps(v);

        // Zero masked lanes
        v = _mm256_and_ps(v, mask);

        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    _mm256_store_ps(tmp, vsum);
    double sum = 0.0;
    for (int k = 0; k < 8; k++) {
        sum += tmp[k];
    }

    for (; i < n; i++) {
        float xi = x[i];
        if (xi == -INFINITY) {
            x[i] = 0.0f;
            continue;
        }

        float t = xi - max_val;
        __m256 v = exp256_ps(_mm256_set1_ps(t));
        _mm_store_ss(&x[i], _mm256_castps256_ps128(v));
        sum += x[i];
    }

    // ---------- ATTENTION SAFETY ----------
    if (!(sum > 0.0) || !isfinite(sum)) {
        memset(x, 0, n * sizeof(float));
        return;
    }

    // -------------------------
    // 3. normalize
    // -------------------------
    float inv_sum = (float)(1.0 / sum);
    __m256 v_inv_sum = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, v_inv_sum);
        _mm256_storeu_ps(x + i, v);
    }

    for (; i < n; i++) {
        x[i] *= inv_sum;
    }
}
#else
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
#endif

void attn_scores_all_heads(
    const float *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    for (size_t h = 0; h < attn_heads; h++) {
        const float *__restrict q_head = (const float *)q->ptr({0, h});
        float       *__restrict att_head = (float *)att->ptr({0, h});

        const float *__restrict k_ptr = key_cache + 1ll * (h / kv_mul) * sh_offset;
        
        // Main unrolled loop: 8 positions at a time
        #pragma omp parallel for
        for (int t = 0; t <= pos - 8; t += 8) {
            const float *k0_ptr = k_ptr + (t+0) * head_dim; 
            const float *k1_ptr = k_ptr + (t+1) * head_dim;
            const float *k2_ptr = k_ptr + (t+2) * head_dim;
            const float *k3_ptr = k_ptr + (t+3) * head_dim;
            const float *k4_ptr = k_ptr + (t+4) * head_dim; 
            const float *k5_ptr = k_ptr + (t+5) * head_dim;
            const float *k6_ptr = k_ptr + (t+6) * head_dim;
            const float *k7_ptr = k_ptr + (t+7) * head_dim;

            __m256 s0 = _mm256_setzero_ps();
            __m256 s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps();
            __m256 s3 = _mm256_setzero_ps();
            __m256 s4 = _mm256_setzero_ps();
            __m256 s5 = _mm256_setzero_ps();
            __m256 s6 = _mm256_setzero_ps();
            __m256 s7 = _mm256_setzero_ps();

            // head_dim % 8 == 0
            for (int i = 0; i < head_dim; i += 8) {
                __m256 q_vec = _mm256_loadu_ps(q_head + i);

                s0 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k0_ptr + i), s0);
                s1 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k1_ptr + i), s1);
                s2 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k2_ptr + i), s2);
                s3 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k3_ptr + i), s3);
                s4 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k4_ptr + i), s4);
                s5 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k5_ptr + i), s5);
                s6 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k6_ptr + i), s6);
                s7 = _mm256_fmadd_ps(q_vec, _mm256_loadu_ps(k7_ptr + i), s7);
            }

            att_head[t+0] = add_reduce_mm_256_layer(s0) * inv_sqrt_d;
            att_head[t+1] = add_reduce_mm_256_layer(s1) * inv_sqrt_d;
            att_head[t+2] = add_reduce_mm_256_layer(s2) * inv_sqrt_d;
            att_head[t+3] = add_reduce_mm_256_layer(s3) * inv_sqrt_d;
            att_head[t+4] = add_reduce_mm_256_layer(s4) * inv_sqrt_d;
            att_head[t+5] = add_reduce_mm_256_layer(s5) * inv_sqrt_d;
            att_head[t+6] = add_reduce_mm_256_layer(s6) * inv_sqrt_d;
            att_head[t+7] = add_reduce_mm_256_layer(s7) * inv_sqrt_d;
        }

        // Tail: 0–7 remaining positions
        for (int t = (pos / 8) * 8; t <= pos; ++t) {
            float score = 0.0f;
            const float *k_head = k_ptr + (size_t)t * head_dim;

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

// Fix attn_weighted_sum_all_heads function:
// This function was already correct. 'loff' passed from forward_text
// corresponds to the layer_cache_start.
void attn_weighted_sum_all_heads(
    const float *__restrict value_cache,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos
) {
    // tb shape: (B=1, attn_heads, head_dim)
    float *__restrict tb_base = (float *)tb->ptr();
    const float *__restrict att_base = (const float *)att->ptr();
    const size_t seq_len = att->shape[att->ndim - 1];

    #pragma omp parallel for
    for (int h = 0; h < attn_heads; h++) {
        float *__restrict tb_head = tb_base + 1ll * h * head_dim;
        const float *__restrict att_head = att_base + 1ll * h * seq_len;

        const int kv_head_idx = h / kv_mul;
        const float *__restrict v_head_base = value_cache + 1ll * kv_head_idx * sh_offset;

        for (int hd = 0; hd < head_dim; hd += 64) {
            // Load accumulators
            __m256 acc_0 = _mm256_setzero_ps();
            __m256 acc_1 = _mm256_setzero_ps();
            __m256 acc_2 = _mm256_setzero_ps();
            __m256 acc_3 = _mm256_setzero_ps();
            __m256 acc_4 = _mm256_setzero_ps();
            __m256 acc_5 = _mm256_setzero_ps();
            __m256 acc_6 = _mm256_setzero_ps();
            __m256 acc_7 = _mm256_setzero_ps();

            for (int t = 0; t <= pos; ++t) {
                __m256 a = _mm256_set1_ps(att_head[t]);
                const float *v = v_head_base + (size_t)t * head_dim + hd;

                acc_0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v), acc_0);
                acc_1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 8), acc_1);
                acc_2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 16), acc_2);
                acc_3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 24), acc_3);
                acc_4 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 32), acc_4);
                acc_5 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 40), acc_5);
                acc_6 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 48), acc_6);
                acc_7 = _mm256_fmadd_ps(a, _mm256_loadu_ps(v + 56), acc_7);
            }

            // Store back
            _mm256_storeu_ps(tb_head + hd, acc_0);
            _mm256_storeu_ps(tb_head + hd + 8, acc_1);
            _mm256_storeu_ps(tb_head + hd + 16, acc_2);
            _mm256_storeu_ps(tb_head + hd + 24, acc_3);
            _mm256_storeu_ps(tb_head + hd + 32, acc_4);
            _mm256_storeu_ps(tb_head + hd + 40, acc_5);
            _mm256_storeu_ps(tb_head + hd + 48, acc_6);
            _mm256_storeu_ps(tb_head + hd + 56, acc_7);
        }
    }
}

void apply_rotary(
    Tensor *__restrict x /*[n_heads*hd]*/,
    const Tensor *__restrict cos_table /*[seq_len*hd/2]*/,
    const Tensor *__restrict sin_table /*[seq_len*hd/2]*/,
    int n_heads, int head_dim, int pos
) {
    const int half = head_dim >> 1;

    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();
    float *__restrict x_buf = (float *)x->ptr();

    const float *__restrict cos_row = cos_buf + pos * half;
    const float *__restrict sin_row = sin_buf + pos * half;

    constexpr int VEC = 8; // AVX2: 8 floats

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
        float *__restrict x1p = x_buf + h * head_dim;
        float *__restrict x2p = x1p + half;

        int i = 0;

        // --- AVX2 main loop ---
        for (; i + VEC <= half; i += VEC) {
            __m256 x1 = _mm256_loadu_ps(x1p + i);
            __m256 x2 = _mm256_loadu_ps(x2p + i);
            __m256 c  = _mm256_loadu_ps(cos_row + i);
            __m256 s  = _mm256_loadu_ps(sin_row + i);

            // x1' = x1*c - x2*s
            __m256 y1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));

            // x2' = x1*s + x2*c
            __m256 y2 = _mm256_fmadd_ps(x1, s, _mm256_mul_ps(x2, c));

            _mm256_storeu_ps(x1p + i, y1);
            _mm256_storeu_ps(x2p + i, y2);
        }

        // --- scalar tail --- half % 8 == 0
        /*
            for (; i < half; i++) {
                float c = cos_row[i];
                float s = sin_row[i];
                float x1 = x1p[i];
                float x2 = x2p[i];
                x1p[i] = x1 * c - x2 * s;
                x2p[i] = x1 * s + x2 * c;
            }
        */
    }    
}

void apply_rotary_cache(
    const Tensor *__restrict in /*[n_heads*hd]*/,
    float *__restrict k_out      /*[n_heads*seq_len*hd]*/,
    const Tensor *__restrict cos_table /*[seq_len*hd/2]*/,
    const Tensor *__restrict sin_table /*[seq_len*hd/2]*/,
    int n_heads, int head_dim, int pos, size_t sh_off
) {
    const int half = head_dim >> 1;

    const float *__restrict in_ptr  = (const float *)in->ptr();
    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();

    const float *__restrict cos_row = cos_buf + pos * half;
    const float *__restrict sin_row = sin_buf + pos * half;

    constexpr int VEC = 8; // AVX2

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < n_heads; h++) {
        const float *__restrict x1p = in_ptr + h * head_dim;
        const float *__restrict x2p = x1p + half;

        float *__restrict y1p = k_out + h * sh_off;
        float *__restrict y2p = y1p + half;

        int i = 0;

        // ---- AVX2 main loop ----
        for (; i + VEC <= half; i += VEC) {
            __m256 x1 = _mm256_loadu_ps(x1p + i);
            __m256 x2 = _mm256_loadu_ps(x2p + i);
            __m256 c  = _mm256_loadu_ps(cos_row + i);
            __m256 s  = _mm256_loadu_ps(sin_row + i);

            // y1 = x1*c - x2*s
            __m256 y1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));

            // y2 = x1*s + x2*c
            __m256 y2 = _mm256_fmadd_ps(x1, s, _mm256_mul_ps(x2, c));

            _mm256_storeu_ps(y1p + i, y1);
            _mm256_storeu_ps(y2p + i, y2);
        }

        // ---- scalar tail ---- half % 8 == 0
        /*
            for (; i < half; i++) {
                float c = cos_row[i];
                float s = sin_row[i];
                float x1 = x1p[i];
                float x2 = x2p[i];
                y1p[i] = x1 * c - x2 * s;
                y2p[i] = x1 * s + x2 * c;
            }
        */
    }    
}

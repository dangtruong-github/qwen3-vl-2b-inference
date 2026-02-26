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
        hid_states->ptr(), emb_ptr.buf, emb_ptr.scale, nullptr,
        nullptr, nullptr, logits->ptr(), 1, vocab_size, hidden_size, true,
        hid_states->dtype, embedding->dtype, embedding->scale_dtype,
        logits->dtype, embedding->group_quantized, embedding->group_size
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

void attn_scores_all_heads_prefill(
    const float *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos, int prefill_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    // q_offset and att_offset should be the stride between consecutive heads
    // Usually q->shape[2] is head_dim, so we need the full stride to the next head.
    const size_t att_stride = att->shape[2]; // Stride for attention scores (usually max_seq_len)

    // Outer loop: iterate through groups of Q heads that share one K head
    for (size_t b = 0; b < prefill_size; ++b) {
        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
            // K pointer remains the same for all kv_mul Q-heads
            const float *__restrict k_ptr = key_cache + 1ll * (h_base / kv_mul) * sh_offset;

            // Base pointers for the first Q head and first Attention head in this group
            const float *__restrict q_group_base = (const float *)q->ptr({b, h_base});
            float       *__restrict att_group_base = (float *)att->ptr({b, h_base});

            size_t t_end = ((pos + b) + 1) & ~3; // Round down to multiple of 4

            // Parallelize the time (sequence) dimension
            #pragma omp parallel for
            for (size_t t = 0; t < t_end; t += 4) {
                const float *k0_ptr = k_ptr + (size_t)t * head_dim;
                const float *k1_ptr = k_ptr + (size_t)(t + 1) * head_dim;
                const float *k2_ptr = k_ptr + (size_t)(t + 2) * head_dim;
                const float *k3_ptr = k_ptr + (size_t)(t + 3) * head_dim;

                // Use fixed-size arrays for accumulators (stack allocated)
                __m256 s0[kv_mul], s1[kv_mul], s2[kv_mul], s3[kv_mul];

                for (int m = 0; m < kv_mul; ++m) {
                    s0[m] = _mm256_setzero_ps();
                    s1[m] = _mm256_setzero_ps();
                    s2[m] = _mm256_setzero_ps();
                    s3[m] = _mm256_setzero_ps();
                }

                for (int i = 0; i < head_dim; i += 8) {
                    __m256 k0v = _mm256_loadu_ps(k0_ptr + i);
                    __m256 k1v = _mm256_loadu_ps(k1_ptr + i);
                    __m256 k2v = _mm256_loadu_ps(k2_ptr + i);
                    __m256 k3v = _mm256_loadu_ps(k3_ptr + i);

                    for (int m = 0; m < kv_mul; ++m) {
                        __m256 q_vec = _mm256_loadu_ps(q_group_base + (m * head_dim) + i);
                        s0[m] = _mm256_fmadd_ps(q_vec, k0v, s0[m]);
                        s1[m] = _mm256_fmadd_ps(q_vec, k1v, s1[m]);
                        s2[m] = _mm256_fmadd_ps(q_vec, k2v, s2[m]);
                        s3[m] = _mm256_fmadd_ps(q_vec, k3v, s3[m]);
                    }
                }

                for (int m = 0; m < kv_mul; ++m) {
                    float *att_ptr = att_group_base + (m * att_stride);
                    att_ptr[t + 0] = add_reduce_mm_256_layer(s0[m]) * inv_sqrt_d;
                    att_ptr[t + 1] = add_reduce_mm_256_layer(s1[m]) * inv_sqrt_d;
                    att_ptr[t + 2] = add_reduce_mm_256_layer(s2[m]) * inv_sqrt_d;
                    att_ptr[t + 3] = add_reduce_mm_256_layer(s3[m]) * inv_sqrt_d;
                }
            }

            #pragma omp parallel for
            for (size_t t = t_end; t <= pos + b; ++t) {
                const float *k_head = k_ptr + (size_t)t * head_dim;

                __m256 s_acc[kv_mul];

                for (int m = 0; m < kv_mul; ++m) {
                    s_acc[m] = _mm256_setzero_ps();
                }

                for (int i = 0; i < head_dim; i += 8) {
                    __m256 k0v = _mm256_loadu_ps(k_head + i);

                    for (int m = 0; m < kv_mul; ++m) {
                        __m256 q_vec = _mm256_loadu_ps(q_group_base + (m * head_dim) + i);
                        s_acc[m] = _mm256_fmadd_ps(q_vec, k0v, s_acc[m]);
                    }
                }

                for (int m = 0; m < kv_mul; ++m) {
                    att_group_base[m * att_stride + t] = add_reduce_mm_256_layer(s_acc[m]) * inv_sqrt_d;
                }
            }

            // Final Softmax for each head in the group
            for (int m = 0; m < kv_mul; ++m) {
                softmax(att_group_base + m * att_stride, (size_t)(pos + b + 1));
            }
        } 
    }
}

void attn_scores_all_heads_decode(
    const float *__restrict key_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, size_t sh_offset, int pos
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    // q_offset and att_offset should be the stride between consecutive heads
    // Usually q->shape[2] is head_dim, so we need the full stride to the next head.
    const size_t att_stride = att->shape[2]; // Stride for attention scores (usually max_seq_len)

    // Outer loop: iterate through groups of Q heads that share one K head
    for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
        // K pointer remains the same for all kv_mul Q-heads
        const float *__restrict k_ptr = key_cache + 1ll * (h_base / kv_mul) * sh_offset;

        // Base pointers for the first Q head and first Attention head in this group
        const float *__restrict q_group_base = (const float *)q->ptr({0, h_base});
        float       *__restrict att_group_base = (float *)att->ptr({0, h_base});

        size_t t_end = (pos + 1) & ~3; // Round down to multiple of 4

        // Parallelize the time (sequence) dimension
        #pragma omp parallel for
        for (size_t t = 0; t < t_end; t += 4) {
            const float *k0_ptr = k_ptr + (size_t)t * head_dim;
            const float *k1_ptr = k_ptr + (size_t)(t + 1) * head_dim;
            const float *k2_ptr = k_ptr + (size_t)(t + 2) * head_dim;
            const float *k3_ptr = k_ptr + (size_t)(t + 3) * head_dim;

            // Use fixed-size arrays for accumulators (stack allocated)
            __m256 s0[kv_mul], s1[kv_mul], s2[kv_mul], s3[kv_mul];

            for (int m = 0; m < kv_mul; ++m) {
                s0[m] = _mm256_setzero_ps();
                s1[m] = _mm256_setzero_ps();
                s2[m] = _mm256_setzero_ps();
                s3[m] = _mm256_setzero_ps();
            }

            for (int i = 0; i < head_dim; i += 8) {
                __m256 k0v = _mm256_loadu_ps(k0_ptr + i);
                __m256 k1v = _mm256_loadu_ps(k1_ptr + i);
                __m256 k2v = _mm256_loadu_ps(k2_ptr + i);
                __m256 k3v = _mm256_loadu_ps(k3_ptr + i);

                for (int m = 0; m < kv_mul; ++m) {
                    __m256 q_vec = _mm256_loadu_ps(q_group_base + (m * head_dim) + i);
                    s0[m] = _mm256_fmadd_ps(q_vec, k0v, s0[m]);
                    s1[m] = _mm256_fmadd_ps(q_vec, k1v, s1[m]);
                    s2[m] = _mm256_fmadd_ps(q_vec, k2v, s2[m]);
                    s3[m] = _mm256_fmadd_ps(q_vec, k3v, s3[m]);
                }
            }

            for (int m = 0; m < kv_mul; ++m) {
                float *att_ptr = att_group_base + (m * att_stride);
                att_ptr[t + 0] = add_reduce_mm_256_layer(s0[m]) * inv_sqrt_d;
                att_ptr[t + 1] = add_reduce_mm_256_layer(s1[m]) * inv_sqrt_d;
                att_ptr[t + 2] = add_reduce_mm_256_layer(s2[m]) * inv_sqrt_d;
                att_ptr[t + 3] = add_reduce_mm_256_layer(s3[m]) * inv_sqrt_d;
            }
        }

        // Tail processing for remaining time steps
        #pragma omp parallel for
        for (size_t t = t_end; t <= pos; ++t) {
            const float *k_head = k_ptr + (size_t)t * head_dim;

            __m256 s_acc[kv_mul];

            for (int m = 0; m < kv_mul; ++m) {
                s_acc[m] = _mm256_setzero_ps();
            }

            for (int i = 0; i < head_dim; i += 8) {
                __m256 k0v = _mm256_loadu_ps(k_head + i);

                for (int m = 0; m < kv_mul; ++m) {
                    __m256 q_vec = _mm256_loadu_ps(q_group_base + (m * head_dim) + i);
                    s_acc[m] = _mm256_fmadd_ps(q_vec, k0v, s_acc[m]);
                }
            }

            for (int m = 0; m < kv_mul; ++m) {
                att_group_base[m * att_stride + t] = add_reduce_mm_256_layer(s_acc[m]) * inv_sqrt_d;
            }
        }

        // Final Softmax for each head in the group
        for (int m = 0; m < kv_mul; ++m) {
            softmax(att_group_base + m * att_stride, (size_t)(pos + 1));
        }
    } 
}

void attn_weighted_sum_all_heads(
    const float *__restrict value_cache,
    const Tensor *__restrict att, Tensor *__restrict tb,
    int attn_heads, int kv_mul, int head_dim, int kv_dim,
    size_t sh_offset, int pos, int prefill_size
) {
    const size_t seq_len = att->shape[att->ndim - 1];

    for (size_t b = 0; b < prefill_size; ++b) {
        float *__restrict tb_base = (float *)tb->ptr({b});
        const float *__restrict att_base = (const float *)att->ptr({b});

        #pragma omp parallel for
        for (size_t h_base = 0; h_base < attn_heads; h_base += kv_mul) {
            float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;
            const float *__restrict att_head = att_base + 1ll * h_base * seq_len;
            const float *__restrict v_head_base = value_cache + 1ll * (h_base / kv_mul) * sh_offset;

            for (int hd = 0; hd < head_dim; hd += 32) {
                // Load accumulators
                __m256 acc_0[kv_mul];
                __m256 acc_1[kv_mul];
                __m256 acc_2[kv_mul];
                __m256 acc_3[kv_mul];

                for (int m = 0; m < kv_mul; ++m) {
                    acc_0[m] = _mm256_setzero_ps();
                    acc_1[m] = _mm256_setzero_ps();
                    acc_2[m] = _mm256_setzero_ps();
                    acc_3[m] = _mm256_setzero_ps();
                }

                for (size_t t = 0; t <= (pos + b); ++t) {
                    const float *v = v_head_base + t * head_dim + hd;

                    __m256 v0 = _mm256_loadu_ps(v);
                    __m256 v1 = _mm256_loadu_ps(v + 8);
                    __m256 v2 = _mm256_loadu_ps(v + 16);
                    __m256 v3 = _mm256_loadu_ps(v + 24);

                    for (int m = 0; m < kv_mul; ++m) {
                        __m256 a = _mm256_broadcast_ss(att_head + seq_len * m + t);

                        acc_0[m] = _mm256_fmadd_ps(a, v0, acc_0[m]);
                        acc_1[m] = _mm256_fmadd_ps(a, v1, acc_1[m]);
                        acc_2[m] = _mm256_fmadd_ps(a, v2, acc_2[m]);
                        acc_3[m] = _mm256_fmadd_ps(a, v3, acc_3[m]);
                    }
                }

                // Store back
                for (int m = 0; m < kv_mul; ++m) {
                    _mm256_storeu_ps(tb_head + head_dim * m + hd, acc_0[m]);
                    _mm256_storeu_ps(tb_head + head_dim * m + hd + 8, acc_1[m]);
                    _mm256_storeu_ps(tb_head + head_dim * m + hd + 16, acc_2[m]);
                    _mm256_storeu_ps(tb_head + head_dim * m + hd + 24, acc_3[m]);
                }
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

    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();
    float *__restrict x_buf = (float *)x->ptr();

    // Tables are indexed by the shared position
    const float *__restrict cos_row_base = cos_buf + pos * half;
    const float *__restrict sin_row_base = sin_buf + pos * half;

    constexpr int VEC = 8; 

    // Parallelize across both batch and heads for maximum throughput
    #pragma omp parallel for schedule(static) collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            // Offset: (batch_index * total_elements_per_batch) + (head_index * elements_per_head)
            float *__restrict x_base = x_buf + (b * n_heads * head_dim) + (h * head_dim);
            float *__restrict x1p = x_base;
            float *__restrict x2p = x_base + half;

            const float *__restrict cos_row = cos_row_base + b * half;
            const float *__restrict sin_row = sin_row_base + b * half;

            int i = 0;
            // --- AVX2 main loop ---
            for (; i <= half - VEC; i += VEC) {
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
        }
    }    
}

void apply_rotary_cache(
    const Tensor *__restrict in /*[n_heads*hd]*/,
    float *__restrict k_out      /*[n_heads*seq_len*hd]*/,
    const Tensor *__restrict cos_table /*[seq_len*hd/2]*/,
    const Tensor *__restrict sin_table /*[seq_len*hd/2]*/,
    int batch_size, int n_heads, int head_dim, int pos, size_t sh_off
) {
    const int half = head_dim >> 1;

    const float *__restrict in_ptr  = (const float *)in->ptr();
    const float *__restrict cos_buf = (const float *)cos_table->ptr();
    const float *__restrict sin_buf = (const float *)sin_table->ptr();

    const float *__restrict cos_row_base = cos_buf + pos * half;
    const float *__restrict sin_row_base = sin_buf + pos * half;

    constexpr int VEC = 8; // AVX2

    #pragma omp parallel for schedule(static) collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            const float *__restrict x1p = in_ptr + (b * n_heads * head_dim) + (h * head_dim);
            const float *__restrict x2p = x1p + half;

            float *__restrict y1p = k_out + b * head_dim + h * sh_off;
            float *__restrict y2p = y1p + half;

            const float *__restrict cos_row = cos_row_base + b * half;
            const float *__restrict sin_row = sin_row_base + b * half;

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
        }    
    }
}

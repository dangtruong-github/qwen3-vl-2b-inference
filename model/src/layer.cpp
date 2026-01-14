#include "../include/layer.hpp"

void embedding_lookup(
    const Tensor *embedding /*[vocab, hidden]*/, size_t token_id,
    Tensor *out /*[hidden]*/, size_t hidden_size
) {
    if (embedding->dtype == DType::FP32) {
        memcpy(
            out->ptr(), embedding->ptr({token_id}),
            hidden_size * sizeof(float)
        );
    } else if (embedding->dtype == DType::FP16) {
        // Get typed pointers
        const half_cpu* src = static_cast<const half_cpu*>(embedding->ptr({token_id}));
        float* dst = static_cast<float*>(out->ptr());

        // Loop-based conversion (memcpy cannot be used here)
        #pragma omp simd
        for (size_t i = 0; i < hidden_size; ++i) {
            dst[i] = static_cast<float>(src[i]);
        }
    } else {
        const int8_t *src_q = static_cast<const int8_t*>(embedding->ptr({token_id}));
        const float *scales = static_cast<const float*>(embedding->ptr({token_id}, true));
        float *dst = static_cast<float*>(out->ptr());

        size_t group_size = embedding->group_size; 

        #pragma omp parallel for
        for (size_t i = 0; i < hidden_size; ++i) {
            // Find which scale group this index belongs to
            size_t group_idx = i / group_size;
            float scale = scales[group_idx];
            
            // Dequantize: float = int8 * scale
            dst[i] = static_cast<float>(src_q[i]) * scale;
        }
    }
}

void rms_norm(
    const float *x /*[hidden]*/, const Tensor *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps,
    size_t batches, size_t layer_offset
) {
    const size_t hidden_size = scale->shape[scale->ndim - 1];
    
    if (scale->dtype == DType::FP32) {
        const float *scale_buf = (const float *)scale->ptr({layer_offset});

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
        const half_cpu *scale_buf = static_cast<const half_cpu*>(scale->ptr({layer_offset}));

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
        const int8_t *scale_q = static_cast<const int8_t*>(scale->ptr({layer_offset}));
        const float *scale_scales = static_cast<const float*>(scale->ptr({layer_offset}, true));
        const size_t group_size = scale->group_size;

        for (size_t i = 0; i < batches; i++) {
            // 1. Calculate sum of squares (standard RMS logic)
            double ss = 0.0;
            #pragma omp simd reduction(+:ss)
            for (size_t j = 0; j < hidden_size; j++) {
                ss += (double)x[j] * x[j];
            }
            
            float inv_rms = static_cast<float>(1.0 / sqrt(ss / hidden_size + eps));

            // 2. Normalize and apply dequantized weights
            // out[j] = (x[j] * inv_rms) * (scale_q[j] * scale_scales[j / group_size])
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; j++) {
                float weight = static_cast<float>(scale_q[j]) * scale_scales[j / group_size];
                out[j] = weight * (inv_rms * x[j]);
            }
            
            x += hidden_size;
            out += hidden_size;
        }
    }
}

void classifier_gemm(
    const Tensor *embedding /*[vocab, hidden]*/,
    const Tensor *hid_states /*[hidden]*/, Tensor *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
) {
    linear(
        (const float *)hid_states->ptr(), embedding->ptr(),
        embedding->group_size > 0 ? embedding->ptr({}, true) : nullptr,
        nullptr, nullptr, (float *)logits->ptr(),
        1, vocab_size, hidden_size, true, embedding->dtype,
        embedding->scale_dtype, embedding->group_size
    );
}

void add_vector(Tensor *add_to, const Tensor *add_from, size_t size_vec) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }
    float *add_to_buf = (float *)add_to->ptr();
    const float *add_from_buf = (const float *)add_from->ptr();

    #pragma omp parallel for simd
    for (size_t i = 0; i < size_vec; i++) {
        add_to_buf[i] += add_from_buf[i];
    }
}

void add_vector(Tensor *add_to, const float *add_from, size_t size_vec) {
    if (size_vec == 0) {
        size_vec = add_to->num_elem();
    }
    float *add_to_buf = (float *)add_to->ptr();
    
    #pragma omp parallel for simd
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

    #pragma omp simd
    for (size_t i = 0; i < size_vec; i++) {
        float x = gate_buf[i];
        float silu = x / (1.0f + expf(-x));  // SiLU(x) = x * sigmoid(x)
        gate_buf[i] = silu * up_buf[i];               // SwiGLU = SiLU(gate) * up
    }
}

#if defined(__AVX2__) && defined(__FMA__)
static inline __m256 exp256_ps(__m256 x) {
    // Clamp to avoid overflow
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    // exp(x) = exp(g + n*ln2)
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f);

    __m256 fx = _mm256_fmadd_ps(x, inv_ln2, _mm256_set1_ps(0.5f));

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);

    __m256 g = _mm256_fnmadd_ps(fx, ln2, x);

    // Polynomial approximation
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.0f));

    // Build 2^n
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
}

void softmax(float *x, size_t n) {
    size_t i = 0;

    // -------------------------
    // 1. max reduction
    // -------------------------
    __m256 vmax = _mm256_set1_ps(-INFINITY);

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    float max_val = -INFINITY;
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, vmax);
    for (int k = 0; k < 8; k++) max_val = fmaxf(max_val, tmp[k]);

    for (; i < n; i++) max_val = fmaxf(max_val, x[i]);

    // -------------------------
    // 2. exp + sum
    // -------------------------
    __m256 vsum = _mm256_setzero_ps();
    __m256 v_max = _mm256_set1_ps(max_val);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, v_max);
        v = exp256_ps(v);
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }

    float sum = 0.f;
    _mm256_store_ps(tmp, vsum);
    for (int k = 0; k < 8; k++) sum += tmp[k];

    for (; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // -------------------------
    // 3. normalize
    // -------------------------
    __m256 v_inv_sum = _mm256_set1_ps(1.0f / sum);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_mul_ps(v, v_inv_sum);
        _mm256_storeu_ps(x + i, v);
    }

    for (; i < n; i++) x[i] /= sum;
}
#else
void softmax(float *x, size_t n) {
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

// **FIXED** attn_scores_all_heads function:
// Changed signature to remove loff_one and calculate offset internally.
void attn_scores_all_heads(
    const Tensor *key_cache, const Tensor *q, Tensor *att,
    size_t layer_offset, size_t attn_heads, int kv_mul, int head_dim,
    int kv_dim, int seq_len, int pos
) {
    // Calculate the start of the current layer in the cache
    const float *key_cache_ptr = (const float *)key_cache->ptr({0, layer_offset});
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    for (size_t h = 0; h < attn_heads; h++) {
        const float *q_head = (const float *)q->ptr({0, h});
        float *att_head = (float *)att->ptr({0, h});
        int kv_head_idx = h / kv_mul;

        // Compute attention scores
        for (int t = 0; t <= pos; t++) {
            // Get the key for this timestep
            const float *k_head = key_cache_ptr + 1ll * t * kv_dim + 1ll * kv_head_idx * head_dim;
            
            float score = 0.0;
            #pragma omp simd reduction(+:score)
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            att_head[t] = score * inv_sqrt_d;
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

            #pragma omp simd
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


#include "../include/text_layer.hpp"

#if defined(__AVX512F__) && defined(__AVX512DQ__)

#elif defined(__AVX2__) && defined(__FMA__)
void flash_attn_decode(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb,  size_t attn_heads,
    int head_dim, int kv_dim, size_t sh_offset,
    int pos, const size_t group_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    const half_cpu *key_cache_fp16 = (const half_cpu *)(key_cache);
    const half_cpu *value_cache_fp16 = (const half_cpu *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    memset(tb_base, 0, attn_heads * head_dim * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {

        const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

        const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
        const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

        const float *__restrict q_group_base = (const float *)q->ptr({0, h_base});
        float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;

        float max_row[2], sum_row[2];
        for (int i = 0; i < 2; ++i) {
            max_row[i] = -INFINITY;
            sum_row[i] = 0.0f;
        }
        float max_row_new, x, alpha, beta;
        
        for (size_t j = 0; j < pos + 1; ++j) {
            __m256 acc_0, acc_1;
            acc_0 = acc_1 =_mm256_setzero_ps();

            const half_cpu* k_j = k_ptr + j * head_dim;
            const half_cpu* v_j = v_ptr + j * head_dim;

            for (size_t k = 0; k < head_dim; k += 8) {
                // load k_j
                __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                
                    __m256 q0_vec = _mm256_loadu_ps(q_group_base + k);
                    __m256 q1_vec = _mm256_loadu_ps(q_group_base + head_dim + k);

                    acc_0 = _mm256_fmadd_ps(q0_vec, k_vec, acc_0);
                    acc_1 = _mm256_fmadd_ps(q1_vec, k_vec, acc_1);
            }

            // --- KV_ID 0 ---
            float x0 = add_reduce_mm_256(acc_0) * inv_sqrt_d;
            float max_row_new0 = max(max_row[0], x0);

            float alpha0 = expf(max_row[0] - max_row_new0);
            float beta0  = expf(x0 - max_row_new0);

            sum_row[0] = sum_row[0] * alpha0 + beta0;
            max_row[0] = max_row_new0;

            __m256 alpha_vec0 = _mm256_set1_ps(alpha0);
            __m256 beta_vec0  = _mm256_set1_ps(beta0);

            // --- KV_ID 1 ---
            float x1 = add_reduce_mm_256(acc_1) * inv_sqrt_d;
            float max_row_new1 = max(max_row[1], x1);

            float alpha1 = expf(max_row[1] - max_row_new1);
            float beta1  = expf(x1 - max_row_new1);

            sum_row[1] = sum_row[1] * alpha1 + beta1;
            max_row[1] = max_row_new1;

            __m256 alpha_vec1 = _mm256_set1_ps(alpha1);
            __m256 beta_vec1  = _mm256_set1_ps(beta1);

            for (size_t k = 0; k < head_dim; k += 8) {
                // load k_j
                __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                
                __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
                __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

                tb_vec0 = _mm256_fmadd_ps(tb_vec0, alpha_vec0, _mm256_mul_ps(v_vec, beta_vec0));
                tb_vec1 = _mm256_fmadd_ps(tb_vec1, alpha_vec1, _mm256_mul_ps(v_vec, beta_vec1));

                _mm256_storeu_ps(tb_head + k, tb_vec0);
                _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
            }
        }

        float inv_sum0 = 1 / (sum_row[0] + 1e-9f);
        __m256 sum_row_vec0 = _mm256_set1_ps(inv_sum0);
        float inv_sum1 = 1 / (sum_row[1] + 1e-9f);
        __m256 sum_row_vec1 = _mm256_set1_ps(inv_sum1);

        for (size_t k = 0; k < head_dim; k += 8) {
            __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
            __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

            tb_vec0 = _mm256_mul_ps(tb_vec0, sum_row_vec0);
            tb_vec1 = _mm256_mul_ps(tb_vec1, sum_row_vec1);

            _mm256_storeu_ps(tb_head + k, tb_vec0);
            _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
        }
    }
}

void flash_attn_prefill(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb, size_t attn_heads,
    int head_dim, int kv_dim, size_t sh_offset, int pos,
    const size_t group_size, const size_t prefill_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    const half_cpu *key_cache_fp16 = (const half_cpu *)(key_cache);
    const half_cpu *value_cache_fp16 = (const half_cpu *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    memset(tb_base, 0, prefill_size * attn_heads * head_dim * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < prefill_size; b += 2) {
        for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {
            const size_t b_size = std::min(prefill_size - b, (size_t)2);
            const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

            const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
            const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

            if (b_size == 2) {
                const float *__restrict q0_base = (const float *)q->ptr({b, h_base});
                const float *__restrict q1_base = (const float *)q->ptr({b + 1, h_base});
                float *__restrict tb0_head = (float *)tb->ptr({b}) + 1ll * h_base * head_dim;
                float *__restrict tb1_head = (float *)tb->ptr({b + 1}) + 1ll * h_base * head_dim;;

                float max_row[4], sum_row[4];
                for (int i = 0; i < 4; ++i) {
                    max_row[i] = -INFINITY;
                    sum_row[i] = 0.0f;
                }
                float max_row_new, x, alpha, beta;
                
                for (size_t j = 0; j < pos + b + 1; ++j) {
                    __m256 acc_00, acc_01, acc_10, acc_11;
                    acc_00 = acc_01 = acc_10 = acc_11 = _mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + j * head_dim;
                    const half_cpu* v_j = v_ptr + j * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                            __m256 q00_vec = _mm256_loadu_ps(q0_base + k);
                            __m256 q01_vec = _mm256_loadu_ps(q0_base + head_dim + k);
                            __m256 q10_vec = _mm256_loadu_ps(q1_base + k);
                            __m256 q11_vec = _mm256_loadu_ps(q1_base + head_dim + k);

                            acc_00 = _mm256_fmadd_ps(q00_vec, k_vec, acc_00);
                            acc_01 = _mm256_fmadd_ps(q01_vec, k_vec, acc_01);
                            acc_10 = _mm256_fmadd_ps(q10_vec, k_vec, acc_10);
                            acc_11 = _mm256_fmadd_ps(q11_vec, k_vec, acc_11);
                    }

                    // --- KV_ID 00 (Batch 0, Head 0) ---
                    float x00 = add_reduce_mm_256(acc_00) * inv_sqrt_d;
                    float max_row_new00 = max(max_row[0], x00);
                    float alpha00 = expf(max_row[0] - max_row_new00);
                    float beta00  = expf(x00 - max_row_new00);
                    sum_row[0] = sum_row[0] * alpha00 + beta00;
                    max_row[0] = max_row_new00;
                    __m256 alpha_vec00 = _mm256_set1_ps(alpha00);
                    __m256 beta_vec00  = _mm256_set1_ps(beta00);

                    // --- KV_ID 01 (Batch 0, Head 1) ---
                    float x01 = add_reduce_mm_256(acc_01) * inv_sqrt_d;
                    float max_row_new01 = max(max_row[1], x01);
                    float alpha01 = expf(max_row[1] - max_row_new01);
                    float beta01  = expf(x01 - max_row_new01);
                    sum_row[1] = sum_row[1] * alpha01 + beta01;
                    max_row[1] = max_row_new01;
                    __m256 alpha_vec01 = _mm256_set1_ps(alpha01);
                    __m256 beta_vec01  = _mm256_set1_ps(beta01);

                    // --- KV_ID 10 (Batch 1, Head 0) ---
                    float x10 = add_reduce_mm_256(acc_10) * inv_sqrt_d;
                    float max_row_new10 = max(max_row[2], x10);
                    float alpha10 = expf(max_row[2] - max_row_new10);
                    float beta10  = expf(x10 - max_row_new10);
                    sum_row[2] = sum_row[2] * alpha10 + beta10;
                    max_row[2] = max_row_new10;
                    __m256 alpha_vec10 = _mm256_set1_ps(alpha10);
                    __m256 beta_vec10  = _mm256_set1_ps(beta10);

                    // --- KV_ID 11 (Batch 1, Head 1) ---
                    float x11 = add_reduce_mm_256(acc_11) * inv_sqrt_d;
                    float max_row_new11 = max(max_row[3], x11);
                    float alpha11 = expf(max_row[3] - max_row_new11);
                    float beta11  = expf(x11 - max_row_new11);
                    sum_row[3] = sum_row[3] * alpha11 + beta11;
                    max_row[3] = max_row_new11;
                    __m256 alpha_vec11 = _mm256_set1_ps(alpha11);
                    __m256 beta_vec11  = _mm256_set1_ps(beta11);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec00 = _mm256_loadu_ps(tb0_head + k);
                        __m256 tb_vec01 = _mm256_loadu_ps(tb0_head + head_dim + k);
                        __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                        __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                        tb_vec00 = _mm256_fmadd_ps(tb_vec00, alpha_vec00, _mm256_mul_ps(v_vec, beta_vec00));
                        tb_vec01 = _mm256_fmadd_ps(tb_vec01, alpha_vec01, _mm256_mul_ps(v_vec, beta_vec01));
                        tb_vec10 = _mm256_fmadd_ps(tb_vec10, alpha_vec10, _mm256_mul_ps(v_vec, beta_vec10));
                        tb_vec11 = _mm256_fmadd_ps(tb_vec11, alpha_vec11, _mm256_mul_ps(v_vec, beta_vec11));

                        _mm256_storeu_ps(tb0_head + k, tb_vec00);
                        _mm256_storeu_ps(tb0_head + head_dim + k, tb_vec01);
                        _mm256_storeu_ps(tb1_head + k, tb_vec10);
                        _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                    }
                }

                {
                    __m256 acc_10, acc_11;
                    acc_10 = acc_11 = _mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + (pos + b + 1) * head_dim;
                    const half_cpu* v_j = v_ptr + (pos + b + 1) * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                        __m256 q10_vec = _mm256_loadu_ps(q1_base + k);
                        __m256 q11_vec = _mm256_loadu_ps(q1_base + head_dim + k);

                        acc_10 = _mm256_fmadd_ps(q10_vec, k_vec, acc_10);
                        acc_11 = _mm256_fmadd_ps(q11_vec, k_vec, acc_11);
                    }

                    // --- KV_ID 10 (Batch 1, Head 0) ---
                    float x10 = add_reduce_mm_256(acc_10) * inv_sqrt_d;
                    float max_row_new10 = max(max_row[2], x10);
                    float alpha10 = expf(max_row[2] - max_row_new10);
                    float beta10  = expf(x10 - max_row_new10);
                    sum_row[2] = sum_row[2] * alpha10 + beta10;
                    max_row[2] = max_row_new10;
                    __m256 alpha_vec10 = _mm256_set1_ps(alpha10);
                    __m256 beta_vec10  = _mm256_set1_ps(beta10);

                    // --- KV_ID 11 (Batch 1, Head 1) ---
                    float x11 = add_reduce_mm_256(acc_11) * inv_sqrt_d;
                    float max_row_new11 = max(max_row[3], x11);
                    float alpha11 = expf(max_row[3] - max_row_new11);
                    float beta11  = expf(x11 - max_row_new11);
                    sum_row[3] = sum_row[3] * alpha11 + beta11;
                    max_row[3] = max_row_new11;
                    __m256 alpha_vec11 = _mm256_set1_ps(alpha11);
                    __m256 beta_vec11  = _mm256_set1_ps(beta11);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                        __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                        tb_vec10 = _mm256_fmadd_ps(tb_vec10, alpha_vec10, _mm256_mul_ps(v_vec, beta_vec10));
                        tb_vec11 = _mm256_fmadd_ps(tb_vec11, alpha_vec11, _mm256_mul_ps(v_vec, beta_vec11));

                        _mm256_storeu_ps(tb1_head + k, tb_vec10);
                        _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                    }
                }

                __m256 sum_row_vec[4];
                for (int i = 0; i < 4; ++i) {
                    float inv_sum = 1 / (sum_row[i] + 1e-9f);
                    sum_row_vec[i] = _mm256_set1_ps(inv_sum);
                }

                for (size_t k = 0; k < head_dim; k += 8) {
                    __m256 tb_vec00 = _mm256_loadu_ps(tb0_head + k);
                    __m256 tb_vec01 = _mm256_loadu_ps(tb0_head + head_dim + k);
                    __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                    __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                    tb_vec00 = _mm256_mul_ps(tb_vec00, sum_row_vec[0]);
                    tb_vec01 = _mm256_mul_ps(tb_vec01, sum_row_vec[1]);
                    tb_vec10 = _mm256_mul_ps(tb_vec10, sum_row_vec[2]);
                    tb_vec11 = _mm256_mul_ps(tb_vec11, sum_row_vec[3]);

                    _mm256_storeu_ps(tb0_head + k, tb_vec00);
                    _mm256_storeu_ps(tb0_head + head_dim + k, tb_vec01);
                    _mm256_storeu_ps(tb1_head + k, tb_vec10);
                    _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                }
            } else {
                const float *__restrict q_group_base = (const float *)q->ptr({b, h_base});
                float *__restrict tb_head = (float *)tb->ptr({b}) + 1ll * h_base * head_dim;

                float max_row[2], sum_row[2];
                for (int i = 0; i < 2; ++i) {
                    max_row[i] = -INFINITY;
                    sum_row[i] = 0.0f;
                }
                float max_row_new, x, alpha, beta;
                
                for (size_t j = 0; j < pos + b + 1; ++j) {
                    __m256 acc_0, acc_1;
                    acc_0 = acc_1 =_mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + j * head_dim;
                    const half_cpu* v_j = v_ptr + j * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                            __m256 q0_vec = _mm256_loadu_ps(q_group_base + k);
                            __m256 q1_vec = _mm256_loadu_ps(q_group_base + head_dim + k);

                            acc_0 = _mm256_fmadd_ps(q0_vec, k_vec, acc_0);
                            acc_1 = _mm256_fmadd_ps(q1_vec, k_vec, acc_1);
                    }

                    // --- KV_ID 0 ---
                    float x0 = add_reduce_mm_256(acc_0) * inv_sqrt_d;
                    float max_row_new0 = max(max_row[0], x0);

                    float alpha0 = expf(max_row[0] - max_row_new0);
                    float beta0  = expf(x0 - max_row_new0);

                    sum_row[0] = sum_row[0] * alpha0 + beta0;
                    max_row[0] = max_row_new0;

                    __m256 alpha_vec0 = _mm256_set1_ps(alpha0);
                    __m256 beta_vec0  = _mm256_set1_ps(beta0);

                    // --- KV_ID 1 ---
                    float x1 = add_reduce_mm_256(acc_1) * inv_sqrt_d;
                    float max_row_new1 = max(max_row[1], x1);

                    float alpha1 = expf(max_row[1] - max_row_new1);
                    float beta1  = expf(x1 - max_row_new1);

                    sum_row[1] = sum_row[1] * alpha1 + beta1;
                    max_row[1] = max_row_new1;

                    __m256 alpha_vec1 = _mm256_set1_ps(alpha1);
                    __m256 beta_vec1  = _mm256_set1_ps(beta1);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
                        __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

                        tb_vec0 = _mm256_fmadd_ps(tb_vec0, alpha_vec0, _mm256_mul_ps(v_vec, beta_vec0));
                        tb_vec1 = _mm256_fmadd_ps(tb_vec1, alpha_vec1, _mm256_mul_ps(v_vec, beta_vec1));

                        _mm256_storeu_ps(tb_head + k, tb_vec0);
                        _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
                    }
                }

                float inv_sum0 = 1 / (sum_row[0] + 1e-9f);
                __m256 sum_row_vec0 = _mm256_set1_ps(inv_sum0);
                float inv_sum1 = 1 / (sum_row[1] + 1e-9f);
                __m256 sum_row_vec1 = _mm256_set1_ps(inv_sum1);

                for (size_t k = 0; k < head_dim; k += 8) {
                    __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
                    __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

                    tb_vec0 = _mm256_mul_ps(tb_vec0, sum_row_vec0);
                    tb_vec1 = _mm256_mul_ps(tb_vec1, sum_row_vec1);

                    _mm256_storeu_ps(tb_head + k, tb_vec0);
                    _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
                }
            }
        }
    }
}
#endif

void flash_attn_decode(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb,  size_t attn_heads,
    int head_dim, int kv_dim, size_t sh_offset,
    int pos, const size_t group_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    const half_cpu *key_cache_fp16 = (const half_cpu *)(key_cache);
    const half_cpu *value_cache_fp16 = (const half_cpu *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    memset(tb_base, 0, attn_heads * head_dim * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {

        const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

        const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
        const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

        const float *__restrict q_group_base = (const float *)q->ptr({0, h_base});
        float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;

        float max_row[2], sum_row[2];
        for (int i = 0; i < 2; ++i) {
            max_row[i] = -INFINITY;
            sum_row[i] = 0.0f;
        }
        float max_row_new, x, alpha, beta;
        
        for (size_t j = 0; j < pos + 1; ++j) {
            __m256 acc_0, acc_1;
            acc_0 = acc_1 =_mm256_setzero_ps();

            const half_cpu* k_j = k_ptr + j * head_dim;
            const half_cpu* v_j = v_ptr + j * head_dim;

            for (size_t k = 0; k < head_dim; k += 8) {
                // load k_j
                __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                
                    __m256 q0_vec = _mm256_loadu_ps(q_group_base + k);
                    __m256 q1_vec = _mm256_loadu_ps(q_group_base + head_dim + k);

                    acc_0 = _mm256_fmadd_ps(q0_vec, k_vec, acc_0);
                    acc_1 = _mm256_fmadd_ps(q1_vec, k_vec, acc_1);
            }


            // --- KV_ID 0 ---
            float x0 = add_reduce_mm_256(acc_0) * inv_sqrt_d;
            float max_row_new0 = max(max_row[0], x0);

            float alpha0 = expf(max_row[0] - max_row_new0);
            float beta0  = expf(x0 - max_row_new0);

            sum_row[0] = sum_row[0] * alpha0 + beta0;
            max_row[0] = max_row_new0;

            __m512 alpha_vec0 = _mm512_set1_ps(alpha0);
            __m512 beta_vec0  = _mm512_set1_ps(beta0);

            // --- KV_ID 1 ---
            float x1 = add_reduce_mm_256(acc_1) * inv_sqrt_d;
            float max_row_new1 = max(max_row[1], x1);

            float alpha1 = expf(max_row[1] - max_row_new1);
            float beta1  = expf(x1 - max_row_new1);

            sum_row[1] = sum_row[1] * alpha1 + beta1;
            max_row[1] = max_row_new1;

            __m512 alpha_vec1 = _mm512_set1_ps(alpha1);
            __m512 beta_vec1  = _mm512_set1_ps(beta1);

            for (size_t k = 0; k < head_dim; k += 16) {
                // load v_j
                __m256i v_h = _mm256_loadu_si256((__m256i const*)(v_j + k));
                __m512 v_vec = _mm512_cvtph_ps(v_h); 
                
                __m512 tb_vec0 = _mm512_loadu_ps(tb_head + k);
                __m512 tb_vec1 = _mm512_loadu_ps(tb_head + head_dim + k);

                tb_vec0 = _mm512_fmadd_ps(tb_vec0, alpha_vec0, _mm512_mul_ps(v_vec, beta_vec0));
                tb_vec1 = _mm512_fmadd_ps(tb_vec1, alpha_vec1, _mm512_mul_ps(v_vec, beta_vec1));

                _mm512_storeu_ps(tb_head + k, tb_vec0);
                _mm512_storeu_ps(tb_head + head_dim + k, tb_vec1);
            }
        }

        float inv_sum0 = 1 / (sum_row[0] + 1e-9f);
        __m512 sum_row_vec0 = _mm512_set1_ps(inv_sum0);
        float inv_sum1 = 1 / (sum_row[1] + 1e-9f);
        __m512 sum_row_vec1 = _mm512_set1_ps(inv_sum1);

        for (size_t k = 0; k < head_dim; k += 16) {
            __m512 tb_vec0 = _mm512_loadu_ps(tb_head + k);
            __m512 tb_vec1 = _mm512_loadu_ps(tb_head + head_dim + k);

            tb_vec0 = _mm512_mul_ps(tb_vec0, sum_row_vec0);
            tb_vec1 = _mm512_mul_ps(tb_vec1, sum_row_vec1);

            _mm512_storeu_ps(tb_head + k, tb_vec0);
            _mm512_storeu_ps(tb_head + head_dim + k, tb_vec1);
        }
    }
}

void sdpa_attn_decode(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb, size_t attn_heads,
    int head_dim, int kv_dim, size_t sh_offset,
    int pos, const size_t group_size
)  {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    const size_t seq_len = att->shape[att->ndim - 1];

    const half_cpu *key_cache_fp16 = (const half_cpu *)(key_cache);
    const half_cpu *value_cache_fp16 = (const half_cpu *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {

        const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

        const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
        const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

        const float *__restrict q_group_base = (const float *)q->ptr({0, h_base});
        float *__restrict att_group_base = (float *)att->ptr({0, h_base});
        float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;

        // gemm qk
        float max_att0, max_att1;
        max_att0 = max_att1 = -INFINITY;

        #pragma omp parallel for schedule(static) reduction(max:max_att0) reduction(max:max_att1)
        for (size_t j = 0; j < pos + 1; ++j) {
            __m256 sum0_vec, sum1_vec;
            sum0_vec = sum1_vec = _mm256_setzero_ps();

            const half_cpu *k_j = k_ptr + j * head_dim;

            for (size_t k = 0; k < head_dim; k += 8) {
                // load k_j
                __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k));
                __m256 k_vec = _mm256_cvtph_ps(k_half);
                __m256 q0_vec = _mm256_loadu_ps(q_group_base + k);
                __m256 q1_vec = _mm256_loadu_ps(q_group_base + head_dim + k);
                
                sum0_vec = _mm256_fmadd_ps(q0_vec, k_vec, sum0_vec);
                sum1_vec = _mm256_fmadd_ps(q1_vec, k_vec, sum1_vec);
            }

            float value0 = add_reduce_mm_256(sum0_vec) * inv_sqrt_d;
            float value1 = add_reduce_mm_256(sum1_vec) * inv_sqrt_d;

            att_group_base[j] = value0;
            att_group_base[j + seq_len] = value1;

            max_att0 = max(value0, max_att0);
            max_att1 = max(value1, max_att1);
        }

        softmax_with_max(att_group_base, max_att0, pos + 1);
        softmax_with_max(att_group_base + seq_len, max_att1, pos + 1);

        #pragma omp parallel for
        for (size_t k = 0; k < head_dim; k += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (size_t j = 0; j < pos + 1; ++j) {
                // Load 8 contiguous values from V
                __m128i v_half = _mm_loadu_si128((__m128i const*)(v_ptr + j * head_dim + k));
                __m256 b = _mm256_cvtph_ps(v_half);

                // Broadcast scalars
                __m256 a0 = _mm256_set1_ps(att_group_base[j]);
                __m256 a1 = _mm256_set1_ps(att_group_base[seq_len + j]);

                // FMA
                acc0 = _mm256_fmadd_ps(a0, b, acc0);
                acc1 = _mm256_fmadd_ps(a1, b, acc1);
            }

            _mm256_storeu_ps(tb_head + k, acc0);
            _mm256_storeu_ps(tb_head + head_dim + k, acc1);
        }
    }
}

void flash_attn_prefill(
    const char *__restrict key_cache,
    const char *__restrict value_cache,
    const Tensor *__restrict q, Tensor *__restrict att,
    Tensor *__restrict tb, size_t attn_heads,
    int head_dim, int kv_dim, size_t sh_offset, int pos,
    const size_t group_size, const size_t prefill_size
) {
    const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    const half_cpu *key_cache_fp16 = (const half_cpu *)(key_cache);
    const half_cpu *value_cache_fp16 = (const half_cpu *)(value_cache);
    float *__restrict tb_base = (float *)tb->ptr();

    memset(tb_base, 0, prefill_size * attn_heads * head_dim * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < prefill_size; b += 2) {
        for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {
            const size_t b_size = std::min(prefill_size - b, (size_t)2);
            const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

            const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
            const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

            if (b_size == 2) {
                const float *__restrict q0_base = (const float *)q->ptr({b, h_base});
                const float *__restrict q1_base = (const float *)q->ptr({b + 1, h_base});
                float *__restrict tb0_head = (float *)tb->ptr({b}) + 1ll * h_base * head_dim;
                float *__restrict tb1_head = (float *)tb->ptr({b + 1}) + 1ll * h_base * head_dim;;

                float max_row[4], sum_row[4];
                for (int i = 0; i < 4; ++i) {
                    max_row[i] = -INFINITY;
                    sum_row[i] = 0.0f;
                }
                float max_row_new, x, alpha, beta;
                
                for (size_t j = 0; j < pos + b + 1; ++j) {
                    __m256 acc_00, acc_01, acc_10, acc_11;
                    acc_00 = acc_01 = acc_10 = acc_11 = _mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + j * head_dim;
                    const half_cpu* v_j = v_ptr + j * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                            __m256 q00_vec = _mm256_loadu_ps(q0_base + k);
                            __m256 q01_vec = _mm256_loadu_ps(q0_base + head_dim + k);
                            __m256 q10_vec = _mm256_loadu_ps(q1_base + k);
                            __m256 q11_vec = _mm256_loadu_ps(q1_base + head_dim + k);

                            acc_00 = _mm256_fmadd_ps(q00_vec, k_vec, acc_00);
                            acc_01 = _mm256_fmadd_ps(q01_vec, k_vec, acc_01);
                            acc_10 = _mm256_fmadd_ps(q10_vec, k_vec, acc_10);
                            acc_11 = _mm256_fmadd_ps(q11_vec, k_vec, acc_11);
                    }

                    // --- KV_ID 00 (Batch 0, Head 0) ---
                    float x00 = add_reduce_mm_256(acc_00) * inv_sqrt_d;
                    float max_row_new00 = max(max_row[0], x00);
                    float alpha00 = expf(max_row[0] - max_row_new00);
                    float beta00  = expf(x00 - max_row_new00);
                    sum_row[0] = sum_row[0] * alpha00 + beta00;
                    max_row[0] = max_row_new00;
                    __m256 alpha_vec00 = _mm256_set1_ps(alpha00);
                    __m256 beta_vec00  = _mm256_set1_ps(beta00);

                    // --- KV_ID 01 (Batch 0, Head 1) ---
                    float x01 = add_reduce_mm_256(acc_01) * inv_sqrt_d;
                    float max_row_new01 = max(max_row[1], x01);
                    float alpha01 = expf(max_row[1] - max_row_new01);
                    float beta01  = expf(x01 - max_row_new01);
                    sum_row[1] = sum_row[1] * alpha01 + beta01;
                    max_row[1] = max_row_new01;
                    __m256 alpha_vec01 = _mm256_set1_ps(alpha01);
                    __m256 beta_vec01  = _mm256_set1_ps(beta01);

                    // --- KV_ID 10 (Batch 1, Head 0) ---
                    float x10 = add_reduce_mm_256(acc_10) * inv_sqrt_d;
                    float max_row_new10 = max(max_row[2], x10);
                    float alpha10 = expf(max_row[2] - max_row_new10);
                    float beta10  = expf(x10 - max_row_new10);
                    sum_row[2] = sum_row[2] * alpha10 + beta10;
                    max_row[2] = max_row_new10;
                    __m256 alpha_vec10 = _mm256_set1_ps(alpha10);
                    __m256 beta_vec10  = _mm256_set1_ps(beta10);

                    // --- KV_ID 11 (Batch 1, Head 1) ---
                    float x11 = add_reduce_mm_256(acc_11) * inv_sqrt_d;
                    float max_row_new11 = max(max_row[3], x11);
                    float alpha11 = expf(max_row[3] - max_row_new11);
                    float beta11  = expf(x11 - max_row_new11);
                    sum_row[3] = sum_row[3] * alpha11 + beta11;
                    max_row[3] = max_row_new11;
                    __m256 alpha_vec11 = _mm256_set1_ps(alpha11);
                    __m256 beta_vec11  = _mm256_set1_ps(beta11);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec00 = _mm256_loadu_ps(tb0_head + k);
                        __m256 tb_vec01 = _mm256_loadu_ps(tb0_head + head_dim + k);
                        __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                        __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                        tb_vec00 = _mm256_fmadd_ps(tb_vec00, alpha_vec00, _mm256_mul_ps(v_vec, beta_vec00));
                        tb_vec01 = _mm256_fmadd_ps(tb_vec01, alpha_vec01, _mm256_mul_ps(v_vec, beta_vec01));
                        tb_vec10 = _mm256_fmadd_ps(tb_vec10, alpha_vec10, _mm256_mul_ps(v_vec, beta_vec10));
                        tb_vec11 = _mm256_fmadd_ps(tb_vec11, alpha_vec11, _mm256_mul_ps(v_vec, beta_vec11));

                        _mm256_storeu_ps(tb0_head + k, tb_vec00);
                        _mm256_storeu_ps(tb0_head + head_dim + k, tb_vec01);
                        _mm256_storeu_ps(tb1_head + k, tb_vec10);
                        _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                    }
                }

                {
                    __m256 acc_10, acc_11;
                    acc_10 = acc_11 = _mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + (pos + b + 1) * head_dim;
                    const half_cpu* v_j = v_ptr + (pos + b + 1) * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                        __m256 q10_vec = _mm256_loadu_ps(q1_base + k);
                        __m256 q11_vec = _mm256_loadu_ps(q1_base + head_dim + k);

                        acc_10 = _mm256_fmadd_ps(q10_vec, k_vec, acc_10);
                        acc_11 = _mm256_fmadd_ps(q11_vec, k_vec, acc_11);
                    }

                    // --- KV_ID 10 (Batch 1, Head 0) ---
                    float x10 = add_reduce_mm_256(acc_10) * inv_sqrt_d;
                    float max_row_new10 = max(max_row[2], x10);
                    float alpha10 = expf(max_row[2] - max_row_new10);
                    float beta10  = expf(x10 - max_row_new10);
                    sum_row[2] = sum_row[2] * alpha10 + beta10;
                    max_row[2] = max_row_new10;
                    __m256 alpha_vec10 = _mm256_set1_ps(alpha10);
                    __m256 beta_vec10  = _mm256_set1_ps(beta10);

                    // --- KV_ID 11 (Batch 1, Head 1) ---
                    float x11 = add_reduce_mm_256(acc_11) * inv_sqrt_d;
                    float max_row_new11 = max(max_row[3], x11);
                    float alpha11 = expf(max_row[3] - max_row_new11);
                    float beta11  = expf(x11 - max_row_new11);
                    sum_row[3] = sum_row[3] * alpha11 + beta11;
                    max_row[3] = max_row_new11;
                    __m256 alpha_vec11 = _mm256_set1_ps(alpha11);
                    __m256 beta_vec11  = _mm256_set1_ps(beta11);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                        __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                        tb_vec10 = _mm256_fmadd_ps(tb_vec10, alpha_vec10, _mm256_mul_ps(v_vec, beta_vec10));
                        tb_vec11 = _mm256_fmadd_ps(tb_vec11, alpha_vec11, _mm256_mul_ps(v_vec, beta_vec11));

                        _mm256_storeu_ps(tb1_head + k, tb_vec10);
                        _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                    }
                }

                __m256 sum_row_vec[4];
                for (int i = 0; i < 4; ++i) {
                    float inv_sum = 1 / (sum_row[i] + 1e-9f);
                    sum_row_vec[i] = _mm256_set1_ps(inv_sum);
                }

                for (size_t k = 0; k < head_dim; k += 8) {
                    __m256 tb_vec00 = _mm256_loadu_ps(tb0_head + k);
                    __m256 tb_vec01 = _mm256_loadu_ps(tb0_head + head_dim + k);
                    __m256 tb_vec10 = _mm256_loadu_ps(tb1_head + k);
                    __m256 tb_vec11 = _mm256_loadu_ps(tb1_head + head_dim + k);

                    tb_vec00 = _mm256_mul_ps(tb_vec00, sum_row_vec[0]);
                    tb_vec01 = _mm256_mul_ps(tb_vec01, sum_row_vec[1]);
                    tb_vec10 = _mm256_mul_ps(tb_vec10, sum_row_vec[2]);
                    tb_vec11 = _mm256_mul_ps(tb_vec11, sum_row_vec[3]);

                    _mm256_storeu_ps(tb0_head + k, tb_vec00);
                    _mm256_storeu_ps(tb0_head + head_dim + k, tb_vec01);
                    _mm256_storeu_ps(tb1_head + k, tb_vec10);
                    _mm256_storeu_ps(tb1_head + head_dim + k, tb_vec11);
                }
            } else {
                const float *__restrict q_group_base = (const float *)q->ptr({b, h_base});
                float *__restrict tb_head = (float *)tb->ptr({b}) + 1ll * h_base * head_dim;

                float max_row[2], sum_row[2];
                for (int i = 0; i < 2; ++i) {
                    max_row[i] = -INFINITY;
                    sum_row[i] = 0.0f;
                }
                float max_row_new, x, alpha, beta;
                
                for (size_t j = 0; j < pos + b + 1; ++j) {
                    __m256 acc_0, acc_1;
                    acc_0 = acc_1 =_mm256_setzero_ps();

                    const half_cpu* k_j = k_ptr + j * head_dim;
                    const half_cpu* v_j = v_ptr + j * head_dim;

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i k_half = _mm_loadu_si128((__m128i const*)(k_j + k)); // 8 x fp16
                        __m256 k_vec = _mm256_cvtph_ps(k_half); // → 8 x fp32
                        
                            __m256 q0_vec = _mm256_loadu_ps(q_group_base + k);
                            __m256 q1_vec = _mm256_loadu_ps(q_group_base + head_dim + k);

                            acc_0 = _mm256_fmadd_ps(q0_vec, k_vec, acc_0);
                            acc_1 = _mm256_fmadd_ps(q1_vec, k_vec, acc_1);
                    }

                    // --- KV_ID 0 ---
                    float x0 = add_reduce_mm_256(acc_0) * inv_sqrt_d;
                    float max_row_new0 = max(max_row[0], x0);

                    float alpha0 = expf(max_row[0] - max_row_new0);
                    float beta0  = expf(x0 - max_row_new0);

                    sum_row[0] = sum_row[0] * alpha0 + beta0;
                    max_row[0] = max_row_new0;

                    __m256 alpha_vec0 = _mm256_set1_ps(alpha0);
                    __m256 beta_vec0  = _mm256_set1_ps(beta0);

                    // --- KV_ID 1 ---
                    float x1 = add_reduce_mm_256(acc_1) * inv_sqrt_d;
                    float max_row_new1 = max(max_row[1], x1);

                    float alpha1 = expf(max_row[1] - max_row_new1);
                    float beta1  = expf(x1 - max_row_new1);

                    sum_row[1] = sum_row[1] * alpha1 + beta1;
                    max_row[1] = max_row_new1;

                    __m256 alpha_vec1 = _mm256_set1_ps(alpha1);
                    __m256 beta_vec1  = _mm256_set1_ps(beta1);

                    for (size_t k = 0; k < head_dim; k += 8) {
                        // load k_j
                        __m128i v_half = _mm_loadu_si128((__m128i const*)(v_j + k)); // 8 x fp16
                        __m256 v_vec = _mm256_cvtph_ps(v_half); // → 8 x fp32
                        
                        __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
                        __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

                        tb_vec0 = _mm256_fmadd_ps(tb_vec0, alpha_vec0, _mm256_mul_ps(v_vec, beta_vec0));
                        tb_vec1 = _mm256_fmadd_ps(tb_vec1, alpha_vec1, _mm256_mul_ps(v_vec, beta_vec1));

                        _mm256_storeu_ps(tb_head + k, tb_vec0);
                        _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
                    }
                }

                float inv_sum0 = 1 / (sum_row[0] + 1e-9f);
                __m256 sum_row_vec0 = _mm256_set1_ps(inv_sum0);
                float inv_sum1 = 1 / (sum_row[1] + 1e-9f);
                __m256 sum_row_vec1 = _mm256_set1_ps(inv_sum1);

                for (size_t k = 0; k < head_dim; k += 8) {
                    __m256 tb_vec0 = _mm256_loadu_ps(tb_head + k);
                    __m256 tb_vec1 = _mm256_loadu_ps(tb_head + head_dim + k);

                    tb_vec0 = _mm256_mul_ps(tb_vec0, sum_row_vec0);
                    tb_vec1 = _mm256_mul_ps(tb_vec1, sum_row_vec1);

                    _mm256_storeu_ps(tb_head + k, tb_vec0);
                    _mm256_storeu_ps(tb_head + head_dim + k, tb_vec1);
                }
            }
        }
    }
}

void fused_att_dispatch(
    const char *k_cache_l, const char *v_cache_l, const float *k_cache_s,
    const float *v_cache_s, const Tensor *q, Tensor *att, Tensor *qkv_out,
    const int num_heads, const int head_dim, const int kv_mul,
    const int kv_dim, const size_t kv_all_off, const int pos,
    const DType::Type key_dtype, const DType::Type value_dtype,
    const size_t group_size, const size_t prefill_size, bool warm_up
) {
    if (key_dtype == DType::FP16 && value_dtype == DType::FP16 && kv_mul == 2) {

        if (prefill_size > 1) {
            flash_attn_prefill(
                k_cache_l, v_cache_l, q, att,
                qkv_out, num_heads, head_dim,
                kv_dim, kv_all_off, pos,
                group_size, prefill_size
            );
        } else {
            sdpa_attn_decode(
                k_cache_l, v_cache_l, q, att,
                qkv_out, num_heads, head_dim,
                kv_dim, kv_all_off, pos, group_size
            );
        }
    } else {
        if (prefill_size > 1) {
            attn_scores_all_heads_prefill(
                k_cache_l, k_cache_s, q, att,
                num_heads, kv_mul, head_dim,
                kv_dim, kv_all_off, pos,
                group_size, prefill_size, key_dtype
            );
        } else {
            attn_scores_all_heads_decode(
                k_cache_l, k_cache_s, q, att,
                num_heads, kv_mul, head_dim, kv_dim,
                kv_all_off, pos, group_size, key_dtype
            );
        }

        attn_weighted_sum_all_heads(
            v_cache_l, v_cache_s, att, qkv_out,
            num_heads, kv_mul, head_dim, kv_dim,
            kv_all_off, pos, group_size,
            prefill_size, value_dtype
        );
    }

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < prefill_size; ++i) {
                state->att->printDebug("att", {i});
                state->qkv_out->printDebug("qkv_out", {i});
            }
        }
    #endif
}

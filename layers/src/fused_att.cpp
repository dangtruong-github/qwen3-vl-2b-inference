#include "../include/text_layer.hpp"

#ifdef __ARM_NEON
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

    #pragma omp parallel 
    { 
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

        #pragma omp for schedule(static)
        for (size_t h_base = 0; h_base < attn_heads; h_base += 2) {

            float max_row[2] = {-INFINITY, -INFINITY};
            float sum_row[2] = {0.0f, 0.0f};

            const size_t cache_offset = 1ll * (h_base >> 1) * sh_offset;

            const half_cpu *__restrict k_ptr = key_cache_fp16 + cache_offset;
            const half_cpu *__restrict v_ptr = value_cache_fp16 + cache_offset;

            const float *__restrict q_group_base = (const float *)q->ptr({0, h_base});
            float *__restrict tb_head = tb_base + 1ll * h_base * head_dim;
            
            for (size_t j = 0; j < pos + 1; ++j) {
                float32x4_t acc_0, acc_1;
                acc_0 = acc_1 = vdupq_n_f32(0.0f);

                const half_cpu* k_j = k_ptr + j * head_dim;
                const half_cpu* v_j = v_ptr + j * head_dim;

                for (size_t k = 0; k < head_dim; k += 4) {
                    // load k_j
                    float16x4_t k_half = vld1_f16((const __fp16*)(k_j + k));
                    float32x4_t k_vec = vcvt_f32_f16(k_half);
                    
                    float32x4_t q0_vec = vld1q_f32(q_group_base + k);
                    float32x4_t q1_vec = vld1q_f32(q_group_base + head_dim + k);

                    acc_0 = vfmaq_f32(acc_0, q0_vec, k_vec);
                    acc_1 = vfmaq_f32(acc_1, q1_vec, k_vec);
                }

                // --- KV_ID 0 ---
                float x0 = vaddvq_f32(acc_0) * inv_sqrt_d;
                float max_row_new0 = max(max_row[0], x0);

                float alpha0 = expf(max_row[0] - max_row_new0);
                float beta0  = expf(x0 - max_row_new0);

                sum_row[0] = sum_row[0] * alpha0 + beta0;
                max_row[0] = max_row_new0;

                float32x4_t alpha_vec0 = vdupq_n_f32(alpha0);
                float32x4_t beta_vec0  = vdupq_n_f32(beta0);

                // --- KV_ID 1 ---
                float x1 = vaddvq_f32(acc_1) * inv_sqrt_d;
                float max_row_new1 = max(max_row[1], x1);

                float alpha1 = expf(max_row[1] - max_row_new1);
                float beta1  = expf(x1 - max_row_new1);

                sum_row[1] = sum_row[1] * alpha1 + beta1;
                max_row[1] = max_row_new1;

                float32x4_t alpha_vec1 = vdupq_n_f32(alpha1);
                float32x4_t beta_vec1  = vdupq_n_f32(beta1);

                for (size_t k = 0; k < head_dim; k += 4) {
                    // load v_j
                    float16x4_t v_h = vld1_f16((const __fp16*)(v_j + k));
                    float32x4_t v_vec = vcvt_f32_f16(v_h);

                    float32x4_t tb_vec0 = vld1q_f32(tb_head + k);
                    float32x4_t tb_vec1 = vld1q_f32(tb_head + head_dim + k);

                    tb_vec0 = vfmaq_f32(vmulq_f32(v_vec, beta_vec0), tb_vec0, alpha_vec0);
                    tb_vec1 = vfmaq_f32(vmulq_f32(v_vec, beta_vec1), tb_vec1, alpha_vec1);

                    vst1q_f32(tb_head + k, tb_vec0);
                    vst1q_f32(tb_head + head_dim + k, tb_vec1);
                }
            }

            float inv_sum0 = 1 / (sum_row[0] + 1e-9f);
            float32x4_t sum_row_vec0 = vdupq_n_f32(inv_sum0);
            float inv_sum1 = 1 / (sum_row[1] + 1e-9f);
            float32x4_t sum_row_vec1 = vdupq_n_f32(inv_sum1);

            for (size_t k = 0; k < head_dim; k += 4) {
                float32x4_t tb_vec0 = vld1q_f32(tb_head + k);
                float32x4_t tb_vec1 = vld1q_f32(tb_head + head_dim + k);

                tb_vec0 = vmulq_f32(tb_vec0, sum_row_vec0);
                tb_vec1 = vmulq_f32(tb_vec1, sum_row_vec1);

                vst1q_f32(tb_head + k, tb_vec0);
                vst1q_f32(tb_head + head_dim + k, tb_vec1);
            }
        }
    }
}
#endif

void fused_att_dispatch(
    const char *k_cache_l, const char *v_cache_l, const float *k_cache_s,
    const float *v_cache_s, const Tensor *q, Tensor *att, Tensor *qkv_out,
    const int num_heads, const int head_dim, const int kv_mul,
    const int kv_dim, const size_t kv_all_off, const int pos,
    const DType::Type key_dtype, const DType::Type value_dtype,
    const size_t group_size, const size_t prefill_size, bool warm_up
) {
    #ifdef __ARM_NEON
        if (
            key_dtype == DType::FP16 && value_dtype == DType::FP16
            && kv_mul == 2 && prefill_size == 1
        ) {
            flash_attn_decode(
                k_cache_l, v_cache_l, q, att,
                qkv_out, num_heads, head_dim,
                kv_dim, kv_all_off, pos, group_size
            );
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
                kv_all_off, pos, prefill_size,
                group_size, value_dtype
            );
        }
    #else
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
            kv_all_off, pos, prefill_size,
            group_size, value_dtype
        );
    #endif

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < prefill_size; ++i) {
                att->printDebug("att", {i});
                qkv_out->printDebug("qkv_out", {i});
            }
        }
    #endif
}

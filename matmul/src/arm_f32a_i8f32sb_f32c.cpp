#include "../include/cpu_wrapper.hpp"

void arm_f32a_i8f32sb_f32c_m1(
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, float* mat_C,
    size_t N, size_t K, size_t group_size, bool add_to_c
) {
    alignas(32) int8_t a_q8[K];
    float a_q8_s[K >> 5];

    for (int kk = 0; kk < K; kk += group_size) {
        float32x4_t v_max = vdupq_n_f32(0.0f);
        for (int k = kk; k < kk + group_size; k += 4) {
            float32x4_t f0 = vld1q_f32(mat_A + k);
            v_max = vmaxq_f32(v_max, vabsq_f32(f0));
        }
        float max_val = vmaxvq_f32(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        float32x4_t invS = vdupq_n_f32(inv_scale);

        // ---------------- quantize ----------------
        for (int k = kk; k < kk + group_size; k += 16) {
            float32x4_t f0 = vld1q_f32(mat_A + k);
            float32x4_t f1 = vld1q_f32(mat_A + k + 4);
            float32x4_t f2 = vld1q_f32(mat_A + k + 8);
            float32x4_t f3 = vld1q_f32(mat_A + k + 12);

            f0 = vmulq_f32(f0, invS);
            f1 = vmulq_f32(f1, invS);
            f2 = vmulq_f32(f2, invS);
            f3 = vmulq_f32(f3, invS);

            int32x4_t i0 = vcvtnq_s32_f32(f0);
            int32x4_t i1 = vcvtnq_s32_f32(f1);
            int32x4_t i2 = vcvtnq_s32_f32(f2);
            int32x4_t i3 = vcvtnq_s32_f32(f3);

            int16x4_t i0_16 = vqmovn_s32(i0); 
            int16x4_t i1_16 = vqmovn_s32(i1);
            int16x4_t i2_16 = vqmovn_s32(i2); 
            int16x4_t i3_16 = vqmovn_s32(i3);

            // 2. Combine them into a single 128-bit vector (int16x8_t)
            int16x8_t p01 = vcombine_s16(i0_16, i1_16);
            int16x8_t p23 = vcombine_s16(i2_16, i3_16);

            // fix lane order// 2. Narrow with SIGNED saturation (clamps to -128...127)
            int8x8_t low_s8  = vqmovn_s16(p01);
            int8x8_t high_s8 = vqmovn_s16(p23);

            // 3. Combine into a single 128-bit signed vector
            int8x16_t q8s = vcombine_s8(low_s8, high_s8);

            // 4. Store 16 bytes (128 bits) as signed integers
            vst1q_s8(a_q8 + k, q8s);
        }
    }

    const size_t K_g = K / group_size;

    const int16x4_t ones16 = vdup_n_s16(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        float32x4_t c0_f = vdupq_n_f32(0.0f);

        const size_t jjK = jj * K;
        const int8_t *__restrict b0_ptr = mat_B_in + jjK;
        const float *__restrict b_s_ptr = mat_B_scales + (jjK / group_size);
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = kk / group_size;
            int32x4_t c0 = vdupq_n_s32(0);

            for (size_t k = kk; k < kk + group_size; k += 16) {
                int8x16_t a_vec = vld1q_s8(a_q8 + k);
                int8x16_t b0 = vld1q_s8(b0_ptr + k);

                c0 = vdotq_s32(c0, a_vec, b0);
            }
            
            c0_f = vfmaq_f32(
                c0_f, vcvtq_f32_s32(c0),
                vdupq_n_f32(a_q8_s[g_off] * b_s_ptr[g_off])
            );
        }
        
        if (add_to_c) {     
            mat_C[jj] += vaddvq_f32(c0_f);
        } else {        
            mat_C[jj] = vaddvq_f32(c0_f);
        }
    }
}

void arm_f32a_i8f32sb_f32c(
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, const int *sum_int8_B, float* mat_C,
    size_t M, size_t N, size_t K, size_t group_size, bool add_to_c
) {
    if (N >= 1024 && K >= 1024) {
        for (size_t i = 0; i < M; ++i) {
            arm_f32a_i8f32sb_f32c_m1(
                mat_A, mat_B_in, mat_B_scales,
                mat_C, N, K, group_size, add_to_c
            );
            mat_A += K;
            mat_C += N;
        }
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;

            // -------- GEMM --------
            for (size_t k = 0; k < K; ++k) {
                float a = mat_A[i * K + k];

                // linear index into B (matches quantizer layout)
                size_t b_linear_idx;
                b_linear_idx = j * K + k;

                size_t scale_idx = b_linear_idx / group_size;
                float scale = mat_B_scales[scale_idx];

                float b = (float)mat_B_in[b_linear_idx] * scale;
                acc += a * b;
            }

            if (add_to_c) {
                mat_C[i * N + j] += acc;
            } else {
                mat_C[i * N + j] = acc;
            }
        }
    }
}
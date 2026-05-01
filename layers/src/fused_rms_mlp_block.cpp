#include "../include/text_layer.hpp"

#ifdef __ARM_NEON
void fused_rms_mlp_swiglu_m1(
    const PtrPair w_rms,  const PtrPair w_gate, const PtrPair w_up,
    const float *x_ptr, float *t_ptr, float *gate_ptr, float *up_ptr,
    const size_t hidden_size, const size_t inter_dim,
    const size_t group_size, const float eps
) {
    const int8_t *__restrict w_rms_w = static_cast<const int8_t*>(w_rms.buf);
    const float *__restrict w_rms_s = static_cast<const float*>(w_rms.scale);

    const int8_t *__restrict w_gate_w = static_cast<const int8_t *>(w_gate.buf);
    const float *__restrict w_gate_s = static_cast<const float *>(w_gate.scale);
    const int8_t *__restrict w_up_w = static_cast<const int8_t *>(w_up.buf);
    const float *__restrict w_up_s = static_cast<const float *>(w_up.scale);

    const float inv_hs = 1.0f / (float)hidden_size;
    
    alignas(32) int8_t a_q8[hidden_size];
    float a_q8_s[hidden_size >> 5];

    // 1. RMS
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (size_t j = 0; j < hidden_size; ++j) {
        ss += x_ptr[j] * x_ptr[j];
    }

    float inv_rms = 1.0f / sqrtf(ss * inv_hs + eps);

    const size_t K_g = hidden_size / group_size;

    // 2. Normalize + dequantized scale (inplace)
    for (size_t g = 0; g < hidden_size; g += group_size) {
        size_t group_id = g / group_size;
        float s = w_rms_s[group_id];
        float combined = s * inv_rms;

        const int8_t *sq = &w_rms_w[g];
        const float *xp = &x_ptr[g];
        float *tp = &t_ptr[g];

        float32x4_t combined_v = vdupq_n_f32(combined);
        float32x4_t v_max = vdupq_n_f32(0.0f);

        for (size_t k = 0; k < group_size; k += 8) {
            // -------- first 8 --------
            float32x4_t xp0 = vld1q_f32(xp + k);
            float32x4_t xp1 = vld1q_f32(xp + k + 4);

            xp0 = vmulq_f32(xp0, combined_v);
            xp1 = vmulq_f32(xp1, combined_v);

            int8x8_t sq8 = vld1_s8(sq + k);
            int16x8_t sq16 = vmovl_s8(sq8);

            int32x4_t sq32_low  = vmovl_s16(vget_low_s16(sq16));
            int32x4_t sq32_high = vmovl_s16(vget_high_s16(sq16));
            
            float32x4_t sqf_low  = vcvtq_f32_s32(sq32_low);
            float32x4_t sqf_high = vcvtq_f32_s32(sq32_high);

            xp0 = vmulq_f32(xp0, sqf_low);
            xp1 = vmulq_f32(xp1, sqf_high);

            v_max = vmaxq_f32(v_max, vabsq_f32(xp0));
            v_max = vmaxq_f32(v_max, vabsq_f32(xp1));

            vst1q_f32(tp + k, xp0);
            vst1q_f32(tp + k + 4, xp1);
        }

        float max_val = vmaxvq_f32(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[group_id] = scale;

        float32x4_t invS = vdupq_n_f32(inv_scale);

        // ---------------- quantize ----------------
        for (int k = g; k < g + group_size; k += 16) {
            float32x4_t f0 = vld1q_f32(t_ptr + k);
            float32x4_t f1 = vld1q_f32(t_ptr + k + 4);
            float32x4_t f2 = vld1q_f32(t_ptr + k + 8);
            float32x4_t f3 = vld1q_f32(t_ptr + k + 12);

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

    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < inter_dim; ++jj) {
        float32x4_t c_up_f = vdupq_n_f32(0.0f);
        float32x4_t c_gate_f = vdupq_n_f32(0.0f);

        const size_t jjK = jj * hidden_size;
        const size_t jjK_g = jj * K_g;
        const int8_t *__restrict up_w_ptr = w_up_w + jjK;
        const float *__restrict up_s_ptr = w_up_s + jjK_g;
        const int8_t *__restrict gate_w_ptr = w_gate_w + jjK;
        const float *__restrict gate_s_ptr = w_gate_s + jjK_g;
        
        for (size_t kk = 0; kk < hidden_size; kk += group_size) {
            const size_t g_off = kk / group_size;
            int32x4_t c_up = vdupq_n_s32(0);
            int32x4_t c_gate = vdupq_n_s32(0);

            for (size_t k = kk; k < kk + group_size; k += 16) {
                int8x16_t a_vec = vld1q_s8(a_q8 + k);
                int8x16_t b_up = vld1q_s8(up_w_ptr + k);
                int8x16_t b_gate = vld1q_s8(gate_w_ptr + k);

                c_up = vdotq_s32(c_up, a_vec, b_up);
                c_gate = vdotq_s32(c_gate, a_vec, b_gate);
            }

            const float a_q8_s_val = a_q8_s[g_off];
            
            c_up_f = vfmaq_f32(
                c_up_f, vcvtq_f32_s32(c_up),
                vdupq_n_f32(a_q8_s_val * up_s_ptr[g_off])
            );
            c_gate_f = vfmaq_f32(
                c_gate_f, vcvtq_f32_s32(c_gate),
                vdupq_n_f32(a_q8_s_val * gate_s_ptr[g_off])
            );
        }

        const float gate_tmp = vaddvq_f32(c_gate_f); 
        const float silu = gate_tmp / (1.0f + expf(-gate_tmp));

        const float up_tmp = vaddvq_f32(c_up_f);
        gate_ptr[jj] = up_tmp * silu;
    }
}
#endif

void fused_rms_mlp_swiglu_dispatch(
    const Tensor *rms_attn_w, const Tensor *w_mlp_gate, const Tensor *w_mlp_up,
    const Tensor *x, Tensor *t, Tensor *gate, Tensor *up, const size_t M,
    const size_t hidden_size, const size_t inter_dim,
    const DType::Type dtype_w, const DType::Type dtype_s,
    const bool text_gq, const float eps, const size_t group_size,
    const size_t layer_offset, bool warm_up
) { 
    /*
    #ifdef __ARM_NEON
        if (
            dtype_w == DType::INT8 && dtype_s == DType::FP32 && text_gq
            && !w_mlp_gate->permuted && t->dtype == DType::FP32
            && gate->dtype == DType::FP32 && up->dtype == DType::FP32
        ) {
            PtrPair w_gate = w_mlp_gate->ptr_all({layer_offset});
            PtrPair w_up = w_mlp_up->ptr_all({layer_offset});
            PtrPair w_rms_attn = rms_attn_w->ptr_all({layer_offset});

            const float *x_ptr = (const float *)x->ptr();
            float *t_ptr = (float *)t->ptr();
            float *gate_ptr = (float *)gate->ptr();
            float *up_ptr = (float *)up->ptr();
            
            size_t i = 0;
            for (; i < M; ++i) {
                fused_rms_mlp_swiglu_m1(
                    w_rms_attn, w_gate, w_up, x_ptr,
                    t_ptr, gate_ptr, up_ptr, hidden_size,
                    inter_dim, group_size, eps
                );
                x_ptr += (hidden_size);
                gate_ptr += (inter_dim);
            }
        } else {
            rms_norm(
                x, rms_attn_w, t, eps, M, layer_offset
            );

            PtrPair w_gate = w_mlp_gate->ptr_all({layer_offset});
            PtrPair w_up = w_mlp_up->ptr_all({layer_offset});
            linear(
                t->ptr(), w_gate.buf, w_gate.scale, w_gate.sum_int8,
                nullptr, nullptr, gate->ptr(), M, inter_dim,
                hidden_size, !w_mlp_gate->permuted, t->dtype,
                dtype_w, dtype_s, gate->dtype,
                text_gq, group_size, false
            );
            linear(
                t->ptr(), w_up.buf, w_up.scale, w_up.sum_int8, nullptr,
                nullptr, up->ptr(), M, inter_dim,
                hidden_size, !w_mlp_up->permuted, t->dtype,
                dtype_w, dtype_s, up->dtype, text_gq,
                group_size, false
            );
            
            swiglu(gate, up, M * inter_dim);
        }
    #else
    */
        rms_norm(
            x, rms_attn_w, t, eps, M, layer_offset
        );

        PtrPair w_gate = w_mlp_gate->ptr_all({layer_offset});
        PtrPair w_up = w_mlp_up->ptr_all({layer_offset});
        linear(
            t->ptr(), w_gate.buf, w_gate.scale, w_gate.sum_int8,
            nullptr, nullptr, gate->ptr(), M, inter_dim,
            hidden_size, !w_mlp_gate->permuted, t->dtype,
            dtype_w, dtype_s, gate->dtype,
            text_gq, group_size, false
        );
        linear(
            t->ptr(), w_up.buf, w_up.scale, w_up.sum_int8, nullptr,
            nullptr, up->ptr(), M, inter_dim,
            hidden_size, !w_mlp_up->permuted, t->dtype,
            dtype_w, dtype_s, up->dtype, text_gq,
            group_size, false
        );
        
        swiglu(gate, up, M * inter_dim);
    // #endif

    #ifdef PRINT_LOGITS
        if (!warm_up) {
            for (size_t i = 0; i < M; ++i) { 
                gate->printDebug("gate", {i});
            }
        }
    #endif
}

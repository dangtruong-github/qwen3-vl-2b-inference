#include "../include/cpu_int8_fp32s_avx2_kernel.hpp"

void linear_int8_fp32s(
    const float *mat_A,          /* [M, K] */
    const int8_t *mat_B_in,      /* [N, K] if !trans, else [K, N] */
    const float *mat_B_scales,   /* Scales for B */
    const int8_t *mat_bias_in,   /* [N] */
    const float *mat_bias_scale, /* Scales for bias */
    float *mat_C,                /* [M, N] */
    size_t M, size_t N, size_t K,
    bool mat_B_transpose,
    size_t group_size            /* e.g., 32 or 64 */
) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float accumulator = 0.0f;

            for (size_t k = 0; k < K; ++k) {
                float a_val = mat_A[i * K + k];
                
                // 1. Handle Weight (B) indexing and dequantization
                size_t b_idx, scale_idx;
                if (!mat_B_transpose) {
                    // Standard: B is [N, K]
                    b_idx = j * K + k;
                    scale_idx = b_idx / group_size;
                } else {
                    // Transposed: B is [K, N]
                    b_idx = k * N + j;
                    // Note: scales must match the memory layout of mat_B_in
                    scale_idx = b_idx / group_size;
                }

                float b_weight = static_cast<float>(mat_B_in[b_idx]) * mat_B_scales[scale_idx];
                accumulator += a_val * b_weight;
            }

            // 2. Handle Bias dequantization
            // Bias is usually [N], one scale per bias_group_size
            // Assuming bias uses the same or its own group_size logic
            float bias = 0.0f;
            if (mat_bias_in) {
                bias = static_cast<float>(mat_bias_in[j]) * mat_bias_scale[j / group_size];
            }

            mat_C[i * N + j] = accumulator + bias;
        }
    }
}
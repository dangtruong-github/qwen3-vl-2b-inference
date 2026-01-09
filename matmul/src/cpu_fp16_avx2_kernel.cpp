#include "../include/cpu_fp16_avx2_kernel.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void fp16_avx2_kernel(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C,  size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    // 1. Initialize mat_C with fp16 bias (expanded to fp32) or zeros
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (mat_bias != nullptr) {
                // F16C conversion happens here
                mat_C[i * N + j] = static_cast<float>(mat_bias[j]);
            } else {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }

    // 2. Compute Matrix Multiplication
    if (mat_B_transpose) {
        // A: [M, K], B: [N, K] (Stored as N rows of length K)
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = mat_C[i * N + j];
                const float* row_A = &mat_A[i * K];
                const half_cpu* row_B = &mat_B[j * K];

                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < K; ++k) {
                    // Both inputs are now in registers as fp32
                    sum += row_A[k] * static_cast<float>(row_B[k]);
                }
                mat_C[i * N + j] = sum;
            }
        }
    } else {
        // B is [K, N]. Standard GEMM.
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = mat_C[i * N + j];
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < K; ++k) {
                    sum += mat_A[i * K + k] * static_cast<float>(mat_B[k * N + j]);
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
}
#endif

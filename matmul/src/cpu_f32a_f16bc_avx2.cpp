#include "../include/cpu_f32a_f16bc_avx2.hpp"

// #if defined(__AVX2__) && defined(__FMA__)
void f32a_f16bc_avx2_kernel(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    half_cpu *__restrict mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = mat_bias ? (float)mat_bias[j] : 0.0f;

            const float *A_row = mat_A + i * K;

            if (!mat_B_transpose) {
                // B is (K, N)
                for (size_t k = 0; k < K; ++k) {
                    float a = A_row[k];
                    float b = (float)mat_B[k * N + j];
                    sum += a * b;
                }
            } else {
                // B is (N, K)
                const half_cpu *B_row = mat_B + j * K;
                for (size_t k = 0; k < K; ++k) {
                    float a = A_row[k];
                    float b = (float)B_row[k];
                    sum += a * b;
                }
            }

            mat_C[i * N + j] = (half_cpu)sum;
        }
    }
}
// #endif
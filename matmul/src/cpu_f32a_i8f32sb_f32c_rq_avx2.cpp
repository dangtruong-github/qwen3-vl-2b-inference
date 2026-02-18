#include "../include/cpu_wrapper.hpp"

// #if defined(__AVX2__) && defined(__FMA__)
void f32a_i8f32sb_f32c_rq_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            float scale = mat_B_scales[j];

            // -------- GEMM --------
            for (size_t k = 0; k < K; ++k) {
                float a = mat_A[i * K + k];

                // linear index into B (matches quantizer layout)
                size_t b_linear_idx;
                b_linear_idx = k * N + j;

                float b = (float)mat_B_in[b_linear_idx] * scale;
                acc += a * b;
            }

            mat_C[i * N + j] = acc;
        }
    }
}
// #endif
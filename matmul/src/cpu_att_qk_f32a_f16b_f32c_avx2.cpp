#include "../include/cpu_wrapper.hpp"

// #if defined(__AVX2__) && defined(__FMA__)
void qk_att_m2_k128_avx2(
    const float *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t N, const size_t max_pos
) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < max_pos; ++j) {

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        const half_cpu *mat_B_ptr = mat_B + (j << 7);

        for (size_t k = 0; k < 128; k += 8) {
            __m256 a0 = _mm256_loadu_ps(mat_A + k);
            __m256 a1 = _mm256_loadu_ps(mat_A + 128 + k);
            __m128i b_vec_fp16 = _mm_loadu_si128((const __m128i*)(mat_B_ptr + k));
            __m256 b_vec = _mm256_cvtph_ps(b_vec_fp16);

            sum0 = _mm256_fmadd_ps(a0, b_vec, sum0);
            sum1 = _mm256_fmadd_ps(a1, b_vec, sum1);
        }

        mat_C[j] = add_reduce_mm_256(sum0) * scale;
        mat_C[N + j] = add_reduce_mm_256(sum1) * scale;
    }
}

void qk_att_f32a_f16b_f32c_avx2_wrapper(
    const float *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N,
    size_t K, const size_t max_pos
) {
    if (M == 2 && K == 128) {
        qk_att_m2_k128_avx2(mat_A, mat_B, mat_C, scale, N, max_pos);
        return;
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < max_pos; ++j) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k) {
                float a = mat_A[i * K + k];
                float b = (float)(mat_B[j * K + k]);

                sum += a * b;
            }

            mat_C[i * N + j] = sum * scale;
        }
    }
}
// #endif
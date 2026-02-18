#include "../include/cpu_wrapper.hpp"

// #if defined(__AVX2__) && defined(__FMA__)
void gemm_att_transpose_k64(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N
) {
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        const float *b = mat_B + n * 64;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // 64 = 8 * 8, unrolled
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A +  0),
                               _mm256_loadu_ps(b     +  0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A +  8),
                               _mm256_loadu_ps(b     +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 16),
                               _mm256_loadu_ps(b     + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 24),
                               _mm256_loadu_ps(b     + 24), acc3);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 32),
                               _mm256_loadu_ps(b     + 32), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 40),
                               _mm256_loadu_ps(b     + 40), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 48),
                               _mm256_loadu_ps(b     + 48), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 56),
                               _mm256_loadu_ps(b     + 56), acc3);

        // Reduce accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // Horizontal reduction
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        mat_C[n] = _mm_cvtss_f32(sum) * scale;
    }
}

void gemm_att_transpose_k64_s1(
    const float *mat_A, const float *mat_B, float *mat_C, size_t N
) {
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        const float *b = mat_B + n * 64;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // 64 = 8 * 8, unrolled
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A +  0),
                               _mm256_loadu_ps(b     +  0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A +  8),
                               _mm256_loadu_ps(b     +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 16),
                               _mm256_loadu_ps(b     + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 24),
                               _mm256_loadu_ps(b     + 24), acc3);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 32),
                               _mm256_loadu_ps(b     + 32), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 40),
                               _mm256_loadu_ps(b     + 40), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 48),
                               _mm256_loadu_ps(b     + 48), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(mat_A + 56),
                               _mm256_loadu_ps(b     + 56), acc3);

        // Reduce accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        // Horizontal reduction
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        mat_C[n] = _mm_cvtss_f32(sum);
    }
}

void gemm_att_n64(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t K
) {
    
    __m256 acc[8];
    for (int i = 0; i < 8; i++)
        acc[i] = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 a = _mm256_broadcast_ss(mat_A + k);

        const float *bk = mat_B + k * 64;

        acc[0] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk +  0), acc[0]);
        acc[1] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk +  8), acc[1]);
        acc[2] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 16), acc[2]);
        acc[3] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 24), acc[3]);
        acc[4] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 32), acc[4]);
        acc[5] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 40), acc[5]);
        acc[6] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 48), acc[6]);
        acc[7] = _mm256_fmadd_ps(a, _mm256_loadu_ps(bk + 56), acc[7]);
    }

    __m256 vscale = _mm256_set1_ps(scale);
    for (int i = 0; i < 8; i++)
        _mm256_storeu_ps(mat_C + i * 8,
                         _mm256_mul_ps(acc[i], vscale));
}

void att_fp32_full_avx2_kernel(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose
) {
    if (mat_B_transpose) {
        if (K == 64) {
            if (scale == 1.0f) {
                gemm_att_transpose_k64_s1(mat_A, mat_B, mat_C, N);
            } else {
                gemm_att_transpose_k64(mat_A, mat_B, mat_C, scale, N);
            }
            return;
        }
    } else {
        if (N == 64) {
            gemm_att_n64(mat_A, mat_B, mat_C, scale, K);
            return;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        float sum = 0.0f;

        for (size_t k = 0; k < K; ++k) {
            if (!mat_B_transpose) {
                // B: (K, N)
                sum += mat_A[k] * mat_B[k * N + n];
            } else {
                // B: (N, K)
                sum += mat_A[k] * mat_B[n * K + k];
            }
        }

        sum *= scale;
        mat_C[n] = sum;
    }
}
// #endif
#include "../include/matmul_cpu.hpp"

static inline float add_reduce_mm_256(__m256 vec) {
    // Step 1: Split into two 128-bit halves
    __m128 low  = _mm256_castps256_ps128(vec);          // lower 128 bits
    __m128 high = _mm256_extractf128_ps(vec, 1);        // upper 128 bits

    // Step 2: Add the halves together
    __m128 sum128 = _mm_add_ps(low, high);

    // Step 3: Horizontal add within 128-bit register
    sum128 = _mm_hadd_ps(sum128, sum128);               // pairwise add
    sum128 = _mm_hadd_ps(sum128, sum128);               // final reduction

    // Step 4: Extract scalar result
    return _mm_cvtss_f32(sum128);
}

void lg_M_N_K_transpose(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TM = 32;   // tile size for rows of A / C
    constexpr size_t TN = 64;   // tile size for columns of C / rows of B
    constexpr size_t threads_each = 32;

    // B is N x K, stored as B^T row-major: mat_B[j * K + k]
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads_each)
    for (size_t ii = 0; ii < M; ii += TM) {
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t i_end = std::min(ii + TM, M);
            size_t j_end = std::min(jj + TN, N);

            size_t i = ii;
            for (; i + 2 <= i_end; i += 2) {
                const float *a0_ptr = mat_A + i * K;
                const float *a1_ptr = mat_A + (i+1) * K;

                size_t j = jj;

                // ---- vectorized j loop (4 columns at once) ----
                for (; j + 4 <= j_end; j += 4) {
                    const float *b0_ptr = mat_B + (j + 0) * K;
                    const float *b1_ptr = mat_B + (j + 1) * K;
                    const float *b2_ptr = mat_B + (j + 2) * K;
                    const float *b3_ptr = mat_B + (j + 3) * K;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();

                    size_t k = 0;

                    // ---- vectorized k loop ----
                    for (; k + 8 <= K; k += 8) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a00_vec = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a10_vec = _mm256_loadu_ps(a1_ptr + k);
                        __m256 b020_vec = _mm256_loadu_ps(b0_ptr + k);
                        __m256 b130_vec = _mm256_loadu_ps(b1_ptr + k);

                        // ---- block 0 (k..k+7) ----
                        c00 = _mm256_fmadd_ps(a00_vec, b020_vec, c00);
                        c01 = _mm256_fmadd_ps(a00_vec, b130_vec, c01);
                        c10 = _mm256_fmadd_ps(a10_vec, b020_vec, c10);
                        c11 = _mm256_fmadd_ps(a10_vec, b130_vec, c11);

                        b020_vec = _mm256_loadu_ps(b2_ptr + k);
                        b130_vec = _mm256_loadu_ps(b3_ptr + k);
                        
                        c02 = _mm256_fmadd_ps(a00_vec, b020_vec, c02);
                        c03 = _mm256_fmadd_ps(a00_vec, b130_vec, c03);
                        c12 = _mm256_fmadd_ps(a10_vec, b020_vec, c12);
                        c13 = _mm256_fmadd_ps(a10_vec, b130_vec, c13);
                    }

                    // reduce SIMD accumulators
                    float sum00 = add_reduce_mm_256(c00);
                    float sum01 = add_reduce_mm_256(c01);
                    float sum02 = add_reduce_mm_256(c02);
                    float sum03 = add_reduce_mm_256(c03);
                    float sum10 = add_reduce_mm_256(c10);
                    float sum11 = add_reduce_mm_256(c11);
                    float sum12 = add_reduce_mm_256(c12);
                    float sum13 = add_reduce_mm_256(c13);

                    // ---- scalar k cleanup ----
                    for (; k < K; ++k) {
                        float a0 = a0_ptr[k];
                        float a1 = a1_ptr[k];
                        float b0 = b0_ptr[k];
                        float b1 = b1_ptr[k];
                        float b2 = b2_ptr[k];
                        float b3 = b3_ptr[k];
                        sum00 += a0 * b0;
                        sum01 += a0 * b1;
                        sum02 += a0 * b2;
                        sum03 += a0 * b3;
                        sum10 += a1 * b0;
                        sum11 += a1 * b1;
                        sum12 += a1 * b2;
                        sum13 += a1 * b3;
                    }

                    mat_C[i * N + j + 0] = sum00;
                    mat_C[i * N + j + 1] = sum01;
                    mat_C[i * N + j + 2] = sum02;
                    mat_C[i * N + j + 3] = sum03;
                    mat_C[(i+1) * N + j + 0] = sum10;
                    mat_C[(i+1) * N + j + 1] = sum11;
                    mat_C[(i+1) * N + j + 2] = sum12;
                    mat_C[(i+1) * N + j + 3] = sum13;
                }

                // ---- scalar j cleanup ----
                for (; j < j_end; ++j) {
                    const float *mat_B_ptr = mat_B + j * K;

                    // Initialize accumulators
                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 8 <= K; k += 8) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a0_vec = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1_vec = _mm256_loadu_ps(a1_ptr + k);
                        __m256 b_vec = _mm256_loadu_ps(mat_B_ptr + k);

                        // ---- block 0 (k..k+7) ----
                        c00 = _mm256_fmadd_ps(a0_vec, b_vec, c00);
                        c10 = _mm256_fmadd_ps(a1_vec, b_vec, c10);
                    }

                    float sum00 = add_reduce_mm_256(c00);
                    float sum10 = add_reduce_mm_256(c10);

                    // Handle leftover elements (< 32)
                    for (; k < K; ++k) {
                        float a0 = a0_ptr[k];
                        float a1 = a1_ptr[k];
                        float b0 = mat_B_ptr[k];
                        sum00 += a0 * b0;
                        sum10 += a1 * b0;
                    }

                    mat_C[i * N + j] = sum00;
                    mat_C[(i+1) * N + j] = sum10;
                }
            }
        
            for (; i < i_end; ++i) {

                const float *mat_A_ptr = mat_A + i * K;

                size_t j = jj;

                // ---- vectorized j loop (4 columns at once) ----
                for (; j + 4 <= j_end; j += 4) {
                    const float *b0_ptr = mat_B + (j + 0) * K;
                    const float *b1_ptr = mat_B + (j + 1) * K;
                    const float *b2_ptr = mat_B + (j + 2) * K;
                    const float *b3_ptr = mat_B + (j + 3) * K;

                    __m256 c0 = _mm256_setzero_ps();
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();

                    size_t k = 0;

                    // ---- vectorized k loop ----
                    for (; k + 64 <= K; k += 64) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a0_vec = _mm256_loadu_ps(mat_A_ptr + k);
                        __m256 a1_vec = _mm256_loadu_ps(mat_A_ptr + k + 8);
                        __m256 a2_vec = _mm256_loadu_ps(mat_A_ptr + k + 16);
                        __m256 a3_vec = _mm256_loadu_ps(mat_A_ptr + k + 24);
                        __m256 a4_vec = _mm256_loadu_ps(mat_A_ptr + k + 32);
                        __m256 a5_vec = _mm256_loadu_ps(mat_A_ptr + k + 40);
                        __m256 a6_vec = _mm256_loadu_ps(mat_A_ptr + k + 48);
                        __m256 a7_vec = _mm256_loadu_ps(mat_A_ptr + k + 56);

                        // ---- block 0 (k..k+7) ----
                        c0 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b0_ptr + k), c0);
                        c1 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b1_ptr + k), c1);
                        c2 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b2_ptr + k), c2);
                        c3 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b3_ptr + k), c3);

                        // ---- block 1 (k+8..k+15) ----
                        c0 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b0_ptr + k + 8), c0);
                        c1 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b1_ptr + k + 8), c1);
                        c2 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b2_ptr + k + 8), c2);
                        c3 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b3_ptr + k + 8), c3);

                        // ---- block 2 (k+16..k+23) ----
                        c0 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b0_ptr + k + 16), c0);
                        c1 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b1_ptr + k + 16), c1);
                        c2 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b2_ptr + k + 16), c2);
                        c3 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b3_ptr + k + 16), c3);

                        // ---- block 3 (k+24..k+31) ----
                        c0 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b0_ptr + k + 24), c0);
                        c1 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b1_ptr + k + 24), c1);
                        c2 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b2_ptr + k + 24), c2);
                        c3 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b3_ptr + k + 24), c3);

                        // ---- block 4 (k+32..k+39) ----
                        c0 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b0_ptr + k + 32), c0);
                        c1 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b1_ptr + k + 32), c1);
                        c2 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b2_ptr + k + 32), c2);
                        c3 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b3_ptr + k + 32), c3);

                        // ---- block 5 (k+40..k+47) ----
                        c0 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b0_ptr + k + 40), c0);
                        c1 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b1_ptr + k + 40), c1);
                        c2 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b2_ptr + k + 40), c2);
                        c3 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b3_ptr + k + 40), c3);

                        // ---- block 6 (k+48..k+55) ----
                        c0 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b0_ptr + k + 48), c0);
                        c1 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b1_ptr + k + 48), c1);
                        c2 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b2_ptr + k + 48), c2);
                        c3 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b3_ptr + k + 48), c3);

                        // ---- block 7 (k+56..k+63) ----
                        c0 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b0_ptr + k + 56), c0);
                        c1 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b1_ptr + k + 56), c1);
                        c2 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b2_ptr + k + 56), c2);
                        c3 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b3_ptr + k + 56), c3);
                    }

                    // reduce SIMD accumulators
                    float sum0 = add_reduce_mm_256(c0);
                    float sum1 = add_reduce_mm_256(c1);
                    float sum2 = add_reduce_mm_256(c2);
                    float sum3 = add_reduce_mm_256(c3);

                    // ---- scalar k cleanup ----
                    for (; k < K; ++k) {
                        float a = mat_A_ptr[k];
                        sum0 += a * b0_ptr[k];
                        sum1 += a * b1_ptr[k];
                        sum2 += a * b2_ptr[k];
                        sum3 += a * b3_ptr[k];
                    }

                    mat_C[i * N + j + 0] = sum0;
                    mat_C[i * N + j + 1] = sum1;
                    mat_C[i * N + j + 2] = sum2;
                    mat_C[i * N + j + 3] = sum3;
                }

                // ---- scalar j cleanup ----
                for (; j < j_end; ++j) {
                    const float *mat_A_ptr = mat_A + i * K;
                    const float *mat_B_ptr = mat_B + j * K;

                    // Initialize accumulators
                    __m256 c0 = _mm256_setzero_ps();
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 32 <= K; k += 32) {
                        // First block
                        __m256 a0 = _mm256_loadu_ps(mat_A_ptr + k);
                        __m256 b0 = _mm256_loadu_ps(mat_B_ptr + k);
                        c0 = _mm256_fmadd_ps(a0, b0, c0);

                        // Second block
                        __m256 a1 = _mm256_loadu_ps(mat_A_ptr + k + 8);
                        __m256 b1 = _mm256_loadu_ps(mat_B_ptr + k + 8);
                        c1 = _mm256_fmadd_ps(a1, b1, c1);

                        // Third block
                        __m256 a2 = _mm256_loadu_ps(mat_A_ptr + k + 16);
                        __m256 b2 = _mm256_loadu_ps(mat_B_ptr + k + 16);
                        c2 = _mm256_fmadd_ps(a2, b2, c2);

                        // Fourth block
                        __m256 a3 = _mm256_loadu_ps(mat_A_ptr + k + 24);
                        __m256 b3 = _mm256_loadu_ps(mat_B_ptr + k + 24);
                        c3 = _mm256_fmadd_ps(a3, b3, c3);
                    }

                    float sum = add_reduce_mm_256(
                        _mm256_add_ps(_mm256_add_ps(c0, c1),
                                    _mm256_add_ps(c2, c3))
                    );

                    // Handle leftover elements (< 32)
                    #pragma omp simd reduction(+:sum)
                    for (size_t k_t = k; k_t < K; ++k_t) {
                        sum += mat_A_ptr[k_t] * mat_B_ptr[k_t];
                    }

                    mat_C[i * N + j] = sum;
                }
            }
        }
    }
}

void lg_M_N_sm_K_transpose(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TM = 32;   // tile size for rows of A / C
    constexpr size_t TN = 64;   // tile size for columns of C / rows of B

    // B is N x K, stored as B^T row-major: mat_B[j * K + k]
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < M; ii += TM) {
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t i_end = std::min(ii + TM, M);
            size_t j_end = std::min(jj + TN, N);

            size_t i = ii;
            for (; i + 2 <= i_end; i += 2) {
                const float *a0_ptr = mat_A + i * K;
                const float *a1_ptr = mat_A + (i+1) * K;

                size_t j = jj;

                // ---- vectorized j loop (4 columns at once) ----
                for (; j + 4 <= j_end; j += 4) {
                    const float *b0_ptr = mat_B + (j + 0) * K;
                    const float *b1_ptr = mat_B + (j + 1) * K;
                    const float *b2_ptr = mat_B + (j + 2) * K;
                    const float *b3_ptr = mat_B + (j + 3) * K;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();

                    size_t k = 0;

                    // ---- vectorized k loop ----
                    for (; k + 8 <= K; k += 8) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a00_vec = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a10_vec = _mm256_loadu_ps(a1_ptr + k);
                        __m256 b020_vec = _mm256_loadu_ps(b0_ptr + k);
                        __m256 b130_vec = _mm256_loadu_ps(b1_ptr + k);

                        // ---- block 0 (k..k+7) ----
                        c00 = _mm256_fmadd_ps(a00_vec, b020_vec, c00);
                        c01 = _mm256_fmadd_ps(a00_vec, b130_vec, c01);
                        c10 = _mm256_fmadd_ps(a10_vec, b020_vec, c10);
                        c11 = _mm256_fmadd_ps(a10_vec, b130_vec, c11);

                        b020_vec = _mm256_loadu_ps(b2_ptr + k);
                        b130_vec = _mm256_loadu_ps(b3_ptr + k);
                        
                        c02 = _mm256_fmadd_ps(a00_vec, b020_vec, c02);
                        c03 = _mm256_fmadd_ps(a00_vec, b130_vec, c03);
                        c12 = _mm256_fmadd_ps(a10_vec, b020_vec, c12);
                        c13 = _mm256_fmadd_ps(a10_vec, b130_vec, c13);
                    }

                    // reduce SIMD accumulators
                    float sum00 = add_reduce_mm_256(c00);
                    float sum01 = add_reduce_mm_256(c01);
                    float sum02 = add_reduce_mm_256(c02);
                    float sum03 = add_reduce_mm_256(c03);
                    float sum10 = add_reduce_mm_256(c10);
                    float sum11 = add_reduce_mm_256(c11);
                    float sum12 = add_reduce_mm_256(c12);
                    float sum13 = add_reduce_mm_256(c13);

                    // ---- scalar k cleanup ----
                    for (; k < K; ++k) {
                        float a0 = a0_ptr[k];
                        float a1 = a1_ptr[k];
                        float b0 = b0_ptr[k];
                        float b1 = b1_ptr[k];
                        float b2 = b2_ptr[k];
                        float b3 = b3_ptr[k];
                        sum00 += a0 * b0;
                        sum01 += a0 * b1;
                        sum02 += a0 * b2;
                        sum03 += a0 * b3;
                        sum10 += a1 * b0;
                        sum11 += a1 * b1;
                        sum12 += a1 * b2;
                        sum13 += a1 * b3;
                    }

                    mat_C[i * N + j + 0] = sum00;
                    mat_C[i * N + j + 1] = sum01;
                    mat_C[i * N + j + 2] = sum02;
                    mat_C[i * N + j + 3] = sum03;
                    mat_C[(i+1) * N + j + 0] = sum10;
                    mat_C[(i+1) * N + j + 1] = sum11;
                    mat_C[(i+1) * N + j + 2] = sum12;
                    mat_C[(i+1) * N + j + 3] = sum13;
                }

                // ---- scalar j cleanup ----
                for (; j < j_end; ++j) {
                    const float *mat_B_ptr = mat_B + j * K;

                    // Initialize accumulators
                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 8 <= K; k += 8) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a0_vec = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1_vec = _mm256_loadu_ps(a1_ptr + k);
                        __m256 b_vec = _mm256_loadu_ps(mat_B_ptr + k);

                        // ---- block 0 (k..k+7) ----
                        c00 = _mm256_fmadd_ps(a0_vec, b_vec, c00);
                        c10 = _mm256_fmadd_ps(a1_vec, b_vec, c10);
                    }

                    float sum00 = add_reduce_mm_256(c00);
                    float sum10 = add_reduce_mm_256(c10);

                    // Handle leftover elements (< 32)
                    for (; k < K; ++k) {
                        float a0 = a0_ptr[k];
                        float a1 = a1_ptr[k];
                        float b0 = mat_B_ptr[k];
                        sum00 += a0 * b0;
                        sum10 += a1 * b0;
                    }

                    mat_C[i * N + j] = sum00;
                    mat_C[(i+1) * N + j] = sum10;
                }
            }
        
            for (; i < i_end; ++i) {

                const float *mat_A_ptr = mat_A + i * K;

                size_t j = jj;

                // ---- vectorized j loop (4 columns at once) ----
                for (; j + 4 <= j_end; j += 4) {
                    const float *b0_ptr = mat_B + (j + 0) * K;
                    const float *b1_ptr = mat_B + (j + 1) * K;
                    const float *b2_ptr = mat_B + (j + 2) * K;
                    const float *b3_ptr = mat_B + (j + 3) * K;

                    __m256 c0 = _mm256_setzero_ps();
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();

                    size_t k = 0;

                    // ---- vectorized k loop ----
                    for (; k + 64 <= K; k += 64) {
                        // Load 8 blocks of 8 floats from A row
                        __m256 a0_vec = _mm256_loadu_ps(mat_A_ptr + k);
                        __m256 a1_vec = _mm256_loadu_ps(mat_A_ptr + k + 8);
                        __m256 a2_vec = _mm256_loadu_ps(mat_A_ptr + k + 16);
                        __m256 a3_vec = _mm256_loadu_ps(mat_A_ptr + k + 24);
                        __m256 a4_vec = _mm256_loadu_ps(mat_A_ptr + k + 32);
                        __m256 a5_vec = _mm256_loadu_ps(mat_A_ptr + k + 40);
                        __m256 a6_vec = _mm256_loadu_ps(mat_A_ptr + k + 48);
                        __m256 a7_vec = _mm256_loadu_ps(mat_A_ptr + k + 56);

                        // ---- block 0 (k..k+7) ----
                        c0 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b0_ptr + k), c0);
                        c1 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b1_ptr + k), c1);
                        c2 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b2_ptr + k), c2);
                        c3 = _mm256_fmadd_ps(a0_vec, _mm256_loadu_ps(b3_ptr + k), c3);

                        // ---- block 1 (k+8..k+15) ----
                        c0 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b0_ptr + k + 8), c0);
                        c1 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b1_ptr + k + 8), c1);
                        c2 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b2_ptr + k + 8), c2);
                        c3 = _mm256_fmadd_ps(a1_vec, _mm256_loadu_ps(b3_ptr + k + 8), c3);

                        // ---- block 2 (k+16..k+23) ----
                        c0 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b0_ptr + k + 16), c0);
                        c1 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b1_ptr + k + 16), c1);
                        c2 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b2_ptr + k + 16), c2);
                        c3 = _mm256_fmadd_ps(a2_vec, _mm256_loadu_ps(b3_ptr + k + 16), c3);

                        // ---- block 3 (k+24..k+31) ----
                        c0 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b0_ptr + k + 24), c0);
                        c1 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b1_ptr + k + 24), c1);
                        c2 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b2_ptr + k + 24), c2);
                        c3 = _mm256_fmadd_ps(a3_vec, _mm256_loadu_ps(b3_ptr + k + 24), c3);

                        // ---- block 4 (k+32..k+39) ----
                        c0 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b0_ptr + k + 32), c0);
                        c1 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b1_ptr + k + 32), c1);
                        c2 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b2_ptr + k + 32), c2);
                        c3 = _mm256_fmadd_ps(a4_vec, _mm256_loadu_ps(b3_ptr + k + 32), c3);

                        // ---- block 5 (k+40..k+47) ----
                        c0 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b0_ptr + k + 40), c0);
                        c1 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b1_ptr + k + 40), c1);
                        c2 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b2_ptr + k + 40), c2);
                        c3 = _mm256_fmadd_ps(a5_vec, _mm256_loadu_ps(b3_ptr + k + 40), c3);

                        // ---- block 6 (k+48..k+55) ----
                        c0 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b0_ptr + k + 48), c0);
                        c1 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b1_ptr + k + 48), c1);
                        c2 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b2_ptr + k + 48), c2);
                        c3 = _mm256_fmadd_ps(a6_vec, _mm256_loadu_ps(b3_ptr + k + 48), c3);

                        // ---- block 7 (k+56..k+63) ----
                        c0 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b0_ptr + k + 56), c0);
                        c1 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b1_ptr + k + 56), c1);
                        c2 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b2_ptr + k + 56), c2);
                        c3 = _mm256_fmadd_ps(a7_vec, _mm256_loadu_ps(b3_ptr + k + 56), c3);
                    }

                    // reduce SIMD accumulators
                    float sum0 = add_reduce_mm_256(c0);
                    float sum1 = add_reduce_mm_256(c1);
                    float sum2 = add_reduce_mm_256(c2);
                    float sum3 = add_reduce_mm_256(c3);

                    // ---- scalar k cleanup ----
                    for (; k < K; ++k) {
                        float a = mat_A_ptr[k];
                        sum0 += a * b0_ptr[k];
                        sum1 += a * b1_ptr[k];
                        sum2 += a * b2_ptr[k];
                        sum3 += a * b3_ptr[k];
                    }

                    mat_C[i * N + j + 0] = sum0;
                    mat_C[i * N + j + 1] = sum1;
                    mat_C[i * N + j + 2] = sum2;
                    mat_C[i * N + j + 3] = sum3;
                }

                // ---- scalar j cleanup ----
                for (; j < j_end; ++j) {
                    const float *mat_A_ptr = mat_A + i * K;
                    const float *mat_B_ptr = mat_B + j * K;

                    // Initialize accumulators
                    __m256 c0 = _mm256_setzero_ps();
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 32 <= K; k += 32) {
                        // First block
                        __m256 a0 = _mm256_loadu_ps(mat_A_ptr + k);
                        __m256 b0 = _mm256_loadu_ps(mat_B_ptr + k);
                        c0 = _mm256_fmadd_ps(a0, b0, c0);

                        // Second block
                        __m256 a1 = _mm256_loadu_ps(mat_A_ptr + k + 8);
                        __m256 b1 = _mm256_loadu_ps(mat_B_ptr + k + 8);
                        c1 = _mm256_fmadd_ps(a1, b1, c1);

                        // Third block
                        __m256 a2 = _mm256_loadu_ps(mat_A_ptr + k + 16);
                        __m256 b2 = _mm256_loadu_ps(mat_B_ptr + k + 16);
                        c2 = _mm256_fmadd_ps(a2, b2, c2);

                        // Fourth block
                        __m256 a3 = _mm256_loadu_ps(mat_A_ptr + k + 24);
                        __m256 b3 = _mm256_loadu_ps(mat_B_ptr + k + 24);
                        c3 = _mm256_fmadd_ps(a3, b3, c3);
                    }

                    float sum = add_reduce_mm_256(
                        _mm256_add_ps(_mm256_add_ps(c0, c1),
                                    _mm256_add_ps(c2, c3))
                    );

                    // Handle leftover elements (< 32)
                    #pragma omp simd reduction(+:sum)
                    for (size_t k_t = k; k_t < K; ++k_t) {
                        sum += mat_A_ptr[k_t] * mat_B_ptr[k_t];
                    }

                    mat_C[i * N + j] = sum;
                }
            }
        }
    }
}

void sm_M_lg_N_K_transpose(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    const size_t j_end_stride = N / 4 * 4;

    // B is N x K, stored as B^T row-major: mat_B[j * K + k]
    for (size_t i = 0; i < M; ++i) {
        const float *mat_A_ptr = mat_A + i * K;

        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < j_end_stride; j += 4) {
            const float *b0_ptr = mat_B + j * K;
            const float *b1_ptr = mat_B + (j+1) * K;
            const float *b2_ptr = mat_B + (j+2) * K;
            const float *b3_ptr = mat_B + (j+3) * K;

            // Initialize accumulators
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 64 <= K; k += 64) {
                // Load 8 blocks of 8 floats from A
                __m256 a_vec0 = _mm256_loadu_ps(mat_A_ptr + k);
                __m256 a_vec1 = _mm256_loadu_ps(mat_A_ptr + k + 8);
                __m256 a_vec2 = _mm256_loadu_ps(mat_A_ptr + k + 16);
                __m256 a_vec3 = _mm256_loadu_ps(mat_A_ptr + k + 24);
                __m256 a_vec4 = _mm256_loadu_ps(mat_A_ptr + k + 32);
                __m256 a_vec5 = _mm256_loadu_ps(mat_A_ptr + k + 40);
                __m256 a_vec6 = _mm256_loadu_ps(mat_A_ptr + k + 48);
                __m256 a_vec7 = _mm256_loadu_ps(mat_A_ptr + k + 56);

                // Accumulate using a_vec0
                c0 = _mm256_fmadd_ps(a_vec0, _mm256_loadu_ps(b0_ptr + k), c0);
                c1 = _mm256_fmadd_ps(a_vec0, _mm256_loadu_ps(b1_ptr + k), c1);
                c2 = _mm256_fmadd_ps(a_vec0, _mm256_loadu_ps(b2_ptr + k), c2);
                c3 = _mm256_fmadd_ps(a_vec0, _mm256_loadu_ps(b3_ptr + k), c3);

                // Accumulate using a_vec1
                c0 = _mm256_fmadd_ps(a_vec1, _mm256_loadu_ps(b0_ptr + k + 8), c0);
                c1 = _mm256_fmadd_ps(a_vec1, _mm256_loadu_ps(b1_ptr + k + 8), c1);
                c2 = _mm256_fmadd_ps(a_vec1, _mm256_loadu_ps(b2_ptr + k + 8), c2);
                c3 = _mm256_fmadd_ps(a_vec1, _mm256_loadu_ps(b3_ptr + k + 8), c3);

                // Accumulate using a_vec2
                c0 = _mm256_fmadd_ps(a_vec2, _mm256_loadu_ps(b0_ptr + k + 16), c0);
                c1 = _mm256_fmadd_ps(a_vec2, _mm256_loadu_ps(b1_ptr + k + 16), c1);
                c2 = _mm256_fmadd_ps(a_vec2, _mm256_loadu_ps(b2_ptr + k + 16), c2);
                c3 = _mm256_fmadd_ps(a_vec2, _mm256_loadu_ps(b3_ptr + k + 16), c3);

                // Accumulate using a_vec3
                c0 = _mm256_fmadd_ps(a_vec3, _mm256_loadu_ps(b0_ptr + k + 24), c0);
                c1 = _mm256_fmadd_ps(a_vec3, _mm256_loadu_ps(b1_ptr + k + 24), c1);
                c2 = _mm256_fmadd_ps(a_vec3, _mm256_loadu_ps(b2_ptr + k + 24), c2);
                c3 = _mm256_fmadd_ps(a_vec3, _mm256_loadu_ps(b3_ptr + k + 24), c3);

                // Accumulate using a_vec4
                c0 = _mm256_fmadd_ps(a_vec4, _mm256_loadu_ps(b0_ptr + k + 32), c0);
                c1 = _mm256_fmadd_ps(a_vec4, _mm256_loadu_ps(b1_ptr + k + 32), c1);
                c2 = _mm256_fmadd_ps(a_vec4, _mm256_loadu_ps(b2_ptr + k + 32), c2);
                c3 = _mm256_fmadd_ps(a_vec4, _mm256_loadu_ps(b3_ptr + k + 32), c3);

                // Accumulate using a_vec5
                c0 = _mm256_fmadd_ps(a_vec5, _mm256_loadu_ps(b0_ptr + k + 40), c0);
                c1 = _mm256_fmadd_ps(a_vec5, _mm256_loadu_ps(b1_ptr + k + 40), c1);
                c2 = _mm256_fmadd_ps(a_vec5, _mm256_loadu_ps(b2_ptr + k + 40), c2);
                c3 = _mm256_fmadd_ps(a_vec5, _mm256_loadu_ps(b3_ptr + k + 40), c3);

                // Accumulate using a_vec6
                c0 = _mm256_fmadd_ps(a_vec6, _mm256_loadu_ps(b0_ptr + k + 48), c0);
                c1 = _mm256_fmadd_ps(a_vec6, _mm256_loadu_ps(b1_ptr + k + 48), c1);
                c2 = _mm256_fmadd_ps(a_vec6, _mm256_loadu_ps(b2_ptr + k + 48), c2);
                c3 = _mm256_fmadd_ps(a_vec6, _mm256_loadu_ps(b3_ptr + k + 48), c3);

                // Accumulate using a_vec7
                c0 = _mm256_fmadd_ps(a_vec7, _mm256_loadu_ps(b0_ptr + k + 56), c0);
                c1 = _mm256_fmadd_ps(a_vec7, _mm256_loadu_ps(b1_ptr + k + 56), c1);
                c2 = _mm256_fmadd_ps(a_vec7, _mm256_loadu_ps(b2_ptr + k + 56), c2);
                c3 = _mm256_fmadd_ps(a_vec7, _mm256_loadu_ps(b3_ptr + k + 56), c3);
            }

            float sum0 = add_reduce_mm_256(c0);
            float sum1 = add_reduce_mm_256(c1);
            float sum2 = add_reduce_mm_256(c2);
            float sum3 = add_reduce_mm_256(c3);

            // Handle leftover elements (< 32)
            for (; k < K; ++K) {
                const float a = mat_A_ptr[k];
                sum0 += a * b0_ptr[k];
                sum1 += a * b1_ptr[k];
                sum2 += a * b2_ptr[k];
                sum3 += a * b3_ptr[k];
            }

            mat_C[i * N + j] = sum0;
            mat_C[i * N + j + 1] = sum1;
            mat_C[i * N + j + 2] = sum2;
            mat_C[i * N + j + 3] = sum3;
        }
    
        #pragma omp parallel for
        for (size_t j = j_end_stride; j < N; ++j) {
            const float *mat_B_ptr = mat_B + j * K;

            // Initialize accumulators
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            size_t k = 0;
            for (; k + 32 <= K; k += 32) {
                // First block
                __m256 a0 = _mm256_loadu_ps(mat_A_ptr + k);
                __m256 b0 = _mm256_loadu_ps(mat_B_ptr + k);
                c0 = _mm256_fmadd_ps(a0, b0, c0);

                // Second block
                __m256 a1 = _mm256_loadu_ps(mat_A_ptr + k + 8);
                __m256 b1 = _mm256_loadu_ps(mat_B_ptr + k + 8);
                c1 = _mm256_fmadd_ps(a1, b1, c1);

                // Third block
                __m256 a2 = _mm256_loadu_ps(mat_A_ptr + k + 16);
                __m256 b2 = _mm256_loadu_ps(mat_B_ptr + k + 16);
                c2 = _mm256_fmadd_ps(a2, b2, c2);

                // Fourth block
                __m256 a3 = _mm256_loadu_ps(mat_A_ptr + k + 24);
                __m256 b3 = _mm256_loadu_ps(mat_B_ptr + k + 24);
                c3 = _mm256_fmadd_ps(a3, b3, c3);
            }

            float sum = add_reduce_mm_256(
                _mm256_add_ps(_mm256_add_ps(c0, c1),
                            _mm256_add_ps(c2, c3))
            );

            // Handle leftover elements (< 32)
            #pragma omp simd reduction(+:sum)
            for (size_t k_t = k; k_t < K; ++k_t) {
                sum += mat_A_ptr[k_t] * mat_B_ptr[k_t];
            }

            mat_C[i * N + j] = sum;
        }
    }
}

void lg_M_sm_N_lg_K(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TM = 32;   // tile size for rows of A / C
    constexpr size_t TK = 48;   // tile size for rows of A / C

    memset(mat_C, 0, M * N * sizeof(float));
    
    #pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < M; ii += TM) {
        size_t i_end = std::min(ii + TM, M);

        for (size_t kk = 0; kk < K; kk += TK) {
            size_t k_end = std::min(kk + TK, K);

            for (size_t i = ii; i < i_end; ++i) {
                float *c_ptr = mat_C + i * N;
                const float *a_ptr = mat_A + i * K;

                size_t k = kk;

                // ---- Vectorized K loop (k += 4) ----
                for (; k + 6 <= k_end; k += 6) {
                    __m256 a0 = _mm256_set1_ps(a_ptr[k]);
                    __m256 a1 = _mm256_set1_ps(a_ptr[k + 1]);
                    __m256 a2 = _mm256_set1_ps(a_ptr[k + 2]);
                    __m256 a3 = _mm256_set1_ps(a_ptr[k + 3]);
                    __m256 a4 = _mm256_set1_ps(a_ptr[k + 4]);
                    __m256 a5 = _mm256_set1_ps(a_ptr[k + 5]);

                    const float *b0_ptr = mat_B + (k + 0) * N;
                    const float *b1_ptr = mat_B + (k + 1) * N;
                    const float *b2_ptr = mat_B + (k + 2) * N;
                    const float *b3_ptr = mat_B + (k + 3) * N;
                    const float *b4_ptr = mat_B + (k + 4) * N;
                    const float *b5_ptr = mat_B + (k + 5) * N;

                    size_t j = 0;
                    for (; j + 8 <= N; j += 8) {
                        __m256 c = _mm256_loadu_ps(c_ptr + j);

                        c = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0_ptr + j), c);
                        c = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1_ptr + j), c);
                        c = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2_ptr + j), c);
                        c = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3_ptr + j), c);
                        c = _mm256_fmadd_ps(a4, _mm256_loadu_ps(b4_ptr + j), c);
                        c = _mm256_fmadd_ps(a5, _mm256_loadu_ps(b5_ptr + j), c);

                        _mm256_storeu_ps(c_ptr + j, c);
                    }

                    // ---- j cleanup (scalar columns) ----
                    for (; j < N; ++j) {
                        c_ptr[j] +=
                            a_ptr[k + 0] * b0_ptr[j] +
                            a_ptr[k + 1] * b1_ptr[j] +
                            a_ptr[k + 2] * b2_ptr[j] +
                            a_ptr[k + 3] * b3_ptr[j] +
                            a_ptr[k + 4] * b4_ptr[j] +
                            a_ptr[k + 5] * b5_ptr[j];
                    }
                }

                // ---- k cleanup (remaining k < 4) ----
                for (; k < k_end; ++k) {
                    const float a = a_ptr[k];
                    const float *b_ptr = mat_B + k * N;

                    size_t j = 0;
                    for (; j + 8 <= N; j += 8) {
                        __m256 c = _mm256_loadu_ps(c_ptr + j);
                        c = _mm256_fmadd_ps(_mm256_set1_ps(a),
                                            _mm256_loadu_ps(b_ptr + j),
                                            c);
                        _mm256_storeu_ps(c_ptr + j, c);
                    }

                    for (; j < N; ++j) {
                        c_ptr[j] += a * b_ptr[j];
                    }
                }
            }
        }
    }
}

void lg_M_N_K(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TM = 32;   // tile size for rows of A / C
    constexpr size_t TN = 32;   // tile size for rows of A / C
    constexpr size_t TK = 8;   // tile size for rows of A / C

    memset(mat_C, 0, M * N * sizeof(float));

    for (size_t kk = 0; kk < K; kk += TK) {
        size_t k_end = std::min(kk + TK, K);

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t ii = 0; ii < M; ii += TM) {
            for (size_t jj = 0; jj < N; jj += TN) {
                size_t i_end = std::min(ii + TM, M);
                size_t j_end = std::min(jj + TN, N);
                
                size_t i = ii; 
                for (; i + 8 <= i_end; i += 8) {
                    const float *a0_ptr = mat_A + i * K;
                    const float *a1_ptr = mat_A + (i+1) * K;
                    const float *a2_ptr = mat_A + (i+2) * K;
                    const float *a3_ptr = mat_A + (i+3) * K;
                    const float *a4_ptr = mat_A + (i+4) * K;
                    const float *a5_ptr = mat_A + (i+5) * K;
                    const float *a6_ptr = mat_A + (i+6) * K;
                    const float *a7_ptr = mat_A + (i+7) * K;

                    size_t j = jj; 
                    for (; j + 8 <= j_end; j += 8) {

                        __m256 c0, c1, c2, c3, c4, c5, c6, c7;
                        __m256 a04, a15, a26, a37;
                        __m256 b_vec;

                        const float *b_ptr = mat_B + j;
                        
                        c0 = _mm256_loadu_ps(mat_C + i * N + j);
                        c1 = _mm256_loadu_ps(mat_C + (i+1) * N + j);
                        c2 = _mm256_loadu_ps(mat_C + (i+2) * N + j);
                        c3 = _mm256_loadu_ps(mat_C + (i+3) * N + j);
                        c4 = _mm256_loadu_ps(mat_C + (i+4) * N + j);
                        c5 = _mm256_loadu_ps(mat_C + (i+5) * N + j);
                        c6 = _mm256_loadu_ps(mat_C + (i+6) * N + j);
                        c7 = _mm256_loadu_ps(mat_C + (i+7) * N + j);

                        for (size_t k = kk; k < k_end; ++k) {
                            b_vec = _mm256_loadu_ps(b_ptr + k * N);

                            a04 = _mm256_set1_ps(a0_ptr[k]);
                            c0 = _mm256_fmadd_ps(a04, b_vec, c0);

                            a15 = _mm256_set1_ps(a1_ptr[k]);
                            c1 = _mm256_fmadd_ps(a15, b_vec, c1);

                            a26 = _mm256_set1_ps(a2_ptr[k]);
                            c2 = _mm256_fmadd_ps(a26, b_vec, c2);

                            a37 = _mm256_set1_ps(a3_ptr[k]);
                            c3 = _mm256_fmadd_ps(a37, b_vec, c3);

                            a04 = _mm256_set1_ps(a4_ptr[k]);
                            c4 = _mm256_fmadd_ps(a04, b_vec, c4);

                            a15 = _mm256_set1_ps(a5_ptr[k]);
                            c5 = _mm256_fmadd_ps(a15, b_vec, c5);

                            a26 = _mm256_set1_ps(a6_ptr[k]);
                            c6 = _mm256_fmadd_ps(a26, b_vec, c6);

                            a37 = _mm256_set1_ps(a7_ptr[k]);
                            c7 = _mm256_fmadd_ps(a37, b_vec, c7);
                        }

                        _mm256_storeu_ps(mat_C + i * N + j, c0);
                        _mm256_storeu_ps(mat_C + (i+1) * N + j, c1);
                        _mm256_storeu_ps(mat_C + (i+2) * N + j, c2);
                        _mm256_storeu_ps(mat_C + (i+3) * N + j, c3);
                        _mm256_storeu_ps(mat_C + (i+4) * N + j, c4);
                        _mm256_storeu_ps(mat_C + (i+5) * N + j, c5);
                        _mm256_storeu_ps(mat_C + (i+6) * N + j, c6);
                        _mm256_storeu_ps(mat_C + (i+7) * N + j, c7);
                    }
                
                    // cleanup loop

                    // cleanup columns for 8-row block
                    for (; j < j_end; ++j) {
                        float c0s = mat_C[(i+0) * N + j];
                        float c1s = mat_C[(i+1) * N + j];
                        float c2s = mat_C[(i+2) * N + j];
                        float c3s = mat_C[(i+3) * N + j];
                        float c4s = mat_C[(i+4) * N + j];
                        float c5s = mat_C[(i+5) * N + j];
                        float c6s = mat_C[(i+6) * N + j];
                        float c7s = mat_C[(i+7) * N + j];

                        for (size_t k = kk; k < k_end; ++k) {
                            float b = mat_B[k * N + j];
                            c0s += a0_ptr[k] * b;
                            c1s += a1_ptr[k] * b;
                            c2s += a2_ptr[k] * b;
                            c3s += a3_ptr[k] * b;
                            c4s += a4_ptr[k] * b;
                            c5s += a5_ptr[k] * b;
                            c6s += a6_ptr[k] * b;
                            c7s += a7_ptr[k] * b;
                        }

                        mat_C[(i+0) * N + j] = c0s;
                        mat_C[(i+1) * N + j] = c1s;
                        mat_C[(i+2) * N + j] = c2s;
                        mat_C[(i+3) * N + j] = c3s;
                        mat_C[(i+4) * N + j] = c4s;
                        mat_C[(i+5) * N + j] = c5s;
                        mat_C[(i+6) * N + j] = c6s;
                        mat_C[(i+7) * N + j] = c7s;
                    }

                }

                for (; i < i_end; ++i) {
                    const float *a_ptr = mat_A + i * K;

                    for (size_t j = jj; j < j_end; ++j) {
                        float c = mat_C[i * N + j];

                        for (size_t k = kk; k < k_end; ++k) {
                            c += a_ptr[k] * mat_B[k * N + j];
                        }

                        mat_C[i * N + j] = c;
                    }
                }
            }
        }
    }
}

void lg_K_linear_transpose(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K
) { 
    // N >= 128
    if (M >= 32) {
        if (K <= 128) {
            lg_M_N_sm_K_transpose(mat_A, mat_B, mat_C, M, N, K);
        } else {
            lg_M_N_K_transpose(mat_A, mat_B, mat_C, M, N, K);
        }
    } else {
        sm_M_lg_N_K_transpose(mat_A, mat_B, mat_C, M, N, K);
    }

    if (mat_bias != nullptr) {
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {
            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] += mat_bias[j];
            }
        }
    }
}

void linear_normal(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K
) {
    if (N < 128) {
        lg_M_sm_N_lg_K(mat_A, mat_B, mat_C, M, N, K);
    } else {
        lg_M_N_K(mat_A, mat_B, mat_C, M, N, K);
    }

    if (mat_bias != nullptr) {
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {

            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] += mat_bias[j];
            }
        }
    }
}

void linear_kernel(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #ifdef CPU_TIME
        CPUTimer timer("linear");
        printf("Shape of matmul: M=%zu, N=%zu, K=%zu, B transpose=%d\n", M, N, K, mat_B_transpose);
    #endif

    if (!mat_B_transpose) {
        linear_normal(mat_A, mat_B, mat_bias, mat_C, M, N, K);
        return;
    }

    if (K >= 1024 || N >= 1024) {
        lg_K_linear_transpose(
            mat_A, mat_B, mat_bias, mat_C, M, N, K
        );
        return;
    }
    
    if (mat_bias != nullptr) {
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {

            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }
    
    // B is N x K. B^T[k][j] = B[j][k] = mat_B[j * K + k]
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {        // Row of A and C
        for (size_t j = 0; j < N; ++j) {    // Column of B^T and C
            // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
            float sum = mat_C[i * N + j];
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < K; ++k) { // Inner dimension
                // C[i][j] += A[i][k] * B[j][k]
                sum += mat_A[i * K + k] * mat_B[j * K + k];
            }
            mat_C[i * N + j] = sum;
        }
    }
}

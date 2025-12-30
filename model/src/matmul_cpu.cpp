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
    constexpr size_t TK = 64; 

    memset(mat_C, 0, M * N * sizeof(float));

    // B is N x K, stored as B^T row-major: mat_B[j * K + k]
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < M; ii += TM) {
        size_t i_end = std::min(ii + TM, M);

        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);

            for (size_t i = ii; i < i_end; ++i) {
                for (size_t j = jj; j < j_end; ++j) {
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

                    mat_C[i * N + j] += sum;
                }
            }
        }
    }
}

void sm_M_lg_N_K_transpose(
    const float *mat_A, const float *mat_B,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TN = 64;   // tile size for rows of A / C

    memset(mat_C, 0, M * N * sizeof(float));

    // B is N x K, stored as B^T row-major: mat_B[j * K + k]
    for (size_t i = 0; i < M; ++i) {
        const float *mat_A_ptr = mat_A + i * K;

        #pragma omp parallel for schedule(static)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);

            for (size_t j = jj; j < j_end; ++j) {
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

                mat_C[i * N + j] += sum;
            }
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

void lg_K_linear_transpose(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K
) { 
    // N >= 128
    if (M >= 32) {
        lg_M_N_K_transpose(mat_A, mat_B, mat_C, M, N, K);
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
    lg_M_sm_N_lg_K(mat_A, mat_B, mat_C, M, N, K);

    if (mat_bias != nullptr) {
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {

            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    }
}

void linear(
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
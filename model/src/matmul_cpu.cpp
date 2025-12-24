#include "../include/matmul_cpu.hpp"

void large_N_K_linear(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    constexpr size_t TN = 64;   // columns of C / rows of B^T
    constexpr size_t TK = 128;  // reduction dimension

    if (mat_bias != nullptr) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }
    
    if (!mat_B_transpose) {
        // B is K x N. B[k][j] = mat_B[k * N + j]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B and C
                // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                float sum = mat_C[i * N + j];
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // C[i][j] += A[i][k] * B[k][j]
                    sum += mat_A[i * K + k] * mat_B[k * N + j];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
    else {
        // B is N x K, stored as B^T row-major: mat_B[j * K + k]
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < M; ++i) {
            for (size_t jj = 0; jj < N; jj += TN) {
                size_t j_end = std::min(jj + TN, N);

                for (size_t j = jj; j < j_end; ++j) {
                    float sum = mat_C[i * N + j];

                    // K-tile
                    for (size_t kk = 0; kk < K; kk += TK) {
                        size_t k_end = std::min(kk + TK, K);

                        #pragma omp simd reduction(+:sum)
                        for (size_t k = kk; k < k_end; ++k) {
                            sum += mat_A[i * K + k] * mat_B[j * K + k];
                        }
                    }

                    mat_C[i * N + j] = sum;
                }
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
    #endif

    if (N >= 2048 || K >= 2048) {
        large_N_K_linear(
            mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose
        );
        return;
    }
    
    if (mat_bias != nullptr) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }

    if (!mat_B_transpose) {
        // B is K x N. B[k][j] = mat_B[k * N + j]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B and C
                // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                float sum = mat_C[i * N + j];
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // C[i][j] += A[i][k] * B[k][j]
                    sum += mat_A[i * K + k] * mat_B[k * N + j];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
    else {
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

    #ifdef CPU_TIME
        printf("Shape of matmul: M=%zu, N=%zu, K=%zu\n", M, N, K);
    #endif
}
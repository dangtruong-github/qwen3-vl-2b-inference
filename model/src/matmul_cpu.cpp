#include "../include/matmul_cpu.hpp"

void linear(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    // CPUTimer timer("linear");
    // A is M x K
    // B is K x N (or N x K if transposed)
    // C is M x N
    // mat_bias is a vector of length N, or NULL

    // 1. Initialize C and apply Bias
    // C[i][j] = mat_bias[j] (if mat_bias is not NULL) or 0.0f
    if (mat_bias != nullptr) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                // Initialize C[i][j] with the bias value mat_bias[j]
                mat_C[i * N + j] = mat_bias[j];
            }
        }
    } else {
        // If no bias, initialize C to zero
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }

    // 2. Perform Matrix Multiplication: C = A * B + C_initial (where C_initial is the bias)
    // We can use the i-j-k loop order and perform the accumulation directly into mat_C.

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
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B^T and C
                // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                float sum = mat_C[i * N + j];
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // C[i][j] += A[i][k] * B[j][k]
                    sum += mat_A[i * K + k] * mat_B[j * K + k];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
    // printf("Shape of matmul: M=%zu, N=%zu, K=%zu\n", M, N, K);
}
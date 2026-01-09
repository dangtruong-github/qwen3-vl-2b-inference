#include "../include/cpu_wrapper.hpp"

void linear_fp16(
    const float *mat_A /* fp32 */, 
    const half_cpu *mat_B /* fp16 */, 
    const half_cpu *mat_bias /* fp16 */,
    float *mat_C /* fp32 */, 
    size_t M, size_t N, size_t K, 
    bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        fp16_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        fp16_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #else
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
    #endif
}

void linear_fp32(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        fp32_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        fp32_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #else
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

        if (mat_B_transpose) {
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
        } else {
            // B is K x N. So B[k][j] = mat_B[k * N + j]
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < M; ++i) {        // Row of A and C
                for (size_t j = 0; j < N; ++j) {    // Column of B and C
                    // The current value of mat_C[i * N + j] is mat_bias[j] (or 0)
                    float sum = mat_C[i * N + j];
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < K; ++k) { // Inner dimension
                        // C[i][j] += A[i][k] * B[k][j]
                        sum += mat_A[i * K + k] * mat_B[k * N + j];
                    }
                    mat_C[i * N + j] = sum;
                }
            }
        }
    #endif
}

void linear(
    const float *mat_A, const void *mat_B_in, const void *mat_bias_in,
    float *mat_C, size_t M, size_t N, size_t K,
    bool mat_B_transpose, DType::Type type_b
) {
    #ifdef CPU_TIME
        CPUTimer timer("linear");
        printf("Shape of matmul w/ precision %s: M=%zu, N=%zu, K=%zu, bias=%d, B transpose=%d\n", dtypeToStr(type_b), M, N, K, (mat_bias_in != nullptr), mat_B_transpose);
    #endif

    if (type_b == DType::FP16) {
        linear_fp16(
            mat_A,
            static_cast<const half_cpu*>(mat_B_in), 
            static_cast<const half_cpu*>(mat_bias_in),
            mat_C, M, N, K, mat_B_transpose 
        );
        return;
    } else if (type_b == DType::FP32) {
        linear_fp32(
            mat_A, (const float *)mat_B_in, (const float *)mat_bias_in,
            mat_C, M, N, K, mat_B_transpose
        );
        return;
    }

    fprintf(stderr, "DType matmul not supported %s\n", dtypeToStr(type_b));
    exit(1);
}

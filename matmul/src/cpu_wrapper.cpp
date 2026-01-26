#include "../include/cpu_wrapper.hpp"

void linear_f32a_f16b_f32c(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        f32a_f16b_f32c_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        f32a_f16b_f32c_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
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

void linear_fp32_full(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        fp32_full_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        fp32_full_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
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

void linear_f32a_i8f32sb_f32c(
    const float* mat_A, const int8_t* mat_B_in, const float* mat_B_scales,
    const int8_t* mat_bias_in, const float* mat_bias_scale, float* mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose, size_t group_size
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        f32a_i8f32sb_f32c_avx2_kernel(
            mat_A, mat_B_in, mat_B_scales, mat_bias_in, mat_bias_scale,
            mat_C, M, N, K, mat_B_transpose, group_size
        );
    #elif defined(__AVX2__) && defined(__FMA__)
        f32a_i8f32sb_f32c_avx2_kernel(
            mat_A, mat_B_in, mat_B_scales, mat_bias_in, mat_bias_scale,
            mat_C, M, N, K, mat_B_transpose, group_size
        );
    #else
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float acc = 0.0f;

                // -------- GEMM --------
                for (size_t k = 0; k < K; ++k) {
                    float a = mat_A[i * K + k];

                    // linear index into B (matches quantizer layout)
                    size_t b_linear_idx;
                    if (mat_B_transpose) {
                        // B is [N, K]
                        b_linear_idx = j * K + k;
                    } else {
                        // B is [K, N]
                        b_linear_idx = k * N + j;
                    }

                    size_t scale_idx = b_linear_idx / group_size;
                    float scale = mat_B_scales[scale_idx];

                    float b = (float)mat_B_in[b_linear_idx] * scale;
                    acc += a * b;
                }

                mat_C[i * N + j] = acc;
            }
        }

        

        // -------- Bias --------
        if (mat_bias_in && mat_bias_scale) {
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t bias_scale_idx = j / group_size;
                    float bias =
                        (float)mat_bias_in[j] * mat_bias_scale[bias_scale_idx];
                    mat_C[i * N + j] += bias;
                }
            }
        }
    #endif
}

void linear_fp16_full(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = mat_bias ? (float)mat_bias[j] : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                float a = (float)mat_A[i * K + k];
                float b = mat_B_transpose
                        ? (float)mat_B[j * K + k]   // (N, K)
                        : (float)mat_B[k * N + j];  // (K, N)
                sum += a * b;
            }

            mat_C[i * N + j] = (half_cpu)sum;
        }
    }
}

void linear_f16a_f16b_f32c(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = mat_bias ? (float)mat_bias[j] : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                float a = (float)mat_A[i * K + k];
                float b = mat_B_transpose
                        ? (float)mat_B[j * K + k]
                        : (float)mat_B[k * N + j];
                sum += a * b;
            }

            mat_C[i * N + j] = sum;
        }
    }
}

void linear_f32a_f16b_f16c(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
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

void linear(
    const void *mat_A, const void *mat_B_in, const void *mat_B_scale,
    const void *mat_bias_in, const void *mat_bias_scale,
    void *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose,
    DType::Type type_a, DType::Type type_b, DType::Type type_b_scale,
    DType::Type type_c, size_t group_size
) {
    #ifdef CPU_TIME
        CPUTimer timer("linear");
        printf("Shape of matmul w/ precision %s: M=%zu, N=%zu, K=%zu, bias=%d, B transpose=%d\n", dtypeToStr(type_b), M, N, K, (mat_bias_in != nullptr), mat_B_transpose);
    #endif

    if (type_a == DType::FP32 && type_c == DType::FP32) {
        if (type_b == DType::FP16) {
            linear_f32a_f16b_f32c(
                static_cast<const float*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<float*>(mat_C),
                M, N, K, mat_B_transpose
            );
            return;
        } else if (type_b == DType::FP32) {
            linear_fp32_full(
                static_cast<const float*>(mat_A),
                static_cast<const float*>(mat_B_in),
                static_cast<const float*>(mat_bias_in),
                static_cast<float*>(mat_C),
                M, N, K, mat_B_transpose
            );
            return;
        } else if (type_b == DType::INT8 && type_b_scale == DType::FP32)  {
            linear_f32a_i8f32sb_f32c(
                static_cast<const float*>(mat_A),
                static_cast<const int8_t*>(mat_B_in),
                static_cast<const float*>(mat_B_scale),
                static_cast<const int8_t*>(mat_bias_in),
                static_cast<const float*>(mat_bias_scale),
                static_cast<float*>(mat_C),
                M, N, K, mat_B_transpose, group_size
            );
            return;
        }
    } else if (type_a == DType::FP16 && type_c == DType::FP32) {
        if (type_b == DType::FP16) {
            linear_f16a_f16b_f32c(
                static_cast<const half_cpu*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<float*>(mat_C),
                M, N, K, mat_B_transpose
            );
        }
    } else if (type_a == DType::FP32 && type_c == DType::FP16) {
        if (type_b == DType::FP16) {
            linear_f32a_f16b_f16c(
                static_cast<const float*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<half_cpu*>(mat_C),
                M, N, K, mat_B_transpose
            );
        }
    } else if (type_a == DType::FP16 && type_c == DType::FP16) {
        if (type_b == DType::FP16) {
            linear_fp16_full(
                static_cast<const half_cpu*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<half_cpu*>(mat_C),
                M, N, K, mat_B_transpose
            );
        }
    }

    fprintf(stderr, "DType matmul not supported: type_a=%s, type_b=%s, type_b_scale=%s, type_c=%s\n", dtypeToStr(type_a), dtypeToStr(type_b), dtypeToStr(type_b_scale), dtypeToStr(type_c));
    exit(1);
}

void gemm_att(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX2__) && defined(__FMA__)
        att_fp32_full_avx2_kernel(mat_A, mat_B, mat_C, scale, N, K, mat_B_transpose);
    #else
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
    #endif
}
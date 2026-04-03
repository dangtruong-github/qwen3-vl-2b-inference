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
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, const int *sum_int8_B, float* mat_C,
    size_t M, size_t N, size_t K, size_t group_size
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        if (sum_int8_B) {
            f32a_i8f32sb_f32c_avx512_prefix_kernel(
                mat_A, mat_B_in, mat_B_scales, sum_int8_B,
                mat_C, M, N, K, group_size
            );
        } else {
            f32a_i8f32sb_f32c_avx512_kernel(
                mat_A, mat_B_in, mat_B_scales,
                mat_C, M, N, K, group_size
            );
        }
    #elif defined(__AVX2__) && defined(__FMA__)
        if (sum_int8_B) {
            f32a_i8f32sb_f32c_avx2_prefix_kernel(
                mat_A, mat_B_in, mat_B_scales, sum_int8_B,
                mat_C, M, N, K, group_size
            );
        } else {
            f32a_i8f32sb_f32c_avx2_kernel(
                mat_A, mat_B_in, mat_B_scales,
                mat_C, M, N, K, group_size
            );
        }
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
                    b_linear_idx = j * K + k;

                    size_t scale_idx = b_linear_idx / group_size;
                    float scale = mat_B_scales[scale_idx];

                    float b = (float)mat_B_in[b_linear_idx] * scale;
                    acc += a * b;
                }

                mat_C[i * N + j] = acc;
            }
        }
    #endif
}

void linear_f32a_i8f32sb_f32bias_f32c(
    const float* mat_A, const int8_t* mat_B_in, const float* mat_B_scales,
    const int *sum_int8_B, const float *mat_bias, float* mat_C,
    size_t M, size_t N, size_t K, size_t group_size
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        f32a_i8f32sb_f32bias_f32c_avx2_kernel(
            mat_A, mat_B_in, mat_B_scales,
            mat_bias, mat_C, M, N, K, group_size
        );
    #elif defined(__AVX2__) && defined(__FMA__)
        f32a_i8f32sb_f32bias_f32c_avx2_kernel(
            mat_A, mat_B_in, mat_B_scales,
            mat_bias, mat_C, M, N, K, group_size
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
                    b_linear_idx = j * K + k;

                    size_t scale_idx = b_linear_idx / group_size;
                    float scale = mat_B_scales[scale_idx];

                    float b = (float)mat_B_in[b_linear_idx] * scale;
                    acc += a * b;
                }

                mat_C[i * N + j] = acc + mat_bias[j];
            }
        }
    #endif
}

void linear_f32a_i8f32sb_f32c_rq(
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, float* mat_C,
    size_t M, size_t N, size_t K
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

void linear_fp16_full(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        fp16_full_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        fp16_full_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #else
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
    #endif
}

void linear_f16ab_f32c(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        f16ab_f32c_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        f16ab_f32c_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #else
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
    #endif
}

void linear_f32a_f16bc(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        // Must implement AVX512
        f32a_f16bc_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #elif defined(__AVX2__) && defined(__FMA__)
        f32a_f16bc_avx2_kernel(mat_A, mat_B, mat_bias, mat_C, M, N, K, mat_B_transpose);
    #else
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
    #endif
}

void linear(
    const void *mat_A, const void *mat_B_in, const void *mat_B_scale,
    const void *sum_int8_B, const void *mat_bias_in, const void *mat_bias_scale,
    void *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose,
    DType::Type type_a, DType::Type type_b, DType::Type type_b_scale,
    DType::Type type_c, bool group_quantized, size_t group_size
) {
    #ifdef CPU_TIME
        CPUTimer timer("linear");
        printf("A=%s, B=%s, C=%s, M=%zu, N=%zu, K=%zu, bias=%d, B_trans=%d\n", dtypeToStr(type_a), dtypeToStr(type_b), dtypeToStr(type_c), M, N, K, (mat_bias_in != nullptr), mat_B_transpose);
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
        } else if (type_b == DType::INT8 && type_b_scale == DType::FP32 && mat_B_transpose) {
            if (mat_bias_in == nullptr) {
                if (group_quantized) {
                    linear_f32a_i8f32sb_f32c(
                        static_cast<const float*>(mat_A),
                        static_cast<const int8_t*>(mat_B_in),
                        static_cast<const float*>(mat_B_scale),
                        static_cast<const int*>(sum_int8_B),
                        static_cast<float*>(mat_C), M, N, K, group_size
                    );
                    return;
                } else {
                    linear_f32a_i8f32sb_f32c_rq(
                        static_cast<const float*>(mat_A),
                        static_cast<const int8_t*>(mat_B_in),
                        static_cast<const float*>(mat_B_scale),
                        static_cast<float*>(mat_C), M, N, K
                    );
                    return;
                }
            } else {
                linear_f32a_i8f32sb_f32bias_f32c(
                    static_cast<const float*>(mat_A),
                    static_cast<const int8_t*>(mat_B_in),
                    static_cast<const float*>(mat_B_scale),
                    static_cast<const int*>(sum_int8_B),
                    static_cast<const float*>(mat_bias_in),
                    static_cast<float*>(mat_C), M, N, K, group_size
                );
                return;
            }
        }
    } else if (type_a == DType::FP16 && type_c == DType::FP32) {
        if (type_b == DType::FP16) {
            linear_f16ab_f32c(
                static_cast<const half_cpu*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<float*>(mat_C),
                M, N, K, mat_B_transpose
            );
            return;
        }
    } else if (type_a == DType::FP32 && type_c == DType::FP16) {
        if (type_b == DType::FP16 && group_quantized) {
            linear_f32a_f16bc(
                static_cast<const float*>(mat_A),
                static_cast<const half_cpu*>(mat_B_in), 
                static_cast<const half_cpu*>(mat_bias_in),
                static_cast<half_cpu*>(mat_C),
                M, N, K, mat_B_transpose
            );
            return;
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
            return;
        }
    }

    fprintf(stderr, "DType matmul not supported: type_a=%s, type_b=%s, type_b_scale=%s, type_c=%s, group_quantized=%d\n", dtypeToStr(type_a), dtypeToStr(type_b), dtypeToStr(type_b_scale), dtypeToStr(type_c), group_quantized);
    exit(1);
}

void gemm_att_fp32_full(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    #if defined(__AVX2__) && defined(__FMA__)
        // Only use optimized kernel when M == 1
        if (M == 1) {
            att_fp32_full_avx2_kernel(
                mat_A, mat_B, mat_C,
                scale, N, K, mat_B_transpose
            );
            return;
        }
    #endif

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {

        const float* A_row = mat_A + m * K;
        float* C_row = mat_C + m * N;

        for (size_t n = 0; n < N; ++n) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                float a = A_row[k];

                float b = (!mat_B_transpose)
                    ? mat_B[k * N + n]     // B[K,N]
                    : mat_B[n * K + k];    // B[N,K]

                sum += a * b;
            }

            C_row[n] = sum * scale;
        }
    }
}

void gemm_att_f16ab_f32c(
    const half_cpu *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    #if defined(__AVX2__) && defined(__FMA__)
        att_f16ab_f32c_avx2_kernel(
            mat_A, mat_B, mat_C, scale, M, N, K, mat_B_transpose
        );
    #else
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < M; ++m) {

            const half_cpu* A_row = mat_A + m * K;
            float* C_row = mat_C + m * N;

            for (size_t n = 0; n < N; ++n) {

                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k) {

                    float a = (float)A_row[k];

                    float b = (!mat_B_transpose)
                        ? (float)mat_B[k * N + n]
                        : (float)mat_B[n * K + k];

                    sum += a * b;
                }

                C_row[n] = sum * scale;
            }
        }
    #endif
}

void gemm_att(
    const void *mat_A, const void *mat_B, void *mat_C,
    const float scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose, DType::Type type_a,
    DType::Type type_b, DType::Type type_c
) {

    if (type_a == DType::FP32 &&
        type_b == DType::FP32 &&
        type_c == DType::FP32)
    {
        gemm_att_fp32_full(
            static_cast<const float*>(mat_A),
            static_cast<const float*>(mat_B),
            static_cast<float*>(mat_C),
            scale, M, N, K, mat_B_transpose
        );
        return;
    }

    if (type_a == DType::FP16 &&
        type_b == DType::FP16 &&
        type_c == DType::FP32)
    {
        gemm_att_f16ab_f32c(
            static_cast<const half_cpu*>(mat_A),
            static_cast<const half_cpu*>(mat_B),
            static_cast<float*>(mat_C),
            scale, M, N, K, mat_B_transpose
        );
        return;
    }

    fprintf(stderr,
        "DType gemm att not supported: type_a=%s, type_b=%s, type_c=%s\n",
        dtypeToStr(type_a),
        dtypeToStr(type_b),
        dtypeToStr(type_c)
    );
    exit(1);
}

void gemm_att_fp32_full_multiple_scale(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float *scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    #if defined(__AVX2__) && defined(__FMA__)
        if (M == 1) {
            // For M=1 just reuse old kernel
            att_fp32_full_avx2_kernel(
                mat_A, mat_B, mat_C,
                scale[0], N, K, mat_B_transpose
            );
            return;
        }
    #endif

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {

        const float* A_row = mat_A + m * K;
        float* C_row = mat_C + m * N;
        float row_scale = scale[m];

        for (size_t n = 0; n < N; ++n) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                float a = A_row[k];

                float b = (!mat_B_transpose)
                    ? mat_B[k * N + n]
                    : mat_B[n * K + k];

                sum += a * b;
            }

            C_row[n] = sum * row_scale;
        }
    }
}

void gemm_att_f32a_f16bc_multiple_scale(
    const float *mat_A, const half_cpu *mat_B, half_cpu *mat_C,
    const float *scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    #if defined(__AVX2__) && defined(__FMA__)
        att_f32a_f16bc_mul_scale_avx2_kernel(
            mat_A, mat_B, mat_C, scale, M, N, K, mat_B_transpose
        );
    #else
        #pragma omp parallel for schedule(static)
        for (size_t m = 0; m < M; ++m) {

            const float* A_row = mat_A + m * K;
            half_cpu* C_row = mat_C + m * N;
            float row_scale = scale ? scale[m] : 1.0f;

            for (size_t n = 0; n < N; ++n) {

                float sum = 0.0f;

                for (size_t k = 0; k < K; ++k) {

                    float a = A_row[k];

                    float b = (!mat_B_transpose)
                        ? (float)mat_B[k * N + n]
                        : (float)mat_B[n * K + k];

                    sum += a * b;
                }

                C_row[n] = (half_cpu)(sum * row_scale);
            }
        }
    #endif
}

void gemm_att_multiple_scale(
    const void *mat_A, const void *mat_B, void *mat_C,
    const float *scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose, DType::Type type_a,
    DType::Type type_b, DType::Type type_c
) {

    if (type_a == DType::FP32 &&
        type_b == DType::FP32 &&
        type_c == DType::FP32)
    {
        gemm_att_fp32_full_multiple_scale(
            static_cast<const float*>(mat_A),
            static_cast<const float*>(mat_B),
            static_cast<float*>(mat_C),
            scale, M, N, K, mat_B_transpose
        );
        return;
    }

    if (type_a == DType::FP32 &&
        type_b == DType::FP16 &&
        type_c == DType::FP16)
    {
        gemm_att_f32a_f16bc_multiple_scale(
            static_cast<const float*>(mat_A),
            static_cast<const half_cpu*>(mat_B),
            static_cast<half_cpu*>(mat_C),
            scale, M, N, K, mat_B_transpose
        );
        return;
    }

    fprintf(stderr,
        "DType gemm att not supported: type_a=%s, type_b=%s, type_c=%s\n",
        dtypeToStr(type_a),
        dtypeToStr(type_b),
        dtypeToStr(type_c)
    );
    exit(1);
}

void qk_att_fp32(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, float *max_row, size_t kv_mul,
    size_t seq_len, size_t head_dim, const size_t max_pos
) {
    if (max_row) {
        for (size_t i = 0; i < kv_mul; ++i) {
            float *mat_C_now = mat_C + i * seq_len;

            #if defined(_OPENMP) && (_OPENMP >= 201307)  // OpenMP 4.0+ (safe for max reduction)
                float row_max = -INFINITY;

                #pragma omp parallel for reduction(max:row_max)
                for (size_t j = 0; j < max_pos; ++j) {
                    float sum = 0.0f;

                    for (size_t k = 0; k < head_dim; ++k) {
                        float a = mat_A[i * head_dim + k];
                        float b = (float)(mat_B[j * head_dim + k]);
                        sum += a * b;
                    }

                    float value = sum * scale;
                    mat_C_now[j] = value;

                    row_max = std::max(row_max, value);
                }

                max_row[i] = row_max;

            #else
                // 🔁 Fallback: manual reduction (portable)

                float row_max = -INFINITY;

                #pragma omp parallel
                {
                    float local_max = -INFINITY;

                    #pragma omp for nowait
                    for (size_t j = 0; j < max_pos; ++j) {
                        float sum = 0.0f;

                        for (size_t k = 0; k < head_dim; ++k) {
                            float a = mat_A[i * head_dim + k];
                            float b = (float)(mat_B[j * head_dim + k]);
                            sum += a * b;
                        }

                        float value = sum * scale;
                        mat_C_now[j] = value;

                        local_max = std::max(local_max, value);
                    }

                    #pragma omp critical
                    {
                        row_max = std::max(row_max, local_max);
                    }
                }

                max_row[i] = row_max;

            #endif
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < kv_mul; ++i) {
            for (size_t j = 0; j < max_pos; ++j) {
                float sum = 0.0f;

                for (size_t k = 0; k < head_dim; ++k) {
                    float a = mat_A[i * head_dim + k];
                    float b = mat_B[j * head_dim + k];
                    sum += a * b;
                }

                mat_C[i * seq_len + j] = sum * scale;
            }
        }
    }
}

void qk_att_f32a_f16b_f32c(
    const float *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, float *max_row, size_t kv_mul,
    size_t seq_len, size_t head_dim, const size_t max_pos
) {
    if (max_row) {
        for (size_t i = 0; i < kv_mul; ++i) {
            float *mat_C_now = mat_C + i * seq_len;

            #if defined(_OPENMP) && (_OPENMP >= 201307)  // OpenMP 4.0+ (safe for max reduction)
                float row_max = -INFINITY;

                #pragma omp parallel for reduction(max:row_max)
                for (size_t j = 0; j < max_pos; ++j) {
                    float sum = 0.0f;

                    for (size_t k = 0; k < head_dim; ++k) {
                        float a = mat_A[i * head_dim + k];
                        float b = (float)(mat_B[j * head_dim + k]);
                        sum += a * b;
                    }

                    float value = sum * scale;
                    mat_C_now[j] = value;

                    row_max = std::max(row_max, value);
                }

                max_row[i] = row_max;

            #else
                // 🔁 Fallback: manual reduction (portable)

                float row_max = -INFINITY;

                #pragma omp parallel
                {
                    float local_max = -INFINITY;

                    #pragma omp for nowait
                    for (size_t j = 0; j < max_pos; ++j) {
                        float sum = 0.0f;

                        for (size_t k = 0; k < head_dim; ++k) {
                            float a = mat_A[i * head_dim + k];
                            float b = (float)(mat_B[j * head_dim + k]);
                            sum += a * b;
                        }

                        float value = sum * scale;
                        mat_C_now[j] = value;

                        local_max = std::max(local_max, value);
                    }

                    #pragma omp critical
                    {
                        row_max = std::max(row_max, local_max);
                    }
                }

                max_row[i] = row_max;

            #endif
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < kv_mul; ++i) {
            for (size_t j = 0; j < max_pos; ++j) {
                float sum = 0.0f;

                for (size_t k = 0; k < head_dim; ++k) {
                    float a = mat_A[i * head_dim + k];
                    float b = (float)(mat_B[j * head_dim + k]);
                    sum += a * b;
                }

                mat_C[i * seq_len + j] = sum * scale;
            }
        }
    }      
}

void qk_att_f32a_i8f32b_f32c(
    const float *mat_A, const int8_t *mat_B, const float *mat_B_scales,
    float *mat_C, const float scale, float *max_row, size_t kv_mul,
    size_t seq_len, size_t head_dim, const size_t max_pos, const size_t group_size
) {
    if (max_row) {
        for (size_t i = 0; i < kv_mul; ++i) {
            float *mat_C_now = mat_C + i * seq_len;

            #if defined(_OPENMP) && (_OPENMP >= 201307)  // OpenMP 4.0+ (safe for max reduction)
                float row_max = -INFINITY;

                #pragma omp parallel for reduction(max:row_max)
                for (size_t j = 0; j < max_pos; ++j) {
                    float sum = 0.0f;

                    for (size_t k = 0; k < head_dim; ++k) {
                        float a = mat_A[i * head_dim + k];
                        size_t b_id = j * head_dim + k;
                        float b = (float)(mat_B[b_id]) * (float)(mat_B_scales[b_id / group_size]);
                        sum += a * b;
                    }

                    float value = sum * scale;
                    mat_C_now[j] = value;

                    row_max = std::max(row_max, value);
                }

                max_row[i] = row_max;

            #else
                // 🔁 Fallback: manual reduction (portable)

                float row_max = -INFINITY;

                #pragma omp parallel
                {
                    float local_max = -INFINITY;

                    #pragma omp for nowait
                    for (size_t j = 0; j < max_pos; ++j) {
                        float sum = 0.0f;

                        for (size_t k = 0; k < head_dim; ++k) {
                            float a = mat_A[i * head_dim + k];
                            size_t b_id = j * head_dim + k;
                            float b = (float)(mat_B[b_id]) * (float)(mat_B_scales[b_id / group_size]);
                            sum += a * b;
                        }

                        float value = sum * scale;
                        mat_C_now[j] = value;

                        local_max = std::max(local_max, value);
                    }

                    #pragma omp critical
                    {
                        row_max = std::max(row_max, local_max);
                    }
                }

                max_row[i] = row_max;

            #endif
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < kv_mul; ++i) {
            for (size_t j = 0; j < max_pos; ++j) {
                float sum = 0.0f;

                for (size_t k = 0; k < head_dim; ++k) {
                    float a = mat_A[i * head_dim + k];
                    size_t b_id = j * head_dim + k;
                    float b = (float)(mat_B[b_id]) * (float)(mat_B_scales[b_id / group_size]);
                    sum += a * b;
                }

                mat_C[i * seq_len + j] = sum * scale;
            }
        }
    }      
}

void gemm_text_qk_att(
    const void *mat_A, const void *mat_B, const void *mat_B_scales,
    void *mat_C, const float scale, float *max_row, size_t kv_mul,
    size_t seq_len, size_t head_dim, const size_t max_pos,
    const size_t group_size, DType::Type type_a, DType::Type type_b,
    DType::Type type_b_s, DType::Type type_c
) {
    if (
        type_a == DType::FP32 && type_b == DType::INT8
        && type_b_s == DType::FP32 && type_c == DType::FP32
    ) {
        qk_att_f32a_i8f32b_f32c(
            static_cast<const float *>(mat_A),
            static_cast<const int8_t *>(mat_B),
            static_cast<const float *>(mat_B_scales),
            static_cast<float *>(mat_C),
            scale, max_row, kv_mul, seq_len,
            head_dim, max_pos, group_size
        );
        return;
    } else if (
        type_a == DType::FP32 && type_b == DType::FP16
        && type_c == DType::FP32
    ) {
        qk_att_f32a_f16b_f32c(
            static_cast<const float *>(mat_A),
            static_cast<const half_cpu *>(mat_B),
            static_cast<float *>(mat_C),
            scale, max_row, kv_mul,
            seq_len, head_dim, max_pos
        );
        return;
    } else if (
        type_a == DType::FP32 && type_b == DType::FP32
        && type_c == DType::FP32
    ) {
        qk_att_fp32(
            static_cast<const float *>(mat_A),
            static_cast<const float *>(mat_B),
            static_cast<float *>(mat_C),
            scale, max_row, kv_mul,
            seq_len, head_dim, max_pos
        );
        return;
    }

    fprintf(stderr, "DType gemm_text_att not supported: type_a=%s, type_b=%s, type_c=%s\n", dtypeToStr(type_a), dtypeToStr(type_b), dtypeToStr(type_c));
    exit(1);
}

void kv_att_fp32(
    const float *mat_A, const float *mat_B, float *mat_C,
    size_t kv_mul, size_t head_dim, size_t seq_len, const size_t max_pos
) {
    // A: [kv_mul, seq_len]
    // B: [seq_len, head_dim] (row-major, NOT transposed)
    // C: [kv_mul, head_dim]

    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < kv_mul; ++m) {
        for (size_t n = 0; n < head_dim; ++n) {

            float acc = 0.0f;

            for (size_t k = 0; k < max_pos; ++k) {
                float a = mat_A[m * seq_len + k];
                float b = mat_B[k * head_dim + n];
                acc += a * b;
            }

            mat_C[m * head_dim + n] = acc;
        }
    }
}

void kv_att_f32a_f16b_f32c(
    const float *mat_A, const half_cpu *mat_B, float *mat_C,
    size_t kv_mul, size_t head_dim, size_t seq_len, const size_t max_pos
) {
    // A: [kv_mul, seq_len]
    // B: [seq_len, head_dim] (row-major, NOT transposed)
    // C: [kv_mul, head_dim]

    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < kv_mul; ++m) {
        for (size_t n = 0; n < head_dim; ++n) {

            float acc = 0.0f;

            for (size_t k = 0; k < max_pos; ++k) {
                float a = mat_A[m * seq_len + k];
                float b = (float)(mat_B[k * head_dim + n]);
                acc += a * b;
            }

            mat_C[m * head_dim + n] = acc;
        }
    }
}

void kv_att_f32a_i8s32b_f32c(
    const float *mat_A, const int8_t *mat_B, const float *mat_B_scales,
    float *mat_C, size_t kv_mul, size_t head_dim,
    size_t seq_len, const size_t max_pos, const size_t group_size
) {
    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < kv_mul; ++m) {
        for (size_t n = 0; n < head_dim; ++n) {

            float acc = 0.0f;

            for (size_t k = 0; k < max_pos; ++k) {
                float a = mat_A[m * seq_len + k];
                size_t b_id = k * head_dim + n;
                float b = (float)(mat_B[b_id]) * (float)(mat_B_scales[b_id / group_size]);
                acc += a * b;
            }

            mat_C[m * head_dim + n] = acc;
        }
    }
}

void gemm_text_kv_att(
    const void *mat_A, const void *mat_B, const void *mat_B_scales,
    void *mat_C, size_t kv_mul, size_t head_dim, size_t seq_len,
    const size_t max_pos, const size_t group_size, DType::Type type_a,
    DType::Type type_b, DType::Type type_b_s, DType::Type type_c
) {
    if (
        type_a == DType::FP32 && type_b == DType::INT8
        && type_b_s == DType::FP32 && type_c == DType::FP32
    ) {
        kv_att_f32a_i8s32b_f32c(
            static_cast<const float *>(mat_A),
            static_cast<const int8_t *>(mat_B),
            static_cast<const float *>(mat_B_scales),
            static_cast<float *>(mat_C),
            kv_mul, head_dim, seq_len, max_pos, group_size
        );
        return;
    } else if (
        type_a == DType::FP32 && type_b == DType::FP16
        && type_c == DType::FP32
    ) {
        kv_att_f32a_f16b_f32c(
            static_cast<const float *>(mat_A),
            static_cast<const half_cpu *>(mat_B),
            static_cast<float *>(mat_C),
            kv_mul, head_dim, seq_len, max_pos
        );
        return;
    } else if (
        type_a == DType::FP32 && type_b == DType::FP32
        && type_c == DType::FP32
    ) {
        kv_att_fp32(
            static_cast<const float *>(mat_A),
            static_cast<const float *>(mat_B),
            static_cast<float *>(mat_C),
            kv_mul, head_dim, seq_len, max_pos
        );
        return;
    }

    fprintf(stderr, "DType gemm_text_att not supported: type_a=%s, type_b=%s, type_c=%s\n", dtypeToStr(type_a), dtypeToStr(type_b), dtypeToStr(type_c));
    exit(1);
}

#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"
#include "simd_utils.hpp"

void linear(
    const void *mat_A, const void *mat_B_in, const void *mat_B_scale,
    const void *sum_int8_B, const void *mat_bias_in, const void *mat_bias_scale,
    void *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose,
    DType::Type type_a, DType::Type type_b, DType::Type type_b_scale,
    DType::Type type_c, bool group_quantized, size_t group_size
);
void gemm_att(
    const void *mat_A, const void *mat_B, void *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose,
    DType::Type type_a, DType::Type type_b, DType::Type type_c
);

#if defined(__AVX2__) && defined(__FMA__)
void fp32_full_avx2_kernel(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
);
void fp16_full_avx2_kernel(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
);
void f32a_i8f32sb_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int8_t *__restrict mat_bias_in,
    const float *__restrict mat_bias_scale,
    float *__restrict mat_C, size_t M, size_t N, size_t K,
    bool mat_B_transpose, size_t group_size
);
void f32a_f16bc_avx2_kernel(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    half_cpu *__restrict mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose
);
void f32a_f16b_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose
);
void f32a_f16b_f32c_avx2_kernel_untranspose(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
);
void f32a_i8f32sb_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K,
    bool mat_B_transpose, size_t group_size
);
void f32a_i8f32sb_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K, size_t group_size
);
#endif

#if defined(__AVX2__) && defined(__FMA__)
void att_fp32_full_avx2_kernel(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose
);
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
void f32a_i8f32sb_f32c_avx512_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K, size_t group_size
);
void f32a_i8f32sb_f32c_avx512_prefix_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int *__restrict sum_int8_B,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K, size_t group_size
);
#endif

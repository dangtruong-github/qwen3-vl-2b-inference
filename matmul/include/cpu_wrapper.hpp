#pragma once

#include <stdio.h>
#include <omp.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET

#include "../../utils/module.hpp"

#ifdef __ARM_NEON
#include "arm_utils.hpp"
#endif

void linear(
    const void *mat_A, const void *mat_B_in, const void *mat_B_scale,
    const void *sum_int8_B, const void *mat_bias_in, const void *mat_bias_scale,
    void *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose,
    DType::Type type_a, DType::Type type_b, DType::Type type_b_scale,
    DType::Type type_c, bool group_quantized, size_t group_size, bool add_to_c
);
void gemm_att(
    const void *mat_A, const void *mat_B, void *mat_C,
    const float scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose, DType::Type type_a,
    DType::Type type_b, DType::Type type_c
);
void gemm_att_multiple_scale(
    const void *mat_A, const void *mat_B, void *mat_C,
    const float *scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose, DType::Type type_a,
    DType::Type type_b, DType::Type type_c
);
void gemm_text_qk_att(
    const void *mat_A, const void *mat_B, const void *mat_B_scales,
    void *mat_C, const float scale, float *max_row, size_t kv_mul,
    size_t seq_len, size_t head_dim, const size_t max_pos,
    const size_t group_size, DType::Type type_a, DType::Type type_b,
    DType::Type type_b_s, DType::Type type_c
);
void gemm_text_kv_att(
    const void *mat_A, const void *mat_B, const void *mat_B_scales,
    void *mat_C, size_t kv_mul, size_t head_dim, size_t seq_len,
    const size_t max_pos, const size_t group_size, DType::Type type_a,
    DType::Type type_b, DType::Type type_b_s, DType::Type type_c
);

#ifdef __ARM_NEON
void arm_f32a_i8f32sb_f32c(
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, const int *sum_int8_B, float* mat_C,
    size_t M, size_t N, size_t K, size_t group_size, bool add_to_c
);
void arm_f32a_i8f32sb_f16c(
    const float* mat_A, const int8_t* mat_B_in,
    const float* mat_B_scales, const int *sum_int8_B, half_cpu* mat_C,
    size_t M, size_t N, size_t K, size_t group_size, bool add_to_c
);
#endif

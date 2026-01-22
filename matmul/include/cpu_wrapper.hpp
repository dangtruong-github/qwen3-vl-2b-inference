#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"
#include "cpu_fp32_avx2_kernel.hpp"
#include "cpu_fp16_avx2_kernel.hpp"
#include "cpu_int8_fp32s_avx2_kernel.hpp"
#include "cpu_gemm_att.hpp"

void linear(
    const float *mat_A, const void *mat_B_in, const void *mat_B_scale,
    const void *mat_bias_in, const void *mat_bias_scale,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose,DType::Type type_b, DType::Type type_b_scale, size_t group_size
);
void gemm_att(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose
);

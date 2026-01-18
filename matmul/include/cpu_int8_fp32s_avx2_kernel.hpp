#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void linear_int8_fp32s_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    const int8_t *__restrict mat_bias_in,
    const float *__restrict mat_bias_scale,
    float *__restrict mat_C, size_t M, size_t N, size_t K,
    bool mat_B_transpose, size_t group_size
);
#endif
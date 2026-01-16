#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void linear_int8_fp32s_avx2_kernel(
    const float *mat_A,          /* [M, K] */
    const int8_t *mat_B_in,      /* [N, K] if !trans, else [K, N] */
    const float *mat_B_scales,   /* Scales for B */
    const int8_t *mat_bias_in,   /* [N] */
    const float *mat_bias_scale, /* Scales for bias */
    float *mat_C,                /* [M, N] */
    size_t M, size_t N, size_t K,
    bool mat_B_transpose,
    size_t group_size            /* e.g., 32 or 64 */
);
#endif
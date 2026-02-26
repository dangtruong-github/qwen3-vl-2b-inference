#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"
#include "simd_utils.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void gemv_lg_N_K(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
);
void gemv_lg_N_K_g128(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K
);
void gemv_lg_N_K_decode(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
);
#endif

#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void att_fp32_full_avx2_kernel(
    const float *mat_A, const float *mat_B, float *mat_C,
    const float scale, size_t N, size_t K, bool mat_B_transpose
);
#endif
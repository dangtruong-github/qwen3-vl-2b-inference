#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void fp16_full_avx2_kernel(
    const half_cpu *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    half_cpu *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
);
#endif
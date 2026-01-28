#pragma once

#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include <sched.h> // for sched_setaffinity(), cpu_set_t, CPU_ZERO, CPU_SET
#include "../../utils/module.hpp"
#include "cpu_f32a_f16b_f32c_avx2_untranspose.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void f32a_f16bc_avx2_kernel(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    half_cpu *__restrict mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose
);
#endif

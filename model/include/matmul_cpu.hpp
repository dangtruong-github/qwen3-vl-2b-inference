#pragma once

#include <stdio.h>
#include <omp.h>
#include "../../utils/module.hpp"

void linear(
    const float *mat_A, const float *mat_B, const float *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
);
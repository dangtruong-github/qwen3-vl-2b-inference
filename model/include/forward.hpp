#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <cmath>

#include "../config.hpp"
#include "../../layers/module.hpp"

// ================================================================
// Forward Functions
// ================================================================
void forward_img(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight, float *img_data,
    int img_h, int img_w, int grid_h, int grid_w, bool warm_up = false
);
void forward_text_prefill(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight, int *token_list,
    const size_t prefill_size, size_t pos, bool warm_up = false
); 
float *forward_text_decode(
    QwenConfig *config, QwenRunState *state, QwenWeight *weight,
    int token_id, size_t pos, bool warm_up = false
);

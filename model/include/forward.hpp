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
void forward_img(QwenConfig *config, QwenRunState *state, QwenWeight *weight, float *img_data, int img_h, int img_w, int grid_h, int grid_w);
float *forward_text(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token, size_t pos);

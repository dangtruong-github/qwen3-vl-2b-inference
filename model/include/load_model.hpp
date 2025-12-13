#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../config.hpp"

void init_model_weights(const char* path, QwenConfig* config, QwenWeight* weights);
void init_model_run_state(QwenRunState* state, const QwenConfig* config);

void free_model_weights(QwenWeight* weights);
void free_model_run_state(QwenRunState* state);
void qwen_rope_precompute(
    float *cos_all_out,  // (seq_len * head_dim/2)
    float *sin_all_out,  // (seq_len * head_dim/2)
    const QwenConfig *config
);
void qwen_vision_rope_precompute(float *vision_freqs, const QwenConfig *config);
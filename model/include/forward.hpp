#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <cmath>

#include "../config.hpp"
#include "layer.hpp"

// ================================================================
// Forward Functions
// ================================================================
void forward_image_encoder(QwenRunState* state, const QwenWeight* weights, const float* image);
void forward_language(QwenRunState* state, const QwenWeight* weights, const int* input_tokens, int n_tokens);
void forward_transformer(QwenRunState* state, const QwenWeight* weights);
void extract_image_patches(const float* img, float* patches, const QwenConfig* config);

// Utility functions
void matrix_multiply_add(float* output, const float* input, const float* weight,
                        const float* bias, int rows, int cols, int inner_dim);
void softmax(float* output, const float* input, int size);
float *forward_llm(QwenConfig *config, QwenRunState *state, QwenWeight *weight, int token, int pos);

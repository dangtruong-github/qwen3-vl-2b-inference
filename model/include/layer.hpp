#pragma once
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cstring>

// Forward declarations for helper functions
void layer_norm(float* buffer, const float* input, const float* weight, const float* bias,
                int hidden_size, float eps);
float gelu(float x);
int argmax(const float* array, int size);



// ------------------------ Helper functions ------------------------


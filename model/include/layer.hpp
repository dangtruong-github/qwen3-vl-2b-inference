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
void embedding_lookup(
    const float *embedding /*[vocab, hidden]*/,
    int token_id, float *out /*[hidden]*/,
    size_t vocab_size, size_t hidden
);
void rms_norm(
    const float *x /*[hidden]*/, const float *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps, size_t hidden_size
);
void classifier_gemm(
    const float *embedding /*[vocab, hidden]*/,
    const float *hid_states /*[hidden]*/, float *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
);
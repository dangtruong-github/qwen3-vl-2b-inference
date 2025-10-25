#include "../include/layer.hpp"

void layer_norm(float* buffer, const float* input, const float* weight,
                const float* bias, int hidden_size, float eps) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < hidden_size; ++i) mean += input[i];
    mean /= hidden_size;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;
    float inv_std = 1.f / sqrtf(var + eps);
    for (int i = 0; i < hidden_size; ++i) {
        buffer[i] = (input[i] - mean) * inv_std;
        if (weight) buffer[i] *= weight[i];
        if (bias) buffer[i] += bias[i];
    }
    memcpy((void*)input, buffer, hidden_size * sizeof(float)); // overwrite
}

float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.79788456f * x * (1.f + 0.044715f * x * x)));
}

int argmax(const float* array, int size) {
    int idx = 0;
    float max_val = array[0];
    for (int i = 1; i < size; ++i) {
        if (array[i] > max_val) {
            max_val = array[i];
            idx = i;
        }
    }
    return idx;
}

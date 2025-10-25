#include "../include/utils.hpp"

int greedy_decode(float* logits, int vocab_size) {
    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

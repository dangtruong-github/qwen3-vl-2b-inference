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

void tokenizer_example(TokenizerStruct *tokenizer) {
    // ---
    // Print Vocabulary (vocab)
    // ---
    printf("========================================\n");
    printf("VOCABULARY (Size: %d)\n", tokenizer->vocab_size);
    printf("========================================\n");

    for (int i = 0; i < tokenizer->vocab_size; i++) {
        // Assuming vocab[i] is a valid C-style string (char*)
        printf("Index %-5d | Token: \"%s\"\n", i, tokenizer->vocab[i]);
    }

    // ---
    // Print Merges (merges)
    // ---
    printf("\n");
    printf("========================================\n");
    printf("MERGES (Size: %d)\n", tokenizer->merges_size);
    printf("========================================\n");

    for (int i = 0; i < tokenizer->merges_size; i++) {
        // Assuming merges[i] is a valid C-style string (char*) representing a merge pair
        printf("Merge %-5d | Pair: \"%s\"\n", i, tokenizer->merges[i]);
    }
    
    printf("\n");
    printf("--- General Info ---\n");
    printf("Max Token Length: %u\n", tokenizer->max_token_length);
    printf("--------------------\n");
}

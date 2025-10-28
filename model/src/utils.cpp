#include "../include/utils.hpp"

void forward_example(QwenConfig *config, QwenWeight *weights, QwenRunState* state) {
    // Example inputs
    float* image = nullptr; // load_image_as_float("example.jpg"); // Implement your own
    int input_tokens[] = {101, 234, 543, 99};          // Already tokenized text
    int n_tokens = sizeof(input_tokens) / sizeof(int);

    // Forward pass
    forward_image_encoder(state, weights, image);
    forward_language(state, weights, input_tokens, n_tokens);
    forward_transformer(state, weights);

    // Get final logits
    float* logits = state->logits; // [vocab_size]

    printf("Logits: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", logits[i]);
    }
    printf("\n");

    // Greedy decoding
    int next_token = greedy_decode(logits, config->vocab_size);
    printf("Predicted token: %d\n", next_token);

    printf("Finished!\n");
    if (image) free(image);
}
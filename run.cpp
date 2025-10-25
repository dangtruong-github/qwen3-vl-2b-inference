#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "model/include/load_model.hpp"
#include "model/include/forward.hpp"
#include "model/config.hpp"
#include "tokenizer/include/utils.hpp"

// ------------------------------------------------------------
// Main entry
// ------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s --model_path <path/to/model.bin>\n", argv[0]);
        return 1;
    }

    const char* model_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model_path required\n");
        return 1;
    }

    // Initialize
    QwenConfig *config = new QwenConfig;
    QwenWeight *weights = new QwenWeight;
    QwenRunState *state = new QwenRunState;

    init_model_weights(model_path, config, weights);
    init_model_run_state(state, config);

    printf("Model initialized from: %s\n", model_path);
    fflush(stdout);


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

    // Cleanup
    free_model_run_state(state);
    free_model_weights(weights);
    if (image) free(image);
    delete config;
    delete weights;
    delete state;

    printf("Finished!\n");
    return 0;
}

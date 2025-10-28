#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "model/module.hpp"
#include "tokenizer/module.hpp"

int main(int argc, char** argv) {
    // Check if there are enough arguments for both paths (at least 5: program name, -flag1, path1, -flag2, path2)
    if (argc < 5) {
        printf("Usage: %s --model_path <path/to/model.bin> --tokenizer_path <path/to/tokenizer.bin>\n", argv[0]);
        return 1;
    }

    const char* model_path = NULL;
    const char* tokenizer_path = NULL;

    // Iterate through command-line arguments to find both paths
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_path") == 0) {
            if (i + 1 < argc) {
                model_path = argv[i + 1];
                i++; // Skip the next argument as it's the path value
            } else {
                fprintf(stderr, "Error: Value missing after --model_path\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--tokenizer_path") == 0) {
            if (i + 1 < argc) {
                tokenizer_path = argv[i + 1];
                i++; // Skip the next argument as it's the path value
            } else {
                fprintf(stderr, "Error: Value missing after --tokenizer_path\n");
                return 1;
            }
        }
    }

    // Check if both paths were successfully found
    if (!model_path) {
        fprintf(stderr, "Error: --model_path required\n");
        return 1;
    }
    if (!tokenizer_path) {
        fprintf(stderr, "Error: --tokenizer_path required\n");
        return 1;
    }

    // Example of using the paths (you'd put the rest of your program logic here)
    printf("Model path: %s\n", model_path);
    printf("Tokenizer path: %s\n", tokenizer_path);

    // Initialize
    QwenConfig *config = new QwenConfig;
    QwenWeight *weights = new QwenWeight;
    QwenRunState *state = new QwenRunState;
    TokenizerStruct* tokenizer = new TokenizerStruct;

    init_tokenizer(tokenizer, tokenizer_path);
    init_model_weights(model_path, config, weights);
    init_model_run_state(state, config);

    printf("Model initialized from: %s\n", model_path);
    fflush(stdout);

    forward_example(config, weights, state);
    // tokenizer_example(tokenizer);

    // Cleanup
    free_model_run_state(state);
    free_model_weights(weights);
    free_tokenizer(tokenizer);
    delete config;
    delete weights;
    delete state;
    delete tokenizer;

    return 0;
}

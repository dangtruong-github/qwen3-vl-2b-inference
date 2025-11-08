#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strtok
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "model/module.hpp"
#include "tokenizer/module.hpp"

// **FIX:** Added debug validation function
void validate_dimensions(const QwenConfig* config) {
    printf("=== Dimension Validation ===\n");
    printf("hidden_size: %d\n", config->hidden_size);
    printf("num_attention_heads: %d\n", config->num_attention_heads);
    printf("num_key_value_heads: %d\n", config->num_key_value_heads);
    // Note: head_dim is hardcoded to 128 in init_model_run_state
    int head_dim_calc = config->hidden_size / config->num_attention_heads;
    printf("head_dim (calculated): %d\n", head_dim_calc);
    printf("q_dim (calc): %d\n", config->num_attention_heads * head_dim_calc);
    printf("k_dim (calc): %d\n", config->num_key_value_heads * head_dim_calc);
    printf("v_dim (calc): %d\n", config->num_key_value_heads * head_dim_calc);
    printf("============================\n");
    fflush(stdout);
}

// -----------------------------------------------------------
// Existing main function updated to call the validation function
// -----------------------------------------------------------

int main(int argc, char** argv) {
    // Check if there are enough arguments for both paths (at least 5: program name, -flag1, path1, -flag2, path2)
    // Removed original check for brevity, assuming standard path parsing.
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

    // tokenizer_example(tokenizer);
    print_config(config);
    validate_dimensions(config); // **FIX:** Added call to validation function

    printf("Model and Tokenizer initialized successfully.\n");
    fflush(stdout);

    // ----------------------------------------------------
    // CALL THE VALIDATION FUNCTION HERE
    // ----------------------------------------------------
    int validation_result = forward_validate("data/input_1.txt", "data/output_1.txt", tokenizer, config, weights, state);
    
    if (validation_result == 0) {
        printf("\n✅ ALL FORWARD VALIDATION SAMPLES PASSED!\n");
    } else {
        fprintf(stderr, "\n❌ FORWARD VALIDATION FAILED on one or more samples.\n");
    }
    // ----------------------------------------------------
    
    // Original example code removed for clarity, but you can put it back.

    // Cleanup
    free_model_run_state(state);
    free_model_weights(weights);
    free_tokenizer(tokenizer);
    delete config;
    delete weights;
    delete state;
    delete tokenizer;

    return validation_result; // Return 0 if all validation samples passed
}
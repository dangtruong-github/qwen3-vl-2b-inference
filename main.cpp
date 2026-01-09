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
    if (argc < 5) {
        printf(
            "Usage: %s "
            "--model_path <path/to/model.bin> "
            "--tokenizer_path <path/to/tokenizer.bin> "
            "[--model_dtype fp32|fp16]\n",
            argv[0]
        );
        return 1;
    }

    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    const char* model_dtype = "fp32";  // default

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model_path") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                fprintf(stderr, "Error: Value missing after --model_path\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--tokenizer_path") == 0) {
            if (i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else {
                fprintf(stderr, "Error: Value missing after --tokenizer_path\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--model_dtype") == 0) {
            if (i + 1 < argc) {
                model_dtype = argv[++i];
                if (strcmp(model_dtype, "fp32") != 0 &&
                    strcmp(model_dtype, "fp16") != 0) {
                    fprintf(stderr,
                        "Error: --model_dtype must be 'fp32' or 'fp16'\n");
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: Value missing after --model_dtype\n");
                return 1;
            }
        }
    }

    if (!model_path || !tokenizer_path) {
        fprintf(stderr,
            "Error: --model_path and --tokenizer_path are required\n");
        return 1;
    }

    printf("Model path: %s\n", model_path);
    printf("Tokenizer path: %s\n", tokenizer_path);
    printf("Model dtype: %s\n", model_dtype);

    // Initialize
    QwenConfig *config = new QwenConfig;
    QwenWeight *weights = new QwenWeight;
    QwenRunState *state = new QwenRunState;
    TokenizerStruct* tokenizer = new TokenizerStruct;

    DType::Type dtype_weight;
    if (strcmp(model_dtype, "fp32") == 0) {
        dtype_weight = DType::FP32;
    } else if (strcmp(model_dtype, "fp16") == 0) {
        dtype_weight = DType::FP16;
    }

    init_tokenizer(tokenizer, tokenizer_path);
    init_model_weights(model_path, config, weights, dtype_weight);
    init_model_run_state(state, config);

    // tokenizer_example(tokenizer);
    print_config(config);
    validate_dimensions(config); // **FIX:** Added call to validation function

    printf("Model and Tokenizer initialized successfully.\n");
    fflush(stdout);

    // ----------------------------------------------------
    // CALL THE VALIDATION FUNCTION HERE
    // ----------------------------------------------------
    int validation_result = forward_validate("data/input_1.txt", "data/image_path.txt", "data/output_1.txt", tokenizer, config, weights, state); // forward_validate("data/input_tokens.txt", "data/image_path.txt", "data/output_tokens_gen_truth.txt", tokenizer, config, weights, state);
    if (validation_result == 0) {
        printf("\n✅ ALL FORWARD VALIDATION SAMPLES PASSED!\n");
    } else {
        fprintf(stderr, "\n❌ FORWARD VALIDATION FAILED on one or more samples.\n");
    }
    // forward_generate("data/input_tokens.txt", "data/image_path.txt", "data/output_tokens_gen.txt", tokenizer, config, weights, state);
    // image_processor_validate("data/image_path.txt",tokenizer, config, weights, state);
    
    
    // ----------------------------------------------------
    
    // Original example code removed for clarity, but you can put it back.

    // Cleanup

    printf("STARTING FREE\n");
    fflush(stdout);
    free_model_run_state(state);
    printf("FINISH FREE state\n");
    fflush(stdout);
    free_model_weights(weights);
    printf("FINISH FREE weights\n");
    fflush(stdout);
    free_model_config(config);
    printf("FINISH FREE config\n");
    fflush(stdout);
    free_tokenizer(tokenizer);
    printf("FINISH FREE tokenizer\n");
    fflush(stdout);
    delete config;
    printf("FINISH DELETE config\n");
    fflush(stdout);
    delete weights;
    printf("FINISH DELETE weights\n");
    fflush(stdout);
    delete state;
    printf("FINISH DELETE state\n");
    fflush(stdout);
    delete tokenizer;
    printf("FINISH FREE AND DELETE\n");
    fflush(stdout);

    return 0; // Return 0 if all validation samples passed
}
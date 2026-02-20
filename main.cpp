#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "model/module.hpp"
#include "tokenizer/module.hpp"

void validate_dimensions(const QwenConfig* config) {
    printf("=== Dimension Validation ===\n");
    printf("hidden_size: %d\n", config->hidden_size);
    printf("num_attention_heads: %d\n", config->num_attention_heads);
    printf("num_key_value_heads: %d\n", config->num_key_value_heads);

    int head_dim_calc = config->hidden_size / config->num_attention_heads;
    printf("head_dim (calculated): %d\n", head_dim_calc);
    printf("q_dim (calc): %d\n", config->num_attention_heads * head_dim_calc);
    printf("k_dim (calc): %d\n", config->num_key_value_heads * head_dim_calc);
    printf("v_dim (calc): %d\n", config->num_key_value_heads * head_dim_calc);
    printf("============================\n");
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);

    if (argc < 11) {
        printf(
            "Usage: %s "
            "--model_path <model.bin> "
            "--tokenizer_path <tokenizer.bin> "
            "--input_path <input.txt> "
            "--image_path <image.txt> "
            "--output_val_path <output.txt>\n",
            argv[0]
        );
        return 1;
    }

    const char* model_path = nullptr;
    const char* tokenizer_path = nullptr;
    const char* input_path = nullptr;
    const char* image_path = nullptr;
    const char* output_val_path = nullptr;

    auto parse_arg = [&](const char* flag, const char*& var, int &i) {
        if (strcmp(argv[i], flag) == 0) {
            if (i + 1 < argc) {
                var = argv[++i];  // advances main loop index
            } else {
                fprintf(stderr, "Missing value after %s\n", flag);
                exit(1);
            }
            return true;
        }
        return false;
    };

    for (int i = 1; i < argc; i++) {
        parse_arg("--model_path", model_path, i);
        parse_arg("--tokenizer_path", tokenizer_path, i);
        parse_arg("--input_path", input_path, i);
        parse_arg("--image_path", image_path, i);
        parse_arg("--output_val_path", output_val_path, i);
    }

    if (!model_path || !tokenizer_path || !input_path || !image_path || !output_val_path) {
        fprintf(stderr,
            "Error: --model_path, --tokenizer_path, --input_path, --image_path, and --output_val_path are required\n");
        return 1;
    }

    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        printf("AVX512 enabled\n");
    #elif defined(__AVX2__) && defined(__FMA__)
        printf("AVX2 enabled\n");
    #else
        printf("Default fallback, no AVX2 or AVX512\n");
    #endif

    printf("Model path: %s\n", model_path);
    printf("Tokenizer path: %s\n", tokenizer_path);
    printf("Input path: %s\n", input_path);
    printf("Image path: %s\n", image_path);
    printf("Output val path: %s\n", output_val_path);

    QwenConfig *config = new QwenConfig;
    QwenWeight *weights = new QwenWeight;
    QwenRunState *state = new QwenRunState;
    TokenizerStruct* tokenizer = new TokenizerStruct;

    init_tokenizer(tokenizer, tokenizer_path);
    init_model_weights(model_path, config, weights);
    init_model_run_state(state, config);

    print_config(config);
    validate_dimensions(config);

    printf("Model and Tokenizer initialized successfully.\n");

    warm_up(tokenizer, config, weights, state);

    int validation_result = forward_validate(
        input_path, image_path, output_val_path,
        tokenizer, config, weights, state
    );

    if (validation_result == 0) {
        printf("\n✅ ALL FORWARD VALIDATION SAMPLES PASSED!\n");
    } else {
        fprintf(stderr, "\n❌ FORWARD VALIDATION FAILED on one or more samples.\n");
    }

    printf("STARTING FREE\n");
    free_model_run_state(state);
    printf("FINISH FREE state\n");
    free_model_weights(weights);
    printf("FINISH FREE weights\n");
    free_model_config(config);
    printf("FINISH FREE config\n");
    free_tokenizer(tokenizer);
    printf("FINISH FREE tokenizer\n");
    delete config;
    printf("FINISH DELETE config\n");
    delete weights;
    printf("FINISH DELETE weights\n");
    delete state;
    printf("FINISH DELETE state\n");
    delete tokenizer;
    printf("FINISH FREE AND DELETE\n");

    return 0;
}
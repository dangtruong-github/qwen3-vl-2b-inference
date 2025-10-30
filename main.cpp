#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strtok
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "model/module.hpp"
#include "tokenizer/module.hpp"

char* read_full_line(FILE* f) {
    if (!f) return NULL;

    size_t bufsize = 8192;
    size_t len = 0;
    char* buffer = (char*)malloc(bufsize);
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for line buffer\n");
        exit(EXIT_FAILURE);
    }

    int c;
    while ((c = fgetc(f)) != EOF) {
        if (len + 1 >= bufsize) {
            bufsize *= 2;
            char* new_buffer = (char*)realloc(buffer, bufsize);
            if (!new_buffer) {
                fprintf(stderr, "Memory reallocation failed for line buffer\n");
                free(buffer);
                exit(EXIT_FAILURE);
            }
            buffer = new_buffer;
        }

        buffer[len++] = (char)c;
        if (c == '\n') break;
    }

    if (len == 0 && c == EOF) {
        free(buffer);
        return NULL;
    }

    buffer[len] = '\0';
    return buffer;
}

int tokenizer_validate(
    TokenizerStruct* tokenizer,
    const char* prompt_file_path,
    const char* tokens_file_path,
    const char* img_file_path,
    int patch_size
) {
    FILE* prompt_file = fopen(prompt_file_path, "r");
    FILE* tokens_file = fopen(tokens_file_path, "r");
    FILE* img_file = fopen(img_file_path, "r");

    if (!prompt_file) {
        fprintf(stderr, "Error: Could not open prompt file: %s\n", prompt_file_path);
        return 1;
    }
    if (!tokens_file) {
        fprintf(stderr, "Error: Could not open token IDs file: %s\n", tokens_file_path);
        fclose(prompt_file);
        return 1;
    }
    if (!img_file) {
        fprintf(stderr, "Error: Could not open image path file: %s\n", img_file_path);
        fclose(prompt_file);
        fclose(tokens_file);
        return 1;
    }

    printf("\nStarting Tokenizer Validation...\n");
    setbuf(stdout, NULL);

    int validation_failures = 0;
    int sample_count = 0;

    while (1) {
        char* prompt_line = read_full_line(prompt_file);
        char* tokens_line = read_full_line(tokens_file);
        char* img_line = read_full_line(img_file);

        if (!prompt_line || !tokens_line || !img_line) {
            free(prompt_line);
            free(tokens_line);
            free(img_line);
            break;
        }

        sample_count++;
        printf("Starting new cycle %d...\n", sample_count);
        int is_valid = 1;

        // ------------------------------------------------------------
        // 1. Parse expected tokens
        // ------------------------------------------------------------
        int expected_tokens[8192];
        int expected_count = 0;

        char* token_ptr = strtok(tokens_line, " \t\r\n");
        while (token_ptr && expected_count < 8192) {
            expected_tokens[expected_count++] = atoi(token_ptr);
            token_ptr = strtok(NULL, " \t\r\n");
        }

        // ------------------------------------------------------------
        // 2. Clean up input strings
        // ------------------------------------------------------------
        size_t len_prompt = strlen(prompt_line);
        if (len_prompt > 0 && (prompt_line[len_prompt - 1] == '\n' || prompt_line[len_prompt - 1] == '\r'))
            prompt_line[len_prompt - 1] = '\0';

        size_t len_img = strlen(img_line);
        if (len_img > 0 && (img_line[len_img - 1] == '\n' || img_line[len_img - 1] == '\r'))
            img_line[len_img - 1] = '\0';

        // ------------------------------------------------------------
        // 3. Encode
        // ------------------------------------------------------------
        int max_tokens = (int)strlen(prompt_line) + 100000;
        int* actual_tokens = (int*)malloc(max_tokens * sizeof(int));
        int num_actual_tokens = 0;

        if (!actual_tokens) {
            fprintf(stderr, "Error: Memory allocation failed for actual_tokens.\n");
            exit(EXIT_FAILURE);
        }

        encode(tokenizer, prompt_line, actual_tokens, &num_actual_tokens, img_line, patch_size);

        // ------------------------------------------------------------
        // 4. Compare Results
        // ------------------------------------------------------------
        if (num_actual_tokens != expected_count) {
            printf("\n❌ Sample %d FAILED: Token count mismatch.\n", sample_count);
            printf("  Prompt: %s\n", prompt_line);
            printf("  Expected Count: %d, Actual Count: %d\n", expected_count, num_actual_tokens);
            is_valid = 0;
        } else {
            for (int i = 0; i < num_actual_tokens; ++i) {
                if (actual_tokens[i] != expected_tokens[i]) {
                    printf("\n❌ Sample %d FAILED: Token ID mismatch at index %d.\n", sample_count, i);
                    printf("  Prompt: %s\n", prompt_line);
                    is_valid = 0;
                    break;
                }
            }
        }

        // ------------------------------------------------------------
        // 5. Print detailed mismatch if failed
        // ------------------------------------------------------------
        if (!is_valid) {
            validation_failures++;
            printf("  Expected Tokens (%d): ", expected_count);
            for (int i = 0; i < expected_count; ++i)
                printf("%d ", expected_tokens[i]);
            printf("\n");
            printf("  Actual Tokens (%d):   ", num_actual_tokens);
            for (int i = 0; i < num_actual_tokens; ++i)
                printf("%d ", actual_tokens[i]);
            printf("\n");
        } else {
            printf("✅ Sample %d validated successfully.\n", sample_count);
        }

        // Cleanup
        free(prompt_line);
        free(tokens_line);
        free(img_line);
        free(actual_tokens);

        printf("End of cycle %d\n", sample_count);
    }

    fclose(prompt_file);
    fclose(tokens_file);
    fclose(img_file);

    printf("\n--- Validation Summary ---\n");
    printf("Total Samples Processed: %d\n", sample_count);
    printf("Total Failures: %d\n", validation_failures);

    return (validation_failures > 0) ? 1 : 0;
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

    printf("Model and Tokenizer initialized successfully.\n");
    fflush(stdout);

    // ----------------------------------------------------
    // CALL THE VALIDATION FUNCTION HERE
    // ----------------------------------------------------
    int validation_result = tokenizer_validate(
        tokenizer, "data/input_text.txt", "data/input_tokens.txt",
        "data/image_path.txt", config->vision_patch_size);
    
    if (validation_result == 0) {
        printf("\n✅ ALL TOKENIZER VALIDATION SAMPLES PASSED!\n");
    } else {
        fprintf(stderr, "\n❌ TOKENIZER VALIDATION FAILED on one or more samples.\n");
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
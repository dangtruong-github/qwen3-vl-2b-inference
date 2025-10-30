#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strtok
#include <math.h>
#include <float.h>

// Note: Assuming these headers contain the necessary struct definitions and function prototypes
// for QwenConfig, QwenWeight, QwenRunState, TokenizerStruct,
// init_tokenizer, init_model_weights, init_model_run_state,
// free_model_run_state, free_model_weights, free_tokenizer,
// and the crucial: encode(tokenizer, prompt, prompt_tokens, num_prompt_tokens_ptr)
#include "model/module.hpp"
#include "tokenizer/module.hpp"

// Define a maximum buffer size for reading lines from files
#define MAX_LINE_LENGTH 4096

/**
 * @brief Validates the tokenizer by comparing the output of the encode function
 * with expected token IDs from two files: prompt.txt and token_ids.txt.
 *
 * @param tokenizer A pointer to the initialized TokenizerStruct.
 * @param prompt_file_path Path to the file containing text prompts (one per line).
 * @param tokens_file_path Path to the file containing expected token IDs (space-separated integers, one line per prompt).
 * @return int 0 on successful validation of all samples, 1 if any sample fails validation or a file error occurs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int tokenizer_validate(TokenizerStruct* tokenizer,
                       const char* prompt_file_path,
                       const char* tokens_file_path) {
    FILE* prompt_file = fopen(prompt_file_path, "r");
    FILE* tokens_file = fopen(tokens_file_path, "r");
    int validation_failures = 0;
    int sample_count = 0;

    if (!prompt_file) {
        fprintf(stderr, "Error: Could not open prompt file: %s\n", prompt_file_path);
        return 1;
    }
    if (!tokens_file) {
        fprintf(stderr, "Error: Could not open token IDs file: %s\n", tokens_file_path);
        fclose(prompt_file);
        return 1;
    }

    printf("\nStarting Tokenizer Validation...\n");
    setbuf(stdout, NULL);

    char prompt_line[8192];
    char tokens_line[8192];

    while (fgets(prompt_line, sizeof(prompt_line), prompt_file) &&
           fgets(tokens_line, sizeof(tokens_line), tokens_file)) {
        printf("Starting new cycle...\n");

        sample_count++;
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

        printf("Finish step 1 out\n");

        // ------------------------------------------------------------
        // 2. Encode the prompt
        // ------------------------------------------------------------
        int max_tokens = (int)strlen(prompt_line) + 8;
        int* actual_tokens = (int*)malloc(max_tokens * sizeof(int));
        int num_actual_tokens = 0;

        if (!actual_tokens) {
            fprintf(stderr, "Error: Memory allocation failed for actual_tokens.\n");
            abort();
        }

        // remove trailing newline
        size_t len = strlen(prompt_line);
        if (len > 0 && (prompt_line[len - 1] == '\n' || prompt_line[len - 1] == '\r'))
            prompt_line[len - 1] = '\0';

        char* prompt_cstr = (char*)malloc(strlen(prompt_line) + 2);
        if (!prompt_cstr) {
            fprintf(stderr, "Error: Memory allocation failed for prompt_cstr.\n");
            abort();
        }
        strcpy(prompt_cstr, prompt_line);

        encode(tokenizer, prompt_cstr, actual_tokens, &num_actual_tokens, NULL);

        printf("Finish step 2 out\n");

        // ------------------------------------------------------------
        // 3. Compare Results
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

        printf("Finish step 3 out\n");

        // ------------------------------------------------------------
        // 4. Print detailed mismatch if failed
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

        printf("Finish step 4 out\n");

        // ------------------------------------------------------------
        // Clean up with pointer sanity checks
        // ------------------------------------------------------------
        if (prompt_cstr) {
            uintptr_t addr = (uintptr_t)prompt_cstr;
            if (addr % alignof(char) != 0) {
                fprintf(stderr, "\nInvalid pointer alignment detected for prompt_cstr at sample %d\n", sample_count);
                abort();
            }
            free(prompt_cstr);
        }

        if (actual_tokens) {
            uintptr_t addr = (uintptr_t)actual_tokens;
            if (addr % alignof(int) != 0) {
                fprintf(stderr, "\nInvalid pointer alignment detected for actual_tokens at sample %d\n", sample_count);
                abort();
            }
            free(actual_tokens);
        }

        printf("End of cycle %d\n", sample_count);
    }

    // Check if files ended simultaneously
    int prompt_eof = feof(prompt_file);
    int tokens_eof = feof(tokens_file);
    if (prompt_eof != tokens_eof) {
        fprintf(stderr, "\nWarning: prompt.txt and token_ids.txt do not have the same number of lines. Validation may be incomplete.\n");
    }

    fclose(prompt_file);
    fclose(tokens_file);

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
    int validation_result = tokenizer_validate(tokenizer, "data/input_text_null.txt", "data/input_tokens_null.txt");
    
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
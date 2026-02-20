#include "../include/test_utils.hpp"

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

void get_expected_tokens(const char *tokens_line, int **out_tokens, int *out_count) {
    if (!tokens_line || !out_tokens || !out_count) {
        fprintf(stderr, "get_expected_tokens: invalid arguments\n");
        exit(EXIT_FAILURE);
    }

    // Duplicate the input string since strtok modifies it
    char *line_copy = strdup(tokens_line);
    if (!line_copy) {
        fprintf(stderr, "Memory allocation failed in get_expected_tokens\n");
        exit(EXIT_FAILURE);
    }

    int capacity = 8192;
    int *tokens = (int *)malloc(capacity * sizeof(int));
    if (!tokens) {
        fprintf(stderr, "Memory allocation failed for tokens\n");
        free(line_copy);
        exit(EXIT_FAILURE);
    }

    int count = 0;
    char *token_ptr = strtok(line_copy, " \t\r\n");

    while (token_ptr) {
        if (count >= capacity) {
            capacity *= 2;
            int *tmp = (int *)realloc(tokens, capacity * sizeof(int));
            if (!tmp) {
                fprintf(stderr, "Memory reallocation failed\n");
                free(tokens);
                free(line_copy);
                exit(EXIT_FAILURE);
            }
            tokens = tmp;
        }

        tokens[count++] = atoi(token_ptr);
        token_ptr = strtok(NULL, " \t\r\n");
    }

    *out_tokens = tokens;
    *out_count = count;
    free(line_copy);
}

void convert_endl(char **prompt_line) {
    if (prompt_line == NULL || *prompt_line == NULL)
        return;

    char *src = *prompt_line;
    size_t len = strlen(src);

    // Allocate new buffer (same size is enough since result is shorter)
    char *dst = (char *)malloc((len + 1) * sizeof(char));
    if (!dst) {
        fprintf(stderr, "Memory allocation failed in convert_endl\n");
        exit(EXIT_FAILURE);
    }

    char *out = dst;
    while (*src) {
        if (src[0] == '\\' && src[1] == 'n') {
            *out++ = '\n';
            src += 2; // skip both '\' and 'n'
        } else {
            *out++ = *src++;
        }
    }
    *out = '\0';

    free(*prompt_line);
    *prompt_line = dst;
}

int tokenizer_validate(
    TokenizerStruct* tokenizer,
    QwenConfig *config,
    const char* prompt_file_path,
    const char* tokens_file_path,
    const char* img_file_path,
    int patch_size, int merge_size
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

        convert_endl(&prompt_line);

        sample_count++;
        printf("Starting new cycle %d...\n", sample_count);
        int is_valid = 1;

        // ------------------------------------------------------------
        // 1. Parse expected tokens
        // ------------------------------------------------------------
        int *expected_tokens = NULL;
        int expected_count = 0;

        get_expected_tokens(tokens_line, &expected_tokens, &expected_count);

        // Use expected_tokens[0..expected_count-1]
        // ...

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

        encode(tokenizer, config, prompt_line, actual_tokens, &num_actual_tokens, img_line, patch_size, merge_size);

        // ------------------------------------------------------------
        // 4. Compare Results
        // ------------------------------------------------------------
        if (num_actual_tokens != expected_count) {
            printf("\n❌ Sample %d FAILED: Token count mismatch.\n", sample_count);
            // printf("  Prompt: %s\n", prompt_line);
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
            for (int i = 0; i < expected_count; ++i) {
                if (expected_tokens[i] != 151655) {   
                    printf("%s ", tokenizer->vocab[expected_tokens[i]]);
                }
            }
            printf("\n");
            printf("  Actual Tokens (%d):   ", num_actual_tokens);
            for (int i = 0; i < num_actual_tokens; ++i) {
                if (actual_tokens[i] != 151655) {   
                    printf("%s ", tokenizer->vocab[actual_tokens[i]]);
                }
            }
            printf("\n");
        } else {
            printf("  Expected Tokens (%d): ", expected_count);
            printf("\n");
            printf("  Actual Tokens (%d):   ", num_actual_tokens);
            printf("\n");
            printf("✅ Sample %d validated successfully.\n", sample_count);
        }

        // Cleanup
        free(prompt_line);
        free(tokens_line);
        free(img_line);
        free(actual_tokens);
        free(expected_tokens);  // don't forget to free after use

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

int decode_validate(
    TokenizerStruct* tokenizer,
    const char* prompt_file_path,
    const char* tokens_file_path
) {
    FILE* prompt_file = fopen(prompt_file_path, "r");
    FILE* tokens_file = fopen(tokens_file_path, "r");

    if (!prompt_file) {
        fprintf(stderr, "Error: Could not open prompt file: %s\n", prompt_file_path);
        return 1;
    }
    if (!tokens_file) {
        fprintf(stderr, "Error: Could not open token IDs file: %s\n", tokens_file_path);
        fclose(prompt_file);
        return 1;
    }

    printf("\nStarting Decode Validation...\n");

    int validation_failures = 0;
    int sample_count = 0;

    while (1) {
        char* prompt_line = read_full_line(prompt_file);
        char* tokens_line = read_full_line(tokens_file);

        if (!prompt_line || !tokens_line) {
            free(prompt_line);
            free(tokens_line);
            break;
        }

        convert_endl(&prompt_line);
        sample_count++;
        printf("Starting new cycle %d...\n", sample_count);

        // ------------------------------------------------------------
        // 1. Parse token IDs
        // ------------------------------------------------------------
        int *tokens = NULL;
        int n_tokens = 0;
        get_expected_tokens(tokens_line, &tokens, &n_tokens);

        // ------------------------------------------------------------
        // 2. Clean expected prompt text
        // ------------------------------------------------------------
        size_t len_prompt = strlen(prompt_line);
        if (len_prompt > 0 && (prompt_line[len_prompt - 1] == '\n' || prompt_line[len_prompt - 1] == '\r'))
            prompt_line[len_prompt - 1] = '\0';

        // ------------------------------------------------------------
        // 3. Allocate decode buffer safely
        // ------------------------------------------------------------
        size_t buf_size = len_prompt * 6 + 8192;
        char *decoded_text = (char*)calloc(1, buf_size);
        if (!decoded_text) {
            fprintf(stderr, "Error: Memory allocation failed for decoded_text.\n");
            exit(EXIT_FAILURE);
        }

        // ------------------------------------------------------------
        // 4. Decode
        // ------------------------------------------------------------
        decode(tokenizer, decoded_text, buf_size, tokens, n_tokens);

        // ------------------------------------------------------------
        // 5. Compare decoded text with expected
        // ------------------------------------------------------------
        int is_valid = 1;
        if (strcmp(prompt_line, decoded_text) != 0) {
            is_valid = 0;
            validation_failures++;

            printf("\n❌ Sample %d FAILED: Decoded text mismatch.\n", sample_count);
            printf("  Expected: \"%s\"\n", prompt_line);
            printf("  Decoded:  \"%s\"\n", decoded_text);

            // Optional: show first differing character
            size_t mismatch_idx = 0;
            while (prompt_line[mismatch_idx] &&
                   decoded_text[mismatch_idx] &&
                   prompt_line[mismatch_idx] == decoded_text[mismatch_idx])
                mismatch_idx++;

            printf("  Diff starts at index %zu:\n", mismatch_idx);
            printf("    expected: '%c' (0x%02X)\n", prompt_line[mismatch_idx], (unsigned char)prompt_line[mismatch_idx]);
            printf("    decoded:  '%c' (0x%02X)\n", decoded_text[mismatch_idx], (unsigned char)decoded_text[mismatch_idx]);
        } else {
            printf("✅ Sample %d validated successfully.\n", sample_count);
        }

        // ------------------------------------------------------------
        // 6. Cleanup
        // ------------------------------------------------------------
        free(prompt_line);
        free(tokens_line);
        free(tokens);
        free(decoded_text);

        printf("End of cycle %d\n", sample_count);
    }

    fclose(prompt_file);
    fclose(tokens_file);

    printf("\n--- Decode Validation Summary ---\n");
    printf("Total Samples Processed: %d\n", sample_count);
    printf("Total Failures: %d\n", validation_failures);
    
    return (validation_failures > 0) ? 1 : 0;
}
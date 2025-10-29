#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For strcmp, strtok
#include <math.h>
#include <float.h>
#include <vector> // Using std::vector for dynamic arrays in C++
#include <iostream> // For std::cout, std::cerr
#include <fstream> // For std::ifstream
#include <sstream> // For std::stringstream

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
int tokenizer_validate(TokenizerStruct* tokenizer, const char* prompt_file_path, const char* tokens_file_path) {
    std::ifstream prompt_file(prompt_file_path);
    std::ifstream tokens_file(tokens_file_path);
    int validation_failures = 0;
    int sample_count = 0;

    if (!prompt_file.is_open()) {
        std::cerr << "Error: Could not open prompt file: " << prompt_file_path << std::endl;
        return 1;
    }
    if (!tokens_file.is_open()) {
        std::cerr << "Error: Could not open token IDs file: " << tokens_file_path << std::endl;
        return 1;
    }

    std::string prompt_line;
    std::string tokens_line;

    std::cout << "\nStarting Tokenizer Validation..." << std::endl;

    // Read both files line by line simultaneously
    while (std::getline(prompt_file, prompt_line) && std::getline(tokens_file, tokens_line)) {
        sample_count++;
        bool is_valid = true;

        // 1. Get Expected Tokens
        std::vector<int> expected_tokens;
        std::stringstream ss(tokens_line);
        int token_id;
        while (ss >> token_id) {
            expected_tokens.push_back(token_id);
        }

        // 2. Encode the Prompt
        // Allocate a buffer large enough for the encoded tokens
        // A generous estimate is prompt length + a margin (e.g., 100)
        int max_tokens = prompt_line.length() + 100;
        int* actual_tokens = (int*)malloc(max_tokens * sizeof(int));
        int num_actual_tokens = 0;

        if (!actual_tokens) {
            std::cerr << "Error: Memory allocation failed for actual_tokens." << std::endl;
            return 1;
        }
        
        // C-style string required for the existing `encode` function
        char* prompt_cstr = (char*)prompt_line.c_str();
        encode(tokenizer, prompt_cstr, actual_tokens, max_tokens, &num_actual_tokens);

        // 3. Compare Results
        if (num_actual_tokens != expected_tokens.size()) {
            std::cout << "\n❌ Sample " << sample_count << " FAILED: Token count mismatch." << std::endl;
            std::cout << "  Prompt: " << prompt_line << std::endl;
            std::cout << "  Expected Count: " << expected_tokens.size() << ", Actual Count: " << num_actual_tokens << std::endl;
            is_valid = false;
        } else {
            for (size_t i = 0; i < num_actual_tokens; ++i) {
                if (actual_tokens[i] != expected_tokens[i]) {
                    std::cout << "\n❌ Sample " << sample_count << " FAILED: Token ID mismatch at index " << i << "." << std::endl;
                    std::cout << "  Prompt: " << prompt_line << std::endl;
                    is_valid = false;
                    break;
                }
            }
        }

        // 4. Output detailed mismatch if failed
        if (!is_valid) {
            validation_failures++;
            
            // Print Expected Tokens
            std::cout << "  Expected Tokens (" << expected_tokens.size() << "): ";
            for (int token : expected_tokens) {
                std::cout << token << " ";
            }
            std::cout << std::endl;

            // Print Actual Tokens
            std::cout << "  Actual Tokens (" << num_actual_tokens << "):   ";
            for (int i = 0; i < num_actual_tokens; ++i) {
                std::cout << actual_tokens[i] << " ";
            }
            std::cout << std::endl;
        } else {
            // Optional: Print a success message for each sample
            std::cout << "✅ Sample " << sample_count << " validated successfully." << std::endl;
        }

        // Clean up the allocated buffer for the current sample
        free(actual_tokens);
    }

    // Check if the files ended simultaneously (i.e., same number of lines)
    if (prompt_file.eof() != tokens_file.eof()) {
        std::cerr << "\nWarning: prompt.txt and token_ids.txt do not have the same number of lines. Validation may be incomplete." << std::endl;
    }

    std::cout << "\n--- Validation Summary ---" << std::endl;
    std::cout << "Total Samples Processed: " << sample_count << std::endl;
    std::cout << "Total Failures: " << validation_failures << std::endl;

    return validation_failures > 0 ? 1 : 0;
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

    printf("Model and Tokenizer initialized successfully.\n");
    fflush(stdout);

    // ----------------------------------------------------
    // CALL THE VALIDATION FUNCTION HERE
    // ----------------------------------------------------
    int validation_result = tokenizer_validate(tokenizer, "data/input_text.txt", "data/input_tokens.txt");
    
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
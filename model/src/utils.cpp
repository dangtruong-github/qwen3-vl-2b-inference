#include "../include/utils.hpp"

void forward_example(QwenConfig *config, QwenWeight *weights, QwenRunState* state) {
    // Example inputs
    float* image = nullptr; // load_image_as_float("example.jpg"); // Implement your own
    int input_tokens[] = {101, 234, 543, 99};          // Already tokenized text
    int n_tokens = sizeof(input_tokens) / sizeof(int);

    // Forward pass
    forward_image_encoder(state, weights, image);
    

    // Get final logits
    float* logits = forward_llm(config, state, weights, input_tokens[0], 0); // [vocab_size]

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

void print_config(QwenConfig *config) {
    printf("vocab_size: %d\n", config->vocab_size);
    printf("hidden_size: %d\n", config->hidden_size);
    printf("intermediate_size: %d\n", config->intermediate_size);
    printf("num_hidden_layers: %d\n", config->num_hidden_layers);
    printf("num_attention_heads: %d\n", config->num_attention_heads);
    printf("num_key_value_heads: %d\n", config->num_key_value_heads);
    printf("max_position_embeddings: %d\n", config->max_position_embeddings);
    printf("rope_theta: %d\n", config->rope_theta);
    printf("rms_norm_eps: %f\n", config->rms_norm_eps);
    printf("vision_hidden_size: %d\n", config->vision_hidden_size);
    printf("vision_depth: %d\n", config->vision_depth);
    printf("vision_patch_size: %d\n", config->vision_patch_size);
    printf("vision_spatial_merge_size: %d\n", config->vision_spatial_merge_size);
    printf("vision_temporal_patch_size: %d\n", config->vision_temporal_patch_size);
    printf("vision_num_heads: %d\n", config->vision_num_heads);
    printf("vision_intermediate_size: %d\n", config->vision_intermediate_size);
    printf("out_hidden_size: %d\n", config->out_hidden_size);
    printf("image_token_id: %d\n", config->image_token_id);
    printf("vision_start_token_id: %d\n", config->vision_start_token_id);
    printf("vision_end_token_id: %d\n", config->vision_end_token_id);
    printf("video_token_id: %d\n", config->video_token_id);
}

int forward_validate(const char *in_token_file, const char *out_token_file, QwenConfig *config, QwenWeight *weight, QwenRunState *state) {
    FILE* in_file = fopen(in_token_file, "r");
    FILE* out_file = fopen(out_token_file, "r");

    if (!in_file) {
        fprintf(stderr, "Error: Could not open input token file: %s\n", in_token_file);
        return 1;
    }
    if (!out_file) {
        fprintf(stderr, "Error: Could not open output token file: %s\n", out_token_file);
        fclose(in_file);
        return 1;
    }

    printf("\nStarting Forward Validation...\n");
    setbuf(stdout, NULL);

    int validation_failures = 0;
    int sample_count = 0;

    while (1) {
        char* in_line = read_full_line(in_file);
        char* out_line = read_full_line(out_file);

        if (!in_line || !out_line) {
            free(in_line);
            free(out_line);
            break;
        }

        sample_count++;
        printf("Starting new forward cycle %d...\n", sample_count);

        int *input_tokens = NULL;
        int input_count = 0;
        get_expected_tokens(in_line, &input_tokens, &input_count);\

        int *expected_tokens = NULL;
        int expected_count = 0;
        get_expected_tokens(out_line, &expected_tokens, &expected_count);

        // ------------------------------------------------------------
        // 3. Reset state and run initial forward pass
        // ------------------------------------------------------------
        forward_image_encoder(state, weight, nullptr);  // No image
        
        // ------------------------------------------------------------
        // 4. Generation loop - matching the structure from run.cpp
        // ------------------------------------------------------------
        int *generated_tokens = (int*)malloc((input_count + 1024) * sizeof(int));
        int total_generated_count = 0;
        
        // Copy initial input tokens
        for (int i = 0; i < input_count; i++) {
            generated_tokens[total_generated_count++] = input_tokens[i];
        }

        // Start the main loop - similar to generate() in run.cpp
        int next; // will store the next token in the sequence
        int token = input_tokens[0]; // kick off with the first token in the prompt
        int pos = 0; // position in the sequence
        int im_end_count = 0;

        while (pos < 1024) { // max steps
            // Forward the transformer to get logits for the next token
            // Using your existing forward functions
            float *logits = forward_llm(config, state, weight, token, pos);

            // Advance the state machine - similar to run.cpp
            if (pos < input_count - 1) {
                // If we are still processing the input prompt, force the next prompt token
                next = input_tokens[pos + 1];
            } else {
                // Otherwise use greedy decoding (temperature=0 equivalent)
                next = greedy_decode(logits, config->vocab_size);
            }
            pos++;

            // Add generated token to sequence (only after processing prompt)
            if (pos >= input_count) {
                generated_tokens[total_generated_count++] = next;
            }

            printf("%d ", generated_tokens[pos]);
            printf("\nLogits: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", logits[i]);
            }
            printf("\n");

            // Data-dependent terminating condition - match your EOS tokens
            // Using the same condition as original forward_validate
            if (next == 151645) { // <im_end> token
                // Count consecutive im_end tokens like original
                im_end_count++;
                if (im_end_count >= 2) {
                    printf("  Stopping generation: <im_end> token encountered twice\n");
                    break;
                }
            }

            // Also check for other termination conditions
            if (total_generated_count >= expected_count) {
                break;
            }

            token = next; // Update token for next iteration
        }

        printf("  Generated %d total tokens (%d new tokens)\n", total_generated_count, total_generated_count - input_count);

        // ------------------------------------------------------------
        // 5. Compare generated sequence with expected sequence
        // ------------------------------------------------------------
        int min_length = total_generated_count < expected_count ? total_generated_count : expected_count;
        int is_valid = 1;
        
        for (int i = 0; i < min_length; i++) {
            if (generated_tokens[i] != expected_tokens[i]) {
                is_valid = 0;
                validation_failures++;
                
                printf("\n❌ Forward Sample %d FAILED: Token mismatch at position %d\n", sample_count, i);
                printf("  Expected token: %d\n", expected_tokens[i]);
                printf("  Generated token: %d\n", generated_tokens[i]);
                
                // Show context around the mismatch
                int start = i > 5 ? i - 5 : 0;
                int end = (i + 5) < min_length ? (i + 5) : min_length - 1;
                
                printf("  Context (expected): ");
                for (int j = start; j <= end; j++) {
                    if (j == i) printf("[%d] ", expected_tokens[j]);
                    else printf("%d ", expected_tokens[j]);
                }
                printf("\n");
                
                printf("  Context (generated): ");
                for (int j = start; j <= end; j++) {
                    if (j == i) printf("[%d] ", generated_tokens[j]);
                    else printf("%d ", generated_tokens[j]);
                }
                printf("\n");
                break;
            }
        }
        
        // Check if lengths match
        if (is_valid && total_generated_count != expected_count) {
            is_valid = 0;
            validation_failures++;
            printf("\n❌ Forward Sample %d FAILED: Length mismatch\n", sample_count);
            printf("  Expected length: %d\n", expected_count);
            printf("  Generated length: %d\n", total_generated_count);
        }

        if (is_valid) {
            printf("✅ Forward Sample %d validated successfully.\n", sample_count);
            printf("  Input length: %d tokens\n", input_count);
            printf("  Generated length: %d tokens\n", total_generated_count);
        }

        // ------------------------------------------------------------
        // 6. Cleanup
        // ------------------------------------------------------------
        free(in_line);
        free(out_line);
        free(input_tokens);
        free(expected_tokens);
        free(generated_tokens);

        printf("End of forward cycle %d\n", sample_count);
    }

    fclose(in_file);
    fclose(out_file);

    printf("\n--- Forward Validation Summary ---\n");
    printf("Total Samples Processed: %d\n", sample_count);
    printf("Total Failures: %d\n", validation_failures);
    
    return (validation_failures > 0) ? 1 : 0;
}

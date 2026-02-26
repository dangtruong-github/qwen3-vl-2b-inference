#include "../include/utils.hpp"

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
    printf("text_bits: %d\n", config->text_bits);
    printf("group_size: %d\n", config->group_size);
    printf("vision_bits: %d\n", config->vision_bits);
    printf("group_quantized: %d\n", config->group_quantized);
}

void print_token(TokenizerStruct *tokenizer, int token_id) {
    if (token_id != 151655) {
        print_normalized_utf8(tokenizer->vocab[token_id]);
    }
}

int forward_validate(const char *in_token_file, const char *in_img_path, const char *out_token_file, TokenizerStruct *tokenizer, QwenConfig *config, QwenWeight *weight, QwenRunState *state) {
    FILE* in_file = fopen(in_token_file, "r");
    FILE* in_img_file = fopen(in_img_path, "r");
    FILE* out_file = fopen(out_token_file, "r");

    if (!in_file) {
        fprintf(stderr, "Error: Could not open input token file: %s\n", in_token_file);
        return 1;
    }
    if (!in_img_file) {
        fprintf(stderr, "Error: Could not open input image path file: %s\n", in_img_path);
        fclose(in_file);
        return 1;
    }
    if (!out_file) {
        fprintf(stderr, "Error: Could not open output token file: %s\n", out_token_file);
        fclose(in_file);
        fclose(in_img_file);
        return 1;
    }

    printf("\nStarting Forward Validation...\n");

    int validation_failures = 0;
    int sample_count = 0;

    double first_tok_gen_time_total = 0.0;
    double gen_time_total = 0.0;
    int new_tokens_gen_num = 0;

    int max_seq_len = config->seq_len;

    while (1) {
        char* in_line = read_full_line(in_file);
        char* in_img_line = read_full_line(in_img_file);
        char* out_line = read_full_line(out_file);

        if (!in_line || !in_img_line || !out_line) {
            free(in_line);
            free(in_img_line);
            free(in_line);
            break;
        }

        size_t len = strlen(in_img_line);
        if (len > 0 && in_img_line[len - 1] == '\n') {
            in_img_line[len - 1] = '\0';
        }

        sample_count++;
        printf("Starting new forward cycle %d...\n", sample_count);

        int *input_tokens = NULL;
        int input_count = 0;
        get_expected_tokens(in_line, &input_tokens, &input_count);

        int *expected_tokens = NULL;
        int expected_count = 0;
        get_expected_tokens(out_line, &expected_tokens, &expected_count);

        if (input_count >= max_seq_len) continue;
        
        // if (sample_count <= 3) continue;
        // if (sample_count >= 5) continue;

        // ------------------------------------------------------------
        // 3. Reset state and run initial forward pass
        // ------------------------------------------------------------
        float *img_processed_output;
        int img_processed_h = 0, img_processed_w = 0;
        int img_grid_h = 0, img_grid_w = 0;
        bool img_true = image_processor(
            in_img_line, config->vision_patch_size, config->vision_spatial_merge_size, config->min_pixels,
            config->max_pixels, &img_processed_output, &img_processed_h,
            &img_processed_w, &img_grid_h, &img_grid_w
        );

        printf("Finish processing images\n");

        double t_gen_start = now_sec();
        double t_first_token = 0.0;
        double t_gen_end = 0.0;

        int first_token_recorded = 0;

        // if (!img_true) continue;

        if (img_true) {
            forward_img(config, state, weight, img_true ? img_processed_output : nullptr, img_processed_h, img_processed_w, img_grid_h, img_grid_w);
        }

        printf("Finish forward images\n");
        printf("max_seq_len=%d, input_count=%d\n", max_seq_len, input_count);
        
        // ------------------------------------------------------------
        // 4. Generation loop - matching the structure from run.cpp
        // ------------------------------------------------------------
        int *generated_tokens = (int*)malloc(max_seq_len * sizeof(int));
        int total_generated_count = 0;

        int pos = 0; // position in the sequence
        for (; pos < input_count - 1; pos += config->max_prefill_size) {
            size_t cur_prefill_size = std::min(input_count - 1 - pos, config->max_prefill_size);

            // Process a chunk of prompt tokens
            forward_text_prefill(config, state, weight, input_tokens + pos, cur_prefill_size, pos);

            // Print each prompt token properly
            for (size_t i = 0; i < cur_prefill_size; ++i) {
                print_token(tokenizer, input_tokens[pos + i]);
            }
        }

        #ifdef PRINT_LOGITS
            exit(1);
        #endif
        
        pos = input_count - 1;
        int token = input_tokens[pos];
        print_token(tokenizer, token);

        while (pos < max_seq_len) {
            float *logits = forward_text_decode(config, state, weight, token, pos);

            int next = greedy_decode(logits, config->vocab_size);

            if (!first_token_recorded) {
                t_first_token = now_sec();
                first_token_recorded = 1;
            }

            generated_tokens[total_generated_count++] = next;

            #ifdef PRINT_LOGITS
                printf("%d %s\n", next, tokenizer->vocab[next]);
                printf("Logits: ");
                for (int i = 0; i < 5; i++) {
                    printf("%.6f ", logits[i]);
                }
                printf("\n");
            #else
                print_token(tokenizer, next);
            #endif

            // EOS handling
            if (next == 151645) {  // <im_end>
                printf("\nStopping generation: <im_end> token encountered twice\n");
                break;
            }

            token = next;
            pos++;
        }

        t_gen_end = now_sec();

        printf("  Generated %d total tokens\n", total_generated_count);

        new_tokens_gen_num += total_generated_count;
        first_tok_gen_time_total += (t_first_token - t_gen_start);
        gen_time_total += (t_gen_end - t_gen_start);

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
        printf("Start freeing data\n");
        free(in_line);
        printf("Freeing in_line successfully\n");
        free(in_img_line);
        printf("Freeing in_img_line successfully\n");
        // if (img_true) free(img_processed_output);
        printf("Freeing img_processed_output successfully\n");
        free(out_line);
        printf("Freeing out_line successfully\n");
        free(input_tokens);
        printf("Freeing input_tokens successfully\n");
        free(expected_tokens);
        printf("Freeing expected_tokens successfully\n");
        free(generated_tokens);
        printf("Freeing generated_tokens successfully\n");

        printf("End of forward cycle %d\n", sample_count);
    }

    double avg_ttft = first_tok_gen_time_total / sample_count;
    double avg_tps = new_tokens_gen_num * sample_count / gen_time_total;

    fclose(in_file);
    fclose(in_img_file);
    fclose(out_file);

    printf("\n--- Forward Validation Summary ---\n");
    printf("Total Samples Processed: %d\n", sample_count);
    printf("Total Failures: %d\n", validation_failures);
    printf("Average TTFT: %lf (s)\n", avg_ttft);
    printf("Average generated tokens per second: %lf (toks/s)\n", avg_tps);
    
    return (validation_failures > 0) ? 1 : 0;
}

void forward_generate(const char *in_token_file, const char *in_img_path, const char *out_token_file, TokenizerStruct *tokenizer, QwenConfig *config, QwenWeight *weight, QwenRunState *state) {
    FILE* in_file = fopen(in_token_file, "r");
    FILE* in_img_file = fopen(in_img_path, "r");
    FILE* out_file = fopen(out_token_file, "w");

    if (!in_file) {
        fprintf(stderr, "Error: Could not open input token file: %s\n", in_token_file);
        return;
    }
    if (!in_img_file) {
        fprintf(stderr, "Error: Could not open input image path file: %s\n", in_img_path);
        fclose(in_file);
        return;
    }
    if (!out_file) {
        fprintf(stderr, "Error: Could not open output token file: %s\n", out_token_file);
        fclose(in_file);
        fclose(in_img_file);
        return;
    }

    printf("\nStarting Forward Validation...\n");

    int sample_count = 0;

    double first_tok_gen_time_total = 0.0;
    double gen_time_total = 0.0;
    int new_tokens_gen_num = 0;

    int max_seq_len = config->seq_len;

    while (1) {
        char* in_line = read_full_line(in_file);
        char* in_img_line = read_full_line(in_img_file);

        if (!in_line || !in_img_line) {
            free(in_line);
            free(in_img_line);
            break;
        }

        size_t len = strlen(in_img_line);
        if (len > 0 && in_img_line[len - 1] == '\n') {
            in_img_line[len - 1] = '\0';
        }

        sample_count++;
        printf("Starting new forward cycle %d...\n", sample_count);

        int *input_tokens = NULL;
        int input_count = 0;
        get_expected_tokens(in_line, &input_tokens, &input_count);

        if (input_count >= max_seq_len) {
            fprintf(out_file, "None\n");
            fflush(out_file);
            continue;
        }

        // ------------------------------------------------------------
        // 3. Reset state and run initial forward pass
        // ------------------------------------------------------------
        float *img_processed_output;
        int img_processed_h = 0, img_processed_w = 0;
        int img_grid_h = 0, img_grid_w = 0;
        bool img_true = image_processor(
            in_img_line, config->vision_patch_size, config->vision_spatial_merge_size, config->min_pixels,
            config->max_pixels, &img_processed_output, &img_processed_h,
            &img_processed_w, &img_grid_h, &img_grid_w
        );

        printf("Finish processing images\n");

        double t_gen_start = now_sec();
        double t_first_token = 0.0;
        double t_gen_end = 0.0;

        int first_token_recorded = 0;

        if (img_true) {
            forward_img(config, state, weight, img_true ? img_processed_output : nullptr, img_processed_h, img_processed_w, img_grid_h, img_grid_w);
        }

        printf("Finish forward images\n");
        printf("max_seq_len=%d, input_count=%d\n", max_seq_len, input_count);
        
        // ------------------------------------------------------------
        // 4. Generation loop - matching the structure from run.cpp
        // ------------------------------------------------------------
        int *generated_tokens = (int*)malloc(max_seq_len * sizeof(int));
        int total_generated_count = 0;

        // Start the main loop - similar to generate() in run.cpp
        int next; // will store the next token in the sequence
        int token = input_tokens[0]; // kick off with the first token in the prompt
        int pos = 0; // position in the sequence
        int im_end_count = 0;

        while (pos < max_seq_len) { // max steps
            // Forward the transformer to get logits for the next token
            // Using your existing forward functions
            float *logits = forward_text_decode(config, state, weight, token, pos);

            // Advance the state machine - similar to run.cpp
            if (pos < input_count - 1) {
                // If we are still processing the input prompt, force the next prompt token
                next = input_tokens[pos + 1];
            } else {
                // Otherwise use greedy decoding (temperature=0 equivalent)
                next = greedy_decode(logits, config->vocab_size);
                if (!first_token_recorded) {
                    t_first_token = now_sec();
                    first_token_recorded = 1;
                }
            }
            pos++;

            // Add generated token to sequence (only after processing prompt)
            if (pos >= input_count) {
                generated_tokens[total_generated_count++] = next;
            }

            #ifdef PRINT_LOGITS
                if (pos >= input_count) {
                    printf("%d %s", next, tokenizer->vocab[next]);
                    printf("\nLogits: ");
                    for (int i = 0; i < 5; i++) {
                        printf("%.6f ", logits[i]);
                    }
                    printf("\n");
                } else {
                    if (next != 151655) {   
                        printf("%s ", tokenizer->vocab[next]);
                    }
                }
            #else
                if (next != 151655) {   
                    printf("%s ", tokenizer->vocab[next]);
                }
            #endif

            // Data-dependent terminating condition - match your EOS tokens
            // Using the same condition as original forward_validate
            if (next == 151645) { // <im_end> token
                // Count consecutive im_end tokens like original
                im_end_count++;
                if (im_end_count >= 2) {
                    printf("\nStopping generation: <im_end> token encountered twice\n");
                    break;
                }
            }

            token = next; // Update token for next iteration
        }
        
        t_gen_end = now_sec();

        for (int i = 0; i < total_generated_count; i++) {
            fprintf(out_file, "%d ", generated_tokens[i]);
        }
        fprintf(out_file, "\n");
        fflush(out_file);

        printf("  Generated %d total tokens\n", total_generated_count);

        new_tokens_gen_num += total_generated_count;
        first_tok_gen_time_total += (t_first_token - t_gen_start);
        gen_time_total += (t_gen_end - t_gen_start);

        // ------------------------------------------------------------
        // 5. Compare generated sequence with expected sequence
        // ------------------------------------------------------------
        
        // ------------------------------------------------------------
        // 6. Cleanup
        // ------------------------------------------------------------
        printf("Start freeing data\n");
        free(in_line);
        printf("Freeing in_line successfully\n");
        free(in_img_line);
        printf("Freeing in_img_line successfully\n");
        // if (img_true) free(img_processed_output);
        printf("Freeing img_processed_output successfully\n");
        free(input_tokens);
        printf("Freeing input_tokens successfully\n");
        free(generated_tokens);
        printf("Freeing generated_tokens successfully\n");

        printf("End of forward cycle %d\n", sample_count);
    }

    double avg_ttft = first_tok_gen_time_total / sample_count;
    double avg_tps = new_tokens_gen_num * sample_count / gen_time_total;

    fclose(in_file);
    fclose(in_img_file);
    fclose(out_file);

    printf("\n--- Forward Validation Summary ---\n");
    printf("Total Samples Processed: %d\n", sample_count);
    printf("Average TTFT: %lf (s)\n", avg_ttft);
    printf("Average generated tokens per second: %lf (toks/s)\n", avg_tps);
    
}

int image_processor_validate(
    const char *in_img_path,
    TokenizerStruct *tokenizer,
    QwenConfig *config,
    QwenWeight *weight,
    QwenRunState *state
) {
    FILE* img_file = fopen(in_img_path, "r");
    if (!img_file) {
        fprintf(stderr, "Error: Could not open image path file: %s\n", in_img_path);
        return 1;
    }

    printf("\nStarting Image Processor Validation...\n");

    int sample_count = 0;
    int validation_failures = 0;

    float *prev_embedding = NULL;        // used to compare image features
    int embed_dim = config->hidden_size; // typical 4096 for QwenVL

    while (1) {
        char* img_path = read_full_line(img_file);

        if (img_path == NULL) {
            break;
        }

        size_t len = strlen(img_path);
        if (len > 0 && img_path[len - 1] == '\n') {
            img_path[len - 1] = '\0';
        }

        // if len = 0, skip
        if (strlen(img_path) == 0) {
            free(img_path);
            continue;
        }

        sample_count++;
        printf("\n--- Image Validation Sample %d ---\n", sample_count);
        printf("Image Path: %s\n", img_path);

        // ------------------------------------------------------------
        // Run Vision Encoder Input (same call used in forward_validate)
        // ------------------------------------------------------------
        float *img_processed_output;
        int img_processed_h, img_processed_w;
        int img_grid_h, img_grid_w;
        bool img_true = image_processor(
            img_path, config->vision_patch_size,
            config->vision_spatial_merge_size, config->min_pixels,
            config->max_pixels, &img_processed_output, &img_processed_h,
            &img_processed_w, &img_grid_h, &img_grid_w
        );

        printf("Finish validation sample %d\n", sample_count);

        forward_img(config, state, weight, img_true ? img_processed_output : nullptr, img_processed_h, img_processed_w, img_grid_h, img_grid_w);

        printf("Finish forward validation sample %d\n", sample_count);

        free(img_path);
        if (img_true) free(img_processed_output);

        printf("Finish free sample %d\n", sample_count);
    }

    printf("Finish validation\n");

    fclose(img_file);
    free(prev_embedding);

    // ------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------
    printf("\n--- Image Processor Validation Summary ---\n");
    printf("Total Images Processed: %d\n", sample_count);
    printf("Total Failures: %d\n", validation_failures);

    return (validation_failures > 0) ? 1 : 0;
}

void warm_up(
    TokenizerStruct *tokenizer,
    QwenConfig *config,
    QwenWeight *weight,
    QwenRunState *state
) {
    printf("🔥 Warm-up start...\n");

    // ---- 1. Fake token (BOS or random vocab token) ----
    int warmup_token = 1;  // usually <bos> = 1, adjust if needed
    int pos = 0;

    // ---- 2. Process one image ----
    int img_grid_h = 2, img_grid_w = 2;
    int img_processed_h = img_grid_h * img_grid_w, img_processed_w = 1536;
    float *img_processed_output = (float *)malloc(img_processed_h * img_processed_w * sizeof(float));

    forward_img(config, state, weight,
                img_processed_output,
                img_processed_h, img_processed_w,
                img_grid_h, img_grid_w, true);
    
    state->vision_embed_tokens = 0;
    state->cur_img_token_id = 0;

    printf("🔥 Warm-up image finished.\n");

    // ---- 3. Run one transformer token forward ----
    float *logits;

    for (int i = 0; i < 5; ++i) {
        forward_text_prefill(config, state, weight, &warmup_token, 1, pos, true);
        logits = forward_text_decode(config, state, weight, warmup_token, pos, true);
    }

    (void)logits; // suppress unused warning

    printf("🔥 Warm-up done.\n");
}

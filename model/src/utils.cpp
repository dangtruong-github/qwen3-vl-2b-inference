#include "../include/utils.hpp"

void forward_example(QwenConfig *config, QwenWeight *weights, QwenRunState* state) {
    // Example inputs
    float* image = nullptr; // load_image_as_float("example.jpg"); // Implement your own
    int input_tokens[] = {101, 234, 543, 99};          // Already tokenized text
    int n_tokens = sizeof(input_tokens) / sizeof(int);

    // Forward pass
    forward_image_encoder(state, weights, image);
    forward_language(state, weights, input_tokens, n_tokens);
    forward_transformer(state, weights);

    // Get final logits
    float* logits = state->logits; // [vocab_size]

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
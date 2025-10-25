#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/load_model.hpp"
#include "model/config.hpp"

int main(int argc, char* argv[]) {
    const char* load_model_path = nullptr;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model_path") == 0 && i + 1 < argc) {
            load_model_path = argv[i + 1];
            i++; // skip next argument since it's the value
        }
    }

    if (!load_model_path) {
        fprintf(stderr, "Usage: %s --model_path <path_to_model>\n", argv[0]);
        return 1;
    }

    printf("Loading model from: %s\n", load_model_path);

    QwenConfig *config = new QwenConfig;
    QwenWeight *weights = new QwenWeight;
    QwenRunState *run_state = new QwenRunState;

    // Replace "export.bin" with the actual path to your binary file
    init_model_weights(load_model_path, config, weights);

    // Example usage:
    if (config->num_hidden_layers > 0 && weights->input_layernorm_weight != NULL) {
        // Accessing the input_layernorm_weight of the 5th layer (index 4)
        // Weight is at index (4 * hidden_size) + 0
        float first_weight_of_layer_5 = weights->input_layernorm_weight[4 * config->hidden_size];
        printf("Config Hidden Size: %d\n", config->hidden_size);
        printf("First weight of input_layernorm for Layer 5: %f\n", first_weight_of_layer_5);
    }

    // Don't forget to free the memory!
    free_model_weights(weights);
    delete config;
    delete weights;
    delete run_state;

    return 0;
}
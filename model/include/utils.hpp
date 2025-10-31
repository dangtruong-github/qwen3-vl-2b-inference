#pragma once

#include "../config.hpp"
#include "../include/forward.hpp"
#include "../../tokenizer/include/utils.hpp"

void forward_example(QwenConfig *config, QwenWeight *weights, QwenRunState* state);
void print_config(QwenConfig *config);
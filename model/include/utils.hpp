#pragma once

#include "../config.hpp"
#include "../include/forward.hpp"
#include "../../tokenizer/include/utils.hpp"
#include "../../tokenizer/include/test_utils.hpp"

void forward_example(QwenConfig *config, QwenWeight *weights, QwenRunState* state);
void print_config(QwenConfig *config);
int forward_validate(const char *in_token_file, const char *out_token_file, TokenizerStruct *tokenizer, QwenConfig *config, QwenWeight *weight, QwenRunState *state);
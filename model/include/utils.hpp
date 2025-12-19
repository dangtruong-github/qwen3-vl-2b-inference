#pragma once

#include "../config.hpp"
#include "../include/forward.hpp"
#include "../../tokenizer/include/utils.hpp"
#include "../../tokenizer/include/test_utils.hpp"
#include "../../utils/module.hpp"

void print_config(QwenConfig *config);
int forward_validate(const char *in_token_file, const char *in_img_path, const char *out_token_file, TokenizerStruct *tokenizer, QwenConfig *config, QwenWeight *weight, QwenRunState *state);
int image_processor_validate(const char *in_img_path,
                             TokenizerStruct *tokenizer,
                             QwenConfig *config,
                             QwenWeight *weight,
                             QwenRunState *state);
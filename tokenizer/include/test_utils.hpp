#pragma once

#include "../config.hpp"
#include "utils.hpp"

int tokenizer_validate(
    TokenizerStruct* tokenizer, const char* prompt_file_path,
    const char* tokens_file_path, const char* img_file_path,
    int patch_size, int merge_size
);
int decode_validate(
    TokenizerStruct* tokenizer, const char* prompt_file_path,
    const char* tokens_file_path
);
#pragma once

#include "../config.hpp"
#include "utils.hpp"

char* read_full_line(FILE* f);
void get_expected_tokens(const char *tokens_line, int **out_tokens, int *out_count);
int tokenizer_validate(
    TokenizerStruct* tokenizer,
    QwenConfig *config,
    const char* prompt_file_path,
    const char* tokens_file_path,
    const char* img_file_path,
    int patch_size, int merge_size
);
int decode_validate(
    TokenizerStruct* tokenizer, const char* prompt_file_path,
    const char* tokens_file_path
);
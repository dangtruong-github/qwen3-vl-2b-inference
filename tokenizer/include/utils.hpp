#pragma once

#include <float.h>

#include "../config.hpp"
#include "img_processor.hpp"
#include "../../model/config.hpp"

int compare_tokens(const void *a, const void *b);
int greedy_decode(float* logits, int vocab_size);
void tokenizer_example(TokenizerStruct *tokenizer);
void encode(
    TokenizerStruct *t, QwenConfig *config, char *text, int *tokens,
    int *n_tokens, char *img_path, int patch_size, int merge_size
);
void decode(
    TokenizerStruct *t, char *text, size_t text_size, int *tokens, int n_tokens
);
int starts_with_utf8(const char *s, const char *utf8_char);
int ends_with_utf8(const char *s, const char *utf8_char);
void print_normalized_utf8(const char *word);
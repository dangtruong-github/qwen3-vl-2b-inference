#pragma once

#include <float.h>

#include "../config.hpp"

int compare_tokens(const void *a, const void *b);
int greedy_decode(float* logits, int vocab_size);
void tokenizer_example(TokenizerStruct *tokenizer);
void encode(TokenizerStruct *t, char *text, int *tokens, int *n_tokens, char *img_path);
#pragma once

#include <float.h>

#include "../config.hpp"

int greedy_decode(float* logits, int vocab_size);
void tokenizer_example(TokenizerStruct *tokenizer);
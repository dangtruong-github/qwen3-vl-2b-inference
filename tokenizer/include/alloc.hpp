#pragma once

#include "../config.hpp"
#include "utils.hpp"

void init_tokenizer(TokenizerStruct* t, const char* tokenizer_path);
void free_tokenizer(TokenizerStruct* t);
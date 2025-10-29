#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include <ctype.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    TokenIndex* sorted_vocab;
    int vocab_size;
    char **merges;
    TokenIndex* sorted_merge;
    int merges_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} TokenizerStruct;

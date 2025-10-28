#include "../include/alloc.hpp"

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void init_tokenizer(TokenizerStruct* t, const char* tokenizer_path) {
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    printf("Open file success \n");
    fflush(stdout);

    // i should have written the vocab_size into the tokenizer file... sigh
    if (fread(&(t->vocab_size), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read vocab size\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&(t->merges_size), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read merges size\n");
        exit(EXIT_FAILURE);
    }

    printf("Read vocab size and merges size success \n");
    fflush(stdout);

    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));
    t->merges = (char**)malloc(t->merges_size * sizeof(char*));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // read in the file
    if (fread(&(t->max_token_length), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }

    printf("Read max_token_length success \n");
    fflush(stdout);

    int len;
    for (int i = 0; i < t->vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }

    printf("Read vocab success \n");
    fflush(stdout);

    for (int i = 0; i < t->merges_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->merges[i] = (char *)malloc(len + 1);
        if (fread(t->merges[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->merges[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);

    printf("Read merges success \n");
    fflush(stdout);
    
    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    printf("Read sorted_vocab success \n");
    fflush(stdout);
}

void free_tokenizer(TokenizerStruct* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->sorted_vocab);

    for (int i = 0; i < t->merges_size; i++) {
        free(t->merges[i]);
    }
    free(t->merges);
}
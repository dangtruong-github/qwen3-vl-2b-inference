#include "../include/alloc.hpp"

void init_tokenizer(TokenizerStruct* t, const char* tokenizer_path) {
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    // i should have written the vocab_size into the tokenizer file... sigh
    if (fread(&(t->vocab_size), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read vocab size\n");
        exit(EXIT_FAILURE);
    }
    if (fread(&(t->merges_size), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read merges size\n");
        exit(EXIT_FAILURE);
    }

    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));
    t->merges = (char**)malloc(t->merges_size * sizeof(char*));
    t->sorted_vocab = NULL; // initialized lazily
    t->sorted_merge = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // read in the file
    if (fread(&(t->max_token_length), sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < t->vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }

    for (int i = 0; i < t->merges_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->merges[i] = (char *)malloc(len + 1);
        if (fread(t->merges[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->merges[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
    
    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    if (t->sorted_merge == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_merge = (TokenIndex*)malloc(t->merges_size * sizeof(TokenIndex));
        for (int i = 0; i < t->merges_size; i++) {
            t->sorted_merge[i].str = t->merges[i];
            t->sorted_merge[i].id = i;
        }
        qsort(t->sorted_merge, t->merges_size, sizeof(TokenIndex), compare_tokens);
    }
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
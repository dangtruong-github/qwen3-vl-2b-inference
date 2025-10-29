#include "../include/utils.hpp"

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void concat_merge(const char *a, const char *b, char **c) {
    // Calculate total length: a + space + b + null terminator
    size_t len = strlen(a) + 1 + strlen(b) + 1;
    *c = (char *)malloc(len);

    if (*c == NULL) {
        printf("Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    snprintf(*c, len, "%s %s", a, b);
}

int greedy_decode(float* logits, int vocab_size) {
    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = (char *)str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

int merge_lookup(
    char *str_a, char *str_b, TokenIndex *sorted_merge, int merges_size
) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    char *str_comb;
    concat_merge(str_a, str_b, &str_comb);
    TokenIndex tok = { .str = str_comb }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_merge, merges_size, sizeof(TokenIndex), compare_tokens);
    free(str_comb);
    return res != NULL ? res->id : -1;
}

void tokenizer_example(TokenizerStruct *tokenizer) {
    // ---
    // Print Vocabulary (vocab)
    // ---
    printf("========================================\n");
    printf("VOCABULARY (Size: %d)\n", tokenizer->vocab_size);
    printf("========================================\n");

    for (int i = 0; i < tokenizer->vocab_size; i++) {
        // Assuming vocab[i] is a valid C-style string (char*)
        printf("Index %-5d | Token: \"%s\"\n", i, tokenizer->vocab[i]);
    }

    // ---
    // Print Merges (merges)
    // ---
    printf("\n");
    printf("========================================\n");
    printf("MERGES (Size: %d)\n", tokenizer->merges_size);
    printf("========================================\n");

    for (int i = 0; i < tokenizer->merges_size; i++) {
        // Assuming merges[i] is a valid C-style string (char*) representing a merge pair
        printf("Merge %-5d | Pair: \"%s\"\n", i, tokenizer->merges[i]);
    }
    
    printf("\n");
    printf("--- General Info ---\n");
    printf("Max Token Length: %u\n", tokenizer->max_token_length);
    printf("--------------------\n");
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Assuming TokenizerStruct and lookup functions are defined elsewhere.
// For the purpose of this solution, we define a common PAD token ID.
#define PAD_TOKEN_ID 0

void encode(TokenizerStruct *t, char *text, int *tokens, int max_len, int *n_tokens) {
    if (!text) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    // --- Allocate str_buffer immediately for safe cleanup ---
    char *str_buffer = (char*)malloc((t->max_token_length * 2 + 4) * sizeof(char));
    if (!str_buffer) {
        perror("malloc failed for str_buffer");
        exit(EXIT_FAILURE);
    }
    *n_tokens = 0;

    // The wrapper requires 8 tokens. If max_len is too small, return 0 tokens.
    if (max_len < 3 + 5) {
        *n_tokens = 0;
        free(str_buffer);
        return; 
    }
    
    // ------------------------------------------------------------
    // 1. Append 'Ċ' (as in original code)
    // ------------------------------------------------------------
    size_t text_len = strlen(text);
    char *text_with_nl = (char*)malloc(text_len + 2);
    if (!text_with_nl) {
        perror("malloc failed for text_with_nl");
        free(str_buffer);
        exit(EXIT_FAILURE);
    }
    strcpy(text_with_nl, text);
    strcat(text_with_nl, "Ċ");
    
    // ------------------------------------------------------------
    // 2. UTF-8 aware byte tokenization (as in original code)
    // ------------------------------------------------------------
    size_t str_len = 0;
    for (char *c = text_with_nl; *c != '\0'; c++) {
        // NOTE: The original loop did not check against max_len here,
        // relying on BPE to reduce token count later. We keep that behavior.
        if ((*c & 0xC0) != 0x80) str_len = 0;

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            int space_id = str_lookup("Ġ", t->sorted_vocab, t->vocab_size);
            tokens[(*n_tokens)++] = space_id;
        }

        str_len = 0;
    }
    free(text_with_nl);

    // ------------------------------------------------------------
    // 3. BPE merges (lowest-rank first) (as in original code)
    // ------------------------------------------------------------
    // ... BPE merge loop as in original code ...
    while (1) {
        int best_rank = t->merges_size + 1;
        int best_idx = -1;
        int best_id = -1;
        // ... (The rest of the BPE logic) ...
        for (int i = 0; i < (*n_tokens - 1); i++) {
            int id1 = tokens[i], id2 = tokens[i + 1];
            if (id1 < 0 || id2 < 0) continue;

            int rank = merge_lookup(t->vocab[id1], t->vocab[id2],
                                     t->sorted_merge, t->merges_size);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_idx = i;
                snprintf(str_buffer, t->max_token_length * 2 + 4,
                          "%s%s", t->vocab[id1], t->vocab[id2]);
                best_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            }
        }

        if (best_idx == -1 || best_id == -1) break;

        tokens[best_idx] = best_id;
        for (int j = best_idx + 1; j < (*n_tokens - 1); j++)
            tokens[j] = tokens[j + 1];
        (*n_tokens)--;
    }


    // ------------------------------------------------------------
    // 4. Remove placeholders (-1) if any remain (as in original code)
    // ------------------------------------------------------------
    int valid_n = 0;
    for (int i = 0; i < *n_tokens; i++) {
        if (tokens[i] != -1)
            tokens[valid_n++] = tokens[i];
    }
    *n_tokens = valid_n;

    // ------------------------------------------------------------
    // 5. Wrap chat-style prompt and Truncate if needed
    // ------------------------------------------------------------
    const int im_start = 151644;
    const int im_end = 151645;
    const int user_tok = 872;
    const int assistant_tok = 77091;
    const int newline_tok = 198;

    int prefix[] = {im_start, user_tok, newline_tok};
    int suffix[] = {im_end, newline_tok, im_start, assistant_tok, newline_tok};
    const int prefix_len = 3;
    const int suffix_len = 5;

    int orig_n = *n_tokens;
    int min_total_n = prefix_len + suffix_len; // 8 tokens for wrapper
    
    // Calculate the number of tokens available for the original content
    int content_max_len = max_len - min_total_n;
    
    // Truncate the original content if it's too long
    if (orig_n > content_max_len) {
        // If the original content is too long, truncate it.
        orig_n = content_max_len > 0 ? content_max_len : 0;
    }
    
    // The final total number of tokens (after truncation and wrapping)
    int wrapped_n = prefix_len + orig_n + suffix_len;
    
    // --- Perform the wrapper insertion/truncation ---

    // 1. Make space for the prefix and content (or truncate the content)
    // We only move up to orig_n elements (the potentially truncated content).
    memmove(tokens + prefix_len, tokens, orig_n * sizeof(int));
    
    // 2. Insert the prefix
    memcpy(tokens, prefix, prefix_len * sizeof(int));
    
    // 3. Insert the suffix right after the content
    memcpy(tokens + prefix_len + orig_n, suffix, suffix_len * sizeof(int));
    
    // ------------------------------------------------------------
    // 6. Set final token count (NO PADDING)
    // ------------------------------------------------------------

    // If wrapped_n > max_len, this is only possible if content_max_len was negative
    // (i.e., max_len was < 8), but we handled that with an early return.
    // If we reached here, the final length is guaranteed to be <= max_len.
    *n_tokens = wrapped_n;
    
    free(str_buffer);
}


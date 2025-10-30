#include <opencv2/opencv.hpp>

#include "../include/utils.hpp"

void read_img_size(const char *img_path, int *height, int *width) {
    if (img_path == nullptr) {
        std::cerr << "Error: image file not found: " << std::endl;
        *height = 0;
        *width = 0;
        return;
    }

    // Read the image using OpenCV
    cv::Mat img = cv::imread(img_path);

    if (img.empty()) {
        std::cerr << "Error: Could not read the image file: " << img_path << std::endl;
        *height = 0;
        *width = 0;
        return;
    }

    // Assign height and width
    *height = img.rows;
    *width = img.cols;
}

int get_num_img_pad(const char *img_path, int patch_size) {
    int height, width;
    read_img_size(img_path, &height, &width);

    if (height <= 0 || width <= 0) {
        fprintf(stderr, "Invalid image dimensions.\n");
        return 0;
    }

    // Compute number of patch rows/cols (ceil division)
    int Hp = (height + patch_size - 1) / patch_size;
    int Wp = (width  + patch_size - 1) / patch_size;

    // Actual patch tokens
    int actual_patches = Hp * Wp;

    // Pad to square grid (Qwen-like behavior)
    int S = (Hp > Wp) ? Hp : Wp;
    int expected_patches = S * S;

    // Expected padding tokens
    int exp_img_pad = (expected_patches - actual_patches) * 3 / 4;

    printf("Height=%d, Width=%d, Hp=%d, Wp=%d, Actual=%d, Expected=%d, Padding=%d\n",
           height, width, Hp, Wp, actual_patches, expected_patches, exp_img_pad);

    return exp_img_pad;
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void concat_merge(const char *a, const char *b, char **c) {
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Invalid NULL argument to concat_merge\n");
        exit(EXIT_FAILURE);
    }

    // Calculate total length: a + space + b + null terminator
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    size_t total_len = len_a + 1 + len_b + 1;

    *c = (char *)malloc(total_len);
    if (*c == NULL) {
        fprintf(stderr, "Memory allocation failed in concat_merge\n");
        exit(EXIT_FAILURE);
    }

    // Perform concatenation manually
    strcpy(*c, a);
    strcat(*c, " ");
    strcat(*c, b);
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

void encode(
    TokenizerStruct *t, char *text, int *tokens, int *n_tokens,
    char *img_path, int patch_size
) {
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    char *str_buffer = (char*)malloc((t->max_token_length * 2 + 4) * sizeof(char));
    if (!str_buffer) {
        fprintf(stderr, "Error: failed to allocate str_buffer\n");
        exit(EXIT_FAILURE);
    }

    *n_tokens = 0;

    // ------------------------------------------------------------
    // 1. Append "Ċ" if missing
    // ------------------------------------------------------------
    size_t text_len = strlen(text);
    char *text_with_nl = (char*)malloc(text_len + 4);
    if (!text_with_nl) {
        fprintf(stderr, "Error: failed to allocate text_with_nl\n");
        exit(EXIT_FAILURE);
    }
    strcpy(text_with_nl, text);
    strcat(text_with_nl, "Ċ");

    // ------------------------------------------------------------
    // 2. UTF-8 aware byte tokenization
    // ------------------------------------------------------------
    size_t str_len = 0;
    char *c = text_with_nl;

    while (*c != '\0') {
        if (((unsigned char)(*c) & 0xC0) != 0x80)
            str_len = 0;

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if (((unsigned char)*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            c++;
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[*n_tokens] = id;
            (*n_tokens)++;
        } else {
            int space_id = str_lookup("Ġ", t->sorted_vocab, t->vocab_size);
            tokens[*n_tokens] = space_id;
            (*n_tokens)++;
        }

        str_len = 0;
        c++;
    }

    free(text_with_nl);

    // ------------------------------------------------------------
    // 3. BPE merges (lowest-rank first)
    // ------------------------------------------------------------
    while (1) {
        int best_rank = t->merges_size + 1;
        int best_idx = -1;
        int best_id = -1;

        if (*n_tokens < 2)
            break;

        for (int i = 0; i < *n_tokens - 1; i++) {
            int id1 = tokens[i];
            int id2 = tokens[i + 1];
            if (id1 < 0 || id2 < 0)
                continue;

            int rank = merge_lookup(t->vocab[id1], t->vocab[id2],
                                    t->sorted_merge, t->merges_size);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_idx = i;

                // safe string concatenation
                int len1 = (int)strlen(t->vocab[id1]);
                int len2 = (int)strlen(t->vocab[id2]);
                if (len1 + len2 + 1 < (int)(t->max_token_length * 2 + 4)) {
                    strcpy(str_buffer, t->vocab[id1]);
                    strcat(str_buffer, t->vocab[id2]);
                    best_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
                } else {
                    fprintf(stderr, "Merged token too long, skipping.\n");
                    best_id = -1;
                }
            }
        }

        if (best_idx == -1 || best_id == -1)
            break;

        tokens[best_idx] = best_id;
        for (int j = best_idx + 1; j < *n_tokens - 1; j++) {
            tokens[j] = tokens[j + 1];
        }
        (*n_tokens)--;
    }

    // ------------------------------------------------------------
    // 4. Remove placeholders (-1)
    // ------------------------------------------------------------
    {
        int valid_n = 0;
        for (int i = 0; i < *n_tokens; i++) {
            if (tokens[i] != -1)
                tokens[valid_n++] = tokens[i];
        }
        *n_tokens = valid_n;
    }

    // ------------------------------------------------------------
    // 5. Wrap chat-style prompt
    // ------------------------------------------------------------
    const int im_start = 151644;
    const int im_end = 151645;
    const int user_tok = 872;
    const int assistant_tok = 77091;
    const int newline_tok = 198;
    const int vision_start = 151652;
    const int image_pad = 151655;
    const int vision_end = 151653;

    int num_img_pad = get_num_img_pad(img_path, patch_size);  // dynamically computed later if needed
    int prefix_len = (num_img_pad > 0) ? (4 + num_img_pad + 2) : 3;

    int *prefix = (int*)malloc(prefix_len * sizeof(int));
    if (!prefix) {
        fprintf(stderr, "Error: failed to allocate prefix\n");
        exit(EXIT_FAILURE);
    }

    int idx = 0;
    prefix[idx++] = im_start;
    prefix[idx++] = user_tok;
    prefix[idx++] = newline_tok;

    if (num_img_pad > 0) {
        prefix[idx++] = vision_start;
        for (int i = 0; i < num_img_pad; i++)
            prefix[idx++] = image_pad;
        prefix[idx++] = vision_end;
        prefix[idx++] = newline_tok;
    }

    int suffix[5];
    suffix[0] = im_end;
    suffix[1] = newline_tok;
    suffix[2] = im_start;
    suffix[3] = assistant_tok;
    suffix[4] = newline_tok;

    int orig_n = *n_tokens;
    int total_n = prefix_len + orig_n + 5;

    // Shift tokens to make room for prefix
    memmove(tokens + prefix_len, tokens, orig_n * sizeof(int));
    memcpy(tokens, prefix, prefix_len * sizeof(int));
    memcpy(tokens + prefix_len + orig_n, suffix, 5 * sizeof(int));

    *n_tokens = total_n;

    free(prefix);
    free(str_buffer);
}


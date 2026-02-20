#include "../include/utils.hpp"

int max(int a, int b) {
    return (a > b) ? a : b;
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

#if defined(__AVX512F__) && defined(__AVX512DQ__)
int greedy_decode(float* logits, int vocab_size) {
    // 1. Initialize 16-lane vectors
    __m512 v_max_vals = _mm512_set1_ps(-FLT_MAX);
    __m512i v_max_idxs = _mm512_setzero_si512();
    
    // Index tracker: {0, 1, ..., 15}
    __m512i v_current_idxs = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m512i v_step = _mm512_set1_epi32(16);

    int i = 0;
    // 2. Main Loop (16 elements at a time)
    for (; i <= vocab_size - 16; i += 16) {
        __m512 v_logits = _mm512_loadu_ps(&logits[i]);
        
        // Compare into a 16-bit mask register
        __mmask16 mask = _mm512_cmp_ps_mask(v_logits, v_max_vals, _CMP_GT_OQ);
        
        // Use the mask to update values and indices (only where mask bit is 1)
        v_max_vals = _mm512_mask_mov_ps(v_max_vals, mask, v_logits);
        v_max_idxs = _mm512_mask_mov_epi32(v_max_idxs, mask, v_current_idxs);
        
        v_current_idxs = _mm512_add_epi32(v_current_idxs, v_step);
    }

    // 3. Horizontal Reduction of the 16 lanes
    float temp_vals[16];
    int temp_idxs[16];
    _mm512_storeu_ps(temp_vals, v_max_vals);
    _mm512_storeu_si512((__m512i*)temp_idxs, v_max_idxs);

    float final_max = temp_vals[0];
    int final_idx = temp_idxs[0];
    for (int j = 1; j < 16; ++j) {
        if (temp_vals[j] > final_max) {
            final_max = temp_vals[j];
            final_idx = temp_idxs[j];
        }
    }

    // 4. Tail Handling
    for (; i < vocab_size; ++i) {
        if (logits[i] > final_max) {
            final_max = logits[i];
            final_idx = i;
        }
    }

    return final_idx;
}
#elif defined(__AVX2__) && defined(__FMA__)
int greedy_decode_avx2(float* logits, int vocab_size) {
    // 1. Initialize vectors
    __m256 v_max_vals = _mm256_set1_ps(-FLT_MAX);
    __m256i v_max_idxs = _mm256_setzero_si256();
    
    // Index tracker: {0, 1, 2, 3, 4, 5, 6, 7}
    __m256i v_current_idxs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i v_step = _mm256_set1_epi32(8);

    int i = 0;
    // 2. Main Loop (8 elements at a time)
    for (; i <= vocab_size - 8; i += 8) {
        __m256 v_logits = _mm256_loadu_ps(&logits[i]);
        
        // Compare: result is 0xFFFFFFFF where logits[i] > current_max
        __m256 v_mask = _mm256_cmp_ps(v_logits, v_max_vals, _CMP_GT_OQ);
        
        // Update max values and max indices
        v_max_vals = _mm256_blendv_ps(v_max_vals, v_logits, v_mask);
        v_max_idxs = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(v_max_idxs), 
            _mm256_castsi256_ps(v_current_idxs), 
            v_mask));
        
        // Increment current indices for next iteration
        v_current_idxs = _mm256_add_epi32(v_current_idxs, v_step);
    }

    // 3. Horizontal Reduction of the 8 lanes
    float temp_vals[8];
    int temp_idxs[8];
    _mm256_storeu_ps(temp_vals, v_max_vals);
    _mm256_storeu_si256((__m256i*)temp_idxs, v_max_idxs);

    float final_max = temp_vals[0];
    int final_idx = temp_idxs[0];
    for (int j = 1; j < 8; ++j) {
        if (temp_vals[j] > final_max) {
            final_max = temp_vals[j];
            final_idx = temp_idxs[j];
        }
    }

    // 4. Tail Handling (since vocab_size might not be multiple of 8)
    for (; i < vocab_size; ++i) {
        if (logits[i] > final_max) {
            final_max = logits[i];
            final_idx = i;
        }
    }

    return final_idx;
}
#else
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
#endif

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

static bool byte_to_unicode_initialized = false;
static char *byte_to_unicode[256];

// encode a single Unicode codepoint to UTF-8 bytes; returns number of bytes written (1..4)
static int codepoint_to_utf8(int cp, char *out) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    } else if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | ((cp >> 6) & 0x1F));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | ((cp >> 12) & 0x0F));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (char)(0xF0 | ((cp >> 18) & 0x07));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

// Initialize byte -> printable-unicode mapping following OpenAI's encoder.py approach
static void init_byte_to_unicode() {
    if (byte_to_unicode_initialized) return;

    // Build list 'bs' of bytes that should map to themselves (printable latin-1 ranges).
    // The Python pseudocode uses ranges: ord("!")..ord("~"), ord("¡")..ord("¬"), ord("®")..ord("ÿ")
    int bs_list[512];
    int bs_len = 0;
    for (int i = (int)'!'; i <= (int)'~'; ++i) bs_list[bs_len++] = i;
    for (int i = 0xA1; i <= 0xAC; ++i) bs_list[bs_len++] = i; // '¡'..'¬'
    for (int i = 0xAE; i <= 0xFF; ++i) bs_list[bs_len++] = i; // '®'..'ÿ'

    // cs = bs initially; then for remaining bytes, map to 256 + n
    int cs_list[512];
    memcpy(cs_list, bs_list, bs_len * sizeof(int));
    int cs_len = bs_len;

    int n = 0;
    bool in_bs[256];
    memset(in_bs, 0, sizeof(in_bs));
    for (int i = 0; i < bs_len; ++i) in_bs[bs_list[i]] = true;

    for (int b = 0; b < 256; ++b) {
        if (!in_bs[b]) {
            bs_list[bs_len++] = b;
            cs_list[cs_len++] = 256 + n;
            ++n;
        }
    }

    // Now zip bs_list -> cs_list into byte_to_unicode mapping.
    // For each pair: byte value = bs_list[i], codepoint = cs_list[i]
    for (int i = 0; i < bs_len; ++i) {
        int byte_val = bs_list[i];
        int codepoint = cs_list[i];
        char utf8buf[5] = {0};
        int L = codepoint_to_utf8(codepoint, utf8buf);
        utf8buf[L] = '\0';
        byte_to_unicode[byte_val] = strdup(utf8buf);
        if (!byte_to_unicode[byte_val]) {
            fprintf(stderr, "Error: strdup failed in init_byte_to_unicode\n");
            exit(EXIT_FAILURE);
        }
    }

    byte_to_unicode_initialized = true;
}

// Helper: determine next UTF-8 sequence length from leading byte
static int utf8_char_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1; // fallback
}

void encode(
    TokenizerStruct *t, QwenConfig *config, char *text, int *tokens,
    int *n_tokens, char *img_path, int patch_size, int merge_size
) {
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    // init mapping if needed
    init_byte_to_unicode();

    // allocate working buffers
    size_t text_len = strlen(text);

    // mapped buffer: each input byte maps to up to 4 UTF-8 bytes; add room for appended newline mapping
    size_t mapped_capacity = (text_len + 2) * 4 + 8;
    char *mapped_buf = (char*)malloc(mapped_capacity);
    if (!mapped_buf) {
        fprintf(stderr, "Error: failed to allocate mapped_buf\n");
        exit(EXIT_FAILURE);
    }
    size_t mapped_len = 0;

    // ------------------------------------------------------------
    // Build mapped buffer by mapping each raw byte of input -> printable unicode string
    // ------------------------------------------------------------
    unsigned char *u = (unsigned char*)text;
    for (size_t i = 0; i < text_len; ++i) {
        unsigned char b = u[i];
        const char *m = byte_to_unicode[b];
        if (!m) m = byte_to_unicode[ (unsigned char)'?' ]; // very unlikely; safe fallback

        size_t mlen = strlen(m);
        if (mapped_len + mlen + 8 >= mapped_capacity) {
            // grow
            mapped_capacity = (mapped_capacity + mlen + 8) * 2;
            mapped_buf = (char*)realloc(mapped_buf, mapped_capacity);
            if (!mapped_buf) {
                fprintf(stderr, "Error: mapped_buf realloc failed\n");
                exit(EXIT_FAILURE);
            }
        }
        memcpy(mapped_buf + mapped_len, m, mlen);
        mapped_len += mlen;
    }

    // Append a newline byte mapping (this mirrors "append newline if missing" behavior,
    // but done in byte-mapped domain). We map raw byte 10 (LF).
    const char *nl_map = byte_to_unicode[10];
    if (nl_map) {
        size_t mlen = strlen(nl_map);
        if (mapped_len + mlen + 4 >= mapped_capacity) {
            mapped_capacity = (mapped_capacity + mlen + 4) * 2;
            mapped_buf = (char*)realloc(mapped_buf, mapped_capacity);
            if (!mapped_buf) {
                fprintf(stderr, "Error: mapped_buf realloc failed (nl)\n");
                exit(EXIT_FAILURE);
            }
        }
        memcpy(mapped_buf + mapped_len, nl_map, mlen);
        mapped_len += mlen;
    }

    mapped_buf[mapped_len] = '\0';

    // Encode each sentence first then combine
    // ------------------------------------------------------------
    // Now iterate over mapped_buf by UTF-8 codepoints and look each "character" up in vocab
    // Each mapping char is intended to be a single token before BPE merges.
    // ------------------------------------------------------------
    char *str_buffer = (char*)malloc((t->max_token_length * 4 + 16) * sizeof(char));
    if (!str_buffer) {
        fprintf(stderr, "Error: failed to allocate str_buffer\n");
        exit(EXIT_FAILURE);
    }

    *n_tokens = 0;
    size_t p = 0;
    while (p < mapped_len) {
        unsigned char lead = (unsigned char)mapped_buf[p];
        int clen = utf8_char_len(lead);
        if (p + clen > mapped_len) clen = (int)(mapped_len - p); // safety

        if (clen >= (int)(t->max_token_length * 4 + 16)) {
            fprintf(stderr, "Error: unicode char too long\n");
            clen = (int)(t->max_token_length * 4 + 15);
        }

        memcpy(str_buffer, mapped_buf + p, clen);
        str_buffer[clen] = '\0';

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[*n_tokens] = id;
            (*n_tokens)++;
        } else {
            // fallback: map unknown to space token 'Ġ' if exists, else -1
            int space_id = str_lookup("Ġ", t->sorted_vocab, t->vocab_size);
            if (space_id != -1) {
                tokens[*n_tokens] = space_id;
                (*n_tokens)++;
            } else {
                tokens[*n_tokens] = -1;
                (*n_tokens)++;
            }
        }

        p += clen;
    }

    // free mapped buffer
    free(mapped_buf);

    // ------------------------------------------------------------
    // BPE merges (your existing loop, slightly improved)
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
                    // too long, mark invalid
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
    // 5. Wrap chat-style prompt (unchanged from your original)
    // ------------------------------------------------------------
    const int im_start = 151644;
    const int im_end = 151645;
    const int user_tok = 872;
    const int assistant_tok = 77091;
    const int newline_tok = 198;
    const int vision_start = 151652;
    const int image_pad = 151655;
    const int vision_end = 151653;

    const long long min_pixels = config->min_pixels;
    const long long max_pixels = config->max_pixels;
    int num_img_pad = get_num_img_pad(img_path, patch_size, merge_size, min_pixels, max_pixels);  // dynamically computed later if needed
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

int starts_with_utf8(const char *s, const char *utf8_char) {
    return (strncmp(s, utf8_char, strlen(utf8_char)) == 0);
}

int ends_with_utf8(const char *s, const char *utf8_char) {
    size_t s_len = strlen(s);
    size_t u_len = strlen(utf8_char);

    if (u_len > s_len) return 0;

    return (memcmp(s + s_len - u_len, utf8_char, u_len) == 0);
}

void print_normalized_utf8(const char *word) {
    const char *utf8_space = "Ġ";
    const char *utf8_newline = "Ċ";
    size_t len = strlen(word);
    size_t nl_len = strlen(utf8_newline);
    size_t sp_len = strlen(utf8_space);

    size_t start = 0;
    size_t end = len;

    int had_nl_leading = 0;
    int had_sp_leading = 0;

    /* ---- strip leading ---- */
    for (;;) {
        if (nl_len && start + nl_len <= end &&
            memcmp(word + start, utf8_newline, nl_len) == 0) {
            start += nl_len;
            had_nl_leading = 1;
        } else if (sp_len && start + sp_len <= end &&
                   memcmp(word + start, utf8_space, sp_len) == 0) {
            start += sp_len;
            had_sp_leading = 1;
        } else {
            break;
        }
    }

    int had_nl_trailing = 0;
    int had_sp_trailing = 0;

    /* ---- strip trailing ---- */
    for (;;) {
        if (nl_len && end >= nl_len &&
            memcmp(word + end - nl_len, utf8_newline, nl_len) == 0) {
            end -= nl_len;
            had_nl_trailing = 1;
        } else if (sp_len && end >= sp_len &&
                   memcmp(word + end - sp_len, utf8_space, sp_len) == 0) {
            end -= sp_len;
            had_sp_trailing = 1;
        } else {
            break;
        }
    }

    /* ---- emit normalized leading ---- */
    if (had_nl_leading) {
        putchar('\n');
    } else if (had_sp_leading) {
        putchar(' ');
    }

    /* ---- emit core word ---- */
    if (end > start) {
        printf("%.*s", (int)(end - start), word + start);
    }

    if (had_nl_trailing) {
        putchar('\n');
    } else if (had_nl_trailing) {
        putchar(' ');
    }
}

void decode(
    TokenizerStruct *t, char *text, size_t text_size, int *tokens, int n_tokens
) {
    if (!t || !text || !tokens || text_size == 0) return;

    char *out = text;
    size_t remaining = text_size - 1; // leave space for null terminator

    const char *utf8_space = "Ġ";
    const char *utf8_newline = "Ċ";
    size_t len_space = strlen(utf8_space);
    size_t len_newline = strlen(utf8_newline);

    for (int i = 0; i < n_tokens; i++) {
        const char *cur_word = t->vocab[tokens[i]];
        if (!cur_word) continue;

        int start = 0;
        while (starts_with_utf8(cur_word + start, utf8_space))
            start += len_space;

        // --- one leading space if Ġ found ---
        if (start > 0 && remaining > 0) {
            *out++ = ' ';
            remaining--;
        }

        // --- copy with Ċ → '\n' replacement ---
        for (int j = start; cur_word[j] != '\0' && remaining > 0; ) {
            if (starts_with_utf8(cur_word + j, utf8_newline)) {
                *out++ = '\n';
                remaining--;
                j += len_newline;
            } else {
                *out++ = cur_word[j++];
                remaining--;
            }
        }
    }

    *out = '\0';
}

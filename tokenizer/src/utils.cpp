#include <opencv2/opencv.hpp>

#include "../include/utils.hpp"

int max(int a, int b) {
    return (a > b) ? a : b;
}

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

static inline double round_bankers(double x) {
    double r = floor(x + 0.5);
    if (fabs(x - (r - 0.5)) < 1e-9) {
        // exactly halfway: round to even
        if (fmod(r, 2.0) != 0.0)
            r -= 1.0;
    }
    return r;
}

void smart_resize_qwen3(
    int height, int width,
    int *h_bar, int *w_bar,
    int factor,
    long long min_pixels, long long max_pixels
) {
    // const double EPS = 1e-9;

    // 1. Check aspect ratio
    double aspect_ratio = (height > width)
                              ? (double)height / width
                              : (double)width / height;
    if (aspect_ratio > 200.0) {
        fprintf(stderr, "Error: aspect ratio %.2f exceeds 200.0\n", aspect_ratio);
        exit(EXIT_FAILURE);
    }

    // 2. Round to nearest multiple of factor (banker's rounding)
    double h_div = (double)height / factor;
    double w_div = (double)width / factor;
    *h_bar = (int)(round_bankers(h_div) * factor);
    *w_bar = (int)(round_bankers(w_div) * factor);

    long long pixels = (long long)(*h_bar) * (*w_bar);

    // 3. Adjust if area is too large
    if (pixels > max_pixels) {
        double beta = sqrt((double)(height * width) / (double)max_pixels);
        double new_h = (height / beta) / factor;
        double new_w = (width / beta) / factor;
        *h_bar = (int)fmax(factor, floor(new_h) * factor);
        *w_bar = (int)fmax(factor, floor(new_w) * factor);
    }

    // 4. Adjust if area is too small
    else if (pixels < min_pixels) {
        double beta = sqrt((double)min_pixels / (double)(height * width));
        double new_h = (height * beta) / factor;
        double new_w = (width * beta) / factor;
        *h_bar = (int)(ceil(new_h) * factor);
        *w_bar = (int)(ceil(new_w) * factor);
    }
}

int get_num_img_pad(const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels) {
    int height, width, h_bar, w_bar;

    // TODO: Implement your OpenCV read_img_size(img_path, &height, &width)
    read_img_size(img_path, &height, &width);

    if (height <= 0 || width <= 0) {
        fprintf(stderr, "Invalid image dimensions for %s\n", img_path);
        return 0;
    }

    const int factor = patch_size * merge_size;  // 32

    smart_resize_qwen3(height, width, &h_bar, &w_bar, factor, min_pixels, max_pixels);

    int grid_h = h_bar / patch_size;
    int grid_w = w_bar / patch_size;

    int num_patches = grid_h * grid_w;
    int num_image_tokens = num_patches / (merge_size * merge_size);

    printf("Image: %s\n", img_path);
    printf("Orig HxW = %dx%d → Resized = %dx%d\n", height, width, h_bar, w_bar);
    printf("Grid = %dx%d → Num patches = %d → Num image tokens = %d\n",
           grid_h, grid_w, num_patches, num_image_tokens);

    return num_image_tokens;
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

// Replace your existing encode(...) with this version.
// NOTE: requires standard headers already included: <stdio.h>, <stdlib.h>, <string.h>, <stdint.h>, <stdbool.h>

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
    TokenizerStruct *t, char *text, int *tokens, int *n_tokens,
    char *img_path, int patch_size, int merge_size
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

    const long long min_pixels = 256ll * 256ll;
    const long long max_pixels = 16777216ll; // 16777216
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

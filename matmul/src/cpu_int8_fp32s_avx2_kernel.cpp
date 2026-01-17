#include "../include/cpu_int8_fp32s_avx2_kernel.hpp"

#if defined(__AVX2__) && defined(__FMA__)
static inline float add_reduce_mm_256(__m256 vec) {
    // Step 1: Split into two 128-bit halves
    __m128 low  = _mm256_castps256_ps128(vec);          // lower 128 bits
    __m128 high = _mm256_extractf128_ps(vec, 1);        // upper 128 bits

    // Step 2: Add the halves together
    __m128 sum128 = _mm_add_ps(low, high);

    // Step 3: Horizontal add within 128-bit register
    sum128 = _mm_hadd_ps(sum128, sum128);               // pairwise add
    sum128 = _mm_hadd_ps(sum128, sum128);               // final reduction

    // Step 4: Extract scalar result
    return _mm_cvtss_f32(sum128);
}

static inline float add_reduce_m256i(__m256i vec) {
    // 1. Extract the top 128 bits and add them to the bottom 128 bits
    __m128i hi128 = _mm256_extractf128_si256(vec, 1);
    __m128i lo128 = _mm256_castsi256_si128(vec);
    __m128i sum128 = _mm_add_epi32(hi128, lo128);

    // 2. Horizontal add within the remaining 128 bits
    // [A, B, C, D] -> [B, A, D, C]
    __m128i shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
    sum128 = _mm_add_epi32(sum128, shuf);

    // 3. Final shuffle and add
    // [A+B, B+A, C+D, D+C] -> [C+D, D+C, A+B, B+A]
    shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2));
    sum128 = _mm_add_epi32(sum128, shuf);

    // 4. Extract the lower 32-bit integer and convert to float
    return (float)_mm_cvtsi128_si32(sum128);
}

void sm_M_lg_N_lg_K_transpose_prepacked(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    constexpr size_t MAX_GS = 128; // assume MAX_GS == group_size
    constexpr size_t TN = 4 * 2;
    constexpr size_t TK = MAX_GS * 2;

    for (size_t i = 0; i < M; ++i) {
        int16_t a_q16[K];
        float a_q16_s[K / group_size];
        const float *a0_ptr = mat_A + i * K;

        #pragma omp parallel for
        for (size_t kk = 0; kk < K; kk += group_size) {
            float sumsq = 0.0f;

            #pragma omp simd
            for (size_t k = kk; k < kk + group_size; ++k) {
                sumsq += a0_ptr[k] * a0_ptr[k];
            }

            float rms = sqrtf(sumsq / group_size + 1e-6f);
            float inv_Sa = 127.0f / rms;
            a_q16_s[kk / group_size] = 1 / inv_Sa;

            #pragma omp simd
            for (size_t k = kk; k < kk + group_size; ++k) {
                a_q16[k] = (int16_t)(a0_ptr[k] * inv_Sa);
            }
        }

        #pragma omp parallel for
        for (size_t j_tile = 0; j_tile < N; j_tile += TN) {
            for (size_t k_tile = 0; k_tile < K; k_tile += TK) {
                int16_t b_q16[TN * TK];
                float b_q16_s[TN * TK / group_size];
                float acc[TN] = {0.0f};

                for (size_t k = 0; k < TK; k += 32) {
                    for (size_t j = 0; j < TN; ++j) {
                        if (k % group_size == 0) {
                            b_q16_s[(j * TK + k) / group_size] = mat_B_scales[((j + j_tile) * K + k_tile + k) / group_size];
                        }
                        // load 32 int8 and save 32 int16
                        __m256i b8 = _mm256_loadu_si256((__m256i*)(mat_B_in + (j + j_tile) * K + k_tile + k));
                        __m256i b16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                        __m256i b16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));

                        _mm256_storeu_si256((__m256i*)(b_q16 + j * TK + k), b16_0);
                        _mm256_storeu_si256((__m256i*)(b_q16 + j * TK + k + 16), b16_1);
                    }
                }

                for (size_t kk = 0; kk < TK; kk += group_size) {
                    const int16_t *aq_now = a_q16 + kk + k_tile;
                    const float a_s = a_q16_s[(kk + k_tile) / group_size];
                    const size_t group = kk / group_size;
                
                    for (size_t jj = 0; jj < TN; jj += 4) {
                        const int16_t *b0_ptr = b_q16 + jj * TK + kk;
                        const int16_t *b1_ptr = b_q16 + (jj + 1) * TK + kk;
                        const int16_t *b2_ptr = b_q16 + (jj + 2) * TK + kk;
                        const int16_t *b3_ptr = b_q16 + (jj + 3) * TK + kk;

                        __m256i c0 = _mm256_setzero_si256();
                        __m256i c1 = _mm256_setzero_si256();
                        __m256i c2 = _mm256_setzero_si256();
                        __m256i c3 = _mm256_setzero_si256();

                        for (size_t k = 0; k < group_size; k += 32) {
                            __m256i a0 = _mm256_loadu_si256((__m256i*)(aq_now + k));
                            __m256i a1 = _mm256_loadu_si256((__m256i*)(aq_now + k + 16));

                            __m256i b00 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));
                            __m256i b10 = _mm256_loadu_si256((__m256i*)(b1_ptr + k));
                            __m256i b20 = _mm256_loadu_si256((__m256i*)(b2_ptr + k));
                            __m256i b30 = _mm256_loadu_si256((__m256i*)(b3_ptr + k));

                            __m256i b01 = _mm256_loadu_si256((__m256i*)(b0_ptr + k + 16));
                            __m256i b11 = _mm256_loadu_si256((__m256i*)(b1_ptr + k + 16));
                            __m256i b21 = _mm256_loadu_si256((__m256i*)(b2_ptr + k + 16));
                            __m256i b31 = _mm256_loadu_si256((__m256i*)(b3_ptr + k + 16));

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, b00));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, b10));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, b20));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, b30));
                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a1, b01));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1, b11));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a1, b21));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a1, b31));
                        }

                        acc[jj + 0] += add_reduce_m256i(c0) * a_s * b_q16_s[jj * TK / group_size + group];
                        acc[jj + 1] += add_reduce_m256i(c1) * a_s * b_q16_s[(jj + 1) * TK / group_size + group];
                        acc[jj + 2] += add_reduce_m256i(c2) * a_s * b_q16_s[(jj + 2) * TK / group_size + group];
                        acc[jj + 3] += add_reduce_m256i(c3) * a_s * b_q16_s[(jj + 3) * TK / group_size + group];
                    }
                }

                for (size_t j = 0; j < TN; ++j) {
                    size_t out_idx = i * N + j_tile + j;
                    if (k_tile == 0) {
                        float bias = 0.0f;
                        if (mat_bias_in && mat_bias_scale) {
                            bias = (float)mat_bias_in[j_tile + j] * mat_bias_scale[(j_tile + j) / group_size];
                        }
                        mat_C[out_idx] = acc[j] + bias;
                    } else {
                        mat_C[out_idx] += acc[j];
                    }
                }
            
                /** TILE J out TILE K
                for (size_t jj = 0; jj < TN; jj += 4) {
                    float acc[4] = {0.0f};

                    const float *b0_s_ptr = b_q16_s + jj * TK / group_size;
                    const float *b1_s_ptr = b_q16_s + (jj + 1) * TK / group_size;
                    const float *b2_s_ptr = b_q16_s + (jj + 2) * TK / group_size;
                    const float *b3_s_ptr = b_q16_s + (jj + 3) * TK / group_size;

                    for (size_t kk = 0; kk < TK; kk += group_size) {
                        const int16_t *aq_now = a_q16 + kk + k_tile;

                        const int16_t *b0_ptr = b_q16 + jj * TK + kk;
                        const int16_t *b1_ptr = b_q16 + (jj + 1) * TK + kk;
                        const int16_t *b2_ptr = b_q16 + (jj + 2) * TK + kk;
                        const int16_t *b3_ptr = b_q16 + (jj + 3) * TK + kk;

                        __m256i c0 = _mm256_setzero_si256();
                        __m256i c1 = _mm256_setzero_si256();
                        __m256i c2 = _mm256_setzero_si256();
                        __m256i c3 = _mm256_setzero_si256();

                        for (size_t k = 0; k < group_size; k += 32) {
                            __m256i a0 = _mm256_loadu_si256((__m256i*)(aq_now + k));
                            __m256i a1 = _mm256_loadu_si256((__m256i*)(aq_now + k + 16));

                            __m256i b00 = _mm256_loadu_si256((__m256i*)(b0_ptr + k));
                            __m256i b10 = _mm256_loadu_si256((__m256i*)(b1_ptr + k));
                            __m256i b20 = _mm256_loadu_si256((__m256i*)(b2_ptr + k));
                            __m256i b30 = _mm256_loadu_si256((__m256i*)(b3_ptr + k));

                            __m256i b01 = _mm256_loadu_si256((__m256i*)(b0_ptr + k + 16));
                            __m256i b11 = _mm256_loadu_si256((__m256i*)(b1_ptr + k + 16));
                            __m256i b21 = _mm256_loadu_si256((__m256i*)(b2_ptr + k + 16));
                            __m256i b31 = _mm256_loadu_si256((__m256i*)(b3_ptr + k + 16));

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, b00));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, b10));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, b20));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, b30));

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a1, b01));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1, b11));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a1, b21));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a1, b31));
                        }

                        const float a_s = a_q16_s[(kk + k_tile) / group_size];
                        const size_t group = kk / group_size;
                        acc[0] += add_reduce_m256i(c0) * a_s * b0_s_ptr[group];
                        acc[1] += add_reduce_m256i(c1) * a_s * b1_s_ptr[group];
                        acc[2] += add_reduce_m256i(c2) * a_s * b2_s_ptr[group];
                        acc[3] += add_reduce_m256i(c3) * a_s * b3_s_ptr[group];
                    }

                    for (int x = 0; x < 4; ++x) {
                        size_t out_idx = i * N + (jj + j_tile) + x;
                        if (k_tile == 0) {
                            float bias = 0.0f;
                            if (mat_bias_in && mat_bias_scale) {
                                bias = (float)mat_bias_in[jj + j_tile + x] * mat_bias_scale[(jj + j_tile) / group_size];
                            }
                            mat_C[out_idx] = acc[x] + bias;
                        } else {
                            mat_C[out_idx] += acc[x];
                        }
                    }
                }
                 */
            }
        }
    }
}

void sm_M_lg_N_lg_K_transpose(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    constexpr size_t TN = 16;
    // Pre-calculate this once outside all loops
    const float inv_group_size = 1.0f / (float)group_size;
    const float inv_127 = 1.0f / 127.0f;

    for (size_t i = 0; i < M; ++i) {
        int16_t a_q16[K];
        float a_q16_s[K / group_size];
        const float *a0_ptr = mat_A + i * K;

        for (size_t kk = 0; kk < K; kk += group_size) {
            __m256 v_sumsq0 = _mm256_setzero_ps();
            __m256 v_sumsq1 = _mm256_setzero_ps();

            for (size_t k = kk; k < kk + group_size; k += 16) {
                __m256 v_a0 = _mm256_loadu_ps(a0_ptr + k);
                __m256 v_a1 = _mm256_loadu_ps(a0_ptr + k + 8);

                v_sumsq0 = _mm256_fmadd_ps(v_a0, v_a0, v_sumsq0);
                v_sumsq1 = _mm256_fmadd_ps(v_a1, v_a1, v_sumsq1);
            }

            // Combine at the end
            float sumsq = add_reduce_mm_256(_mm256_add_ps(v_sumsq0, v_sumsq1));

            // Inside the loop:
            float rms = sqrtf(sumsq * inv_group_size + 1e-6f);
            float s_val = rms * inv_127; 

            a_q16_s[kk / group_size] = s_val;

            __m256 v_inv_Sa = _mm256_set1_ps(1 / s_val);

            for (size_t k = kk; k < kk + group_size; k += 16) {
                // 1. Load 16 floats
                __m256 f0 = _mm256_loadu_ps(a0_ptr + k);
                __m256 f1 = _mm256_loadu_ps(a0_ptr + k + 8);

                // 2. Scale and Convert to i32
                __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, v_inv_Sa));
                __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, v_inv_Sa));

                // 3. Pack (This creates the "Lane Mess")
                __m256i packed = _mm256_packs_epi32(i0, i1);

                // 4. FIX THE LANES (The Magic Step)
                // This permutes 64-bit quads to restore linear order: 0, 2, 1, 3
                __m256i linear = _mm256_permute4x64_epi64(packed, 0xD8); 

                // 5. Store
                _mm256_storeu_si256((__m256i*)(a_q16 + k), linear);
            }
        }

        #pragma omp parallel for
        for (size_t jj = 0; jj < N; jj += 4) {
            float total_acc_0 = 0.0f;
            float total_acc_1 = 0.0f;
            float total_acc_2 = 0.0f;
            float total_acc_3 = 0.0f;

            const int8_t *b0_ptr = mat_B_in + jj * K;
            const int8_t *b1_ptr = mat_B_in + (jj + 1) * K;
            const int8_t *b2_ptr = mat_B_in + (jj + 2) * K;
            const int8_t *b3_ptr = mat_B_in + (jj + 3) * K;
            
            for (size_t kk = 0; kk < K; kk += group_size) {
                __m256i c00 = _mm256_setzero_si256();
                __m256i c10 = _mm256_setzero_si256();
                __m256i c20 = _mm256_setzero_si256();
                __m256i c30 = _mm256_setzero_si256();

                __m256i c01 = _mm256_setzero_si256();
                __m256i c11 = _mm256_setzero_si256();
                __m256i c21 = _mm256_setzero_si256();
                __m256i c31 = _mm256_setzero_si256();

                for (size_t k = kk; k < kk + group_size; k += 32) {
                    __m256i a0 = _mm256_loadu_si256((__m256i*)(a_q16 + k));
                    __m256i a1 = _mm256_loadu_si256((__m256i*)(a_q16 + k + 16));

                    __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b0_ptr + k)));
                    __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b1_ptr + k)));
                    __m256i b2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b2_ptr + k)));
                    __m256i b3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b3_ptr + k)));

                    c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(a0, b0));
                    c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(a0, b1));
                    c20 = _mm256_add_epi32(c20, _mm256_madd_epi16(a0, b2));
                    c30 = _mm256_add_epi32(c30, _mm256_madd_epi16(a0, b3));

                    b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b0_ptr + k + 16)));
                    b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b1_ptr + k + 16)));
                    b2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b2_ptr + k + 16)));
                    b3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b3_ptr + k + 16)));

                    c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(a1, b0));
                    c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(a1, b1));
                    c21 = _mm256_add_epi32(c21, _mm256_madd_epi16(a1, b2));
                    c31 = _mm256_add_epi32(c31, _mm256_madd_epi16(a1, b3));
                }

                float b0_q16_s = mat_B_scales[(jj * K + kk) / group_size];
                float b1_q16_s = mat_B_scales[((jj + 1) * K + kk) / group_size];
                float b2_q16_s = mat_B_scales[((jj + 2) * K + kk) / group_size];
                float b3_q16_s = mat_B_scales[((jj + 3) * K + kk) / group_size];
                const float a_s = a_q16_s[kk / group_size];

                total_acc_0 += add_reduce_m256i(_mm256_add_epi32(c00, c01)) * a_s * b0_q16_s;
                total_acc_1 += add_reduce_m256i(_mm256_add_epi32(c10, c11)) * a_s * b1_q16_s;
                total_acc_2 += add_reduce_m256i(_mm256_add_epi32(c20, c21)) * a_s * b2_q16_s;
                total_acc_3 += add_reduce_m256i(_mm256_add_epi32(c30, c31)) * a_s * b3_q16_s;
            }
            
            if (mat_bias_in && mat_bias_scale) {
                mat_C[i * N + jj] = (float)mat_bias_in[jj] * mat_bias_scale[jj / group_size] + total_acc_0;
                mat_C[i * N + jj + 1] = (float)mat_bias_in[jj + 1] * mat_bias_scale[(jj + 1) / group_size] + total_acc_1;
                mat_C[i * N + jj + 2] = (float)mat_bias_in[jj + 2] * mat_bias_scale[(jj + 2) / group_size] + total_acc_2;
                mat_C[i * N + jj + 3] = (float)mat_bias_in[jj + 3] * mat_bias_scale[(jj + 3) / group_size] + total_acc_3;
            } else {
                mat_C[i * N + jj] = total_acc_0;
                mat_C[i * N + jj + 1] = total_acc_1;
                mat_C[i * N + jj + 2] = total_acc_2;
                mat_C[i * N + jj + 3] = total_acc_3;
            }
        }
    }
}

// if defined INT16 x INT16
void linear_int8_fp32s_int32acc_transpose(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    if (M == 1 && N >= 1024 && K >= 1024) {
        sm_M_lg_N_lg_K_transpose(
            mat_A, mat_B_in, mat_B_scales, mat_bias_in,
            mat_bias_scale, mat_C, M, N, K, group_size
        );
        return;
    }

    constexpr size_t TN = 4 * 4;
    constexpr size_t TK = 128 * 2;
    // Pre-calculate this once outside all loops
    const float inv_group_size = 1.0f / (float)group_size;
    const float inv_127 = 1.0f / 127.0f;

    for (size_t i = 0; i < M; ++i) {
        int16_t a_q16[K];
        float a_q16_s[K / group_size];
        const float *a0_ptr = mat_A + i * K;

        for (size_t kk = 0; kk < K; kk += group_size) {
            __m256 v_sumsq0 = _mm256_setzero_ps();
            __m256 v_sumsq1 = _mm256_setzero_ps();

            for (size_t k = kk; k < kk + group_size; k += 16) {
                __m256 v_a0 = _mm256_loadu_ps(a0_ptr + k);
                __m256 v_a1 = _mm256_loadu_ps(a0_ptr + k + 8);

                v_sumsq0 = _mm256_fmadd_ps(v_a0, v_a0, v_sumsq0);
                v_sumsq1 = _mm256_fmadd_ps(v_a1, v_a1, v_sumsq1);
            }

            // Combine at the end
            float sumsq = add_reduce_mm_256(_mm256_add_ps(v_sumsq0, v_sumsq1));

            // Inside the loop:
            float rms = sqrtf(sumsq * inv_group_size + 1e-6f);
            float s_val = rms * inv_127; 

            a_q16_s[kk / group_size] = s_val;

            __m256 v_inv_Sa = _mm256_set1_ps(1 / s_val);

            for (size_t k = kk; k < kk + group_size; k += 16) {
                // 1. Load 16 floats
                __m256 f0 = _mm256_loadu_ps(a0_ptr + k);
                __m256 f1 = _mm256_loadu_ps(a0_ptr + k + 8);

                // 2. Scale and Convert to i32
                __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, v_inv_Sa));
                __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, v_inv_Sa));

                // 3. Pack (This creates the "Lane Mess")
                __m256i packed = _mm256_packs_epi32(i0, i1);

                // 4. FIX THE LANES (The Magic Step)
                // This permutes 64-bit quads to restore linear order: 0, 2, 1, 3
                __m256i linear = _mm256_permute4x64_epi64(packed, 0xD8); 

                // 5. Store
                _mm256_storeu_si256((__m256i*)(a_q16 + k), linear);
            }
        }

        #pragma omp parallel for
        for (size_t jj = 0; jj < N; jj += TN) {
            const size_t j_end = std::min(jj + TN, N);
            const size_t j_size = j_end - jj;
            float acc[j_size] = {0.0f};

            for (size_t kk = 0; kk < K; kk += TK) {
                const size_t k_end = std::min(kk + TK, K);
                const size_t k_size = k_end - kk;

                size_t k = kk;
                for (; k + group_size <= k_end; k += group_size) {
                    const size_t group = k / group_size;

                    size_t j = jj;
                    for (; j < j_end; j += 4) {
                        const int8_t *b0_ptr = mat_B_in + j * K;
                        const int8_t *b1_ptr = mat_B_in + (j+1) * K;
                        const int8_t *b2_ptr = mat_B_in + (j+2) * K;
                        const int8_t *b3_ptr = mat_B_in + (j+3) * K;

                        // group_size % 16 == 0
                        __m256i c0 = _mm256_setzero_si256();
                        __m256i c1 = _mm256_setzero_si256();
                        __m256i c2 = _mm256_setzero_si256();
                        __m256i c3 = _mm256_setzero_si256();

                        /*
                        for (size_t k_i = k; k_i < k + group_size; k_i += 16) {
                            __m256i a0 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i));
                            __m256i a1 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i + 16));

                            __m256i b8 = _mm256_loadu_si256((__m256i*)(b0_ptr + k_i));
                            __m256i b16_00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                            __m256i b16_01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));

                            b8 = _mm256_loadu_si256((__m256i*)(b1_ptr + k_i));
                            __m256i b16_10 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                            __m256i b16_11 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, b16_00));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, b16_10));
                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a1, b16_01));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1, b16_11));

                            b8 = _mm256_loadu_si256((__m256i*)(b2_ptr + k_i));
                            b16_00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                            b16_01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));

                            b8 = _mm256_loadu_si256((__m256i*)(b3_ptr + k_i));
                            b16_10 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                            b16_11 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));

                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, b16_00));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, b16_10));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a1, b16_01));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a1, b16_11));
                        }
                        */

                        for (size_t k_i = k; k_i < k + group_size; k_i += 16) {
                            __m256i a0 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i));
                            
                            __m128i b80 = _mm_loadu_si128((const __m128i*)(b0_ptr + k_i));
                            __m256i b16_00 = _mm256_cvtepi8_epi16(b80);
                            
                            __m128i b81 = _mm_loadu_si128((const __m128i*)(b1_ptr + k_i));
                            __m256i b16_10 = _mm256_cvtepi8_epi16(b81);
                            
                            __m128i b82 = _mm_loadu_si128((const __m128i*)(b2_ptr + k_i));
                            __m256i b16_20 = _mm256_cvtepi8_epi16(b82);
                            
                            __m128i b83 = _mm_loadu_si128((const __m128i*)(b3_ptr + k_i));
                            __m256i b16_30 = _mm256_cvtepi8_epi16(b83);

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, b16_00));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, b16_10));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, b16_20));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, b16_30));
                        }

                        const float a_q_group = a_q16_s[group];
                        acc[j - jj] += add_reduce_m256i(c0) * a_q_group * mat_B_scales[j * K / group_size + group];
                        acc[j + 1 - jj] += add_reduce_m256i(c1) * a_q_group * mat_B_scales[(j + 1) * K / group_size + group];
                        acc[j + 2 - jj] += add_reduce_m256i(c2) * a_q_group * mat_B_scales[(j + 2) * K / group_size + group];
                        acc[j + 3 - jj] += add_reduce_m256i(c3) * a_q_group * mat_B_scales[(j + 3) * K / group_size + group];
                    }
                
                    for (; j < j_end; ++j) {
                        const int8_t *b0_ptr = mat_B_in + j * K;

                        // group_size % 16 == 0
                        __m256i c0 = _mm256_setzero_si256();
                        __m256i c1 = _mm256_setzero_si256();
                        __m256i c2 = _mm256_setzero_si256();
                        __m256i c3 = _mm256_setzero_si256();

                        for (size_t k_i = k; k_i < k + group_size; k_i += 64) {
                            __m256i a0 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i));
                            __m256i a1 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i + 16));
                            __m256i a2 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i + 32));
                            __m256i a3 = _mm256_loadu_si256((__m256i*)(a_q16 + k_i + 48));

                            __m256i b80 = _mm256_loadu_si256((__m256i*)(b0_ptr + k_i));
                            __m256i b16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b80));
                            __m256i b16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b80, 1));

                            __m256i b81 = _mm256_loadu_si256((__m256i*)(b0_ptr + k_i + 32));
                            __m256i b16_2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b81));
                            __m256i b16_3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b81, 1));

                            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, b16_0));
                            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a1, b16_1));
                            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a2, b16_2));
                            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a3, b16_3));
                        }

                        c0 = _mm256_add_epi32(c0, c1);
                        c2 = _mm256_add_epi32(c2, c3);

                        acc[j - jj] += add_reduce_m256i(_mm256_add_epi32(c0, c2)) * a_q16_s[group] * mat_B_scales[j * K / group_size + group];
                    }
                }
            }

            for (size_t j = jj; j < j_end; ++j) {
                size_t out_idx = i * N + j;
                float bias = 0.0f;
                if (mat_bias_in && mat_bias_scale) {
                    bias = (float)mat_bias_in[j] * mat_bias_scale[j / group_size];
                }
                mat_C[out_idx] = acc[j - jj] + bias;
            }
        }
    }
}

void linear_int8_fp32s_transpose(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    linear_int8_fp32s_int32acc_transpose(
        mat_A, mat_B_in, mat_B_scales, mat_bias_in,
        mat_bias_scale, mat_C, M, N, K, group_size
    );
    return;
    // default fallback
    // KERNEL ENSURE group_size >= 16
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const float *a0_ptr = mat_A + i * K;
            const int8_t *b0_ptr = mat_B_in + j * K;
            const float *b_s_ptr = mat_B_scales + j * K / group_size;

            float acc = 0.0f;
            if (mat_bias_in && mat_bias_scale) {
                acc = (float)mat_bias_in[j] * mat_bias_scale[j / group_size];
            }

            size_t k = 0;
            for (; k + group_size <= K; k += group_size) {
                // group_size % 16 == 0
                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();
                for (size_t k_i = k; k_i < k + group_size; k_i += 32) {
                    __m256 a0 = _mm256_loadu_ps(a0_ptr + k_i);
                    __m256 a1 = _mm256_loadu_ps(a0_ptr + k_i + 8);
                    __m256 a2 = _mm256_loadu_ps(a0_ptr + k_i + 16);
                    __m256 a3 = _mm256_loadu_ps(a0_ptr + k_i + 24);
                    
                    // 1. Load 16 int8 values into an xmm register (128-bit)// 1. Load all 32 int8 values into a single 256-bit register
                    __m256i q32 = _mm256_loadu_si256((const __m256i*)(b0_ptr + k_i));

                    // 2. Extract 128-bit chunks (16 bytes each)
                    __m128i low_16 = _mm256_castsi256_si128(q32);
                    __m128i high_16 = _mm256_extracti128_si256(q32, 1);

                    // 3. Convert first 16 bytes (low_16) into 2 float registers (8 floats each)
                    // Low 8 of low_16
                    __m256i i32_0 = _mm256_cvtepi8_epi32(low_16); 
                    __m256 b0 = _mm256_cvtepi32_ps(i32_0);
                    
                    // High 8 of low_16 (shift right by 8 bytes first)
                    __m256i i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(low_16, 8));
                    __m256 b1 = _mm256_cvtepi32_ps(i32_1);

                    // 4. Convert next 16 bytes (high_16) into 2 float registers
                    // Low 8 of high_16
                    __m256i i32_2 = _mm256_cvtepi8_epi32(high_16);
                    __m256 b2 = _mm256_cvtepi32_ps(i32_2);
                    
                    // High 8 of high_16 (shift right by 8 bytes first)
                    __m256i i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(high_16, 8));
                    __m256 b3 = _mm256_cvtepi32_ps(i32_3);

                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a2, b2, c2);
                    c3 = _mm256_fmadd_ps(a3, b3, c3);
                }

                c0 = _mm256_add_ps(c0, c1);
                c2 = _mm256_add_ps(c2, c3);

                acc += add_reduce_mm_256(_mm256_add_ps(c0, c2)) * b_s_ptr[k / group_size];
            }

            // Horizontal sum of all accumulators

            // Remainder loop
            for (; k < K; ++k) {
                float scale = mat_B_scales[(j * K + k) / group_size];
                acc += a0_ptr[k] * ((float)b0_ptr[k] * scale);
            }

            mat_C[i * N + j] = acc;
        }
    }
}

void linear_int8_fp32s_avx2_kernel(
    const float* mat_A,          // [M, K] FP32
    const int8_t* mat_B_in,      // [N, K] or [K, N] INT8
    const float* mat_B_scales,   // groupwise scales over linear B storage
    const int8_t* mat_bias_in,   // [N] INT8 (optional, can be nullptr)
    const float* mat_bias_scale, // groupwise scales over linear bias storage
    float* mat_C,                // [M, N] FP32
    size_t M,
    size_t N,
    size_t K,
    bool mat_B_transpose,
    size_t group_size
) {
    if (mat_B_transpose) {
        linear_int8_fp32s_transpose(
            mat_A, mat_B_in, mat_B_scales, mat_bias_in,
            mat_bias_scale, mat_C, M, N, K, group_size
        );
        return;
    }

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;

            // -------- GEMM --------
            for (size_t k = 0; k < K; ++k) {
                float a = mat_A[i * K + k];

                // linear index into B (matches quantizer layout)
                size_t b_linear_idx;
                b_linear_idx = k * N + j;

                size_t scale_idx = b_linear_idx / group_size;
                float scale = mat_B_scales[scale_idx];

                float b = (float)mat_B_in[b_linear_idx] * scale;
                acc += a * b;
            }

            mat_C[i * N + j] = acc;
        }
    }

    

    // -------- Bias --------
    if (mat_bias_in && mat_bias_scale) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t bias_scale_idx = j / group_size;
                float bias =
                    (float)mat_bias_in[j] * mat_bias_scale[bias_scale_idx];
                mat_C[i * N + j] += bias;
            }
        }
    }
}
#endif
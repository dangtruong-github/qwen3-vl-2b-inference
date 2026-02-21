#include "../include/cpu_wrapper.hpp"

#define VERY_LARGE_N 65536

#if defined(__AVX2__) && defined(__FMA__)
void gemv_lg_N_K(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    // Pre-calculate this once outside all loops
    const float inv_group_size = 1.0f / (float)group_size;
    const float inv_127 = 1.0f / 127.0f;

    alignas(32) int16_t a_q16[K];
    float a_q16_s[K / group_size];

    for (size_t kk = 0; kk < K; kk += group_size) {
        __m256 v_sumsq0 = _mm256_setzero_ps();
        __m256 v_sumsq1 = _mm256_setzero_ps();

        for (size_t k = kk; k < kk + group_size; k += 16) {
            __m256 v_a0 = _mm256_loadu_ps(mat_A + k);
            __m256 v_a1 = _mm256_loadu_ps(mat_A + k + 8);

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
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 f1 = _mm256_loadu_ps(mat_A + k + 8);

            // 2. Scale and Convert to i32
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, v_inv_Sa));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, v_inv_Sa));

            // 3. Pack (This creates the "Lane Mess")
            __m256i packed = _mm256_packs_epi32(i0, i1);

            // 4. FIX THE LANES (The Magic Step)
            // This permutes 64-bit quads to restore linear order: 0, 2, 1, 3
            __m256i linear = _mm256_permute4x64_epi64(packed, 0xD8); 

            // 5. Store
            _mm256_store_si256((__m256i*)(a_q16 + k), linear);
        }
    }

    const size_t K_g = K / group_size;
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; jj += 4) {
        __m256 c0_f = _mm256_setzero_ps();
        __m256 c1_f = _mm256_setzero_ps();
        __m256 c2_f = _mm256_setzero_ps();
        __m256 c3_f = _mm256_setzero_ps();

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        const int8_t *__restrict b1_ptr = mat_B_in + (jj + 1) * K;
        const int8_t *__restrict b2_ptr = mat_B_in + (jj + 2) * K;
        const int8_t *__restrict b3_ptr = mat_B_in + (jj + 3) * K;
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = (jj * K + kk) / group_size;
            const float *__restrict b_s_ptr = mat_B_scales + g_off;

            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();

            for (size_t k = kk; k < kk + group_size; k += 16) {
                __m256i a0 = _mm256_load_si256((__m256i*)(a_q16 + k));

                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b0_ptr + k)))));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b1_ptr + k)))));
                c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b2_ptr + k)))));
                c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b3_ptr + k)))));
            }

            const float a_s = a_q16_s[kk / group_size];
            
            c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_s * b_s_ptr[0]), c0_f);
            c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a_s * b_s_ptr[K_g]), c1_f);
            c2_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), _mm256_set1_ps(a_s * b_s_ptr[2 * K_g]), c2_f);
            c3_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), _mm256_set1_ps(a_s * b_s_ptr[3 * K_g]), c3_f);
        }
        
        mat_C[jj] = add_reduce_mm_256(c0_f);
        mat_C[jj + 1] = add_reduce_mm_256(c1_f);
        mat_C[jj + 2] = add_reduce_mm_256(c2_f);
        mat_C[jj + 3] = add_reduce_mm_256(c3_f);
    }
}

void gemv_lg_N_K_2_new(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    // Pre-calculate this once outside all loops
    const float inv_group_size = 1.0f / (float)group_size;
    const float inv_127 = 2.0f / (127.0f);

    alignas(64) uint8_t a_q8[K];
    float a_q8_s[K / group_size];

    for (int kk = 0; kk < K; kk += group_size) {
        // 1. Find Max Absolute instead of RMS for better range coverage
        __m256 v_max = _mm256_setzero_ps();
        for (int k = kk; k < kk + group_size; k += 8) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 abs_f0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), f0);
            v_max = _mm256_max_ps(v_max, abs_f0);
        }
        float max_val = max_reduce_mm_256(v_max); 
        float scale = max_val / 127.0f;
        float inv_scale = (max_val > 0) ? 1.0f / scale : 0.0f;
        a_q8_s[kk / group_size] = scale;

        __m256 invS = _mm256_set1_ps(inv_scale);
        __m256 fmin = _mm256_set1_ps(-128.0f);
        __m256 fmax = _mm256_set1_ps(127.0f);
        __m256i zp = _mm256_set1_epi8((char)128);

        // ---------------- quantize ----------------
        for (int k = kk; k < kk + group_size; k += 32) {
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 f1 = _mm256_loadu_ps(mat_A + k + 8);
            __m256 f2 = _mm256_loadu_ps(mat_A + k + 16);
            __m256 f3 = _mm256_loadu_ps(mat_A + k + 24);

            f0 = _mm256_mul_ps(f0, invS);
            f1 = _mm256_mul_ps(f1, invS);
            f2 = _mm256_mul_ps(f2, invS);
            f3 = _mm256_mul_ps(f3, invS);

            f0 = _mm256_min_ps(_mm256_max_ps(f0, fmin), fmax);
            f1 = _mm256_min_ps(_mm256_max_ps(f1, fmin), fmax);
            f2 = _mm256_min_ps(_mm256_max_ps(f2, fmin), fmax);
            f3 = _mm256_min_ps(_mm256_max_ps(f3, fmin), fmax);

            __m256i i0 = _mm256_cvtps_epi32(f0);
            __m256i i1 = _mm256_cvtps_epi32(f1);
            __m256i i2 = _mm256_cvtps_epi32(f2);
            __m256i i3 = _mm256_cvtps_epi32(f3);

            // int32 -> int16
            __m256i p01 = _mm256_packs_epi32(i0, i1);
            __m256i p23 = _mm256_packs_epi32(i2, i3);

            // fix lane order
            p01 = _mm256_permute4x64_epi64(p01, 0xD8);
            p23 = _mm256_permute4x64_epi64(p23, 0xD8);

            // int16 -> int8 (SIGNED)
            __m256i q8s = _mm256_packs_epi16(p01, p23);
            q8s = _mm256_permute4x64_epi64(q8s, _MM_SHUFFLE(3, 1, 2, 0));

            // 🔥 ADD 128 to convert signed → unsigned
            __m256i q8u = _mm256_add_epi8(q8s, zp);

            // store uint8
            _mm256_store_si256((__m256i*)(a_q8 + k), q8u);
        }
    }

    const size_t K_g = K / group_size;

    const __m512i ones8 = _mm512_set1_epi8(1);
    
    #pragma omp parallel for schedule(static)
    for (size_t jj = 0; jj < N; ++jj) {
        __m512 c0_f = _mm512_setzero_ps();

        const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
        
        for (size_t kk = 0; kk < K; kk += group_size) {
            const size_t g_off = (jj * K + kk) / group_size;
            const float *__restrict b_s_ptr = mat_B_scales + g_off;

            __m512i c0 = _mm512_setzero_si512();
            __m512i corr32 = _mm512_setzero_si512();

            for (size_t k = kk; k < kk + group_size; k += 64) {
                __m512i a0 = _mm512_loadu_si512((__m512i*)(a_q8 + k));
                __m512i b0 = _mm512_loadu_si512((__m512i*)(b0_ptr + k));

                corr32 = _mm512_dpbusd_epi32(corr32, ones8, b0);
                c0 = _mm512_dpbusd_epi32(c0, a0, b0);
            }

            const float a_s = a_q8_s[kk / group_size];

            corr32 = _mm512_slli_epi32(corr32, 7);
            c0 = _mm512_sub_epi32(c0, corr32);
            
            c0_f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(a_s * b_s_ptr[0]), c0_f);
        }
        
        mat_C[jj] = _mm512_reduce_add_ps(c0_f);
    }
}

void gemv_lg_N_K_decode(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t N, size_t K, size_t group_size
) {
    constexpr size_t TN = 1024;
    constexpr size_t PF_DT = TN / 16;
    constexpr size_t PF_S = 4 * TN * 2048;
    // Pre-calculate this once outside all loops
    const float inv_group_size = 1.0f / (float)group_size;
    const float inv_127 = 1.0f / 127.0f;

    alignas(32) int16_t a_q16[K];
    float a_q16_s[K / group_size];

    // fetch 4 lines ahead
    for (size_t p = 0; p < PF_DT; ++p) {
        _mm_prefetch((const char *)(mat_B_in + p * PF_S), _MM_HINT_T1);
    }

    for (size_t kk = 0; kk < K; kk += group_size) {
        __m256 v_sumsq0 = _mm256_setzero_ps();
        __m256 v_sumsq1 = _mm256_setzero_ps();

        for (size_t k = kk; k < kk + group_size; k += 16) {
            __m256 v_a0 = _mm256_loadu_ps(mat_A + k);
            __m256 v_a1 = _mm256_loadu_ps(mat_A + k + 8);

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
            __m256 f0 = _mm256_loadu_ps(mat_A + k);
            __m256 f1 = _mm256_loadu_ps(mat_A + k + 8);

            // 2. Scale and Convert to i32
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, v_inv_Sa));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, v_inv_Sa));

            // 3. Pack (This creates the "Lane Mess")
            __m256i packed = _mm256_packs_epi32(i0, i1);

            // 4. FIX THE LANES (The Magic Step)
            // This permutes 64-bit quads to restore linear order: 0, 2, 1, 3
            __m256i linear = _mm256_permute4x64_epi64(packed, 0xD8); 

            // 5. Store
            _mm256_store_si256((__m256i*)(a_q16 + k), linear);
        }
    }

    const size_t K_g = K / group_size;
    
    for (size_t j_out = 0; j_out < N; j_out += TN) {
        size_t j_end = std::min(j_out + TN, N);
        
        if (j_end < N) {
            // fetch 4 lines ahead
            const size_t off = j_end * K * 8;
            for (size_t p = 0; p < PF_DT; ++p) {
                _mm_prefetch((const char *)(mat_B_in + off + p * PF_S), _MM_HINT_T1);
            }
        }

        #pragma omp parallel for
        for (size_t jj = j_out; jj < j_end; jj += 4) {
            const size_t rs_base = jj * K_g;
            __m256 c0_f = _mm256_setzero_ps();
            __m256 c1_f = _mm256_setzero_ps();
            __m256 c2_f = _mm256_setzero_ps();
            __m256 c3_f = _mm256_setzero_ps();
            // __m256 c4_f = _mm256_setzero_ps();

            const int8_t *__restrict b0_ptr = mat_B_in + jj * K;
            const int8_t *__restrict b1_ptr = mat_B_in + (jj + 1) * K;
            const int8_t *__restrict b2_ptr = mat_B_in + (jj + 2) * K;
            const int8_t *__restrict b3_ptr = mat_B_in + (jj + 3) * K;
            // const int8_t *__restrict b4_ptr = mat_B_in + (jj + 4) * K;
            
            for (size_t kk = 0; kk < K; kk += group_size) {
                const float *__restrict b_s_ptr = mat_B_scales + rs_base + kk / group_size;

                __m256i c0 = _mm256_setzero_si256();
                __m256i c1 = _mm256_setzero_si256();
                __m256i c2 = _mm256_setzero_si256();
                __m256i c3 = _mm256_setzero_si256();
                // __m256i c4 = _mm256_setzero_si256();

                for (size_t k = kk; k < kk + group_size; k += 16) {
                    __m256i a0 = _mm256_load_si256((__m256i*)(a_q16 + k));

                    c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b0_ptr + k)))));
                    c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b1_ptr + k)))));
                    c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b2_ptr + k)))));
                    c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b3_ptr + k)))));
                    // c4 = _mm256_add_epi32(c4, _mm256_madd_epi16(a0, _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(b4_ptr + k)))));
                }

                const float a_s = a_q16_s[kk / group_size];
                
                c0_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_set1_ps(a_s * b_s_ptr[0]), c0_f);
                c1_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_set1_ps(a_s * b_s_ptr[K_g]), c1_f);
                c2_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), _mm256_set1_ps(a_s * b_s_ptr[2 * K_g]), c2_f);
                c3_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), _mm256_set1_ps(a_s * b_s_ptr[3 * K_g]), c3_f);
                // c4_f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c4), _mm256_set1_ps(a_s * b_s_ptr[4 * K_g]), c4_f);
            }
            
            mat_C[jj] = add_reduce_mm_256(c0_f);
            mat_C[jj + 1] = add_reduce_mm_256(c1_f);
            mat_C[jj + 2] = add_reduce_mm_256(c2_f);
            mat_C[jj + 3] = add_reduce_mm_256(c3_f);
            // mat_C[jj + 4] = add_reduce_mm_256(c4_f);
        }
    }
}

// if defined INT16 x INT16
void f32a_i8f32sb_f32c_avx2_kernel(
    const float *__restrict mat_A,
    const int8_t *__restrict mat_B_in,
    const float *__restrict mat_B_scales,
    float *__restrict mat_C, size_t M, size_t N, size_t K, size_t group_size
) {
    if (M == 1 && N >= 1024 && K >= 1024) {
        if (N > VERY_LARGE_N) {
            gemv_lg_N_K_decode(
                mat_A, mat_B_in, mat_B_scales, mat_C, N, K, group_size
            );
        } else {
            gemv_lg_N_K(
                mat_A, mat_B_in, mat_B_scales, mat_C, N, K, group_size
            );
        }
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
                mat_C[out_idx] = acc[j - jj];
            }
        }
    }
}
#endif

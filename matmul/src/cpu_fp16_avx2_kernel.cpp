#include "../include/cpu_fp16_avx2_kernel.hpp"

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

void sm_M_lg_K_N_transpose(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K
) {
    memset(mat_C, 0, M * N * sizeof(float));
    constexpr size_t TN = 64;

    for (size_t i = 0; i < M; ++i) {
        const float *A_ptr = mat_A + i * K;
        float *C_ptr = mat_C + i * N;

        #pragma omp parallel for
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);

            size_t j = jj;
            for (; j + 4 <= j_end; j += 4) {
                const half_cpu *b0_ptr = mat_B + j * K;
                const half_cpu *b1_ptr = mat_B + (j+1) * K;
                const half_cpu *b2_ptr = mat_B + (j+2) * K;
                const half_cpu *b3_ptr = mat_B + (j+3) * K;

                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();

                size_t k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m256 a0 = _mm256_loadu_ps(A_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(A_ptr + k + 8);

                    __m128i hb00 = _mm_loadu_si128((const __m128i*)(b0_ptr + k));
                    __m256 b00 = _mm256_cvtph_ps(hb00);
                    __m128i hb10 = _mm_loadu_si128((const __m128i*)(b1_ptr + k));
                    __m256 b10 = _mm256_cvtph_ps(hb10);
                    __m128i hb20 = _mm_loadu_si128((const __m128i*)(b2_ptr + k));
                    __m256 b20 = _mm256_cvtph_ps(hb20);
                    __m128i hb30 = _mm_loadu_si128((const __m128i*)(b3_ptr + k));
                    __m256 b30 = _mm256_cvtph_ps(hb30);

                    __m128i hb01 = _mm_loadu_si128((const __m128i*)(b0_ptr + k + 8));
                    __m256 b01 = _mm256_cvtph_ps(hb01);
                    __m128i hb11 = _mm_loadu_si128((const __m128i*)(b1_ptr + k + 8));
                    __m256 b11 = _mm256_cvtph_ps(hb11);
                    __m128i hb21 = _mm_loadu_si128((const __m128i*)(b2_ptr + k + 8));
                    __m256 b21 = _mm256_cvtph_ps(hb21);
                    __m128i hb31 = _mm_loadu_si128((const __m128i*)(b3_ptr + k + 8));
                    __m256 b31 = _mm256_cvtph_ps(hb31);

                    c0 = _mm256_fmadd_ps(a0, b00, c0);
                    c1 = _mm256_fmadd_ps(a0, b10, c1);
                    c2 = _mm256_fmadd_ps(a0, b20, c2);
                    c3 = _mm256_fmadd_ps(a0, b30, c3);

                    c0 = _mm256_fmadd_ps(a1, b01, c0);
                    c1 = _mm256_fmadd_ps(a1, b11, c1);
                    c2 = _mm256_fmadd_ps(a1, b21, c2);
                    c3 = _mm256_fmadd_ps(a1, b31, c3);
                }

                float res_c0 = add_reduce_mm_256(c0);
                float res_c1 = add_reduce_mm_256(c1);
                float res_c2 = add_reduce_mm_256(c2);
                float res_c3 = add_reduce_mm_256(c3);

                // cleanup
                for (; k < K; ++k) {
                    float a = A_ptr[k];
                    res_c0 += a * static_cast<float>(b0_ptr[k]);
                    res_c1 += a * static_cast<float>(b1_ptr[k]);
                    res_c2 += a * static_cast<float>(b2_ptr[k]);
                    res_c3 += a * static_cast<float>(b3_ptr[k]);
                }

                C_ptr[j] = res_c0;
                C_ptr[j + 1] = res_c1;
                C_ptr[j + 2] = res_c2;
                C_ptr[j + 3] = res_c3;
            }

            for (; j + 2 <= j_end; j += 2) {
                const half_cpu *b0_ptr = mat_B + j * K;
                const half_cpu *b1_ptr = mat_B + (j+1) * K;

                __m256 c00 = _mm256_setzero_ps();
                __m256 c10 = _mm256_setzero_ps();
                __m256 c01 = _mm256_setzero_ps();
                __m256 c11 = _mm256_setzero_ps();

                size_t k = 0;
                for (; k + 24 <= K; k += 24) {
                    __m256 a0 = _mm256_loadu_ps(A_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(A_ptr + k + 8);
                    __m256 a2 = _mm256_loadu_ps(A_ptr + k + 16);

                    __m128i hb00 = _mm_loadu_si128((const __m128i*)(b0_ptr + k));
                    __m256 b00 = _mm256_cvtph_ps(hb00);
                    __m128i hb10 = _mm_loadu_si128((const __m128i*)(b1_ptr + k));
                    __m256 b10 = _mm256_cvtph_ps(hb10);

                    __m128i hb01 = _mm_loadu_si128((const __m128i*)(b0_ptr + k + 8));
                    __m256 b01 = _mm256_cvtph_ps(hb01);
                    __m128i hb11 = _mm_loadu_si128((const __m128i*)(b1_ptr + k + 8));
                    __m256 b11 = _mm256_cvtph_ps(hb11);

                    __m128i hb02 = _mm_loadu_si128((const __m128i*)(b0_ptr + k + 16));
                    __m256 b02 = _mm256_cvtph_ps(hb02);
                    __m128i hb12 = _mm_loadu_si128((const __m128i*)(b1_ptr + k + 16));
                    __m256 b12 = _mm256_cvtph_ps(hb12);

                    c00 = _mm256_fmadd_ps(a0, b00, c00);
                    c10 = _mm256_fmadd_ps(a0, b10, c10);
                    c01 = _mm256_fmadd_ps(a1, b01, c01);
                    c11 = _mm256_fmadd_ps(a1, b11, c11);
                    c00 = _mm256_fmadd_ps(a2, b02, c00);
                    c10 = _mm256_fmadd_ps(a2, b12, c10);
                }

                float res_c0 = add_reduce_mm_256(_mm256_add_ps(c00, c01));
                float res_c1 = add_reduce_mm_256(_mm256_add_ps(c10, c11));

                // cleanup
                for (; k < K; ++k) {
                    float a = A_ptr[k];
                    res_c0 += a * static_cast<float>(b0_ptr[k]);
                    res_c1 += a * static_cast<float>(b1_ptr[k]);
                }

                C_ptr[j] = res_c0;
                C_ptr[j + 1] = res_c1;
            }

            for (; j < j_end; ++j) {
                const half_cpu *B_ptr = mat_B + j * K;

                __m256 c0 = _mm256_setzero_ps();
                __m256 c1 = _mm256_setzero_ps();
                __m256 c2 = _mm256_setzero_ps();
                __m256 c3 = _mm256_setzero_ps();

                size_t k = 0;
                for (; k + 32 <= K; k += 32) {
                    __m128i hb0 = _mm_loadu_si128((const __m128i*)(B_ptr + k));
                    __m256 b0 = _mm256_cvtph_ps(hb0);
                    __m128i hb1 = _mm_loadu_si128((const __m128i*)(B_ptr + k + 8));
                    __m256 b1 = _mm256_cvtph_ps(hb1);
                    __m128i hb2 = _mm_loadu_si128((const __m128i*)(B_ptr + k + 16));
                    __m256 b2 = _mm256_cvtph_ps(hb2);
                    __m128i hb3 = _mm_loadu_si128((const __m128i*)(B_ptr + k + 24));
                    __m256 b3 = _mm256_cvtph_ps(hb3);

                    __m256 a0 = _mm256_loadu_ps(A_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(A_ptr + k + 8);
                    __m256 a2 = _mm256_loadu_ps(A_ptr + k + 16);
                    __m256 a3 = _mm256_loadu_ps(A_ptr + k + 24);

                    c0 = _mm256_fmadd_ps(a0, b0, c0);
                    c1 = _mm256_fmadd_ps(a1, b1, c1);
                    c2 = _mm256_fmadd_ps(a2, b2, c2);
                    c3 = _mm256_fmadd_ps(a3, b3, c3);
                }

                c0 = _mm256_add_ps(c0, c1);
                c2 = _mm256_add_ps(c2, c3);
                float res_C = add_reduce_mm_256(_mm256_add_ps(c0, c2));

                // cleanup
                for (; k < K; ++k) {
                    res_C += A_ptr[k] * static_cast<float>(B_ptr[k]);
                }

                C_ptr[j] = res_C;
            }
        }
    }

    if (mat_bias != nullptr) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                // F16C conversion happens here
                mat_C[i * N + j] += static_cast<float>(mat_bias[j]);
            }
        }
    }
}

void lg_M_N_K_transpose(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K
) {
    constexpr size_t TN = 8;
    constexpr size_t TK = 256;
    
    
    for (size_t jj = 0; jj < N; jj += TN) {
        size_t j_end = std::min(jj + TN, N);
        size_t j_size = j_end - jj;

        float packed_B[j_size * TK] __attribute__((aligned(32)));

        for (size_t kk = 0; kk < K; kk += TK) {
            size_t k_end = std::min(kk + TK, K);
            size_t k_size = k_end - kk;

            for (size_t j = 0; j < j_size; ++j) {
                size_t k = 0;
                const half_cpu* b_ptr = &mat_B[(j + jj) * K + kk];
                float *packed_B_ptr = packed_B + j * TK;

                // Process 32 elements per iteration (4 * 8)
                for (; k + 63 < k_size; k += 64) {
                    // Load 8 blocks of 128-bit (8 x fp16)
                    __m128i v16_0 = _mm_loadu_si128((const __m128i*)(b_ptr + k));
                    __m128i v16_1 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 8));
                    __m128i v16_2 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 16));
                    __m128i v16_3 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 24));
                    __m128i v16_4 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 32));
                    __m128i v16_5 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 40));
                    __m128i v16_6 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 48));
                    __m128i v16_7 = _mm_loadu_si128((const __m128i*)(b_ptr + k + 56));

                    // Store converted results (Aligned store is key here)
                    _mm256_store_ps(packed_B_ptr + k,      _mm256_cvtph_ps(v16_0));
                    _mm256_store_ps(packed_B_ptr + k + 8,  _mm256_cvtph_ps(v16_1));
                    _mm256_store_ps(packed_B_ptr + k + 16, _mm256_cvtph_ps(v16_2));
                    _mm256_store_ps(packed_B_ptr + k + 24, _mm256_cvtph_ps(v16_3));
                    _mm256_store_ps(packed_B_ptr + k + 32, _mm256_cvtph_ps(v16_4));
                    _mm256_store_ps(packed_B_ptr + k + 40, _mm256_cvtph_ps(v16_5));
                    _mm256_store_ps(packed_B_ptr + k + 48, _mm256_cvtph_ps(v16_6));
                    _mm256_store_ps(packed_B_ptr + k + 56, _mm256_cvtph_ps(v16_7));
                }

                for (; k + 7 < k_size; k += 8) {
                    // Load 128 bits (8 x 16-bit floats)
                    __m128i v16 = _mm_loadu_si128((const __m128i*)(b_ptr + k));
                    __m256 v32 = _mm256_cvtph_ps(v16);
                    _mm256_store_ps(packed_B_ptr + k, v32);
                }

                for (; k < k_size; ++k) {
                    packed_B_ptr[k] = (float)b_ptr[k];
                }
            }

            size_t ii;
            #pragma omp parallel for
            for (ii = 0; ii <= M - 2; ii += 2) {
                const float *a0_ptr = mat_A + ii * K + kk;
                const float *a1_ptr = mat_A + (ii + 1) * K + kk;

                float *c0_ptr = mat_C + ii * N + jj;
                float *c1_ptr = mat_C + (ii+1) * N + jj;

                size_t j = 0;
                for (; j + 4 <= j_size; j += 4) {
                    const float *packed_b0_ptr = packed_B + j * TK;
                    const float *packed_b1_ptr = packed_B + (j+1) * TK;
                    const float *packed_b2_ptr = packed_B + (j+2) * TK;
                    const float *packed_b3_ptr = packed_B + (j+3) * TK;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();
                    
                    size_t k = 0;
                    for (; k + 8 <= k_size; k += 8) {
                        __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1 = _mm256_loadu_ps(a1_ptr + k);
                        
                        __m256 b0 = _mm256_load_ps(packed_b0_ptr + k);
                        __m256 b1 = _mm256_load_ps(packed_b1_ptr + k);
                        __m256 b2 = _mm256_load_ps(packed_b2_ptr + k);
                        __m256 b3 = _mm256_load_ps(packed_b3_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a0, b1, c01);
                        c02 = _mm256_fmadd_ps(a0, b2, c02);
                        c03 = _mm256_fmadd_ps(a0, b3, c03);
                        c10 = _mm256_fmadd_ps(a1, b0, c10);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);
                        c12 = _mm256_fmadd_ps(a1, b2, c12);
                        c13 = _mm256_fmadd_ps(a1, b3, c13);
                    }

                    float acc_00 = add_reduce_mm_256(c00);
                    float acc_01 = add_reduce_mm_256(c01);
                    float acc_02 = add_reduce_mm_256(c02);
                    float acc_03 = add_reduce_mm_256(c03);
                    float acc_10 = add_reduce_mm_256(c10);
                    float acc_11 = add_reduce_mm_256(c11);
                    float acc_12 = add_reduce_mm_256(c12);
                    float acc_13 = add_reduce_mm_256(c13);

                    for (; k < k_size; ++k) {
                        const float pb0 = packed_b0_ptr[k];
                        const float pb1 = packed_b1_ptr[k];
                        const float pb2 = packed_b2_ptr[k];
                        const float pb3 = packed_b3_ptr[k];
                        const float a0 = a0_ptr[k];
                        const float a1 = a1_ptr[k];
                        acc_00 += a0 * pb0;
                        acc_01 += a0 * pb1;
                        acc_02 += a0 * pb2;
                        acc_03 += a0 * pb3;
                        acc_10 += a1 * pb0;
                        acc_11 += a1 * pb1;
                        acc_12 += a1 * pb2;
                        acc_13 += a1 * pb3;
                    }

                    if (kk == 0) {
                        float bias[4] = {0.0f};
                        if (mat_bias) {
                            for (int x = 0; x < 4; ++x) {
                                bias[x] = (float)(mat_bias[jj + j + x]);
                            }
                        }
                        c0_ptr[j] = acc_00 + bias[0];
                        c0_ptr[j + 1] = acc_01 + bias[1];
                        c0_ptr[j + 2] = acc_02 + bias[2];
                        c0_ptr[j + 3] = acc_03 + bias[3];
                        c1_ptr[j] = acc_10 + bias[0];
                        c1_ptr[j + 1] = acc_11 + bias[1];
                        c1_ptr[j + 2] = acc_12 + bias[2];
                        c1_ptr[j + 3] = acc_13 + bias[3];
                    } else {
                        c0_ptr[j] += acc_00;
                        c0_ptr[j + 1] += acc_01;
                        c0_ptr[j + 2] += acc_02;
                        c0_ptr[j + 3] += acc_03;
                        c1_ptr[j] += acc_10;
                        c1_ptr[j + 1] += acc_11;
                        c1_ptr[j + 2] += acc_12;
                        c1_ptr[j + 3] += acc_13;
                    }
                }
                
                for (; j + 2 <= j_size; j += 2) {
                    const float *packed_b0_ptr = packed_B + j * TK;
                    const float *packed_b1_ptr = packed_B + (j+1) * TK;

                    __m256 c00a = _mm256_setzero_ps();
                    __m256 c10a = _mm256_setzero_ps();
                    __m256 c01a = _mm256_setzero_ps();
                    __m256 c11a = _mm256_setzero_ps();

                    __m256 c00b = _mm256_setzero_ps();
                    __m256 c10b = _mm256_setzero_ps();
                    __m256 c01b = _mm256_setzero_ps();
                    __m256 c11b = _mm256_setzero_ps();
                    
                    size_t k = 0;
                    for (; k + 16 <= k_size; k += 16) {
                        __m256 a00 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a10 = _mm256_loadu_ps(a1_ptr + k);
                        
                        __m256 b00 = _mm256_load_ps(packed_b0_ptr + k);
                        __m256 b10 = _mm256_load_ps(packed_b1_ptr + k);

                        c00a = _mm256_fmadd_ps(a00, b00, c00a);
                        c10a = _mm256_fmadd_ps(a10, b00, c10a);
                        c01a = _mm256_fmadd_ps(a00, b10, c01a);
                        c11a = _mm256_fmadd_ps(a10, b10, c11a);

                        a00 = _mm256_loadu_ps(a0_ptr + k + 8);
                        a10 = _mm256_loadu_ps(a1_ptr + k + 8);

                        b00 = _mm256_load_ps(packed_b0_ptr + k + 8);
                        b10 = _mm256_load_ps(packed_b1_ptr + k + 8);

                        c00b = _mm256_fmadd_ps(a00, b00, c00b);
                        c10b = _mm256_fmadd_ps(a10, b00, c10b);
                        c01b = _mm256_fmadd_ps(a00, b10, c01b);
                        c11b = _mm256_fmadd_ps(a10, b10, c11b);
                    }

                    float acc_00 = add_reduce_mm_256(_mm256_add_ps(c00a, c00b));
                    float acc_10 = add_reduce_mm_256(_mm256_add_ps(c10a, c10b));
                    float acc_01 = add_reduce_mm_256(_mm256_add_ps(c01a, c01b));
                    float acc_11 = add_reduce_mm_256(_mm256_add_ps(c11a, c11b));

                    for (; k < k_size; ++k) {
                        const float pb0 = packed_b0_ptr[k];
                        const float pb1 = packed_b1_ptr[k];
                        const float a0 = a0_ptr[k];
                        const float a1 = a1_ptr[k];
                        acc_00 += a0 * pb0;
                        acc_10 += a1 * pb0;
                        acc_01 += a0 * pb1;
                        acc_11 += a1 * pb1;
                    }

                    if (kk == 0) {
                        float bias[2] = {0.0f};
                        if (mat_bias) {
                            for (int x = 0; x < 2; ++x) {
                                bias[x] = (float)(mat_bias[jj + j + x]);
                            }
                        }
                        c0_ptr[j] = acc_00 + bias[0];
                        c0_ptr[j + 1] = acc_01 + bias[1];
                        c1_ptr[j] = acc_10 + bias[0];
                        c1_ptr[j + 1] = acc_11 + bias[1];
                    } else {
                        c0_ptr[j] += acc_00;
                        c0_ptr[j + 1] += acc_01;
                        c1_ptr[j] += acc_10;
                        c1_ptr[j + 1] += acc_11;
                    }

                    j += 2;
                }
                
                if (j_size - j >= 1) {
                    const float *packed_b_ptr = packed_B + j * TK;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 24 <= k_size; k += 24) {
                        __m256 b0 = _mm256_load_ps(packed_b_ptr + k);
                        __m256 b1 = _mm256_load_ps(packed_b_ptr + k + 8);

                        __m256 a00 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a10 = _mm256_loadu_ps(a1_ptr + k);
                        __m256 a01 = _mm256_loadu_ps(a0_ptr + k + 8);
                        __m256 a11 = _mm256_loadu_ps(a1_ptr + k + 8);

                        c00 = _mm256_fmadd_ps(a00, b0, c00);
                        c10 = _mm256_fmadd_ps(a10, b0, c10);
                        c01 = _mm256_fmadd_ps(a01, b1, c01);
                        c11 = _mm256_fmadd_ps(a11, b1, c11);

                        __m256 b2 = _mm256_load_ps(packed_b_ptr + k + 16);
                        a00 = _mm256_loadu_ps(a0_ptr + k + 16);
                        a10 = _mm256_loadu_ps(a1_ptr + k + 16);
                        c02 = _mm256_fmadd_ps(a00, b2, c02);
                        c12 = _mm256_fmadd_ps(a10, b2, c12);
                    }

                    c01 = _mm256_add_ps(c01, c00);
                    c11 = _mm256_add_ps(c11, c10);
                    
                    for (; k + 8 <= k_size; k += 8) {
                        __m256 b0 = _mm256_load_ps(packed_b_ptr + k);

                        __m256 a00 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a10 = _mm256_loadu_ps(a1_ptr + k);

                        c02 = _mm256_fmadd_ps(a00, b0, c02);
                        c12 = _mm256_fmadd_ps(a10, b0, c12);
                    }

                    float acc_0 = add_reduce_mm_256(_mm256_add_ps(c02, c01));
                    float acc_1 = add_reduce_mm_256(_mm256_add_ps(c12, c11));

                    for (; k < k_size; ++k) {
                        const float pb = packed_b_ptr[k];
                        acc_0 += a0_ptr[k] * pb;
                        acc_1 += a1_ptr[k] * pb;
                    }

                    if (kk == 0) {
                        const float bias = mat_bias ? (float)(mat_bias[jj + j]) : 0.0f;
                        c0_ptr[j] = acc_0 + bias;
                        c1_ptr[j] = acc_1 + bias;
                    } else {
                        c0_ptr[j] += acc_0;
                        c1_ptr[j] += acc_1;
                    }
                }
            
                // ii += 2;
            }
        
            if (M - ii >= 1) {
                const float *a0_ptr = mat_A + ii * K + kk;
                float *c0_ptr = mat_C + ii * N + jj;

                size_t j = 0;
                for (; j + 4 <= j_size; j += 4) {
                    const float *packed_b0_ptr = packed_B + j * TK;
                    const float *packed_b1_ptr = packed_B + (j+1) * TK;
                    const float *packed_b2_ptr = packed_B + (j+2) * TK;
                    const float *packed_b3_ptr = packed_B + (j+3) * TK;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c20 = _mm256_setzero_ps();
                    __m256 c30 = _mm256_setzero_ps();
                    
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c21 = _mm256_setzero_ps();
                    __m256 c31 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 16 <= k_size; k += 16) {
                        __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1 = _mm256_loadu_ps(a0_ptr + k + 8);

                        __m256 b0 = _mm256_load_ps(packed_b0_ptr + k);
                        __m256 b1 = _mm256_load_ps(packed_b1_ptr + k);
                        __m256 b2 = _mm256_load_ps(packed_b2_ptr + k);
                        __m256 b3 = _mm256_load_ps(packed_b3_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c10 = _mm256_fmadd_ps(a0, b1, c10);
                        c20 = _mm256_fmadd_ps(a0, b2, c20);
                        c30 = _mm256_fmadd_ps(a0, b3, c30);

                        b0 = _mm256_load_ps(packed_b0_ptr + k + 8);
                        b1 = _mm256_load_ps(packed_b1_ptr + k + 8);
                        b2 = _mm256_load_ps(packed_b2_ptr + k + 8);
                        b3 = _mm256_load_ps(packed_b3_ptr + k + 8);

                        c01 = _mm256_fmadd_ps(a1, b0, c01);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);
                        c21 = _mm256_fmadd_ps(a1, b2, c21);
                        c31 = _mm256_fmadd_ps(a1, b3, c31);
                    }

                    float acc_0 = add_reduce_mm_256(_mm256_add_ps(c00, c01));
                    float acc_1 = add_reduce_mm_256(_mm256_add_ps(c10, c11));
                    float acc_2 = add_reduce_mm_256(_mm256_add_ps(c20, c21));
                    float acc_3 = add_reduce_mm_256(_mm256_add_ps(c30, c31));

                    for (; k < k_size; ++k) {
                        const float a = a0_ptr[k];
                        acc_0 += a * packed_b0_ptr[k];
                        acc_1 += a * packed_b1_ptr[k];
                        acc_2 += a * packed_b2_ptr[k];
                        acc_3 += a * packed_b3_ptr[k];
                    }

                    if (kk == 0) {
                        float bias[4] = {0.0f};
                        if (mat_bias) {
                            for (int x = 0; x < 4; ++x) {
                                bias[x] = (float)(mat_bias[jj + j + x]);
                            }
                        }
                        c0_ptr[j] = acc_0 + bias[0];
                        c0_ptr[j + 1] = acc_1 + bias[1];
                        c0_ptr[j + 2] = acc_2 + bias[2];
                        c0_ptr[j + 3] = acc_3 + bias[3];
                    } else {
                        c0_ptr[j] += acc_0;
                        c0_ptr[j + 1] += acc_1;
                        c0_ptr[j + 2] += acc_2;
                        c0_ptr[j + 3] += acc_3;
                    }
                }
            
                if (j_size - j >= 2) {
                    const float *packed_b0_ptr = packed_B + j * TK;
                    const float *packed_b1_ptr = packed_B + (j+1) * TK;

                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 32 <= k_size; k += 32) {
                        __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1 = _mm256_loadu_ps(a0_ptr + k + 8);

                        __m256 b00 = _mm256_load_ps(packed_b0_ptr + k);
                        __m256 b01 = _mm256_load_ps(packed_b1_ptr + k);
                        __m256 b10 = _mm256_load_ps(packed_b0_ptr + k + 8);
                        __m256 b11 = _mm256_load_ps(packed_b1_ptr + k + 8);

                        c00 = _mm256_fmadd_ps(a0, b00, c00);
                        c10 = _mm256_fmadd_ps(a0, b01, c10);
                        c01 = _mm256_fmadd_ps(a1, b10, c01);
                        c11 = _mm256_fmadd_ps(a1, b11, c11);
                        
                        a0 = _mm256_loadu_ps(a0_ptr + k + 16);
                        a1 = _mm256_loadu_ps(a0_ptr + k + 24);

                        b00 = _mm256_load_ps(packed_b0_ptr + k + 16);
                        b01 = _mm256_load_ps(packed_b1_ptr + k + 16);
                        b10 = _mm256_load_ps(packed_b0_ptr + k + 24);
                        b11 = _mm256_load_ps(packed_b1_ptr + k + 24);

                        c02 = _mm256_fmadd_ps(a0, b00, c02);
                        c12 = _mm256_fmadd_ps(a0, b01, c12);
                        c03 = _mm256_fmadd_ps(a1, b10, c03);
                        c13 = _mm256_fmadd_ps(a1, b11, c13);
                    }

                    c00 = _mm256_add_ps(c00, c01);
                    c02 = _mm256_add_ps(c02, c03);
                    c10 = _mm256_add_ps(c10, c11);
                    c12 = _mm256_add_ps(c12, c13);

                    float acc_0 = add_reduce_mm_256(_mm256_add_ps(c00, c02));
                    float acc_1 = add_reduce_mm_256(_mm256_add_ps(c10, c12));

                    for (; k < k_size; ++k) {
                        const float a = a0_ptr[k];
                        acc_0 += a * packed_b0_ptr[k];
                        acc_1 += a * packed_b1_ptr[k];
                    }

                    if (kk == 0) {
                        float bias[2] = {0.0f};
                        if (mat_bias) {
                            for (int x = 0; x < 2; ++x) {
                                bias[x] = (float)(mat_bias[jj + j + x]);
                            }
                        }
                        c0_ptr[j] = acc_0 + bias[0];
                        c0_ptr[j + 1] = acc_1 + bias[1];
                    } else {
                        c0_ptr[j] += acc_0;
                        c0_ptr[j + 1] += acc_1;
                    }

                    j += 2;
                }
            
                if (j_size - j >= 1) {
                    __m256 c0 = _mm256_setzero_ps();
                    __m256 c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps();
                    __m256 c3 = _mm256_setzero_ps();
                    const float *packed_b_ptr = packed_B + j * TK;

                    size_t k = 0;
                    for (; k + 32 <= k_size; k += 32) {
                        __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 b0 = _mm256_load_ps(packed_b_ptr + k);
                        __m256 a1 = _mm256_loadu_ps(a0_ptr + k + 8);
                        __m256 b1 = _mm256_load_ps(packed_b_ptr + k + 8);
                        __m256 a2 = _mm256_loadu_ps(a0_ptr + k + 16);
                        __m256 b2 = _mm256_load_ps(packed_b_ptr + k + 16);
                        __m256 a3 = _mm256_loadu_ps(a0_ptr + k + 24);
                        __m256 b3 = _mm256_load_ps(packed_b_ptr + k + 24);

                        c0 = _mm256_fmadd_ps(a0, b0, c0);
                        c1 = _mm256_fmadd_ps(a1, b1, c1);
                        c2 = _mm256_fmadd_ps(a2, b2, c2);
                        c3 = _mm256_fmadd_ps(a3, b3, c3);
                    }

                    c0 = _mm256_add_ps(c0, c1);
                    c2 = _mm256_add_ps(c2, c3);

                    float acc_0 = add_reduce_mm_256(_mm256_add_ps(c0, c2));

                    for (; k < k_size; ++k) {
                        acc_0 += a0_ptr[k] * packed_b_ptr[k];
                    }

                    if (kk == 0) {
                        const float bias = mat_bias ? (float)(mat_bias[jj + j]) : 0.0f;
                        c0_ptr[j] = acc_0 + bias;
                    } else {
                        c0_ptr[j] += acc_0;
                    }
                }
            }
        }
    }
}

void fp16_avx2_kernel(
    const float *mat_A, const half_cpu *mat_B, const half_cpu *mat_bias,
    float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    #ifdef CPU_TIME
        CPUTimer timer("linear");
        printf("Shape of matmul FP16: M=%zu, N=%zu, K=%zu, bias=%d, B transpose=%d\n", M, N, K, (mat_bias != nullptr), mat_B_transpose);
    #endif

    if (K >= 1024 && N >= 1024) {
        if (M <= 64) {
            sm_M_lg_K_N_transpose(mat_A, mat_B, mat_bias, mat_C, M, N, K);
            return;
        } else {
            lg_M_N_K_transpose(mat_A, mat_B, mat_bias, mat_C, M, N, K);
            return;
        }
    }

    // 1. Initialize mat_C with fp16 bias (expanded to fp32) or zeros
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (mat_bias != nullptr) {
                // F16C conversion happens here
                mat_C[i * N + j] = static_cast<float>(mat_bias[j]);
            } else {
                mat_C[i * N + j] = 0.0f;
            }
        }
    }

    // 2. Compute Matrix Multiplication
    if (mat_B_transpose) {
        // A: [M, K], B: [N, K] (Stored as N rows of length K)
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = mat_C[i * N + j];
                const float* row_A = &mat_A[i * K];
                const half_cpu* row_B = &mat_B[j * K];

                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < K; ++k) {
                    // Both inputs are now in registers as fp32
                    sum += row_A[k] * static_cast<float>(row_B[k]);
                }
                mat_C[i * N + j] = sum;
            }
        }
    } else {
        // B is [K, N]. Standard GEMM.
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = mat_C[i * N + j];
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < K; ++k) {
                    sum += mat_A[i * K + k] * static_cast<float>(mat_B[k * N + j]);
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
}
#endif

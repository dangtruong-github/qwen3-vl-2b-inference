#include "../include/cpu_fp16_avx2_untranspose.hpp"

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

void lg_M_N_K(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
) {
    constexpr size_t TN = 32;
    constexpr size_t TK = 64;

    if (mat_bias) {
        // Step 1: Convert the bias chunk for this TN block ONCE
        float local_bias[N] __attribute__((aligned(32)));

        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            // 1. Load 4 chunks of 128-bit FP16 (8 elements each)
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 24));

            // 2. Convert to FP32 vectors
            __m256 v32_0 = _mm256_cvtph_ps(v16_0);
            __m256 v32_1 = _mm256_cvtph_ps(v16_1);
            __m256 v32_2 = _mm256_cvtph_ps(v16_2);
            __m256 v32_3 = _mm256_cvtph_ps(v16_3);

            _mm256_store_ps(local_bias + jb,      v32_0);
            _mm256_store_ps(local_bias + jb + 8,  v32_1);
            _mm256_store_ps(local_bias + jb + 16, v32_2);
            _mm256_store_ps(local_bias + jb + 24, v32_3);
        }
        for (; jb < N; ++jb) {
            local_bias[jb] = static_cast<float>(mat_bias[jb]);
        }

        // Step 2: Broadcast to all rows M
        const size_t copy_size = N * sizeof(float);
        #pragma omp parallel for
        for (size_t i = 0; i < M; ++i) {
            memcpy(mat_C + i * N, local_bias, copy_size);
        }
    } else {
        memset(mat_C, 0, M * N * sizeof(float));
    }
    
    #pragma omp parallel
    {
        // thread-private packed_B
        alignas(32) float packed_B[TK * TN];
        
        for (size_t kk = 0; kk < K; kk += TK) {
            size_t k_end = std::min(kk + TK, K);
            size_t k_size = k_end - kk;

            #pragma omp for schedule(static)
            for (size_t jj = 0; jj < N; jj += TN) {
                size_t j_end = std::min(jj + TN, N);
                size_t j_size = j_end - jj;
                
                for (size_t k = 0; k < k_size; ++k) {
                    size_t j = 0;
                    const half_cpu* b_ptr = &mat_B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;

                    for (; j + 8 <= j_size; j += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                        __m256 v32 = _mm256_cvtph_ps(v16);
                        _mm256_store_ps(packed_B_ptr + j, v32);
                    }

                    for (; j < j_size; ++j) {
                        packed_B_ptr[j] = (float)b_ptr[j];
                    }
                }

                for (size_t ii = 0; ii < M; ++ii) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;

                    size_t j_tile = 0;
                    for (; j_tile + 32 <= j_size; j_tile += 32) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c0 = _mm256_loadu_ps(c0_ptr + j_tile);
                        __m256 c1 = _mm256_loadu_ps(c0_ptr + j_tile + 8);
                        __m256 c2 = _mm256_loadu_ps(c0_ptr + j_tile + 16);
                        __m256 c3 = _mm256_loadu_ps(c0_ptr + j_tile + 24);

                        size_t k = 0;
                        for (; k + 2 <= k_size; k += 2) {
                            const float *pb0 = pb_j + k * TN;
                            const float *pb1 = pb_j + (k+1) * TN;
                            __m256 a0 = _mm256_set1_ps(a0_ptr[k]);
                            __m256 a1 = _mm256_set1_ps(a0_ptr[k+1]);

                            c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                            c1 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 8), c1);
                            c2 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 16), c2);
                            c3 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 24), c3);
                            c0 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1), c0);
                            c1 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1 + 8), c1);
                            c2 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1 + 16), c2);
                            c3 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1 + 24), c3);
                        }

                        for (; k < k_size; ++k) {
                            __m256 a0 = _mm256_set1_ps(a0_ptr[k]);
                            const float *pb0 = packed_B + k * TN + j_tile;

                            c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                            c1 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 8), c1);
                            c2 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 16), c2);
                            c3 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0 + 24), c3);
                        }
                        
                        _mm256_storeu_ps(c0_ptr + j_tile, c0);
                        _mm256_storeu_ps(c0_ptr + j_tile + 8, c1);
                        _mm256_storeu_ps(c0_ptr + j_tile + 16, c2);
                        _mm256_storeu_ps(c0_ptr + j_tile + 24, c3);
                    }

                    for (; j_tile + 8 <= j_size; j_tile += 8) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c0 = _mm256_loadu_ps(c0_ptr + j_tile);
                        __m256 c1 = _mm256_setzero_ps();
                        __m256 c2 = _mm256_setzero_ps();
                        __m256 c3 = _mm256_setzero_ps();

                        size_t k = 0;
                        for (; k + 4 <= k_size; k += 4) {
                            __m256 b0 = _mm256_load_ps(pb_j + k * TN);
                            __m256 b1 = _mm256_load_ps(pb_j + (k+1) * TN);
                            __m256 b2 = _mm256_load_ps(pb_j + (k+2) * TN);
                            __m256 b3 = _mm256_load_ps(pb_j + (k+3) * TN);
                            __m256 a0 = _mm256_set1_ps(a0_ptr[k]);
                            __m256 a1 = _mm256_set1_ps(a0_ptr[k+1]);
                            __m256 a2 = _mm256_set1_ps(a0_ptr[k+2]);
                            __m256 a3 = _mm256_set1_ps(a0_ptr[k+3]);

                            c0 = _mm256_fmadd_ps(a0, b0, c0);
                            c1 = _mm256_fmadd_ps(a1, b1, c1);
                            c2 = _mm256_fmadd_ps(a2, b2, c2);
                            c3 = _mm256_fmadd_ps(a3, b3, c3);
                        }

                        c2 = _mm256_add_ps(c2, c3);

                        for (; k < k_size; ++k) {
                            __m256 a0 = _mm256_set1_ps(a0_ptr[k]);
                            const float *pb0 = packed_B + k * TN + j_tile;

                            c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pb0), c0);
                        }
                        
                        c0 = _mm256_add_ps(c0, c1);
                        
                        _mm256_storeu_ps(c0_ptr + j_tile, _mm256_add_ps(c0, c2));
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc = c0_ptr[j_tile];
                        for (size_t k = 0; k < k_size; ++k) {
                            acc += a0_ptr[k] * packed_B[k * TN + j_tile];
                        }
                        c0_ptr[j_tile] = acc;
                    }
                }
            }
        }
    }
}

template <size_t TK>
void lg_M_N_K_even_tn32(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
) {
    #ifdef CPU_TIME_FP16_EVAL
        CPUTimer timer("linear");
        printf("Shape of matmul FP16 tn32: M=%zu, N=%zu, K=%zu, TK=%zu\n", M, N, K, TK);
    #endif

    constexpr size_t TN = 32;

    alignas(32) float local_bias[N];
    
    if (mat_bias) {
        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 24));

            __m256 v32_0 = _mm256_cvtph_ps(v16_0);
            __m256 v32_1 = _mm256_cvtph_ps(v16_1);
            __m256 v32_2 = _mm256_cvtph_ps(v16_2);
            __m256 v32_3 = _mm256_cvtph_ps(v16_3);

            _mm256_store_ps(local_bias + jb,      v32_0);
            _mm256_store_ps(local_bias + jb + 8,  v32_1);
            _mm256_store_ps(local_bias + jb + 16, v32_2);
            _mm256_store_ps(local_bias + jb + 24, v32_3);
        }
    } else {
        memset(local_bias, 0, N * sizeof(float));
    }
    
    #pragma omp parallel
    {
        alignas(32) float packed_B[TK * TN];
        
        for (size_t kk = 0; kk < K; kk += TK) {

            #pragma omp for schedule(static)
            for (size_t jj = 0; jj < N; jj += TN) {

                for (size_t k = 0; k < TK; ++k) {
                    const half_cpu* b_ptr = &mat_B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;
                    // Load 4 chunks of 8 half-precision floats (128-bit each)
                    __m128i v0 = _mm_loadu_si128((const __m128i*)(b_ptr + 0));
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(b_ptr + 8));
                    __m128i v2 = _mm_loadu_si128((const __m128i*)(b_ptr + 16));
                    __m128i v3 = _mm_loadu_si128((const __m128i*)(b_ptr + 24));

                    // Convert to float (F16C instruction) and store (256-bit each)
                    _mm256_store_ps(packed_B_ptr + 0,  _mm256_cvtph_ps(v0));
                    _mm256_store_ps(packed_B_ptr + 8,  _mm256_cvtph_ps(v1));
                    _mm256_store_ps(packed_B_ptr + 16, _mm256_cvtph_ps(v2));
                    _mm256_store_ps(packed_B_ptr + 24, _mm256_cvtph_ps(v3));
                }

                const float *pb0_j = packed_B;
                const float *pb1_j = packed_B + 8;
                const float *pb2_j = packed_B + 16;
                const float *pb3_j = packed_B + 24;

                size_t ii = 0;
                                
                for (; ii + 2 <= M; ii += 2) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    const float *a1_ptr = mat_A + (ii + 1) * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    float *c1_ptr = mat_C + (ii + 1) * N + jj;
                    
                    __m256 c00, c10, c20, c30, c01, c11, c21, c31;
                    if (kk == 0) {
                        c00 = _mm256_load_ps(local_bias + jj);
                        c10 = _mm256_load_ps(local_bias + jj + 8);
                        c20 = _mm256_load_ps(local_bias + jj + 16);
                        c30 = _mm256_load_ps(local_bias + jj + 24);
                        c01 = c00; c11 = c10; c21 = c20; c31 = c30;
                    } else {
                        c00 = _mm256_loadu_ps(c0_ptr);
                        c01 = _mm256_loadu_ps(c1_ptr);
                        c10 = _mm256_loadu_ps(c0_ptr + 8);
                        c11 = _mm256_loadu_ps(c1_ptr + 8);
                        c20 = _mm256_loadu_ps(c0_ptr + 16);
                        c21 = _mm256_loadu_ps(c1_ptr + 16);
                        c30 = _mm256_loadu_ps(c0_ptr + 24);
                        c31 = _mm256_loadu_ps(c1_ptr + 24);
                    }

                    size_t k = 0;
                    for (; k + 2 <= TK; k += 2) {
                        __m256 a00 = _mm256_broadcast_ss(a0_ptr + k);
                        __m256 a10 = _mm256_broadcast_ss(a1_ptr + k);
                        __m256 a01 = _mm256_broadcast_ss(a0_ptr + k + 1);
                        __m256 a11 = _mm256_broadcast_ss(a1_ptr + k + 1);

                        c00 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb0_j + k * TN), c00);
                        c01 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb0_j + k * TN), c01);
                        c10 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb1_j + k * TN), c10);
                        c11 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb1_j + k * TN), c11);
                        c20 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb2_j + k * TN), c20);
                        c21 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb2_j + k * TN), c21);
                        c30 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb3_j + k * TN), c30);
                        c31 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb3_j + k * TN), c31);

                        c00 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb0_j + (k + 1) * TN), c00);
                        c01 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb0_j + (k + 1) * TN), c01);
                        c10 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb1_j + (k + 1) * TN), c10);
                        c11 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb1_j + (k + 1) * TN), c11);
                        c20 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb2_j + (k + 1) * TN), c20);
                        c21 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb2_j + (k + 1) * TN), c21);
                        c30 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb3_j + (k + 1) * TN), c30);
                        c31 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb3_j + (k + 1) * TN), c31);
                    }
                    
                    _mm256_storeu_ps(c0_ptr, c00);
                    _mm256_storeu_ps(c0_ptr + 8, c10);
                    _mm256_storeu_ps(c0_ptr + 16, c20);
                    _mm256_storeu_ps(c0_ptr + 24, c30);
                    _mm256_storeu_ps(c1_ptr, c01);
                    _mm256_storeu_ps(c1_ptr + 8, c11);
                    _mm256_storeu_ps(c1_ptr + 16, c21);
                    _mm256_storeu_ps(c1_ptr + 24, c31);
                }

                if (M > ii) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    
                    __m256 c00, c10, c20, c30, c01, c11, c21, c31;
                    if (kk == 0) {
                        c00 = _mm256_load_ps(local_bias + jj);
                        c10 = _mm256_load_ps(local_bias + jj + 8);
                        c20 = _mm256_load_ps(local_bias + jj + 16);
                        c30 = _mm256_load_ps(local_bias + jj + 24);
                    } else {
                        c00 = _mm256_loadu_ps(c0_ptr);
                        c10 = _mm256_loadu_ps(c0_ptr + 8);
                        c20 = _mm256_loadu_ps(c0_ptr + 16);
                        c30 = _mm256_loadu_ps(c0_ptr + 24);
                    }

                    c01 = c11 = c21 = c31 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 2 <= TK; k += 2) {
                        __m256 a0 = _mm256_broadcast_ss(a0_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0_j + k * TN), c00);
                        c10 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb1_j + k * TN), c10);
                        c20 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb2_j + k * TN), c20);
                        c30 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb3_j + k * TN), c30);

                        __m256 a1 = _mm256_broadcast_ss(a0_ptr + k + 1);

                        c01 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb0_j + (k + 1) * TN), c01);
                        c11 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1_j + (k + 1) * TN), c11);
                        c21 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb2_j + (k + 1) * TN), c21);
                        c31 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb3_j + (k + 1) * TN), c31);
                    }
                    
                    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(c00, c01));
                    _mm256_storeu_ps(c0_ptr + 8, _mm256_add_ps(c10, c11));
                    _mm256_storeu_ps(c0_ptr + 16, _mm256_add_ps(c20, c21));
                    _mm256_storeu_ps(c0_ptr + 24, _mm256_add_ps(c30, c31));
                }
            }
        }
    }
}

template <size_t TK>
void lg_M_N_K_even_tn16_tm4(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
) {
    #ifdef CPU_TIME_FP16_EVAL
        CPUTimer timer("linear");
        printf("Shape of matmul FP16 tn16: M=%zu, N=%zu, K=%zu, TK=%zu\n", M, N, K, TK);
    #endif

    constexpr size_t TN = 16;

    alignas(32) float local_bias[N];
    
    if (mat_bias) {
        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 24));

            __m256 v32_0 = _mm256_cvtph_ps(v16_0);
            __m256 v32_1 = _mm256_cvtph_ps(v16_1);
            __m256 v32_2 = _mm256_cvtph_ps(v16_2);
            __m256 v32_3 = _mm256_cvtph_ps(v16_3);

            _mm256_store_ps(local_bias + jb,      v32_0);
            _mm256_store_ps(local_bias + jb + 8,  v32_1);
            _mm256_store_ps(local_bias + jb + 16, v32_2);
            _mm256_store_ps(local_bias + jb + 24, v32_3);
        }
    } else {
        memset(local_bias, 0, N * sizeof(float));
    }
    
    #pragma omp parallel
    {
        // thread-private packed_B
        alignas(32) float packed_B[TK * TN];
        
        for (size_t kk = 0; kk < K; kk += TK) {

            #pragma omp for schedule(static)
            for (size_t jj = 0; jj < N; jj += TN) {

                for (size_t k = 0; k < TK; ++k) {
                    size_t j = 0;
                    const half_cpu* b_ptr = &mat_B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;

                    // Load 2 chunks of 8 half-precision floats (128-bit each)
                    __m128i v0 = _mm_loadu_si128((const __m128i*)(b_ptr + 0));
                    __m128i v1 = _mm_loadu_si128((const __m128i*)(b_ptr + 8));

                    // Convert and Store (256-bit each)
                    _mm256_store_ps(packed_B_ptr + 0, _mm256_cvtph_ps(v0));
                    _mm256_store_ps(packed_B_ptr + 8, _mm256_cvtph_ps(v1));
                }

                const float *pb0_j = packed_B;
                const float *pb1_j = packed_B + 8;

                size_t ii = 0;
                for (; ii + 4 <= M; ii += 4) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    const float *a1_ptr = mat_A + (ii + 1) * K + kk;
                    const float *a2_ptr = mat_A + (ii + 2) * K + kk;
                    const float *a3_ptr = mat_A + (ii + 3) * K + kk;

                    float *c0_ptr = mat_C + ii * N + jj;
                    float *c1_ptr = mat_C + (ii + 1) * N + jj;
                    float *c2_ptr = mat_C + (ii + 2) * N + jj;
                    float *c3_ptr = mat_C + (ii + 3) * N + jj;
                    
                    __m256 c00, c10, c01, c11, c02, c12, c03, c13;
                    if (kk == 0) {
                        c00 = _mm256_load_ps(local_bias + jj);
                        c10 = _mm256_load_ps(local_bias + jj + 8);
                        c01 = c02 = c03 = c00;
                        c11 = c12 = c13 = c10;
                    } else {
                        c00 = _mm256_loadu_ps(c0_ptr);
                        c01 = _mm256_loadu_ps(c1_ptr);
                        c02 = _mm256_loadu_ps(c2_ptr);
                        c03 = _mm256_loadu_ps(c3_ptr);
                        c10 = _mm256_loadu_ps(c0_ptr + 8);
                        c11 = _mm256_loadu_ps(c1_ptr + 8);
                        c12 = _mm256_loadu_ps(c2_ptr + 8);
                        c13 = _mm256_loadu_ps(c3_ptr + 8);
                    }

                    size_t k = 0;
                    for (; k + 2 <= TK; k += 2) {
                        __m256 b00 = _mm256_load_ps(pb0_j + k * TN);
                        __m256 b10 = _mm256_load_ps(pb1_j + k * TN);

                        c00 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k), b00, c00);
                        c10 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k), b10, c10);

                        c01 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k), b00, c01);
                        c11 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k), b10, c11);

                        c02 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k), b00, c02);
                        c12 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k), b10, c12);

                        c03 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k), b00, c03);
                        c13 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k), b10, c13);

                        __m256 b01 = _mm256_load_ps(pb0_j + (k+1) * TN);
                        __m256 b11 = _mm256_load_ps(pb1_j + (k+1) * TN);

                        c00 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 1), b01, c00);
                        c10 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 1), b11, c10);

                        c01 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 1), b01, c01);
                        c11 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 1), b11, c11);

                        c02 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k + 1), b01, c02);
                        c12 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k + 1), b11, c12);

                        c03 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k + 1), b01, c03);
                        c13 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k + 1), b11, c13);
                    }
                    
                    _mm256_storeu_ps(c0_ptr, c00);
                    _mm256_storeu_ps(c1_ptr, c01);
                    _mm256_storeu_ps(c2_ptr, c02);
                    _mm256_storeu_ps(c3_ptr, c03);
                    _mm256_storeu_ps(c0_ptr + 8, c10);
                    _mm256_storeu_ps(c1_ptr + 8, c11);
                    _mm256_storeu_ps(c2_ptr + 8, c12);
                    _mm256_storeu_ps(c3_ptr + 8, c13);
                }

                if (M - ii >= 2) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    const float *a1_ptr = mat_A + (ii + 1) * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    float *c1_ptr = mat_C + (ii + 1) * N + jj;
                    
                    __m256 c00, c10, c01, c11, c00b, c10b, c01b, c11b;
                    if (kk == 0) {
                        c00 = _mm256_load_ps(local_bias + jj);
                        c10 = _mm256_load_ps(local_bias + jj + 8);
                        c01 = c00; c11 = c10;
                    } else {
                        c00 = _mm256_loadu_ps(c0_ptr);
                        c01 = _mm256_loadu_ps(c1_ptr);
                        c10 = _mm256_loadu_ps(c0_ptr + 8);
                        c11 = _mm256_loadu_ps(c1_ptr + 8);
                    }
                    c00b = c10b = c01b = c11b = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 2 <= TK; k += 2) {
                        __m256 a00 = _mm256_broadcast_ss(a0_ptr + k);
                        __m256 a10 = _mm256_broadcast_ss(a1_ptr + k);
                        __m256 a01 = _mm256_broadcast_ss(a0_ptr + k + 1);
                        __m256 a11 = _mm256_broadcast_ss(a1_ptr + k + 1);

                        c00 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb0_j + k * TN), c00);
                        c01 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb0_j + k * TN), c01);
                        c10 = _mm256_fmadd_ps(a00, _mm256_load_ps(pb1_j + k * TN), c10);
                        c11 = _mm256_fmadd_ps(a10, _mm256_load_ps(pb1_j + k * TN), c11);

                        c00 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb0_j + (k + 1) * TN), c00);
                        c01 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb0_j + (k + 1) * TN), c01);
                        c10 = _mm256_fmadd_ps(a01, _mm256_load_ps(pb1_j + (k + 1) * TN), c10);
                        c11 = _mm256_fmadd_ps(a11, _mm256_load_ps(pb1_j + (k + 1) * TN), c11);
                    }
                    
                    _mm256_storeu_ps(c0_ptr, c00);
                    _mm256_storeu_ps(c0_ptr + 8, c10);
                    _mm256_storeu_ps(c1_ptr, c01);
                    _mm256_storeu_ps(c1_ptr + 8, c11);

                    ii += 2;
                }

                if (M > ii) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    
                    __m256 c00, c10, c01, c11, c02, c12, c03, c13;
                    if (kk == 0) {
                        c00 = _mm256_load_ps(local_bias + jj);
                        c10 = _mm256_load_ps(local_bias + jj + 8);
                    } else {
                        c00 = _mm256_loadu_ps(c0_ptr);
                        c10 = _mm256_loadu_ps(c0_ptr + 8);
                    }

                    c01 = c11 = c02 = c12 = c03 = c13 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 4 <= TK; k += 4) {
                        __m256 a0 = _mm256_broadcast_ss(a0_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0_j + k * TN), c00);
                        c10 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb1_j + k * TN), c10);

                        __m256 a1 = _mm256_broadcast_ss(a0_ptr + k + 1);

                        c01 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb0_j + (k + 1) * TN), c01);
                        c11 = _mm256_fmadd_ps(a1, _mm256_load_ps(pb1_j + (k + 1) * TN), c11);

                        __m256 a2 = _mm256_broadcast_ss(a0_ptr + k + 2);

                        c02 = _mm256_fmadd_ps(a2, _mm256_load_ps(pb0_j + (k + 2) * TN), c02);
                        c12 = _mm256_fmadd_ps(a2, _mm256_load_ps(pb1_j + (k + 2) * TN), c12);

                        __m256 a3 = _mm256_broadcast_ss(a0_ptr + k + 3);

                        c03 = _mm256_fmadd_ps(a3, _mm256_load_ps(pb0_j + (k + 3) * TN), c03);
                        c13 = _mm256_fmadd_ps(a3, _mm256_load_ps(pb1_j + (k + 3) * TN), c13);
                    }
                    
                    c00 = _mm256_add_ps(c00, c02);
                    c10 = _mm256_add_ps(c10, c12);
                    c01 = _mm256_add_ps(c01, c03);
                    c11 = _mm256_add_ps(c11, c13);

                    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(c00, c01));
                    _mm256_storeu_ps(c0_ptr + 8, _mm256_add_ps(c10, c11));
                }
            }
        }
    }
}

template <size_t TK>
void lg_M_N_K_even_tn8_tm4(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
) {
    #ifdef CPU_TIME_FP16_EVAL
        CPUTimer timer("linear");
        printf("Shape of matmul FP16 tn8: M=%zu, N=%zu, K=%zu, TK=%zu\n", M, N, K, TK);
    #endif

    constexpr size_t TN = 8;

    alignas(32) float local_bias[N];
    
    if (mat_bias) {
        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_bias + jb + 24));

            __m256 v32_0 = _mm256_cvtph_ps(v16_0);
            __m256 v32_1 = _mm256_cvtph_ps(v16_1);
            __m256 v32_2 = _mm256_cvtph_ps(v16_2);
            __m256 v32_3 = _mm256_cvtph_ps(v16_3);

            _mm256_store_ps(local_bias + jb,      v32_0);
            _mm256_store_ps(local_bias + jb + 8,  v32_1);
            _mm256_store_ps(local_bias + jb + 16, v32_2);
            _mm256_store_ps(local_bias + jb + 24, v32_3);
        }
    } else {
        memset(local_bias, 0, N * sizeof(float));
    }
    
    #pragma omp parallel
    {
        // thread-private packed_B
        alignas(32) float packed_B[TK * TN];
        
        for (size_t kk = 0; kk < K; kk += TK) {

            #pragma omp for schedule(static)
            for (size_t jj = 0; jj < N; jj += TN) {

                for (size_t k = 0; k < TK; ++k) {
                    size_t j = 0;
                    const half_cpu* b_ptr = &mat_B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;

                    // Load 2 chunks of 8 half-precision floats (128-bit each)
                    __m128i v0 = _mm_loadu_si128((const __m128i*)(b_ptr + 0));

                    // Convert and Store (256-bit each)
                    _mm256_store_ps(packed_B_ptr + 0, _mm256_cvtph_ps(v0));
                }

                size_t ii = 0;
                for (; ii + 4 <= M; ii += 4) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    const float *a1_ptr = mat_A + (ii + 1) * K + kk;
                    const float *a2_ptr = mat_A + (ii + 2) * K + kk;
                    const float *a3_ptr = mat_A + (ii + 3) * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    float *c1_ptr = mat_C + (ii + 1) * N + jj;
                    float *c2_ptr = mat_C + (ii + 2) * N + jj;
                    float *c3_ptr = mat_C + (ii + 3) * N + jj;
                    
                    __m256 c0a, c0b, c2a, c2b;
                    __m256 c1a, c1b, c3a, c3b;
                    if (kk == 0) {
                        c0a = _mm256_load_ps(local_bias + jj);
                        c1a = c2a = c3a = c0a;
                    } else {
                        c0a = _mm256_loadu_ps(c0_ptr);
                        c1a = _mm256_loadu_ps(c1_ptr);
                        c2a = _mm256_loadu_ps(c2_ptr);
                        c3a = _mm256_loadu_ps(c3_ptr);
                    }
                    c0b = c1b = c2b = c3b = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 2 <= TK; k += 2) {
                        __m256 b0 = _mm256_load_ps(packed_B + k * TN);
                        __m256 b1 = _mm256_load_ps(packed_B + (k+1) * TN);

                        c0a = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k), b0, c0a);
                        c1a = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k), b0, c1a);
                        c2a = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k), b0, c2a);
                        c3a = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k), b0, c3a);

                        c0b = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 1), b1, c0b);
                        c1b = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 1), b1, c1b);
                        c2b = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr + k + 1), b1, c2b);
                        c3b = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr + k + 1), b1, c3b);
                    }

                    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(c0a, c0b));
                    _mm256_storeu_ps(c1_ptr, _mm256_add_ps(c1a, c1b));
                    _mm256_storeu_ps(c2_ptr, _mm256_add_ps(c2a, c2b));
                    _mm256_storeu_ps(c3_ptr, _mm256_add_ps(c3a, c3b));
                }
            
                if (M - ii >= 2) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    const float *a1_ptr = mat_A + (ii + 1) * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    float *c1_ptr = mat_C + (ii + 1) * N + jj;
                    
                    __m256 c0a, c0b, c0c, c0d;
                    __m256 c1a, c1b, c1c, c1d;
                    if (kk == 0) {
                        c0a = _mm256_load_ps(local_bias + jj);
                        c1a = c0a;
                    } else {
                        c0a = _mm256_loadu_ps(c0_ptr);
                        c1a = _mm256_loadu_ps(c1_ptr);
                    }
                    c0b = c1b = c0c = c1c = c0d = c1d = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 4 <= TK; k += 4) {
                        __m256 b0 = _mm256_load_ps(packed_B + k * TN);
                        __m256 b1 = _mm256_load_ps(packed_B + (k+1) * TN);

                        c0a = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k), b0, c0a);
                        c1a = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k), b0, c1a);
                        c0b = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 1), b1, c0b);
                        c1b = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 1), b1, c1b);

                        __m256 b2 = _mm256_load_ps(packed_B + (k+2) * TN);
                        __m256 b3 = _mm256_load_ps(packed_B + (k+3) * TN);

                        c0c = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 2), b2, c0c);
                        c1c = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 2), b2, c1c);
                        c0d = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 3), b3, c0d);
                        c1d = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr + k + 3), b3, c1d);
                    }

                    c0a = _mm256_add_ps(c0a, c0c);
                    c1a = _mm256_add_ps(c1a, c1c);
                    c0b = _mm256_add_ps(c0b, c0d);
                    c1b = _mm256_add_ps(c1b, c1d);
                    
                    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(c0a, c0b));
                    _mm256_storeu_ps(c1_ptr, _mm256_add_ps(c1a, c1b));

                    ii += 2;
                }
            
                if (M > ii) {
                    const float *a0_ptr = mat_A + ii * K + kk;
                    float *c0_ptr = mat_C + ii * N + jj;
                    
                    __m256 c0, c1, c2, c3;
                    if (kk == 0) {
                        c0 = _mm256_load_ps(local_bias + jj);
                    } else {
                        c0 = _mm256_loadu_ps(c0_ptr);
                    }

                    c1 = c2 = c3 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k + 8 <= TK; k += 8) {
                        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k), _mm256_load_ps(packed_B + k * TN), c0);
                        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 1), _mm256_load_ps(packed_B + (k+1) * TN), c1);
                        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 2), _mm256_load_ps(packed_B + (k+2) * TN), c2);
                        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 3), _mm256_load_ps(packed_B + (k+3) * TN), c3);
                        
                        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 4), _mm256_load_ps(packed_B + (k+4) * TN), c0);
                        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 5), _mm256_load_ps(packed_B + (k+5) * TN), c1);
                        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 6), _mm256_load_ps(packed_B + (k+6) * TN), c2);
                        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr + k + 7), _mm256_load_ps(packed_B + (k+7) * TN), c3);
                    }

                    c0 = _mm256_add_ps(c0, c2);
                    c1 = _mm256_add_ps(c1, c3);

                    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(c0, c1));
                }
            }
        }
    }
}

void fp16_avx2_kernel_untranspose(
    const float *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K
) {
    if (M > 64) {
        if (N % 32 == 0 && K % 64 == 0) {
            // if cache size, change tn16_tm4 TK = 256, tn8_tm4 TK = 
            lg_M_N_K_even_tn16_tm4<512>(mat_A, mat_B, mat_bias, mat_C, M, N, K);
        } else {
            lg_M_N_K(mat_A, mat_B, mat_bias, mat_C, M, N, K);
        }
        return;
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
#endif

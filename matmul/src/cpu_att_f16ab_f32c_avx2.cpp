#include "../include/cpu_wrapper.hpp"

#if defined(__AVX2__) && defined(__FMA__)
void gemm_att_f16ab_f32c_k64(
    const half_cpu *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N
) {
    constexpr size_t TN = 32;

    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        
        alignas(32) float packed_B[(TN << 6)];

        #pragma omp for schedule(static)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);
            size_t j_size = j_end - jj;

            // pack B
            const half_cpu *b_ptr = mat_B + (jj << 6);
            for (size_t j = 0; j < j_size; ++j) {
                const size_t j6 = (j << 6);
                const half_cpu* b_row_ptr = b_ptr + j6;
                float* packed_row_ptr = packed_B + j6;

                // Load 8 chunks of 8 halves (128-bit each)
                __m128i h0 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 0));
                __m128i h1 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 8));
                __m128i h2 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 16));
                __m128i h3 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 24));
                __m128i h4 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 32));
                __m128i h5 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 40));
                __m128i h6 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 48));
                __m128i h7 = _mm_loadu_si128((const __m128i*)(b_row_ptr + 56));

                // Convert and Store
                _mm256_storeu_ps(packed_row_ptr + 0,  _mm256_cvtph_ps(h0));
                _mm256_storeu_ps(packed_row_ptr + 8,  _mm256_cvtph_ps(h1));
                _mm256_storeu_ps(packed_row_ptr + 16, _mm256_cvtph_ps(h2));
                _mm256_storeu_ps(packed_row_ptr + 24, _mm256_cvtph_ps(h3));
                _mm256_storeu_ps(packed_row_ptr + 32, _mm256_cvtph_ps(h4));
                _mm256_storeu_ps(packed_row_ptr + 40, _mm256_cvtph_ps(h5));
                _mm256_storeu_ps(packed_row_ptr + 48, _mm256_cvtph_ps(h6));
                _mm256_storeu_ps(packed_row_ptr + 56, _mm256_cvtph_ps(h7));
            }

            for (size_t ii = 0; ii < M; ++ii) {
                float *c0_ptr = mat_C + ii * N + jj;
                const half_cpu *a0_ptr = mat_A + (ii << 6);
                
                __m256 a_v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 0)));;
                __m256 a_v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 8)));;
                __m256 a_v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 16)));;
                __m256 a_v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 24)));;
                __m256 a_v4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 32)));
                __m256 a_v5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 40)));
                __m256 a_v6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 48)));
                __m256 a_v7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a0_ptr + 56)));

                // perform matmul here
                for (size_t j = 0; j < j_size; ++j) {
                    // Use the already packed_B (which is Row-Major FP32)
                    const float* b_row_ptr = packed_B + (j << 6);

                    // Multiply A row by B row (Dot Product)
                    __m256 sum0 = _mm256_mul_ps(a_v0, _mm256_load_ps(b_row_ptr + 0));
                    sum0 = _mm256_fmadd_ps(a_v1, _mm256_load_ps(b_row_ptr + 8),  sum0);
                    sum0 = _mm256_fmadd_ps(a_v2, _mm256_load_ps(b_row_ptr + 16), sum0);
                    sum0 = _mm256_fmadd_ps(a_v3, _mm256_load_ps(b_row_ptr + 24), sum0);
                    sum0 = _mm256_fmadd_ps(a_v4, _mm256_load_ps(b_row_ptr + 32), sum0);
                    sum0 = _mm256_fmadd_ps(a_v5, _mm256_load_ps(b_row_ptr + 40), sum0);
                    sum0 = _mm256_fmadd_ps(a_v6, _mm256_load_ps(b_row_ptr + 48), sum0);
                    sum0 = _mm256_fmadd_ps(a_v7, _mm256_load_ps(b_row_ptr + 56), sum0);

                    // Reduce the 8-float vector to a single scalar
                    float dot_product = add_reduce_mm_256(sum0);

                    // Apply scale and store
                    c0_ptr[j] = dot_product * scale;
                }
            }
        }
    }
}

void gemm_att_f16ab_f32c_k64_pack_transposed(
    const half_cpu *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N
) {
    constexpr size_t TN = 32;

    __m256 scale_vec = _mm256_set1_ps(scale);

    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        
        alignas(32) float packed_B[(TN << 6)];
        alignas(32) float tmp[8];
        alignas(32) float packed_A[4 * 64];
        const float *pack_a1_ptr = packed_A + 64;
        const float *pack_a2_ptr = packed_A + 2 * 64;
        const float *pack_a3_ptr = packed_A + 3 * 64;

        #pragma omp for schedule(static)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);
            size_t j_size = j_end - jj;

            // pack B
            for (size_t j = 0; j < j_size; ++j) {
                const half_cpu* b_row_ptr = mat_B + ((jj + j) << 6);

                // load 64 FP16 values from B[j]
                for (size_t k = 0; k < 64; k += 8) {
                    __m128i h = _mm_loadu_si128((const __m128i*)(b_row_ptr + k));
                    __m256 f = _mm256_cvtph_ps(h);

                    // store transposed: packed_B[k][j]
                    float* dst = packed_B + k * TN + j;

                    // scatter 8 floats vertically
                    // cannot store directly as vector because we need stride TN
                    _mm256_store_ps(tmp, f);

                    for (int x = 0; x < 8; ++x)
                        dst[x * TN] = tmp[x];
                }
            }

            size_t ii = 0;
            for (; ii + 4 <= M; ii += 4) {
                float *c0_ptr = mat_C + ii * N + jj;
                float *c1_ptr = mat_C + (ii + 1) * N + jj;
                float *c2_ptr = mat_C + (ii + 2) * N + jj;
                float *c3_ptr = mat_C + (ii + 3) * N + jj;

                // pack A
                {
                    // Ensure packed_A is at least 4 * TK elements and 32-byte aligned
                    size_t k = 0;
                    for (; k + 8 <= 64; k += 8) {
                        // 1. Concurrent Loads (Matrix A is FP16)
                        __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 0) << 6) + k));
                        __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 1) << 6) + k));
                        __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 2) << 6) + k));
                        __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 3) << 6) + k));

                        // 2. Interleaved Conversions (FP16 -> FP32)
                        __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                        __m256 v32_1 = _mm256_cvtph_ps(v16_1);
                        __m256 v32_2 = _mm256_cvtph_ps(v16_2);
                        __m256 v32_3 = _mm256_cvtph_ps(v16_3);

                        // 3. Aligned Stores to packed_A
                        _mm256_store_ps(packed_A + k, v32_0);
                        _mm256_store_ps(packed_A + 64 + k, v32_1);
                        _mm256_store_ps(packed_A + 2 * 64 + k, v32_2);
                        _mm256_store_ps(packed_A + 3 * 64 + k, v32_3);
                    }
                }

                size_t j_tile = 0;
                for (; j_tile + 16 <= j_size; j_tile += 16) {
                    const float *pb_j = packed_B + j_tile;
                    
                    __m256 c00, c01, c02, c03, c10, c11, c12, c13;
                    
                    c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k < 64; ++k) {
                        const float *pb0 = pb_j + k * TN;
                        __m256 b0 = _mm256_load_ps(pb0);
                        __m256 b1 = _mm256_load_ps(pb0 + 8);

                        __m256 a0 = _mm256_broadcast_ss(packed_A + k);
                        __m256 a1 = _mm256_broadcast_ss(pack_a1_ptr + k);
                        __m256 a2 = _mm256_broadcast_ss(pack_a2_ptr + k);
                        __m256 a3 = _mm256_broadcast_ss(pack_a3_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c10 = _mm256_fmadd_ps(a0, b1, c10);

                        c01 = _mm256_fmadd_ps(a1, b0, c01);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);

                        c02 = _mm256_fmadd_ps(a2, b0, c02);
                        c12 = _mm256_fmadd_ps(a2, b1, c12);

                        c03 = _mm256_fmadd_ps(a3, b0, c03);
                        c13 = _mm256_fmadd_ps(a3, b1, c13);
                    }

                    c00 = _mm256_mul_ps(c00, scale_vec);
                    c01 = _mm256_mul_ps(c01, scale_vec);
                    c02 = _mm256_mul_ps(c02, scale_vec);
                    c03 = _mm256_mul_ps(c03, scale_vec);
                    
                    _mm256_storeu_ps(c0_ptr + j_tile, c00);
                    _mm256_storeu_ps(c1_ptr + j_tile, c01);
                    _mm256_storeu_ps(c2_ptr + j_tile, c02);
                    _mm256_storeu_ps(c3_ptr + j_tile, c03);

                    c10 = _mm256_mul_ps(c10, scale_vec);
                    c11 = _mm256_mul_ps(c11, scale_vec);
                    c12 = _mm256_mul_ps(c12, scale_vec);
                    c13 = _mm256_mul_ps(c13, scale_vec);
                    
                    _mm256_storeu_ps(c0_ptr + j_tile + 8, c10);
                    _mm256_storeu_ps(c1_ptr + j_tile + 8, c11);
                    _mm256_storeu_ps(c2_ptr + j_tile + 8, c12);
                    _mm256_storeu_ps(c3_ptr + j_tile + 8, c13);
                }

                for (; j_tile < j_size; ++j_tile) {
                    float acc0, acc1, acc2, acc3;
                    acc0 = acc1 = acc2 = acc3 = 0.0f;

                    for (size_t k = 0; k < 64; ++k) {
                        float b_val = packed_B[k * TN + j_tile];
                        acc0 += packed_A[k] * b_val;
                        acc1 += pack_a1_ptr[k] * b_val;
                        acc2 += pack_a2_ptr[k] * b_val;
                        acc3 += pack_a3_ptr[k] * b_val;
                    }

                    acc0 *= scale;
                    acc1 *= scale;
                    acc2 *= scale;
                    acc3 *= scale;

                    c0_ptr[j_tile] = acc0;
                    c1_ptr[j_tile] = acc1;
                    c2_ptr[j_tile] = acc2;
                    c3_ptr[j_tile] = acc3;
                }
            }

            if (ii + 2 <= M) {
                float *c0_ptr = mat_C + ii * N + jj;
                float *c1_ptr = mat_C + (ii + 1) * N + jj;

                // pack A
                for (int i = 0; i < 2; ++i) {
                    size_t k = 0;
                    const half_cpu *a_ptr = mat_A + ((ii + i) << 6);
                    float *packed_A_ptr = packed_A + (i << 6);
                    for (; k + 8 <= 64; k += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(a_ptr + k));
                        __m256 v32 = _mm256_cvtph_ps(v16);
                        _mm256_store_ps(packed_A_ptr + k, v32);
                    }
                }

                size_t j_tile = 0;
                for (; j_tile + 8 <= j_size; j_tile += 8) {
                    const float *pb_j = packed_B + j_tile;
                    
                    __m256 c00, c01; // , c10, c20, c30, c01, c11, c21, c31;

                    c00 = c01 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k < 64; ++k) {
                        const float *pb0 = pb_j + k * TN;
                        __m256 a0 = _mm256_broadcast_ss(packed_A + k);
                        __m256 a1 = _mm256_broadcast_ss(pack_a1_ptr + k);
                        
                        __m256 b0 = _mm256_load_ps(pb0);

                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a1, b0, c01);
                    }
                    
                    c00 = _mm256_mul_ps(c00, scale_vec);
                    c01 = _mm256_mul_ps(c01, scale_vec);
                    
                    _mm256_storeu_ps(c0_ptr + j_tile, c00);
                    _mm256_storeu_ps(c1_ptr + j_tile, c01);
                }

                for (; j_tile < j_size; ++j_tile) {
                    float acc0, acc1;
                    acc0 = acc1 = 0.0f;

                    for (size_t k = 0; k < 64; ++k) {
                        float b_val = packed_B[k * TN + j_tile];
                        acc0 += packed_A[k] * b_val;
                        acc1 += pack_a1_ptr[k] * b_val;
                    }

                    acc0 *= scale;
                    acc1 *= scale;

                    c0_ptr[j_tile] = acc0;
                    c1_ptr[j_tile] = acc1;
                }

                ii += 2;
            }

            if (ii < M) {
                float *c0_ptr = mat_C + ii * N + jj;

                // pack A
                {
                    size_t k = 0;
                    const half_cpu *a_ptr = mat_A + (ii << 6);
                    for (; k + 8 <= 64; k += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(a_ptr + k));
                        __m256 v32 = _mm256_cvtph_ps(v16);
                        _mm256_store_ps(packed_A + k, v32);
                    }
                }

                size_t j_tile = 0;
                for (; j_tile + 8 <= j_size; j_tile += 8) {
                    const float *pb_j = packed_B + j_tile;
                    
                    __m256 c0 = _mm256_setzero_ps(); //, c1, c2, c3;

                    size_t k = 0;
                    for (; k < 64; ++k) {
                        const float *pb0 = pb_j + k * TN;
                        __m256 a0 = _mm256_broadcast_ss(packed_A + k);

                        c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                    }
                    
                    c0 = _mm256_mul_ps(c0, scale_vec);
                    
                    _mm256_storeu_ps(c0_ptr + j_tile, c0);
                }

                for (; j_tile < j_size; ++j_tile) {
                    float acc_h = 0.0f;

                    for (size_t k = 0; k < 64; ++k) {
                        // FP16 mul → FP16 add (scalar, rounded each step)
                        acc_h += packed_A[k] * packed_B[k * TN + j_tile];
                    }

                    acc_h *= scale;
                    c0_ptr[j_tile] = acc_h;
                }
            }
        }
    }
}

void gemm_att_f16ab_f32c_k64_pack_transposed_tm4(
    const half_cpu *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N
) {
    constexpr size_t TN = 32;

    __m256 scale_vec = _mm256_set1_ps(scale);

    #pragma omp parallel
    {
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        
        alignas(32) float packed_B[(TN << 6)];
        alignas(32) float tmp[8];
        alignas(32) float packed_A[4 * 64];
        const float *pack_a1_ptr = packed_A + 64;
        const float *pack_a2_ptr = packed_A + 2 * 64;
        const float *pack_a3_ptr = packed_A + 3 * 64;

        #pragma omp for schedule(static)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);
            size_t j_size = j_end - jj;

            // pack B
            for (size_t j = 0; j < j_size; ++j) {
                const half_cpu* b_row_ptr = mat_B + ((jj + j) << 6);

                // load 64 FP16 values from B[j]
                for (size_t k = 0; k < 64; k += 8) {
                    __m128i h = _mm_loadu_si128((const __m128i*)(b_row_ptr + k));
                    __m256 f = _mm256_cvtph_ps(h);

                    // store transposed: packed_B[k][j]
                    float* dst = packed_B + k * TN + j;

                    // scatter 8 floats vertically
                    // cannot store directly as vector because we need stride TN
                    _mm256_store_ps(tmp, f);

                    for (int x = 0; x < 8; ++x)
                        dst[x * TN] = tmp[x];
                }
            }

            for (size_t ii = 0; ii + 4 <= M; ii += 4) {
                float *c0_ptr = mat_C + ii * N + jj;
                float *c1_ptr = mat_C + (ii + 1) * N + jj;
                float *c2_ptr = mat_C + (ii + 2) * N + jj;
                float *c3_ptr = mat_C + (ii + 3) * N + jj;

                // pack A
                {
                    // Ensure packed_A is at least 4 * TK elements and 32-byte aligned
                    size_t k = 0;
                    for (; k + 8 <= 64; k += 8) {
                        // 1. Concurrent Loads (Matrix A is FP16)
                        __m128i v16_0 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 0) << 6) + k));
                        __m128i v16_1 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 1) << 6) + k));
                        __m128i v16_2 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 2) << 6) + k));
                        __m128i v16_3 = _mm_loadu_si128((const __m128i*)(mat_A + ((ii + 3) << 6) + k));

                        // 2. Interleaved Conversions (FP16 -> FP32)
                        __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                        __m256 v32_1 = _mm256_cvtph_ps(v16_1);
                        __m256 v32_2 = _mm256_cvtph_ps(v16_2);
                        __m256 v32_3 = _mm256_cvtph_ps(v16_3);

                        // 3. Aligned Stores to packed_A
                        _mm256_store_ps(packed_A + k, v32_0);
                        _mm256_store_ps(packed_A + 64 + k, v32_1);
                        _mm256_store_ps(packed_A + 2 * 64 + k, v32_2);
                        _mm256_store_ps(packed_A + 3 * 64 + k, v32_3);
                    }
                }

                size_t j_tile = 0;
                for (; j_tile + 8 <= j_size; j_tile += 8) {
                    const float *pb_j = packed_B + j_tile;
                    
                    __m256 c00, c01, c02, c03;
                    
                    c00 = c01 = c02 = c03 = _mm256_setzero_ps();

                    size_t k = 0;
                    for (; k < 64; ++k) {
                        const float *pb0 = pb_j + k * TN;
                        __m256 b0 = _mm256_load_ps(pb0);

                        __m256 a0 = _mm256_broadcast_ss(packed_A + k);
                        __m256 a1 = _mm256_broadcast_ss(pack_a1_ptr + k);
                        __m256 a2 = _mm256_broadcast_ss(pack_a2_ptr + k);
                        __m256 a3 = _mm256_broadcast_ss(pack_a3_ptr + k);

                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a1, b0, c01);
                        c02 = _mm256_fmadd_ps(a2, b0, c02);
                        c03 = _mm256_fmadd_ps(a3, b0, c03);
                    }

                    c00 = _mm256_mul_ps(c00, scale_vec);
                    c01 = _mm256_mul_ps(c01, scale_vec);
                    c02 = _mm256_mul_ps(c02, scale_vec);
                    c03 = _mm256_mul_ps(c03, scale_vec);
                    
                    _mm256_storeu_ps(c0_ptr + j_tile, c00);
                    _mm256_storeu_ps(c1_ptr + j_tile, c01);
                    _mm256_storeu_ps(c2_ptr + j_tile, c02);
                    _mm256_storeu_ps(c3_ptr + j_tile, c03);
                }

                for (; j_tile < j_size; ++j_tile) {
                    float acc0, acc1, acc2, acc3;
                    acc0 = acc1 = acc2 = acc3 = 0.0f;

                    for (size_t k = 0; k < 64; ++k) {
                        float b_val = packed_B[k * TN + j_tile];
                        acc0 += packed_A[k] * b_val;
                        acc1 += pack_a1_ptr[k] * b_val;
                        acc2 += pack_a2_ptr[k] * b_val;
                        acc3 += pack_a3_ptr[k] * b_val;
                    }

                    acc0 *= scale;
                    acc1 *= scale;
                    acc2 *= scale;
                    acc3 *= scale;

                    c0_ptr[j_tile] = acc0;
                    c1_ptr[j_tile] = acc1;
                    c2_ptr[j_tile] = acc2;
                    c3_ptr[j_tile] = acc3;
                }
            }
        }
    }
}

void att_f16ab_f32c_avx2_kernel(
    const half_cpu *mat_A, const half_cpu *mat_B, float *mat_C,
    const float scale, size_t M, size_t N, size_t K,
    bool mat_B_transpose
) {
    if (mat_B_transpose && K == 64) {
        if (M % 4 == 0) {
            gemm_att_f16ab_f32c_k64_pack_transposed_tm4(
                mat_A, mat_B, mat_C, scale, M, N
            );
        } else {
            gemm_att_f16ab_f32c_k64_pack_transposed(
                mat_A, mat_B, mat_C, scale, M, N
            );
        }
        return;
    }

    // printf("M=%zu, N=%zu, K=%zu\n", M, N, K);

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {

        const half_cpu* A_row = mat_A + m * K;
        float* C_row = mat_C + m * N;

        for (size_t n = 0; n < N; ++n) {

            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                float a = (float)A_row[k];

                float b = (!mat_B_transpose)
                    ? (float)mat_B[k * N + n]
                    : (float)mat_B[n * K + k];

                sum += a * b;
            }

            C_row[n] = sum * scale;
        }
    }
}
#endif

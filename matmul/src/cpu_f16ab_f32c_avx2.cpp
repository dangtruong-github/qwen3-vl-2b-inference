#include "../include/cpu_wrapper.hpp"

#if defined(__AVX2__) && defined(__FMA__)
template <size_t TN, size_t TK>
void linear_f16ab_fp32c_normal_tm_4(
    const half_cpu *A, const half_cpu *B, const half_cpu *bias, float *C,
    const size_t M, const size_t N, const size_t K
) {
    #ifdef CPU_TIME_FP16_EVAL
        CPUTimer timer("linear");
        printf("Shape of matmul FP16 tm4: M=%zu, N=%zu, K=%zu, TN=%zu, TK=%zu\n", M, N, K, TN, TK);
    #endif

    alignas(32) float local_bias[N];
    
    if (bias) {
        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(bias + jb + 24));

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
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

        alignas(32) float packed_B[TK * TN];
        alignas(32) float packed_A[4 * TK];
        const float *pack_a1_ptr = packed_A + TK;
        const float *pack_a2_ptr = packed_A + 2 * TK;
        const float *pack_a3_ptr = packed_A + 3 * TK;

        #pragma omp for schedule(dynamic, 1)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);
            size_t j_size = j_end - jj;

            for (size_t kk = 0; kk < K; kk += TK) {
                size_t k_end = std::min(kk + TK, K);
                size_t k_size = k_end - kk;

                // pack B
                
                for (size_t k = 0; k < k_size; ++k) {
                    const half_cpu* b_ptr = &B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;

                    size_t j = 0;
                    // Unroll by 16 (2 AVX2 registers)
                    for (; j + 16 <= j_size; j += 16) {
                        // Load two 128-bit chunks (8 halves each)
                        __m128i v16_0 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                        __m128i v16_1 = _mm_loadu_si128((const __m128i*)(b_ptr + j + 8));

                        // Convert to two 256-bit float registers
                        __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                        __m256 v32_1 = _mm256_cvtph_ps(v16_1);

                        // Store (using aligned stores if packed_B is 32-byte aligned)
                        _mm256_store_ps(packed_B_ptr + j, v32_0);
                        _mm256_store_ps(packed_B_ptr + j + 8, v32_1);
                    }

                    // Handle remainder of j_size (8-element SIMD step)
                    for (; j + 8 <= j_size; j += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                        _mm256_store_ps(packed_B_ptr + j, _mm256_cvtph_ps(v16));
                    }

                    // Scalar tail for the absolute remainder
                    for (; j < j_size; ++j) {
                        packed_B_ptr[j] = (float)b_ptr[j];
                    }
                }

                size_t ii = 0;
                for (; ii + 4 <= M; ii += 4) {
                    float *c0_ptr = C + ii * N + jj;
                    float *c1_ptr = C + (ii + 1) * N + jj;
                    float *c2_ptr = C + (ii + 2) * N + jj;
                    float *c3_ptr = C + (ii + 3) * N + jj;

                    // pack A
                    {
                        // Ensure packed_A is at least 4 * TK elements and 32-byte aligned
                        size_t k = 0;
                        for (; k + 8 <= k_size; k += 8) {
                            // 1. Concurrent Loads (Matrix A is FP16)
                            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(A + (ii + 0) * K + kk + k));
                            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(A + (ii + 1) * K + kk + k));
                            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(A + (ii + 2) * K + kk + k));
                            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(A + (ii + 3) * K + kk + k));

                            // 2. Interleaved Conversions (FP16 -> FP32)
                            __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                            __m256 v32_1 = _mm256_cvtph_ps(v16_1);
                            __m256 v32_2 = _mm256_cvtph_ps(v16_2);
                            __m256 v32_3 = _mm256_cvtph_ps(v16_3);

                            // 3. Aligned Stores to packed_A
                            _mm256_store_ps(packed_A + 0 * TK + k, v32_0);
                            _mm256_store_ps(packed_A + 1 * TK + k, v32_1);
                            _mm256_store_ps(packed_A + 2 * TK + k, v32_2);
                            _mm256_store_ps(packed_A + 3 * TK + k, v32_3);
                        }

                        // Tail loop for remaining k < 8
                        for (; k < k_size; ++k) {
                            packed_A[0 * TK + k] = (float)A[(ii + 0) * K + kk + k];
                            packed_A[1 * TK + k] = (float)A[(ii + 1) * K + kk + k];
                            packed_A[2 * TK + k] = (float)A[(ii + 2) * K + kk + k];
                            packed_A[3 * TK + k] = (float)A[(ii + 3) * K + kk + k];
                        }
                    }

                    size_t j_tile = 0;
                    for (; j_tile + 16 <= j_size; j_tile += 16) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c00, c01, c02, c03, c10, c11, c12, c13;
                        
                        if (kk == 0) {
                            c00 = _mm256_load_ps(local_bias + jj + j_tile);
                            c10 = _mm256_load_ps(local_bias + jj + j_tile + 8);

                            c01 = c02 = c03 = c00;
                            c11 = c12 = c13 = c10;
                        } else {
                            c00 = _mm256_loadu_ps(c0_ptr + j_tile);
                            c01 = _mm256_loadu_ps(c1_ptr + j_tile);
                            c02 = _mm256_loadu_ps(c2_ptr + j_tile);
                            c03 = _mm256_loadu_ps(c3_ptr + j_tile);
                            c10 = _mm256_loadu_ps(c0_ptr + j_tile + 8);
                            c11 = _mm256_loadu_ps(c1_ptr + j_tile + 8);
                            c12 = _mm256_loadu_ps(c2_ptr + j_tile + 8);
                            c13 = _mm256_loadu_ps(c3_ptr + j_tile + 8);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
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
                        
                        _mm256_storeu_ps(c0_ptr + j_tile, c00);
                        _mm256_storeu_ps(c1_ptr + j_tile, c01);
                        _mm256_storeu_ps(c2_ptr + j_tile, c02);
                        _mm256_storeu_ps(c3_ptr + j_tile, c03);
                        
                        _mm256_storeu_ps(c0_ptr + j_tile + 8, c10);
                        _mm256_storeu_ps(c1_ptr + j_tile + 8, c11);
                        _mm256_storeu_ps(c2_ptr + j_tile + 8, c12);
                        _mm256_storeu_ps(c3_ptr + j_tile + 8, c13);
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc0, acc1, acc2, acc3;

                        if (kk == 0) {
                            acc0 = acc1 = acc2 = acc3 = local_bias[jj + j_tile];
                        } else {
                            acc0 = c0_ptr[j_tile];
                            acc1 = c1_ptr[j_tile];
                            acc2 = c2_ptr[j_tile];
                            acc3 = c3_ptr[j_tile];
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            float b_val = packed_B[k * TN + j_tile];
                            acc0 += packed_A[k] * b_val;
                            acc1 += pack_a1_ptr[k] * b_val;
                            acc2 += pack_a2_ptr[k] * b_val;
                            acc3 += pack_a3_ptr[k] * b_val;
                        }

                        c0_ptr[j_tile] = acc0;
                        c1_ptr[j_tile] = acc1;
                        c2_ptr[j_tile] = acc2;
                        c3_ptr[j_tile] = acc3;
                    }
                }

                if (ii + 2 <= M) {
                    float *c0_ptr = C + ii * N + jj;
                    float *c1_ptr = C + (ii + 1) * N + jj;

                    // pack A
                    for (int i = 0; i < 2; ++i) {
                        size_t k = 0;
                        const half_cpu *a_ptr = A + (ii + i) * K + kk;
                        float *packed_A_ptr = packed_A + i * TK;
                        for (; k + 8 <= k_size; k += 8) {
                            __m128i v16 = _mm_loadu_si128((const __m128i*)(a_ptr + k));
                            __m256 v32 = _mm256_cvtph_ps(v16);
                            _mm256_store_ps(packed_A_ptr + k, v32);
                        }

                        for (; k < k_size; ++k) {
                            packed_A_ptr[k] = (float)a_ptr[k];
                        }
                    }

                    size_t j_tile = 0;
                    for (; j_tile + 8 <= j_size; j_tile += 8) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c00, c01; // , c10, c20, c30, c01, c11, c21, c31;
                        
                        if (kk == 0) {
                            c00 = c01 = _mm256_load_ps(local_bias + jj + j_tile);
                        } else {
                            c00 = _mm256_loadu_ps(c0_ptr + j_tile);
                            c01 = _mm256_loadu_ps(c1_ptr + j_tile);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
                            const float *pb0 = pb_j + k * TN;
                            __m256 a0 = _mm256_broadcast_ss(packed_A + k);
                            __m256 a1 = _mm256_broadcast_ss(pack_a1_ptr + k);
                            
                            __m256 b0 = _mm256_load_ps(pb0);

                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a1, b0, c01);
                        }
                        
                        _mm256_storeu_ps(c0_ptr + j_tile, c00);
                        _mm256_storeu_ps(c1_ptr + j_tile, c01);
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc0, acc1;

                        if (kk == 0) {
                            acc0 = acc1 = local_bias[jj + j_tile];
                        } else {
                            acc0 = c0_ptr[j_tile];
                            acc1 = c1_ptr[j_tile];
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            float b_val = packed_B[k * TN + j_tile];
                            acc0 += packed_A[k] * b_val;
                            acc1 += pack_a1_ptr[k] * b_val;
                        }

                        c0_ptr[j_tile] = acc0;
                        c1_ptr[j_tile] = acc1;
                    }

                    ii += 2;
                }

                if (ii < M) {
                    float *c0_ptr = C + ii * N + jj;

                    // pack A
                    {
                        size_t k = 0;
                        const half_cpu *a_ptr = A + ii * K + kk;
                        for (; k + 8 <= k_size; k += 8) {
                            __m128i v16 = _mm_loadu_si128((const __m128i*)(a_ptr + k));
                            __m256 v32 = _mm256_cvtph_ps(v16);
                            _mm256_store_ps(packed_A + k, v32);
                        }

                        for (; k < k_size; ++k) {
                            packed_A[k] = (float)a_ptr[k];
                        }
                    }

                    size_t j_tile = 0;
                    for (; j_tile + 8 <= j_size; j_tile += 8) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c0; //, c1, c2, c3;
                        
                        if (kk == 0) {
                            c0 = _mm256_load_ps(local_bias + jj + j_tile);
                        } else {
                            c0 = _mm256_loadu_ps(c0_ptr + j_tile);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
                            const float *pb0 = pb_j + k * TN;
                            __m256 a0 = _mm256_broadcast_ss(packed_A + k);

                            c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                        }
                        
                        _mm256_storeu_ps(c0_ptr + j_tile, c0);
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc_h;

                        if (kk == 0) {
                            acc_h = bias ? bias[jj + j_tile] : 0.0f;
                        } else {
                            acc_h = c0_ptr[j_tile];
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            // FP16 mul → FP16 add (scalar, rounded each step)
                            acc_h += packed_A[k] * packed_B[k * TN + j_tile];
                        }

                        c0_ptr[j_tile] = acc_h;
                    }
                }
            }
        }
    }
}

template <size_t TN, size_t TK>
void linear_f16ab_fp32c_normal_tm_8(
    const half_cpu *A, const half_cpu *B, const half_cpu *bias, float *C,
    const size_t M, const size_t N, const size_t K
) {
    #ifdef CPU_TIME_FP16_EVAL
        CPUTimer timer("linear");
        printf("Shape of matmul FP16 tm8: M=%zu, N=%zu, K=%zu, TN=%zu, TK=%zu\n", M, N, K, TN, TK);
    #endif

    alignas(32) float local_bias[N];
    
    if (bias) {
        size_t jb = 0;
        for (; jb + 31 < N; jb += 32) {
            __m128i v16_0 = _mm_loadu_si128((const __m128i*)(bias + jb));
            __m128i v16_1 = _mm_loadu_si128((const __m128i*)(bias + jb + 8));
            __m128i v16_2 = _mm_loadu_si128((const __m128i*)(bias + jb + 16));
            __m128i v16_3 = _mm_loadu_si128((const __m128i*)(bias + jb + 24));

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
        int cpu_id = omp_get_thread_num(); 
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset); 

        pthread_t current_thread = pthread_self();
        pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

        alignas(32) float packed_B[TK * TN];
        alignas(32) float packed_A[8 * TK];

        #pragma omp for schedule(dynamic, 1)
        for (size_t jj = 0; jj < N; jj += TN) {
            size_t j_end = std::min(jj + TN, N);
            size_t j_size = j_end - jj;

            for (size_t kk = 0; kk < K; kk += TK) {
                size_t k_end = std::min(kk + TK, K);
                size_t k_size = k_end - kk;

                // pack B
                for (size_t k = 0; k < k_size; ++k) {
                    const half_cpu* b_ptr = &B[(k + kk) * N + jj];
                    float *packed_B_ptr = packed_B + k * TN;

                    size_t j = 0;
                    // Unroll by 16 (2 AVX2 registers)
                    for (; j + 16 <= j_size; j += 16) {
                        // Load two 128-bit chunks (8 halves each)
                        __m128i v16_0 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                        __m128i v16_1 = _mm_loadu_si128((const __m128i*)(b_ptr + j + 8));

                        // Convert to two 256-bit float registers
                        __m256 v32_0 = _mm256_cvtph_ps(v16_0);
                        __m256 v32_1 = _mm256_cvtph_ps(v16_1);

                        // Store (using aligned stores if packed_B is 32-byte aligned)
                        _mm256_store_ps(packed_B_ptr + j, v32_0);
                        _mm256_store_ps(packed_B_ptr + j + 8, v32_1);
                    }

                    // Handle remainder of j_size (8-element SIMD step)
                    for (; j + 8 <= j_size; j += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(b_ptr + j));
                        _mm256_store_ps(packed_B_ptr + j, _mm256_cvtph_ps(v16));
                    }

                    // Scalar tail for the absolute remainder
                    for (; j < j_size; ++j) {
                        packed_B_ptr[j] = (float)b_ptr[j];
                    }
                }

                size_t i_tile = 0;
                for (; i_tile + 8 <= M; i_tile += 8) {
                    // pack A
                    {
                        // Pack 8 rows of A simultaneously
                        size_t k = 0;
                        for (; k + 8 <= k_size; k += 8) {
                            // Load 8 half_cpu-precision elements from 8 different rows
                            __m128i v16_r0 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 0) * K + kk + k));
                            __m128i v16_r1 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 1) * K + kk + k));
                            __m128i v16_r2 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 2) * K + kk + k));
                            __m128i v16_r3 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 3) * K + kk + k));
                            __m128i v16_r4 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 4) * K + kk + k));
                            __m128i v16_r5 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 5) * K + kk + k));
                            __m128i v16_r6 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 6) * K + kk + k));
                            __m128i v16_r7 = _mm_loadu_si128((const __m128i*)(A + (i_tile + 7) * K + kk + k));

                            // Convert all to float (F16C)
                            // This interleaves the conversion latency across 8 registers
                            __m256 v32_r0 = _mm256_cvtph_ps(v16_r0);
                            __m256 v32_r1 = _mm256_cvtph_ps(v16_r1);
                            __m256 v32_r2 = _mm256_cvtph_ps(v16_r2);
                            __m256 v32_r3 = _mm256_cvtph_ps(v16_r3);
                            __m256 v32_r4 = _mm256_cvtph_ps(v16_r4);
                            __m256 v32_r5 = _mm256_cvtph_ps(v16_r5);
                            __m256 v32_r6 = _mm256_cvtph_ps(v16_r6);
                            __m256 v32_r7 = _mm256_cvtph_ps(v16_r7);

                            // Store into packed_A (TK is the stride)
                            _mm256_store_ps(packed_A + 0 * TK + k, v32_r0);
                            _mm256_store_ps(packed_A + 1 * TK + k, v32_r1);
                            _mm256_store_ps(packed_A + 2 * TK + k, v32_r2);
                            _mm256_store_ps(packed_A + 3 * TK + k, v32_r3);
                            _mm256_store_ps(packed_A + 4 * TK + k, v32_r4);
                            _mm256_store_ps(packed_A + 5 * TK + k, v32_r5);
                            _mm256_store_ps(packed_A + 6 * TK + k, v32_r6);
                            _mm256_store_ps(packed_A + 7 * TK + k, v32_r7);
                        }

                        // Tail loop for k (if k_size is not multiple of 8)
                        for (; k < k_size; ++k) {
                            for (int i = 0; i < 8; ++i) {
                                packed_A[i * TK + k] = (float)A[(i_tile + i) * K + kk + k];
                            }
                        }
                    }

                    for (size_t ii = 0; ii + 4 <= 8; ii += 4) {
                        half_cpu *c0_ptr = C + (i_tile + ii) * N + jj;
                        half_cpu *c1_ptr = C + (i_tile + ii + 1) * N + jj;
                        half_cpu *c2_ptr = C + (i_tile + ii + 2) * N + jj;
                        half_cpu *c3_ptr = C + (i_tile + ii + 3) * N + jj;
                        const float *pack_a0_ptr = packed_A + ii * TK;
                        const float *pack_a1_ptr = packed_A + (ii + 1) * TK;
                        const float *pack_a2_ptr = packed_A + (ii + 2) * TK;
                        const float *pack_a3_ptr = packed_A + (ii + 3) * TK;

                        size_t j_tile = 0;
                        for (; j_tile + 16 <= j_size; j_tile += 16) {
                            const float *pb_j = packed_B + j_tile;
                            
                            __m256 c00, c01, c02, c03, c10, c11, c12, c13;
                            
                            if (kk == 0) {
                                c00 = _mm256_load_ps(local_bias + jj + j_tile);
                                c10 = _mm256_load_ps(local_bias + jj + j_tile + 8);

                                c01 = c02 = c03 = c00;
                                c11 = c12 = c13 = c10;
                            } else {
                                __m128i c00_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile));
                                __m128i c01_16 = _mm_loadu_si128((const __m128i*)(c1_ptr + j_tile));
                                __m128i c02_16 = _mm_loadu_si128((const __m128i*)(c2_ptr + j_tile));
                                __m128i c03_16 = _mm_loadu_si128((const __m128i*)(c3_ptr + j_tile));
                                __m128i c10_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile + 8));
                                __m128i c11_16 = _mm_loadu_si128((const __m128i*)(c1_ptr + j_tile + 8));
                                __m128i c12_16 = _mm_loadu_si128((const __m128i*)(c2_ptr + j_tile + 8));
                                __m128i c13_16 = _mm_loadu_si128((const __m128i*)(c3_ptr + j_tile + 8));

                                c00 = _mm256_cvtph_ps(c00_16);
                                c01 = _mm256_cvtph_ps(c01_16);
                                c02 = _mm256_cvtph_ps(c02_16);
                                c03 = _mm256_cvtph_ps(c03_16);
                                c10 = _mm256_cvtph_ps(c10_16);
                                c11 = _mm256_cvtph_ps(c11_16);
                                c12 = _mm256_cvtph_ps(c12_16);
                                c13 = _mm256_cvtph_ps(c13_16);
                            }

                            size_t k = 0;
                            for (; k < k_size; ++k) {
                                const float *pb0 = pb_j + k * TN;
                                __m256 b0 = _mm256_load_ps(pb0);
                                __m256 b1 = _mm256_load_ps(pb0 + 8);

                                __m256 a0 = _mm256_broadcast_ss(pack_a0_ptr + k);
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
                            
                            _mm_storeu_si128((__m128i*)(c0_ptr + j_tile), _mm256_cvtps_ph(c00, 0));
                            _mm_storeu_si128((__m128i*)(c1_ptr + j_tile), _mm256_cvtps_ph(c01, 0));
                            _mm_storeu_si128((__m128i*)(c2_ptr + j_tile), _mm256_cvtps_ph(c02, 0));
                            _mm_storeu_si128((__m128i*)(c3_ptr + j_tile), _mm256_cvtps_ph(c03, 0));
                            _mm_storeu_si128((__m128i*)(c0_ptr + j_tile + 8), _mm256_cvtps_ph(c10, 0));
                            _mm_storeu_si128((__m128i*)(c1_ptr + j_tile + 8), _mm256_cvtps_ph(c11, 0));
                            _mm_storeu_si128((__m128i*)(c2_ptr + j_tile + 8), _mm256_cvtps_ph(c12, 0));
                            _mm_storeu_si128((__m128i*)(c3_ptr + j_tile + 8), _mm256_cvtps_ph(c13, 0));
                        }

                        for (; j_tile < j_size; ++j_tile) {
                            float acc0, acc1, acc2, acc3;

                            if (kk == 0) {
                                acc0 = acc1 = acc2 = acc3 = local_bias[jj + j_tile];
                            } else {
                                acc0 = (float)c0_ptr[j_tile];
                                acc1 = (float)c1_ptr[j_tile];
                                acc2 = (float)c2_ptr[j_tile];
                                acc3 = (float)c3_ptr[j_tile];
                            }

                            for (size_t k = 0; k < k_size; ++k) {
                                float b_val = packed_B[k * TN + j_tile];
                                acc0 += pack_a0_ptr[k] * b_val;
                                acc1 += pack_a1_ptr[k] * b_val;
                                acc2 += pack_a2_ptr[k] * b_val;
                                acc3 += pack_a3_ptr[k] * b_val;
                            }

                            c0_ptr[j_tile] = (half_cpu)acc0;
                            c1_ptr[j_tile] = (half_cpu)acc1;
                            c2_ptr[j_tile] = (half_cpu)acc2;
                            c3_ptr[j_tile] = (half_cpu)acc3;
                        }
                    }
                }

                // pack A
                for (size_t i = 0; i < (M - i_tile); ++i) {
                    size_t k = 0;
                    const half_cpu *a_ptr = A + (i_tile + i) * K + kk;
                    float *packed_A_ptr = packed_A + i * TK;
                    for (; k + 8 <= k_size; k += 8) {
                        __m128i v16 = _mm_loadu_si128((const __m128i*)(a_ptr + k));
                        __m256 v32 = _mm256_cvtph_ps(v16);
                        _mm256_store_ps(packed_A_ptr + k, v32);
                    }

                    for (; k < k_size; ++k) {
                        packed_A_ptr[k] = (float)a_ptr[k];
                    }
                }
                
                const float *pack_a_ptr = packed_A;

                if (i_tile + 4 <= M) {
                    half_cpu *c0_ptr = C + i_tile * N + jj;
                    half_cpu *c1_ptr = C + (i_tile + 1) * N + jj;
                    half_cpu *c2_ptr = C + (i_tile + 2) * N + jj;
                    half_cpu *c3_ptr = C + (i_tile + 3) * N + jj;

                    size_t j_tile = 0;
                    for (; j_tile + 16 <= j_size; j_tile += 16) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c00, c01, c02, c03, c10, c11, c12, c13;
                        
                        if (kk == 0) {
                            c00 = _mm256_load_ps(local_bias + jj + j_tile);
                            c10 = _mm256_load_ps(local_bias + jj + j_tile + 8);

                            c01 = c02 = c03 = c00;
                            c11 = c12 = c13 = c10;
                        } else {
                            __m128i c00_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile));
                            __m128i c01_16 = _mm_loadu_si128((const __m128i*)(c1_ptr + j_tile));
                            __m128i c02_16 = _mm_loadu_si128((const __m128i*)(c2_ptr + j_tile));
                            __m128i c03_16 = _mm_loadu_si128((const __m128i*)(c3_ptr + j_tile));
                            __m128i c10_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile + 8));
                            __m128i c11_16 = _mm_loadu_si128((const __m128i*)(c1_ptr + j_tile + 8));
                            __m128i c12_16 = _mm_loadu_si128((const __m128i*)(c2_ptr + j_tile + 8));
                            __m128i c13_16 = _mm_loadu_si128((const __m128i*)(c3_ptr + j_tile + 8));

                            c00 = _mm256_cvtph_ps(c00_16);
                            c01 = _mm256_cvtph_ps(c01_16);
                            c02 = _mm256_cvtph_ps(c02_16);
                            c03 = _mm256_cvtph_ps(c03_16);
                            c10 = _mm256_cvtph_ps(c10_16);
                            c11 = _mm256_cvtph_ps(c11_16);
                            c12 = _mm256_cvtph_ps(c12_16);
                            c13 = _mm256_cvtph_ps(c13_16);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
                            const float *pb0 = pb_j + k * TN;
                            __m256 b0 = _mm256_load_ps(pb0);
                            __m256 b1 = _mm256_load_ps(pb0 + 8);

                            __m256 a0 = _mm256_broadcast_ss(pack_a_ptr + k);
                            __m256 a1 = _mm256_broadcast_ss(pack_a_ptr + TK + k);
                            __m256 a2 = _mm256_broadcast_ss(pack_a_ptr + 2 * TK + k);
                            __m256 a3 = _mm256_broadcast_ss(pack_a_ptr + 3 * TK + k);

                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c10 = _mm256_fmadd_ps(a0, b1, c10);

                            c01 = _mm256_fmadd_ps(a1, b0, c01);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);

                            c02 = _mm256_fmadd_ps(a2, b0, c02);
                            c12 = _mm256_fmadd_ps(a2, b1, c12);

                            c03 = _mm256_fmadd_ps(a3, b0, c03);
                            c13 = _mm256_fmadd_ps(a3, b1, c13);
                        }
                        
                        _mm_storeu_si128((__m128i*)(c0_ptr + j_tile), _mm256_cvtps_ph(c00, 0));
                        _mm_storeu_si128((__m128i*)(c1_ptr + j_tile), _mm256_cvtps_ph(c01, 0));
                        _mm_storeu_si128((__m128i*)(c2_ptr + j_tile), _mm256_cvtps_ph(c02, 0));
                        _mm_storeu_si128((__m128i*)(c3_ptr + j_tile), _mm256_cvtps_ph(c03, 0));
                        _mm_storeu_si128((__m128i*)(c0_ptr + j_tile + 8), _mm256_cvtps_ph(c10, 0));
                        _mm_storeu_si128((__m128i*)(c1_ptr + j_tile + 8), _mm256_cvtps_ph(c11, 0));
                        _mm_storeu_si128((__m128i*)(c2_ptr + j_tile + 8), _mm256_cvtps_ph(c12, 0));
                        _mm_storeu_si128((__m128i*)(c3_ptr + j_tile + 8), _mm256_cvtps_ph(c13, 0));
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc0, acc1, acc2, acc3;

                        if (kk == 0) {
                            acc0 = acc1 = acc2 = acc3 = local_bias[jj + j_tile];
                        } else {
                            acc0 = (float)c0_ptr[j_tile];
                            acc1 = (float)c1_ptr[j_tile];
                            acc2 = (float)c2_ptr[j_tile];
                            acc3 = (float)c3_ptr[j_tile];
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            float b_val = packed_B[k * TN + j_tile];
                            acc0 += pack_a_ptr[k] * b_val;
                            acc1 += pack_a_ptr[TK + k] * b_val;
                            acc2 += pack_a_ptr[2 * TK + k] * b_val;
                            acc3 += pack_a_ptr[3 * TK + k] * b_val;
                        }

                        c0_ptr[j_tile] = (half_cpu)acc0;
                        c1_ptr[j_tile] = (half_cpu)acc1;
                        c2_ptr[j_tile] = (half_cpu)acc2;
                        c3_ptr[j_tile] = (half_cpu)acc3;
                    }
                
                    i_tile += 4;
                    pack_a_ptr += 4 * TK;
                }

                if (i_tile + 2 <= M) {
                    half_cpu *c0_ptr = C + i_tile * N + jj;
                    half_cpu *c1_ptr = C + (i_tile + 1) * N + jj;

                    size_t j_tile = 0;
                    for (; j_tile + 8 <= j_size; j_tile += 8) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c00, c01; // , c10, c20, c30, c01, c11, c21, c31;
                        
                        if (kk == 0) {
                            c00 = c01 = _mm256_load_ps(local_bias + jj + j_tile);
                        } else {
                            __m128i c00_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile));
                            __m128i c01_16 = _mm_loadu_si128((const __m128i*)(c1_ptr + j_tile));

                            c00 = _mm256_cvtph_ps(c00_16);
                            c01 = _mm256_cvtph_ps(c01_16);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
                            const float *pb0 = pb_j + k * TN;
                            __m256 a0 = _mm256_broadcast_ss(pack_a_ptr + k);
                            __m256 a1 = _mm256_broadcast_ss(pack_a_ptr + TK + k);
                            
                            __m256 b0 = _mm256_load_ps(pb0);

                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a1, b0, c01);
                        }
                        
                        _mm_storeu_si128((__m128i*)(c0_ptr + j_tile), _mm256_cvtps_ph(c00, 0));
                        _mm_storeu_si128((__m128i*)(c1_ptr + j_tile), _mm256_cvtps_ph(c01, 0));
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc0, acc1;

                        if (kk == 0) {
                            acc0 = acc1 = local_bias[jj + j_tile];
                        } else {
                            acc0 = (float)c0_ptr[j_tile];
                            acc1 = (float)c1_ptr[j_tile];
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            float b_val = packed_B[k * TN + j_tile];
                            acc0 += pack_a_ptr[k] * b_val;
                            acc1 += pack_a_ptr[TK + k] * b_val;
                        }

                        c0_ptr[j_tile] = (half_cpu)acc0;
                        c1_ptr[j_tile] = (half_cpu)acc1;
                    }

                    i_tile += 2;
                    pack_a_ptr += 2 * TK;
                }

                if (i_tile < M) {
                    half_cpu *c0_ptr = C + i_tile * N + jj;

                    size_t j_tile = 0;
                    for (; j_tile + 8 <= j_size; j_tile += 8) {
                        const float *pb_j = packed_B + j_tile;
                        
                        __m256 c0; //, c1, c2, c3;
                        
                        if (kk == 0) {
                            c0 = _mm256_load_ps(local_bias + jj + j_tile);
                        } else {
                            __m128i c0_16 = _mm_loadu_si128((const __m128i*)(c0_ptr + j_tile));

                            c0 = _mm256_cvtph_ps(c0_16);
                        }

                        size_t k = 0;
                        for (; k < k_size; ++k) {
                            const float *pb0 = pb_j + k * TN;
                            __m256 a0 = _mm256_broadcast_ss(pack_a_ptr + k);

                            c0 = _mm256_fmadd_ps(a0, _mm256_load_ps(pb0), c0);
                        }
                        
                        _mm_storeu_si128((__m128i*)(c0_ptr + j_tile), _mm256_cvtps_ph(c0, 0));
                    }

                    for (; j_tile < j_size; ++j_tile) {
                        float acc_h;

                        if (kk == 0) {
                            acc_h = bias ? (float)(bias[jj + j_tile]) : 0.0f;
                        } else {
                            acc_h = (float)(c0_ptr[j_tile]);
                        }

                        for (size_t k = 0; k < k_size; ++k) {
                            // FP16 mul → FP16 add (scalar, rounded each step)
                            acc_h += pack_a_ptr[k] * packed_B[k * TN + j_tile];
                        }

                        c0_ptr[j_tile] = (half_cpu)acc_h;
                    }
                }
            }
        }
    }
}

void f16ab_f32c_avx2_kernel(
    const half_cpu *__restrict mat_A,
    const half_cpu *__restrict mat_B,
    const half_cpu *__restrict mat_bias,
    float *__restrict mat_C,
    size_t M, size_t N, size_t K, bool mat_B_transpose
) {
    if (!mat_B_transpose) {
        if (N >= 1024 && K >= 1024) {
            linear_f16ab_fp32c_normal_tm_4<64, 256>(mat_A, mat_B, mat_bias, mat_C, M, N, K);
            return;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = mat_bias ? (float)mat_bias[j] : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                float a = (float)mat_A[i * K + k];
                float b = mat_B_transpose
                        ? (float)mat_B[j * K + k]
                        : (float)mat_B[k * N + j];
                sum += a * b;
            }

            mat_C[i * N + j] = sum;
        }
    }
}
#endif
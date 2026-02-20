// tensor.cpp
#include "../include/tensor.hpp"

const char* dtypeToStr(DType::Type dtype) {
    switch (dtype) {
        case DType::FP32: return "FP32";
        case DType::INT32: return "INT32";
        case DType::FP16: return "FP16";
        case DType::INT8: return "INT8";
        case DType::INT4: return "INT4";
        default: return "Unknown";
    }
}

Tensor::Tensor(
    const vector<size_t> &shape_, DType::Type dtype_
) : shape(shape_), dtype(dtype_), owns_host_buf(true) {
    if (dtype != DType::FP32 && dtype != DType::FP16) {
        fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(dtype));
        exit(1);   
    }

    ndim = shape_.size();
    size_t N_ = num_elem();
    size_t each_elem = get_dtype_size();

    buf = calloc(N_, each_elem);
    CHECK_ALLOC(buf, N_ * each_elem);

    printf("Allocate tensor with size %lf MB\n", double(N_ * each_elem) / 1024.0 / 1024.0);
}

Tensor::Tensor(
    const vector<size_t> &shape_, void *buf_, DType::Type dtype_
) : shape(shape_), buf(buf_), dtype(dtype_), owns_host_buf(false) {
    ndim = shape_.size();

    size_t size_buf = num_elem() * get_dtype_size();

    printf("New tensor wrapper with size %lf MB\n", double(size_buf) / 1024.0 / 1024.0);
}

Tensor::Tensor(
    const std::vector<size_t> &shape_, void *buf_, void *scale_buf_,
    size_t group_size_, bool group_quantized_,
    DType::Type dtype_, DType::Type scale_dtype_
) : shape(shape_), buf(buf_), scale_buf(scale_buf_), group_size(group_size_), group_quantized(group_quantized_), dtype(dtype_), scale_dtype(scale_dtype_), owns_host_buf(false) {
    ndim = shape_.size();

    size_t size_buf = num_elem() * get_dtype_size();
    size_t size_scale_buf = get_dtype_size(true);

    if (group_quantized_) {
        size_scale_buf *= (num_elem() / group_size);
    } else {
        size_scale_buf *= (num_elem() / shape_[ndim - 1]);
    }

    printf("Dtype tensor %s with scale %s\n", dtypeToStr(dtype), dtypeToStr(scale_dtype));
    printf("New tensor wrapper with size %lf MB\n", double(size_buf) / 1024.0 / 1024.0);
    printf("New tensor wrapper scale with size %lf MB\n", double(size_scale_buf) / 1024.0 / 1024.0);
}

Tensor::~Tensor() {
    static size_t page_size = sysconf(_SC_PAGESIZE);

    auto safe_munmap = [&](void* ptr, size_t data_size) {
        if (!ptr) return;

        uintptr_t addr = (uintptr_t)ptr;
        uintptr_t aligned_addr = (addr / page_size) * page_size;
        size_t offset_in_page = addr - aligned_addr;
        size_t total_mapped_size = data_size + offset_in_page;

        if (munmap((void*)aligned_addr, total_mapped_size) == -1) {
            perror("munmap failed");
        }
    };

    // --------------------
    // Free main tensor buf
    // --------------------
    if (buf) {
        if (owns_host_buf) {
            free(buf);
        } else {
            size_t data_size = num_elem() * get_dtype_size();
            safe_munmap(buf, data_size);
        }
        buf = nullptr;
    }

    // --------------------
    // Free scale buffer
    // --------------------
    if (scale_buf) {
        size_t stride = group_quantized ? group_size : shape[ndim - 1];
        size_t num_scales = (num_elem() + stride - 1) / stride;
        size_t data_size = num_scales * get_dtype_size(true);

        safe_munmap(scale_buf, data_size);
        scale_buf = nullptr;
    }

    if (sum_int8_buf) {
        free(sum_int8_buf);
    }
}

void* Tensor::ptr(const std::vector<size_t>& indices, bool get_scale) const {
    if (get_scale && scale_buf == nullptr) {
        fprintf(stderr, "scale_buf is null\n");
        exit(1);
    }
    void *work_buf = get_scale ? scale_buf : buf;

    if (indices.empty()) return work_buf;

    if (indices.size() > ndim) {
        fprintf(stderr, "Error: indices size %zu exceeds ndim %zu\n", indices.size(), ndim);
        exit(1);
    }

    size_t offset = 0;
    size_t remaining_elements = num_elem();

    // Calculate element-wise offset assuming row-major contiguity
    for (size_t i = 0; i < indices.size(); ++i) {
        remaining_elements /= shape[i];
        offset += indices[i] * remaining_elements;
    }

    if (get_scale) {
        size_t stride = group_quantized ? group_size : shape[ndim - 1];
        if (offset % stride) {
            fprintf(stderr, "scale offset misaligned\n");
            exit(1);
        }
        offset /= stride;
    }

    // Apply offset based on the size of the data type
    // dtype_size() should return sizeof(float), sizeof(int), etc.
    char* base_ptr = static_cast<char*>(work_buf);
    return base_ptr + (offset * get_dtype_size(get_scale));
}

PtrPair Tensor::ptr_all(const std::vector<size_t> &indices) const {
    PtrPair out{};

    // Fast path: no indexing
    if (indices.empty()) {
        out.buf   = buf;
        out.scale = scale_buf;
        out.sum_int8 = sum_int8_buf;
        return out;
    }
    
    if (indices.size() > ndim) {
        fprintf(stderr, "Error: indices size %zu exceeds ndim %zu\n", indices.size(), ndim);
        exit(1);
    }

    size_t offset = 0;
    size_t remaining_elements = num_elem();

    // Calculate element-wise offset assuming row-major contiguity
    for (size_t i = 0; i < indices.size(); ++i) {
        remaining_elements /= shape[i];
        offset += indices[i] * remaining_elements;
    }

    // Apply offset based on the size of the data type
    // dtype_size() should return sizeof(float), sizeof(int), etc.
    char* buf_ptr = static_cast<char*>(buf);
    out.buf = buf_ptr + (offset * get_dtype_size());
    if (scale_buf) {
        size_t stride = group_quantized ? group_size : shape[ndim - 1];
        if (offset % stride) {
            fprintf(stderr, "scale offset misaligned\n");
            exit(1);
        }
        size_t offset_s = offset / stride;
        char* scale_ptr = static_cast<char*>(scale_buf);
        out.scale = scale_ptr + (offset_s * get_dtype_size(true));
    } else {
        out.scale = nullptr;
    }

    if (sum_int8_buf && dtype == DType::INT8) {
        size_t stride = group_quantized ? group_size : shape[ndim - 1];
        if (offset % stride) {
            fprintf(stderr, "scale offset misaligned\n");
            exit(1);
        }

        char* sum_int8_ptr = static_cast<char*>(sum_int8_buf);
        size_t offset_s = offset / stride;

        #if defined(__AVX512F__) && defined(__AVX512DQ__)
            offset_s <<= 4;
        #elif defined(__AVX2__) && defined(__FMA__)
            offset_s <<= 3;
        #endif

        out.sum_int8 = sum_int8_ptr + (offset_s * sizeof(int));
    } else {
        out.sum_int8 = nullptr;
    }

    return out;
}

void Tensor::offline_sum_int8() {
    const size_t num_groups = num_elem() / group_size;
    
    // Define vector width based on architecture
    #if defined(__AVX512F__) && defined(__AVX512DQ__)
        const size_t vals_per_group = 16; 
    #elif defined(__AVX2__) && defined(__FMA__)
        const size_t vals_per_group = 8;
    #else
        const size_t vals_per_group = 1;
    #endif

    // Allocate aligned memory for the correction buffer
    size_t total_ints = num_groups * vals_per_group;
    if (sum_int8_buf) free(sum_int8_buf);
    sum_int8_buf = aligned_alloc(64, total_ints * sizeof(int32_t));
    int32_t* sum_ptr = (int32_t*)sum_int8_buf;

    const int8_t* b_data = (const int8_t*)(buf); 

    for (size_t g = 0; g < num_groups; ++g) {
        const int8_t* group_ptr = b_data + (g * group_size);
        int32_t* current_out = sum_ptr + (g * vals_per_group);

        #if defined(__AVX512F__) && defined(__AVX512DQ__)
            // AVX-512 Dynamic Loop
            __m512i v_sum = _mm512_setzero_si512();
            __m512i v_ones = _mm512_set1_epi8(1);
            
            for (size_t k = 0; k < group_size; k += 64) {
                // Process 64 bytes at a time (full ZMM)
                v_sum = _mm512_dpbusd_epi32(v_sum, v_ones, _mm512_loadu_si512((__m512i*)(group_ptr + k)));
            }
            
            // Multiply by 128 and store the 16 partial sums
            v_sum = _mm512_slli_epi32(v_sum, 7);
            _mm512_store_si512((__m512i*)current_out, v_sum);

        #elif defined(__AVX2__) && defined(__FMA__)
            // AVX2 Dynamic Loop
            __m256i v_sum = _mm256_setzero_si256();
            __m256i v_ones = _mm256_set1_epi8(1);
            __m256i v_madd_ones = _mm256_set1_epi16(1);
            
            for (size_t k = 0; k < group_size; k += 32) {
                __m256i b = _mm256_loadu_si256((__m256i*)(group_ptr + k));
                // Standard AVX2 horizontal sum pattern: maddubs -> madd -> add
                __m256i mad = _mm256_maddubs_epi16(v_ones, b);
                v_sum = _mm256_add_epi32(v_sum, _mm256_madd_epi16(mad, v_madd_ones));
            }
            
            v_sum = _mm256_slli_epi32(v_sum, 7);
            _mm256_store_si256((__m256i*)current_out, v_sum);

        #else
            // Scalar fallback
            int32_t s = 0;
            for (size_t k = 0; k < group_size; ++k) {
                s += (int32_t)group_ptr[k];
            }
            current_out[0] = s << 7;
        #endif
    }
    has_sum_int8 = true;
}

size_t Tensor::num_elem() const {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

size_t Tensor::get_dtype_size(bool get_scale) const {
    DType::Type get_dtype = get_scale ? scale_dtype : dtype;
    switch (get_dtype) {
        case DType::FP32: return sizeof(float);
        case DType::INT32: return sizeof(int);
        case DType::FP16: return sizeof(half_cpu);
        case DType::INT8: return sizeof(int8_t);
        default: 
            fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(get_dtype));
            exit(1);
    }
}

void Tensor::reshape(const vector<size_t> &shape_) {
    size_t n = 1;
    ndim = shape_.size();  // ndim<=5
    for (size_t i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
        n *= shape[i];
    }
}

void Tensor::printShape(const std::string &descr) const {
    #pragma omp critical
    {
        printf("Shape of %s tensor: ", descr.c_str());
        for (size_t i = 0; i < ndim; i++) {
            printf("%zu ", shape[i]);
        }
        printf("\n");

        if (scale_buf) {
            printf("Shape of %s scale: ", descr.c_str());
            for (size_t i = 0; i < ndim - 1; i++) {
                printf("%zu ", shape[i]);
            }
            if (group_quantized) {
                printf("%zu", shape[ndim - 1] / group_size);
            }
            printf("\n");
        }
    }
}

void Tensor::printDebug(const std::string &descr, bool full_tensor) const {
    #pragma omp critical
    {
        size_t elem = num_elem();
        size_t batches = elem / shape[ndim - 1];

        if (!full_tensor) {
            elem = std::min((size_t)5, elem);
        } else {
            elem = shape[ndim - 1];
        }

        printf("Print debug %s: ", descr.c_str());
        if (dtype == DType::FP32) {
            float *fp32_buf = (float *)buf;
            if (full_tensor) {
                printf("\n");
                for (size_t i = 0; i < batches; i++) {
                    for (size_t j = 0; j < elem; j++) {
                        printf("%.2f ", fp32_buf[i * elem + j]);
                    }
                    printf("\n");
                }
            } else {
                for (size_t i = 0; i < elem; i++) {
                    printf("%.2f ", fp32_buf[i]);
                }
            }
        } else if (dtype == DType::FP16) {
            half_cpu *fp16_buf = (half_cpu *)buf;
            if (full_tensor) {
                printf("\n");
                for (size_t i = 0; i < batches; i++) {
                    for (size_t j = 0; j < elem; j++) {
                        printf("%.2f ", (float)(fp16_buf[i * elem + j]));
                    }
                    printf("\n");
                }
            } else {
                for (size_t i = 0; i < elem; i++) {
                    printf("%.2f ", (float)(fp16_buf[i]));
                }
            }
        } else if (dtype == DType::INT32) {
            int *int32_buf = (int *)buf;
            if (full_tensor) {
                printf("\n");
                for (size_t i = 0; i < batches; i++) {
                    for (size_t j = 0; j < elem; j++) {
                        printf("%d ", int32_buf[i * elem + j]);
                    }
                    printf("\n");
                }
            } else {
                for (size_t i = 0; i < elem; i++) {
                    printf("%d ", int32_buf[i]);
                }
            }
        }
        printf("\n");

    }
}

// right now only works for CPU
void Tensor::permute(const std::vector<size_t> &order) {
    // Verify the permutation order is valid
    if (order.size() != ndim) {
        throw std::invalid_argument("Permute order size must match tensor ndim");
    }
    
    // Create the new shape based on the permutation
    std::vector<size_t> new_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = shape[order[i]];
    }
    
    // Create a buffer for the permuted tensor
    void *new_buf = malloc(num_elem() * get_dtype_size());
    if (!new_buf) {
        throw std::bad_alloc();
    }
    
    // Calculate strides for both original and new shapes
    std::vector<size_t> old_strides(ndim);
    std::vector<size_t> new_strides(ndim);
    
    // Calculate original strides (row-major order)
    old_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * shape[i + 1];
    }
    
    // Calculate new strides based on permuted shape
    new_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }
    
    // Calculate the size of each element
    size_t elem_size = get_dtype_size();
    size_t total_elements = num_elem();
    
    // Helper function to compute index from coordinates
    auto compute_flat_index = [](const std::vector<size_t> &indices,
                                 const std::vector<size_t> &strides,
                                 size_t ndim) -> size_t {
        size_t idx = 0;
        for (size_t i = 0; i < ndim; ++i) {
            idx += indices[i] * strides[i];
        }
        return idx;
    };
    
    // Iterate through all elements in the original tensor
    std::vector<size_t> old_indices(ndim, 0);
    std::vector<size_t> new_indices(ndim, 0);
    
    // For small ndim, we could use nested loops, but for arbitrary ndim,
    // we use a systematic approach
    for (size_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
        // Convert flat index to multi-dimensional indices in original order
        size_t temp = flat_idx;
        for (int i = ndim - 1; i >= 0; --i) {
            old_indices[i] = temp % shape[i];
            temp /= shape[i];
        }
        
        // Apply permutation to get indices in new order
        for (size_t i = 0; i < ndim; ++i) {
            new_indices[i] = old_indices[order[i]];
        }
        
        // Calculate positions in both buffers
        size_t old_pos = compute_flat_index(old_indices, old_strides, ndim);
        size_t new_pos = compute_flat_index(new_indices, new_strides, ndim);
        
        // Copy the element
        memcpy(static_cast<char*>(new_buf) + new_pos * elem_size,
               static_cast<char*>(buf) + old_pos * elem_size,
               elem_size);
    }
    
    // Update the tensor
    if (owns_host_buf) {
        free(buf);
    } else {
        static size_t page_size = sysconf(_SC_PAGESIZE);
        uintptr_t addr = (uintptr_t)buf;
        uintptr_t aligned_addr = (addr / page_size) * page_size;
        size_t offset_in_page = addr - aligned_addr;

        // USE YOUR SIZE CALCULATION HERE
        size_t data_size = num_elem() * get_dtype_size(); 
        size_t total_mapped_size = data_size + offset_in_page;

        if (munmap((void*)aligned_addr, total_mapped_size) == -1) {
            perror("munmap failed");
        }
    }
    buf = new_buf;
    shape = new_shape;
    permuted = true;
    owns_host_buf = true;
}

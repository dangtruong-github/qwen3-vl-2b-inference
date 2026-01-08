// tensor.cpp
#include "../include/tensor.hpp"

const char* dtypeToStr(DType::Type dtype) {
    switch (dtype) {
        case DType::FP32: return "FP32";
        case DType::INT32: return "INT32";
        default: return "Unknown";
    }
}

Tensor::Tensor(
    const vector<size_t> &shape_, DType::Type dtype
) : shape(shape_), dtype(dtype), owns_host_buf(true) {
    ndim = shape_.size();
    size_t N_ = num_elem();
    if (dtype == DType::FP32) {
        buf = calloc(N_, sizeof(float));
        CHECK_ALLOC(buf, N_ * sizeof(float));
    } else {
        fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(dtype));
        exit(1);
    }
}

Tensor::Tensor(
    const vector<size_t> &shape_, void *buf_, DType::Type dtype
) : shape(shape_), buf(buf_), dtype(dtype), owns_host_buf(false) {
    ndim = shape_.size();
}

Tensor::~Tensor() {
    if (buf != nullptr) {
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
        buf = nullptr;
    }
}

void* Tensor::ptr(const std::vector<size_t>& strides_) const {
    if (strides_.empty()) {
        return buf;
    }

    if (strides_.size() > ndim) {
        fprintf(stderr, "Error in shape: stride=%ld ndim=%ld", strides_.size(), ndim);
        exit(1);
    }

    size_t stride_buf = num_elem();
    if (dtype == DType::FP32) {
        const float *cur_buf = (const float *)buf;
        for (int i = 0; i < strides_.size(); i++) {
            stride_buf /= shape[i];
            cur_buf += strides_[i] * stride_buf;
        }
        return const_cast<float*>(cur_buf);
    } else if (dtype == DType::INT32) {
        const int *cur_buf = (const int *)buf;
        for (int i = 0; i < strides_.size(); i++) {
            stride_buf /= shape[i];
            cur_buf += strides_[i] * stride_buf;
        }
        return const_cast<int*>(cur_buf);
    }

    return nullptr;
}

size_t Tensor::num_elem() const {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

size_t Tensor::get_dtype_size() const {
    if (dtype == DType::FP32) {
        return sizeof(float);
    } else if (dtype == DType::INT32) {
        return sizeof(int);
    } else {
        fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(dtype));
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
        fflush(stdout);
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
        fflush(stdout);
    }
}

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
    owns_host_buf = true;
}

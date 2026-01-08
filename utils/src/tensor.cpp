// tensor.cpp
#include "../include/tensor.hpp"

const char* dtypeToStr(DType::Type dtype) {
    switch (dtype) {
        case DType::FP32: return "FP32";
        case DType::INT32: return "INT32";
        case DType::FP16: return "FP16";
        default: return "Unknown";
    }
}

Tensor::Tensor(
    const vector<size_t> &shape_, DType::Type dtype_, DType::Type gpu_dtype_
) : shape(shape_), dtype(dtype_), owns_host_buf(true) {
    ndim = shape_.size();
    size_t N_ = num_elem();
    if (dtype == DType::FP32) {
        buf = calloc(N_, sizeof(float));
        CHECK_ALLOC(buf, N_ * sizeof(float));
    } else {
        fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(dtype));
        exit(1);
    }

    gpu_dtype = gpu_dtype_;
}

Tensor::Tensor(
    const vector<size_t> &shape_, void *buf_,
    DType::Type dtype_, DType::Type gpu_dtype_
) : shape(shape_), buf(buf_), dtype(dtype_), owns_host_buf(false) {
    ndim = shape_.size();

    gpu_dtype = gpu_dtype_;
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

    if (gpu_buf != nullptr) {
        free_device();
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
    } else if (dtype == DType::FP16) {
        return sizeof(half_cpu);
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
    owns_host_buf = true;
}

// GPU API
size_t Tensor::get_gpu_dtype_size() const {
    if (dtype == DType::FP32) {
        return sizeof(float);
    } else if (dtype == DType::INT32) {
        return sizeof(int);
    } else if (dtype == DType::FP16) {
        return sizeof(half);
    } else {
        fprintf(stderr, "Dtype not implemented %s\n", dtypeToStr(dtype));
        exit(1);
    }
}

void Tensor::malloc_device() {
    if (gpu_buf) {
        fprintf(stderr, "Tensor has already been allocated");
        exit(1);
    }

    size_t malloc_size = num_elem() * get_gpu_dtype_size();

    printf("Malloc GPU with num elem = %zu\n", num_elem());
    printf("Malloc GPU with dtype size = %zu\n", get_gpu_dtype_size());
    printf("Malloc GPU with size %lf MB\n", float(malloc_size) / 1024.0);
    fflush(stdout);

    CHECK_CUDA(cudaMalloc(&gpu_buf, malloc_size));
}

void Tensor::free_device() {
    if (gpu_buf == nullptr) {
        fprintf(stderr, "Tensor has already been deallocated");
        exit(1);
    }

    CHECK_CUDA(cudaFree(gpu_buf));
    gpu_buf = nullptr;
}

void Tensor::to_gpu(cudaStream_t stream) {
    if (gpu_buf == nullptr) {
        malloc_device();
    }

    size_t n = num_elem();
    size_t gpu_bytes = n * get_gpu_dtype_size();

    if (dtype == gpu_dtype) {
        // Direct copy if types match
        CHECK_CUDA(cudaMemcpyAsync(gpu_buf, buf, gpu_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // 1. Allocate a temporary host buffer of the target GPU type
        void* temp_host_buf = malloc(gpu_bytes);
        if (!temp_host_buf) { 
            fprintf(stderr, "Failed to allocate temporary buffer for conversion\n");
            exit(1);
        }

        // 2. Perform CPU conversion
        if (dtype == DType::FP32 && gpu_dtype == DType::FP16) {
            float* src = (float*)buf;
            half_cpu* dst = (half_cpu*)temp_host_buf;
            for (size_t i = 0; i < n; ++i) dst[i] = (half_cpu)src[i];
        } 
        else if (dtype == DType::FP16 && gpu_dtype == DType::FP32) {
            half_cpu* src = (half_cpu*)buf;
            float* dst = (float*)temp_host_buf;
            for (size_t i = 0; i < n; ++i) dst[i] = (float)src[i];
        }
        else {
            fprintf(stderr, "CPU conversion from %s to %s not implemented\n", dtypeToStr(dtype), dtypeToStr(gpu_dtype));
            free(temp_host_buf);
            exit(1);
        }

        // 3. Copy the converted data to the GPU
        CHECK_CUDA(cudaMemcpyAsync(gpu_buf, temp_host_buf, gpu_bytes, cudaMemcpyHostToDevice, stream));

        // 4. Synchronize and free temporary memory
        // We must sync because temp_host_buf is used by the async copy
        cudaStreamSynchronize(stream);
        free(temp_host_buf);
    }
}

void Tensor::from_gpu(cudaStream_t stream) {
    if (gpu_buf == nullptr) {
        fprintf(stderr, "Error: GPU buffer hasn't been allocated.\n");
        exit(1);
    }

    size_t n = num_elem();
    size_t gpu_bytes = n * get_gpu_dtype_size();
    size_t host_bytes = n * get_dtype_size();

    if (dtype == gpu_dtype) {
        // Direct copy if types match
        CHECK_CUDA(cudaMemcpyAsync(buf, gpu_buf, host_bytes, cudaMemcpyDeviceToHost, stream));
    } else {
        // 1. Allocate a temporary host buffer to receive raw GPU data
        void* temp_host_buf = malloc(gpu_bytes);
        if (!temp_host_buf) {
            fprintf(stderr, "Failed to allocate temporary buffer for conversion\n");
            exit(1);
        }

        // 2. Copy from GPU to temporary host buffer
        CHECK_CUDA(cudaMemcpyAsync(temp_host_buf, gpu_buf, gpu_bytes, cudaMemcpyDeviceToHost, stream));

        // 3. Synchronize so we can read the data on CPU
        cudaStreamSynchronize(stream);

        // 4. Perform CPU conversion back to original host dtype
        if (gpu_dtype == DType::FP16 && dtype == DType::FP32) {
            half_cpu* src = (half_cpu*)temp_host_buf;
            float* dst = (float*)buf;
            for (size_t i = 0; i < n; ++i) dst[i] = (float)src[i];
        }
        else if (gpu_dtype == DType::FP32 && dtype == DType::FP16) {
            float* src = (float*)temp_host_buf;
            half_cpu* dst = (half_cpu*)buf;
            for (size_t i = 0; i < n; ++i) dst[i] = (half_cpu)src[i];
        }
        else {
            fprintf(stderr, "CPU conversion from %s back to %s not implemented\n", dtypeToStr(gpu_dtype), dtypeToStr(dtype));
            free(temp_host_buf);
            exit(1);
        }

        free(temp_host_buf);
    }
}


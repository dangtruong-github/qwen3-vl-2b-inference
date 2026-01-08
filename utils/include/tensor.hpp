#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "half.hpp" /* for half on CPU ('half_cpu') */
#include "cuda_fp16.h" /* for half on GPU ('half') */

using half_cpu = half_float::half;        // Bring the type into scope

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                             \
    do {                                                              \
        cudaError_t status_ = call;                                    \
        if (status_ != cudaSuccess) {                                   \
            fprintf(                                                     \
                stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,\
                cudaGetErrorName(status_), cudaGetErrorString(status_)     \
            );                                                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                     \
    } while (0)


// Helper to check for allocation errors
#define CHECK_ALLOC(ptr, size) if (!ptr) { \
    fprintf(stderr, "Error: failed to allocate memory for %ld bytes\n", (long)(size)); \
    return; \
}

using std::vector;

struct DType {
    enum Type { FP32, INT32, FP16 };
};

const char* dtypeToStr(DType::Type dtype);

struct Tensor {
    void* buf = nullptr;
    size_t ndim = 0;
    std::vector<size_t> shape;
    DType::Type dtype;
    bool owns_host_buf = false;
    bool use_gpu = false;
    
    // Constructors & Destructor
    Tensor(
        const std::vector<size_t> &shape_,
        DType::Type dtype_ = DType::FP32,
        DType::Type gpu_dtype_ = DType::FP32
    );
    Tensor(
        const std::vector<size_t> &shape_, void *buf_,
        DType::Type dtype_ = DType::FP32,
        DType::Type gpu_dtype_ = DType::FP32
    );
    ~Tensor();

    // Public API
    size_t num_elem() const;
    size_t get_dtype_size() const;
    void* ptr(const std::vector<size_t> &strides_ = {}) const;
    void reshape(const std::vector<size_t> &shape_);
    void printShape(const std::string &descr) const;
    void printDebug(const std::string &descr, bool full_tensor = false) const;
    void permute(const std::vector<size_t> &order);

    // GPU API
    void *gpu_buf = nullptr;
    DType::Type gpu_dtype;
    size_t get_gpu_dtype_size() const;
    void malloc_device();
    void free_device();
    void to_gpu(cudaStream_t stream = 0);
    void from_gpu(cudaStream_t stream = 0);
};


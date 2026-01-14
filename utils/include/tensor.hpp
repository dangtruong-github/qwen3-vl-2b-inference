#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

#include "half.hpp" /* for half on CPU ('half_cpu') */

using half_cpu = half_float::half;        // Bring the type into scope


// Helper to check for allocation errors
#define CHECK_ALLOC(ptr, size) if (!ptr) { \
    fprintf(stderr, "Error: failed to allocate memory for %ld bytes\n", (long)(size)); \
    return; \
}

using std::vector;

struct DType {
    enum Type { FP32, INT32, FP16, INT8, INT4 };
};

const char* dtypeToStr(DType::Type dtype);

struct Tensor {
    void* buf = nullptr;
    size_t ndim = 0;
    std::vector<size_t> shape;
    DType::Type dtype;
    bool owns_host_buf = false;
    bool use_gpu = false;

    void* scale_buf = nullptr;
    DType::Type scale_dtype;
    size_t group_size = 0;
    
    // Constructors & Destructor
    // FP32 and FP16
    Tensor(
        const std::vector<size_t> &shape_,
        DType::Type dtype_ = DType::FP32
    );
    Tensor(
        const std::vector<size_t> &shape_, void *buf_,
        DType::Type dtype_ = DType::FP32
    );
    // INT8 and INT4, only weight
    Tensor(
        const std::vector<size_t> &shape_, void *buf_, void *scale_buf_,
        size_t group_size_, DType::Type dtype_, DType::Type scale_dtype_
    );
    ~Tensor();

    // Public API
    size_t num_elem() const;
    size_t get_dtype_size(bool get_scale = false) const;
    void* ptr(
        const std::vector<size_t> &strides_ = {}, bool get_scale = false
    ) const;
    void reshape(const std::vector<size_t> &shape_);
    void printShape(const std::string &descr) const;
    // NEEDS IMPLEMENTING FOR scale_buf
    void printDebug(const std::string &descr, bool full_tensor = false) const;
    void permute(const std::vector<size_t> &order);
};

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>

// Helper to check for allocation errors
#define CHECK_ALLOC(ptr, size) if (!ptr) { \
    fprintf(stderr, "Error: failed to allocate memory for %ld bytes\n", (long)(size)); \
    return; \
}

using std::vector;

struct DType {
    enum Type { FP32, INT32 };
};

const char* dtypeToStr(DType::Type dtype);

struct Tensor {
    size_t ndim = 0;
    vector<size_t> shape;
    void *buf = nullptr;
    DType::Type dtype;
    bool owns_host_buf;

    Tensor(const vector<size_t> &shape_, DType::Type dtype = DType::FP32);
    Tensor(
        const vector<size_t> &shape_, void *buf_,
        DType::Type dtype = DType::FP32
    );
    ~Tensor();

    size_t num_elem() const;
    size_t get_dtype_size() const;  // Add this declaration
    void reshape(const vector<int> &shape_);
    void printShape(const std::string &descr) const;
    void printDebug(const std::string &descr, bool full_tensor = false) const;
};

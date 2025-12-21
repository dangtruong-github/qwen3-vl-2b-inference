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
        printShape("success type 1");
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
    if (owns_host_buf && buf != nullptr) {
        free(buf);
    }
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
void Tensor::reshape(const vector<int> &shape_) {
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

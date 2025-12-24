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

void* Tensor::ptr(const std::vector<size_t>& strides_) const {
    if (strides_.empty()) {
        return buf;
    }

    if (strides_.size() > ndim) {
        fprintf(stderr, "Error in shape: stride=%d ndim=%d", strides_.size(), ndim);
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

#include "../include/layer.hpp"

void embedding_lookup(
    const float *embedding /*[vocab, hidden]*/,
    int token_id, float *out /*[hidden]*/,
    size_t vocab_size, size_t hidden_size
) {
    memcpy(out, embedding + token_id * hidden_size, hidden_size * sizeof(float));
}

void rms_norm(
    const float *x /*[hidden]*/, const float *scale /*[hidden]*/,
    float *out /*[hidden]*/, float eps, size_t hidden_size
) {
    // calculate sum of squares
    double ss = 0.0;
    for (size_t j = 0; j < hidden_size; j++) {
        ss += x[j] * x[j];
    }
    ss /= hidden_size;
    ss += eps;
    ss = 1.0 / sqrt(ss);
    // normalize and scale
    for (size_t j = 0; j < hidden_size; j++) {
        out[j] = scale[j] * (ss * x[j]);
    }
}

void gemm(const float *mat_A, const float *mat_B, float *mat_C, size_t M, size_t N, size_t K, bool mat_B_transpose) {
    // A is M x K
    // B is K x N (or N x K if transposed)
    // C is M x N

    // Initialize C to zero
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat_C[i * N + j] = 0.0f;
        }
    }

    // Standard Matrix Multiplication: C[i][j] += A[i][k] * B[k][j]
    // Loop structure: i (M) -> j (N) -> k (K)
    if (!mat_B_transpose) {
        // B is K x N. B[k][j] = mat_B[k * N + j]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B and C
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // A[i][k] * B[k][j]
                    sum += mat_A[i * K + k] * mat_B[k * N + j];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
    // Matrix Multiplication with Transposed B: C[i][j] += A[i][k] * B^T[k][j]
    // Since B^T[k][j] is B[j][k] (from B's perspective, B is N x K)
    // We are multiplying A (M x K) by B^T (K x N).
    // B is stored in row-major order as N x K.
    else {
        // B is N x K. B^T[k][j] = B[j][k] = mat_B[j * K + k]
        for (size_t i = 0; i < M; ++i) {        // Row of A and C
            for (size_t j = 0; j < N; ++j) {    // Column of B^T and C
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) { // Inner dimension
                    // A[i][k] * B^T[k][j] which is A[i][k] * B[j][k]
                    sum += mat_A[i * K + k] * mat_B[j * K + k];
                }
                mat_C[i * N + j] = sum;
            }
        }
    }
}

void classifier_gemm(
    const float *embedding /*[vocab, hidden]*/,
    const float *hid_states /*[hidden]*/, float *logits /*[vocab]*/,
    size_t vocab_size, size_t hidden_size
) {
    gemm(hid_states, embedding, logits, 1, vocab_size, hidden_size, true);
}

void layer_norm(float* buffer, const float* input, const float* weight,
                const float* bias, int hidden_size, float eps) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < hidden_size; ++i) mean += input[i];
    mean /= hidden_size;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= hidden_size;
    float inv_std = 1.f / sqrtf(var + eps);
    for (int i = 0; i < hidden_size; ++i) {
        buffer[i] = (input[i] - mean) * inv_std;
        if (weight) buffer[i] *= weight[i];
        if (bias) buffer[i] += bias[i];
    }
    memcpy((void*)input, buffer, hidden_size * sizeof(float)); // overwrite
}

float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.79788456f * x * (1.f + 0.044715f * x * x)));
}

int argmax(const float* array, int size) {
    int idx = 0;
    float max_val = array[0];
    for (int i = 1; i < size; ++i) {
        if (array[i] > max_val) {
            max_val = array[i];
            idx = i;
        }
    }
    return idx;
}

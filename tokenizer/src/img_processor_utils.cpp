#include "../include/img_processor_utils.hpp"

void resize_bicubic(const cv::Mat img, cv::Mat &resized_img, int new_height, int new_width) {
    // Bicubic interpolation
    cv::resize(img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
}

// ===== Reference kernel (from previous message) =====
static inline float cubic_convolution1(float x, float A) {
    return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
}

static inline float cubic_convolution2(float x, float A) {
    return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
}

static inline float cubic_kernel(float x) {
    constexpr float A = -0.75f;
    x = std::abs(x);
    if (x < 1.0f) return cubic_convolution1(x, A);
    if (x < 2.0f) return cubic_convolution2(x, A);
    return 0.0f;
}

static inline float compute_scale(int in, int out) {
    return static_cast<float>(in) / out;
}

static inline float compute_source_index(float scale, int dst) {
    return scale * (dst + 0.5f) - 0.5f;
}

// ====================================================
// REQUIRED WRAPPER (signature unchanged)
// ====================================================
void resize_bicubic_torchvision(
    const cv::Mat img, cv::Mat &resized_img,
    int new_height, int new_width
) {
    CV_Assert(img.isContinuous());

    const int inH = img.rows;
    const int inW = img.cols;
    const int C   = img.channels();

    resized_img.create(new_height, new_width, img.type());

    const float scale_y = compute_scale(inH, new_height);
    const float scale_x = compute_scale(inW, new_width);

    // Temporary buffer: H x newW x C
    std::vector<float> temp(inH * new_width * C);

    const float* input  = img.ptr<float>();
    float* output = resized_img.ptr<float>();

    // ---------- Horizontal pass ----------
    for (int y = 0; y < inH; ++y) {
        for (int ox = 0; ox < new_width; ++ox) {
            float in_x = compute_source_index(scale_x, ox);

            float invscale_x = (scale_x < 1.0f) ? scale_x : 1.0f;
            float kernel_radius =
                (scale_x < 1.0f) ? (2.0f / scale_x) : 2.0f;

            int x0 = static_cast<int>(std::floor(in_x - kernel_radius));
            int x1 = static_cast<int>(std::ceil (in_x + kernel_radius));

            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                float acc = 0.0f;

                for (int ix = x0; ix <= x1; ++ix) {
                    int sx = std::min(std::max(ix, 0), inW - 1);
                    float w = cubic_kernel((in_x - ix) * invscale_x);
                    sum += w;
                    acc += w * input[(y * inW + sx) * C + c];
                }

                temp[(y * new_width + ox) * C + c] = acc / sum;
            }
        }
    }

    // ---------- Vertical pass ----------
    for (int oy = 0; oy < new_height; ++oy) {
        float in_y = compute_source_index(scale_y, oy);

        float invscale_y = (scale_y < 1.0f) ? scale_y : 1.0f;
        float kernel_radius =
            (scale_y < 1.0f) ? (2.0f / scale_y) : 2.0f;

        int y0 = static_cast<int>(std::floor(in_y - kernel_radius));
        int y1 = static_cast<int>(std::ceil (in_y + kernel_radius));

        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < C; ++c) {
                float sum = 0.0f;
                float acc = 0.0f;

                for (int iy = y0; iy <= y1; ++iy) {
                    int sy = std::min(std::max(iy, 0), inH - 1);
                    float w = cubic_kernel((in_y - iy) * invscale_y);
                    sum += w;
                    acc += w * temp[(sy * new_width + x) * C + c];
                }

                output[(oy * new_width + x) * C + c] = acc / sum;
            }
        }
    }

}

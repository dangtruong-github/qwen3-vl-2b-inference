#include "../include/resize.hpp"

// Bicubic kernel function (a = -0.75 as in PyTorch)
static inline float bicubic_kernel(float x) {
    x = std::abs(x);
    if (x < 1.0f) {
        return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    } else if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    } else {
        return 0.0f;
    }
}

// Compute weights and indices for one dimension
static void compute_weights_and_indices(
    int input_size,
    int output_size,
    float scale,
    bool antialias,
    std::vector<std::vector<int>>& indices,
    std::vector<std::vector<float>>& weights,
    int& kernel_size) {
    
    indices.clear();
    weights.clear();
    indices.resize(output_size);
    weights.resize(output_size);
    
    // For bicubic with anti-aliasing, we need to scale the kernel
    float kernel_scale = antialias ? std::max(1.0f / scale, 1.0f) : 1.0f;
    kernel_size = static_cast<int>(std::ceil(4.0f * kernel_scale));
    
    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        // align_corners = false mapping
        float in_idx_f = (out_idx + 0.5f) / scale - 0.5f;
        
        int left = static_cast<int>(std::floor(in_idx_f)) - kernel_size / 2 + 1;
        int right = left + kernel_size;
        
        // Clamp indices to valid range
        left = std::max(left, 0);
        right = std::min(right, input_size);
        
        // Compute weights
        float sum_weights = 0.0f;
        for (int k = left; k < right; ++k) {
            float x = (k - in_idx_f) / kernel_scale;
            float w = bicubic_kernel(x);
            
            if (std::abs(w) > 1e-6f) {
                indices[out_idx].push_back(k);
                weights[out_idx].push_back(w);
                sum_weights += w;
            }
        }
        
        // Normalize weights
        if (std::abs(sum_weights) > 1e-6f) {
            for (float& w : weights[out_idx]) {
                w /= sum_weights;
            }
        }
    }
}

// Apply 1D resampling horizontally
template<int CHANNELS>
static void apply_horizontal_resample(
    const cv::Mat& src,
    cv::Mat& dst,
    const std::vector<std::vector<int>>& indices,
    const std::vector<std::vector<float>>& weights) {
    
    int height = src.rows;
    int dst_width = dst.cols;
    
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.ptr<uint8_t>(y);
        uint8_t* dst_row = dst.ptr<uint8_t>(y);
        
        for (int x = 0; x < dst_width; ++x) {
            float sum[CHANNELS] = {0.0f};
            
            const auto& idx_vec = indices[x];
            const auto& weight_vec = weights[x];
            
            for (size_t k = 0; k < idx_vec.size(); ++k) {
                int src_x = idx_vec[k];
                float w = weight_vec[k];
                
                if (src_x >= 0 && src_x < src.cols) {
                    const uint8_t* src_pixel = src_row + src_x * CHANNELS;
                    for (int c = 0; c < CHANNELS; ++c) {
                        sum[c] += w * src_pixel[c];
                    }
                }
            }
            
            uint8_t* dst_pixel = dst_row + x * CHANNELS;
            for (int c = 0; c < CHANNELS; ++c) {
                dst_pixel[c] = static_cast<uint8_t>(std::clamp(std::round(sum[c]), 0.0f, 255.0f));
            }
        }
    }
}

// Apply 1D resampling vertically
template<int CHANNELS>
static void apply_vertical_resample(
    const cv::Mat& src,
    cv::Mat& dst,
    const std::vector<std::vector<int>>& indices,
    const std::vector<std::vector<float>>& weights) {
    
    int width = src.cols;
    int dst_height = dst.rows;
    
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < dst_height; ++y) {
            float sum[CHANNELS] = {0.0f};
            
            const auto& idx_vec = indices[y];
            const auto& weight_vec = weights[y];
            
            for (size_t k = 0; k < idx_vec.size(); ++k) {
                int src_y = idx_vec[k];
                float w = weight_vec[k];
                
                if (src_y >= 0 && src_y < src.rows) {
                    const uint8_t* src_pixel = src.ptr<uint8_t>(src_y) + x * CHANNELS;
                    for (int c = 0; c < CHANNELS; ++c) {
                        sum[c] += w * src_pixel[c];
                    }
                }
            }
            
            uint8_t* dst_pixel = dst.ptr<uint8_t>(y) + x * CHANNELS;
            for (int c = 0; c < CHANNELS; ++c) {
                dst_pixel[c] = static_cast<uint8_t>(std::clamp(std::round(sum[c]), 0.0f, 255.0f));
            }
        }
    }
}

// Main resize function
void resize_bicubic_kernel(const cv::Mat img, cv::Mat &resized_img, int new_height, int new_width) {
    // Check if no resize needed
    if (img.cols == new_width && img.rows == new_height) {
        img.copyTo(resized_img);
        return;
    }
    
    // Validate input
    if (img.empty()) {
        return;
    }
    
    int channels = img.channels();
    CV_Assert(channels == 1 || channels == 3 || channels == 4);
    
    // Convert to uint8 if needed
    cv::Mat src;
    if (img.depth() != CV_8U) {
        img.convertTo(src, CV_8U);
    } else {
        src = img;
    }
    
    // Compute scales
    float scale_x = static_cast<float>(new_width) / img.cols;
    float scale_y = static_cast<float>(new_height) / img.rows;
    
    bool need_horizontal = new_width != img.cols;
    bool need_vertical = new_height != img.rows;
    
    // Prepare for resampling
    std::vector<std::vector<int>> horiz_indices, vert_indices;
    std::vector<std::vector<float>> horiz_weights, vert_weights;
    int ksize_horiz = 0, ksize_vert = 0;
    
    if (need_horizontal) {
        compute_weights_and_indices(
            img.cols, new_width, scale_x, true,
            horiz_indices, horiz_weights, ksize_horiz
        );
    }
    
    if (need_vertical) {
        compute_weights_and_indices(
            img.rows, new_height, scale_y, true,
            vert_indices, vert_weights, ksize_vert
        );
    }
    
    // Intermediate buffers
    cv::Mat temp_horiz, temp_vert;
    
    // Apply resampling
    if (need_horizontal && need_vertical) {
        // Two-pass: horizontal then vertical
        if (channels == 1) {
            temp_horiz.create(img.rows, new_width, CV_8UC1);
            apply_horizontal_resample<1>(src, temp_horiz, horiz_indices, horiz_weights);
            
            resized_img.create(new_height, new_width, CV_8UC1);
            apply_vertical_resample<1>(temp_horiz, resized_img, vert_indices, vert_weights);
        } else if (channels == 3) {
            temp_horiz.create(img.rows, new_width, CV_8UC3);
            apply_horizontal_resample<3>(src, temp_horiz, horiz_indices, horiz_weights);
            
            resized_img.create(new_height, new_width, CV_8UC3);
            apply_vertical_resample<3>(temp_horiz, resized_img, vert_indices, vert_weights);
        } else { // channels == 4
            temp_horiz.create(img.rows, new_width, CV_8UC4);
            apply_horizontal_resample<4>(src, temp_horiz, horiz_indices, horiz_weights);
            
            resized_img.create(new_height, new_width, CV_8UC4);
            apply_vertical_resample<4>(temp_horiz, resized_img, vert_indices, vert_weights);
        }
    } else if (need_horizontal) {
        // Horizontal only
        resized_img.create(img.rows, new_width, img.type());
        if (channels == 1) {
            apply_horizontal_resample<1>(src, resized_img, horiz_indices, horiz_weights);
        } else if (channels == 3) {
            apply_horizontal_resample<3>(src, resized_img, horiz_indices, horiz_weights);
        } else {
            apply_horizontal_resample<4>(src, resized_img, horiz_indices, horiz_weights);
        }
    } else if (need_vertical) {
        // Vertical only
        resized_img.create(new_height, img.cols, img.type());
        if (channels == 1) {
            apply_vertical_resample<1>(src, resized_img, vert_indices, vert_weights);
        } else if (channels == 3) {
            apply_vertical_resample<3>(src, resized_img, vert_indices, vert_weights);
        } else {
            apply_vertical_resample<4>(src, resized_img, vert_indices, vert_weights);
        }
    }
}

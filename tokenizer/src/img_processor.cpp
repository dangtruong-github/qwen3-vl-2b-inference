#include "../include/img_processor.hpp"

void read_img_size(const char *img_path, int *height, int *width) {
    if (img_path == nullptr) {
        std::cerr << "Error: image file not found: " << std::endl;
        *height = 0;
        *width = 0;
        return;
    }

    // Read the image using OpenCV
    cv::Mat img = cv::imread(img_path);

    if (img.empty()) {
        std::cerr << "Error: Could not read the image file: " << img_path << std::endl;
        *height = 0;
        *width = 0;
        return;
    }

    // Assign height and width
    *height = img.rows;
    *width = img.cols;
}

static inline double round_bankers(double x) {
    double r = floor(x + 0.5);
    if (fabs(x - (r - 0.5)) < 1e-9) {
        // exactly halfway: round to even
        if (fmod(r, 2.0) != 0.0)
            r -= 1.0;
    }
    return r;
}

void smart_resize_qwen3(
    int height, int width, int *h_bar, int *w_bar, int factor,
    long long min_pixels, long long max_pixels
) {
    // const double EPS = 1e-9;

    // 1. Check aspect ratio
    double aspect_ratio = (height > width)
                              ? (double)height / width
                              : (double)width / height;
    if (aspect_ratio > 200.0) {
        fprintf(stderr, "Error: aspect ratio %.2f exceeds 200.0\n", aspect_ratio);
        exit(EXIT_FAILURE);
    }

    // 2. Round to nearest multiple of factor (banker's rounding)
    double h_div = (double)height / factor;
    double w_div = (double)width / factor;
    *h_bar = (int)(round_bankers(h_div) * factor);
    *w_bar = (int)(round_bankers(w_div) * factor);

    long long pixels = (long long)(*h_bar) * (*w_bar);

    // 3. Adjust if area is too large
    if (pixels > max_pixels) {
        double beta = sqrt((double)(height * width) / (double)max_pixels);
        double new_h = (height / beta) / factor;
        double new_w = (width / beta) / factor;
        *h_bar = (int)fmax(factor, floor(new_h) * factor);
        *w_bar = (int)fmax(factor, floor(new_w) * factor);
    }

    // 4. Adjust if area is too small
    else if (pixels < min_pixels) {
        double beta = sqrt((double)min_pixels / (double)(height * width));
        double new_h = (height * beta) / factor;
        double new_w = (width * beta) / factor;
        *h_bar = (int)(ceil(new_h) * factor);
        *w_bar = (int)(ceil(new_w) * factor);
    }
}

int get_num_img_pad(const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels) {
    int height, width, h_bar, w_bar;

    // TODO: Implement your OpenCV read_img_size(img_path, &height, &width)
    read_img_size(img_path, &height, &width);

    if (height <= 0 || width <= 0) {
        fprintf(stderr, "Invalid image dimensions for %s\n", img_path);
        return 0;
    }

    const int factor = patch_size * merge_size;  // 32

    smart_resize_qwen3(height, width, &h_bar, &w_bar, factor, min_pixels, max_pixels);

    int grid_h = h_bar / patch_size;
    int grid_w = w_bar / patch_size;

    int num_patches = grid_h * grid_w;
    int num_image_tokens = num_patches / (merge_size * merge_size);

    printf("Image: %s\n", img_path);
    printf("Orig HxW = %dx%d → Resized = %dx%d\n", height, width, h_bar, w_bar);
    printf("Grid = %dx%d → Num patches = %d → Num image tokens = %d\n",
           grid_h, grid_w, num_patches, num_image_tokens);

    return num_image_tokens;
}

void normalize_inplace(
    float *img, const double *mean, const double *std,
    size_t C, size_t H, size_t W
) {
    for (size_t c = 0; c < C; c++) {
        float cur_mean = mean[c];
        float cur_std = std[c];

        #pragma omp parallel for collapse(2)
        for (size_t y = 0; y < H; y++) {
            for (size_t x = 0; x < W; x++) {
                size_t stride = c * H * W + y * W;
                img[stride + x] = (img[stride + x] - cur_mean) / cur_std;
            }
        }
    }
}

// original dims = D0,D1,D2,D3,D4,D5,D6,D7
// permute order = 2,5,3,6,1,0,4,7 (new dims)
void permute_8d(
    float* out, const float* in, int D0, int D1, int D2,
    int D3, int D4, int D5, int D6, int D7
) {
    const int dims[8] = {D0, D1, D2, D3, D4, D5, D6, D7};
    const int P[8]   = {2, 5, 3, 6, 1, 0, 4, 7};

    int stride[8], new_stride[8];
    stride[7] = 1;
    for (int i=6;i>=0;i--) stride[i] = stride[i+1] * dims[i+1];

    int new_dims[8];
    for (int i=0;i<8;i++) new_dims[i] = dims[P[i]];

    new_stride[7] = 1;
    for (int i=6;i>=0;i--) new_stride[i] = new_stride[i+1] * new_dims[i+1];

    long total = 1L*D0*D1*D2*D3*D4*D5*D6*D7;

    #pragma omp parallel for
    for (long idx=0; idx<total; idx++) {
        int tmp = idx, coord[8]={0};

        // Convert flat -> multi-index (new permuted shape)
        for (int i=0; i<8; i++){
            coord[i] = tmp / new_stride[i];
            tmp     %= new_stride[i];
        }

        // Map back to original coordinates
        long old_idx = 0;
        for (int i=0; i<8; i++)
            old_idx += coord[i] * stride[P[i]];

        out[idx] = in[old_idx];
    }
}

void duplicate_to_batch(
    const float* img,   // CHW, size = C * H * W
    float* batch,       // NCHW, size = B * C * H * W
    int B, int C, int H, int W
) {
    const size_t HW = (size_t)H * W;
    const size_t CHW = (size_t)C * HW;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; ++b) {

        // CHW → CHW (just copy)
        for (int c = 0; c < C; ++c) {
            float* batch_b = batch + (size_t)b * CHW;
            const float* src_c = img + (size_t)c * HW;
            float* dst_c = batch_b + (size_t)c * HW;

            memcpy(dst_c, src_c, HW * sizeof(float));
        }
    }
}

bool str_none(const char *str_to_check) {
    // Skip if "none"
    if (strcmp(str_to_check, "none") == 0 || strcmp(str_to_check, "NONE") == 0 || strcmp(str_to_check, "None") == 0) {
        return true;
    }
    if (strcmp(str_to_check, "none\n") == 0 || strcmp(str_to_check, "NONE\n") == 0 || strcmp(str_to_check, "None\n") == 0) {
        return true;
    }

    return false;
}

void resize_bicubic(const cv::Mat img, cv::Mat &resized_img, int new_height, int new_width) {
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    if (img.type() != CV_8UC1 &&
        img.type() != CV_8UC3 &&
        img.type() != CV_8UC4) {
        throw std::runtime_error("Only CV_8UC1 / CV_8UC3 / CV_8UC4 supported");
    }

    if (new_height <= 0 || new_width <= 0) {
        throw std::runtime_error("Invalid target size");
    }

    // Bicubic interpolation
    cv::resize(img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
}

void print_cv_img(const cv::Mat *img) {
    if (!img || img->empty()) {
        printf("Invalid or empty image\n");
        return;
    }

    CV_Assert(img->isContinuous());
    CV_Assert(img->type() == CV_32FC3);

    const int H = img->rows;
    const int W = img->cols;
    const int C = img->channels();  // should be 3

    const float* data = (const float*)img->data;

    // OpenCV layout: HWC
    // index = (h * W + w) * C + c

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                size_t idx = ((size_t)h * W + w) * C + c;
                printf("%.2f ", data[idx]);
            }
            printf("\n");
        }
    }
}

void hwc_to_chw(const cv::Mat& hwc_img, float* chw_out) {
    CV_Assert(hwc_img.type() == CV_32FC3);
    CV_Assert(hwc_img.isContinuous());

    const int H = hwc_img.rows;
    const int W = hwc_img.cols;
    const int C = 3;

    const float* hwc = (const float*)hwc_img.data;

    // HWC index: (h * W + w) * C + c
    // CHW index: c * H * W + h * W + w
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                chw_out[c * H * W + h * W + w] =
                    hwc[(h * W + w) * C + c];
            }
        }
    }
}

bool image_processor(
    const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels, float **out_data, int *out_h, int *out_w,
    int *out_grid_h, int *out_grid_w 
) {
    if (str_none(img_path)) {
        printf("Skipping 'none' image path.\n");
        return false;
    }

    int orig_height, orig_width, h_bar, w_bar;

    printf("img_path=%s\n", img_path);
    read_img_size(img_path, &orig_height, &orig_width);

    const int factor = patch_size * merge_size;  // 32
    smart_resize_qwen3(orig_height, orig_width, &h_bar, &w_bar, factor, min_pixels, max_pixels);
    printf("orig_height=%d, orig_width=%d, h_bar=%d, w_bar=%d\n", orig_height, orig_width, h_bar, w_bar);

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat resized_img;

    // 🔴 IMPORTANT
    if (!resized_img.isContinuous()) {
        resized_img = resized_img.clone();
    }
    if (!img.isContinuous()) {
        img = img.clone();
    }

    resize_bicubic_kernel(img, resized_img, h_bar, w_bar);

    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32FC3);

    if (!resized_img.isContinuous()) {
        resized_img = resized_img.clone();
    }

    // print_cv_img(&resized_img);
    // fflush(stdout);
    // exit(1);

    // Normalization parameters
    const double mean[3] = {127.5000, 127.5000, 127.5000};
    const double std[3]  = {127.5000, 127.5000, 127.5000};

    int temporal_patch_size = 2;
    int channels = 3;

    float *resized_img_ptr = (float *)malloc(sizeof(float) * 1ll * channels * h_bar * w_bar);
    hwc_to_chw(resized_img, resized_img_ptr);

    normalize_inplace(resized_img_ptr, mean, std, channels, h_bar, w_bar);

    float *batch_img = (float *)malloc(sizeof(float) * 1ll * temporal_patch_size * channels * h_bar * w_bar);

    duplicate_to_batch(resized_img_ptr, batch_img, temporal_patch_size, channels, h_bar, w_bar);

    int grid_h = h_bar / patch_size;
    int grid_w = w_bar / patch_size;

    *out_data = (float *)malloc(
        sizeof(float) * 1ll * temporal_patch_size * channels *
        (grid_h / merge_size) * merge_size * patch_size *
        (grid_w / merge_size) * merge_size * patch_size
    );

    permute_8d(
        *out_data,
        batch_img,
        temporal_patch_size,
        channels,
        grid_h / merge_size,
        merge_size,
        patch_size,
        grid_w / merge_size,
        merge_size,
        patch_size
    );

    int final_h = grid_h * grid_w;
    int final_w = temporal_patch_size * channels * patch_size * patch_size;
    
    *out_h = final_h;
    *out_w = final_w;
    *out_grid_h = grid_h;
    *out_grid_w = grid_w;
    free(batch_img);
    return true;
}

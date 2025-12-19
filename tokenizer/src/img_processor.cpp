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

void normalize_inplace(cv::Mat &img, const double *mean, const double *std, int C) {

    // Convert image to float inplace if needed
    if (img.type() != CV_32FC3) {
        img.convertTo(img, CV_32FC3);
    }

    int H = img.rows;
    int W = img.cols;

    for (int y = 0; y < H; y++) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                row[x][c] = (row[x][c] - mean[c]) / std[c];
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

void duplicate_to_batch(const cv::Mat& img, float* batch, int B, int C, int H, int W) {
    if (img.rows != H || img.cols != W || img.channels() != C) {
        printf("Error: input image dimensions do not match HWC parameters\n");
        return;
    }

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                const cv::Vec3f* row_ptr = img.ptr<cv::Vec3f>(i);
                float* dst = batch + b * C * H * W + c * H * W + i * W;
                for (int j = 0; j < W; j++) {
                    dst[j] = row_ptr[j][c];
                }
            }
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

bool image_processor(
    const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels, float **out_data, int *out_h, int *out_w,
    int *out_grid_h, int *out_grid_w 
) {
    if (str_none(img_path)) {
        printf("Skipping 'none' image path.\n");
        return false;
    }

    int orig_height, orig_width, h_bar, w_bar;
    printf("Start reading %s\n", img_path);
    fflush(stdout);

    read_img_size(img_path, &orig_height, &orig_width);

    printf("Finish read_img_size: %d %d\n", orig_height, orig_width);
    fflush(stdout);

    const int factor = patch_size * merge_size;  // 32
    smart_resize_qwen3(orig_height, orig_width, &h_bar, &w_bar, factor, min_pixels, max_pixels);

    printf("Finish smart_resize_qwen3: %d %d\n", h_bar, w_bar);
    fflush(stdout);

    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat resized_img;

    resize_bicubic(img, resized_img, h_bar, w_bar);
    img.convertTo(img, CV_32FC3);

    /*
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < h_bar; i++) {
            for (int j = 0; j < w_bar; j++) {
                float pixel_value = resized_img.at<cv::Vec3f>(i, j)[c];
                printf("%.2f ", pixel_value);
            }
            printf("\n");
        }
        printf("\n");
    }
    */

    printf("Finish resize_bicubic: %d %d\n", h_bar, w_bar);
    fflush(stdout);

    // Normalization parameters
    const double mean[3] = {127.5000, 127.5000, 127.5000};
    const double std[3]  = {127.5000, 127.5000, 127.5000};

    int temporal_patch_size = 2;
    int channels = 3;

    normalize_inplace(resized_img, mean, std, channels);

    /*
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < h_bar; i++) {
            for (int j = 0; j < w_bar; j++) {
                float pixel_value = resized_img.at<cv::Vec3f>(i, j)[c];
                printf("%.2f ", pixel_value);
            }
            printf("\n");
        }
        printf("\n");
    }
    */

    printf("Finish normalize_inplace: %d %d\n", h_bar, w_bar);
    fflush(stdout);

    float *batch_img = (float *)malloc(sizeof(float) * 1ll * temporal_patch_size * channels * h_bar * w_bar);

    duplicate_to_batch(resized_img, batch_img, temporal_patch_size, channels, h_bar, w_bar);

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

    printf("Finish permute_8d: %d %d\n", h_bar, w_bar);
    fflush(stdout);

    int final_h = grid_h * grid_w;
    int final_w = temporal_patch_size * channels * patch_size * patch_size;

    printf("final_h: %d, final_w: %d\n", final_h, final_w);
    fflush(stdout);

    /*
    for (int i = 0; i < final_h; i++) {
        for (int j = 0; j < final_w; j++) {
            printf("%.2f ", out_data[i * final_w + j]);
        }
        printf("\n");
    }
    */
    
    *out_h = final_h;
    *out_w = final_w;
    *out_grid_h = grid_h;
    *out_grid_w = grid_w;
    free(batch_img);
    return true;
}

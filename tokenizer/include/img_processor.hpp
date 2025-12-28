#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>   // for printf()
#include <stdlib.h>  // for exit()
#include "resize.hpp"

int get_num_img_pad(const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels);
bool image_processor(
    const char *img_path, int patch_size, int merge_size, long long min_pixels, long long max_pixels, float **out_data, int *out_h, int *out_w,
    int *out_grid_h, int *out_grid_w 
);
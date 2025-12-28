#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

void resize_bicubic_kernel(const cv::Mat img, cv::Mat &resized_img, int new_height, int new_width);
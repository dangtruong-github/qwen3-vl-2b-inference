#pragma once


#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

void resize_bicubic(const cv::Mat img, cv::Mat &resized_img, int new_height, int new_width);
void resize_bicubic_torchvision(
    const cv::Mat img, cv::Mat &resized_img,
    int new_height, int new_width
);

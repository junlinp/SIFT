#ifndef SIFT_H_
#define SIFT_H_
#include "opencv2/core.h"

void GaussianBlur(const cv::Mat& gray_input_image, double sigma, cv::Mat& gray_output_image);
#endif // SIFT_H_
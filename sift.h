#ifndef SIFT_H_
#define SIFT_H_
#include "opencv2/core.hpp"

int symmetrization_functor(int value, int value_bound);
void GaussianBlur(const cv::Mat& gray_input_image, double sigma, cv::Mat& gray_output_image);
#endif // SIFT_H_
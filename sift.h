#ifndef SIFT_H_
#define SIFT_H_
#include "opencv2/core.hpp"

int symmetrization_functor(int value, int value_bound);
void GaussianBlur(const cv::Mat& gray_input_image, double sigma,
                  cv::Mat& gray_output_image);

void BilinearInterpolationImage(cv::Mat& input_image, cv::Mat& output_image, double sample_distance);

void GaussianPyramid(const cv::Mat& input_image,
                     std::vector<cv::Mat>& gaussian_pyramid, int num_octaves,
                     int layers_per_octave, double init_blur_sigma);
#endif  // SIFT_H_
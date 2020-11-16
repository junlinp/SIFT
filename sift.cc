#include "sift.h"

int symmetrization_functor(int value, int value_bound) {
  return std::min(std::max(value, -1 - value), 2 * value_bound - 1 - value);
}

void GaussianBlur(const cv::Mat& gray_input_image, double sigma,
                  cv::Mat& gray_output_image) {
  gray_output_image = gray_input_image.clone();
  int r_ = gray_input_image.rows;
  int c_ = gray_input_image.cols;
  cv::Mat Temp(r_, c_, CV_32FC1);

  int gaussian_range = static_cast<int>(4.0 * sigma);

  std::vector<double> gaussian_value(gaussian_range + 1);
  double sum = 0.0;
  for (int i = 0; i <= gaussian_range; i++) {
    double v = -i * i / 2.0 / sigma / sigma;
    gaussian_value[i] = std::exp(v);
    sum += (i == 0 ? gaussian_value[i] : 2 * gaussian_value[i]);
  }
  for (double& i : gaussian_value) {
    i /= sum;
  }

  for (int row = 0; row < gray_input_image.rows; row++) {
    for (int col = 0; col < gray_input_image.cols; col++) {
      double value = 0.0;
      for (int k = -gaussian_range; k <= gaussian_range; k++) {
        int c = symmetrization_functor(col + k, c_);
        uchar pixel_value = gray_input_image.at<uchar>(row, c);
        value += pixel_value * gaussian_value[std::abs(k)];
      }
      Temp.at<float>(row, col) = value;
    }
  }

  for (int col = 0; col < gray_input_image.cols; col++) {
    for (int row = 0; row < gray_input_image.rows; row++) {
      double value = 0.0;
      for (int k = -gaussian_range; k <= gaussian_range; k++) {
        int r = symmetrization_functor(row + k, r_);
        float pixel_value = Temp.at<float>(r, col);
        double v = pixel_value * gaussian_value[std::abs(k)];
        value += v;
      }
      gray_output_image.at<uchar>(row, col) = static_cast<uchar>(value);
    }
  }
}
/*
void BilinearInterpolationImage(cv::Mat& input_image, cv::Mat& output_image,
                                double sample_distance) {
  int sample_row = static_cast<int>(input_image.rows / sample_distance);
  int sample_col = static_cast<int>(input_image.cols / sample_distance);
  output_image = cv::Mat(sample_row, sample_col, CV_8UC1);
  for (int row = 0; row < sample_row; row++) {
    for (int col = 0; col < sample_col; col++) {
      double unsample_row = row * sample_distance;
      double unsample_col = col * sample_distance;
      output_image.at<uchar>(row, col) = 
        (unsample_row - std::floor(unsample_row)) * ( (unsample_col - std::floor(unsample_col)) * input_image.at<uchar>(symmetrization_functor(std::ceil(unsample_row), input_image.rows), symmetrization_functor(std::ceil(unsample_col))) + (std::ceil(unsample_col) - unsample_col) * input_image.at<uchar>(symmetrization_functor(std::ceil(unsample_row)), symmetrization_functor(std::floor(unsample_col)))) +
        (std::ceil(unsample_row) - unsample_row) * ( (unsample_col - std::floor(unsample_col)) * u(std::floor(unsample_row), std::ceil(unsample_col)) + (std::ceil(unsample_col) - unsample_col) * u(std::floor(unsample_row), std::floor(unsample_col));
    }
  }
}
*/
uchar BilinearInterpolationImage(const cv::Mat& input_image, double row, double col) {
  size_t low_row = size_t(row);
  size_t low_col = size_t(col);
  size_t max_row = input_image.rows;
  size_t max_col = input_image.cols;
  size_t high_row = std::min(low_row + 1, max_row);
  size_t high_col = std::min(low_col + 1, max_col);

  uchar low_low_value = input_image.at<uchar>(low_row, low_col);
  uchar low_high_value = input_image.at<uchar>(low_row, high_col);
  uchar high_low_value = input_image.at<uchar>(high_row, low_col);
  uchar high_high_value = input_image.at<uchar>(high_row, high_col);
  
  float low_value = (row - low_row) * (high_low_value - low_low_value);
  float high_value = (row - low_row) * (high_high_value - low_high_value);
  float value = (col - low_col) * (high_value - low_value);
  
  return static_cast<uchar>(value);
}

void GaussianPyramid(const cv::Mat& input_image,
                     std::vector<cv::Mat>& gaussian_pyramid, int num_octaves,
                     int layers_per_octave, double init_blur_sigma) {
  int scale_up_width = 2 * input_image.cols;
  int scale_up_height = 2 * input_image.rows;
  cv::Mat scale_image(scale_up_height, scale_up_width, CV_8UC1);

  for (int row = 0; row < scale_up_height; row++) {
    for (int col = 0; col < scale_up_width; col++) {
      scale_image.at<uchar>(row, col) = BilinearInterpolationImage(input_image, row * 0.5, col * 0.5);
    }
  }
}

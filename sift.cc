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
        value +=  v;
      }
      gray_output_image.at<uchar>(row, col) = static_cast<uchar>(value);
    }
  }
}
#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sift.h"
TEST(sym, t) {
    int row = 16;

    for (int i = -row; i < 0; i++) {
        EXPECT_EQ(-1 - i, symmetrization_functor(i, row));
    }

    for (int i = 0; i < row; i++) {
        EXPECT_EQ(i, symmetrization_functor(i, row));
    }

    for (int i = row; i < 2 * row; i++) {
        EXPECT_EQ(2 * row - 1 - i, symmetrization_functor(i, row));
    }
}

TEST(Gaussian, Blur) {
    std::string file_path = std::string(Test_DIR) + "/a.jpg";
    cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    cv::imwrite(std::string(Test_DIR) + "/a_gray.jpg", img);

    cv::Mat my_output;
    GaussianBlur(img, 1.6, my_output);

    cv::imwrite(std::string(Test_DIR) + "/a_blur.jpg", my_output);

    cv::Mat opencv_output;
    
    cv::GaussianBlur(img, opencv_output, cv::Size(0, 0), 1.6, 1.6);

    cv::imwrite(std::string(Test_DIR) + "/a_blur_opencv.jpg", opencv_output);

}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
#include "sift.h"

#include "gtest/gtest.h"
#include "matcher.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class TimeUse {
 public:
  TimeUse() { start_ = std::chrono::high_resolution_clock::now(); }
  ~TimeUse() {
    std::chrono::milliseconds duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_);
    std::cout << "Time excaped :" << duration.count() << " milliseconds"
              << std::endl;
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

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
#include <random>
TEST(Match, match) {
  std::vector<Descriptor> lhs;
  std::vector<int> shuffle;
  int n = 1024 * 16;
  std::default_random_engine engine;
  std::uniform_real_distribution<float> uniform_d(0, 255);
  for (int i = 0; i < n; i++) {
    Descriptor d;
    for (int k = 0; k < 128; k++) {
      d[k] = uniform_d(engine);
    }
    lhs.push_back(d);
    shuffle.push_back(i);
  };
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffle.begin(), shuffle.end(), engine);
  std::vector<Descriptor> rhs;
  for (int i = 0; i < n; i++) {
    rhs.push_back(lhs[shuffle[i]]);
  }

  std::vector<int> match_idx;
  {
    TimeUse t;
    match6(rhs, lhs, match_idx);
    std::cout << "match 6 : " << std::endl;
  }
  {
    TimeUse t;
    match5(rhs, lhs, match_idx);
    std::cout << "match 5 : " << std::endl;
  }
  {
    TimeUse t;
    match3(rhs, lhs, match_idx);
    std::cout << "match 3 : " << std::endl;
  }

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(match_idx[i], shuffle[i]);
  }
}
int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
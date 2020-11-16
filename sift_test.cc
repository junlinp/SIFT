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
  int n = 128;
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
    match5(rhs, lhs, match_idx);
    std::cout << "match 5 : " << std::endl;
  }
  {
    TimeUse t;
    //match3(rhs, lhs, match_idx);
    std::cout << "match 3 : " << std::endl;
  }
{
    TimeUse t;
    match6(rhs, lhs, match_idx);
    std::cout << "match 6 : " << std::endl;
  }

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(match_idx[i], shuffle[i]);
  }
}


TEST(AVX, SCALR) {

    std::cout << "malloc " << std::endl;
    int count = 128 / 8;
    int count2 = count * 8;
    float* ptr_a = (float*) _mm_malloc(count * 8 * sizeof(float), 32);
    float* ptr_b = (float*) _mm_malloc(count * 8 * sizeof(float), 32);
    float* ptr_avx_output = (float*) _mm_malloc(count * 8 * sizeof(float), 32);
    float* ptr_cpu_output = (float*) _mm_malloc(count * 8 * sizeof(float), 32);
    float* temp = (float*) _mm_malloc(8 * sizeof(float), 32);
    std::cout << "init data " << std::endl;

    float all_sum = 0.0;
    for (int i = 0; i < count * 8 ; i++) {
        ptr_a[i] = i;
        ptr_b[i] = 2 * i;
        ptr_cpu_output[i] = i * i; 
        ptr_avx_output[i] = 0;
        all_sum += ptr_cpu_output[i];
    }
    {
        TimeUse t;
    // ymm1
    // 0 1 2 3 4 5  6  7 
    // ymm2
    // 0 2 4 6 8 10 12 14
    // ymm3
    // 0 1 2 3 4 5  6  7 
    // 0 1 4 9 16 25  36  49
    asm (
        "vxorps %%ymm5, %%ymm5, %%ymm5;"
        "lea (%%rax), %%r8;"
        "lea (%%rbx), %%r9;"
        "lea (%%rcx), %%r10;"
        //"lea 0x100(%%r8), %%r8;"
        //"lea 0x100(%%r9), %%r9;"
        //"lea 0x100(%%r10), %%r10;"
        "loop_:\t vmovups (%%r8), %%ymm1;"
        "vmovups (%%r9), %%ymm2;"
        "vsubps %%ymm1, %%ymm2, %%ymm3;"
        "vmulps %%ymm3, %%ymm3, %%ymm4;"
        "vaddps %%ymm4, %%ymm5, %%ymm5;"
        "vmovups %%ymm4, (%%r10);"
        //"loop_:\t flds (%%r8);"
        //"flds (%%r9);"
        //"faddp;"
        //"fstps (%%r10);"
        "lea 0x20(%%r8), %%r8;"
        "lea 0x20(%%r9), %%r9;"
        "lea 0x20(%%r10), %%r10;"
        "decl %0;"
        "jnz loop_;"
        "vmovups %%ymm5, (%4);"
        "flds (%4);"
        "flds 0x4(%4);"
        "flds 0x8(%4);"
        "flds 0xC(%4);"
        "flds 0x10(%4);"
        "flds 0x14(%4);"
        "flds 0x18(%4);"
        "flds 0x1C(%4);"
        "faddp;"
        "faddp;"
        "faddp;"
        "faddp;"
        "faddp;"
        "faddp;"
        "faddp;"
        "fstps (%4);"
         : 
         : "r"(count), "a" (ptr_a), "b" (ptr_b), "c" (ptr_avx_output), "r" (temp)
         : "%ymm1", "%ymm2", "%ymm3", "%ymm4","%ymm5", "%r8", "%r9", "%r10"
    );
    }
    
    for(int i = 0; i < count2; i++) {
        EXPECT_EQ(ptr_cpu_output[i], ptr_avx_output[i]);
        //std::cout << ptr_avx_output[i] << std::endl;
    }
    EXPECT_EQ(temp[0], all_sum);
}
int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
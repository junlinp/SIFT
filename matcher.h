
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  ThreadPool(size_t max_thread_num = 1)
      : max_thread_num_(max_thread_num), b_stop(false) {
    auto thread_base_task = [this]() {
      while (true) {
        std::function<void(void)> task;
        {
          std::unique_lock<std::mutex> ulk(mutex_);
          cv_.wait(ulk, [this]() { return b_stop || !tasks.empty(); });

          if (b_stop && tasks.empty()) {
            return;
          }
          task = std::move(tasks.front());
          tasks.pop();
        }
        task();
      }
    };
    for (size_t i = 0; i < max_thread_num_; i++) {
      works.emplace_back(thread_base_task);
    }
  }
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> ulk(mutex_);
      b_stop = true;
      cv_.notify_all();
    }
    for (auto& thread : works) {
      thread.join();
    }
  }
  template <class Functor, class... Args,
            class ReturnType =
                std::result_of_t<std::remove_reference_t<Functor>(Args...)>>
  std::future<ReturnType> Enqueue(Functor&& functor, Args&&... args) {
    auto f =
        std::bind(std::forward<Functor>(functor), std::forward<Args>(args)...);
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(f);
    auto future = task->get_future();
    {
      std::unique_lock<std::mutex> ulk(mutex_);
      if (b_stop) {
        throw std::runtime_error("ThreadPool has stopped.");
      }
      tasks.push([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

  template <class Functor, class ReturnType = std::result_of_t<
                               std::remove_reference_t<Functor>()>>
  std::future<ReturnType> Enqueue(Functor&& functor) {
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(functor);
    auto future = task->get_future();
    {
      std::unique_lock<std::mutex> ulk(mutex_);
      if (b_stop) {
        throw std::runtime_error("ThreadPool has stopped.");
      }
      tasks.push([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

 private:
  size_t max_thread_num_;
  std::queue<std::function<void(void)>> tasks;
  std::vector<std::thread> works;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool b_stop;
};

using Descriptor = std::array<float, 128>;

void match(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
           std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    std::vector<std::pair<int, double>> s;
    for (int j = 0; j < rhs.size(); j++) {
      double sum = 0.0;
      for (int k = 0; k < 128; k++) {
        double d = lhs[i][k] - rhs[j][k];
        sum += d * d;
      }
      s.push_back({j, sum});
    }

    std::sort(s.begin(), s.end(),
              [](auto& lhs, auto& rhs) { return lhs.second < rhs.second; });

    if (s[0].second < 0.6 * s[1].second) {
      match_idx[i] = s[0].first;
    } else {
      match_idx[i] = -1;
    }
  }
}

void match1(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());
  for (int i = 0; i < lhs.size(); i++) {
    std::vector<std::pair<int, double>> s;
    s.reserve(rhs.size());
    for (int j = 0; j < rhs.size(); j++) {
      double sum = 0.0;
      for (int k = 0; k < 128; k++) {
        double d = lhs[i][k] - rhs[j][k];
        sum += d * d;
      }
      s.push_back({j, sum});
    }

    std::sort(s.begin(), s.end(),
              [](auto& lhs, auto& rhs) { return lhs.second < rhs.second; });

    if (s[0].second < 0.6 * s[1].second) {
      match_idx[i] = s[0].first;
    } else {
      match_idx[i] = -1;
    }
  }
}

void match2(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());

  std::vector<int> min_index(lhs.size());
  std::vector<double> min_distance(lhs.size());
  std::vector<int> second_min_index(lhs.size());
  std::vector<double> second_min_distance(lhs.size());

  for (int i = 0; i < lhs.size(); i++) {
    min_distance[i] = 1e300;
    min_index[i] = -1;
    second_min_distance[i] = 1e300;
    second_min_index[i] = -1;
    for (int j = 0; j < rhs.size(); j++) {
      double sum = 0.0;

      for (int k = 0; k < 128; k++) {
        double d = lhs[i][k] - rhs[j][k];
        sum += d * d;
      }

      if (sum < second_min_distance[i]) {
        second_min_distance[i] = sum;
        second_min_index[i] = j;
        if (second_min_distance[i] < min_distance[i]) {
          std::swap(second_min_distance[i], min_distance[i]);
          std::swap(second_min_index[i], min_index[i]);
        }
      }
    }
  }

  for (int i = 0; i < lhs.size(); i++) {
    // match_idx[i] = min_index[i];
    if (min_distance[i] < 0.6 * second_min_distance[i]) {
      match_idx[i] = min_index[i];
    } else {
      match_idx[i] = -1;
    }
  }
}

void match3(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());

  std::vector<int> min_index(lhs.size());
  std::vector<double> min_distance(lhs.size());
  std::vector<int> second_min_index(lhs.size());
  std::vector<double> second_min_distance(lhs.size());

  for (int i = 0; i < lhs.size(); i++) {
    min_distance[i] = 1e300;
    min_index[i] = -1;
    second_min_distance[i] = 1e300;
    second_min_index[i] = -1;
    float* p1_ptr = lhs[i].begin();
    for (int j = 0; j < rhs.size(); j++) {
      __m256 m_sum = _mm256_setzero_ps();
      for (int d = 0; d < 128; d += 8) {
        __m256 v1 = _mm256_load_ps(p1_ptr + d);
        __m256 v2 = _mm256_load_ps(rhs[j].begin() + d);
        __m256 sub = _mm256_sub_ps(v1, v2);
        m_sum = _mm256_fmadd_ps(sub, sub, m_sum);
      }
      m_sum = _mm256_add_ps(m_sum, _mm256_permute2f128_ps(m_sum, m_sum, 1));
      m_sum = _mm256_hadd_ps(m_sum, m_sum);
      float sum = _mm256_cvtss_f32(_mm256_hadd_ps(m_sum, m_sum));

      if (sum < second_min_distance[i]) {
        second_min_distance[i] = sum;
        second_min_index[i] = j;
        if (second_min_distance[i] < min_distance[i]) {
          std::swap(second_min_distance[i], min_distance[i]);
          std::swap(second_min_index[i], min_index[i]);
        }
      }
    }
  }

  for (int i = 0; i < lhs.size(); i++) {
    // match_idx[i] = min_index[i];
    if (min_distance[i] < 0.6 * second_min_distance[i]) {
      match_idx[i] = min_index[i];
    } else {
      match_idx[i] = -1;
    }
  }
}
void match4(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());

  std::vector<int> min_index(lhs.size());
  std::vector<double> min_distance(lhs.size());
  std::vector<int> second_min_index(lhs.size());
  std::vector<double> second_min_distance(lhs.size());

  auto functor = [&](int i) {
    min_distance[i] = 1e300;
    min_index[i] = -1;
    second_min_distance[i] = 1e300;
    second_min_index[i] = -1;
    for (int j = 0; j < rhs.size(); j++) {
      __m256 m_sum = _mm256_setzero_ps();
      for (int d = 0; d < 128; d += 8) {
        __m256 v1 = _mm256_load_ps(lhs[i].begin() + d);
        __m256 v2 = _mm256_load_ps(rhs[j].begin() + d);
        __m256 sub = _mm256_sub_ps(v1, v2);
        m_sum = _mm256_fmadd_ps(sub, sub, m_sum);
      }
      m_sum = _mm256_add_ps(m_sum, _mm256_permute2f128_ps(m_sum, m_sum, 1));
      m_sum = _mm256_hadd_ps(m_sum, m_sum);
      float sum = _mm256_cvtss_f32(_mm256_hadd_ps(m_sum, m_sum));

      if (sum < second_min_distance[i]) {
        second_min_distance[i] = sum;
        second_min_index[i] = j;
        if (second_min_distance[i] < min_distance[i]) {
          std::swap(second_min_distance[i], min_distance[i]);
          std::swap(second_min_index[i], min_index[i]);
        }
      }
    }
  };

  {
    ThreadPool thread_pool;
    for (int i = 0; i < lhs.size(); i++) thread_pool.Enqueue(functor, i);
  }

  for (int i = 0; i < lhs.size(); i++) {
    // match_idx[i] = min_index[i];
    if (min_distance[i] < 0.6 * second_min_distance[i]) {
      match_idx[i] = min_index[i];
    } else {
      match_idx[i] = -1;
    }
  }
}
void match5(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());

  std::vector<int> min_index(lhs.size());
  std::vector<double> min_distance(lhs.size());
  std::vector<int> second_min_index(lhs.size());
  std::vector<double> second_min_distance(lhs.size());

  const int BATCH_SIZE = 512;
  for (int i = 0; i < lhs.size(); i++) {
    min_distance[i] = 1e300;
    min_index[i] = -1;
    second_min_distance[i] = 1e300;
    second_min_index[i] = -1;
  }
  // assume that we have L1 cache with E cachelines.
  // meanwhile each cacheline can store B points.
  // in match3 version code.
  // it seems that we have lhs.size() / B * rhs.size() / B times cache miss.
  // but if we match the vector in a subset which can be totally store in cache
  // cache miss can be reduced to lhs.size() / (E / 2 * B) * rhs.size() / b
  // (we assume each vector in lhs and rhs fill with half of cache)
  for (int b1 = 0; b1 < lhs.size(); b1 += BATCH_SIZE) {
    for (int b2 = 0; b2 < rhs.size(); b2 += BATCH_SIZE) {
      for (int i = b1; i < b1 + BATCH_SIZE; i++) {
        float* p1_ptr = lhs[i].begin();
        for (int j = b2; j < b2 + BATCH_SIZE; j++) {
          float* p2_ptr = rhs[j].begin();
          __m256 m_sum = _mm256_setzero_ps();
          for (int d = 0; d < 128; d += 8) {
            __m256 v1 = _mm256_load_ps(p1_ptr + d);
            __m256 v2 = _mm256_load_ps(p2_ptr + d);
            __m256 sub = _mm256_sub_ps(v1, v2);
            m_sum = _mm256_fmadd_ps(sub, sub, m_sum);
          }
          m_sum = _mm256_add_ps(m_sum, _mm256_permute2f128_ps(m_sum, m_sum, 1));
          m_sum = _mm256_hadd_ps(m_sum, m_sum);
          float sum = _mm256_cvtss_f32(_mm256_hadd_ps(m_sum, m_sum));

          if (sum < second_min_distance[i]) {
            second_min_distance[i] = sum;
            second_min_index[i] = j;
            if (second_min_distance[i] < min_distance[i]) {
              std::swap(second_min_distance[i], min_distance[i]);
              std::swap(second_min_index[i], min_index[i]);
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < lhs.size(); i++) {
    // match_idx[i] = min_index[i];
    if (min_distance[i] < 0.6 * second_min_distance[i]) {
      match_idx[i] = min_index[i];
    } else {
      match_idx[i] = -1;
    }
  }
}
void match6(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs,
            std::vector<int>& match_idx) {
  match_idx.resize(lhs.size());

  std::vector<int> min_index(lhs.size());
  std::vector<double> min_distance(lhs.size());
  std::vector<int> second_min_index(lhs.size());
  std::vector<double> second_min_distance(lhs.size());

  const int BATCH_SIZE = 512;
  for (int i = 0; i < lhs.size(); i++) {
    min_distance[i] = 1e300;
    min_index[i] = -1;
    second_min_distance[i] = 1e300;
    second_min_index[i] = -1;
  }
  for (int b1 = 0; b1 < lhs.size(); b1 += BATCH_SIZE) {
    for (int b2 = 0; b2 < rhs.size(); b2 += BATCH_SIZE) {
      for (int i = b1; i < b1 + BATCH_SIZE; i++) {
        float* p1_ptr = lhs[i].begin();
        for (int j = b2; j < b2 + BATCH_SIZE; j++) {
          float* p2_ptr = rhs[j].begin();
          __m256 m_sum = _mm256_setzero_ps();
          __m256 m_sum2 = _mm256_setzero_ps();
          // unroll by hand
          
          for (int d = 0; d < 128; d += 16) {
            __m256 v1 = _mm256_load_ps(p1_ptr + d);
            __m256 v2 = _mm256_load_ps(p2_ptr + d);
            __m256 sub = _mm256_sub_ps(v1, v2);
            m_sum = _mm256_fmadd_ps(sub, sub, m_sum);
            __m256 v3 = _mm256_load_ps(p1_ptr + d + 8);
            __m256 v4 = _mm256_load_ps(p2_ptr + d + 8);
            __m256 sub2 = _mm256_sub_ps(v3, v4); 
            m_sum2 = _mm256_fmadd_ps(sub2, sub2, m_sum2);
          }
          m_sum = _mm256_add_ps(m_sum, m_sum2);
          m_sum = _mm256_add_ps(m_sum, _mm256_permute2f128_ps(m_sum, m_sum, 1));
          m_sum = _mm256_hadd_ps(m_sum, m_sum);
          float sum = _mm256_cvtss_f32(_mm256_hadd_ps(m_sum, m_sum));

          if (sum < second_min_distance[i]) {
            second_min_distance[i] = sum;
            second_min_index[i] = j;
            if (second_min_distance[i] < min_distance[i]) {
              std::swap(second_min_distance[i], min_distance[i]);
              std::swap(second_min_index[i], min_index[i]);
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < lhs.size(); i++) {
    // match_idx[i] = min_index[i];
    if (min_distance[i] < 0.6 * second_min_distance[i]) {
      match_idx[i] = min_index[i];
    } else {
      match_idx[i] = -1;
    }
  }
}

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <vector>

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
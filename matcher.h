
#include <array>
#include <vector>
#include <algorithm>

using Descriptor = std::array<float, 128>;

void match(std::vector<Descriptor>& lhs, std::vector<Descriptor>& rhs, std::vector<int>& match_idx) {
    match_idx.resize(lhs.size());
    for(int i = 0; i < lhs.size(); i++) {
        std::vector<std::pair<int, double>> s;
        for (int j = 0; j < rhs.size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < 128; k++) {
                double d = lhs[i][k] - rhs[j][k];
                sum += d * d;
            }
            s.push_back({j, sum});
        }

        std::sort(s.begin(), s.end(), [](auto&lhs, auto&rhs){
            return lhs.second < rhs.second;
        });

        if (s[0].second < 0.6 * s[1].second) {
            match_idx[i] = s[0].first;
        } else {
            match_idx[i] = -1;
        }
    }
}
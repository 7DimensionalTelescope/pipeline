#ifndef SIGMA_CLIPPED_STATS_H
#define SIGMA_CLIPPED_STATS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

// Computes sigma-clipped mean, median, and stddev of 1D float array
void sigma_clipped_stats(
    const std::vector<float>& data,
    float& mean_val,
    float& median_val,
    float& std_val,
    float sigma = 3.0f,
    int maxiters = 5
) {
    std::vector<float> clipped = data;

    for (int iter = 0; iter < maxiters; ++iter) {
        if (clipped.empty()) break;

        // Median
        std::vector<float> temp = clipped;
        size_t mid = temp.size() / 2;
        std::nth_element(temp.begin(), temp.begin() + mid, temp.end());
        float median = temp[mid];
        if (temp.size() % 2 == 0) {
            std::nth_element(temp.begin(), temp.begin() + mid - 1, temp.end());
            median = 0.5f * (median + temp[mid - 1]);
        }

        // Stddev (centered on median)
        double sum = 0.0, sq_sum = 0.0;
        for (float v : clipped) {
            double diff = v - median;
            sum += diff;
            sq_sum += diff * diff;
        }
        float stddev = std::sqrt(sq_sum / std::max(1.0, static_cast<double>(clipped.size() - 1)));

        // Clipping
        std::vector<float> new_clipped;
        for (float v : clipped) {
            if (std::abs(v - median) < sigma * stddev)
                new_clipped.push_back(v);
        }

        if (new_clipped.size() == clipped.size()) break;
        clipped.swap(new_clipped);
    }

    if (!clipped.empty()) {
        // Mean
        double sum = 0.0;
        for (float v : clipped) sum += v;
        mean_val = sum / clipped.size();

        // Median
        std::nth_element(clipped.begin(), clipped.begin() + clipped.size() / 2, clipped.end());
        median_val = clipped[clipped.size() / 2];
        if (clipped.size() % 2 == 0) {
            std::nth_element(clipped.begin(), clipped.begin() + clipped.size() / 2 - 1, clipped.end());
            median_val = 0.5f * (median_val + clipped[clipped.size() / 2 - 1]);
        }

        // Stddev
        double sq_sum = 0.0;
        for (float v : clipped) {
            double diff = v - mean_val;
            sq_sum += diff * diff;
        }
        std_val = std::sqrt(sq_sum / std::max(1.0, static_cast<double>(clipped.size() - 1)));
    } else {
        mean_val = median_val = std_val = 0.0f;
    }
}

#endif // SIGMA_CLIPPED_STATS_H


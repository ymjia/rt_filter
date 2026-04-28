#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "RigidMatrix.h"

namespace output_alg {

struct TimedRigidResult {
    Sn3DAlgorithm::RigidMatrix rigid;
    std::int64_t elapsed_time_ns = 0;
};

struct TimedFilterTrajectoryResult {
    std::vector<Sn3DAlgorithm::RigidMatrix> rigids;
    std::vector<std::int64_t> per_pose_time_ns;
    std::int64_t total_time_ns = 0;
};

struct TimingSummary {
    std::int64_t total_time_ns = 0;
    double total_time_ms = 0.0;
    double mean_time_us = 0.0;
    double p95_time_us = 0.0;
    double max_time_us = 0.0;
};

inline std::vector<std::int64_t> UniformPerPoseTimeNs(std::size_t count, std::int64_t total_time_ns) {
    std::vector<std::int64_t> values(count, 0);
    if (count == 0 || total_time_ns <= 0) {
        return values;
    }

    const std::int64_t base = total_time_ns / static_cast<std::int64_t>(count);
    const std::int64_t remainder = total_time_ns % static_cast<std::int64_t>(count);
    std::fill(values.begin(), values.end(), base);
    for (std::int64_t i = 0; i < remainder; ++i) {
        values[static_cast<std::size_t>(i)] += 1;
    }
    return values;
}

inline void FinalizePerPoseTimeNs(std::vector<std::int64_t>& partial_time_ns, std::int64_t total_time_ns) {
    if (partial_time_ns.empty()) {
        return;
    }

    std::int64_t measured_time_ns = 0;
    for (const std::int64_t value : partial_time_ns) {
        measured_time_ns += value;
    }

    const std::int64_t remaining_time_ns = total_time_ns - measured_time_ns;
    if (remaining_time_ns <= 0) {
        return;
    }

    const auto adjustment = UniformPerPoseTimeNs(partial_time_ns.size(), remaining_time_ns);
    for (std::size_t i = 0; i < partial_time_ns.size(); ++i) {
        partial_time_ns[i] += adjustment[i];
    }
}

inline double LinearPercentileNs(const std::vector<std::int64_t>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }

    std::vector<std::int64_t> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    if (sorted.size() == 1) {
        return static_cast<double>(sorted[0]);
    }

    if (percentile <= 0.0) {
        return static_cast<double>(sorted.front());
    }
    if (percentile >= 100.0) {
        return static_cast<double>(sorted.back());
    }

    const double position =
        percentile * static_cast<double>(sorted.size() - 1) / 100.0;
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return (1.0 - fraction) * static_cast<double>(sorted[lower]) +
           fraction * static_cast<double>(sorted[upper]);
}

inline TimingSummary SummarizeTiming(
    const std::vector<std::int64_t>& per_pose_time_ns,
    std::int64_t total_time_ns) {
    TimingSummary summary;
    summary.total_time_ns = total_time_ns;
    summary.total_time_ms = static_cast<double>(total_time_ns) / 1000000.0;

    if (per_pose_time_ns.empty()) {
        return summary;
    }

    double sum_ns = 0.0;
    std::int64_t max_ns = 0;
    for (const std::int64_t value : per_pose_time_ns) {
        sum_ns += static_cast<double>(value);
        if (value > max_ns) {
            max_ns = value;
        }
    }

    summary.mean_time_us =
        sum_ns / static_cast<double>(per_pose_time_ns.size()) / 1000.0;
    summary.p95_time_us = LinearPercentileNs(per_pose_time_ns, 95.0) / 1000.0;
    summary.max_time_us = static_cast<double>(max_ns) / 1000.0;
    return summary;
}

}  // namespace output_alg

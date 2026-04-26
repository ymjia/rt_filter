#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

namespace output_alg {

struct OneEuroZParameters {
    // Lower min_cutoff gives stronger static denoising and more lag.
    // Higher beta reduces lag during motion and lets more motion-time noise pass.
    // derivative_deadband ignores small filtered Z velocity before adapting.
    double min_cutoff = 0.02;
    double beta = 6.0;
    double d_cutoff = 2.0;
    double derivative_deadband = 1.0;
    double sample_rate_hz = 100.0;
    std::size_t history_size = 0;  // 0 means keep all filtered frames in history.
    bool strict_timestamps = false;
};

class OneEuroZRealtimeFilter {
public:
    using Pose = std::array<double, 16>;  // Row-major 4x4 pose matrix. Z translation is index 11.

    explicit OneEuroZRealtimeFilter(OneEuroZParameters params = OneEuroZParameters{});

    void Reset();
    void SetParameters(const OneEuroZParameters& params, bool reset = false);
    const OneEuroZParameters& Parameters() const { return params_; }

    // Filter one incoming pose. Only pose[11] (m23, translation Z) is changed.
    Pose Update(const Pose& pose, std::optional<double> timestamp = std::nullopt);

    // Filter every frame and return the filtered trajectory.
    std::vector<Pose> FilterTrajectory(
        const std::vector<Pose>& poses,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    // Filter a bounded history and return only the latest filtered display pose.
    Pose FilterLatestFromHistory(
        const std::vector<Pose>& poses,
        const std::vector<double>* timestamps = nullptr) const;

    const std::deque<Pose>& History() const { return history_; }

private:
    static constexpr std::size_t kZIndex = 11;

    static void Validate(const OneEuroZParameters& params);
    static double LowpassAlpha(double cutoff, double dt);
    double DeltaTime(std::optional<double> timestamp) const;
    void PushHistory(const Pose& pose);

    OneEuroZParameters params_;
    bool initialized_ = false;
    double last_raw_z_ = 0.0;
    double filtered_z_ = 0.0;
    double derivative_hat_ = 0.0;
    std::optional<double> last_timestamp_;
    std::deque<Pose> history_;
};

std::vector<OneEuroZRealtimeFilter::Pose> FilterOneEuroZTrajectory(
    const std::vector<OneEuroZRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps = nullptr,
    OneEuroZParameters params = OneEuroZParameters{});

OneEuroZRealtimeFilter::Pose FilterOneEuroZLatestFromHistory(
    const std::vector<OneEuroZRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps = nullptr,
    OneEuroZParameters params = OneEuroZParameters{});

}  // namespace output_alg

#include "one_euro_z.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace output_alg {

OneEuroZRealtimeFilter::OneEuroZRealtimeFilter(OneEuroZParameters params)
    : params_(params) {
    Validate(params_);
    Reset();
}

void OneEuroZRealtimeFilter::Reset() {
    initialized_ = false;
    last_raw_z_ = 0.0;
    filtered_z_ = 0.0;
    derivative_hat_ = 0.0;
    last_timestamp_.reset();
    history_.clear();
}

void OneEuroZRealtimeFilter::SetParameters(const OneEuroZParameters& params, bool reset) {
    Validate(params);
    params_ = params;
    if (reset) {
        Reset();
        return;
    }
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

OneEuroZRealtimeFilter::Pose OneEuroZRealtimeFilter::Update(
    const Pose& pose,
    std::optional<double> timestamp) {
    Pose filtered = pose;
    const double raw_z = pose[kZIndex];

    if (!initialized_) {
        initialized_ = true;
        last_raw_z_ = raw_z;
        filtered_z_ = raw_z;
        derivative_hat_ = 0.0;
        last_timestamp_ = timestamp;
        filtered[kZIndex] = raw_z;
        PushHistory(filtered);
        return filtered;
    }

    const double dt = DeltaTime(timestamp);
    const double derivative = (raw_z - last_raw_z_) / dt;
    const double derivative_alpha = LowpassAlpha(params_.d_cutoff, dt);
    derivative_hat_ =
        derivative_alpha * derivative + (1.0 - derivative_alpha) * derivative_hat_;

    const double effective_derivative =
        std::max(std::abs(derivative_hat_) - params_.derivative_deadband, 0.0);
    const double cutoff = params_.min_cutoff + params_.beta * effective_derivative;
    const double value_alpha = LowpassAlpha(cutoff, dt);
    filtered_z_ = value_alpha * raw_z + (1.0 - value_alpha) * filtered_z_;

    last_raw_z_ = raw_z;
    if (timestamp.has_value()) {
        last_timestamp_ = timestamp;
    }
    filtered[kZIndex] = filtered_z_;
    PushHistory(filtered);
    return filtered;
}

std::vector<OneEuroZRealtimeFilter::Pose> OneEuroZRealtimeFilter::FilterTrajectory(
    const std::vector<Pose>& poses,
    const std::vector<double>* timestamps,
    bool reset) {
    if (poses.empty()) {
        throw std::invalid_argument("poses must contain at least one frame");
    }
    if (timestamps != nullptr && timestamps->size() != poses.size()) {
        throw std::invalid_argument("timestamps size must match poses size");
    }
    if (reset) {
        Reset();
    }

    std::vector<Pose> output;
    output.reserve(poses.size());
    for (std::size_t i = 0; i < poses.size(); ++i) {
        std::optional<double> timestamp = std::nullopt;
        if (timestamps != nullptr) {
            timestamp = (*timestamps)[i];
        }
        output.push_back(Update(poses[i], timestamp));
    }
    return output;
}

OneEuroZRealtimeFilter::Pose OneEuroZRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Pose>& poses,
    const std::vector<double>* timestamps) const {
    OneEuroZRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(poses, timestamps, true);
    return filtered.back();
}

void OneEuroZRealtimeFilter::Validate(const OneEuroZParameters& params) {
    if (params.min_cutoff <= 0.0) {
        throw std::invalid_argument("min_cutoff must be positive");
    }
    if (params.beta < 0.0) {
        throw std::invalid_argument("beta must be >= 0");
    }
    if (params.d_cutoff <= 0.0) {
        throw std::invalid_argument("d_cutoff must be positive");
    }
    if (params.derivative_deadband < 0.0) {
        throw std::invalid_argument("derivative_deadband must be >= 0");
    }
    if (params.sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
}

double OneEuroZRealtimeFilter::LowpassAlpha(double cutoff, double dt) {
    const double tau = 1.0 / (2.0 * std::acos(-1.0) * cutoff);
    return dt / (dt + tau);
}

double OneEuroZRealtimeFilter::DeltaTime(std::optional<double> timestamp) const {
    const double nominal = 1.0 / params_.sample_rate_hz;
    if (!timestamp.has_value() || !last_timestamp_.has_value()) {
        return nominal;
    }
    const double dt = *timestamp - *last_timestamp_;
    if (dt > 0.0) {
        return dt;
    }
    if (params_.strict_timestamps) {
        throw std::invalid_argument("timestamps must be strictly increasing");
    }
    return nominal;
}

void OneEuroZRealtimeFilter::PushHistory(const Pose& pose) {
    history_.push_back(pose);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<OneEuroZRealtimeFilter::Pose> FilterOneEuroZTrajectory(
    const std::vector<OneEuroZRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterTrajectory(poses, timestamps, true);
}

OneEuroZRealtimeFilter::Pose FilterOneEuroZLatestFromHistory(
    const std::vector<OneEuroZRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(poses, timestamps);
}

}  // namespace output_alg

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

Sn3DAlgorithm::RigidMatrix OneEuroZRealtimeFilter::Update(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    Sn3DAlgorithm::RigidMatrix filtered = rigid;
    const double raw_z = rigid.get_translation().z();

    if (!initialized_) {
        initialized_ = true;
        last_raw_z_ = raw_z;
        filtered_z_ = raw_z;
        derivative_hat_ = 0.0;
        last_timestamp_ = timestamp;
        filtered.get_translation().z() = raw_z;
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
    filtered.get_translation().z() = filtered_z_;
    PushHistory(filtered);
    return filtered;
}

std::vector<Sn3DAlgorithm::RigidMatrix> OneEuroZRealtimeFilter::FilterTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    bool reset) {
    if (rigids.empty()) {
        throw std::invalid_argument("rigids must contain at least one frame");
    }
    if (timestamps != nullptr && timestamps->size() != rigids.size()) {
        throw std::invalid_argument("timestamps size must match rigids size");
    }
    if (reset) {
        Reset();
    }

    std::vector<Sn3DAlgorithm::RigidMatrix> output;
    output.reserve(rigids.size());
    for (std::size_t i = 0; i < rigids.size(); ++i) {
        if (timestamps != nullptr) {
            output.push_back(Update(rigids[i], (*timestamps)[i]));
        } else {
            output.push_back(Update(rigids[i]));
        }
    }
    return output;
}

Sn3DAlgorithm::RigidMatrix OneEuroZRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps) const {
    OneEuroZRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(rigids, timestamps, true);
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

double OneEuroZRealtimeFilter::DeltaTime(OptionalDouble timestamp) const {
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

void OneEuroZRealtimeFilter::PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid) {
    history_.push_back(rigid);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<Sn3DAlgorithm::RigidMatrix> FilterOneEuroZTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterTrajectory(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterOneEuroZLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

}  // namespace output_alg

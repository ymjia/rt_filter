#include "one_euro_z.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace output_alg {
namespace {

std::size_t DelayWindowSize(std::size_t delay_frames) {
    return delay_frames * 2 + 1;
}

std::size_t DelayedTargetIndex(std::size_t count, std::size_t delay_frames) {
    if (count == 0) {
        return 0;
    }
    if (count > delay_frames) {
        return count - 1 - delay_frames;
    }
    return 0;
}

}  // namespace

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
    raw_buffer_.clear();
    timestamp_buffer_.clear();
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
    const std::size_t max_delay_size = DelayWindowSize(params_.delay_frames);
    while (raw_buffer_.size() > max_delay_size) {
        raw_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
}

Sn3DAlgorithm::RigidMatrix OneEuroZRealtimeFilter::Update(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    if (params_.delay_frames > 0) {
        if (timestamp.has_value() && last_timestamp_.has_value()) {
            const double dt = *timestamp - *last_timestamp_;
            if (dt <= 0.0 && params_.strict_timestamps) {
                throw std::invalid_argument("timestamps must be strictly increasing");
            }
        }
        initialized_ = true;
        PushDelayBuffer(rigid, timestamp);
        if (timestamp.has_value()) {
            last_timestamp_ = timestamp;
        }
        Sn3DAlgorithm::RigidMatrix filtered = DelayedWindowOutput();
        PushHistory(filtered);
        return filtered;
    }

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

TimedRigidResult OneEuroZRealtimeFilter::UpdateTimed(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    const auto started = std::chrono::steady_clock::now();
    TimedRigidResult result;
    result.rigid = Update(rigid, timestamp);
    result.elapsed_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started)
                                 .count();
    return result;
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

TimedFilterTrajectoryResult OneEuroZRealtimeFilter::FilterTrajectoryTimed(
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

    TimedFilterTrajectoryResult result;
    result.rigids.reserve(rigids.size());
    result.per_pose_time_ns.reserve(rigids.size());

    const auto started = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < rigids.size(); ++i) {
        const TimedRigidResult timed =
            timestamps != nullptr ? UpdateTimed(rigids[i], (*timestamps)[i]) : UpdateTimed(rigids[i]);
        result.rigids.push_back(timed.rigid);
        result.per_pose_time_ns.push_back(timed.elapsed_time_ns);
    }
    result.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started)
                               .count();
    FinalizePerPoseTimeNs(result.per_pose_time_ns, result.total_time_ns);
    return result;
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

void OneEuroZRealtimeFilter::PushDelayBuffer(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    raw_buffer_.push_back(rigid);
    timestamp_buffer_.push_back(timestamp);
    const std::size_t max_size = DelayWindowSize(params_.delay_frames);
    while (raw_buffer_.size() > max_size) {
        raw_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
}

Sn3DAlgorithm::RigidMatrix OneEuroZRealtimeFilter::DelayedWindowOutput() const {
    if (raw_buffer_.empty()) {
        throw std::invalid_argument("delay buffer is empty");
    }
    const std::size_t target_index =
        DelayedTargetIndex(raw_buffer_.size(), params_.delay_frames);
    Sn3DAlgorithm::RigidMatrix filtered = raw_buffer_[target_index];
    filtered.get_translation().z() = SmoothedDelayedZ();
    return filtered;
}

double OneEuroZRealtimeFilter::SmoothedDelayedZ() const {
    const std::size_t count = raw_buffer_.size();
    const std::size_t target_index = DelayedTargetIndex(count, params_.delay_frames);
    if (count <= 1) {
        return raw_buffer_[target_index].get_translation().z();
    }

    const double dt = BufferMeanDt();
    const double derivative_hat = EstimateWindowDerivative(target_index);
    const double effective_derivative =
        std::max(std::abs(derivative_hat) - params_.derivative_deadband, 0.0);
    const double cutoff = params_.min_cutoff + params_.beta * effective_derivative;
    const double value_alpha = LowpassAlpha(cutoff, dt);
    const double decay = 1.0 - value_alpha;

    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double distance =
            std::abs(static_cast<double>(index) - static_cast<double>(target_index));
        const double weight = std::pow(decay, distance);
        weighted_sum += weight * raw_buffer_[index].get_translation().z();
        weight_sum += weight;
    }
    if (weight_sum <= 0.0) {
        return raw_buffer_[target_index].get_translation().z();
    }
    return weighted_sum / weight_sum;
}

double OneEuroZRealtimeFilter::BufferDt(std::size_t previous_index, std::size_t index) const {
    const double nominal = 1.0 / params_.sample_rate_hz;
    if (previous_index >= timestamp_buffer_.size() || index >= timestamp_buffer_.size()) {
        return nominal;
    }
    const OptionalDouble& previous = timestamp_buffer_[previous_index];
    const OptionalDouble& current = timestamp_buffer_[index];
    if (!previous.has_value() || !current.has_value()) {
        return nominal;
    }
    const double dt = *current - *previous;
    if (dt > 0.0) {
        return dt;
    }
    if (params_.strict_timestamps) {
        throw std::invalid_argument("timestamps must be strictly increasing");
    }
    return nominal;
}

double OneEuroZRealtimeFilter::BufferMeanDt() const {
    if (raw_buffer_.size() <= 1) {
        return 1.0 / params_.sample_rate_hz;
    }
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t index = 1; index < raw_buffer_.size(); ++index) {
        const double dt = BufferDt(index - 1, index);
        if (dt > 0.0) {
            sum += dt;
            ++count;
        }
    }
    if (count == 0) {
        return 1.0 / params_.sample_rate_hz;
    }
    return sum / static_cast<double>(count);
}

double OneEuroZRealtimeFilter::EstimateWindowDerivative(std::size_t target_index) const {
    if (raw_buffer_.size() <= 1) {
        return 0.0;
    }

    const double derivative_alpha = LowpassAlpha(params_.d_cutoff, BufferMeanDt());
    const double derivative_decay = 1.0 - derivative_alpha;
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t index = 1; index < raw_buffer_.size(); ++index) {
        const double dt = BufferDt(index - 1, index);
        const double previous_z = raw_buffer_[index - 1].get_translation().z();
        const double current_z = raw_buffer_[index].get_translation().z();
        const double derivative = (current_z - previous_z) / dt;
        const double step_center = static_cast<double>(index) - 0.5;
        const double distance = std::abs(step_center - static_cast<double>(target_index));
        const double weight = std::pow(derivative_decay, distance);
        weighted_sum += weight * derivative;
        weight_sum += weight;
    }
    if (weight_sum <= 0.0) {
        return 0.0;
    }
    return weighted_sum / weight_sum;
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

TimedFilterTrajectoryResult FilterOneEuroZTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterTrajectoryTimed(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterOneEuroZLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    OneEuroZParameters params) {
    OneEuroZRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

}  // namespace output_alg

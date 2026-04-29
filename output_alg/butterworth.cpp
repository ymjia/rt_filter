#include "butterworth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace output_alg {
namespace {

constexpr double kPi = 3.14159265358979323846;

void ValidateRuntimeSampleRate(const ButterworthParameters& params, double sample_rate_hz) {
    if (sample_rate_hz <= 0.0) {
        throw std::invalid_argument("sample_rate_hz must be positive");
    }
    const double nyquist_hz = 0.5 * sample_rate_hz;
    if (params.cutoff_hz >= nyquist_hz) {
        throw std::invalid_argument("cutoff_hz must be smaller than Nyquist frequency");
    }
}

ButterworthSectionState MakeFirstOrderSection(double cutoff_hz, double sample_rate_hz) {
    const double k = std::tan(kPi * cutoff_hz / sample_rate_hz);
    const double norm = 1.0 / (1.0 + k);

    ButterworthSectionState section;
    section.b0 = k * norm;
    section.b1 = section.b0;
    section.b2 = 0.0;
    section.a1 = (k - 1.0) * norm;
    section.a2 = 0.0;
    return section;
}

ButterworthSectionState MakeSecondOrderSection(double cutoff_hz, double sample_rate_hz, double quality) {
    const double k = std::tan(kPi * cutoff_hz / sample_rate_hz);
    const double norm = 1.0 / (1.0 + k / quality + k * k);

    ButterworthSectionState section;
    section.b0 = k * k * norm;
    section.b1 = 2.0 * section.b0;
    section.b2 = section.b0;
    section.a1 = 2.0 * (k * k - 1.0) * norm;
    section.a2 = (1.0 - k / quality + k * k) * norm;
    return section;
}

std::vector<ButterworthSectionState> DesignSections(
    const ButterworthParameters& params,
    double sample_rate_hz) {
    ValidateRuntimeSampleRate(params, sample_rate_hz);

    std::vector<ButterworthSectionState> sections;
    sections.reserve(static_cast<std::size_t>((params.order + 1) / 2));

    if ((params.order % 2) != 0) {
        sections.push_back(MakeFirstOrderSection(params.cutoff_hz, sample_rate_hz));
    }

    const int pair_count = params.order / 2;
    for (int pair_index = 1; pair_index <= pair_count; ++pair_index) {
        const double theta =
            kPi * static_cast<double>(2 * pair_index - 1) / (2.0 * static_cast<double>(params.order));
        const double quality = 1.0 / (2.0 * std::sin(theta));
        sections.push_back(MakeSecondOrderSection(params.cutoff_hz, sample_rate_hz, quality));
    }
    return sections;
}

void InitializeSectionSteadyState(ButterworthSectionState& section, double value) {
    section.s1 = value * (1.0 - section.b0);
    section.s2 = value * (section.b2 - section.a2);
}

void InitializeSections(std::vector<ButterworthSectionState>& sections, double value) {
    for (ButterworthSectionState& section : sections) {
        InitializeSectionSteadyState(section, value);
    }
}

double ApplySection(ButterworthSectionState& section, double value) {
    const double output = section.b0 * value + section.s1;
    const double next_s1 = section.b1 * value - section.a1 * output + section.s2;
    const double next_s2 = section.b2 * value - section.a2 * output;
    section.s1 = next_s1;
    section.s2 = next_s2;
    return output;
}

double ApplySections(std::vector<ButterworthSectionState>& sections, double value) {
    double filtered = value;
    for (ButterworthSectionState& section : sections) {
        filtered = ApplySection(section, filtered);
    }
    return filtered;
}

double DeltaTimeFromTimestamp(
    OptionalDouble timestamp,
    OptionalDouble last_timestamp,
    double sample_rate_hz,
    bool strict_timestamps) {
    const double nominal = 1.0 / sample_rate_hz;
    if (!timestamp.has_value() || !last_timestamp.has_value()) {
        return nominal;
    }
    const double dt = *timestamp - *last_timestamp;
    if (dt > 0.0) {
        return dt;
    }
    if (strict_timestamps) {
        throw std::invalid_argument("timestamps must be strictly increasing");
    }
    return nominal;
}

bool NeedsRedesign(double current_sample_rate_hz, double next_sample_rate_hz) {
    if (current_sample_rate_hz <= 0.0) {
        return true;
    }
    const double scale = std::max(std::abs(current_sample_rate_hz), std::abs(next_sample_rate_hz));
    return std::abs(current_sample_rate_hz - next_sample_rate_hz) > scale * 1e-6;
}

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

double SampleRateFromDelayBuffer(
    const std::deque<OptionalDouble>& timestamps,
    double fallback_sample_rate_hz,
    bool strict_timestamps) {
    double dt_sum = 0.0;
    std::size_t dt_count = 0;
    for (std::size_t index = 1; index < timestamps.size(); ++index) {
        const OptionalDouble& previous = timestamps[index - 1];
        const OptionalDouble& current = timestamps[index];
        if (!previous.has_value() || !current.has_value()) {
            continue;
        }
        const double dt = *current - *previous;
        if (dt > 0.0) {
            dt_sum += dt;
            ++dt_count;
            continue;
        }
        if (strict_timestamps) {
            throw std::invalid_argument("timestamps must be strictly increasing");
        }
    }
    if (dt_count == 0 || dt_sum <= 0.0) {
        return fallback_sample_rate_hz;
    }
    return static_cast<double>(dt_count) / dt_sum;
}

double ApplyForwardBackwardWindow(
    const std::vector<double>& values,
    std::size_t target_index,
    const ButterworthParameters& params,
    double sample_rate_hz) {
    if (values.empty()) {
        return 0.0;
    }
    if (values.size() == 1) {
        return values[0];
    }

    std::vector<ButterworthSectionState> sections = DesignSections(params, sample_rate_hz);
    InitializeSections(sections, values.front());
    std::vector<double> forward(values.size(), 0.0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        forward[index] = ApplySections(sections, values[index]);
    }

    sections = DesignSections(params, sample_rate_hz);
    InitializeSections(sections, forward.back());
    std::vector<double> smoothed(values.size(), 0.0);
    for (std::size_t reverse_index = values.size(); reverse_index-- > 0;) {
        smoothed[reverse_index] = ApplySections(sections, forward[reverse_index]);
    }
    return smoothed[target_index];
}

}  // namespace

ButterworthRealtimeFilter::ButterworthRealtimeFilter(ButterworthParameters params)
    : params_(params) {
    Validate(params_);
    Reset();
}

void ButterworthRealtimeFilter::Reset() {
    initialized_ = false;
    current_sample_rate_hz_ = 0.0;
    last_output_.setZero();
    last_timestamp_.reset();
    for (auto& axis_sections : sections_) {
        axis_sections.clear();
    }
    history_.clear();
}

void ButterworthRealtimeFilter::SetParameters(const ButterworthParameters& params, bool reset) {
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
    if (initialized_) {
        const double sample_rate_hz =
            current_sample_rate_hz_ > 0.0 ? current_sample_rate_hz_ : params_.sample_rate_hz;
        Redesign(sample_rate_hz, last_output_);
    }
}

Sn3DAlgorithm::RigidMatrix ButterworthRealtimeFilter::Update(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    Sn3DAlgorithm::RigidMatrix filtered = rigid;
    Eigen::Vector3d translation = rigid.get_translation();

    if (!initialized_) {
        Redesign(params_.sample_rate_hz, translation);
        initialized_ = true;
        last_output_ = translation;
        last_timestamp_ = timestamp;
        filtered.set_translation(translation);
        PushHistory(filtered);
        return filtered;
    }

    const double dt = DeltaTime(timestamp);
    const double sample_rate_hz = 1.0 / dt;
    if (NeedsRedesign(current_sample_rate_hz_, sample_rate_hz)) {
        Redesign(sample_rate_hz, last_output_);
    }

    for (int axis = 0; axis < 3; ++axis) {
        translation(axis) = ApplySections(sections_[static_cast<std::size_t>(axis)], translation(axis));
    }

    last_output_ = translation;
    if (timestamp.has_value()) {
        last_timestamp_ = timestamp;
    }
    filtered.set_translation(translation);
    PushHistory(filtered);
    return filtered;
}

Eigen::Vector3d ButterworthRealtimeFilter::Update(
    const Eigen::Vector3d& point,
    OptionalDouble timestamp) {
    Sn3DAlgorithm::RigidMatrix rigid(Eigen::Matrix3d::Identity(), point);
    return Update(rigid, timestamp).get_translation();
}

TimedRigidResult ButterworthRealtimeFilter::UpdateTimed(
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

std::vector<Sn3DAlgorithm::RigidMatrix> ButterworthRealtimeFilter::FilterTrajectory(
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

TimedFilterTrajectoryResult ButterworthRealtimeFilter::FilterTrajectoryTimed(
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

Sn3DAlgorithm::RigidMatrix ButterworthRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps) const {
    ButterworthRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(rigids, timestamps, true);
    return filtered.back();
}

void ButterworthRealtimeFilter::Validate(const ButterworthParameters& params) {
    if (params.cutoff_hz <= 0.0) {
        throw std::invalid_argument("cutoff_hz must be positive");
    }
    if (params.order < 1) {
        throw std::invalid_argument("order must be >= 1");
    }
    ValidateRuntimeSampleRate(params, params.sample_rate_hz);
}

double ButterworthRealtimeFilter::DeltaTime(OptionalDouble timestamp) const {
    return DeltaTimeFromTimestamp(
        timestamp,
        last_timestamp_,
        params_.sample_rate_hz,
        params_.strict_timestamps);
}

void ButterworthRealtimeFilter::Redesign(double sample_rate_hz, const Eigen::Vector3d& anchor_value) {
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<ButterworthSectionState>& axis_sections = sections_[static_cast<std::size_t>(axis)];
        axis_sections = DesignSections(params_, sample_rate_hz);
        InitializeSections(axis_sections, anchor_value(axis));
    }
    current_sample_rate_hz_ = sample_rate_hz;
}

void ButterworthRealtimeFilter::PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid) {
    history_.push_back(rigid);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

ButterworthZRealtimeFilter::ButterworthZRealtimeFilter(ButterworthZParameters params)
    : params_(params) {
    Validate(params_);
    Reset();
}

void ButterworthZRealtimeFilter::Reset() {
    initialized_ = false;
    current_sample_rate_hz_ = 0.0;
    last_output_z_ = 0.0;
    last_timestamp_.reset();
    sections_.clear();
    raw_buffer_.clear();
    timestamp_buffer_.clear();
    history_.clear();
}

void ButterworthZRealtimeFilter::SetParameters(const ButterworthZParameters& params, bool reset) {
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
    if (initialized_) {
        const double sample_rate_hz =
            current_sample_rate_hz_ > 0.0 ? current_sample_rate_hz_ : params_.sample_rate_hz;
        Redesign(sample_rate_hz, last_output_z_);
    }
}

Sn3DAlgorithm::RigidMatrix ButterworthZRealtimeFilter::Update(
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
        last_output_z_ = filtered.get_translation().z();
        PushHistory(filtered);
        return filtered;
    }

    Sn3DAlgorithm::RigidMatrix filtered = rigid;
    const double raw_z = rigid.get_translation().z();

    if (!initialized_) {
        Redesign(params_.sample_rate_hz, raw_z);
        initialized_ = true;
        last_output_z_ = raw_z;
        last_timestamp_ = timestamp;
        filtered.get_translation().z() = raw_z;
        PushHistory(filtered);
        return filtered;
    }

    const double dt = DeltaTime(timestamp);
    const double sample_rate_hz = 1.0 / dt;
    if (NeedsRedesign(current_sample_rate_hz_, sample_rate_hz)) {
        Redesign(sample_rate_hz, last_output_z_);
    }

    last_output_z_ = ApplySections(sections_, raw_z);
    if (timestamp.has_value()) {
        last_timestamp_ = timestamp;
    }
    filtered.get_translation().z() = last_output_z_;
    PushHistory(filtered);
    return filtered;
}

Eigen::Vector3d ButterworthZRealtimeFilter::Update(
    const Eigen::Vector3d& point,
    OptionalDouble timestamp) {
    Sn3DAlgorithm::RigidMatrix rigid(Eigen::Matrix3d::Identity(), point);
    return Update(rigid, timestamp).get_translation();
}

TimedRigidResult ButterworthZRealtimeFilter::UpdateTimed(
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

std::vector<Sn3DAlgorithm::RigidMatrix> ButterworthZRealtimeFilter::FilterTrajectory(
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

TimedFilterTrajectoryResult ButterworthZRealtimeFilter::FilterTrajectoryTimed(
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

Sn3DAlgorithm::RigidMatrix ButterworthZRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps) const {
    ButterworthZRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(rigids, timestamps, true);
    return filtered.back();
}

void ButterworthZRealtimeFilter::Validate(const ButterworthZParameters& params) {
    if (params.cutoff_hz <= 0.0) {
        throw std::invalid_argument("cutoff_hz must be positive");
    }
    if (params.order < 1) {
        throw std::invalid_argument("order must be >= 1");
    }
    ValidateRuntimeSampleRate(params, params.sample_rate_hz);
}

double ButterworthZRealtimeFilter::DeltaTime(OptionalDouble timestamp) const {
    return DeltaTimeFromTimestamp(
        timestamp,
        last_timestamp_,
        params_.sample_rate_hz,
        params_.strict_timestamps);
}

void ButterworthZRealtimeFilter::Redesign(double sample_rate_hz, double anchor_value) {
    sections_ = DesignSections(params_, sample_rate_hz);
    InitializeSections(sections_, anchor_value);
    current_sample_rate_hz_ = sample_rate_hz;
}

void ButterworthZRealtimeFilter::PushDelayBuffer(
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

Sn3DAlgorithm::RigidMatrix ButterworthZRealtimeFilter::DelayedWindowOutput() const {
    if (raw_buffer_.empty()) {
        throw std::invalid_argument("delay buffer is empty");
    }
    const std::size_t target_index =
        DelayedTargetIndex(raw_buffer_.size(), params_.delay_frames);
    std::vector<double> z_values;
    z_values.reserve(raw_buffer_.size());
    for (const Sn3DAlgorithm::RigidMatrix& frame : raw_buffer_) {
        z_values.push_back(frame.get_translation().z());
    }
    const double sample_rate_hz = SampleRateFromDelayBuffer(
        timestamp_buffer_,
        params_.sample_rate_hz,
        params_.strict_timestamps);
    Sn3DAlgorithm::RigidMatrix filtered = raw_buffer_[target_index];
    filtered.get_translation().z() =
        ApplyForwardBackwardWindow(z_values, target_index, params_, sample_rate_hz);
    return filtered;
}

void ButterworthZRealtimeFilter::PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid) {
    history_.push_back(rigid);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<Sn3DAlgorithm::RigidMatrix> FilterButterworthTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthParameters params) {
    ButterworthRealtimeFilter filter(params);
    return filter.FilterTrajectory(rigids, timestamps, true);
}

TimedFilterTrajectoryResult FilterButterworthTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthParameters params) {
    ButterworthRealtimeFilter filter(params);
    return filter.FilterTrajectoryTimed(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterButterworthLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthParameters params) {
    ButterworthRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

std::vector<Sn3DAlgorithm::RigidMatrix> FilterButterworthZTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthZParameters params) {
    ButterworthZRealtimeFilter filter(params);
    return filter.FilterTrajectory(rigids, timestamps, true);
}

TimedFilterTrajectoryResult FilterButterworthZTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthZParameters params) {
    ButterworthZRealtimeFilter filter(params);
    return filter.FilterTrajectoryTimed(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterButterworthZLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    ButterworthZParameters params) {
    ButterworthZRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

}  // namespace output_alg

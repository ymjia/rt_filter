#include "adaptive_local_line.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <Eigen/SVD>

namespace output_alg {
namespace {

std::size_t HalfWindow(std::size_t window) {
    return window / 2;
}

std::size_t DelayWindowSize(std::size_t window) {
    return window;
}

std::size_t DelayedTargetIndex(std::size_t count, std::size_t half_window) {
    if (count == 0) {
        return 0;
    }
    if (count > half_window) {
        return count - 1 - half_window;
    }
    return 0;
}

Eigen::Vector3d NormalizedDirection(const Eigen::Vector3d& direction) {
    const double norm = direction.norm();
    if (norm <= 0.0) {
        throw std::invalid_argument("line_direction must be non-zero");
    }
    return direction / norm;
}

Eigen::Vector3d MeanPoint(const std::vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& point : points) {
        mean += point;
    }
    return mean / static_cast<double>(points.size());
}

std::vector<Eigen::Vector3d> PositionsFromRigids(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(rigids.size());
    for (const auto& rigid : rigids) {
        points.push_back(rigid.get_translation());
    }
    return points;
}

}  // namespace

AdaptiveLocalLineRealtimeFilter::AdaptiveLocalLineRealtimeFilter(
    AdaptiveLocalLineParameters params)
    : params_(params) {
    Validate(params_);
    Reset();
}

void AdaptiveLocalLineRealtimeFilter::Reset() {
    raw_buffer_.clear();
    timestamp_buffer_.clear();
    history_.clear();
}

void AdaptiveLocalLineRealtimeFilter::SetParameters(
    const AdaptiveLocalLineParameters& params,
    bool reset) {
    Validate(params);
    params_ = params;
    if (reset) {
        Reset();
        return;
    }
    while (raw_buffer_.size() > DelayWindowSize(params_.window)) {
        raw_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

Sn3DAlgorithm::RigidMatrix AdaptiveLocalLineRealtimeFilter::Update(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    PushDelayBuffer(rigid, timestamp);
    Sn3DAlgorithm::RigidMatrix filtered = DelayedWindowOutput();
    PushHistory(filtered);
    return filtered;
}

Eigen::Vector3d AdaptiveLocalLineRealtimeFilter::Update(
    const Eigen::Vector3d& point,
    OptionalDouble timestamp) {
    Sn3DAlgorithm::RigidMatrix rigid(Eigen::Matrix3d::Identity(), point);
    return Update(rigid, timestamp).get_translation();
}

TimedRigidResult AdaptiveLocalLineRealtimeFilter::UpdateTimed(
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

std::vector<Sn3DAlgorithm::RigidMatrix> AdaptiveLocalLineRealtimeFilter::FilterTrajectory(
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

    const auto filtered = FilterTrajectoryImpl(rigids, timestamps, params_);
    history_.clear();
    for (const auto& rigid : filtered) {
        PushHistory(rigid);
    }
    return filtered;
}

TimedFilterTrajectoryResult AdaptiveLocalLineRealtimeFilter::FilterTrajectoryTimed(
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

    const auto started = std::chrono::steady_clock::now();
    TimedFilterTrajectoryResult result;
    result.rigids = FilterTrajectoryImpl(rigids, timestamps, params_);
    result.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started)
                               .count();
    result.per_pose_time_ns =
        UniformPerPoseTimeNs(result.rigids.size(), result.total_time_ns);
    history_.clear();
    for (const auto& rigid : result.rigids) {
        PushHistory(rigid);
    }
    return result;
}

Sn3DAlgorithm::RigidMatrix AdaptiveLocalLineRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps) const {
    if (rigids.empty()) {
        throw std::invalid_argument("rigids must contain at least one frame");
    }
    if (timestamps != nullptr && timestamps->size() != rigids.size()) {
        throw std::invalid_argument("timestamps size must match rigids size");
    }

    AdaptiveLocalLineRealtimeFilter filter(params_);
    Sn3DAlgorithm::RigidMatrix output;
    for (std::size_t i = 0; i < rigids.size(); ++i) {
        output = timestamps != nullptr ? filter.Update(rigids[i], (*timestamps)[i])
                                       : filter.Update(rigids[i]);
    }
    return output;
}

void AdaptiveLocalLineRealtimeFilter::Validate(const AdaptiveLocalLineParameters& params) {
    if (params.window < 3 || (params.window % 2) == 0) {
        throw std::invalid_argument("window must be an odd integer >= 3");
    }
    if (params.target_noise_mm < 0.0 || params.min_strength < 0.0 ||
        params.max_strength > 1.0 || params.min_strength > params.max_strength ||
        params.response <= 0.0 || params.sample_rate_hz <= 0.0) {
        throw std::invalid_argument(
            "invalid adaptive_local_line parameters: target_noise_mm >= 0, "
            "0 <= min_strength <= max_strength <= 1, response > 0, sample_rate_hz > 0");
    }
    CanonicalReferenceMode(params.reference_mode);
    if (params.use_line_origin && !params.use_line_direction) {
        throw std::invalid_argument("line_origin requires line_direction");
    }
    if (params.use_line_direction) {
        (void)NormalizedDirection(params.line_direction);
    }
}

std::string AdaptiveLocalLineRealtimeFilter::CanonicalReferenceMode(
    const std::string& mode) {
    if (mode == "global" || mode == "GLOBAL") {
        return "global";
    }
    if (mode == "local" || mode == "LOCAL") {
        return "local";
    }
    throw std::invalid_argument("reference_mode must be global or local");
}

AdaptiveLocalLineRealtimeFilter::Line AdaptiveLocalLineRealtimeFilter::FitPrincipalLine(
    const std::vector<Eigen::Vector3d>& points) {
    Line line;
    line.origin = MeanPoint(points);
    if (points.size() <= 1) {
        line.direction = Eigen::Vector3d::UnitX();
        return line;
    }

    Eigen::MatrixXd centered(points.size(), 3);
    for (std::size_t i = 0; i < points.size(); ++i) {
        centered.row(static_cast<Eigen::Index>(i)) = (points[i] - line.origin).transpose();
    }
    if (centered.norm() <= 0.0) {
        line.direction = Eigen::Vector3d::UnitX();
        return line;
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
    line.direction = svd.matrixV().col(0);
    if (line.direction.x() < 0.0) {
        line.direction = -line.direction;
    }
    line.direction.normalize();
    return line;
}

AdaptiveLocalLineRealtimeFilter::Line AdaptiveLocalLineRealtimeFilter::ReferenceLine(
    const std::vector<Eigen::Vector3d>& points,
    const AdaptiveLocalLineParameters& params) {
    if (params.use_line_direction) {
        Line line;
        line.origin = params.use_line_origin ? params.line_origin : MeanPoint(points);
        line.direction = NormalizedDirection(params.line_direction);
        return line;
    }
    return FitPrincipalLine(points);
}

double AdaptiveLocalLineRealtimeFilter::Strength(
    double noise,
    double target_noise,
    double min_strength,
    double max_strength,
    double response) {
    if (noise <= target_noise || noise <= 0.0) {
        return 0.0;
    }
    const double raw = target_noise > 0.0
                           ? 1.0 - std::pow(target_noise / noise, response)
                           : 1.0;
    return std::min(max_strength, std::max(min_strength, raw));
}

Eigen::Vector3d AdaptiveLocalLineRealtimeFilter::LocalLineCenterPerpResidual(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<double>& relative_times,
    std::size_t center_index,
    double* noise) {
    if (points.empty() || points.size() != relative_times.size() ||
        center_index >= points.size()) {
        throw std::invalid_argument("invalid local line window");
    }

    double mean_t = 0.0;
    Eigen::Vector3d mean_p = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        mean_t += relative_times[i];
        mean_p += points[i];
    }
    mean_t /= static_cast<double>(points.size());
    mean_p /= static_cast<double>(points.size());

    double denominator = 0.0;
    Eigen::Vector3d numerator = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double dt = relative_times[i] - mean_t;
        denominator += dt * dt;
        numerator += dt * (points[i] - mean_p);
    }

    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    if (denominator > 0.0) {
        velocity = numerator / denominator;
    }
    const Eigen::Vector3d intercept = mean_p - velocity * mean_t;
    Eigen::Vector3d direction =
        velocity.norm() > 0.0 ? velocity.normalized() : FitPrincipalLine(points).direction;

    double sum_sq = 0.0;
    Eigen::Vector3d center_perp = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector3d model = intercept + velocity * relative_times[i];
        const Eigen::Vector3d residual = points[i] - model;
        const Eigen::Vector3d perp = residual - residual.dot(direction) * direction;
        sum_sq += perp.squaredNorm();
        if (i == center_index) {
            center_perp = perp;
        }
    }
    if (noise != nullptr) {
        *noise = std::sqrt(sum_sq / static_cast<double>(points.size()));
    }
    return center_perp;
}

std::vector<double> AdaptiveLocalLineRealtimeFilter::TimeValues(
    const std::vector<double>* timestamps,
    std::size_t count,
    double sample_rate_hz,
    bool strict_timestamps) {
    std::vector<double> values(count, 0.0);
    if (timestamps == nullptr) {
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = static_cast<double>(i) / sample_rate_hz;
        }
        return values;
    }
    if (timestamps->size() != count) {
        throw std::invalid_argument("timestamps size must match rigids size");
    }
    values = *timestamps;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[i - 1]) {
            continue;
        }
        if (strict_timestamps) {
            throw std::invalid_argument("timestamps must be strictly increasing");
        }
        values[i] = values[i - 1] + 1.0 / sample_rate_hz;
    }
    return values;
}

std::vector<Sn3DAlgorithm::RigidMatrix> AdaptiveLocalLineRealtimeFilter::FilterTrajectoryImpl(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    const AdaptiveLocalLineParameters& params) {
    Validate(params);
    if (rigids.empty()) {
        throw std::invalid_argument("rigids must contain at least one frame");
    }

    const std::size_t count = rigids.size();
    const std::size_t half = HalfWindow(params.window);
    std::vector<Sn3DAlgorithm::RigidMatrix> output = rigids;
    if (count < params.window) {
        return output;
    }

    const std::vector<Eigen::Vector3d> points = PositionsFromRigids(rigids);
    const std::vector<double> times =
        TimeValues(timestamps, count, params.sample_rate_hz, params.strict_timestamps);
    const std::string mode = CanonicalReferenceMode(params.reference_mode);

    if (mode == "global") {
        const Line line = ReferenceLine(points, params);
        std::vector<Eigen::Vector3d> perp(count, Eigen::Vector3d::Zero());
        std::vector<double> distances(count, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            const Eigen::Vector3d centered = points[i] - line.origin;
            perp[i] = centered - centered.dot(line.direction) * line.direction;
            distances[i] = perp[i].norm();
        }
        for (std::size_t center = half; center + half < count; ++center) {
            double sum_sq = 0.0;
            for (std::size_t j = center - half; j <= center + half; ++j) {
                sum_sq += distances[j] * distances[j];
            }
            const double local_noise =
                std::sqrt(sum_sq / static_cast<double>(params.window));
            const double strength = Strength(
                local_noise,
                params.target_noise_mm,
                params.min_strength,
                params.max_strength,
                params.response);
            output[center].get_translation() = points[center] - strength * perp[center];
        }
        return output;
    }

    for (std::size_t center = half; center + half < count; ++center) {
        std::vector<Eigen::Vector3d> window_points;
        std::vector<double> relative_times;
        window_points.reserve(params.window);
        relative_times.reserve(params.window);
        for (std::size_t j = center - half; j <= center + half; ++j) {
            window_points.push_back(points[j]);
            relative_times.push_back(times[j] - times[center]);
        }
        double local_noise = 0.0;
        const Eigen::Vector3d center_perp =
            LocalLineCenterPerpResidual(window_points, relative_times, half, &local_noise);
        const double strength = Strength(
            local_noise,
            params.target_noise_mm,
            params.min_strength,
            params.max_strength,
            params.response);
        output[center].get_translation() = points[center] - strength * center_perp;
    }
    return output;
}

Sn3DAlgorithm::RigidMatrix AdaptiveLocalLineRealtimeFilter::DelayedWindowOutput() const {
    if (raw_buffer_.empty()) {
        return Sn3DAlgorithm::RigidMatrix();
    }
    const std::size_t half = HalfWindow(params_.window);
    const std::size_t target_index = DelayedTargetIndex(raw_buffer_.size(), half);
    Sn3DAlgorithm::RigidMatrix filtered = raw_buffer_[target_index];
    if (raw_buffer_.size() < params_.window) {
        return filtered;
    }

    std::vector<Sn3DAlgorithm::RigidMatrix> rigids(raw_buffer_.begin(), raw_buffer_.end());
    std::vector<double> timestamps;
    bool has_all_timestamps = true;
    timestamps.reserve(timestamp_buffer_.size());
    for (const OptionalDouble& timestamp : timestamp_buffer_) {
        if (!timestamp.has_value()) {
            has_all_timestamps = false;
            break;
        }
        timestamps.push_back(*timestamp);
    }
    const std::vector<double>* timestamps_ptr = has_all_timestamps ? &timestamps : nullptr;

    AdaptiveLocalLineParameters params = params_;
    if (CanonicalReferenceMode(params.reference_mode) == "global" && !params.use_line_direction) {
        params.reference_mode = "local";
    }
    const auto output = FilterTrajectoryImpl(rigids, timestamps_ptr, params);
    return output[target_index];
}

void AdaptiveLocalLineRealtimeFilter::PushDelayBuffer(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    if (timestamp.has_value() && !timestamp_buffer_.empty() &&
        timestamp_buffer_.back().has_value()) {
        const double dt = *timestamp - *timestamp_buffer_.back();
        if (dt <= 0.0 && params_.strict_timestamps) {
            throw std::invalid_argument("timestamps must be strictly increasing");
        }
    }
    raw_buffer_.push_back(rigid);
    timestamp_buffer_.push_back(timestamp);
    while (raw_buffer_.size() > DelayWindowSize(params_.window)) {
        raw_buffer_.pop_front();
        timestamp_buffer_.pop_front();
    }
}

void AdaptiveLocalLineRealtimeFilter::PushHistory(
    const Sn3DAlgorithm::RigidMatrix& rigid) {
    history_.push_back(rigid);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<Sn3DAlgorithm::RigidMatrix> FilterAdaptiveLocalLineTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    AdaptiveLocalLineParameters params) {
    AdaptiveLocalLineRealtimeFilter filter(params);
    return filter.FilterTrajectory(rigids, timestamps, true);
}

TimedFilterTrajectoryResult FilterAdaptiveLocalLineTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    AdaptiveLocalLineParameters params) {
    AdaptiveLocalLineRealtimeFilter filter(params);
    return filter.FilterTrajectoryTimed(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterAdaptiveLocalLineLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    AdaptiveLocalLineParameters params) {
    AdaptiveLocalLineRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

}  // namespace output_alg

#include "ukf.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Geometry>

namespace output_alg {

UkfRealtimeFilter::UkfRealtimeFilter(UkfParameters params)
    : params_(std::move(params)) {
    Validate(params_);
    Reset();
}

void UkfRealtimeFilter::Reset() {
    initialized_ = false;
    motion_model_ = CanonicalMotionModel(params_.motion_model);
    order_ = motion_model_ == "constant_acceleration" ? 3 : 2;
    scale_ = 0.0;
    reference_rotation_.setIdentity();
    state_.resize(0);
    covariance_.resize(0, 0);
    weights_mean_.resize(0);
    weights_cov_.resize(0);
    last_timestamp_.reset();
    history_.clear();
}

void UkfRealtimeFilter::SetParameters(const UkfParameters& params, bool reset) {
    Validate(params);
    params_ = params;
    if (reset) {
        Reset();
        return;
    }
    motion_model_ = CanonicalMotionModel(params_.motion_model);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

Sn3DAlgorithm::RigidMatrix UkfRealtimeFilter::Update(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    if (!initialized_) {
        return Initialize(rigid, timestamp);
    }

    const Eigen::VectorXd measurement = MeasurementFromRigid(rigid, reference_rotation_);
    const double dt = DeltaTime(timestamp);
    const Eigen::MatrixXd sigma_points = SigmaPoints(state_, covariance_, scale_);

    Eigen::MatrixXd predicted_sigma(sigma_points.rows(), sigma_points.cols());
    for (int row = 0; row < sigma_points.rows(); ++row) {
        predicted_sigma.row(row) =
            PredictSigma(sigma_points.row(row).transpose(), kMeasurementDims, dt, motion_model_).transpose();
    }

    const Eigen::VectorXd state_pred = predicted_sigma.transpose() * weights_mean_;
    Eigen::MatrixXd state_diff(predicted_sigma.rows(), predicted_sigma.cols());
    for (int row = 0; row < predicted_sigma.rows(); ++row) {
        state_diff.row(row) = predicted_sigma.row(row) - state_pred.transpose();
    }

    Eigen::MatrixXd covariance_pred = ProcessCovariance(
        kMeasurementDims,
        order_,
        dt,
        params_.process_noise);
    for (int row = 0; row < state_diff.rows(); ++row) {
        covariance_pred += weights_cov_(row) * state_diff.row(row).transpose() * state_diff.row(row);
    }
    covariance_pred = Symmetrize(covariance_pred);

    const Eigen::MatrixXd measurement_sigma = predicted_sigma.leftCols(kMeasurementDims);
    const Eigen::VectorXd measurement_pred = measurement_sigma.transpose() * weights_mean_;
    Eigen::MatrixXd measurement_diff(measurement_sigma.rows(), measurement_sigma.cols());
    for (int row = 0; row < measurement_sigma.rows(); ++row) {
        measurement_diff.row(row) = measurement_sigma.row(row) - measurement_pred.transpose();
    }

    Eigen::MatrixXd innovation_covariance =
        Eigen::MatrixXd::Identity(kMeasurementDims, kMeasurementDims) * params_.measurement_noise;
    Eigen::MatrixXd cross_covariance =
        Eigen::MatrixXd::Zero(state_.size(), kMeasurementDims);
    for (int row = 0; row < measurement_diff.rows(); ++row) {
        innovation_covariance +=
            weights_cov_(row) * measurement_diff.row(row).transpose() * measurement_diff.row(row);
        cross_covariance +=
            weights_cov_(row) * state_diff.row(row).transpose() * measurement_diff.row(row);
    }
    innovation_covariance = Symmetrize(innovation_covariance);

    const Eigen::MatrixXd gain =
        innovation_covariance.transpose().ldlt().solve(cross_covariance.transpose()).transpose();
    const Eigen::VectorXd innovation = measurement - measurement_pred;
    state_ = state_pred + gain * innovation;
    covariance_ = Symmetrize(covariance_pred - gain * innovation_covariance * gain.transpose());

    if (timestamp.has_value()) {
        last_timestamp_ = timestamp;
    }

    Sn3DAlgorithm::RigidMatrix filtered =
        RigidFromMeasurement(state_.head(kMeasurementDims), reference_rotation_);
    PushHistory(filtered);
    return filtered;
}

TimedRigidResult UkfRealtimeFilter::UpdateTimed(
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

std::vector<Sn3DAlgorithm::RigidMatrix> UkfRealtimeFilter::FilterTrajectory(
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

TimedFilterTrajectoryResult UkfRealtimeFilter::FilterTrajectoryTimed(
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

Sn3DAlgorithm::RigidMatrix UkfRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps) const {
    UkfRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(rigids, timestamps, true);
    return filtered.back();
}

void UkfRealtimeFilter::Validate(const UkfParameters& params) {
    CanonicalMotionModel(params.motion_model);
    if (params.process_noise <= 0.0 || params.measurement_noise <= 0.0 ||
        params.initial_covariance <= 0.0 || params.alpha <= 0.0 ||
        params.sample_rate_hz <= 0.0) {
        throw std::invalid_argument(
            "process_noise, measurement_noise, initial_covariance, alpha, and sample_rate_hz must be positive");
    }
}

std::string UkfRealtimeFilter::CanonicalMotionModel(const std::string& value) {
    std::string model = value;
    std::transform(model.begin(), model.end(), model.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::replace(model.begin(), model.end(), '-', '_');
    if (model == "constant_velocity" || model == "cv") {
        return "constant_velocity";
    }
    if (model == "constant_acceleration" || model == "ca") {
        return "constant_acceleration";
    }
    throw std::invalid_argument("motion_model must be constant_velocity or constant_acceleration");
}

Eigen::VectorXd UkfRealtimeFilter::InitialVelocityVector(const UkfParameters& params) {
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(kMeasurementDims);
    if (params.use_initial_velocity) {
        for (int i = 0; i < kMeasurementDims; ++i) {
            velocity(i) = params.initial_velocity[static_cast<std::size_t>(i)];
        }
        return velocity;
    }
    for (int i = 0; i < 3; ++i) {
        velocity(i) = params.initial_linear_velocity[static_cast<std::size_t>(i)];
        velocity(i + 3) = params.initial_angular_velocity[static_cast<std::size_t>(i)];
    }
    return velocity;
}

Eigen::VectorXd UkfRealtimeFilter::MeasurementFromRigid(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    const Eigen::Matrix3d& reference) {
    Eigen::VectorXd measurement(kMeasurementDims);
    measurement.head<3>() = rigid.get_translation();
    measurement.segment<3>(3) =
        RotVecFromMatrix(reference.transpose() * rigid.get_rotation());
    return measurement;
}

Sn3DAlgorithm::RigidMatrix UkfRealtimeFilter::RigidFromMeasurement(
    const Eigen::VectorXd& measurement,
    const Eigen::Matrix3d& reference) {
    const Eigen::Matrix3d rotation =
        reference * MatrixFromRotVec(measurement.segment<3>(3));
    return Sn3DAlgorithm::RigidMatrix(rotation, measurement.head<3>());
}

Eigen::MatrixXd UkfRealtimeFilter::SigmaPoints(
    const Eigen::VectorXd& state,
    const Eigen::MatrixXd& covariance,
    double scale) {
    const int dim = static_cast<int>(state.size());
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim, dim);
    double jitter = 1e-12;
    Eigen::MatrixXd root(dim, dim);
    bool ok = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
        Eigen::LLT<Eigen::MatrixXd> llt(scale * (Symmetrize(covariance) + jitter * identity));
        if (llt.info() == Eigen::Success) {
            root = llt.matrixL();
            ok = true;
            break;
        }
        jitter *= 10.0;
    }
    if (!ok) {
        Eigen::LLT<Eigen::MatrixXd> llt(scale * (Symmetrize(covariance) + jitter * identity));
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("UKF covariance is not positive definite");
        }
        root = llt.matrixL();
    }

    Eigen::MatrixXd points(2 * dim + 1, dim);
    points.row(0) = state.transpose();
    for (int idx = 0; idx < dim; ++idx) {
        points.row(idx + 1) = (state + root.col(idx)).transpose();
        points.row(idx + 1 + dim) = (state - root.col(idx)).transpose();
    }
    return points;
}

Eigen::VectorXd UkfRealtimeFilter::PredictSigma(
    const Eigen::VectorXd& point,
    int dims,
    double dt,
    const std::string& motion_model) {
    Eigen::VectorXd predicted = point;
    if (motion_model == "constant_acceleration") {
        predicted.head(dims) =
            point.head(dims) + point.segment(dims, dims) * dt +
            0.5 * point.tail(dims) * dt * dt;
        predicted.segment(dims, dims) = point.segment(dims, dims) + point.tail(dims) * dt;
    } else {
        predicted.head(dims) = point.head(dims) + point.segment(dims, dims) * dt;
    }
    return predicted;
}

Eigen::MatrixXd UkfRealtimeFilter::ProcessCovariance(
    int dims,
    int order,
    double dt,
    double process_noise) {
    Eigen::MatrixXd q1d;
    if (order == 3) {
        q1d.resize(3, 3);
        q1d << std::pow(dt, 6) / 36.0, std::pow(dt, 5) / 12.0, std::pow(dt, 4) / 6.0,
            std::pow(dt, 5) / 12.0, std::pow(dt, 4) / 4.0, std::pow(dt, 3) / 2.0,
            std::pow(dt, 4) / 6.0, std::pow(dt, 3) / 2.0, dt * dt;
    } else {
        q1d.resize(2, 2);
        q1d << std::pow(dt, 4) / 4.0, std::pow(dt, 3) / 2.0,
            std::pow(dt, 3) / 2.0, dt * dt;
    }
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(dims * order, dims * order);
    for (int row = 0; row < order; ++row) {
        for (int col = 0; col < order; ++col) {
            q.block(row * dims, col * dims, dims, dims) =
                Eigen::MatrixXd::Identity(dims, dims) * q1d(row, col);
        }
    }
    return q * process_noise;
}

Eigen::MatrixXd UkfRealtimeFilter::Symmetrize(const Eigen::MatrixXd& matrix) {
    return 0.5 * (matrix + matrix.transpose());
}

Eigen::Vector3d UkfRealtimeFilter::RotVecFromMatrix(const Eigen::Matrix3d& matrix) {
    Eigen::AngleAxisd angle_axis(matrix);
    const double angle = angle_axis.angle();
    if (std::abs(angle) < 1e-14) {
        return Eigen::Vector3d::Zero();
    }
    return angle * angle_axis.axis();
}

Eigen::Matrix3d UkfRealtimeFilter::MatrixFromRotVec(const Eigen::Vector3d& rotvec) {
    const double angle = rotvec.norm();
    if (angle < 1e-14) {
        return Eigen::Matrix3d::Identity();
    }
    return Eigen::AngleAxisd(angle, rotvec / angle).toRotationMatrix();
}

Sn3DAlgorithm::RigidMatrix UkfRealtimeFilter::Initialize(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    OptionalDouble timestamp) {
    motion_model_ = CanonicalMotionModel(params_.motion_model);
    order_ = motion_model_ == "constant_acceleration" ? 3 : 2;
    const int state_dim = kMeasurementDims * order_;
    const double lambda =
        params_.alpha * params_.alpha * (state_dim + params_.kappa) - state_dim;
    scale_ = state_dim + lambda;
    if (scale_ <= 0.0) {
        throw std::invalid_argument("invalid UKF scaling; increase alpha or kappa");
    }

    weights_mean_ = Eigen::VectorXd::Constant(2 * state_dim + 1, 0.5 / scale_);
    weights_cov_ = weights_mean_;
    weights_mean_(0) = lambda / scale_;
    weights_cov_(0) = lambda / scale_ + (1.0 - params_.alpha * params_.alpha + params_.beta);

    reference_rotation_ = rigid.get_rotation();
    state_ = Eigen::VectorXd::Zero(state_dim);
    state_.head(kMeasurementDims) = MeasurementFromRigid(rigid, reference_rotation_);
    state_.segment(kMeasurementDims, kMeasurementDims) = InitialVelocityVector(params_);
    covariance_ = Eigen::MatrixXd::Identity(state_dim, state_dim) * params_.initial_covariance;
    last_timestamp_ = timestamp;
    initialized_ = true;
    PushHistory(rigid);
    return rigid;
}

double UkfRealtimeFilter::DeltaTime(OptionalDouble timestamp) const {
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

void UkfRealtimeFilter::PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid) {
    history_.push_back(rigid);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<Sn3DAlgorithm::RigidMatrix> FilterUkfTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    UkfParameters params) {
    UkfRealtimeFilter filter(params);
    return filter.FilterTrajectory(rigids, timestamps, true);
}

TimedFilterTrajectoryResult FilterUkfTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    UkfParameters params) {
    UkfRealtimeFilter filter(params);
    return filter.FilterTrajectoryTimed(rigids, timestamps, true);
}

Sn3DAlgorithm::RigidMatrix FilterUkfLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps,
    UkfParameters params) {
    UkfRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(rigids, timestamps);
}

}  // namespace output_alg

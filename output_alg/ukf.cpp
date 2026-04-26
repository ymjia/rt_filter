#include "ukf.hpp"

#include <algorithm>
#include <cctype>
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

UkfRealtimeFilter::Pose UkfRealtimeFilter::Update(
    const Pose& pose,
    std::optional<double> timestamp) {
    if (!initialized_) {
        return Initialize(pose, timestamp);
    }

    const Eigen::VectorXd measurement = MeasurementFromPose(pose, reference_rotation_);
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

    Pose filtered = PoseFromMeasurement(state_.head(kMeasurementDims), reference_rotation_);
    PushHistory(filtered);
    return filtered;
}

std::vector<UkfRealtimeFilter::Pose> UkfRealtimeFilter::FilterTrajectory(
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

UkfRealtimeFilter::Pose UkfRealtimeFilter::FilterLatestFromHistory(
    const std::vector<Pose>& poses,
    const std::vector<double>* timestamps) const {
    UkfRealtimeFilter filter(params_);
    const auto filtered = filter.FilterTrajectory(poses, timestamps, true);
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

Eigen::Matrix3d UkfRealtimeFilter::RotationFromPose(const Pose& pose) {
    Eigen::Matrix3d rotation;
    rotation << pose[0], pose[1], pose[2],
        pose[4], pose[5], pose[6],
        pose[8], pose[9], pose[10];
    return rotation;
}

Eigen::VectorXd UkfRealtimeFilter::MeasurementFromPose(
    const Pose& pose,
    const Eigen::Matrix3d& reference) {
    Eigen::VectorXd measurement(kMeasurementDims);
    measurement(0) = pose[3];
    measurement(1) = pose[7];
    measurement(2) = pose[11];
    measurement.segment<3>(3) =
        RotVecFromMatrix(reference.transpose() * RotationFromPose(pose));
    return measurement;
}

UkfRealtimeFilter::Pose UkfRealtimeFilter::PoseFromMeasurement(
    const Eigen::VectorXd& measurement,
    const Eigen::Matrix3d& reference) {
    Pose pose = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
    const Eigen::Matrix3d rotation =
        reference * MatrixFromRotVec(measurement.segment<3>(3));
    pose[0] = rotation(0, 0);
    pose[1] = rotation(0, 1);
    pose[2] = rotation(0, 2);
    pose[3] = measurement(0);
    pose[4] = rotation(1, 0);
    pose[5] = rotation(1, 1);
    pose[6] = rotation(1, 2);
    pose[7] = measurement(1);
    pose[8] = rotation(2, 0);
    pose[9] = rotation(2, 1);
    pose[10] = rotation(2, 2);
    pose[11] = measurement(2);
    return pose;
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

UkfRealtimeFilter::Pose UkfRealtimeFilter::Initialize(
    const Pose& pose,
    std::optional<double> timestamp) {
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

    reference_rotation_ = RotationFromPose(pose);
    state_ = Eigen::VectorXd::Zero(state_dim);
    state_.head(kMeasurementDims) = MeasurementFromPose(pose, reference_rotation_);
    state_.segment(kMeasurementDims, kMeasurementDims) = InitialVelocityVector(params_);
    covariance_ = Eigen::MatrixXd::Identity(state_dim, state_dim) * params_.initial_covariance;
    last_timestamp_ = timestamp;
    initialized_ = true;
    PushHistory(pose);
    return pose;
}

double UkfRealtimeFilter::DeltaTime(std::optional<double> timestamp) const {
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

void UkfRealtimeFilter::PushHistory(const Pose& pose) {
    history_.push_back(pose);
    if (params_.history_size > 0) {
        while (history_.size() > params_.history_size) {
            history_.pop_front();
        }
    }
}

std::vector<UkfRealtimeFilter::Pose> FilterUkfTrajectory(
    const std::vector<UkfRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps,
    UkfParameters params) {
    UkfRealtimeFilter filter(params);
    return filter.FilterTrajectory(poses, timestamps, true);
}

UkfRealtimeFilter::Pose FilterUkfLatestFromHistory(
    const std::vector<UkfRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps,
    UkfParameters params) {
    UkfRealtimeFilter filter(params);
    return filter.FilterLatestFromHistory(poses, timestamps);
}

}  // namespace output_alg

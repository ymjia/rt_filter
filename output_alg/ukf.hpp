#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace output_alg {

struct UkfParameters {
    // Supported values: "constant_velocity"/"cv" and "constant_acceleration"/"ca".
    std::string motion_model = "constant_velocity";

    // Higher process_noise follows motion faster. Higher measurement_noise smooths more.
    double process_noise = 1000.0;
    double measurement_noise = 0.001;
    double initial_covariance = 1.0;

    // Initial velocity is [vx, vy, vz, wx, wy, wz]. Linear velocity uses pose
    // distance units per second; angular velocity uses radians per second.
    bool use_initial_velocity = false;
    std::array<double, 6> initial_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 3> initial_linear_velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> initial_angular_velocity = {0.0, 0.0, 0.0};

    double alpha = 1e-3;
    double beta = 2.0;
    double kappa = 0.0;
    double sample_rate_hz = 100.0;
    std::size_t history_size = 0;  // 0 means keep all filtered frames in history.
    bool strict_timestamps = false;
};

class UkfRealtimeFilter {
public:
    using Pose = std::array<double, 16>;  // Row-major 4x4 pose matrix.

    explicit UkfRealtimeFilter(UkfParameters params = UkfParameters{});

    void Reset();
    void SetParameters(const UkfParameters& params, bool reset = false);
    const UkfParameters& Parameters() const { return params_; }

    // Filter one incoming pose and return one display/output pose.
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
    static constexpr int kMeasurementDims = 6;

    static void Validate(const UkfParameters& params);
    static std::string CanonicalMotionModel(const std::string& value);
    static Eigen::VectorXd InitialVelocityVector(const UkfParameters& params);
    static Eigen::Matrix3d RotationFromPose(const Pose& pose);
    static Eigen::VectorXd MeasurementFromPose(const Pose& pose, const Eigen::Matrix3d& reference);
    static Pose PoseFromMeasurement(const Eigen::VectorXd& measurement, const Eigen::Matrix3d& reference);
    static Eigen::MatrixXd SigmaPoints(
        const Eigen::VectorXd& state,
        const Eigen::MatrixXd& covariance,
        double scale);
    static Eigen::VectorXd PredictSigma(
        const Eigen::VectorXd& point,
        int dims,
        double dt,
        const std::string& motion_model);
    static Eigen::MatrixXd ProcessCovariance(int dims, int order, double dt, double process_noise);
    static Eigen::MatrixXd Symmetrize(const Eigen::MatrixXd& matrix);
    static Eigen::Vector3d RotVecFromMatrix(const Eigen::Matrix3d& matrix);
    static Eigen::Matrix3d MatrixFromRotVec(const Eigen::Vector3d& rotvec);

    Pose Initialize(const Pose& pose, std::optional<double> timestamp);
    double DeltaTime(std::optional<double> timestamp) const;
    void PushHistory(const Pose& pose);

    UkfParameters params_;
    bool initialized_ = false;
    std::string motion_model_ = "constant_velocity";
    int order_ = 2;
    double scale_ = 0.0;
    Eigen::Matrix3d reference_rotation_ = Eigen::Matrix3d::Identity();
    Eigen::VectorXd state_;
    Eigen::MatrixXd covariance_;
    Eigen::VectorXd weights_mean_;
    Eigen::VectorXd weights_cov_;
    std::optional<double> last_timestamp_;
    std::deque<Pose> history_;
};

std::vector<UkfRealtimeFilter::Pose> FilterUkfTrajectory(
    const std::vector<UkfRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps = nullptr,
    UkfParameters params = UkfParameters{});

UkfRealtimeFilter::Pose FilterUkfLatestFromHistory(
    const std::vector<UkfRealtimeFilter::Pose>& poses,
    const std::vector<double>* timestamps = nullptr,
    UkfParameters params = UkfParameters{});

}  // namespace output_alg

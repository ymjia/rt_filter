#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "optional_double.hpp"
#include "RigidMatrix.h"

namespace output_alg {

// UkfParameters 保存 SE(3) UKF 滤波器的全部运行参数。
//
// 当前 UKF 把每帧 RigidMatrix 刚体变换转换成 6 维测量：
// [x, y, z, rx, ry, rz]。
// 其中 r* 是相对第一帧参考姿态的旋转向量，单位为 rad。状态中还包含该 6 维
// 测量的一阶速度；constant_acceleration 模型还会额外包含二阶加速度。
struct UkfParameters {
    // 运动模型。支持：
    // - "constant_velocity" 或 "cv"：状态为 [pose6, velocity6]
    // - "constant_acceleration" 或 "ca"：状态为 [pose6, velocity6, acceleration6]
    std::string motion_model = "constant_velocity";

    // 过程噪声。值越大，越允许真实运动偏离当前运动模型，输出更跟手；
    // 值越小，越相信“速度/加速度应连续”，平滑更强但拖影风险更高。
    double process_noise = 1000.0;

    // 观测噪声。值越大，越不相信输入位姿测量，平滑更强；值越小，越贴近原始轨迹。
    double measurement_noise = 0.001;

    // 初始状态协方差。主要影响滤波开始的前几帧，通常保持默认即可。
    double initial_covariance = 1.0;

    // 初始速度可以用两种形式传入：
    // 1. use_initial_velocity=true 时，使用完整 6 维 initial_velocity；
    // 2. 否则使用 initial_linear_velocity + initial_angular_velocity。
    //
    // 6 维顺序为 [vx, vy, vz, wx, wy, wz]。
    // 线速度单位是输入坐标单位/秒，例如 mm/s；角速度单位为 rad/s。
    bool use_initial_velocity = false;
    std::array<double, 6> initial_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::array<double, 3> initial_linear_velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> initial_angular_velocity = {0.0, 0.0, 0.0};

    // UKF sigma 点分布参数。一般情况下不需要调。alpha 过小/过大都会影响数值稳定
    // 和 sigma 点覆盖范围；beta=2 常用于近似高斯分布；kappa 通常保持 0。
    double alpha = 1e-3;
    double beta = 2.0;
    double kappa = 0.0;

    // 当 Update/FilterTrajectory 没有传入时间戳时，用该采样率计算 dt。
    // 如果传入了严格递增时间戳，则优先使用真实帧间隔。
    double sample_rate_hz = 100.0;

    // RealtimeFilter 内部保留的输出历史长度。0 表示不限制长度。
    // 这不影响滤波状态，只影响 History() 返回的缓存大小。
    std::size_t history_size = 0;

    // true 时，非递增时间戳会抛异常；false 时会回退到 sample_rate_hz 对应的 dt。
    bool strict_timestamps = false;
};

// UkfRealtimeFilter 是有状态的实时 SE(3) UKF 滤波器。
//
// 典型用法是每个轨迹/目标/groupId 长期持有一个对象，每帧调用 Update。
// 滤波器内部会保存上一帧状态、协方差、初始参考姿态和时间戳。不要每帧临时构造，
// 否则速度/协方差历史会丢失，UKF 会退化成几乎无历史的单帧处理。
class UkfRealtimeFilter {
public:
    // 使用给定参数创建滤波器，并初始化内部状态。构造时会校验参数合法性。
    explicit UkfRealtimeFilter(UkfParameters params = UkfParameters{});

    // 清空状态、协方差、参考姿态、时间戳和输出历史。参数本身不变。
    void Reset();

    // 更新参数。reset=true 时同时清空状态，适合换用一组完全不同的参数；
    // reset=false 时保留当前状态，仅影响后续帧预测和更新。
    void SetParameters(const UkfParameters& params, bool reset = false);

    // 返回当前参数引用，便于外部界面读取或记录当前生效配置。
    const UkfParameters& Parameters() const { return params_; }

    // 单帧实时接口。输入一帧 RigidMatrix，返回滤波后的显示/输出刚体变换。
    // timestamp 单位为秒；未提供时使用 sample_rate_hz。
    // 第一帧会用于初始化状态和参考姿态，通常会原样输出。
    Sn3DAlgorithm::RigidMatrix Update(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    // 整段轨迹接口。对每一帧顺序调用 Update，并返回与输入等长的滤波结果。
    // timestamps 为 nullptr 时使用固定采样率；非空时长度必须与 rigids 一致。
    // reset=true 表示处理前先从空状态开始，适合离线处理一整条新轨迹。
    std::vector<Sn3DAlgorithm::RigidMatrix> FilterTrajectory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    // 历史窗口接口。传入一段最近帧历史，只返回最后一帧的滤波位姿。
    // 适合界面刷新时“用最近 N 帧计算当前显示位姿”的调用方式。
    // 该接口不会修改当前对象状态；内部会临时创建同参数滤波器。
    Sn3DAlgorithm::RigidMatrix FilterLatestFromHistory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr) const;

    // 已输出的滤波历史。是否限制长度由 UkfParameters::history_size 决定。
    const std::deque<Sn3DAlgorithm::RigidMatrix>& History() const { return history_; }

private:
    static constexpr int kMeasurementDims = 6;

    static void Validate(const UkfParameters& params);
    static std::string CanonicalMotionModel(const std::string& value);
    static Eigen::VectorXd InitialVelocityVector(const UkfParameters& params);
    static Eigen::VectorXd MeasurementFromRigid(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        const Eigen::Matrix3d& reference);
    static Sn3DAlgorithm::RigidMatrix RigidFromMeasurement(
        const Eigen::VectorXd& measurement,
        const Eigen::Matrix3d& reference);
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

    Sn3DAlgorithm::RigidMatrix Initialize(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp);
    double DeltaTime(OptionalDouble timestamp) const;
    void PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid);

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
    OptionalDouble last_timestamp_;
    std::deque<Sn3DAlgorithm::RigidMatrix> history_;
};

// 无状态便捷函数：构造一个临时 UkfRealtimeFilter，处理整段轨迹并返回结果。
// 适合离线工具、demo、测试代码；实时场景建议直接长期持有 UkfRealtimeFilter。
std::vector<Sn3DAlgorithm::RigidMatrix> FilterUkfTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    UkfParameters params = UkfParameters{});

// 无状态便捷函数：用同一组参数处理最近历史，并只返回最后一帧滤波结果。
Sn3DAlgorithm::RigidMatrix FilterUkfLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    UkfParameters params = UkfParameters{});

}  // namespace output_alg

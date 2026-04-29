#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <vector>

#include <Eigen/Core>

#include "filter_timing.hpp"
#include "optional_double.hpp"
#include "RigidMatrix.h"

namespace output_alg {

// ButterworthParameters 保存实时因果 Butterworth 低通滤波器的全部运行参数。
//
// 与 Python 侧的离线 zero-phase `butterworth` / `butterworth_z` 不同，这里的 C++
// 实时版本是有状态、因果的 IIR 低通。它适合扫描界面、实时显示或逐帧处理链路；
// 代价是会引入相位延迟，不能与离线双向滤波的波形完全等价。
struct ButterworthParameters {
    // 低通截止频率，单位 Hz。应严格小于 Nyquist 频率（sample_rate_hz / 2）。
    // 值越小平滑越强，但真实高频波动也会更容易被压掉。
    double cutoff_hz = 20.0;

    // Butterworth 阶数。1 表示一阶；>=2 时内部会自动按一阶/二阶节级联实现。
    int order = 2;

    // 当 Update/FilterTrajectory 没有传入时间戳时，用该采样率计算 dt。
    // 如果传入了严格递增时间戳，则会按实际帧间隔重建本帧使用的 IIR 系数。
    double sample_rate_hz = 100.0;

    // RealtimeFilter 内部保留的输出历史长度。0 表示不限制长度。
    std::size_t history_size = 0;

    // 0 keeps the original causal update. Values > 0 enable fixed-lag smoothing
    // for Z-only realtime filtering: update N returns frame N - delay_frames,
    // filtered with up to 2 * delay_frames + 1 recent raw frames.
    std::size_t delay_frames = 0;

    // true 时，非递增时间戳会抛异常；false 时会回退到 sample_rate_hz 对应的 dt。
    bool strict_timestamps = false;
};

using ButterworthZParameters = ButterworthParameters;

struct ButterworthSectionState {
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
};

// ButterworthRealtimeFilter 是有状态的实时三轴平移 Butterworth 低通滤波器。
//
// 它同时处理位姿矩阵中的 X/Y/Z 平移分量，旋转矩阵保持输入值不变。
// 典型用法是每个轨迹/目标/groupId 长期持有一个对象，然后每帧调用 Update。
class ButterworthRealtimeFilter {
public:
    explicit ButterworthRealtimeFilter(ButterworthParameters params = ButterworthParameters{});

    void Reset();
    void SetParameters(const ButterworthParameters& params, bool reset = false);

    const ButterworthParameters& Parameters() const { return params_; }

    Sn3DAlgorithm::RigidMatrix Update(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    TimedRigidResult UpdateTimed(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    std::vector<Sn3DAlgorithm::RigidMatrix> FilterTrajectory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    TimedFilterTrajectoryResult FilterTrajectoryTimed(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    Sn3DAlgorithm::RigidMatrix FilterLatestFromHistory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr) const;

    const std::deque<Sn3DAlgorithm::RigidMatrix>& History() const { return history_; }

private:
    static void Validate(const ButterworthParameters& params);
    double DeltaTime(OptionalDouble timestamp) const;
    void Redesign(double sample_rate_hz, const Eigen::Vector3d& anchor_value);
    void PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid);

    ButterworthParameters params_;
    bool initialized_ = false;
    double current_sample_rate_hz_ = 0.0;
    Eigen::Vector3d last_output_ = Eigen::Vector3d::Zero();
    OptionalDouble last_timestamp_;
    std::array<std::vector<ButterworthSectionState>, 3> sections_;
    std::deque<Sn3DAlgorithm::RigidMatrix> history_;
};

// ButterworthZRealtimeFilter 是有状态的实时 Z-only Butterworth 低通滤波器。
//
// 它只修改位姿矩阵中的 translation.z()；translation.x/y 和 rotation 会保持输入值。
class ButterworthZRealtimeFilter {
public:
    explicit ButterworthZRealtimeFilter(ButterworthZParameters params = ButterworthZParameters{});

    void Reset();
    void SetParameters(const ButterworthZParameters& params, bool reset = false);

    const ButterworthZParameters& Parameters() const { return params_; }

    Sn3DAlgorithm::RigidMatrix Update(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    TimedRigidResult UpdateTimed(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    std::vector<Sn3DAlgorithm::RigidMatrix> FilterTrajectory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    TimedFilterTrajectoryResult FilterTrajectoryTimed(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    Sn3DAlgorithm::RigidMatrix FilterLatestFromHistory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr) const;

    const std::deque<Sn3DAlgorithm::RigidMatrix>& History() const { return history_; }

private:
    static void Validate(const ButterworthZParameters& params);
    double DeltaTime(OptionalDouble timestamp) const;
    void Redesign(double sample_rate_hz, double anchor_value);
    void PushDelayBuffer(const Sn3DAlgorithm::RigidMatrix& rigid, OptionalDouble timestamp);
    Sn3DAlgorithm::RigidMatrix DelayedWindowOutput() const;
    void PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid);

    ButterworthZParameters params_;
    bool initialized_ = false;
    double current_sample_rate_hz_ = 0.0;
    double last_output_z_ = 0.0;
    OptionalDouble last_timestamp_;
    std::vector<ButterworthSectionState> sections_;
    std::deque<Sn3DAlgorithm::RigidMatrix> raw_buffer_;
    std::deque<OptionalDouble> timestamp_buffer_;
    std::deque<Sn3DAlgorithm::RigidMatrix> history_;
};

std::vector<Sn3DAlgorithm::RigidMatrix> FilterButterworthTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthParameters params = ButterworthParameters{});

TimedFilterTrajectoryResult FilterButterworthTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthParameters params = ButterworthParameters{});

Sn3DAlgorithm::RigidMatrix FilterButterworthLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthParameters params = ButterworthParameters{});

std::vector<Sn3DAlgorithm::RigidMatrix> FilterButterworthZTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthZParameters params = ButterworthZParameters{});

TimedFilterTrajectoryResult FilterButterworthZTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthZParameters params = ButterworthZParameters{});

Sn3DAlgorithm::RigidMatrix FilterButterworthZLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    ButterworthZParameters params = ButterworthZParameters{});

}  // namespace output_alg

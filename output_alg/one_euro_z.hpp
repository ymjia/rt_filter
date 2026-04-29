#pragma once

#include <cstddef>
#include <deque>
#include <vector>

#include "filter_timing.hpp"
#include "optional_double.hpp"
#include "RigidMatrix.h"

namespace output_alg {

// OneEuroZParameters 保存 One Euro Z 滤波器的全部运行参数。
//
// 这个滤波器只处理位姿矩阵中的 Z 平移分量，适合“深度方向噪声明显大于 X/Y”
// 的实时显示或整段轨迹降噪场景。X/Y 平移和旋转矩阵会原样拷贝到输出。
struct OneEuroZParameters {
    // 静止或低速时的最小截止频率。值越小，静止抖动压制越强，但真实 Z 向运动
    // 会更容易产生滞后；值越大，输出越跟手，但静止时降噪变弱。
    double min_cutoff = 1.0;

    // 根据 Z 方向速度动态提高截止频率的强度。值越大，检测到运动时越快放开
    // 低通，拖影越小；代价是运动过程中的噪声会保留更多。
    double beta = 10.0;

    // 对 Z 方向速度估计做低通时使用的截止频率。它影响“当前是否在运动”的
    // 判断稳定性，通常不需要频繁调整。
    double d_cutoff = 8.0;

    // Z 速度死区，单位与输入坐标单位/秒一致。低于该阈值的速度会被视作静止
    // 抖动，不触发 beta 自适应放开滤波。增大该值可增强静止降噪，但会增加慢速
    // Z 向运动的滞后风险。
    double derivative_deadband = 0.02;

    // 当 Update/FilterTrajectory 没有传入时间戳时，用该采样率计算 dt。
    // 如果传入了严格递增时间戳，则优先使用真实帧间隔。
    double sample_rate_hz = 100.0;

    // RealtimeFilter 内部保留的输出历史长度。0 表示不限制长度；如果只用于界面
    // 显示，通常可设置一个较小窗口，避免长期运行时 history_ 增长。
    std::size_t history_size = 0;

    // true 时，非递增时间戳会抛异常；false 时会回退到 sample_rate_hz 对应的 dt。
    bool strict_timestamps = false;
};

// OneEuroZRealtimeFilter 是有状态的实时滤波器。
//
// 典型用法是每个轨迹/目标/groupId 长期持有一个对象，然后每帧调用 Update。
// 不要每帧临时创建滤波器，否则上一帧的滤波状态会丢失，One Euro 的平滑效果
// 和实时自适应都会失效。
class OneEuroZRealtimeFilter {
public:
    // 使用给定参数创建滤波器，并初始化内部状态。构造时会校验参数合法性。
    explicit OneEuroZRealtimeFilter(OneEuroZParameters params = OneEuroZParameters{});

    // 清空滤波状态和历史输出。参数本身不变。
    void Reset();

    // 更新参数。reset=true 时同时清空状态，适合用户切换到一组全新参数；
    // reset=false 时保留已有滤波状态，仅调整后续帧使用的参数。
    void SetParameters(const OneEuroZParameters& params, bool reset = false);

    // 返回当前参数引用，便于界面或调用方读取实际生效配置。
    const OneEuroZParameters& Parameters() const { return params_; }

    // 单帧实时接口。输入一帧 RigidMatrix，返回滤波后的显示/输出刚体变换。
    // 仅 translation.z() 会被修改；translation.x/y 和 rotation 会保持输入值。
    // timestamp 单位为秒；未提供时使用 sample_rate_hz。
    Sn3DAlgorithm::RigidMatrix Update(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    // 单帧实时接口的计时版本。返回本帧滤波结果和该帧计算耗时（ns）。
    TimedRigidResult UpdateTimed(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    // 整段轨迹接口。对每一帧顺序调用 Update，并返回与输入等长的滤波结果。
    // timestamps 为 nullptr 时使用固定采样率；非空时长度必须与 rigids 一致。
    // reset=true 表示处理前先从空状态开始，适合离线处理一整条新轨迹。
    std::vector<Sn3DAlgorithm::RigidMatrix> FilterTrajectory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    // 整段轨迹接口的计时版本。返回逐帧滤波结果、逐帧耗时和总耗时（ns）。
    TimedFilterTrajectoryResult FilterTrajectoryTimed(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr,
        bool reset = true);

    // 历史窗口接口。传入一段最近帧历史，只返回最后一帧的滤波位姿。
    // 适合界面刷新时“用最近 N 帧计算当前显示位姿”的调用方式。
    // 该接口不会修改当前对象状态；内部会临时创建同参数滤波器。
    Sn3DAlgorithm::RigidMatrix FilterLatestFromHistory(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps = nullptr) const;

    // 已输出的滤波历史。是否限制长度由 OneEuroZParameters::history_size 决定。
    const std::deque<Sn3DAlgorithm::RigidMatrix>& History() const { return history_; }

private:
    static void Validate(const OneEuroZParameters& params);
    static double LowpassAlpha(double cutoff, double dt);
    double DeltaTime(OptionalDouble timestamp) const;
    void PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid);

    OneEuroZParameters params_;
    bool initialized_ = false;
    double last_raw_z_ = 0.0;
    double filtered_z_ = 0.0;
    double derivative_hat_ = 0.0;
    OptionalDouble last_timestamp_;
    std::deque<Sn3DAlgorithm::RigidMatrix> history_;
};

// 无状态便捷函数：构造一个临时 OneEuroZRealtimeFilter，处理整段轨迹并返回结果。
// 适合离线工具、demo、测试代码；实时场景建议直接长期持有 OneEuroZRealtimeFilter。
std::vector<Sn3DAlgorithm::RigidMatrix> FilterOneEuroZTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    OneEuroZParameters params = OneEuroZParameters{});

TimedFilterTrajectoryResult FilterOneEuroZTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    OneEuroZParameters params = OneEuroZParameters{});

// 无状态便捷函数：用同一组参数处理最近历史，并只返回最后一帧滤波结果。
Sn3DAlgorithm::RigidMatrix FilterOneEuroZLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    OneEuroZParameters params = OneEuroZParameters{});

}  // namespace output_alg

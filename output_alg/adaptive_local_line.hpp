#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "filter_timing.hpp"
#include "optional_double.hpp"
#include "RigidMatrix.h"

namespace output_alg {

struct AdaptiveLocalLineParameters {
    std::size_t window = 5;
    double target_noise_mm = 0.26;
    double max_strength = 0.5;
    double min_strength = 0.0;
    double response = 1.0;
    std::string reference_mode = "global";
    double sample_rate_hz = 100.0;
    bool strict_timestamps = false;
    std::size_t history_size = 0;

    bool use_line_origin = false;
    bool use_line_direction = false;
    Eigen::Vector3d line_origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d line_direction = Eigen::Vector3d::UnitX();
};

class AdaptiveLocalLineRealtimeFilter {
public:
    explicit AdaptiveLocalLineRealtimeFilter(
        AdaptiveLocalLineParameters params = AdaptiveLocalLineParameters{});

    void Reset();
    void SetParameters(const AdaptiveLocalLineParameters& params, bool reset = false);
    const AdaptiveLocalLineParameters& Parameters() const { return params_; }

    Sn3DAlgorithm::RigidMatrix Update(
        const Sn3DAlgorithm::RigidMatrix& rigid,
        OptionalDouble timestamp = OptionalDouble());

    Eigen::Vector3d Update(
        const Eigen::Vector3d& point,
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
    struct Line {
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        Eigen::Vector3d direction = Eigen::Vector3d::UnitX();
    };

    static void Validate(const AdaptiveLocalLineParameters& params);
    static std::string CanonicalReferenceMode(const std::string& mode);
    static Line FitPrincipalLine(const std::vector<Eigen::Vector3d>& points);
    static Line ReferenceLine(
        const std::vector<Eigen::Vector3d>& points,
        const AdaptiveLocalLineParameters& params);
    static double Strength(
        double noise,
        double target_noise,
        double min_strength,
        double max_strength,
        double response);
    static Eigen::Vector3d LocalLineCenterPerpResidual(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<double>& relative_times,
        std::size_t center_index,
        double* noise);
    static std::vector<double> TimeValues(
        const std::vector<double>* timestamps,
        std::size_t count,
        double sample_rate_hz,
        bool strict_timestamps);
    static std::vector<Sn3DAlgorithm::RigidMatrix> FilterTrajectoryImpl(
        const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
        const std::vector<double>* timestamps,
        const AdaptiveLocalLineParameters& params);

    Sn3DAlgorithm::RigidMatrix DelayedWindowOutput() const;
    void PushDelayBuffer(const Sn3DAlgorithm::RigidMatrix& rigid, OptionalDouble timestamp);
    void PushHistory(const Sn3DAlgorithm::RigidMatrix& rigid);

    AdaptiveLocalLineParameters params_;
    std::deque<Sn3DAlgorithm::RigidMatrix> raw_buffer_;
    std::deque<OptionalDouble> timestamp_buffer_;
    std::deque<Sn3DAlgorithm::RigidMatrix> history_;
};

std::vector<Sn3DAlgorithm::RigidMatrix> FilterAdaptiveLocalLineTrajectory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    AdaptiveLocalLineParameters params = AdaptiveLocalLineParameters{});

TimedFilterTrajectoryResult FilterAdaptiveLocalLineTrajectoryTimed(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    AdaptiveLocalLineParameters params = AdaptiveLocalLineParameters{});

Sn3DAlgorithm::RigidMatrix FilterAdaptiveLocalLineLatestFromHistory(
    const std::vector<Sn3DAlgorithm::RigidMatrix>& rigids,
    const std::vector<double>* timestamps = nullptr,
    AdaptiveLocalLineParameters params = AdaptiveLocalLineParameters{});

}  // namespace output_alg

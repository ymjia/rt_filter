#pragma once

#include <Eigen/Geometry>

#include "butterworth.hpp"
#include "one_euro_z.hpp"
#include "ukf.hpp"

namespace output_alg {

inline Sn3DAlgorithm::RigidMatrix RigidFromPositionEulerZyx(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& euler_zyx) {
    const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(euler_zyx.x(), Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        Eigen::AngleAxisd(euler_zyx.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() *
        Eigen::AngleAxisd(euler_zyx.z(), Eigen::Vector3d::UnitX()).toRotationMatrix();
    return Sn3DAlgorithm::RigidMatrix(rotation, position);
}

inline void PositionEulerZyxFromRigid(
    const Sn3DAlgorithm::RigidMatrix& rigid,
    Eigen::Vector3d& position,
    Eigen::Vector3d& euler_zyx) {
    position = rigid.get_translation();
    euler_zyx = rigid.get_rotation().eulerAngles(2, 1, 0);
}

inline bool FilterOneEuroZEulerZyx(
    OneEuroZRealtimeFilter& filter,
    Eigen::Vector3d& position,
    Eigen::Vector3d& euler_zyx,
    OptionalDouble timestamp = OptionalDouble()) {
    const Sn3DAlgorithm::RigidMatrix filtered =
        filter.Update(RigidFromPositionEulerZyx(position, euler_zyx), timestamp);
    PositionEulerZyxFromRigid(filtered, position, euler_zyx);
    return true;
}

inline bool FilterButterworthZEulerZyx(
    ButterworthZRealtimeFilter& filter,
    Eigen::Vector3d& position,
    Eigen::Vector3d& euler_zyx,
    OptionalDouble timestamp = OptionalDouble()) {
    const Sn3DAlgorithm::RigidMatrix filtered =
        filter.Update(RigidFromPositionEulerZyx(position, euler_zyx), timestamp);
    PositionEulerZyxFromRigid(filtered, position, euler_zyx);
    return true;
}

inline bool FilterButterworthEulerZyx(
    ButterworthRealtimeFilter& filter,
    Eigen::Vector3d& position,
    Eigen::Vector3d& euler_zyx,
    OptionalDouble timestamp = OptionalDouble()) {
    const Sn3DAlgorithm::RigidMatrix filtered =
        filter.Update(RigidFromPositionEulerZyx(position, euler_zyx), timestamp);
    PositionEulerZyxFromRigid(filtered, position, euler_zyx);
    return true;
}

inline bool FilterUkfEulerZyx(
    UkfRealtimeFilter& filter,
    Eigen::Vector3d& position,
    Eigen::Vector3d& euler_zyx,
    OptionalDouble timestamp = OptionalDouble()) {
    const Sn3DAlgorithm::RigidMatrix filtered =
        filter.Update(RigidFromPositionEulerZyx(position, euler_zyx), timestamp);
    PositionEulerZyxFromRigid(filtered, position, euler_zyx);
    return true;
}

}  // namespace output_alg

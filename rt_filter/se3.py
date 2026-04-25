from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp


ArrayF = NDArray[np.float64]


def as_pose_array(data: ArrayLike) -> ArrayF:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        poses = arr.copy()
    elif arr.ndim == 2 and arr.shape[1] == 16:
        poses = arr.reshape((-1, 4, 4)).copy()
    elif arr.ndim == 2 and arr.shape[1] == 7:
        poses = poses_from_xyz_quat_wxyz(arr)
    else:
        raise ValueError(
            "trajectory data must be shaped as (N,4,4), (N,16), or (N,7)"
        )

    poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return poses


def quat_wxyz_to_xyzw(quat: ArrayLike) -> ArrayF:
    q = np.asarray(quat, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_xyzw_to_wxyz(quat: ArrayLike) -> ArrayF:
    q = np.asarray(quat, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def normalize_quaternions(quat: ArrayLike) -> ArrayF:
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero-length quaternion is not valid")
    return q / norm


def enforce_positive_qw(quat_wxyz: ArrayLike) -> ArrayF:
    q = normalize_quaternions(quat_wxyz)
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def ensure_quaternion_continuity_xyzw(quat_xyzw: ArrayLike) -> ArrayF:
    q = normalize_quaternions(quat_xyzw).copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q


def poses_from_xyz_quat_wxyz(data: ArrayLike) -> ArrayF:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError("xyz + quaternion input must have shape (N, 7)")
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], arr.shape[0], axis=0)
    poses[:, :3, 3] = arr[:, :3]
    quat_xyzw = quat_wxyz_to_xyzw(normalize_quaternions(arr[:, 3:7]))
    poses[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    return poses


def poses_to_xyz_quat_wxyz(poses: ArrayLike, positive_qw: bool = True) -> ArrayF:
    pose_arr = as_pose_array(poses)
    xyz = pose_arr[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(pose_arr[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
    if positive_qw:
        quat_wxyz = enforce_positive_qw(quat_wxyz)
    return np.column_stack([xyz, quat_wxyz])


def make_poses(positions: ArrayLike, rotations: Rotation) -> ArrayF:
    positions_arr = np.asarray(positions, dtype=float)
    if positions_arr.ndim != 2 or positions_arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if len(rotations) != positions_arr.shape[0]:
        raise ValueError("positions and rotations must have the same length")
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], positions_arr.shape[0], axis=0)
    poses[:, :3, 3] = positions_arr
    poses[:, :3, :3] = rotations.as_matrix()
    return poses


def invert_poses(poses: ArrayLike) -> ArrayF:
    pose_arr = as_pose_array(poses)
    inv = np.repeat(np.eye(4, dtype=float)[None, :, :], pose_arr.shape[0], axis=0)
    rot_t = np.swapaxes(pose_arr[:, :3, :3], 1, 2)
    inv[:, :3, :3] = rot_t
    inv[:, :3, 3] = -np.einsum("nij,nj->ni", rot_t, pose_arr[:, :3, 3])
    return inv


def compose_poses(left: ArrayLike, right: ArrayLike) -> ArrayF:
    left_arr = as_pose_array(left)
    right_arr = as_pose_array(right)
    return np.einsum("nij,njk->nik", left_arr, right_arr)


def relative_rotvecs(rotations: Rotation, reference: Rotation | None = None) -> ArrayF:
    if reference is None:
        reference = rotations[0]
    return (reference.inv() * rotations).as_rotvec()


def rotations_from_relative_rotvecs(
    rotvecs: ArrayLike, reference: Rotation
) -> Rotation:
    return reference * Rotation.from_rotvec(np.asarray(rotvecs, dtype=float))


def interpolate_trajectory(
    query_timestamps: ArrayLike,
    source_timestamps: ArrayLike,
    source_poses: ArrayLike,
) -> ArrayF:
    query = np.asarray(query_timestamps, dtype=float)
    source_t = np.asarray(source_timestamps, dtype=float)
    source = as_pose_array(source_poses)
    if source.shape[0] != source_t.shape[0]:
        raise ValueError("source timestamps and poses must have the same length")
    if np.any(np.diff(source_t) <= 0):
        raise ValueError("source timestamps must be strictly increasing")

    clipped = np.clip(query, source_t[0], source_t[-1])
    positions = np.column_stack(
        [
            np.interp(clipped, source_t, source[:, axis, 3])
            for axis in range(3)
        ]
    )
    rotations = Slerp(source_t, Rotation.from_matrix(source[:, :3, :3]))(clipped)
    return make_poses(positions, rotations)


def translation_norm(left_poses: ArrayLike, right_poses: ArrayLike) -> ArrayF:
    left = as_pose_array(left_poses)
    right = as_pose_array(right_poses)
    return np.linalg.norm(left[:, :3, 3] - right[:, :3, 3], axis=1)


def rotation_angle(left_poses: ArrayLike, right_poses: ArrayLike) -> ArrayF:
    left = Rotation.from_matrix(as_pose_array(left_poses)[:, :3, :3])
    right = Rotation.from_matrix(as_pose_array(right_poses)[:, :3, :3])
    return (right.inv() * left).magnitude()

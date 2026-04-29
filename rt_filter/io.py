from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from rt_filter.se3 import (
    as_pose_array,
    make_poses,
    poses_from_xyz_quat_wxyz,
    poses_to_xyz_quat_wxyz,
)
from rt_filter.trajectory import Trajectory


TIME_COLUMNS = ("timestamp", "time", "t")
STATUS_COLUMNS = ("status", "valid")
XYZ_COLUMNS = ("x", "y", "z")
Q_WXYZ_COLUMNS = ("qw", "qx", "qy", "qz")
Q_XYZW_COLUMNS = ("qx", "qy", "qz", "qw")
MATRIX_COLUMNS = tuple(f"m{r}{c}" for r in range(4) for c in range(4))
SN_TRACK_COLUMNS = ("x", "y", "z", "xr", "yr", "zr", "time", "rate")
SUPPORTED_TRAJECTORY_SUFFIXES = (".csv", ".txt", ".json", ".npy", ".npz")


def is_trajectory_file(path: str | Path, *, inspect_json: bool = True) -> bool:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_TRAJECTORY_SUFFIXES or not file_path.is_file():
        return False
    if suffix != ".json" or not inspect_json:
        return True

    lowered_name = file_path.name.lower()
    lowered_stem = file_path.stem.lower()
    if lowered_name in {"manifest.json", "rt_filter_gui.json"}:
        return False
    if lowered_stem.endswith("_metadata") or lowered_stem.endswith("_config"):
        return False

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False

    if isinstance(payload, list):
        return True
    if not isinstance(payload, dict):
        return False
    return "poses" in payload or "samples" in payload


def read_trajectory(path: str | Path, *, drop_invalid: bool = True) -> Trajectory:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(file_path, drop_invalid=drop_invalid)
    if suffix == ".txt":
        return _read_txt(file_path)
    if suffix == ".json":
        return _read_json(file_path, drop_invalid=drop_invalid)
    if suffix == ".npy":
        poses = as_pose_array(np.load(file_path))
        return Trajectory(poses=poses, name=file_path.stem, metadata={"source": str(file_path)})
    if suffix == ".npz":
        return _read_npz(file_path, drop_invalid=drop_invalid)
    raise ValueError(f"unsupported trajectory format: {file_path.suffix}")


def write_trajectory(path_or_traj: str | Path | Trajectory, maybe_path: str | Path | None = None) -> None:
    """Write a trajectory.

    Both ``write_trajectory(traj, path)`` and ``write_trajectory(path, traj)`` are
    accepted to keep CLI and notebook usage convenient.
    """

    if isinstance(path_or_traj, Trajectory):
        traj = path_or_traj
        if maybe_path is None:
            raise ValueError("output path is required")
        path = Path(maybe_path)
    else:
        path = Path(path_or_traj)
        if not isinstance(maybe_path, Trajectory):
            raise ValueError("trajectory is required")
        traj = maybe_path

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        trajectory_to_frame(traj).to_csv(path, index=False)
    elif suffix == ".json":
        payload: dict[str, Any] = {
            "name": traj.name,
            "metadata": traj.metadata,
            "poses": traj.poses.tolist(),
        }
        if traj.timestamps is not None:
            payload["timestamps"] = traj.timestamps.tolist()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif suffix == ".npy":
        np.save(path, traj.poses)
    elif suffix == ".npz":
        payload = {"poses": traj.poses}
        if traj.timestamps is not None:
            payload["timestamps"] = traj.timestamps
        np.savez_compressed(path, **payload)
    else:
        raise ValueError(f"unsupported output format: {path.suffix}")


def trajectory_to_frame(traj: Trajectory) -> pd.DataFrame:
    compact = poses_to_xyz_quat_wxyz(traj.poses, positive_qw=True)
    matrix = traj.poses.reshape((traj.count, 16))
    data: dict[str, Any] = {}
    if traj.timestamps is not None:
        data["timestamp"] = traj.timestamps
    for idx, col in enumerate(("x", "y", "z", "qw", "qx", "qy", "qz")):
        data[col] = compact[:, idx]
    for idx, col in enumerate(MATRIX_COLUMNS):
        data[col] = matrix[:, idx]
    return pd.DataFrame(data)


def _read_csv(path: Path, *, drop_invalid: bool) -> Trajectory:
    frame = pd.read_csv(path, sep=None, engine="python")
    sn_track = _read_sn_track_frame(frame, path=path, name=path.stem, format_name="csv")
    if sn_track is not None:
        return sn_track
    frame = _filter_invalid_rows(frame, drop_invalid=drop_invalid)
    timestamps = _extract_timestamps(frame)
    poses = _poses_from_frame(frame)
    return Trajectory(
        poses=poses,
        timestamps=timestamps,
        name=path.stem,
        metadata={"source": str(path), "format": "csv"},
    )


def _read_txt(path: Path) -> Trajectory:
    frame = pd.read_csv(
        path,
        sep=r"[\t, ]+",
        engine="python",
        header=None,
        names=SN_TRACK_COLUMNS,
    )
    sn_track = _read_sn_track_frame(frame, path=path, name=path.stem, format_name="txt")
    if sn_track is None:
        raise ValueError(
            "unsupported txt trajectory format; expected columns "
            "x y z xr yr zr time rate"
        )
    return sn_track


def _read_json(path: Path, *, drop_invalid: bool) -> Trajectory:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = payload.get("name", path.stem) if isinstance(payload, dict) else path.stem
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    metadata = dict(metadata)
    metadata.update({"source": str(path), "format": "json"})

    if isinstance(payload, list):
        poses = as_pose_array(payload)
        return Trajectory(poses=poses, name=name, metadata=metadata)

    if "samples" in payload:
        frame = pd.DataFrame(payload["samples"])
        frame = _filter_invalid_rows(frame, drop_invalid=drop_invalid)
        return Trajectory(
            poses=_poses_from_frame(frame),
            timestamps=_extract_timestamps(frame),
            name=name,
            metadata=metadata,
        )

    poses = as_pose_array(payload["poses"])
    timestamps = payload.get("timestamps")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float)
    status = np.asarray(payload.get("status", []))
    if drop_invalid and status.size:
        valid = status != -1
        poses = poses[valid]
        if timestamps is not None:
            timestamps = timestamps[valid]
    return Trajectory(poses=poses, timestamps=timestamps, name=name, metadata=metadata)


def _read_npz(path: Path, *, drop_invalid: bool) -> Trajectory:
    archive = np.load(path)
    key = next((candidate for candidate in ("poses", "trajectory", "data") if candidate in archive), None)
    if key is None:
        keys = list(archive.keys())
        if len(keys) != 1:
            raise ValueError(f"cannot infer pose array from npz keys: {keys}")
        key = keys[0]
    poses = as_pose_array(archive[key])
    timestamps = np.asarray(archive["timestamps"], dtype=float) if "timestamps" in archive else None
    if drop_invalid and "status" in archive:
        valid = np.asarray(archive["status"]) != -1
        poses = poses[valid]
        if timestamps is not None:
            timestamps = timestamps[valid]
    return Trajectory(
        poses=poses,
        timestamps=timestamps,
        name=path.stem,
        metadata={"source": str(path), "format": "npz"},
    )


def _filter_invalid_rows(frame: pd.DataFrame, *, drop_invalid: bool) -> pd.DataFrame:
    if not drop_invalid:
        return frame
    for col in STATUS_COLUMNS:
        if col in frame.columns:
            if col == "valid":
                return frame.loc[frame[col].astype(bool)].reset_index(drop=True)
            return frame.loc[frame[col] != -1].reset_index(drop=True)
    return frame


def _extract_timestamps(frame: pd.DataFrame) -> np.ndarray | None:
    for col in TIME_COLUMNS:
        if col in frame.columns:
            return frame[col].to_numpy(dtype=float)
    return None


def _poses_from_frame(frame: pd.DataFrame) -> np.ndarray:
    lower_to_original = {str(col).lower(): col for col in frame.columns}

    if all(col in lower_to_original for col in MATRIX_COLUMNS):
        matrix_frame = frame[[lower_to_original[col] for col in MATRIX_COLUMNS]]
        return as_pose_array(matrix_frame.to_numpy(dtype=float))

    if all(col in lower_to_original for col in (*XYZ_COLUMNS, *Q_WXYZ_COLUMNS)):
        cols = [lower_to_original[col] for col in (*XYZ_COLUMNS, *Q_WXYZ_COLUMNS)]
        return poses_from_xyz_quat_wxyz(frame[cols].to_numpy(dtype=float))

    if all(col in lower_to_original for col in (*XYZ_COLUMNS, *Q_XYZW_COLUMNS)):
        cols = [lower_to_original[col] for col in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        return poses_from_xyz_quat_wxyz(frame[cols].to_numpy(dtype=float))

    numeric_cols = [
        col
        for col in frame.select_dtypes(include=["number"]).columns
        if str(col).lower() not in {*TIME_COLUMNS, *STATUS_COLUMNS}
    ]
    if len(numeric_cols) == 16:
        return as_pose_array(frame[numeric_cols].to_numpy(dtype=float))
    if len(numeric_cols) == 7:
        return poses_from_xyz_quat_wxyz(frame[numeric_cols].to_numpy(dtype=float))

    raise ValueError(
        "cannot infer trajectory columns; expected matrix columns m00..m33 or "
        "x,y,z,qw,qx,qy,qz"
    )


def _read_sn_track_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    name: str,
    format_name: str,
) -> Trajectory | None:
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    required_columns = ("x", "y", "z", "xr", "yr", "zr")
    if not all(col in lower_to_original for col in required_columns):
        return None

    selected_columns = [
        lower_to_original[col]
        for col in ("x", "y", "z", "xr", "yr", "zr")
        if col in lower_to_original
    ]
    if "time" in lower_to_original:
        selected_columns.append(lower_to_original["time"])
    if "rate" in lower_to_original:
        selected_columns.append(lower_to_original["rate"])

    numeric = frame[selected_columns].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if numeric.empty:
        raise ValueError(f"{path} does not contain valid numeric SN trajectory rows")

    positions = numeric[[lower_to_original["x"], lower_to_original["y"], lower_to_original["z"]]].to_numpy(
        dtype=float
    )
    rotations = Rotation.from_euler(
        "xyz",
        numeric[[lower_to_original["xr"], lower_to_original["yr"], lower_to_original["zr"]]].to_numpy(dtype=float),
        degrees=True,
    )
    poses = make_poses(positions, rotations)
    timestamps, timestamp_source = _sn_track_timestamps(numeric)
    metadata: dict[str, Any] = {
        "source": str(path),
        "format": format_name,
        "position_unit": "mm",
        "rotation_assumption": "XYZ Euler angles in degrees",
        "timestamp_source": timestamp_source,
    }
    return Trajectory(
        poses=poses,
        timestamps=timestamps,
        name=name,
        metadata=metadata,
    )


def _sn_track_timestamps(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "rate" in frame.columns:
        rate = float(frame["rate"].replace(0, np.nan).dropna().median())
        if np.isfinite(rate) and rate > 0:
            return np.arange(len(frame), dtype=float) / rate, "generated from rate column"

    if "time" in frame.columns:
        time_values = frame["time"].to_numpy(dtype=float)
        if time_values.size >= 2:
            dt = np.diff(time_values)
            if np.all(dt > 0):
                return time_values - time_values[0], "copied from time column"

    return np.arange(len(frame), dtype=float), "generated from sample index"

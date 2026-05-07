from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from rt_filter.io import write_trajectory
from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory


SOURCE_ROOT = Path("ref_data/sn")
INPUT_ROOT = Path("input/sn")
TXT_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "time", "rate"]
CASE0506_COLUMNS = ["time", "x", "y", "z", "a", "b", "c"]
MAX_CONTINUOUS_STEP_S = 0.05
SPECIAL_CASE_FOLDERS = {"\u672c\u5730\u5b9e\u9a8c\u5ba4\u6570\u636e"}
CASE0506_CASES = [
    {
        "case_id": "case_31_dynamic_case0506_scan",
        "case_label": "dynamic_robot_arm",
        "source_path": Path("ref_data/case0506_scan.txt"),
        "variant_id": "case0506_scan",
        "variant_note": "case0506 scan trajectory",
    },
    {
        "case_id": "case_32_dynamic_case0506_sn",
        "case_label": "dynamic_robot_arm",
        "source_path": Path("ref_data/case0506_sn.txt"),
        "variant_id": "case0506_sn",
        "variant_note": "case0506 sn trajectory",
    },
]


@dataclass(frozen=True)
class FolderInfo:
    case_id: str
    case_label: str
    base: str
    target: str
    calibration: str
    exposure: str
    added_points: bool


def main() -> None:
    rows = prepare_sn_data()
    print(f"wrote {len(rows)} trajectories to {INPUT_ROOT}")


def prepare_sn_data() -> list[dict[str, Any]]:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"SN source directory not found: {SOURCE_ROOT}")

    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    folders = [
        path
        for path in sorted(SOURCE_ROOT.iterdir())
        if path.is_dir() and path.name not in SPECIAL_CASE_FOLDERS
    ]
    for folder_index, folder in enumerate(folders, start=1):
        info = _folder_info(folder.name, folder_index)
        case_dir = INPUT_ROOT / info.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        workbook = folder / "数据稳定性.xlsx"

        for file_index, txt_path in enumerate(sorted(folder.glob("track_data_*.txt")), start=1):
            variant = _variant_from_filename(txt_path.name)
            output_stem = f"{info.case_id}_{file_index:02d}_{variant['variant_id']}"
            output_path = case_dir / f"{output_stem}.csv"
            traj, raw_row_count = _read_sn_track(txt_path, output_stem)
            write_trajectory(traj, output_path)

            rows.append(
                {
                    "trajectory": output_stem,
                    "case_id": info.case_id,
                    "case_label": info.case_label,
                    "source_folder": folder.name,
                    "source_file": txt_path.name,
                    "source_path": str(txt_path),
                    "input_csv": str(output_path),
                    "source_workbook": str(workbook) if workbook.exists() else "",
                    "base": info.base,
                    "target": info.target,
                    "calibration": info.calibration,
                    "exposure": info.exposure,
                    "added_points": info.added_points,
                    "variant_id": variant["variant_id"],
                    "tracker_smoothing": variant["tracker_smoothing"],
                    "variant_note": variant["variant_note"],
                    "sample_count": traj.count,
                    "raw_row_count": raw_row_count,
                    "position_unit": "mm",
                    "rotation_source": "xr,yr,zr",
                    "rotation_assumption": "XYZ Euler angles in degrees",
                    "timestamp_source": "generated from rate column",
                    "sample_rate_hz": _sample_rate_from_traj(traj),
                }
            )

    for case in CASE0506_CASES:
        source_path = Path(case["source_path"])
        if source_path.exists():
            rows.append(_prepare_case0506_case(case))

    _write_csv(INPUT_ROOT / "manifest.csv", rows)
    _write_readme()
    return rows


def _read_sn_track(path: Path, name: str) -> tuple[Trajectory, int]:
    frame = pd.read_csv(path, sep="\t")
    missing = [col for col in TXT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[TXT_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    rotations = Rotation.from_euler("xyz", frame[["xr", "yr", "zr"]].to_numpy(dtype=float), degrees=True)
    poses = make_poses(positions, rotations)
    rate = float(frame["rate"].replace(0, np.nan).dropna().median())
    if not np.isfinite(rate) or rate <= 0:
        rate = 100.0
    timestamps = np.arange(len(frame), dtype=float) / rate
    return (
        Trajectory(
            poses,
            timestamps=timestamps,
            name=name,
            metadata={
                "source": str(path),
                "position_unit": "mm",
                "rotation_assumption": "XYZ Euler angles in degrees",
            },
        ),
        len(frame),
    )


def _prepare_case0506_case(case: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(case["source_path"])
    case_id = str(case["case_id"])
    trajectory_name = f"{case_id}_01_{source_path.stem}"
    case_dir = INPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    trajectory, raw_row_count, frame = _read_case0506_track(source_path, trajectory_name)
    input_csv = case_dir / f"{trajectory_name}.csv"
    metadata_json = case_dir / f"{trajectory_name}_metadata.json"
    write_trajectory(trajectory, input_csv)
    metadata = _case0506_metadata(
        case_id=case_id,
        trajectory_name=trajectory_name,
        source_path=source_path,
        raw_row_count=raw_row_count,
        frame=frame,
        trajectory=trajectory,
    )
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "trajectory": trajectory_name,
        "case_id": case_id,
        "case_label": case["case_label"],
        "source_folder": "ref_data",
        "source_file": source_path.name,
        "source_path": str(source_path),
        "input_csv": str(input_csv),
        "source_workbook": "",
        "base": "case0506",
        "target": "robot_arm_motion",
        "calibration": "unknown_calibration",
        "exposure": "default_exposure",
        "added_points": False,
        "variant_id": case["variant_id"],
        "tracker_smoothing": "unknown",
        "variant_note": case["variant_note"],
        "sample_count": trajectory.count,
        "raw_row_count": raw_row_count,
        "position_unit": "mm",
        "rotation_source": "a,b,c",
        "rotation_assumption": "XYZ Euler angles in radians",
        "timestamp_source": "normalized from source time seconds",
        "sample_rate_hz": _sample_rate_from_traj(trajectory),
        "reference_csv": "",
        "metadata_json": str(metadata_json),
    }


def _read_case0506_track(path: Path, name: str) -> tuple[Trajectory, int, pd.DataFrame]:
    raw_frame = pd.read_csv(path, sep="\t")
    raw_row_count = len(raw_frame)
    missing = [col for col in CASE0506_COLUMNS if col not in raw_frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = raw_frame[CASE0506_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} contains no valid numeric trajectory rows")

    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    rotations = Rotation.from_euler("xyz", frame[["a", "b", "c"]].to_numpy(dtype=float), degrees=False)
    poses = make_poses(positions, rotations)
    timestamps = _timestamps_from_case0506(frame)
    return (
        Trajectory(
            poses,
            timestamps=timestamps,
            name=name,
            metadata={
                "source": str(path),
                "position_unit": "mm",
                "rotation_assumption": "XYZ Euler angles in radians",
                "timestamp_source": "normalized from source time seconds",
            },
        ),
        raw_row_count,
        frame,
    )


def _timestamps_from_case0506(frame: pd.DataFrame) -> np.ndarray:
    source_timestamps = frame["time"].to_numpy(dtype=float)
    timestamps = source_timestamps - source_timestamps[0]
    if len(timestamps) > 1 and np.all(np.diff(timestamps) > 0.0):
        return timestamps

    valid_dt = np.diff(source_timestamps)
    valid_dt = valid_dt[valid_dt > 0.0]
    if valid_dt.size:
        dt = float(np.median(valid_dt))
        if np.isfinite(dt) and dt > 0.0:
            return np.arange(len(frame), dtype=float) * dt
    return np.arange(len(frame), dtype=float) / 100.0


def _case0506_metadata(
    *,
    case_id: str,
    trajectory_name: str,
    source_path: Path,
    raw_row_count: int,
    frame: pd.DataFrame,
    trajectory: Trajectory,
) -> dict[str, Any]:
    positions = trajectory.positions
    timestamps = trajectory.timestamps
    assert timestamps is not None
    dt = np.diff(timestamps)
    valid_dt = dt[dt > 0.0]
    euler = frame[["a", "b", "c"]].to_numpy(dtype=float)
    return {
        "case_id": case_id,
        "trajectory": trajectory_name,
        "source_path": str(source_path),
        "source_columns": CASE0506_COLUMNS,
        "raw_row_count": raw_row_count,
        "sample_count": trajectory.count,
        "position_unit": "mm",
        "rotation_source": "a,b,c",
        "rotation_assumption": "XYZ Euler angles in radians",
        "timestamp_source": "normalized from source time seconds",
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "sample_rate_hz_median": _median_sample_rate(timestamps),
        "dt_s_min": float(valid_dt.min()) if valid_dt.size else 0.0,
        "dt_s_median": float(np.median(valid_dt)) if valid_dt.size else 0.0,
        "dt_s_max": float(valid_dt.max()) if valid_dt.size else 0.0,
        "position_min_mm": _axis_dict(positions.min(axis=0)),
        "position_max_mm": _axis_dict(positions.max(axis=0)),
        "position_range_mm": _axis_dict(np.ptp(positions, axis=0)),
        "position_std_mm": _axis_dict(positions.std(axis=0, ddof=0)),
        "euler_min_rad": _axis_dict(euler.min(axis=0), axes=("a", "b", "c")),
        "euler_max_rad": _axis_dict(euler.max(axis=0), axes=("a", "b", "c")),
        "euler_range_rad": _axis_dict(np.ptp(euler, axis=0), axes=("a", "b", "c")),
        "continuous_step_speed_mm_s": _step_speed_stats(positions, timestamps),
    }


def _folder_info(name: str, index: int) -> FolderInfo:
    lower = name.lower()
    if "高底座" in name:
        base = "high_base"
    elif "新底座" in name:
        base = "new_low_base"
    elif "低底座" in name:
        base = "low_base"
    else:
        base = "unknown_base"

    if "拍物体" in name:
        target = "object"
        label = "static_object"
    elif "拍机械臂" in name or "静止机械臂" in name:
        target = "robot_arm"
        label = "static_robot_arm"
    else:
        target = "unknown"
        label = "unknown_target"

    if "180标定后" in name:
        calibration = "after_180_calibration"
    elif "标定后" in name:
        calibration = "after_calibration"
    else:
        calibration = "before_or_unspecified_calibration"

    exposure_match = re.search(r"曝光(\d+)", name)
    exposure = f"exposure_{exposure_match.group(1)}" if exposure_match else "default_exposure"
    added_points = "加点" in name
    case_id = f"case_{index:02d}_{label}_{base}_{calibration}_{exposure}"
    if added_points:
        case_id += "_added_points"

    return FolderInfo(case_id, label, base, target, calibration, exposure, added_points)


def _variant_from_filename(name: str) -> dict[str, str]:
    normalized = name.lower()
    if "true" in normalized or "开启" in name:
        smoothing = "on"
        variant_id = "tracker_smoothing_on"
    elif "false" in normalized or "关闭" in name:
        smoothing = "off"
        variant_id = "tracker_smoothing_off"
    elif "机械臂" in name:
        smoothing = "unspecified"
        variant_id = "robot_arm_record"
    else:
        smoothing = "unspecified"
        match = re.search(r"(\d+)", name)
        variant_id = f"record_{match.group(1)}" if match else "record_default"

    variant_id = re.sub(r"[^a-zA-Z0-9_]+", "_", variant_id).strip("_")
    return {
        "tracker_smoothing": smoothing,
        "variant_id": variant_id,
        "variant_note": name,
    }


def _sample_rate_from_traj(traj: Trajectory) -> float:
    if traj.timestamps is None or traj.count < 2:
        return 0.0
    duration = traj.timestamps[-1] - traj.timestamps[0]
    return float((traj.count - 1) / duration) if duration > 0 else 0.0


def _median_sample_rate(timestamps: np.ndarray | None) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 0.0
    dt = np.diff(timestamps)
    dt = dt[(dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def _step_speed_stats(positions: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    dt = np.diff(timestamps)
    dp = np.diff(positions, axis=0)
    valid = (dt > 0.0) & (dt <= MAX_CONTINUOUS_STEP_S)
    if not np.any(valid):
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    speed_norm = np.linalg.norm(dp[valid], axis=1) / dt[valid]
    return {
        "median": float(np.median(speed_norm)),
        "p95": float(np.percentile(speed_norm, 95)),
        "max": float(np.max(speed_norm)),
    }


def _axis_dict(values: np.ndarray, axes: tuple[str, str, str] = ("x", "y", "z")) -> dict[str, float]:
    return {axis: float(value) for axis, value in zip(axes, values, strict=True)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme() -> None:
    text = """# SN Real Trajectories

These CSV files are converted from `ref_data/sn/*/track_data_*.txt`.

Source TXT columns:

- `x,y,z`: position, assumed to be millimeters
- `xr,yr,zr`: rotation, converted as XYZ Euler angles in degrees
- `rate`: used to generate timestamps because many source `time` columns are zero

The folder name is parsed into case metadata such as base, target, calibration,
exposure, and added-points flags. See `manifest.csv` for the full mapping.

Additional case0506 files are converted from `ref_data/case0506_scan.txt` and
`ref_data/case0506_sn.txt` into case31 and case32. These source files use
`time,x,y,z,a,b,c`, where `time` is seconds and `a,b,c` are interpreted as XYZ
Euler angles in radians.
"""
    (INPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

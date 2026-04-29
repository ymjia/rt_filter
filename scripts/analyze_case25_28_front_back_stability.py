from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CASE_IDS = [
    "case_25_static_robot_arm_track_data_00_hold_01",
    "case_26_static_robot_arm_track_data_00_hold_02",
    "case_27_static_robot_arm_track_data_00_hold_03",
    "case_28_static_robot_arm_track_data_00_hold_04",
    "case_29_static_robot_arm_track_data_00_after_motion_05",
    "case_30_static_robot_arm_track_data_00_before_motion_01",
]
INPUT_ROOT = Path("input/sn")
OUTPUT_ROOT = Path("outputs/case25_track_data_00_static_holds")


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run() -> dict[str, str]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    half_rows: list[dict[str, object]] = []
    compare_rows: list[dict[str, object]] = []
    for case_id in CASE_IDS:
        case_dir = INPUT_ROOT / case_id
        csv_path = next(case_dir.glob("*.csv"))
        metadata_path = next(case_dir.glob("*_metadata.json"))
        frame = pd.read_csv(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        front, back = _split_frame(frame)
        front_row = _part_metrics(case_id, metadata, "front", front)
        back_row = _part_metrics(case_id, metadata, "back", back)
        half_rows.extend([front_row, back_row])
        compare_rows.append(_compare_rows(case_id, front_row, back_row))

    halves_csv = OUTPUT_ROOT / "front_back_halves_summary.csv"
    compare_csv = OUTPUT_ROOT / "front_back_comparison.csv"
    report_md = OUTPUT_ROOT / "front_back_stability_report.md"
    pd.DataFrame(half_rows).to_csv(halves_csv, index=False)
    pd.DataFrame(compare_rows).to_csv(compare_csv, index=False)
    _write_report(report_md, pd.DataFrame(half_rows), pd.DataFrame(compare_rows))

    return {
        "halves_csv": str(halves_csv),
        "compare_csv": str(compare_csv),
        "report_md": str(report_md),
    }


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    midpoint = 0.5 * (timestamps[0] + timestamps[-1])
    front = frame.loc[timestamps <= midpoint].copy()
    back = frame.loc[timestamps > midpoint].copy()
    if back.empty:
        split_index = len(frame) // 2
        front = frame.iloc[:split_index].copy()
        back = frame.iloc[split_index:].copy()
    return front, back


def _part_metrics(
    case_id: str,
    metadata: dict[str, object],
    part: str,
    frame: pd.DataFrame,
) -> dict[str, object]:
    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    std = positions.std(axis=0, ddof=0)
    value_range = positions.max(axis=0) - positions.min(axis=0)
    centered = positions - positions.mean(axis=0, keepdims=True)
    p95 = np.percentile(np.abs(centered), 95, axis=0)
    segment_label = _segment_label(metadata)
    return {
        "case_id": case_id,
        "between_motion_segments": segment_label,
        "source_rows": f"{metadata['source_start_row']}..{metadata['source_stop_row_exclusive'] - 1}",
        "part": part,
        "sample_count": len(frame),
        "duration_s": float(frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0])
        if len(frame) > 1
        else 0.0,
        "x_std_mm": float(std[0]),
        "y_std_mm": float(std[1]),
        "z_std_mm": float(std[2]),
        "x_range_mm": float(value_range[0]),
        "y_range_mm": float(value_range[1]),
        "z_range_mm": float(value_range[2]),
        "x_p95_abs_mm": float(p95[0]),
        "y_p95_abs_mm": float(p95[1]),
        "z_p95_abs_mm": float(p95[2]),
    }


def _segment_label(metadata: dict[str, object]) -> str:
    between = metadata.get("between_motion_segments")
    if isinstance(between, list) and len(between) >= 2:
        return f"{between[0]}-{between[1]}"
    edge_role = metadata.get("edge_role")
    if edge_role is not None:
        return str(edge_role)
    nearest = metadata.get("nearest_motion_segment")
    if nearest is not None:
        return f"motion_{nearest}"
    return ""


def _compare_rows(
    case_id: str,
    front_row: dict[str, object],
    back_row: dict[str, object],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "front_samples": int(front_row["sample_count"]),
        "back_samples": int(back_row["sample_count"]),
        "x_std_ratio_back_to_front": _safe_ratio(float(back_row["x_std_mm"]), float(front_row["x_std_mm"])),
        "y_std_ratio_back_to_front": _safe_ratio(float(back_row["y_std_mm"]), float(front_row["y_std_mm"])),
        "z_std_ratio_back_to_front": _safe_ratio(float(back_row["z_std_mm"]), float(front_row["z_std_mm"])),
        "x_range_ratio_back_to_front": _safe_ratio(
            float(back_row["x_range_mm"]), float(front_row["x_range_mm"])
        ),
        "y_range_ratio_back_to_front": _safe_ratio(
            float(back_row["y_range_mm"]), float(front_row["y_range_mm"])
        ),
        "z_range_ratio_back_to_front": _safe_ratio(
            float(back_row["z_range_mm"]), float(front_row["z_range_mm"])
        ),
        "x_back_more_stable": bool(float(back_row["x_std_mm"]) < float(front_row["x_std_mm"])),
        "y_back_more_stable": bool(float(back_row["y_std_mm"]) < float(front_row["y_std_mm"])),
        "z_back_more_stable": bool(float(back_row["z_std_mm"]) < float(front_row["z_std_mm"])),
        "x_back_smaller_range": bool(float(back_row["x_range_mm"]) < float(front_row["x_range_mm"])),
        "y_back_smaller_range": bool(float(back_row["y_range_mm"]) < float(front_row["y_range_mm"])),
        "z_back_smaller_range": bool(float(back_row["z_range_mm"]) < float(front_row["z_range_mm"])),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def _write_report(path: Path, halves: pd.DataFrame, compare: pd.DataFrame) -> None:
    lines = [
        "# Case 25-30 Front/Back Stability Report",
        "",
        "Each static hold case is split at the time midpoint into front and back halves.",
        "",
        "## Halves",
        "",
        "| case | part | duration s | x std mm | y std mm | z std mm | x range mm | y range mm | z range mm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in halves.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['part']} | {row['duration_s']:.6g} | "
            f"{row['x_std_mm']:.6g} | {row['y_std_mm']:.6g} | {row['z_std_mm']:.6g} | "
            f"{row['x_range_mm']:.6g} | {row['y_range_mm']:.6g} | {row['z_range_mm']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Back vs Front",
            "",
            "| case | x std ratio | y std ratio | z std ratio | x range ratio | y range ratio | z range ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in compare.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['x_std_ratio_back_to_front']:.6g} | "
            f"{row['y_std_ratio_back_to_front']:.6g} | {row['z_std_ratio_back_to_front']:.6g} | "
            f"{row['x_range_ratio_back_to_front']:.6g} | {row['y_range_ratio_back_to_front']:.6g} | "
            f"{row['z_range_ratio_back_to_front']:.6g} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

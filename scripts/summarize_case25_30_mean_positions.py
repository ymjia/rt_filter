from __future__ import annotations

import json
from pathlib import Path

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

    rows: list[dict[str, object]] = []
    for case_id in CASE_IDS:
        case_dir = INPUT_ROOT / case_id
        csv_path = next(case_dir.glob("*.csv"))
        metadata_path = next(case_dir.glob("*_metadata.json"))
        frame = pd.read_csv(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "case_id": case_id,
                "position_group": _position_group(metadata),
                "source_rows": f"{metadata['source_start_row']}..{metadata['source_stop_row_exclusive'] - 1}",
                "sample_count": int(metadata["sample_count"]),
                "duration_s": float(metadata["duration_s"]),
                "mean_x_mm": float(frame["x"].mean()),
                "mean_y_mm": float(frame["y"].mean()),
                "mean_z_mm": float(frame["z"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_ROOT / "case25_30_mean_positions.csv"
    report_path = OUTPUT_ROOT / "case25_30_mean_positions.md"
    summary.to_csv(csv_path, index=False)
    _write_report(report_path, summary)
    return {
        "csv": str(csv_path),
        "report_md": str(report_path),
    }


def _position_group(metadata: dict[str, object]) -> str:
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


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Case 25-30 Mean Position Summary",
        "",
        "| case | position group | rows | duration s | mean x mm | mean y mm | mean z mm |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['position_group']} | {row['source_rows']} | "
            f"{row['duration_s']:.6g} | {row['mean_x_mm']:.6g} | {row['mean_y_mm']:.6g} | "
            f"{row['mean_z_mm']:.6g} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

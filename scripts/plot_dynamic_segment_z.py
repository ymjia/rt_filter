from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rt_filter.io import read_trajectory


DEFAULT_CASE_ID = "case_20_dynamic_robot_arm_track_data_00_segment_01"
OUTPUT_ROOT = Path("outputs/case20_track_data_00_segments")


def main() -> None:
    args = _parse_args()
    case_id = args.case_id
    case_dir = OUTPUT_ROOT / case_id
    comparison_csv = case_dir / f"{case_id}_filter_comparison.csv"
    metadata_json = ROOT / "input" / "sn" / case_id / f"{case_id}_01_track_data_00_metadata.json"
    if not comparison_csv.exists():
        raise FileNotFoundError(f"missing comparison csv: {comparison_csv}")
    if not metadata_json.exists():
        raise FileNotFoundError(f"missing metadata json: {metadata_json}")

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_csv)
    raw_path = ROOT / "input" / "sn" / case_id / f"{case_id}_01_track_data_00.csv"
    raw = read_trajectory(raw_path)

    selected = _select_filters(comparison)
    output_path = case_dir / f"{case_id}_z_comparison.png"
    _plot_case(raw, metadata, selected, output_path)
    print(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw/filtered Z traces for one dynamic segment case.")
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID, help="segment case id under input/sn and outputs/")
    return parser.parse_args()


def _select_filters(comparison: pd.DataFrame) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    ranked = comparison.sort_values(["score", "along_lag_ms_p95_positive"])

    best = ranked.iloc[0]
    selected.append(
        {
            "label": f"best: {best['algorithm']}",
            "path": str(best["filtered_csv"]),
            "color": "#d95f02",
        }
    )

    for algorithm, color in (("savgol", "#1b9e77"), ("one_euro_z", "#7570b3"), ("butterworth_z", "#e7298a")):
        subset = ranked.loc[ranked["algorithm"].astype(str).eq(algorithm)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        path = str(row["filtered_csv"])
        if any(item["path"] == path for item in selected):
            continue
        selected.append(
            {
                "label": algorithm,
                "path": path,
                "color": color,
            }
        )
    return selected


def _plot_case(raw, metadata: dict[str, object], selected: list[dict[str, str]], output_path: Path) -> None:
    timestamps = raw.timestamps
    assert timestamps is not None
    z_raw = raw.positions[:, 2]
    motion_start_idx = int(metadata["motion_start_relative"])
    motion_stop_idx = int(metadata["motion_stop_relative"])
    motion_t0 = float(timestamps[motion_start_idx])
    motion_t1 = float(timestamps[motion_stop_idx - 1])

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    full_ax, zoom_ax, residual_ax = axes

    _shade_segment_regions(full_ax, timestamps, motion_start_idx, motion_stop_idx)
    _shade_segment_regions(residual_ax, timestamps, motion_start_idx, motion_stop_idx)

    full_ax.plot(timestamps, z_raw, color="#4c4c4c", linewidth=1.8, label="raw")

    motion_slice = slice(max(0, motion_start_idx - 10), min(raw.count, motion_stop_idx + 10))
    zoom_t = timestamps[motion_slice]
    zoom_ax.plot(zoom_t, z_raw[motion_slice], color="#4c4c4c", linewidth=1.8, label="raw")

    residual_ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")

    for item in selected:
        filtered = read_trajectory(item["path"])
        z_filtered = filtered.positions[:, 2]
        label = item["label"]
        color = item["color"]
        full_ax.plot(timestamps, z_filtered, color=color, linewidth=1.5, label=label)
        zoom_ax.plot(zoom_t, z_filtered[motion_slice], color=color, linewidth=1.5, label=label)
        residual_ax.plot(timestamps, z_raw - z_filtered, color=color, linewidth=1.3, label=f"raw - {label}")

    full_ax.set_title(
        f"{metadata['case_id']}  Z Trace with Static/Motion Context\n"
        f"pre {metadata['pre_static_duration_s']:.2f}s | motion {metadata['motion_duration_s']:.2f}s | post {metadata['post_static_duration_s']:.2f}s"
    )
    full_ax.set_ylabel("Z (mm)")
    full_ax.legend(loc="best", ncol=2)
    full_ax.grid(True, alpha=0.25)

    zoom_ax.set_title(
        f"Motion Zoom  [{motion_t0:.2f}s, {motion_t1:.2f}s]   "
        f"raw dominant f = {metadata['motion_summary']['dominant_frequency_hz']:.2f} Hz"
    )
    zoom_ax.set_ylabel("Z (mm)")
    zoom_ax.legend(loc="best", ncol=2)
    zoom_ax.grid(True, alpha=0.25)

    residual_ax.set_title("Removed Component  (raw - filtered)")
    residual_ax.set_xlabel("Time (s)")
    residual_ax.set_ylabel("Residual (mm)")
    residual_ax.legend(loc="best", ncol=2)
    residual_ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _shade_segment_regions(ax, timestamps, motion_start_idx: int, motion_stop_idx: int) -> None:
    start_t = float(timestamps[0])
    motion_t0 = float(timestamps[motion_start_idx])
    motion_t1 = float(timestamps[motion_stop_idx - 1])
    end_t = float(timestamps[-1])
    ax.axvspan(start_t, motion_t0, color="#d9edf7", alpha=0.28, label="pre static")
    ax.axvspan(motion_t0, motion_t1, color="#fce5cd", alpha=0.30, label="motion")
    ax.axvspan(motion_t1, end_t, color="#d9ead3", alpha=0.25, label="post static")


if __name__ == "__main__":
    main()

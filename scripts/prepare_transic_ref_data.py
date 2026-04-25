from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

import h5py

from rt_filter.io import write_trajectory
from rt_filter.trajectory import Trajectory


DATASET_BASE_URL = "https://huggingface.co/datasets/transic-robot/data/resolve/main"
REF_ROOT = Path("ref_data/transic")
INPUT_ROOT = Path("input")
POSE_DATASET = "post_intervention_eef_pose"

SELECTED_FILES = [
    "correction_data/insert/2024-01-04-15-25-37.hdf5",
    "correction_data/insert/2024-01-04-15-29-19.hdf5",
    "correction_data/lift_leaned_leg/2024-03-24-13-26-04.hdf5",
    "correction_data/lift_leaned_leg/2024-03-24-13-30-39.hdf5",
    "correction_data/reach_and_grasp/2024-01-12-16-27-03.hdf5",
    "correction_data/reach_and_grasp/2024-01-12-16-28-51.hdf5",
    "correction_data/screw/2024-01-23-15-07-07.hdf5",
    "correction_data/screw/2024-01-23-15-10-32.hdf5",
    "correction_data/stabilize/2024-01-24-19-48-42.hdf5",
    "correction_data/stabilize/2024-01-24-19-55-59.hdf5",
]


def main() -> None:
    REF_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for rel in ("README.md", "LICENSE"):
        _download_if_missing(rel, REF_ROOT / rel)

    manifest_rows = []
    source_records = []
    for index, rel in enumerate(SELECTED_FILES, start=1):
        source_path = REF_ROOT / rel
        _download_if_missing(rel, source_path)
        task = Path(rel).parts[1]
        stem = source_path.stem.replace("-", "")
        output_name = f"transic_{index:02d}_{task}_{stem}.csv"
        output_path = INPUT_ROOT / output_name

        with h5py.File(source_path, "r") as handle:
            poses = handle[POSE_DATASET][:]
            intervention_count = int(handle["is_human_intervention"][:].sum())

        traj = Trajectory(
            poses=poses,
            name=output_path.stem,
            metadata={
                "source_dataset": "TRANSIC",
                "source_file": rel,
                "pose_dataset": POSE_DATASET,
                "unit": "meters",
            },
        )
        write_trajectory(traj, output_path)

        row = {
            "trajectory": output_path.stem,
            "task": task,
            "input_csv": str(output_path),
            "source_hdf5": str(source_path),
            "pose_dataset": POSE_DATASET,
            "sample_count": traj.count,
            "human_intervention_count": intervention_count,
            "unit": "meters",
        }
        manifest_rows.append(row)
        source_records.append(
            {
                **row,
                "source_url": f"{DATASET_BASE_URL}/{quote(rel)}",
            }
        )

    _write_csv(INPUT_ROOT / "manifest.csv", manifest_rows)
    (REF_ROOT / "selected_trajectories.json").write_text(
        json.dumps(source_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_rows)} trajectories to {INPUT_ROOT}")


def _download_if_missing(rel: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    url = f"{DATASET_BASE_URL}/{quote(rel)}"
    print(f"download {rel}")
    urlretrieve(url, destination)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

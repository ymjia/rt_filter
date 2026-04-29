from __future__ import annotations

import json
from pathlib import Path

import run_case16_hotel_analysis as base


base.SOURCE_PATH = Path("ref_data/track_noise_0.1.txt")
base.OUTPUT_ROOT = Path("outputs/case18_track_noise_0_1_static")
base.CASE_ID = "case_18_static_robot_arm_track_noise_0_1"
base.CASE_LABEL = "static_robot_arm"
base.TRAJECTORY = f"{base.CASE_ID}_01_track_noise_0_1"
base.CASE_BASE = "track_noise_0_1"
base.CASE_TARGET = "robot_arm"
base.CASE_VARIANT_ID = "track_noise_0_1"
base.CASE_VARIANT_NOTE = "larger static-noise robot arm trajectory"
base.REPORT_TITLE = "Case 18 Larger Static Noise Robot Arm Report"


def main() -> None:
    output = base.run()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

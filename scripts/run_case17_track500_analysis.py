from __future__ import annotations

import json
from pathlib import Path

import run_case16_hotel_analysis as base


# Reuse the same analysis pipeline as case16 so the noise and lag metrics are
# directly comparable across sites. Only the case identity and source paths
# change here.
base.SOURCE_PATH = Path("ref_data/track_data_500.txt")
base.OUTPUT_ROOT = Path("outputs/case17_track500_static")
base.CASE_ID = "case_17_static_robot_arm_poor_environment"
base.CASE_LABEL = "static_robot_arm"
base.TRAJECTORY = f"{base.CASE_ID}_01_track_data_500"
base.CASE_BASE = "poor_environment"
base.CASE_TARGET = "robot_arm"
base.CASE_VARIANT_ID = "track_data_500"
base.CASE_VARIANT_NOTE = "worse environment static robot arm trajectory"
base.REPORT_TITLE = "Case 17 Poor Environment Static Robot Arm Report"


def main() -> None:
    output = base.run()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

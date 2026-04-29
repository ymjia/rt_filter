# C++ Filter Demo

This demo builds the standalone C++ `butterworth`, `butterworth_z`,
`one_euro_z`, and `ukf` filters and runs them on a trajectory CSV. The output
CSV uses the same columns as the Python framework
(`timestamp,x,y,z,qw,qx,qy,qz,m00..m33`), so it can be read by
`rt_filter.io.read_trajectory` and evaluated with the current tools.

Besides the filtered trajectory CSV, the demo now also writes:

- `*.timing.csv`: per-pose compute time with `frame_index`, `timestamp`,
  `compute_time_ns`, `compute_time_us`, `compute_time_ms`
- `*.metrics.json`: total/mean/P95/max compute-time summary

## Cross-Platform Build

The demo now supports both macOS and Windows through:

- `output_alg/cpp_demo/CMakePresets.json`
- `scripts/build_cpp_demo.py`
- the local Conan workspace under `~/dev/3rd_build`

`scripts/build_cpp_demo.py` first refreshes the local `eigen/5.0.1@local/stable`
package from `3rd_build`, then runs `conan install`, `cmake configure`, and
`cmake build` with the preset that matches the host platform.

### macOS

This path uses the Xcode generator and the `macos-xcode15-arm64` toolchain from
`~/dev/3rd_build`:

```bash
python scripts/build_cpp_demo.py
```

Generated files:

- Xcode project: `build/cpp_demo/macos-xcode/rt_filter_cpp_demo.xcodeproj`
- Executable: `build/cpp_demo/macos-xcode/Release/rt_filter_cpp_demo`

If you want to inspect or build manually inside Xcode:

```bash
open build/cpp_demo/macos-xcode/rt_filter_cpp_demo.xcodeproj
```

### Windows

The default Windows preset uses the Visual Studio generator and the `vs2026`
toolchain entry from `3rd_build`:

```powershell
python scripts/build_cpp_demo.py
```

If your local `3rd_build` workspace uses a different Windows toolchain, override
it explicitly:

```powershell
python scripts/build_cpp_demo.py --third-build-toolchain vs2017
```

Generated files:

- Solution: `build/cpp_demo/windows-vs2022/rt_filter_cpp_demo.sln`
- Executable: `build/cpp_demo/windows-vs2022/Release/rt_filter_cpp_demo.exe`

### Manual Conan + CMake

If you prefer manual steps, the equivalent flow is:

```bash
python ~/dev/3rd_build/scripts/thirdparty_manager.py pipeline eigen --toolchain macos-xcode15-arm64 --build-type Release
CONAN_HOME=~/dev/3rd_build/conan_root \
CONAN1_EXE=~/dev/3rd_build/scripts/conan1_exe.sh \
CONAN1_USER_HOME=~/dev/3rd_build/conan_root_v1 \
~/dev/3rd_build/.tools/conan2_venv/bin/conan install output_alg/cpp_demo \
  -of build/cpp_demo/macos-xcode \
  -s build_type=Release \
  --build=missing \
  -c tools.cmake.cmaketoolchain:generator=Xcode
cmake --preset macos-xcode-release -S output_alg/cpp_demo
cmake --build --preset macos-xcode-release -S output_alg/cpp_demo
```

On Windows, replace the Conan executable path, generator, preset, and
`--toolchain` value accordingly.

## Run

macOS:

```bash
build/cpp_demo/macos-xcode/Release/rt_filter_cpp_demo \
  --algorithm ukf \
  --input input/sn/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv \
  --output outputs/cpp_demo/case11_ukf.csv \
  --motion-model constant_velocity \
  --process-noise 1000 \
  --measurement-noise 0.001 \
  --initial-linear-velocity 0,0,0 \
  --initial-angular-velocity 0,0,0
```

This command writes:

- `outputs/cpp_demo/case11_ukf.csv`
- `outputs/cpp_demo/case11_ukf.timing.csv`
- `outputs/cpp_demo/case11_ukf.metrics.json`

Windows:

```powershell
build\cpp_demo\windows-vs2022\Release\rt_filter_cpp_demo.exe `
  --algorithm ukf `
  --input input\sn\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv `
  --output outputs\cpp_demo\case11_ukf.csv `
  --motion-model constant_velocity `
  --process-noise 1000 `
  --measurement-noise 0.001 `
  --initial-linear-velocity 0,0,0 `
  --initial-angular-velocity 0,0,0
```

`one_euro_z` example:

```bash
build/cpp_demo/macos-xcode/Release/rt_filter_cpp_demo \
  --algorithm one_euro_z \
  --input input/sn/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv \
  --output outputs/cpp_demo/case11_one_euro_z.csv \
  --min-cutoff 1.0 \
  --beta 10.0 \
  --d-cutoff 8.0 \
  --derivative-deadband 0.02
```

`butterworth_z` example:

```bash
build/cpp_demo/macos-xcode/Release/rt_filter_cpp_demo \
  --algorithm butterworth_z \
  --input input/sn/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv \
  --output outputs/cpp_demo/case11_butterworth_z.csv \
  --cutoff-hz 20.0 \
  --order 2
```

`butterworth` example:

```bash
build/cpp_demo/macos-xcode/Release/rt_filter_cpp_demo \
  --algorithm butterworth \
  --input input/sn/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv \
  --output outputs/cpp_demo/case11_butterworth.csv \
  --cutoff-hz 30.0 \
  --order 2
```

## Evaluate Output

```python
from pathlib import Path
from rt_filter.evaluation import compare_filter_result
from rt_filter.io import read_trajectory

raw = read_trajectory(
    Path(
        "input/sn/case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms/"
        "case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv"
    )
)
filtered = read_trajectory(Path("outputs/cpp_demo/case11_ukf.csv"))
print(compare_filter_result(raw, filtered))
```

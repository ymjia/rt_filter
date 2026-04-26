# C++ Filter Demo

This demo builds the standalone C++ `one_euro_z` and `ukf` filters and runs
them on a trajectory CSV. The output CSV uses the same columns as the Python
framework (`timestamp,x,y,z,qw,qx,qy,qz,m00..m33`), so it can be read by
`rt_filter.io.read_trajectory` and evaluated with the current tools.

## Build With Conan

From the repository root:

```powershell
. F:\dev\3rd_build\scripts\use_local_conan.ps1
conan install output_alg\cpp_demo -of build\cpp_demo -s build_type=Release --build=missing
cmake --preset conan-default -S output_alg\cpp_demo
cmake --build build\cpp_demo --config Release
```

The Conan package used by this demo is:

```text
eigen/5.0.1@local/stable
```

## Run

```powershell
build\cpp_demo\Release\rt_filter_cpp_demo.exe `
  --algorithm ukf `
  --input input\sn\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv `
  --output outputs\cpp_demo\case11_ukf.csv `
  --motion-model constant_velocity `
  --process-noise 1000 `
  --measurement-noise 0.001 `
  --initial-linear-velocity 0,0,0 `
  --initial-angular-velocity 0,0,0
```

```powershell
build\cpp_demo\Release\rt_filter_cpp_demo.exe `
  --algorithm one_euro_z `
  --input input\sn\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv `
  --output outputs\cpp_demo\case11_one_euro_z.csv `
  --min-cutoff 0.02 `
  --beta 6.0 `
  --d-cutoff 2.0 `
  --derivative-deadband 1.0
```

## Evaluate Output

```powershell
python - <<'PY'
from pathlib import Path
from rt_filter.evaluation import compare_filter_result
from rt_filter.io import read_trajectory

raw = read_trajectory(Path(r"input\sn\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms\case_11_synthetic_rectangle_high_base_case10_noise_speed_100mms_01_rectangle_100mms.csv"))
filtered = read_trajectory(Path(r"outputs\cpp_demo\case11_ukf.csv"))
print(compare_filter_result(raw, filtered))
PY
```

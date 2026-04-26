# Standalone Realtime Filters

This directory contains standalone Python and C++ implementations of filters
used for realtime scan-view pose display and trajectory denoising.

Available filters:

- `one_euro_z`: Z-only One Euro adaptive low-pass filtering.
- `ukf`: 6D pose Unscented Kalman filtering with constant-velocity or
  constant-acceleration motion prediction.

## OneEuroZ Realtime Filter

The filter accepts homogeneous 4x4 poses and only changes the Z translation:

- Python: `pose[2, 3]`
- C++ row-major pose array: `pose[11]`, equivalent to `m23`

X/Y translation and rotation are copied unchanged.

## Parameters

| parameter | effect |
| --- | --- |
| `min_cutoff` | Lower values denoise static Z more strongly but increase lag. |
| `beta` | Higher values make the filter follow motion faster and reduce lag. |
| `d_cutoff` | Low-pass cutoff for the estimated Z velocity. |
| `derivative_deadband` | Z velocity deadband before adaptive cutoff; higher values keep small static jitter from weakening denoising. |
| `sample_rate_hz` | Used when frame timestamps are not supplied. |
| `history_size` | Number of filtered poses retained by the realtime object; `0` keeps all. |

Recommended initial values for the current SN noise profile:

```text
min_cutoff = 0.02
beta = 6.0
d_cutoff = 2.0
derivative_deadband = 1.0
sample_rate_hz = 100.0
```

## Python Usage

```python
import numpy as np
from output_alg.one_euro_z import OneEuroZParameters, OneEuroZRealtimeFilter

params = OneEuroZParameters(
    min_cutoff=0.02,
    beta=6.0,
    d_cutoff=2.0,
    derivative_deadband=1.0,
    sample_rate_hz=100.0,
)
filter_z = OneEuroZRealtimeFilter(params)

# Realtime UI update: one frame in, one display pose out.
display_pose = filter_z.update(current_pose_4x4, timestamp=current_time_s)

# UI update from a bounded recent history.
display_pose = filter_z.filter_latest_from_history(history_poses_4x4, history_timestamps)

# Per-frame output for an entire trajectory.
filtered_poses = filter_z.filter_trajectory(all_poses_4x4, timestamps)
```

## C++ Usage

```cpp
#include "one_euro_z.hpp"

using output_alg::OneEuroZParameters;
using output_alg::OneEuroZRealtimeFilter;

OneEuroZParameters params;
params.min_cutoff = 0.02;
params.beta = 6.0;
params.d_cutoff = 2.0;
params.derivative_deadband = 1.0;
params.sample_rate_hz = 100.0;

OneEuroZRealtimeFilter filter(params);
OneEuroZRealtimeFilter::Pose display_pose = filter.Update(current_pose, current_time_s);
std::vector<OneEuroZRealtimeFilter::Pose> filtered = filter.FilterTrajectory(poses, &timestamps);
```

Build a C++ object file:

```powershell
g++ -std=c++17 -Ioutput_alg -c output_alg/one_euro_z.cpp -o build/one_euro_z.o
```

## UKF Realtime Filter

The UKF accepts homogeneous 4x4 poses and filters the 6D measurement
`[x,y,z,rx,ry,rz]`. The rotation vector is relative to the first pose after
reset. Linear velocity units are input distance units per second, and angular
velocity is radians per second.

Key parameters:

| parameter | effect |
| --- | --- |
| `motion_model` | `constant_velocity` or `constant_acceleration`. |
| `process_noise` | Higher values follow motion faster and reduce lag. |
| `measurement_noise` | Higher values smooth more and trust observations less. |
| `initial_covariance` | Initial state covariance. |
| `initial_velocity` | Full `[vx,vy,vz,wx,wy,wz]` initial velocity. |
| `initial_linear_velocity` | Split `[vx,vy,vz]` initial linear velocity. |
| `initial_angular_velocity` | Split `[wx,wy,wz]` initial angular velocity. |
| `sample_rate_hz` | Used when frame timestamps are not supplied. |

Python usage:

```python
from output_alg.ukf import UkfParameters, UkfRealtimeFilter

params = UkfParameters(
    motion_model="constant_velocity",
    process_noise=1000.0,
    measurement_noise=0.001,
    initial_linear_velocity=[0.0, 0.0, 0.0],
    initial_angular_velocity=[0.0, 0.0, 0.0],
)
filter_ukf = UkfRealtimeFilter(params)

display_pose = filter_ukf.update(current_pose_4x4, timestamp=current_time_s)
filtered_poses = filter_ukf.filter_trajectory(all_poses_4x4, timestamps)
```

C++ usage:

```cpp
#include "ukf.hpp"

output_alg::UkfParameters params;
params.motion_model = "constant_velocity";
params.process_noise = 1000.0;
params.measurement_noise = 0.001;
params.initial_linear_velocity = {0.0, 0.0, 0.0};
params.initial_angular_velocity = {0.0, 0.0, 0.0};

output_alg::UkfRealtimeFilter filter(params);
output_alg::UkfRealtimeFilter::Pose display_pose = filter.Update(current_pose, current_time_s);
```

## C++ Demo

`output_alg/cpp_demo` contains a CMake + Conan demo executable that reads a
trajectory CSV, runs either `one_euro_z` or `ukf`, and writes a CSV that can be
evaluated by the existing Python framework. See
`output_alg/cpp_demo/README.md` for build and run commands.

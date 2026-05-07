# Standalone Realtime Filters

This directory contains standalone Python and C++ implementations of filters
used for realtime scan-view pose display and trajectory denoising.

Available filters:

- `butterworth`: XYZ realtime causal Butterworth low-pass filtering.
- `butterworth_z`: Z-only realtime causal Butterworth low-pass filtering.
- `one_euro_z`: Z-only One Euro adaptive low-pass filtering.
- `adaptive_local_line`: Fixed-lag line-constrained filtering that preserves
  along-line motion and adaptively attenuates perpendicular jitter.
- `ukf`: 6D pose Unscented Kalman filtering with constant-velocity or
  constant-acceleration motion prediction.

Note: the realtime C++ Butterworth filters are causal IIR filters. They are
intended for scan-view display or frame-by-frame processing, and therefore are
not identical to the Python-side offline zero-phase `sosfiltfilt` implementation.

## OneEuroZ Realtime Filter

The filter accepts homogeneous 4x4 poses in Python and `Sn3DAlgorithm::RigidMatrix`
in C++. It only changes the Z translation:

- Python: `pose[2, 3]`
- C++: `rigid.get_translation().z()`

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
| `delay_frames` | `0` keeps the original causal update. Values such as `3` return a frame delayed by 3 samples and smooth it with up to `2 * delay_frames + 1` recent raw frames. |

Recommended initial values for the current SN noise profile:

```text
min_cutoff = 1.0
beta = 10.0
d_cutoff = 8.0
derivative_deadband = 0.02
sample_rate_hz = 100.0
delay_frames = 0
```

## Python Usage

```python
import numpy as np
from output_alg.one_euro_z import OneEuroZParameters, OneEuroZRealtimeFilter

params = OneEuroZParameters(
    min_cutoff=1.0,
    beta=10.0,
    d_cutoff=8.0,
    derivative_deadband=0.02,
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
#include "RigidMatrix.h"

using output_alg::OneEuroZParameters;
using output_alg::OneEuroZRealtimeFilter;
using Sn3DAlgorithm::RigidMatrix;

OneEuroZParameters params;
params.min_cutoff = 1.0;
params.beta = 10.0;
params.d_cutoff = 8.0;
params.derivative_deadband = 0.02;
params.sample_rate_hz = 100.0;
params.delay_frames = 0;

OneEuroZRealtimeFilter filter(params);
RigidMatrix display_pose = filter.Update(current_rigid, current_time_s);
std::vector<RigidMatrix> filtered = filter.FilterTrajectory(rigids, &timestamps);
```

Build a C++ object file:

```powershell
g++ -std=c++14 -Ioutput_alg -c output_alg/one_euro_z.cpp -o build/one_euro_z.o
```

## Realtime Butterworth Filters

`ButterworthRealtimeFilter` filters `x/y/z` translation together, while
`ButterworthZRealtimeFilter` only changes `translation.z()`. Both preserve the
input rotation matrix and keep internal causal IIR state across frames.

Key parameters:

| parameter | effect |
| --- | --- |
| `cutoff_hz` | Low-pass cutoff; lower values smooth more but remove more local peaks. |
| `order` | Butterworth order; `2` is the recommended starting point. |
| `sample_rate_hz` | Used when frame timestamps are not supplied. |
| `history_size` | Number of filtered poses retained by the realtime object; `0` keeps all. |
| `delay_frames` | For `ButterworthZRealtimeFilter`, `0` keeps the original causal IIR update. Values such as `3` return a Z-filtered frame delayed by 3 samples and use up to 7 recent raw frames for zero-phase local smoothing. |

C++ usage:

```cpp
#include "butterworth.hpp"
#include "RigidMatrix.h"

output_alg::ButterworthParameters xyz_params;
xyz_params.cutoff_hz = 30.0;
xyz_params.order = 2;
xyz_params.sample_rate_hz = 100.0;

output_alg::ButterworthRealtimeFilter xyz_filter(xyz_params);
Sn3DAlgorithm::RigidMatrix xyz_display = xyz_filter.Update(current_rigid, current_time_s);

output_alg::ButterworthZParameters z_params;
z_params.cutoff_hz = 20.0;
z_params.order = 2;
z_params.sample_rate_hz = 100.0;
z_params.delay_frames = 0;

output_alg::ButterworthZRealtimeFilter z_filter(z_params);
Sn3DAlgorithm::RigidMatrix z_display = z_filter.Update(current_rigid, current_time_s);
```

Build C++ object files:

```powershell
g++ -std=c++14 -Ioutput_alg -c output_alg/butterworth.cpp -o build/butterworth.o
```

## Adaptive Local Line Realtime Filter

`AdaptiveLocalLineRealtimeFilter` is intended for straight-line motion. It
keeps the input rotation unchanged, preserves each point's coordinate along the
reference line, and attenuates only the perpendicular residual. The attenuation
strength is computed from a local odd-sized window, so noisier local regions are
pulled toward the line more strongly while quieter regions are left closer to
raw input.

Key parameters:

| parameter | effect |
| --- | --- |
| `window` | Odd local window size. `5` gives a 2-frame fixed lag. |
| `target_noise_mm` | Perpendicular RMS below this level is left unchanged. |
| `max_strength` | Upper bound for how much of the perpendicular residual is removed. |
| `min_strength` | Lower bound used once local noise exceeds the target. |
| `response` | Shape of the noise-to-strength curve. |
| `reference_mode` | `global` fits/uses one reference line; `local` fits each window. |
| `line_origin` | Optional known line origin, used with `line_direction`. |
| `line_direction` | Optional known line direction for realtime use. |
| `sample_rate_hz` | Used when frame timestamps are not supplied. |

C++ usage:

```cpp
#include "adaptive_local_line.hpp"
#include "RigidMatrix.h"

output_alg::AdaptiveLocalLineParameters params;
params.window = 5;
params.target_noise_mm = 0.26;
params.max_strength = 0.5;
params.reference_mode = "global";
params.use_line_direction = true;
params.line_origin = Eigen::Vector3d(0.0, 0.0, 0.0);
params.line_direction = Eigen::Vector3d(1.0, 0.0, 0.0);

output_alg::AdaptiveLocalLineRealtimeFilter filter(params);
Sn3DAlgorithm::RigidMatrix display_pose = filter.Update(current_rigid, current_time_s);
std::vector<Sn3DAlgorithm::RigidMatrix> filtered =
    filter.FilterTrajectory(rigids, &timestamps);
```

## UKF Realtime Filter

The Python UKF accepts homogeneous 4x4 poses. The C++ UKF accepts
`Sn3DAlgorithm::RigidMatrix`. Both filter the 6D measurement
`[x,y,z,rx,ry,rz]`. The rotation vector is relative to the first frame after
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
#include "RigidMatrix.h"

output_alg::UkfParameters params;
params.motion_model = "constant_velocity";
params.process_noise = 1000.0;
params.measurement_noise = 0.001;
params.initial_linear_velocity = {0.0, 0.0, 0.0};
params.initial_angular_velocity = {0.0, 0.0, 0.0};

output_alg::UkfRealtimeFilter filter(params);
Sn3DAlgorithm::RigidMatrix display_pose = filter.Update(current_rigid, current_time_s);
```

For integration code that already stores position plus Euler angles, include
`euler_zyx_interface.hpp` and call `FilterButterworthEulerZyx`,
`FilterButterworthZEulerZyx`, `FilterOneEuroZEulerZyx`, or `FilterUkfEulerZyx`.
The Euler vector is `[z, y, x]` in radians, with `R = Rz * Ry * Rx`; the
helper mutates the input position and Euler vectors with the filtered result.

## C++ Demo

`output_alg/cpp_demo` contains a CMake + Conan demo executable that reads a
trajectory CSV, runs `butterworth`, `butterworth_z`, `one_euro_z`,
`adaptive_local_line`, or `ukf`, and writes a CSV that can be evaluated by the
existing Python framework. See `output_alg/cpp_demo/README.md` for build and
run commands.

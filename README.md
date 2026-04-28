# 机械臂轨迹滤波系统

这个目录现在包含一个可运行的 Python 工程，用于对机械臂 6DoF 轨迹做读取、滤波、批量执行、指标验证和统计可视化。

系统内部统一使用 `N x 4 x 4` 齐次变换矩阵表示轨迹，同时兼容常见的 `x,y,z,qw,qx,qy,qz` 输入。文档中的问题判断表明当前数据可能包含姿态系统误差和杠杆臂放大效应，因此默认评估会同时关注平滑程度、对原始轨迹的偏移、相对参考轨迹的误差，以及滤波引入的滞后风险。

## 安装

```powershell
python -m pip install -e .[dev]
```

## 输入格式

支持 `.csv`、`.txt`、`.json`、`.npy`、`.npz`。

CSV 可使用以下任一形式：

- 位姿列：`x,y,z,qw,qx,qy,qz`
- 位姿列：`x,y,z,qx,qy,qz,qw`
- 矩阵列：`m00,m01,...,m33`
- 可选时间列：`timestamp`、`time` 或 `t`
- 可选状态列：`status`，值为 `-1` 的行默认丢弃

也支持 SN 参考轨迹的 `x,y,z,xr,yr,zr,time,rate` 形式：

- `.csv`：可为逗号分隔或制表符分隔，要求包含表头
- `.txt`：按 `x y z xr yr zr time rate` 列顺序读取，可无表头
- `xr,yr,zr` 按 XYZ 欧拉角 `degrees=True` 解释
- 时间戳默认优先按 `rate` 列生成，与 `prepare_sn_ref_data.py` 保持一致

## 单条轨迹滤波

```powershell
rt-filter filter input.csv outputs/filtered.csv --algorithm moving_average --param window=9
rt-filter filter input.csv outputs/filtered.csv --algorithm savgol --param window=11 --param polyorder=2
rt-filter filter input.csv outputs/filtered.csv --algorithm exponential --param alpha=0.25
rt-filter filter input.csv outputs/filtered.csv --algorithm one_euro_z --param min_cutoff=0.02 --param beta=6.0 --param d_cutoff=2.0 --param derivative_deadband=1.0
```

查看可用算法：

```powershell
rt-filter catalog
```

## 滤波算法说明

当前所有滤波算法都作用在完整 6DoF 位姿轨迹上。平移部分按 `x,y,z` 处理；旋转部分不会直接线性平均四元数，而是使用 SO(3) 旋转平均、相对旋转向量或增量旋转，避免普通欧氏空间插值造成明显姿态错误。

### 算法总览

| 算法 | 主要参数 | 性质 | 适用场景 | 主要风险 |
| --- | --- | --- | --- | --- |
| `moving_average` | `window` | 居中滑动窗口平均；平移做窗口均值，姿态做 SO(3) mean | 静态点稳定性分析、低速轨迹初筛、快速看噪声量级 | 会抹平尖峰和真实动态细节；窗口大时轨迹拐角会被圆滑化 |
| `savgol` | `window`, `polyorder`, `mode` | Savitzky-Golay 多项式平滑；较保形，能比均值滤波更好保留趋势 | 连续运动轨迹、直线/圆弧等较平滑运动、需要少滞后地降噪 | 窗口过大仍会过平滑；轨迹很短时会自动保持原样 |
| `exponential` | `alpha` | 因果指数平滑，只使用当前和历史信息 | 在线实时滤波、不能使用未来帧的场景 | 会产生滞后；速度越高或 `alpha` 越小，位置滞后越明显 |
| `kalman_cv` | `process_noise`, `measurement_noise`, `initial_covariance` | 常速度模型 Kalman 滤波；同时估计位置/旋转向量及其速度 | 静态或低速数据、近似匀速轨迹、需要较强抖动压制的稳定性测试 | 模型假设不适合剧烈加减速；参数过强会把真实运动当噪声 |
| `ukf` | `motion_model`, `process_noise`, `measurement_noise`, `initial_covariance`, `initial_velocity`, `initial_linear_velocity`, `initial_angular_velocity`, `alpha`, `beta`, `kappa` | 无迹 Kalman 滤波；对平移和相对旋转向量使用匀速或匀加速状态预测 | 位姿变化连续、测量噪声较大、希望显式引入运动连续性约束的轨迹 | 当前仅使用位姿测量；若无关节/控制量，复杂机械臂运动学模型无法真正发挥 |
| `one_euro_z` | `min_cutoff`, `beta`, `d_cutoff`, `derivative_deadband`, `sample_rate_hz` | Z 方向 One Euro 自适应低通；只改 Z，保留 X/Y 和姿态 | Z 噪声明显大于 X/Y、需要在线实时且减少拖影的静止或慢速转向场景 | 只处理 Z 随机噪声；`min_cutoff` 太低或 `derivative_deadband` 太大仍会产生滞后 |

如果已经通过 `python3 scripts/build_cpp_demo.py` 构建了独立 C++ demo，GUI 和分析链路里还会出现 `ukf-cpp` 与 `one_euro_z-cpp`。这两个条目会由 Python 调用 `rt_filter_cpp_demo` 可执行程序，并直接导入该程序写出的轨迹结果与每帧耗时数据。

### `moving_average`

`moving_average` 是最直观的平滑方法。`window` 表示窗口长度，值越大，抖动压制越强，但轨迹细节损失也越多。当前位置使用前后邻域的样本平均，因此它不是严格在线算法。

适合：

- 静止采样数据的抖动范围估计
- 快速获得一个低噪声参考曲线
- 对比设备内置平滑开关前后的效果

不适合：

- 高速动态跟踪
- 拐角、启停、变速段需要保持真实形状的轨迹
- 对延迟或局部峰值敏感的评价

推荐先试：

```yaml
name: moving_average
params:
  window: [5, 15, 31]
```

### `savgol`

`savgol` 使用 Savitzky-Golay 多项式平滑。它不是简单平均，而是在局部窗口内拟合低阶多项式，所以通常比移动平均更能保留轨迹形状和斜率。平移直接对 `x,y,z` 做平滑；姿态先转换为相对初始姿态的旋转向量，再平滑后还原为旋转矩阵。

适合：

- 直线、圆弧、慢速连续运动
- 希望降低高频噪声，同时尽量保留运动趋势
- 对比不同窗口长度下的保形能力

参数含义：

- `window`：局部窗口长度，必须为正；偶数会自动调整为较小的奇数
- `polyorder`：多项式阶数，必须小于 `window`
- `mode`：边界处理方式，默认 `interp`

推荐先试：

```yaml
name: savgol
params:
  window: [7, 11, 21, 31]
  polyorder: [2]
```

注意：如果轨迹长度小于等于 `polyorder + 1`，当前实现会返回原轨迹并在 metadata 中记录 warning。

### `exponential`

`exponential` 是因果滤波，当前输出只依赖历史输出和当前观测。`alpha` 越大，越相信当前观测，滞后越小但降噪较弱；`alpha` 越小，平滑越强但滞后越明显。

适合：

- 在线实时系统
- 只允许使用历史数据的处理链路
- 静态或低速采样的轻量降噪

不适合：

- 需要离线最优保形的轨迹分析
- 变速直线、快速启停、动态跟踪误差评价

推荐先试：

```yaml
name: exponential
params:
  alpha: [0.15, 0.25, 0.4, 0.6]
```

工程判断：如果 `jerk_rms_ratio` 很低，但 `to_raw_translation_rmse` 和 `to_reference_translation_rmse` 明显变大，通常说明指数滤波引入了滞后。

### `kalman_cv`

`kalman_cv` 使用常速度模型。平移和姿态旋转向量分别进入同一种常速度 Kalman 滤波器。它通过 `process_noise` 和 `measurement_noise` 控制“相信运动模型”还是“相信观测数据”。

参数含义：

- `process_noise`：过程噪声。越大表示允许状态变化更快，滤波更跟手；越小表示更相信常速度模型，平滑更强。
- `measurement_noise`：测量噪声。越大表示越不相信观测，平滑更强；越小表示更贴近原始数据。
- `initial_covariance`：初始协方差，默认 `1.0`，通常不需要先调。

适合：

- 静态稳定性数据
- 低速、近似匀速的轨迹
- 需要明显压制随机抖动的评估

不适合：

- 真实运动有明显高加速度或频繁改变方向，而仍希望完整保留动态细节的场景
- 系统误差主导、不是随机噪声主导的数据。此时 Kalman 可以让曲线更平滑，但不能证明绝对精度变好。

推荐先试：

```yaml
name: kalman_cv
params:
  process_noise: [0.0001, 0.001]
  measurement_noise: [0.005, 0.01, 0.02]
```

### `ukf`

`ukf` 使用无迹 Kalman 滤波。当前实现把每帧位姿转成 6 维测量 `[x,y,z,rx,ry,rz]`，其中 `rx,ry,rz` 是相对第一帧姿态的旋转向量。状态预测支持 `constant_velocity` 和 `constant_acceleration` 两种运动模型；更新阶段用位姿观测修正预测结果。

参数含义：

- `motion_model`：`constant_velocity` 或 `constant_acceleration`。
- `process_noise`：运动模型过程噪声。越大越跟手，越小越平滑。
- `measurement_noise`：位姿测量噪声。越大越不相信视觉测量，平滑更强但拖影风险更高。
- `initial_covariance`：初始状态协方差。
- `initial_velocity`：6 维初始速度 `[vx,vy,vz,wx,wy,wz]`，线速度单位与输入坐标一致/秒，角速度单位为 rad/s。
- `initial_linear_velocity` / `initial_angular_velocity`：拆分传入的 3 维初始线速度和角速度；不能与 `initial_velocity` 同时使用。
- `alpha`, `beta`, `kappa`：UKF sigma 点分布参数，通常保持默认。

推荐先试：

```yaml
name: ukf
params:
  motion_model: constant_velocity
  process_noise: 1000.0
  measurement_noise: 0.001
  initial_linear_velocity: [0.0, 0.0, 0.0]
  initial_angular_velocity: [0.0, 0.0, 0.0]
```

工程判断：如果只有位姿序列，没有关节角、控制量或机器人运动学模型，UKF 只能表达“轨迹应连续、速度/加速度不应突变”这类通用物理先验。若要让 UKF 真正知道机械臂运动学，还需要同步输入关节角/关节速度、末端速度命令、机器人 DH/URDF 参数，以及这些量的噪声协方差。

### `one_euro_z`

`one_euro_z` 是面向当前 SN 数据现象加入的在线自适应低通滤波。它只对平移的 Z 方向做 One Euro Filter，X/Y 和旋转保持原样；当检测到 Z 方向变化速度变大时会自动提高截止频率。当前默认加入了 Z 速度死区，用来避免静止时的小幅深度抖动过早放开滤波。

参数含义：

- `min_cutoff`：低速或静止时的最小截止频率。越小抖动压制越强，但越容易滞后。
- `beta`：根据速度提高截止频率的强度。越大越跟手，拖影越小，但静止抑噪会变弱。
- `d_cutoff`：速度估计的截止频率，默认 `2.0`。
- `derivative_deadband`：速度自适应死区，单位约为 mm/s。小于该阈值的 Z 速度会被当作静止抖动，不提高截止频率。
- `sample_rate_hz`：无时间戳输入时使用的采样率，默认 `80.0`；有 `timestamp/time/t` 时会直接使用实际时间戳。

推荐先试：

```yaml
name: one_euro_z
params:
  min_cutoff: 0.02
  beta: 6.0
  d_cutoff: 2.0
  derivative_deadband: 1.0
```

工程判断：如果主要问题是 SN 固定位置数据里 Z 抖动比 X/Y 大得多，可以优先看 `z_jerk_rms_ratio`、`to_raw_z_rmse` 和 GUI 中的 `XYZ Neighbor Mean Deviation`。默认增强参数更偏向静止降噪；如果观察到慢速 Z 向运动拖影，优先降低 `derivative_deadband` 或提高 `min_cutoff`。

### 选择建议

| 目标 | 优先尝试 | 观察指标 |
| --- | --- | --- |
| 静止点云稳定性、极差/标准差压制 | `one_euro_z`, `kalman_cv`, `ukf`, `moving_average` | `filtered_range_z`, `std_norm_ratio`, `z_std_ratio`, `to_raw_translation_rmse` |
| 离线运动轨迹降噪且保留形状 | `savgol` | `to_reference_translation_rmse`, `to_raw_translation_rmse`, `jerk_rms_ratio` |
| 在线实时轻量滤波 | `one_euro_z`, `exponential` | `to_raw_translation_rmse`, `to_reference_translation_rmse`, 延迟表现 |
| 快速建立基线 | `moving_average` | `acceleration_rms_ratio`, `jerk_rms_ratio`, `to_raw_translation_max` |
| 有 Leica 或目标轨迹参考 | 多算法批量比较 | `to_reference_translation_rmse`, `reference_rmse_improvement` |
| 无参考、只有真实采样数据 | 多算法批量比较，含 `ukf` 运动模型参数扫描 | `jerk_rms_ratio`, `to_raw_translation_rmse`, SN 数据看 `std_norm_ratio` |

当前问题背景下要特别注意：滤波主要处理随机抖动和高频噪声；如果误差主项是位置相关姿态系统误差，滤波只能让轨迹更平滑，不能从根本上消除系统偏差。因此最终选择不应只看平滑比例，还要同时看相对原始轨迹偏移和参考轨迹误差。

## 批量执行

配置示例见 [examples/batch_config.yaml](examples/batch_config.yaml)。

```powershell
rt-filter batch examples/batch_config.yaml
```

真实 TRANSIC 轨迹示例：

```powershell
python scripts\prepare_transic_ref_data.py
rt-filter batch examples\transic_batch_config.yaml
```

先临 SN 真实静态稳定性数据示例：

```powershell
python scripts\prepare_sn_ref_data.py
rt-filter batch examples\sn_batch_config.yaml
```

生成带测试用例汇总的完整报告：

```powershell
python scripts\run_sn_analysis.py
```

`scripts\prepare_sn_ref_data.py` 会扫描 `ref_data\sn\*\track_data_*.txt`，按文件夹名解析底座、目标、标定状态、曝光、是否加点等条件，并将每条轨迹转换到 `input\sn\case_*\*.csv`。源数据中的 `x,y,z` 按 mm 处理，`xr,yr,zr` 按 XYZ 欧拉角 degree 转换为当前系统使用的齐次位姿矩阵和四元数。

批处理输出结构：

```text
outputs/run_YYYYMMDD_HHMMSS/
  manifest.json
  summary.csv
  trajectory_name/
    input/
      trajectory.csv
      trajectory.vtu
    moving_average/window-5/
      trajectory.csv
      trajectory.vtu
      metrics.json
      timing.csv
    savgol/window-11_polyorder-2/
      trajectory.csv
      trajectory.vtu
      metrics.json
      timing.csv
```

## 评估与统计

单独评估：

```powershell
rt-filter evaluate outputs/filtered.csv --reference reference.csv --output outputs/metrics.json
```

从批处理汇总表生成排名和趋势图：

```powershell
rt-filter report outputs/run_YYYYMMDD_HHMMSS/summary.csv --metric to_reference_translation_rmse
```

如果没有 Leica 或其他参考轨迹，可以先用 `acceleration_rms`、`jerk_rms`、`to_raw_translation_rmse`、`to_raw_translation_max` 观察滤波是否过度平滑或引入明显偏移。

### 评估输出字段说明

评估结果会同时写入每个滤波结果目录下的 `metrics.json` 和批处理总表 `summary.csv`。`rt-filter evaluate` 输出的是同一套指标；GUI 还会额外显示分维度指标和自动结论。

#### 结果文件

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `metrics.json` | 单个“输入轨迹 + 滤波算法 + 参数”的完整指标 | 追踪单个结果细节 |
| `summary.csv` | 所有算法、参数、输入轨迹的汇总表 | 排名、筛选、画趋势图 |
| `manifest.json` | 批处理配置、输入、输出路径、可视化配置 | 复现实验和定位文件 |
| `trajectory.csv` | 滤波后的轨迹 | 后续算法或人工检查 |
| `trajectory.vtu` | 带法向点云 | ParaView 三维对比 |
| `timing.csv` | 每一帧/位姿的滤波耗时（ns/us/ms） | 对比实时性和抖动峰值 |
| `report/ranked_results.csv` | 按指定 metric 排序后的总表 | 找最优参数 |
| `report/*.png` | 算法箱线图、参数趋势图 | 看参数变化趋势 |
| `sn_filter_stability.csv` | SN 静态数据专用稳定性指标 | 比较静态抖动压制效果 |
| `sn_report.md` | SN 测试用例级 Markdown 总结 | 快速看每个工况的推荐结果 |

#### 轨迹自身指标

这组字段描述滤波后轨迹本身的几何范围和动态平滑程度，字段名前缀通常是 `filtered_`。

| 字段 | 含义 | 反映的性质 |
| --- | --- | --- |
| `filtered_sample_count` | 滤波后采样点数量 | 是否有丢点或截断 |
| `filtered_path_length` | 相邻位置距离累计长度 | 轨迹整体运动量；静态数据中越小通常越稳定 |
| `filtered_duration` | 时间戳跨度 | 数据时长 |
| `filtered_sample_rate_mean` | 平均采样率 | 数据采集频率是否正常 |
| `filtered_range_x/y/z` | x/y/z 方向最大值减最小值 | 各轴抖动或运动范围；静态稳定性里是关键指标 |
| `filtered_velocity_rms` | 速度 RMS | 轨迹变化强度 |
| `filtered_acceleration_rms` | 加速度 RMS | 高频抖动和运动突变程度 |
| `filtered_jerk_rms` | jerk RMS | 更敏感的高频噪声指标；越低代表越平滑 |
| `filtered_rotation_step_rms_deg` | 相邻姿态旋转步长 RMS | 姿态抖动程度，单位 degree |

#### 相对原始轨迹的偏移指标

这组字段前缀是 `to_raw_`，用于判断滤波有没有过度改变原始轨迹。

| 字段 | 含义 | 反映的性质 |
| --- | --- | --- |
| `to_raw_translation_mean` | 滤波轨迹相对原始轨迹的位置偏移均值 | 整体偏移水平 |
| `to_raw_translation_rmse` | 位置偏移 RMSE | 滤波引入的平均改变量；过大可能是滞后或过平滑 |
| `to_raw_translation_max` | 最大位置偏移 | 最坏点偏离程度 |
| `to_raw_translation_p95` | 位置偏移 95 分位 | 排除极端点后的高位偏移 |
| `to_raw_rotation_mean_deg` | 姿态偏移均值 | 姿态整体改变量 |
| `to_raw_rotation_rmse_deg` | 姿态偏移 RMSE | 姿态滤波强度 |
| `to_raw_rotation_max_deg` | 最大姿态偏移 | 最坏姿态偏离 |

工程判断上，`to_raw_*` 不是越小越好，而是要和降噪效果一起看：如果 `jerk_rms_ratio` 降很多但 `to_raw_translation_rmse` 很大，说明滤波可能确实平滑了噪声，但也可能引入了明显滞后或形状偏移。

#### 平滑比例指标

| 字段 | 定义 | 反映的性质 |
| --- | --- | --- |
| `acceleration_rms_ratio` | `filtered_acceleration_rms / raw_acceleration_rms` | 加速度抖动压制比例 |
| `jerk_rms_ratio` | `filtered_jerk_rms / raw_jerk_rms` | 高频噪声压制比例 |

比例小于 1 表示滤波后更平滑。静态数据中通常希望它更小；动态轨迹中要防止它过小，因为真实加减速也可能被抹掉。

#### 相对参考轨迹的误差指标

当提供 Leica、目标轨迹或其他参考轨迹时，会输出 `raw_to_reference_*` 和 `to_reference_*` 两组指标。前者是原始轨迹对参考的误差，后者是滤波轨迹对参考的误差。

| 字段 | 含义 | 反映的性质 |
| --- | --- | --- |
| `raw_to_reference_translation_rmse` | 原始轨迹相对参考的位置 RMSE | 原始数据精度基线 |
| `to_reference_translation_rmse` | 滤波轨迹相对参考的位置 RMSE | 滤波后绝对精度 |
| `raw_to_reference_translation_max` | 原始轨迹最大位置误差 | 原始最坏点 |
| `to_reference_translation_max` | 滤波后最大位置误差 | 滤波后最坏点 |
| `*_translation_mean` | 平均位置误差 | 整体偏差水平 |
| `*_translation_p95` | 位置误差 95 分位 | 稳健高位误差 |
| `*_rotation_mean_deg` | 平均姿态误差 | 姿态平均偏差 |
| `*_rotation_rmse_deg` | 姿态误差 RMSE | 姿态整体精度 |
| `*_rotation_max_deg` | 最大姿态误差 | 姿态最坏点 |
| `reference_rmse_improvement` | `raw_to_reference_translation_rmse - to_reference_translation_rmse` | 正值表示滤波让轨迹更接近参考 |
| `reference_max_improvement` | `raw_to_reference_translation_max - to_reference_translation_max` | 正值表示最大误差改善 |

如果没有参考轨迹，就不能判断绝对精度是否变好，只能判断“是否更平滑”和“是否偏离原始轨迹”。

#### GUI 分维度指标

GUI 的分析核心会额外输出每个方向的偏移和平滑比例，便于判断噪声是否集中在某个方向。

| 字段 | 含义 | 反映的性质 |
| --- | --- | --- |
| `to_raw_x/y/z_mean` | 每个轴向的平均偏移 | 某一方向是否有系统性漂移 |
| `to_raw_x/y/z_rmse` | 每个轴向的偏移 RMSE | 哪个轴被滤波改动最大 |
| `to_raw_x/y/z_max_abs` | 每个轴最大绝对偏移 | 单轴最坏偏离 |
| `x/y/z_acceleration_rms_ratio` | 单轴加速度 RMS 比例 | 单轴平滑效果 |
| `x/y/z_jerk_rms_ratio` | 单轴 jerk RMS 比例 | 单轴高频噪声压制效果 |
| `to_raw_rotation_p95_deg` | 姿态偏移 95 分位 | 稳健姿态改变量 |
| `to_reference_x/y/z_rmse` | 单轴参考误差 RMSE | 哪个方向绝对误差较大 |
| `raw_to_reference_x/y/z_rmse` | 原始单轴参考误差 RMSE | 单轴基线误差 |

对于当前机械臂场景，如果 z 方向稳定性明显差，优先看 `to_raw_z_rmse`、`z_jerk_rms_ratio`、`to_reference_z_rmse` 或 SN 报告中的 `z_std_ratio`。

#### SN 静态稳定性专用指标

`scripts\run_sn_analysis.py` 会额外生成 `sn_filter_stability.csv`，把同一文件夹下的真实数据整理成测试用例，并输出更适合静态稳定性判断的字段。

| 字段 | 含义 | 反映的性质 |
| --- | --- | --- |
| `case_id` | 由文件夹名解析出的测试用例 | 同一工况不同参数的归类键 |
| `base` | 底座条件，如低底座、高底座、新底座 | 安装条件 |
| `target` | 拍摄对象，如机械臂、物体 | 测试对象 |
| `calibration` | 标定状态 | 标定变化对稳定性的影响 |
| `exposure` | 曝光条件 | 曝光参数影响 |
| `added_points` | 是否加点 | 标志点几何变化影响 |
| `tracker_smoothing` | 源数据文件名里的设备平滑开关 | 设备内置平滑对比 |
| `raw_range_x/y/z` | 原始轨迹各轴极差 | 原始静态抖动范围 |
| `filtered_range_x/y/z` | 滤波后各轴极差 | 滤波后静态抖动范围 |
| `raw_range_norm` | 原始三轴极差向量模长 | 综合抖动范围 |
| `filtered_range_norm` | 滤波后三轴极差向量模长 | 综合稳定性结果 |
| `raw_std_x/y/z` | 原始各轴标准差 | 原始随机波动强度 |
| `filtered_std_x/y/z` | 滤波后各轴标准差 | 滤波后随机波动强度 |
| `raw_std_norm` | 原始三轴标准差向量模长 | 综合随机抖动 |
| `filtered_std_norm` | 滤波后三轴标准差向量模长 | 综合降噪效果 |
| `range_norm_ratio` | `filtered_range_norm / raw_range_norm` | 综合极差压制比例 |
| `std_norm_ratio` | `filtered_std_norm / raw_std_norm` | 综合标准差压制比例 |
| `z_range_ratio` | `filtered_range_z / raw_range_z` | z 向极差压制比例 |
| `z_std_ratio` | `filtered_std_z / raw_std_z` | z 向标准差压制比例 |
| `raw_radial_p95/max` | 原始点到均值中心的径向 95 分位/最大值 | 原始空间散布程度 |
| `filtered_radial_p95/max` | 滤波后点到均值中心的径向 95 分位/最大值 | 滤波后空间散布程度 |

SN 静态数据没有外部绝对参考，因此推荐用 `filtered_range_norm`、`std_norm_ratio`、`z_std_ratio`、`to_raw_translation_rmse` 一起判断。一个结果如果 `std_norm_ratio` 很低但 `to_raw_translation_rmse` 很高，说明它压低了抖动，但也明显改变了原始点位中心或趋势。

## VTK 位姿点云导出

可以将轨迹导出为带法向点云，输出是 `vtkUnstructuredGrid`：每个位姿是一个 `VTK_VERTEX`，点坐标为平移量，点法向默认取局部 `z` 轴在 Base/世界坐标系下的方向。

```powershell
rt-filter export-vtk input.csv outputs/trajectory.vtu
rt-filter export-vtk input.csv outputs/trajectory.vtk --normal-axis z
```

`.vtu` 为 VTK XML 格式，`.vtk` 为 legacy ASCII 格式。点数据中包含 `Normals`、`XAxis`、`YAxis`、`ZAxis`、`SampleIndex`、可选 `Timestamp` 和 `PathDistance`，可直接在 ParaView 中按法向或三轴向量显示。

## Qt 分析界面

安装 GUI 依赖：

```bash
python -m pip install -e .[gui]
```

启动界面：

```bash
rt-filter gui
# 或
rt-filter-gui
```

GUI 会在运行时读取 `rt_filter_gui.json`，这样默认路径不需要再写死在代码里。
常见搜索位置包括：

- 当前工作目录
- 可执行程序旁边；macOS `.app` 还会额外检查 `.app` 同级目录
- `~/Library/Application Support/rt-filter/rt_filter_gui.json` 或 `%LOCALAPPDATA%\rt-filter\rt_filter_gui.json`
- 仓库根目录

默认配置文件示例：

```json
{
  "input_roots": [
    {
      "path": "input/sn",
      "patterns": ["case_*/*.csv"]
    },
    {
      "path": "examples/demo_data",
      "patterns": ["*.csv"]
    }
  ],
  "reference_path": "",
  "output_dir": "outputs/gui",
  "auto_load_inputs": true,
  "restore_recent_inputs": true,
  "recent_input_files": [],
  "selected_input_file": ""
}
```

字段说明：

- `input_roots`：按顺序定义默认输入目录和自动加载 glob 规则，第一项也会作为文件选择器的默认打开目录
- `reference_path`：默认参考轨迹文件，可留空
- `output_dir`：GUI 默认输出目录
- `auto_load_inputs`：是否在启动时自动加载默认测试用例
- `restore_recent_inputs`：是否优先恢复上一次仍存在的输入文件列表
- `recent_input_files`：上一次 GUI 中已添加且仍存在的输入文件；程序会自动维护
- `selected_input_file`：上一次选中的输入文件；程序会自动维护

相对路径默认按 `rt_filter_gui.json` 所在目录解析；对输入目录、参考文件和最近使用文件，若该位置不存在，程序还会回退到当前工程目录/打包资源目录继续查找，所以源码运行和打包运行都能共用这份配置。

构建 Windows/macOS 可执行程序：

```bash
python -m pip install -e .[gui,build]
python scripts/build_gui.py
```

构建脚本会调用 `PyInstaller`，并自动执行一次 `--smoke-test`：

- macOS 产物：`dist/rt-filter-gui.app`
- Windows 产物：`dist/rt-filter-gui/rt-filter-gui.exe`

构建脚本还会把 `rt_filter_gui.json` 复制到打包产物旁边，便于后续直接修改配置。

如果没有额外修改 `rt_filter_gui.json`，打包后的 GUI 默认会优先加载 `input/sn/` 测试用例，找不到时再回退到 `examples/demo_data/`。
默认输出目录由 `rt_filter_gui.json` 控制；当前默认值是相对配置文件位置的 `outputs/gui`。

界面工作流：

- 选择一个或多个输入轨迹，默认会按 `rt_filter_gui.json` 中的 `input_roots` 自动加载测试用例
- `Add Dir` 会递归扫描所选目录及其子目录中的 `.csv/.txt/.json/.npy/.npz` 轨迹文件
- GUI 会把上一次已添加且仍存在的输入文件写回 `rt_filter_gui.json`，下次启动时优先恢复这些 case
- 在滤波表中勾选算法，参数用 JSON 写法，列表值会展开成参数网格
- 运行后查看指标表、维度结论和曲线图；右下角新增 `Per-frame Compute Time` 图，可直接比较不同滤波方法的逐帧耗时
- 结果会写入 `outputs/gui/run_YYYYMMDD_HHMMSS`
- 每个滤波结果目录会额外写出 `timing.csv`，并在 `summary.csv/metrics.json` 中记录总耗时、单帧均值、P95、最大值
- `Generate ParaView Script` 会写出 `paraview_compare.py`，同时引用原始和各滤波结果 `.vtu`

ParaView 对比脚本运行方式：

```powershell
pvpython outputs\gui\run_YYYYMMDD_HHMMSS\paraview_compare.py
```

脚本会创建并排 RenderView，显示每条轨迹的点云，并用 `Normals` 生成箭头方向用于姿态对比。

## Python API

```python
from rt_filter import read_trajectory, write_trajectory, run_filter, run_filter_timed
from rt_filter.vtk_export import write_vtk_unstructured_grid

traj = read_trajectory("input.csv")
filtered = run_filter("moving_average", traj, {"window": 9})
write_trajectory(filtered, "outputs/filtered.csv")
write_vtk_unstructured_grid(filtered, "outputs/filtered.vtu")

timed = run_filter_timed("moving_average", traj, {"window": 9})
print(timed.total_time_ns, timed.per_pose_time_ns[:5])
```

# 机械臂轨迹滤波系统

这个目录现在包含一个可运行的 Python 工程，用于对机械臂 6DoF 轨迹做读取、滤波、批量执行、指标验证和统计可视化。

系统内部统一使用 `N x 4 x 4` 齐次变换矩阵表示轨迹，同时兼容常见的 `x,y,z,qw,qx,qy,qz` 输入。文档中的问题判断表明当前数据可能包含姿态系统误差和杠杆臂放大效应，因此默认评估会同时关注平滑程度、对原始轨迹的偏移、相对参考轨迹的误差，以及滤波引入的滞后风险。

## 安装

```powershell
python -m pip install -e .[dev]
```

## 输入格式

支持 `.csv`、`.json`、`.npy`、`.npz`。

CSV 可使用以下任一形式：

- 位姿列：`x,y,z,qw,qx,qy,qz`
- 位姿列：`x,y,z,qx,qy,qz,qw`
- 矩阵列：`m00,m01,...,m33`
- 可选时间列：`timestamp`、`time` 或 `t`
- 可选状态列：`status`，值为 `-1` 的行默认丢弃

## 单条轨迹滤波

```powershell
rt-filter filter input.csv outputs/filtered.csv --algorithm moving_average --param window=9
rt-filter filter input.csv outputs/filtered.csv --algorithm savgol --param window=11 --param polyorder=2
rt-filter filter input.csv outputs/filtered.csv --algorithm exponential --param alpha=0.25
```

查看可用算法：

```powershell
rt-filter catalog
```

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
    savgol/window-11_polyorder-2/
      trajectory.csv
      trajectory.vtu
      metrics.json
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

## VTK 位姿点云导出

可以将轨迹导出为带法向点云，输出是 `vtkUnstructuredGrid`：每个位姿是一个 `VTK_VERTEX`，点坐标为平移量，点法向默认取局部 `z` 轴在 Base/世界坐标系下的方向。

```powershell
rt-filter export-vtk input.csv outputs/trajectory.vtu
rt-filter export-vtk input.csv outputs/trajectory.vtk --normal-axis z
```

`.vtu` 为 VTK XML 格式，`.vtk` 为 legacy ASCII 格式。点数据中包含 `Normals`、`XAxis`、`YAxis`、`ZAxis`、`SampleIndex`、可选 `Timestamp` 和 `PathDistance`，可直接在 ParaView 中按法向或三轴向量显示。

## Qt 分析界面

安装 GUI 依赖：

```powershell
python -m pip install -e .[gui]
```

启动界面：

```powershell
rt-filter gui
# 或
rt-filter-gui
```

界面工作流：

- 选择一个或多个输入轨迹，默认会自动加载 `input/transic_*.csv`
- 在滤波表中勾选算法，参数用 JSON 写法，列表值会展开成参数网格
- 运行后查看指标表、维度结论和曲线图
- 结果会写入 `outputs/gui/run_YYYYMMDD_HHMMSS`
- `Generate ParaView Script` 会写出 `paraview_compare.py`，同时引用原始和各滤波结果 `.vtu`

ParaView 对比脚本运行方式：

```powershell
pvpython outputs\gui\run_YYYYMMDD_HHMMSS\paraview_compare.py
```

脚本会创建并排 RenderView，显示每条轨迹的点云，并用 `Normals` 生成箭头方向用于姿态对比。

## Python API

```python
from rt_filter import read_trajectory, write_trajectory, run_filter
from rt_filter.vtk_export import write_vtk_unstructured_grid

traj = read_trajectory("input.csv")
filtered = run_filter("moving_average", traj, {"window": 9})
write_trajectory(filtered, "outputs/filtered.csv")
write_vtk_unstructured_grid(filtered, "outputs/filtered.vtu")
```

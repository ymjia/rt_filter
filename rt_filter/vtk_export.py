from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from rt_filter.trajectory import Trajectory


ArrayF = NDArray[np.float64]


def write_vtk_unstructured_grid(
    traj: Trajectory,
    path: str | Path,
    *,
    normal_axis: str | Iterable[float] = "z",
    include_axes: bool = True,
    include_path_distance: bool = True,
) -> None:
    """Write trajectory poses as a point cloud in a VTK UnstructuredGrid file.

    Each pose is represented as one ``VTK_VERTEX`` cell. Point coordinates are
    pose translations, and point normals are generated from one local pose axis
    transformed into world/base coordinates. ``.vtu`` and legacy ASCII ``.vtk``
    outputs are supported.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".vtu":
        _write_vtu(
            traj,
            output,
            normal_axis=normal_axis,
            include_axes=include_axes,
            include_path_distance=include_path_distance,
        )
        return
    if suffix == ".vtk":
        _write_legacy_vtk(
            traj,
            output,
            normal_axis=normal_axis,
            include_axes=include_axes,
            include_path_distance=include_path_distance,
        )
        return
    raise ValueError("VTK output path must end with .vtu or .vtk")


def trajectory_point_data(
    traj: Trajectory,
    *,
    normal_axis: str | Iterable[float] = "z",
    include_axes: bool = True,
    include_path_distance: bool = True,
) -> dict[str, np.ndarray]:
    """Build point-data arrays used by the VTK exporters."""

    data: dict[str, np.ndarray] = {
        "Normals": _pose_normals(traj, normal_axis),
        "SampleIndex": np.arange(traj.count, dtype=np.int64),
    }
    if traj.timestamps is not None:
        data["Timestamp"] = np.asarray(traj.timestamps, dtype=np.float64)
    if include_path_distance:
        data["PathDistance"] = _path_distance(traj.positions)
    if include_axes:
        data["XAxis"] = _normalized_vectors(traj.poses[:, :3, 0])
        data["YAxis"] = _normalized_vectors(traj.poses[:, :3, 1])
        data["ZAxis"] = _normalized_vectors(traj.poses[:, :3, 2])
    return data


def _write_vtu(
    traj: Trajectory,
    path: Path,
    *,
    normal_axis: str | Iterable[float],
    include_axes: bool,
    include_path_distance: bool,
) -> None:
    vtk_file = ET.Element(
        "VTKFile",
        {
            "type": "UnstructuredGrid",
            "version": "0.1",
            "byte_order": "LittleEndian",
        },
    )
    grid = ET.SubElement(vtk_file, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        {
            "NumberOfPoints": str(traj.count),
            "NumberOfCells": str(traj.count),
        },
    )
    point_data = ET.SubElement(piece, "PointData", {"Normals": "Normals"})
    for name, values in trajectory_point_data(
        traj,
        normal_axis=normal_axis,
        include_axes=include_axes,
        include_path_distance=include_path_distance,
    ).items():
        _add_data_array(point_data, name, values)

    points = ET.SubElement(piece, "Points")
    _add_data_array(points, None, traj.positions, number_of_components=3)

    cells = ET.SubElement(piece, "Cells")
    indices = np.arange(traj.count, dtype=np.int64)
    _add_data_array(cells, "connectivity", indices, vtk_type="Int64")
    _add_data_array(cells, "offsets", indices + 1, vtk_type="Int64")
    _add_data_array(cells, "types", np.ones(traj.count, dtype=np.uint8), vtk_type="UInt8")

    _indent_xml(vtk_file)
    tree = ET.ElementTree(vtk_file)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _write_legacy_vtk(
    traj: Trajectory,
    path: Path,
    *,
    normal_axis: str | Iterable[float],
    include_axes: bool,
    include_path_distance: bool,
) -> None:
    point_data = trajectory_point_data(
        traj,
        normal_axis=normal_axis,
        include_axes=include_axes,
        include_path_distance=include_path_distance,
    )
    lines = [
        "# vtk DataFile Version 3.0",
        f"Trajectory {traj.name}",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {traj.count} double",
        *_format_rows(traj.positions),
        f"CELLS {traj.count} {traj.count * 2}",
        *[f"1 {idx}" for idx in range(traj.count)],
        f"CELL_TYPES {traj.count}",
        *_format_scalar_rows(np.ones(traj.count, dtype=np.uint8)),
        f"POINT_DATA {traj.count}",
        "NORMALS Normals double",
        *_format_rows(point_data.pop("Normals")),
    ]

    for name, values in point_data.items():
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == 3:
            lines.append(f"VECTORS {name} double")
            lines.extend(_format_rows(arr))
        else:
            lines.append(f"SCALARS {name} {_legacy_scalar_type(arr)} 1")
            lines.append("LOOKUP_TABLE default")
            lines.extend(_format_scalar_rows(arr))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_data_array(
    parent: ET.Element,
    name: str | None,
    values: np.ndarray,
    *,
    number_of_components: int | None = None,
    vtk_type: str | None = None,
) -> None:
    arr = np.asarray(values)
    attributes = {"type": _vtk_xml_type(arr) if vtk_type is None else vtk_type, "format": "ascii"}
    if name is not None:
        attributes["Name"] = name
    if number_of_components is None and arr.ndim == 2:
        number_of_components = arr.shape[1]
    if number_of_components is not None:
        attributes["NumberOfComponents"] = str(number_of_components)
    element = ET.SubElement(parent, "DataArray", attributes)
    element.text = "\n" + _format_array(arr) + "\n"


def _pose_normals(traj: Trajectory, normal_axis: str | Iterable[float]) -> ArrayF:
    local_axis = _local_axis_vector(normal_axis)
    normals = np.einsum("nij,j->ni", traj.poses[:, :3, :3], local_axis)
    return _normalized_vectors(normals)


def _local_axis_vector(axis: str | Iterable[float]) -> ArrayF:
    if isinstance(axis, str):
        key = axis.strip().lower()
        sign = -1.0 if key.startswith("-") else 1.0
        key = key[1:] if key.startswith("-") else key
        mapping = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if key not in mapping:
            raise ValueError("normal_axis must be x, y, z, -x, -y, -z, or a 3-vector")
        return sign * mapping[key]

    vec = np.asarray(list(axis), dtype=float)
    if vec.shape != (3,):
        raise ValueError("normal_axis vector must contain exactly 3 values")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("normal_axis vector must be non-zero")
    return vec / norm


def _normalized_vectors(vectors: np.ndarray) -> ArrayF:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _path_distance(positions: np.ndarray) -> ArrayF:
    if len(positions) == 1:
        return np.zeros(1, dtype=float)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _format_array(values: np.ndarray) -> str:
    arr = np.asarray(values)
    if arr.ndim == 1:
        return " ".join(_format_number(value) for value in arr)
    return "\n".join(" ".join(_format_number(value) for value in row) for row in arr)


def _format_rows(values: np.ndarray) -> list[str]:
    return [" ".join(_format_number(value) for value in row) for row in np.asarray(values)]


def _format_scalar_rows(values: np.ndarray) -> list[str]:
    return [_format_number(value) for value in np.asarray(values).reshape(-1)]


def _format_number(value: object) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.17g}"
    return str(value)


def _vtk_xml_type(values: np.ndarray) -> str:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        return "UInt64" if arr.dtype.itemsize > 4 else "UInt32"
    if np.issubdtype(arr.dtype, np.integer):
        return "Int64" if arr.dtype.itemsize > 4 else "Int32"
    return "Float64"


def _legacy_scalar_type(values: np.ndarray) -> str:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        return "unsigned_long" if arr.dtype.itemsize > 4 else "unsigned_int"
    if np.issubdtype(arr.dtype, np.integer):
        return "long" if arr.dtype.itemsize > 4 else "int"
    return "double"


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            _indent_xml(child, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = indent
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent
    else:
        if not element.tail or not element.tail.strip():
            element.tail = indent

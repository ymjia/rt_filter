from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from rt_filter.se3 import make_poses
from rt_filter.trajectory import Trajectory
from rt_filter.vtk_export import trajectory_point_data, write_vtk_unstructured_grid


def test_trajectory_point_data_uses_pose_z_axis_as_default_normal():
    positions = np.zeros((2, 3))
    rotations = Rotation.from_rotvec(
        np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 2.0, 0.0]])
    )
    traj = Trajectory(make_poses(positions, rotations), name="normals")

    data = trajectory_point_data(traj)

    np.testing.assert_allclose(data["Normals"][0], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(data["Normals"][1], [1.0, 0.0, 0.0], atol=1e-12)


def test_write_vtu_unstructured_grid_with_normals(tmp_path):
    positions = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    rotations = Rotation.identity(3)
    traj = Trajectory(
        make_poses(positions, rotations),
        timestamps=np.array([0.0, 0.1, 0.2]),
        name="sample",
    )
    output = tmp_path / "sample.vtu"

    write_vtk_unstructured_grid(traj, output)

    root = ET.parse(output).getroot()
    assert root.attrib["type"] == "UnstructuredGrid"
    piece = root.find("./UnstructuredGrid/Piece")
    assert piece is not None
    assert piece.attrib["NumberOfPoints"] == "3"
    assert piece.attrib["NumberOfCells"] == "3"
    point_data = piece.find("PointData")
    assert point_data is not None
    assert point_data.attrib["Normals"] == "Normals"
    names = {node.attrib.get("Name") for node in point_data.findall("DataArray")}
    assert {"Normals", "XAxis", "YAxis", "ZAxis", "Timestamp", "PathDistance"} <= names


def test_write_legacy_vtk_unstructured_grid(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    traj = Trajectory(make_poses(positions, Rotation.identity(2)), name="legacy")
    output = tmp_path / "sample.vtk"

    write_vtk_unstructured_grid(traj, output)

    text = output.read_text(encoding="utf-8")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "POINT_DATA 2" in text
    assert "NORMALS Normals double" in text

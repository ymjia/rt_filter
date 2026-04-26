from __future__ import annotations

import numpy as np
import pytest

from rt_filter.gui.chart_data import neighbor_mean_deviation


def test_neighbor_mean_deviation_is_zero_for_centered_linear_samples():
    x = np.arange(7, dtype=float)
    positions = np.column_stack([x, 2.0 * x, -x])

    deviations = neighbor_mean_deviation(positions, window=2)

    np.testing.assert_allclose(deviations[2:5], 0.0, atol=1e-12)


def test_neighbor_mean_deviation_highlights_local_spike():
    positions = np.zeros((5, 3), dtype=float)
    positions[2] = [10.0, -4.0, 2.0]

    deviations = neighbor_mean_deviation(positions, window=1)

    np.testing.assert_allclose(deviations[2], [10.0, -4.0, 2.0])
    np.testing.assert_allclose(deviations[1], [-5.0, 2.0, -1.0])
    np.testing.assert_allclose(deviations[3], [-5.0, 2.0, -1.0])


def test_neighbor_mean_deviation_handles_single_sample():
    positions = np.array([[1.0, 2.0, 3.0]])

    deviations = neighbor_mean_deviation(positions, window=10)

    np.testing.assert_allclose(deviations, np.zeros((1, 3)))


def test_neighbor_mean_deviation_validates_shape_and_window():
    with pytest.raises(ValueError, match="shape"):
        neighbor_mean_deviation(np.zeros((4, 2)), window=10)
    with pytest.raises(ValueError, match="window"):
        neighbor_mean_deviation(np.zeros((4, 3)), window=0)

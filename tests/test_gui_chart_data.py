from __future__ import annotations

import numpy as np
import pytest

from rt_filter.gui.chart_data import (
    complete_neighbor_slice,
    fit_expected_path,
    fit_expected_path_cached,
    neighbor_mean_deviation,
    path_deviation,
)


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


def test_complete_neighbor_slice_removes_incomplete_edge_windows():
    values = np.arange(30)

    trimmed = values[complete_neighbor_slice(len(values), window=10)]

    np.testing.assert_array_equal(trimmed, np.arange(10, 20))


def test_complete_neighbor_slice_returns_empty_when_window_does_not_fit():
    values = np.arange(20)

    trimmed = values[complete_neighbor_slice(len(values), window=10)]

    assert trimmed.size == 0


def test_expected_path_uses_static_endpoint_line_when_available():
    rng = np.random.default_rng(3)
    start_static = rng.normal(scale=0.005, size=(6, 3))
    moving_x = np.linspace(0.0, 10.0, 31)
    moving = np.column_stack([moving_x, 0.03 * np.sin(moving_x), np.zeros_like(moving_x)])
    end_static = np.array([10.0, 0.0, 0.0]) + rng.normal(scale=0.005, size=(6, 3))
    positions = np.vstack([start_static, moving, end_static])

    model = fit_expected_path(positions)
    deviation = path_deviation(positions, model)

    assert model.kind == "line-static"
    assert model.details["start_static_samples"] >= 6
    assert model.details["end_static_samples"] >= 6
    assert float(np.percentile(deviation.cross, 95)) < 0.04


def test_expected_path_deviation_exposes_line_lag_as_negative_along_error():
    start_static = np.zeros((6, 3), dtype=float)
    moving_x = np.linspace(0.0, 10.0, 31)
    moving = np.column_stack([moving_x, np.zeros_like(moving_x), np.zeros_like(moving_x)])
    end_static = np.tile(np.array([10.0, 0.0, 0.0]), (6, 1))
    raw = np.vstack([start_static, moving, end_static])
    filtered = raw.copy()
    filtered[8:35, 0] -= 0.5

    model = fit_expected_path(raw)
    deviation = path_deviation(filtered, model)

    assert model.kind == "line-static"
    np.testing.assert_allclose(np.median(deviation.along[10:33]), -0.5, atol=1e-12)
    np.testing.assert_allclose(np.median(deviation.cross[10:33]), 0.0, atol=1e-12)


def test_expected_path_detects_ellipse():
    theta = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    positions = np.column_stack(
        [
            3.0 * np.cos(theta),
            1.5 * np.sin(theta),
            0.02 * np.sin(2.0 * theta),
        ]
    )

    model = fit_expected_path(positions)
    deviation = path_deviation(positions, model)

    assert model.kind == "ellipse"
    assert float(np.percentile(deviation.cross, 95)) < 0.08


def test_expected_path_falls_back_to_polyline_for_cornered_motion():
    first = np.column_stack(
        [
            np.linspace(0.0, 5.0, 40),
            np.zeros(40),
            np.zeros(40),
        ]
    )
    second = np.column_stack(
        [
            np.full(40, 5.0),
            np.linspace(0.0, 4.0, 40),
            np.zeros(40),
        ]
    )
    positions = np.vstack([first, second[1:]])

    model = fit_expected_path(positions)
    deviation = path_deviation(positions, model)

    assert model.kind == "polyline"
    assert model.details["vertices"] >= 3
    assert float(np.percentile(deviation.cross, 95)) < 1e-9


def test_expected_path_uses_high_order_curve_when_simple_models_exceed_threshold():
    x = np.linspace(0.0, 1000.0, 500)
    positions = np.column_stack(
        [
            x,
            200.0 * np.sin(2.0 * np.pi * x / 1000.0),
            np.zeros_like(x),
        ]
    )

    model = fit_expected_path(positions)
    deviation = path_deviation(positions, model)

    assert model.kind == "savgol-curve"
    assert model.details["max_deviation_mm"] <= 10.0
    assert float(np.max(deviation.norm)) <= 10.0


def test_expected_path_cache_round_trips_successful_fit(tmp_path):
    x = np.linspace(0.0, 1000.0, 500)
    positions = np.column_stack(
        [
            x,
            200.0 * np.sin(2.0 * np.pi * x / 1000.0),
            np.zeros_like(x),
        ]
    )
    source = tmp_path / "case_99.csv"
    source.write_text("placeholder", encoding="utf-8")

    first = fit_expected_path_cached(positions, source_path=source, cache_dir=tmp_path)
    cache_files = sorted(tmp_path.glob("case_99__*.npz"))
    second = fit_expected_path_cached(positions, source_path=source, cache_dir=tmp_path)

    assert len(cache_files) == 1
    assert second.kind == first.kind
    np.testing.assert_allclose(second.expected, first.expected)
    np.testing.assert_allclose(second.tangent, first.tangent)

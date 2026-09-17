from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from eeg_viewer.topomap import DualTopomapWidget, TopomapInterpolator


def test_topomap_interpolation_is_constant_inside_and_transparent_outside() -> None:
    positions = np.array(
        [[-0.7, 0.6], [0.7, 0.6], [-0.6, -0.7], [0.6, -0.7], [0.0, 0.0]],
        dtype=float,
    )
    interpolator = TopomapInterpolator(positions, grid_size=48)

    image = interpolator.interpolate(np.full(5, 3.25))

    np.testing.assert_allclose(image[interpolator.inside_mask], 3.25, atol=1e-6)
    assert np.isnan(image[~interpolator.inside_mask]).all()


def test_topomap_interpolation_tolerates_missing_and_collinear_sensors() -> None:
    interpolator = TopomapInterpolator(
        np.array([[-0.8, 0.8], [0.0, 0.0], [0.8, -0.8]], dtype=float),
        grid_size=32,
    )

    image = interpolator.interpolate(np.array([-2.0, np.nan, 4.0]))

    assert np.isfinite(image[interpolator.inside_mask]).all()
    assert image[-6, 5] < image[5, -6]


def test_thin_plate_spline_reproduces_a_smooth_linear_field() -> None:
    positions = np.array(
        [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8], [0.0, 0.0]],
        dtype=float,
    )
    values = positions[:, 0] + 2.0 * positions[:, 1]
    interpolator = TopomapInterpolator(positions, grid_size=65)

    image = interpolator.interpolate(values)

    axis = np.linspace(-1.0, 1.0, 65)
    grid_x, grid_y = np.meshgrid(axis, axis)
    expected = grid_x + 2.0 * grid_y
    np.testing.assert_allclose(
        image[interpolator.inside_mask],
        expected[interpolator.inside_mask],
        atol=1e-5,
    )


def test_dual_topomap_uses_independent_symmetric_scales() -> None:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    positions = np.array(
        [[-0.7, 0.6], [0.7, 0.6], [-0.6, -0.7], [0.6, -0.7]],
        dtype=float,
    )
    widget = DualTopomapWidget(
        positions,
        ("Fp1", "Fp2", "O1", "O2"),
        (0, 1, 2, 3),
        sample_rate_hz=500.0,
    )

    widget.set_maps(
        np.array([-1.0, -0.5, 0.5, 1.0]),
        np.array([-4.0, -2.0, 2.0, 4.0]),
        trial_number=7,
        time_seconds=0.125,
        window_seconds=0.02,
        unit="native [uV]",
    )
    widget.show()
    application.processEvents()

    assert widget.last_clean_limit == 1.0
    assert widget.last_current_limit == 4.0
    assert widget.clean_map.scale.limit == 1.0
    assert widget.current_map.scale.limit == 4.0
    assert widget.clean_map.image.image.shape == (128, 128, 4)
    assert widget.current_map.image.image.shape == (128, 128, 4)
    assert widget.current_map.title.text() == "Trial 7"
    assert "125.0 ms" in widget.time_label.text()
    widget.close()

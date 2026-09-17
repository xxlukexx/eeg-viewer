from __future__ import annotations

import numpy as np

from eeg_viewer.core import (
    ArraySignalSource,
    ViewState,
    fit_average_to_channel,
    reduce_ordered_extrema,
    stack_for_plot,
)


def test_array_source_reads_a_view_and_clamps_bounds() -> None:
    data = np.arange(40, dtype=np.float32).reshape(4, 10)
    source = ArraySignalSource(data, 100.0, ("a", "b", "c", "d"))

    result = source.read(slice(1, 3), -5, 99)

    assert result.shape == (2, 10)
    assert np.shares_memory(result, data)


def test_view_state_clamps_navigation_and_preserves_zoom_anchor() -> None:
    state = ViewState(
        sample_count=10_000,
        sample_rate_hz=1_000.0,
        channel_count=64,
        start_sample=4_000,
        window_seconds=2.0,
        visible_channels=16,
    )
    old_centre = state.start_sample + state.window_samples // 2

    state.zoom_time(0.5, 0.5)

    new_centre = state.start_sample + state.window_samples // 2
    assert abs(new_centre - old_centre) <= 1
    state.move_time_fraction(100)
    assert state.start_sample == state.max_start_sample
    state.move_channels(100)
    assert state.first_channel == state.max_first_channel


def test_reduction_preserves_extrema_and_temporal_order() -> None:
    data = np.zeros((2, 100), dtype=np.float32)
    data[0, 37] = 250.0
    data[0, 40] = -220.0
    data[1, 99] = 310.0  # exercise the padded final bin

    reduced = reduce_ordered_extrema(data, start_sample=1_000, max_time_bins=10)

    assert reduced.samples_per_bin == 10
    assert 250.0 in reduced.values[0]
    assert -220.0 in reduced.values[0]
    assert 310.0 in reduced.values[1]
    spike_index = int(np.flatnonzero(reduced.values[0] == 250.0)[0])
    trough_index = int(np.flatnonzero(reduced.values[0] == -220.0)[0])
    assert spike_index < trough_index
    assert reduced.sample_positions[1, -1] == 1_099


def test_stack_for_plot_separates_channels_with_nan_breaks() -> None:
    data = np.array([[0.0, 10.0], [0.0, -10.0]], dtype=np.float32)
    reduced = reduce_ordered_extrema(data, start_sample=0, max_time_bins=10)

    x, y, offsets = stack_for_plot(reduced, 100.0, 10.0)

    assert offsets.tolist() == [1.0, 0.0]
    assert np.isnan(x[2]) and np.isnan(y[2])
    assert y[1] == 2.0
    assert y[4] == -1.0


def test_clean_average_fits_each_channel_around_zero() -> None:
    average = np.array(
        [[10, 11, 12, np.nan], [-100, 0, 100, 0], [5, 5, 5, 5], [np.nan] * 4],
        dtype=np.float32,
    )

    fitted = fit_average_to_channel(average, half_height=0.42)

    np.testing.assert_allclose(fitted[0, :3], [0.35, 0.385, 0.42])
    np.testing.assert_allclose(fitted[1], [-0.42, 0, 0.42, 0])
    np.testing.assert_allclose(fitted[2], 0.42)
    assert np.isnan(fitted[0, 3])
    assert np.isnan(fitted[3]).all()

from __future__ import annotations

import numpy as np

from eeg_viewer.core import reduce_ordered_extrema
from eeg_viewer.geometry import place_waveforms, placements_from_layout
from eeg_viewer.layout import grid_layout


def test_tile_geometry_batches_all_channels_and_keeps_extrema() -> None:
    data = np.array([[0, 100, -100, 0], [0, 25, -25, 0]], dtype=np.float32)
    reduced = reduce_ordered_extrema(data, start_sample=10, max_time_bins=4)
    placements = placements_from_layout(grid_layout(("a", "b")))

    x, y = place_waveforms(reduced, placements, amplitude_spacing=100.0)

    assert x.shape == y.shape == (10,)
    assert np.isnan(x[4]) and np.isnan(x[9])
    assert np.nanmax(y[:4]) > placements[0].center_y
    assert np.nanmin(y[:4]) < placements[0].center_y


def test_generated_grid_tiles_do_not_overlap() -> None:
    placements = placements_from_layout(grid_layout(tuple(f"E{i}" for i in range(70))))

    for index, first in enumerate(placements):
        for second in placements[index + 1 :]:
            separated = (
                abs(first.center_x - second.center_x) >= first.width
                or abs(first.center_y - second.center_y) >= first.height
            )
            assert separated

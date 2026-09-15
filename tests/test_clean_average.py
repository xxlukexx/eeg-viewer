from __future__ import annotations

import numpy as np
import pytest

from eeg_viewer.model import (
    ArtifactLayer,
    ArtifactState,
    ChannelInfo,
    DatasetKind,
    DatasetViewSource,
    EegDataset,
    InMemorySignalSource,
    SegmentInfo,
)


def test_clean_average_aligns_time_and_excludes_flags_per_channel() -> None:
    blocks = (
        np.array([[[1, 1, 1, 1], [0, 0, 0, 0], [8, 8, 8, 8]]], dtype=np.float32),
        np.array([[[9, 9, 9, 9], [2, 2, 2, 2], [8, 8, 8, 8]]], dtype=np.float32),
        np.array([[[3, 3, np.nan], [10, 10, 10], [8, 8, 8]]], dtype=np.float32),
        np.array([[[7, 7, 7, 7], [4, 4, 4, 4], [8, 8, 8, 8]]], dtype=np.float32),
    )
    artifacts = ArtifactState(
        layers=(
            ArtifactLayer(
                "blink",
                np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
            ),
        ),
        interpolated=np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool),
        cannot_interpolate=np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=bool
        ),
    )
    dataset = EegDataset(
        kind=DatasetKind.SEGMENTED,
        channels=tuple(ChannelInfo(i, f"C{i}") for i in range(3)),
        segments=tuple(
            SegmentInfo(i, f"trial-{i}", block.shape[2], start, 0.01)
            for i, (block, start) in enumerate(zip(blocks, (0.0, 0.0, 0.01, 0.0)))
        ),
        signal=InMemorySignalSource(blocks, sample_rate_hz=100.0),
        artifacts=artifacts,
    )
    source = DatasetViewSource(dataset)

    average = source.read_clean_average(slice(None), 0, 4)
    np.testing.assert_allclose(average[0], [1, 2, 2, 1])
    np.testing.assert_allclose(average[1], [2, 2, 2, 2])
    assert np.isnan(average[2]).all()

    source.select_segment(2)
    shifted = source.read_clean_average(slice(0, 1), 0, 3)
    np.testing.assert_allclose(shifted[0], [2, 2, 1])

    source.dataset = EegDataset(
        kind=DatasetKind.CONTINUOUS,
        channels=dataset.channels,
        segments=dataset.segments,
        signal=dataset.signal,
        artifacts=dataset.artifacts,
    )
    with pytest.raises(ValueError, match="segmented"):
        source.read_clean_average(slice(None), 0, 1)

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


def test_negative_timestamps_define_per_trial_channel_baselines() -> None:
    blocks = (
        np.array(
            [[[10, 12, 14, 16], [5, 7, 9, 11]]],
            dtype=np.float32,
        ),
        np.array(
            [[[20, 22, 26, 28], [15, 19, 23, 27]]],
            dtype=np.float32,
        ),
    )
    dataset = EegDataset(
        kind=DatasetKind.SEGMENTED,
        channels=(ChannelInfo(0, "C1"), ChannelInfo(1, "C2")),
        segments=tuple(
            SegmentInfo(index, f"trial-{index}", 4, -0.02, 0.01)
            for index in range(2)
        ),
        signal=InMemorySignalSource(blocks, sample_rate_hz=100.0),
    )
    source = DatasetViewSource(dataset)

    assert source.baseline_sample_count() == 2
    np.testing.assert_allclose(source.baseline_mean(), [11, 6])
    np.testing.assert_allclose(
        source.read_baseline_corrected(slice(None), 0, 4),
        [[-1, 1, 3, 5], [-1, 1, 3, 5]],
    )
    np.testing.assert_allclose(
        source.read_clean_average(slice(None), 0, 4),
        [[-1, 1, 4, 6], [-1.5, 1.5, 4.5, 7.5]],
    )


def test_nonnegative_timestamps_leave_values_unchanged() -> None:
    block = np.array([[[3, 4, 5]]], dtype=np.float32)
    dataset = EegDataset(
        kind=DatasetKind.SEGMENTED,
        channels=(ChannelInfo(0, "C1"),),
        segments=(SegmentInfo(0, "trial-0", 3, 0.0, 0.01),),
        signal=InMemorySignalSource((block,), sample_rate_hz=100.0),
    )
    source = DatasetViewSource(dataset)

    assert source.baseline_sample_count() == 0
    np.testing.assert_array_equal(
        source.read_baseline_corrected(slice(None), 0, 3),
        [[3, 4, 5]],
    )

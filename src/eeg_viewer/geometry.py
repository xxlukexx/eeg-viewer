"""Batched waveform geometry shared by scalp and grid placement modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import ReducedBlock
from .layout import ResolvedLayout


@dataclass(frozen=True)
class ChannelPlacement:
    channel_index: int
    label: str
    center_x: float
    center_y: float
    width: float
    height: float


def placements_from_layout(layout: ResolvedLayout) -> tuple[ChannelPlacement, ...]:
    count = len(layout.labels)
    if count == 0:
        return ()
    positions = layout.positions
    distances = []
    if count > 1:
        delta = positions[:, None, :] - positions[None, :, :]
        pairwise = np.sqrt(np.sum(delta * delta, axis=2))
        pairwise[pairwise == 0] = np.inf
        distances = np.min(pairwise, axis=1).tolist()
    nearest = float(np.median(distances)) if distances else 0.3
    width = min(0.24, max(0.02, nearest * 1.35))
    height = min(0.18, max(0.016, nearest * 0.82))
    # A spherical montage is irregular. Size the uniform tiles against every
    # pair rather than the median distance so dense temporal rows cannot overlap.
    scale_limit = 1.0
    for first in range(count):
        for second in range(first + 1, count):
            dx = abs(float(positions[first, 0] - positions[second, 0]))
            dy = abs(float(positions[first, 1] - positions[second, 1]))
            if dx < width and dy < height:
                scale_limit = min(scale_limit, max(dx / width, dy / height))
    if scale_limit < 1.0:
        scale = max(0.1, scale_limit * 0.92)
        width *= scale
        height *= scale
    return tuple(
        ChannelPlacement(
            channel_index=index,
            label=label,
            center_x=float(positions[index, 0]),
            center_y=float(positions[index, 1]),
            width=width,
            height=height,
        )
        for index, label in enumerate(layout.labels)
    )


def place_waveforms(
    reduced: ReducedBlock,
    placements: tuple[ChannelPlacement, ...],
    *,
    amplitude_spacing: float,
    sample_range: tuple[int, int] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Map reduced channel traces into small-multiple tiles as one line batch."""

    channels, points = reduced.values.shape
    if len(placements) != channels:
        raise ValueError("one placement is required for every reduced channel")
    x = np.full((channels, points + 1), np.nan, dtype=np.float32)
    y = np.full_like(x, np.nan)
    if points == 0:
        return x.ravel(), y.ravel()

    positions = reduced.sample_positions
    if sample_range is None:
        start = np.nanmin(positions, axis=1)
        stop = np.nanmax(positions, axis=1)
    else:
        start = np.full(channels, sample_range[0], dtype=np.float32)
        stop = np.full(channels, sample_range[1] - 1, dtype=np.float32)
    span = np.maximum(stop - start, 1.0)
    for row, placement in enumerate(placements):
        left = placement.center_x - placement.width / 2.0
        x[row, :points] = left + (
            (positions[row] - start[row]) / span[row] * placement.width
        )
        y[row, :points] = placement.center_y + (
            reduced.values[row] / np.float32(amplitude_spacing) * placement.height
        )
    return x.ravel(), y.ravel()

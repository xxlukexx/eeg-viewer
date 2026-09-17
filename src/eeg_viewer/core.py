"""Renderer-independent signal access, navigation, and display reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class ArraySignalSource:
    data: FloatArray
    sample_rate_hz: float
    channel_labels: tuple[str, ...]
    unit: str = "uV"

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must have shape channels x samples")
        if self.data.shape[0] != len(self.channel_labels):
            raise ValueError("channel label count does not match data")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")

    @property
    def channel_count(self) -> int:
        return self.data.shape[0]

    @property
    def sample_count(self) -> int:
        return self.data.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz

    def read(
        self,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray:
        start = max(0, min(int(start_sample), self.sample_count))
        stop = max(start, min(int(stop_sample), self.sample_count))
        return self.data[channels, start:stop]


@dataclass
class ViewState:
    sample_count: int
    sample_rate_hz: float
    channel_count: int
    start_sample: int = 0
    window_seconds: float = 10.0
    first_channel: int = 0
    visible_channels: int = 32
    amplitude_spacing_uv: float = 100.0

    @property
    def window_samples(self) -> int:
        return max(2, int(round(self.window_seconds * self.sample_rate_hz)))

    @property
    def stop_sample(self) -> int:
        return min(self.sample_count, self.start_sample + self.window_samples)

    @property
    def max_start_sample(self) -> int:
        return max(0, self.sample_count - self.window_samples)

    @property
    def max_first_channel(self) -> int:
        return max(0, self.channel_count - self.visible_channels)

    @property
    def channel_slice(self) -> slice:
        stop = min(self.channel_count, self.first_channel + self.visible_channels)
        return slice(self.first_channel, stop)

    def clamp(self) -> None:
        self.window_seconds = max(
            2.0 / self.sample_rate_hz,
            min(self.window_seconds, self.sample_count / self.sample_rate_hz),
        )
        self.visible_channels = max(1, min(self.visible_channels, self.channel_count))
        self.start_sample = max(0, min(int(self.start_sample), self.max_start_sample))
        self.first_channel = max(0, min(int(self.first_channel), self.max_first_channel))
        self.amplitude_spacing_uv = max(1e-12, float(self.amplitude_spacing_uv))

    def move_time_fraction(self, fraction: float) -> None:
        self.start_sample += int(round(fraction * self.window_samples))
        self.clamp()

    def zoom_time(self, factor: float, anchor_fraction: float = 0.5) -> None:
        if factor <= 0:
            raise ValueError("zoom factor must be positive")
        anchor_fraction = min(1.0, max(0.0, anchor_fraction))
        anchor_sample = self.start_sample + int(anchor_fraction * self.window_samples)
        self.window_seconds *= factor
        self.clamp()
        self.start_sample = anchor_sample - int(anchor_fraction * self.window_samples)
        self.clamp()

    def move_channels(self, amount: int) -> None:
        self.first_channel += int(amount)
        self.clamp()


@dataclass(frozen=True)
class ReducedBlock:
    sample_positions: FloatArray
    values: FloatArray
    samples_per_bin: int


def reduce_ordered_extrema(
    data: FloatArray,
    start_sample: int,
    max_time_bins: int,
) -> ReducedBlock:
    """Reduce samples while retaining min/max values and their temporal order."""

    if data.ndim != 2:
        raise ValueError("data must have shape channels x samples")
    if max_time_bins < 1:
        raise ValueError("max_time_bins must be positive")
    channels, samples = data.shape
    if samples == 0:
        empty = np.empty((channels, 0), dtype=np.float32)
        return ReducedBlock(empty, empty.copy(), 1)

    if samples <= max_time_bins:
        positions = np.arange(
            start_sample, start_sample + samples, dtype=np.float32
        )
        positions = np.broadcast_to(positions, data.shape).copy()
        return ReducedBlock(positions, np.asarray(data, dtype=np.float32), 1)

    samples_per_bin = int(np.ceil(samples / max_time_bins))
    full_bins, remainder = divmod(samples, samples_per_bin)

    def extrema(
        grouped: FloatArray,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
        # The ordinary benchmark contains only finite values. Avoid allocating
        # two channel x sample scratch arrays in that hot path; the explicit
        # non-finite path remains available for discontinuity stress tests.
        if np.isfinite(grouped).all():
            available = np.ones(grouped.shape[:2], dtype=bool)
            return (
                grouped.min(axis=2),
                grouped.max(axis=2),
                grouped.argmin(axis=2),
                grouped.argmax(axis=2),
                available,
            )
        finite = np.isfinite(grouped)
        available = finite.any(axis=2)
        for_min = np.where(finite, grouped, np.inf)
        for_max = np.where(finite, grouped, -np.inf)
        return (
            for_min.min(axis=2),
            for_max.max(axis=2),
            for_min.argmin(axis=2),
            for_max.argmax(axis=2),
            available,
        )

    if full_bins:
        grouped = data[:, : full_bins * samples_per_bin].reshape(
            channels, full_bins, samples_per_bin
        )
        minimum, maximum, minimum_index, maximum_index, any_finite = extrema(grouped)
    else:
        minimum = maximum = np.empty((channels, 0), dtype=np.float32)
        minimum_index = maximum_index = np.empty((channels, 0), dtype=np.intp)
        any_finite = np.empty((channels, 0), dtype=bool)

    if remainder:
        tail = data[:, full_bins * samples_per_bin :].reshape(channels, 1, remainder)
        tail_parts = extrema(tail)
        minimum = np.concatenate((minimum, tail_parts[0]), axis=1)
        maximum = np.concatenate((maximum, tail_parts[1]), axis=1)
        minimum_index = np.concatenate((minimum_index, tail_parts[2]), axis=1)
        maximum_index = np.concatenate((maximum_index, tail_parts[3]), axis=1)
        any_finite = np.concatenate((any_finite, tail_parts[4]), axis=1)

    bins = minimum.shape[1]

    lower_first = minimum_index <= maximum_index
    first_value = np.where(lower_first, minimum, maximum).astype(np.float32)
    second_value = np.where(lower_first, maximum, minimum).astype(np.float32)
    first_index = np.where(lower_first, minimum_index, maximum_index)
    second_index = np.where(lower_first, maximum_index, minimum_index)

    bin_start = np.arange(bins, dtype=np.int64)[None, :] * samples_per_bin
    first_position = start_sample + bin_start + first_index
    second_position = start_sample + bin_start + second_index

    values = np.empty((channels, bins * 2), dtype=np.float32)
    positions = np.empty_like(values)
    values[:, 0::2] = first_value
    values[:, 1::2] = second_value
    positions[:, 0::2] = first_position
    positions[:, 1::2] = second_position

    missing = ~any_finite
    values[:, 0::2][missing] = np.nan
    values[:, 1::2][missing] = np.nan
    return ReducedBlock(positions, values, samples_per_bin)


def stack_for_plot(
    reduced: ReducedBlock,
    sample_rate_hz: float,
    amplitude_spacing_uv: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Flatten channel traces with NaN separators for one batched plot item."""

    channels, points = reduced.values.shape
    offsets = np.arange(channels - 1, -1, -1, dtype=np.float32)
    scaled = reduced.values / np.float32(amplitude_spacing_uv)
    scaled += offsets[:, None]
    seconds = reduced.sample_positions / np.float32(sample_rate_hz)

    x = np.full((channels, points + 1), np.nan, dtype=np.float32)
    y = np.full_like(x, np.nan)
    x[:, :points] = seconds
    y[:, :points] = scaled
    return x.ravel(), y.ravel(), offsets


def fit_average_to_channel(
    average: FloatArray,
    *,
    half_height: float = 0.35,
) -> FloatArray:
    """Fit each mean symmetrically around zero within one channel.

    The result uses channel-spacing units rather than native signal units.
    Zero remains the channel centre so baseline-corrected polarity is preserved.
    Wholly missing channel means remain missing.
    """

    values = np.asarray(average, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("average must have shape channels x samples")
    fitted = np.full(values.shape, np.nan, dtype=np.float32)
    for row, channel in enumerate(values):
        finite = np.isfinite(channel)
        if not finite.any():
            continue
        maximum_absolute = float(np.abs(channel[finite]).max())
        if maximum_absolute == 0:
            fitted[row, finite] = 0.0
        else:
            fitted[row, finite] = (
                channel[finite] * (half_height / maximum_absolute)
            )
    return fitted

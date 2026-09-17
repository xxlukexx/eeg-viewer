"""Small internal data contract shared by importers, layouts, and renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


class SignalStore(Protocol):
    sample_rate_hz: float
    series_labels: tuple[str, ...]

    @property
    def segment_count(self) -> int: ...

    @property
    def series_count(self) -> int: ...

    @property
    def channel_count(self) -> int: ...

    def sample_count(self, segment_index: int) -> int: ...

    def read(
        self,
        segment_index: int,
        series_index: int,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray: ...


class DatasetKind(str, Enum):
    CONTINUOUS = "continuous"
    SEGMENTED = "segmented"
    EVOKED = "evoked"
    GRAND_AVERAGE = "grand_average"


@dataclass(frozen=True)
class ChannelInfo:
    index: int
    label: str
    channel_type: str = "eeg"
    unit: str = "unknown"


@dataclass(frozen=True)
class SegmentInfo:
    index: int
    segment_id: str
    sample_count: int
    start_time_seconds: float
    sample_period_seconds: float
    sample_info: tuple[int, int] | None = None
    trial_info: Any = None


@dataclass(frozen=True)
class MontageCandidate:
    labels: tuple[str, ...]
    positions: FloatArray
    source: str
    coordinate_system: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("montage positions must have shape channels x 2/3")
        if positions.shape[0] != len(self.labels):
            raise ValueError("montage labels and positions do not align")


@dataclass(frozen=True)
class ArtifactLayer:
    name: str
    mask: BoolArray
    source: str = "fieldtrip"

    def __post_init__(self) -> None:
        if np.asarray(self.mask).ndim != 2:
            raise ValueError("artifact masks must have shape channels x segments")


@dataclass(frozen=True)
class ArtifactState:
    layers: tuple[ArtifactLayer, ...] = ()
    interpolated: BoolArray | None = None
    cannot_interpolate: BoolArray | None = None

    def visual_states(self, segment_index: int, channel_count: int) -> tuple[ChannelVisualState, ...]:
        """Return renderer-neutral state; chart and tile views decide its appearance."""

        return tuple(
            ChannelVisualState(
                artifact_types=tuple(
                    layer.name
                    for layer in self.layers
                    if layer.mask[channel, segment_index]
                ),
                interpolated=bool(
                    self.interpolated is not None
                    and self.interpolated[channel, segment_index]
                ),
                cannot_interpolate=bool(
                    self.cannot_interpolate is not None
                    and self.cannot_interpolate[channel, segment_index]
                ),
            )
            for channel in range(channel_count)
        )

    def channel_types(self, segment_index: int) -> tuple[tuple[str, ...], ...]:
        if not self.layers:
            channel_count = 0
            for optional in (self.interpolated, self.cannot_interpolate):
                if optional is not None:
                    channel_count = optional.shape[0]
                    break
            return tuple(() for _ in range(channel_count))
        channel_count = self.layers[0].mask.shape[0]
        return tuple(
            tuple(
                layer.name
                for layer in self.layers
                if layer.mask[channel, segment_index]
            )
            for channel in range(channel_count)
        )


@dataclass(frozen=True)
class ChannelVisualState:
    """Imported display state, deliberately independent of drawing geometry."""

    artifact_types: tuple[str, ...] = ()
    interpolated: bool = False
    cannot_interpolate: bool = False


@dataclass(frozen=True)
class InMemorySignalSource:
    """Segmented in-memory data stored as series x channels x samples blocks."""

    blocks: tuple[FloatArray, ...]
    sample_rate_hz: float
    series_labels: tuple[str, ...] = ("signal",)

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("at least one signal block is required")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")
        first_shape = np.asarray(self.blocks[0]).shape
        if len(first_shape) != 3:
            raise ValueError("blocks must have shape series x channels x samples")
        series_count, channel_count = first_shape[:2]
        if len(self.series_labels) != series_count:
            raise ValueError("series labels do not align with data")
        for block in self.blocks:
            shape = np.asarray(block).shape
            if len(shape) != 3 or shape[:2] != (series_count, channel_count):
                raise ValueError("all blocks must share series and channel axes")

    @property
    def segment_count(self) -> int:
        return len(self.blocks)

    @property
    def series_count(self) -> int:
        return self.blocks[0].shape[0]

    @property
    def channel_count(self) -> int:
        return self.blocks[0].shape[1]

    def sample_count(self, segment_index: int) -> int:
        return int(self.blocks[segment_index].shape[2])

    def read(
        self,
        segment_index: int,
        series_index: int,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray:
        block = self.blocks[segment_index]
        start = max(0, min(int(start_sample), block.shape[2]))
        stop = max(start, min(int(stop_sample), block.shape[2]))
        return block[series_index, channels, start:stop]


@dataclass(frozen=True)
class EegDataset:
    kind: DatasetKind
    channels: tuple[ChannelInfo, ...]
    segments: tuple[SegmentInfo, ...]
    signal: SignalStore
    montage_candidate: MontageCandidate | None = None
    artifacts: ArtifactState = field(default_factory=ArtifactState)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.channels) != self.signal.channel_count:
            raise ValueError("channel metadata does not align with signal data")
        if len(self.segments) != self.signal.segment_count:
            raise ValueError("segment metadata does not align with signal data")
        for index, segment in enumerate(self.segments):
            if segment.sample_count != self.signal.sample_count(index):
                raise ValueError("segment sample count does not align with signal data")
        for layer in self.artifacts.layers:
            if layer.mask.shape != (len(self.channels), len(self.segments)):
                raise ValueError("artifact mask does not align with dataset")


class DatasetViewSource:
    """Select one segment/series and expose the viewer's windowed interface."""

    def __init__(
        self,
        dataset: EegDataset,
        segment_index: int = 0,
        series_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.segment_index = segment_index
        self.series_index = series_index
        self._baseline_cache: dict[tuple[int, int, int], FloatArray] = {}
        self._validate_selection()

    def _validate_selection(self) -> None:
        if not 0 <= self.segment_index < len(self.dataset.segments):
            raise IndexError("segment index out of bounds")
        if not 0 <= self.series_index < self.dataset.signal.series_count:
            raise IndexError("series index out of bounds")

    @property
    def channel_count(self) -> int:
        return len(self.dataset.channels)

    @property
    def channel_labels(self) -> tuple[str, ...]:
        return tuple(channel.label for channel in self.dataset.channels)

    @property
    def sample_rate_hz(self) -> float:
        return self.dataset.signal.sample_rate_hz

    @property
    def sample_count(self) -> int:
        return self.dataset.segments[self.segment_index].sample_count

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz

    @property
    def time_start_seconds(self) -> float:
        return self.dataset.segments[self.segment_index].start_time_seconds

    @property
    def unit(self) -> str:
        units = {channel.unit for channel in self.dataset.channels}
        if len(units) == 1:
            declared = units.pop()
            return "native units" if declared == "unknown" else f"native [{declared}]"
        return "native [mixed metadata]"

    @property
    def segment_count(self) -> int:
        return len(self.dataset.segments)

    @property
    def series_count(self) -> int:
        return self.dataset.signal.series_count

    def select_segment(self, index: int) -> None:
        self.segment_index = int(index)
        self._validate_selection()

    def select_series(self, index: int) -> None:
        self.series_index = int(index)
        self._validate_selection()

    def read(
        self,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray:
        return self.dataset.signal.read(
            self.segment_index,
            self.series_index,
            channels,
            start_sample,
            stop_sample,
        )

    def baseline_sample_count(self, segment_index: int | None = None) -> int:
        """Return the number of samples whose recorded timestamp is below zero."""

        index = self.segment_index if segment_index is None else int(segment_index)
        segment = self.dataset.segments[index]
        if segment.start_time_seconds >= 0.0:
            return 0
        exact_count = -segment.start_time_seconds * self.sample_rate_hz
        nearest_integer = round(exact_count)
        if np.isclose(exact_count, nearest_integer, rtol=0.0, atol=1e-7):
            exact_count = float(nearest_integer)
        return min(segment.sample_count, max(0, int(np.ceil(exact_count))))

    def baseline_mean(
        self,
        segment_index: int | None = None,
        series_index: int | None = None,
    ) -> FloatArray:
        """Return each channel's finite-sample mean over timestamps below zero."""

        segment = self.segment_index if segment_index is None else int(segment_index)
        series = self.series_index if series_index is None else int(series_index)
        key = (id(self.dataset), segment, series)
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached

        stop = self.baseline_sample_count(segment)
        if stop == 0:
            baseline = np.zeros(self.channel_count, dtype=np.float64)
        else:
            values = np.asarray(
                self.dataset.signal.read(segment, series, slice(None), 0, stop),
                dtype=np.float64,
            )
            finite = np.isfinite(values)
            counts = finite.sum(axis=1)
            sums = np.where(finite, values, 0.0).sum(axis=1)
            baseline = np.full(self.channel_count, np.nan, dtype=np.float64)
            np.divide(sums, counts, out=baseline, where=counts > 0)
        self._baseline_cache[key] = baseline
        return baseline

    def read_baseline_corrected(
        self,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray:
        """Read the selected trial after subtracting its negative-time mean."""

        values = np.asarray(self.read(channels, start_sample, stop_sample))
        if self.baseline_sample_count() == 0:
            return values
        selected = np.arange(self.channel_count)[channels]
        return values - self.baseline_mean()[selected, None]

    def read_clean_average(
        self,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> FloatArray:
        """Average unflagged trials on the selected trial's time axis.

        A channel contributes only when it is free of artifacts, interpolation,
        and cannot-interpolate flags. Missing samples and non-finite values do
        not contribute; samples with no clean observations remain NaN.
        """

        if self.dataset.kind != DatasetKind.SEGMENTED:
            raise ValueError("clean-trial averages require segmented data")
        selected = np.arange(self.channel_count)[channels]
        start = max(0, min(int(start_sample), self.sample_count))
        stop = max(start, min(int(stop_sample), self.sample_count))
        sums = np.zeros((len(selected), stop - start), dtype=np.float64)
        counts = np.zeros(sums.shape, dtype=np.int32)
        flagged = np.zeros((self.channel_count, self.segment_count), dtype=bool)
        for layer in self.dataset.artifacts.layers:
            flagged |= layer.mask
        for mask in (
            self.dataset.artifacts.interpolated,
            self.dataset.artifacts.cannot_interpolate,
        ):
            if mask is not None:
                flagged |= mask

        reference_start = self.time_start_seconds
        for index, segment in enumerate(self.dataset.segments):
            eligible_rows = np.flatnonzero(~flagged[selected, index])
            if not eligible_rows.size:
                continue
            # Trial samples may differ in both start time and duration. Match
            # samples by their recorded times rather than by array position.
            shift = round((reference_start - segment.start_time_seconds) * self.sample_rate_hz)
            overlap_start = max(start, -shift)
            overlap_stop = min(stop, segment.sample_count - shift)
            if overlap_start >= overlap_stop:
                continue
            values = np.asarray(
                self.dataset.signal.read(
                    index,
                    self.series_index,
                    selected[eligible_rows],
                    overlap_start + shift,
                    overlap_stop + shift,
                )
            )
            if self.baseline_sample_count(index) > 0:
                baseline = self.baseline_mean(index, self.series_index)
                values = values - baseline[selected[eligible_rows], None]
            finite = np.isfinite(values)
            destination = slice(overlap_start - start, overlap_stop - start)
            sums[eligible_rows, destination] += np.where(finite, values, 0.0)
            counts[eligible_rows, destination] += finite

        average = np.full(sums.shape, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=average, where=counts > 0)
        return average

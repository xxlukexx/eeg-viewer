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

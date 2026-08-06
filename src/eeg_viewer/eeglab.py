"""Lazy read-only EEGLAB adapter for continuous .set/.fdt recordings."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model import (
    ChannelInfo,
    DatasetKind,
    EegDataset,
    MontageCandidate,
    SegmentInfo,
)


class EeglabImportError(ValueError):
    """Raised when an EEGLAB source cannot be represented by the viewer."""


@dataclass
class MneRawSignalSource:
    """Windowed access to an MNE Raw object without preloading the recording."""

    raw: Any
    series_labels: tuple[str, ...] = ("signal",)
    cache_seconds: float = 60.0
    _cache_start: int = field(default=0, init=False, repr=False)
    _cache_data: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.raw.info["sfreq"])

    @property
    def segment_count(self) -> int:
        return 1

    @property
    def series_count(self) -> int:
        return 1

    @property
    def channel_count(self) -> int:
        return len(self.raw.ch_names)

    def sample_count(self, segment_index: int) -> int:
        if segment_index != 0:
            raise IndexError("continuous EEGLAB data has one segment")
        return int(self.raw.n_times)

    def read(
        self,
        segment_index: int,
        series_index: int,
        channels: Sequence[int] | slice,
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray:
        if segment_index != 0 or series_index != 0:
            raise IndexError("continuous EEGLAB data has one segment and one series")
        sample_count = self.sample_count(0)
        start = max(0, min(int(start_sample), sample_count))
        stop = max(start, min(int(stop_sample), sample_count))
        channel_indices = np.arange(self.channel_count)[channels].tolist()
        cache_stop = (
            self._cache_start + self._cache_data.shape[1]
            if self._cache_data is not None
            else 0
        )
        if self._cache_data is None or start < self._cache_start or stop > cache_stop:
            requested = max(1, stop - start)
            cache_samples = max(
                requested,
                int(round(self.cache_seconds * self.sample_rate_hz)),
            )
            margin = max(0, (cache_samples - requested) // 2)
            cache_start = max(0, start - margin)
            cache_end = min(sample_count, cache_start + cache_samples)
            cache_start = max(0, cache_end - cache_samples)
            self._cache_data = np.asarray(
                self.raw.get_data(
                    picks=list(range(self.channel_count)),
                    start=cache_start,
                    stop=cache_end,
                ),
                dtype=np.float32,
            )
            self._cache_start = cache_start
        local_start = start - self._cache_start
        local_stop = stop - self._cache_start
        return self._cache_data[channel_indices, local_start:local_stop]


def read_eeglab(path: str | Path) -> EegDataset:
    """Open an EEGLAB continuous recording lazily through MNE."""

    import mne

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.casefold() == ".fdt":
        paired_set = source_path.with_suffix(".set")
        if not paired_set.exists():
            raise EeglabImportError(
                f"{source_path.name} requires the paired {paired_set.name} file"
            )
        source_path = paired_set
    if source_path.suffix.casefold() != ".set":
        raise EeglabImportError("EEGLAB input must be a .set file or its paired .fdt")

    try:
        raw = mne.io.read_raw_eeglab(source_path, preload=False, verbose="ERROR")
    except Exception as error:
        raise EeglabImportError(f"Could not open EEGLAB recording: {error}") from error
    if raw.n_times < 1 or not raw.ch_names:
        raise EeglabImportError("EEGLAB recording contains no waveform data")

    channel_types = raw.get_channel_types()
    channels = tuple(
        ChannelInfo(
            index=index,
            label=label,
            channel_type=channel_types[index],
            unit="V" if channel_types[index] in {"eeg", "eog", "ecg", "emg"} else "unknown",
        )
        for index, label in enumerate(raw.ch_names)
    )
    montage_candidate = _montage_candidate(raw)
    signal = MneRawSignalSource(raw)
    segment = SegmentInfo(
        index=0,
        segment_id="continuous",
        sample_count=signal.sample_count(0),
        start_time_seconds=float(raw.first_time),
        sample_period_seconds=1.0 / signal.sample_rate_hz,
    )
    return EegDataset(
        kind=DatasetKind.CONTINUOUS,
        channels=channels,
        segments=(segment,),
        signal=signal,
        montage_candidate=montage_candidate,
        metadata={
            "adapter": "eeglab",
            "source_path": str(source_path),
            "annotation_count": len(raw.annotations),
            "preloaded": bool(raw.preload),
        },
    )


def _montage_candidate(raw: Any) -> MontageCandidate | None:
    montage = raw.get_montage()
    if montage is None:
        return None
    positions = montage.get_positions().get("ch_pos", {})
    labels = tuple(
        label
        for label in raw.ch_names
        if label in positions and np.isfinite(positions[label]).all()
    )
    if not labels:
        return None
    return MontageCandidate(
        labels=labels,
        positions=np.asarray([positions[label] for label in labels], dtype=float),
        source="eeglab.chanlocs",
        coordinate_system="head",
        unit="m",
    )

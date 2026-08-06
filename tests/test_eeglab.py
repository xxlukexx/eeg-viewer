from __future__ import annotations

from pathlib import Path

import numpy as np

from eegvis_benchmark.eeglab import MneRawSignalSource, read_eeglab
from eegvis_benchmark.model import DatasetKind, DatasetViewSource


class FakeRaw:
    def __init__(self) -> None:
        self.info = {"sfreq": 250.0}
        self.ch_names = ["Fp1", "Cz", "O2"]
        self.n_times = 1_000
        self.first_time = 0.0
        self.preload = False
        self.annotations = [object(), object()]
        self._data = np.arange(3_000, dtype=float).reshape(3, 1_000) * 1e-6
        self.read_calls = 0

    def get_channel_types(self):
        return ["eeg", "eeg", "eeg"]

    def get_montage(self):
        return None

    def get_data(self, *, picks, start, stop):
        self.read_calls += 1
        return self._data[picks, start:stop]


def test_mne_raw_store_reads_only_the_requested_window() -> None:
    raw = FakeRaw()
    source = MneRawSignalSource(raw)

    result = source.read(0, 0, slice(1, 3), 100, 125)

    assert result.shape == (2, 25)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, raw._data[1:3, 100:125])
    second = source.read(0, 0, slice(0, 1), 130, 150)
    np.testing.assert_allclose(second, raw._data[0:1, 130:150])
    assert raw.read_calls == 1


def test_eeglab_adapter_keeps_continuous_data_lazy(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "recording.set"
    path.write_bytes(b"fake test placeholder")
    raw = FakeRaw()
    monkeypatch.setattr("mne.io.read_raw_eeglab", lambda *args, **kwargs: raw)

    dataset = read_eeglab(path)
    source = DatasetViewSource(dataset)

    assert dataset.kind == DatasetKind.CONTINUOUS
    assert dataset.metadata["preloaded"] is False
    assert dataset.metadata["annotation_count"] == 2
    assert source.duration_seconds == 4.0
    assert source.read(slice(None), 10, 20).shape == (3, 10)

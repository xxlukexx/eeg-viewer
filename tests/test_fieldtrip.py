from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from eegvis_benchmark.fieldtrip import fieldtrip_from_mapping, read_fieldtrip
from eegvis_benchmark.model import DatasetKind, DatasetViewSource


def _segmented_mapping() -> dict[str, object]:
    return {
        "data": {
            "label": ["Fp1", "Cz", "EOG"],
            "fsample": 250.0,
            "trial": [
                np.arange(30, dtype=np.float64).reshape(3, 10),
                np.arange(36, dtype=np.float64).reshape(3, 12),
            ],
            "time": [np.arange(10) / 250.0 - 0.1, np.arange(12) / 250.0 - 0.1],
            "sampleinfo": [[101, 110], [201, 212]],
            "trialinfo": [[1, 7], [2, 9]],
            "elec": {
                "label": ["Fp1", "Cz", "EOG"],
                "chanpos": [[-0.3, 0.8, 0.1], [0, 0, 0.9], [0.8, 0.4, 0]],
                "chanunit": ["uV", "uV", "uV"],
                "chantype": ["eeg", "eeg", "eog"],
                "coordsys": "ctf",
                "unit": "cm",
            },
            "art": np.array(
                [
                    [[1, 0], [0, 1]],
                    [[0, 0], [1, 0]],
                    [[0, 1], [0, 0]],
                ],
                dtype=np.uint8,
            ),
            "art_type": ["blink", "range"],
            "interp": [[0, 1], [0, 0], [1, 0]],
            "cantInterp": [[0, 0], [0, 1], [0, 0]],
        }
    }


def test_segmented_fieldtrip_preserves_metadata_layout_and_artifacts() -> None:
    dataset = fieldtrip_from_mapping(_segmented_mapping())

    assert dataset.kind == DatasetKind.SEGMENTED
    assert [block.shape for block in dataset.signal.blocks] == [(1, 3, 10), (1, 3, 12)]
    assert dataset.signal.blocks[0].dtype == np.float32
    assert dataset.signal.sample_rate_hz == 250.0
    assert dataset.segments[0].start_time_seconds == -0.1
    assert dataset.segments[1].sample_info == (201, 212)
    assert dataset.channels[2].channel_type == "eog"
    assert dataset.montage_candidate is not None
    assert dataset.montage_candidate.source == "fieldtrip.elec.chanpos"
    assert [layer.name for layer in dataset.artifacts.layers] == ["blink", "range"]
    states = dataset.artifacts.visual_states(1, 3)
    assert states[0].artifact_types == ("range",)
    assert states[0].interpolated
    assert states[1].cannot_interpolate


def test_continuous_timelock_and_grand_average_have_distinct_kinds() -> None:
    raw = fieldtrip_from_mapping(
        {"label": ["Cz", "Pz"], "trial": np.zeros((2, 50)), "fsample": 100}
    )
    evoked = fieldtrip_from_mapping(
        {"label": ["Cz", "Pz"], "avg": np.zeros((2, 20)), "time": np.arange(20) / 200}
    )
    grand = fieldtrip_from_mapping(
        {
            "label": ["Cz", "Pz"],
            "individual": np.zeros((4, 2, 20)),
            "time": np.arange(20) / 200,
            "subj": ["s1", "s2", "s3", "s4"],
        }
    )

    assert raw.kind == DatasetKind.CONTINUOUS
    assert evoked.kind == DatasetKind.EVOKED
    assert grand.kind == DatasetKind.GRAND_AVERAGE
    assert grand.signal.series_count == 4
    assert grand.signal.series_labels[-1] == "s4"


def test_multi_condition_average_container_becomes_display_series() -> None:
    common = {
        "label": ["Cz", "Pz"],
        "time": np.arange(20) / 200.0,
        "fsample": 200.0,
    }
    dataset = fieldtrip_from_mapping(
        {
            "erps": {
                "face_up": {**common, "avg": np.zeros((2, 20))},
                "face_inv": {**common, "avg": np.ones((2, 20))},
                "summary": {"trials": 20},
            }
        }
    )

    assert dataset.kind == DatasetKind.EVOKED
    assert dataset.signal.series_labels == ("face_up", "face_inv")
    assert dataset.signal.blocks[0].shape == (2, 2, 20)
    np.testing.assert_array_equal(dataset.signal.blocks[0][1], 1.0)


def test_grand_average_exposes_mean_and_named_subject_series() -> None:
    individual = np.stack((np.zeros((2, 20)), np.full((2, 20), 2.0)))
    dataset = fieldtrip_from_mapping(
        {
            "label": ["Cz", "Pz"],
            "individual": individual,
            "avg": individual.mean(axis=0),
            "time": np.arange(20) / 200,
            "subj": ["s1", "s2"],
        }
    )

    assert dataset.kind == DatasetKind.GRAND_AVERAGE
    assert dataset.signal.series_labels == ("grand average", "s1", "s2")
    np.testing.assert_array_equal(dataset.signal.blocks[0][0], 1.0)


def test_dataset_view_switches_variable_length_segments_without_copying() -> None:
    dataset = fieldtrip_from_mapping(_segmented_mapping())
    source = DatasetViewSource(dataset)

    source.select_segment(1)
    result = source.read(slice(0, 2), 2, 8)

    assert source.sample_count == 12
    assert source.time_start_seconds == -0.1
    assert source.unit == "native [uV]"
    assert result.shape == (2, 6)
    assert np.shares_memory(result, dataset.signal.blocks[1])


def test_reads_a_conventional_mat_file(tmp_path: Path) -> None:
    path = tmp_path / "fieldtrip.mat"
    savemat(
        path,
        {
            "ft_data": {
                "label": np.array(["Cz", "Pz"], dtype=object),
                "trial": np.arange(40, dtype=np.float64).reshape(2, 20),
                "time": np.arange(20, dtype=float) / 100.0,
                "fsample": 100.0,
            }
        },
    )

    dataset = read_fieldtrip(path)

    assert dataset.metadata["matlab_variable"] == "ft_data"
    assert dataset.metadata["source_path"] == str(path.resolve())
    assert dataset.signal.blocks[0].shape == (1, 2, 20)


def test_falls_back_to_scipy_when_primary_reader_rejects_nested_metadata(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "fallback.mat"
    savemat(
        path,
        {
            "data": {
                "label": np.array(["Cz", "Pz"], dtype=object),
                "trial": np.zeros((2, 20)),
                "fsample": 100.0,
                "cfg": {"unexpected": np.array([[1, 2], [3, 4]])},
            }
        },
    )

    def reject(_: str):
        raise ValueError("simulated nested-structure parser failure")

    monkeypatch.setattr("pymatreader.read_mat", reject)
    dataset = read_fieldtrip(path)

    assert dataset.metadata["mat_decoder"] == "scipy-fallback"
    assert dataset.metadata["primary_decoder_error"] == "ValueError"

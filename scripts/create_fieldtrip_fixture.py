"""Create a realistic segmented FieldTrip MAT file for interactive smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat

from eegvis_benchmark.synthetic import SyntheticConfig, generate_recording


LABELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
    "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "PO7", "PO3", "PO4", "PO8", "O1", "Oz", "O2",
)


def build_fixture(output: Path, *, trials: int = 24, seed: int = 71) -> Path:
    sample_rate = 250.0
    samples_per_trial = 750
    recording = generate_recording(
        SyntheticConfig(
            channels=len(LABELS),
            sample_rate_hz=sample_rate,
            duration_seconds=trials * samples_per_trial / sample_rate,
            seed=seed,
        )
    )
    trial_cells = np.empty((1, trials), dtype=object)
    time_cells = np.empty((1, trials), dtype=object)
    time = np.arange(samples_per_trial, dtype=float) / sample_rate - 0.2
    for trial in range(trials):
        start = trial * samples_per_trial
        trial_cells[0, trial] = recording.data_uv[:, start : start + samples_per_trial]
        time_cells[0, trial] = time

    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    channel_positions = np.asarray([positions[label] for label in LABELS])
    artifact = np.zeros((len(LABELS), trials, 3), dtype=np.uint8)
    artifact[:7, [3, 13, 20], 0] = 1  # blink-like frontal flags
    artifact[[9, 18, 28], [6, 17, 22], 1] = 1  # range flags
    artifact[25:, [10, 19], 2] = 1  # high-frequency/posterior flags
    interpolated = np.zeros((len(LABELS), trials), dtype=np.uint8)
    interpolated[9, 6] = 1
    cannot_interpolate = np.zeros_like(interpolated)
    cannot_interpolate[28, 22] = 1

    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output,
        {
            "cleaned": {
                "label": np.asarray(LABELS, dtype=object),
                "fsample": sample_rate,
                "trial": trial_cells,
                "time": time_cells,
                "sampleinfo": np.column_stack(
                    (
                        np.arange(trials) * samples_per_trial + 1,
                        np.arange(1, trials + 1) * samples_per_trial,
                    )
                ),
                "trialinfo": np.column_stack((np.arange(1, trials + 1), np.arange(trials) % 2 + 1)),
                "elec": {
                    "label": np.asarray(LABELS, dtype=object),
                    "chanpos": channel_positions,
                    "elecpos": channel_positions,
                    "chantype": np.asarray(["eeg"] * len(LABELS), dtype=object),
                    "chanunit": np.asarray(["uV"] * len(LABELS), dtype=object),
                    "coordsys": "standard",
                    "unit": "m",
                },
                "art": artifact,
                "art_type": np.asarray(["blink", "range", "high-frequency"], dtype=object),
                "interp": interpolated,
                "cantInterp": cannot_interpolate,
            }
        },
        do_compression=True,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    path = build_fixture(args.output.resolve(), trials=args.trials)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

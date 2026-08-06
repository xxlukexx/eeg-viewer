"""Build a local FieldTrip grand-average fixture from subject average files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

from eegvis_benchmark.fieldtrip import read_fieldtrip


def build_grand_average(
    source_directory: Path,
    output: Path,
    *,
    variable: str,
    subject_count: int,
) -> Path:
    selected: list[tuple[str, object]] = []
    reference_labels: tuple[str, ...] | None = None
    reference_samples: int | None = None
    for path in sorted(source_directory.glob("*.clean.average.mat")):
        print(f"Inspecting {path.name}", flush=True)
        try:
            dataset = read_fieldtrip(path, variable=variable)
        except Exception as error:
            print(f"  skipped: {error}", flush=True)
            continue
        labels = tuple(channel.label for channel in dataset.channels)
        sample_count = dataset.signal.sample_count(0)
        if reference_labels is None:
            reference_labels = labels
            reference_samples = sample_count
        if labels != reference_labels or sample_count != reference_samples:
            print("  skipped: channels or samples differ from the selected group", flush=True)
            continue
        selected.append((path.stem.split(".")[0], dataset))
        print(f"  selected {len(selected)}/{subject_count}", flush=True)
        if len(selected) >= subject_count:
            break
    if len(selected) < 2:
        raise RuntimeError("Fewer than two compatible subject averages were found")

    first = selected[0][1]
    individual = np.stack(
        [dataset.signal.read(0, 0, slice(None), 0, dataset.signal.sample_count(0)) for _, dataset in selected],
        axis=0,
    )
    segment = first.segments[0]
    time = segment.start_time_seconds + np.arange(segment.sample_count) / first.signal.sample_rate_hz
    structure: dict[str, object] = {
        "label": np.asarray(reference_labels, dtype=object),
        "fsample": first.signal.sample_rate_hz,
        "time": time,
        "avg": individual.mean(axis=0),
        "individual": individual,
        "subj": np.asarray([subject for subject, _ in selected], dtype=object),
        "dimord": "subj_chan_time",
    }
    if first.montage_candidate is not None:
        structure["elec"] = {
            "label": np.asarray(first.montage_candidate.labels, dtype=object),
            "chanpos": first.montage_candidate.positions,
            "elecpos": first.montage_candidate.positions,
            "coordsys": first.montage_candidate.coordinate_system or "unknown",
            "unit": first.montage_candidate.unit or "unknown",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, {"grand_average": structure}, do_compression=True)
    print(
        f"Wrote {output} ({individual.shape[0]} subjects x "
        f"{individual.shape[1]} channels x {individual.shape[2]} samples)",
        flush=True,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variable", default="erps.face_up")
    parser.add_argument("--subjects", type=int, default=5)
    args = parser.parse_args()
    build_grand_average(
        args.source_directory,
        args.output.resolve(),
        variable=args.variable,
        subject_count=args.subjects,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

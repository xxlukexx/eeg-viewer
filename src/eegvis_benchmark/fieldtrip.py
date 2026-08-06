"""Read common FieldTrip MATLAB structures into the viewer data contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    ArtifactLayer,
    ArtifactState,
    ChannelInfo,
    DatasetKind,
    EegDataset,
    InMemorySignalSource,
    MontageCandidate,
    SegmentInfo,
)


class FieldTripImportError(ValueError):
    """Raised when a MATLAB variable is not a supported FieldTrip data type."""


def read_fieldtrip(path: str | Path, variable: str | None = None) -> EegDataset:
    """Read conventional or MATLAB 7.3 FieldTrip data without requiring MATLAB."""

    from pymatreader import read_mat

    source_path = Path(path).expanduser().resolve()
    decoder = "pymatreader"
    primary_error: Exception | None = None
    try:
        contents = read_mat(str(source_path))
    except Exception as error:  # third-party readers expose several parser errors
        primary_error = error
        try:
            from scipy.io import loadmat

            contents = loadmat(source_path, simplify_cells=True)
            decoder = "scipy-fallback"
        except Exception as fallback_error:
            raise FieldTripImportError(
                "The MAT file could not be decoded by pymatreader or SciPy "
                f"({type(error).__name__}; {type(fallback_error).__name__})"
            ) from error
    dataset = fieldtrip_from_mapping(
        contents,
        variable=variable,
        source_path=source_path,
    )
    dataset.metadata["mat_decoder"] = decoder
    if primary_error is not None:
        dataset.metadata["primary_decoder_error"] = type(primary_error).__name__
    return dataset


def fieldtrip_from_mapping(
    contents: Mapping[str, Any],
    *,
    variable: str | None = None,
    source_path: Path | None = None,
) -> EegDataset:
    """Convert an already decoded MATLAB mapping (also useful for tests)."""

    if variable is None:
        candidates = _find_candidates(contents)
        if len(candidates) > 1 and all("avg" in candidate for _, candidate in candidates):
            return _average_collection(candidates, source_path)
    name, data = _select_variable(contents, variable)
    labels = _text_tuple(data.get("label"))
    if not labels:
        raise FieldTripImportError(f"FieldTrip variable {name!r} has no channel labels")

    signal_blocks, kind, series_labels = _extract_signal(data, len(labels))
    sample_rate = _sample_rate(data, signal_blocks)
    time_axes = _time_axes(data.get("time"), signal_blocks, sample_rate)
    segments = _segments(data, signal_blocks, time_axes, sample_rate)
    units, channel_types = _channel_metadata(data, labels)
    channels = tuple(
        ChannelInfo(index=i, label=label, channel_type=channel_types[i], unit=units[i])
        for i, label in enumerate(labels)
    )
    montage = _embedded_montage(data)
    artifacts = _artifact_state(data, len(labels), len(signal_blocks))

    original_dtypes = tuple(str(np.asarray(block).dtype) for block in signal_blocks)
    blocks = tuple(np.asarray(block, dtype=np.float32) for block in signal_blocks)
    metadata: dict[str, Any] = {
        "adapter": "fieldtrip",
        "matlab_variable": name,
        "original_dtypes": original_dtypes,
    }
    if source_path is not None:
        metadata["source_path"] = str(source_path)
    if "dimord" in data:
        metadata["fieldtrip_dimord"] = str(data["dimord"])

    return EegDataset(
        kind=kind,
        channels=channels,
        segments=segments,
        signal=InMemorySignalSource(
            blocks=blocks,
            sample_rate_hz=sample_rate,
            series_labels=series_labels,
        ),
        montage_candidate=montage,
        artifacts=artifacts,
        metadata=metadata,
    )


def _select_variable(
    contents: Mapping[str, Any], variable: str | None
) -> tuple[str, Mapping[str, Any]]:
    if variable is not None:
        candidate: Any = contents
        for part in variable.split("."):
            if not isinstance(candidate, Mapping) or part not in candidate:
                raise FieldTripImportError(f"MATLAB variable {variable!r} was not found")
            candidate = candidate[part]
        if not isinstance(candidate, Mapping):
            raise FieldTripImportError(f"MATLAB variable {variable!r} is not a structure")
        return variable, candidate

    candidates = _find_candidates(contents)
    if not candidates:
        raise FieldTripImportError(
            "No supported FieldTrip raw, timelock, or grand-average variable was found"
        )
    if len(candidates) > 1:
        names = ", ".join(name for name, _ in candidates)
        raise FieldTripImportError(
            f"Several FieldTrip variables were found ({names}); choose one explicitly"
        )
    return candidates[0]


def _find_candidates(contents: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if _looks_like_fieldtrip(contents):
        return [("<root>", contents)]
    direct = [
        (str(key), value)
        for key, value in contents.items()
        if not str(key).startswith("__")
        and isinstance(value, Mapping)
        and _looks_like_fieldtrip(value)
    ]
    if direct:
        return direct
    nested: list[tuple[str, Mapping[str, Any]]] = []
    ignored = {"cfg", "hdr", "summary", "previous"}
    for outer_name, outer in contents.items():
        if str(outer_name).startswith("__") or str(outer_name).casefold() in ignored:
            continue
        if not isinstance(outer, Mapping):
            continue
        for inner_name, value in outer.items():
            if isinstance(value, Mapping) and _looks_like_fieldtrip(value):
                nested.append((f"{outer_name}.{inner_name}", value))
    return nested


def _average_collection(
    candidates: list[tuple[str, Mapping[str, Any]]],
    source_path: Path | None,
) -> EegDataset:
    datasets = [fieldtrip_from_mapping(candidate) for _, candidate in candidates]
    first = datasets[0]
    expected_labels = tuple(channel.label for channel in first.channels)
    expected_shape = first.signal.blocks[0].shape[1:]
    for dataset in datasets[1:]:
        labels = tuple(channel.label for channel in dataset.channels)
        if labels != expected_labels or dataset.signal.blocks[0].shape[1:] != expected_shape:
            names = ", ".join(name for name, _ in candidates)
            raise FieldTripImportError(
                f"Average variables {names} do not share channels and samples"
            )
        if not np.isclose(dataset.signal.sample_rate_hz, first.signal.sample_rate_hz):
            raise FieldTripImportError("Average variables do not share a sample rate")
    block = np.concatenate([dataset.signal.blocks[0] for dataset in datasets], axis=0)
    metadata: dict[str, Any] = {
        "adapter": "fieldtrip",
        "matlab_variables": tuple(name for name, _ in candidates),
        "average_collection": True,
    }
    if source_path is not None:
        metadata["source_path"] = str(source_path)
    return EegDataset(
        kind=DatasetKind.EVOKED,
        channels=first.channels,
        segments=first.segments,
        signal=InMemorySignalSource(
            blocks=(block,),
            sample_rate_hz=first.signal.sample_rate_hz,
            series_labels=tuple(name.rsplit(".", 1)[-1] for name, _ in candidates),
        ),
        montage_candidate=next(
            (dataset.montage_candidate for dataset in datasets if dataset.montage_candidate),
            None,
        ),
        metadata=metadata,
    )


def _looks_like_fieldtrip(value: Mapping[str, Any]) -> bool:
    return "label" in value and any(key in value for key in ("trial", "avg", "individual"))


def _extract_signal(
    data: Mapping[str, Any], channel_count: int
) -> tuple[list[np.ndarray], DatasetKind, tuple[str, ...]]:
    if "trial" in data:
        trials = _matrix_list(data["trial"], channel_count)
        if not trials:
            raise FieldTripImportError("FieldTrip trial data is empty")
        blocks = [matrix[None, :, :] for matrix in trials]
        kind = DatasetKind.CONTINUOUS if len(blocks) == 1 else DatasetKind.SEGMENTED
        return blocks, kind, ("signal",)

    if "individual" in data:
        individual = np.asarray(data["individual"])
        individual = np.squeeze(individual)
        if individual.ndim != 3:
            raise FieldTripImportError("FieldTrip individual must be subjects x channels x samples")
        channel_axes = [axis for axis, size in enumerate(individual.shape) if size == channel_count]
        if not channel_axes:
            raise FieldTripImportError("No axis of individual matches the channel labels")
        channel_axis = 1 if individual.shape[1] == channel_count else channel_axes[0]
        individual = np.moveaxis(individual, channel_axis, 1)
        series_labels = _text_tuple(data.get("subj"))
        if len(series_labels) != individual.shape[0]:
            series_labels = tuple(f"subject {i + 1}" for i in range(individual.shape[0]))
        if "avg" in data:
            average = _channel_matrix(data["avg"], channel_count, "avg")
            if average.shape[1] != individual.shape[2]:
                raise FieldTripImportError("Grand-average avg and individual samples do not align")
            individual = np.concatenate((average[None, :, :], individual), axis=0)
            series_labels = ("grand average", *series_labels)
        return [individual], DatasetKind.GRAND_AVERAGE, series_labels

    if "avg" in data:
        average = _channel_matrix(data["avg"], channel_count, "avg")
        return [average[None, :, :]], DatasetKind.EVOKED, ("average",)

    if any(key in data for key in ("powspctrm", "fourierspctrm", "pos")):
        raise FieldTripImportError("FieldTrip frequency and source structures are not waveform data")
    raise FieldTripImportError("Unsupported FieldTrip structure")


def _matrix_list(value: Any, channel_count: int) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        return [_channel_matrix(item, channel_count, "trial") for item in value]
    array = np.asarray(value)
    if array.dtype == object:
        return [_channel_matrix(item, channel_count, "trial") for item in array.ravel()]
    array = np.squeeze(array)
    if array.ndim == 2:
        return [_channel_matrix(array, channel_count, "trial")]
    if array.ndim == 3:
        # Some readers collapse equal-sized FieldTrip cell trials into one array.
        if array.shape[1] == channel_count:
            return [np.asarray(item) for item in array]
        if array.shape[0] == channel_count:
            moved = np.moveaxis(array, 1, 0)
            return [np.asarray(item) for item in moved]
    raise FieldTripImportError("FieldTrip trial must contain channel x sample matrices")


def _channel_matrix(value: Any, channel_count: int, field: str) -> np.ndarray:
    matrix = np.asarray(value)
    matrix = np.squeeze(matrix)
    if matrix.ndim != 2:
        raise FieldTripImportError(f"FieldTrip {field} data must be two-dimensional")
    if matrix.shape[0] == channel_count:
        return matrix
    if matrix.shape[1] == channel_count:
        return matrix.T
    raise FieldTripImportError(f"Neither axis of {field} matches the channel labels")


def _sample_rate(data: Mapping[str, Any], blocks: Sequence[np.ndarray]) -> float:
    explicit = data.get("fsample")
    if explicit is not None:
        values = np.asarray(explicit, dtype=float).ravel()
        if values.size and np.isfinite(values[0]) and values[0] > 0:
            return float(values[0])
    raw_time = data.get("time")
    time_arrays = _array_list(raw_time)
    for axis in time_arrays:
        values = np.asarray(axis, dtype=float).ravel()
        if values.size >= 2:
            differences = np.diff(values)
            period = float(np.median(differences))
            if period > 0 and np.allclose(differences, period, rtol=1e-5, atol=1e-10):
                return 1.0 / period
    raise FieldTripImportError("Sample rate is absent and cannot be inferred from time")


def _time_axes(
    raw_time: Any,
    blocks: Sequence[np.ndarray],
    sample_rate: float,
) -> list[np.ndarray]:
    axes = _array_list(raw_time)
    if not axes:
        return [np.arange(block.shape[2], dtype=float) / sample_rate for block in blocks]
    if len(axes) == 1 and len(blocks) > 1:
        axes = axes * len(blocks)
    if len(axes) != len(blocks):
        raise FieldTripImportError("FieldTrip time and trial counts do not align")
    result: list[np.ndarray] = []
    for axis, block in zip(axes, blocks, strict=True):
        values = np.asarray(axis, dtype=float).ravel()
        if values.size != block.shape[2]:
            raise FieldTripImportError("FieldTrip time and signal sample counts do not align")
        result.append(values)
    return result


def _segments(
    data: Mapping[str, Any],
    blocks: Sequence[np.ndarray],
    time_axes: Sequence[np.ndarray],
    sample_rate: float,
) -> tuple[SegmentInfo, ...]:
    sample_info = np.asarray(data.get("sampleinfo", []))
    if sample_info.size:
        sample_info = np.atleast_2d(sample_info)
    trial_info = data.get("trialinfo")
    trial_rows = np.asarray(trial_info) if trial_info is not None else None
    if trial_rows is not None and trial_rows.ndim == 1 and len(blocks) > 1:
        trial_rows = trial_rows[:, None]
    segments: list[SegmentInfo] = []
    for index, (block, axis) in enumerate(zip(blocks, time_axes, strict=True)):
        bounds = None
        if sample_info.size and index < sample_info.shape[0] and sample_info.shape[1] >= 2:
            bounds = (int(sample_info[index, 0]), int(sample_info[index, 1]))
        info: Any = None
        if trial_rows is not None:
            if trial_rows.ndim >= 1 and index < trial_rows.shape[0]:
                row = trial_rows[index]
                info = row.item() if np.asarray(row).ndim == 0 else np.asarray(row).tolist()
        segments.append(
            SegmentInfo(
                index=index,
                segment_id=f"segment-{index + 1}",
                sample_count=block.shape[2],
                start_time_seconds=float(axis[0]) if axis.size else 0.0,
                sample_period_seconds=1.0 / sample_rate,
                sample_info=bounds,
                trial_info=info,
            )
        )
    return tuple(segments)


def _channel_metadata(
    data: Mapping[str, Any], labels: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sources = [candidate for candidate in (data.get("elec"), data.get("hdr")) if isinstance(candidate, Mapping)]
    units = _aligned_metadata(sources, labels, "chanunit", "unknown")
    types = _aligned_metadata(sources, labels, "chantype", "eeg")
    return units, types


def _aligned_metadata(
    sources: Sequence[Mapping[str, Any]],
    labels: tuple[str, ...],
    field: str,
    default: str,
) -> tuple[str, ...]:
    for source in sources:
        values = _text_tuple(source.get(field))
        if not values:
            continue
        source_labels = _text_tuple(source.get("label"))
        if source_labels and len(values) == len(source_labels):
            lookup = {label.casefold(): value for label, value in zip(source_labels, values, strict=True)}
            return tuple(lookup.get(label.casefold(), default) for label in labels)
        if len(values) == len(labels):
            return values
    return tuple(default for _ in labels)


def _embedded_montage(data: Mapping[str, Any]) -> MontageCandidate | None:
    elec = data.get("elec")
    if not isinstance(elec, Mapping):
        return None
    labels = _text_tuple(elec.get("label"))
    for field in ("chanpos", "elecpos"):
        raw = elec.get(field)
        if raw is None:
            continue
        positions = np.asarray(raw, dtype=float)
        positions = np.squeeze(positions)
        if positions.ndim == 2 and positions.shape[0] != len(labels) and positions.shape[1] == len(labels):
            positions = positions.T
        if labels and positions.ndim == 2 and positions.shape[0] == len(labels) and positions.shape[1] in (2, 3):
            return MontageCandidate(
                labels=labels,
                positions=positions,
                source=f"fieldtrip.elec.{field}",
                coordinate_system=_optional_text(elec.get("coordsys")),
                unit=_optional_text(elec.get("unit")),
            )
    return None


def _artifact_state(
    data: Mapping[str, Any], channel_count: int, segment_count: int
) -> ArtifactState:
    art_container = data.get("art")
    art_value: Any = art_container
    names_value: Any = data.get("art_type")
    if isinstance(art_container, Mapping):
        art_value = art_container.get("art", art_container.get("matrix", art_container.get("mask")))
        names_value = art_container.get("art_type", art_container.get("type", names_value))

    layers: tuple[ArtifactLayer, ...] = ()
    if art_value is not None:
        matrix = _orient_artifacts(art_value, channel_count, segment_count)
        names = _text_tuple(names_value)
        if len(names) != matrix.shape[2]:
            names = tuple(f"artifact {i + 1}" for i in range(matrix.shape[2]))
        layers = tuple(
            ArtifactLayer(name=name, mask=matrix[:, :, index].astype(bool, copy=False))
            for index, name in enumerate(names)
        )

    interpolated = _channel_segment_mask(
        data.get("interp"), channel_count, segment_count, "interp"
    )
    cannot = _channel_segment_mask(
        data.get("cantInterp", data.get("cantinterp")),
        channel_count,
        segment_count,
        "cantInterp",
    )
    return ArtifactState(layers=layers, interpolated=interpolated, cannot_interpolate=cannot)


def _orient_artifacts(value: Any, channels: int, segments: int) -> np.ndarray:
    matrix = np.asarray(value)
    matrix = np.squeeze(matrix)
    if matrix.ndim == 1 and channels == matrix.size and segments == 1:
        matrix = matrix[:, None, None]
    elif matrix.ndim == 2:
        matrix = _orient_channel_segment(matrix, channels, segments)[:, :, None]
    elif matrix.ndim == 3:
        candidates = []
        for channel_axis in range(3):
            for segment_axis in range(3):
                if channel_axis != segment_axis and matrix.shape[channel_axis] == channels and matrix.shape[segment_axis] == segments:
                    layer_axis = next(axis for axis in range(3) if axis not in (channel_axis, segment_axis))
                    candidates.append(np.transpose(matrix, (channel_axis, segment_axis, layer_axis)))
        if not candidates:
            raise FieldTripImportError("Artifact array does not align with channels and segments")
        matrix = candidates[0]
    else:
        raise FieldTripImportError("Artifact array must have one to three dimensions")
    return matrix.astype(bool, copy=False)


def _channel_segment_mask(
    value: Any, channels: int, segments: int, field: str
) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value)
    matrix = np.squeeze(matrix)
    if matrix.ndim == 1 and segments == 1 and matrix.size == channels:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise FieldTripImportError(f"{field} does not align with channels and segments")
    return _orient_channel_segment(matrix, channels, segments).astype(bool, copy=False)


def _orient_channel_segment(matrix: np.ndarray, channels: int, segments: int) -> np.ndarray:
    if matrix.shape == (channels, segments):
        return matrix
    if matrix.shape == (segments, channels):
        return matrix.T
    raise FieldTripImportError("Array does not align with channels and segments")


def _array_list(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [np.asarray(item) for item in value]
    array = np.asarray(value)
    if array.dtype == object:
        return [np.asarray(item) for item in array.ravel()]
    if array.ndim <= 1:
        return [array]
    return [array]


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, bytes):
        return (value.decode(errors="replace"),)
    array = np.asarray(value, dtype=object)
    result: list[str] = []
    for item in array.ravel():
        if isinstance(item, np.ndarray):
            item_array = np.asarray(item)
            if item_array.dtype.kind in "US" and item_array.ndim == 1:
                result.append("".join(str(part) for part in item_array).strip())
                continue
            if item_array.size == 1:
                item = item_array.item()
        if isinstance(item, bytes):
            result.append(item.decode(errors="replace").strip())
        else:
            result.append(str(item).strip())
    return tuple(text for text in result if text)


def _optional_text(value: Any) -> str | None:
    values = _text_tuple(value)
    return values[0] if values else None

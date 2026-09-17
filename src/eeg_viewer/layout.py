"""Independent channel-layout parsing, lookup, and deterministic fallback."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import MontageCandidate


@dataclass(frozen=True)
class ResolvedLayout:
    labels: tuple[str, ...]
    positions: NDArray[np.float64]
    matched: NDArray[np.bool_]
    kind: str
    source: str
    scalp_positions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.positions.shape != (len(self.labels), 2):
            raise ValueError("resolved positions must have shape channels x 2")
        if self.matched.shape != (len(self.labels),):
            raise ValueError("resolved match flags do not align with labels")
        if self.scalp_positions is not None:
            if self.scalp_positions.shape != (len(self.labels), 2):
                raise ValueError("scalp positions must have shape channels x 2")

    @property
    def coverage(self) -> float:
        return float(np.mean(self.matched)) if len(self.labels) else 0.0


def parse_fieldtrip_lay(path: str | Path) -> MontageCandidate:
    """Parse FieldTrip's text .lay format (id, x, y, width, height, label)."""

    layout_path = Path(path).expanduser().resolve()
    labels: list[str] = []
    positions: list[tuple[float, float]] = []
    with layout_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid .lay line {line_number}: expected at least 6 columns")
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError as error:
                raise ValueError(f"Invalid .lay coordinates on line {line_number}") from error
            labels.append(" ".join(parts[5:]))
            positions.append((x, y))
    if not labels:
        raise ValueError(f"No channels were found in {layout_path}")
    return MontageCandidate(
        labels=tuple(labels),
        positions=np.asarray(positions, dtype=float),
        source=str(layout_path),
        coordinate_system="fieldtrip-layout-2d",
    )


def resolve_layout(
    labels: Iterable[str],
    *,
    embedded: MontageCandidate | None = None,
    explicit_layout: str | Path | MontageCandidate | None = None,
    standard_montage: str | None = None,
) -> ResolvedLayout:
    """Resolve embedded, explicit, standard-label, then grid positions."""

    channel_labels = tuple(str(label) for label in labels)
    candidates: list[MontageCandidate] = []
    if embedded is not None:
        candidates.append(embedded)
    if explicit_layout is not None:
        candidates.append(
            explicit_layout
            if isinstance(explicit_layout, MontageCandidate)
            else parse_fieldtrip_lay(explicit_layout)
        )

    for candidate in candidates:
        resolved = _resolve_candidate(channel_labels, candidate)
        if resolved.coverage >= 0.5 or resolved.coverage == 1.0:
            return resolved

    standard = _standard_candidate(channel_labels, standard_montage)
    if standard is not None:
        resolved = _resolve_candidate(channel_labels, standard)
        if resolved.coverage >= 0.5:
            return resolved

    return grid_layout(channel_labels)


def grid_layout(labels: Iterable[str]) -> ResolvedLayout:
    channel_labels = tuple(str(label) for label in labels)
    return ResolvedLayout(
        labels=channel_labels,
        positions=_grid_positions(len(channel_labels)),
        matched=np.zeros(len(channel_labels), dtype=bool),
        kind="grid",
        source="deterministic-grid",
        scalp_positions=None,
    )


def _resolve_candidate(
    labels: tuple[str, ...], candidate: MontageCandidate
) -> ResolvedLayout:
    lookup: dict[str, int] = {}
    for index, label in enumerate(candidate.labels):
        key = canonical_label(label)
        if key and key not in lookup:
            lookup[key] = index

    positions = np.full((len(labels), 2), np.nan, dtype=float)
    matched = np.zeros(len(labels), dtype=bool)
    raw = _project_positions(np.asarray(candidate.positions, dtype=float))
    for channel, label in enumerate(labels):
        candidate_index = lookup.get(canonical_label(label))
        if candidate_index is not None and np.isfinite(raw[candidate_index]).all():
            positions[channel] = raw[candidate_index]
            matched[channel] = True

    if not np.any(matched):
        return ResolvedLayout(
            labels,
            _grid_positions(len(labels)),
            matched,
            "grid",
            candidate.source,
            None,
        )

    normalized = _normalise_xy(positions[matched])
    scalp_positions = np.full((len(labels), 2), np.nan, dtype=float)
    scalp_positions[matched] = _normalise_scalp(positions[matched])
    unmatched_count = int(np.sum(~matched))
    if unmatched_count == 0:
        positions[matched] = 0.08 + 0.84 * normalized
        kind = "sensor"
    else:
        positions[matched, 0] = 0.04 + 0.70 * normalized[:, 0]
        positions[matched, 1] = 0.06 + 0.88 * normalized[:, 1]
        side = _grid_positions(unmatched_count)
        side[:, 0] = 0.79 + 0.18 * side[:, 0]
        side[:, 1] = 0.04 + 0.92 * side[:, 1]
        positions[~matched] = side
        kind = "hybrid"
    return ResolvedLayout(
        labels,
        positions,
        matched,
        kind,
        candidate.source,
        scalp_positions,
    )


def _standard_candidate(
    labels: tuple[str, ...], requested: str | None
) -> MontageCandidate | None:
    import mne

    preferred = [
        "standard_1005",
        "standard_1020",
        "easycap-M1",
        "biosemi128",
        "biosemi64",
        "biosemi32",
        "GSN-HydroCel-129",
        "GSN-HydroCel-128",
        "GSN-HydroCel-64_1.0",
        "GSN-HydroCel-32",
    ]
    if requested:
        names = [requested]
    else:
        builtins = list(mne.channels.get_builtin_montages())
        names = preferred + [name for name in builtins if name not in preferred]

    target = {canonical_label(label) for label in labels}
    generic_numeric = bool(target) and all(re.fullmatch(r"e\d+", key) for key in target)
    best: tuple[float, MontageCandidate] | None = None
    errors: list[Exception] = []
    for name in names:
        try:
            montage = mne.channels.make_standard_montage(name)
        except (ValueError, RuntimeError) as error:
            errors.append(error)
            continue
        channel_positions = montage.get_positions()["ch_pos"]
        source_labels = tuple(channel_positions)
        source_keys = {canonical_label(label) for label in source_labels}
        overlap = len(target & source_keys)
        coverage = overlap / max(1, len(target))
        if generic_numeric:
            size_ratio = len(source_labels) / max(1, len(labels))
            if coverage < 0.90 or not 0.75 <= size_ratio <= 1.35:
                continue
        elif coverage < 0.5:
            continue
        exactness = -abs(len(source_labels) - len(labels)) / max(1, len(labels))
        score = coverage * 10.0 + exactness
        positions = np.asarray([channel_positions[label] for label in source_labels], dtype=float)
        candidate = MontageCandidate(
            labels=source_labels,
            positions=positions,
            source=f"mne:{name}",
            coordinate_system="standard",
            unit="m",
        )
        if best is None or score > best[0]:
            best = (score, candidate)
    if requested and best is None:
        detail = f": {errors[-1]}" if errors else ""
        raise ValueError(f"Standard montage {requested!r} did not match the dataset{detail}")
    return best[1] if best is not None else None


def canonical_label(label: str) -> str:
    text = re.sub(r"[\s_\-\.]+", "", str(label).casefold())
    if text.startswith("eeg") and len(text) > 3:
        text = text[3:]
    return text


def _normalise_xy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    centre = (minimum + maximum) / 2.0
    span = maximum - minimum
    scale = float(max(span.max(), 1e-12))
    return (values - centre) / scale + 0.5


def _normalise_scalp(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centre sensor geometry and fit it just inside a circular head outline."""

    values = np.asarray(values, dtype=float)
    centre = (values.min(axis=0) + values.max(axis=0)) / 2.0
    centred = values - centre
    radius = float(max(np.linalg.norm(centred, axis=1).max(), 1e-12))
    return 0.5 + 0.46 * centred / radius


def _project_positions(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project 3-D head coordinates azimuthally instead of dropping height."""

    if values.shape[1] == 2:
        return values
    norms = np.linalg.norm(values, axis=1)
    projected = np.full((values.shape[0], 2), np.nan, dtype=float)
    valid = np.isfinite(values).all(axis=1) & (norms > 1e-12)
    unit = np.zeros_like(values)
    unit[valid] = values[valid] / norms[valid, None]
    radial = np.hypot(unit[:, 0], unit[:, 1])
    polar_angle = np.arctan2(radial, unit[:, 2])
    off_axis = valid & (radial > 1e-12)
    projected[off_axis, 0] = (
        polar_angle[off_axis] * unit[off_axis, 0] / radial[off_axis]
    )
    projected[off_axis, 1] = (
        polar_angle[off_axis] * unit[off_axis, 1] / radial[off_axis]
    )
    projected[valid & ~off_axis] = 0.0
    return projected


def _grid_positions(count: int) -> NDArray[np.float64]:
    if count <= 0:
        return np.empty((0, 2), dtype=float)
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    result = np.empty((count, 2), dtype=float)
    for index in range(count):
        row, column = divmod(index, columns)
        result[index, 0] = (column + 0.5) / columns
        result[index, 1] = 1.0 - (row + 0.5) / rows
    return result

from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_viewer.layout import grid_layout, parse_fieldtrip_lay, resolve_layout
from eeg_viewer.model import MontageCandidate


def test_parse_fieldtrip_lay_and_preserve_channel_order(tmp_path: Path) -> None:
    path = tmp_path / "test.lay"
    path.write_text(
        "1 -0.5 0.8 0.1 0.1 Fp1\n2 0.0 0.0 0.1 0.1 Cz\n3 0.5 -0.8 0.1 0.1 O2\n",
        encoding="utf-8",
    )

    parsed = parse_fieldtrip_lay(path)
    resolved = resolve_layout(("O2", "Fp1", "Cz"), explicit_layout=parsed)

    assert parsed.labels == ("Fp1", "Cz", "O2")
    assert resolved.labels == ("O2", "Fp1", "Cz")
    assert resolved.kind == "sensor"
    assert resolved.coverage == 1.0
    assert resolved.positions[1, 1] > resolved.positions[0, 1]
    assert resolved.scalp_positions is not None
    assert resolved.scalp_positions[1, 1] > resolved.scalp_positions[0, 1]
    scalp_radius = np.linalg.norm(2.0 * resolved.scalp_positions - 1.0, axis=1)
    assert np.all(scalp_radius <= 0.92 + 1e-12)


def test_embedded_layout_wins_and_unmatched_auxiliary_channel_is_retained() -> None:
    embedded = MontageCandidate(
        labels=("Fp1", "Cz", "O2"),
        positions=np.array([[-1, 1], [0, 0], [0.7, -1]], dtype=float),
        source="embedded-test",
    )

    resolved = resolve_layout(("Fp1", "Cz", "O2", "EOG"), embedded=embedded)

    assert resolved.source == "embedded-test"
    assert resolved.kind == "hybrid"
    assert resolved.matched.tolist() == [True, True, True, False]
    assert resolved.positions[3, 0] > 0.79
    assert resolved.scalp_positions is not None
    assert np.isnan(resolved.scalp_positions[3]).all()


def test_standard_label_lookup_and_grid_fallback() -> None:
    standard = resolve_layout(("Fp1", "Cz", "O2"), standard_montage="standard_1020")
    fallback = resolve_layout(("custom-a", "custom-b", "custom-c"))

    assert standard.kind == "sensor"
    assert standard.source == "mne:standard_1020"
    assert standard.coverage == 1.0
    assert fallback.kind == "grid"
    assert fallback.coverage == 0.0
    assert fallback.scalp_positions is None
    np.testing.assert_allclose(fallback.positions, grid_layout(fallback.labels).positions)


def test_three_dimensional_positions_are_projected_without_collapsing_height() -> None:
    montage = MontageCandidate(
        labels=("upper-right", "lower-right", "upper-front", "lower-front"),
        positions=np.array(
            [[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=float
        ),
        source="3d-test",
    )

    resolved = resolve_layout(montage.labels, embedded=montage)

    assert not np.allclose(resolved.positions[0], resolved.positions[1])
    assert not np.allclose(resolved.positions[2], resolved.positions[3])

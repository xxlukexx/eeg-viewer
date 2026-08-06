from __future__ import annotations

import numpy as np

from eeg_viewer.synthetic import SyntheticConfig, generate_recording


def test_generator_is_deterministic_and_includes_known_stress_events() -> None:
    config = SyntheticConfig(
        channels=12,
        sample_rate_hz=250.0,
        duration_seconds=30.0,
        seed=9,
    )

    first = generate_recording(config)
    second = generate_recording(config)

    assert first.data_uv.shape == (12, 7_500)
    assert first.data_uv.dtype == np.float32
    np.testing.assert_array_equal(first.data_uv, second.data_uv)
    kinds = {event.kind for event in first.events}
    assert {"blink", "large_erp", "single_sample_spike"} <= kinds
    assert np.nanmax(np.abs(first.data_uv)) > 200.0


def test_generator_can_insert_a_nan_gap_explicitly() -> None:
    recording = generate_recording(
        SyntheticConfig(
            channels=4,
            sample_rate_hz=100.0,
            duration_seconds=5.0,
            include_nans=True,
        )
    )

    assert np.isnan(recording.data_uv).any()
    assert any(event.kind == "nan_gap" for event in recording.events)

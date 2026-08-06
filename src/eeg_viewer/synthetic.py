"""Deterministic, visibly EEG-like data for rendering and navigation tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SyntheticConfig:
    channels: int = 129
    sample_rate_hz: float = 1_000.0
    duration_seconds: float = 180.0
    seed: int = 17
    mains_hz: float = 50.0
    include_nans: bool = False

    @property
    def samples(self) -> int:
        return max(2, int(round(self.sample_rate_hz * self.duration_seconds)))


@dataclass(frozen=True)
class SyntheticEvent:
    kind: str
    start_sample: int
    stop_sample: int
    channels: tuple[int, ...]
    amplitude_uv: float | None = None


@dataclass(frozen=True)
class SyntheticRecording:
    data_uv: FloatArray
    sample_rate_hz: float
    channel_labels: tuple[str, ...]
    events: tuple[SyntheticEvent, ...]
    config: SyntheticConfig

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "shape": list(self.data_uv.shape),
            "dtype": str(self.data_uv.dtype),
            "unit": "uV",
            "event_count": len(self.events),
        }


def _sample_bounds(
    centre_seconds: float,
    width_seconds: float,
    sample_rate_hz: float,
    sample_count: int,
) -> tuple[int, int]:
    centre = int(round(centre_seconds * sample_rate_hz))
    radius = max(1, int(round(width_seconds * sample_rate_hz / 2)))
    return max(0, centre - radius), min(sample_count, centre + radius + 1)


def generate_recording(
    config: SyntheticConfig,
    progress: Callable[[str], None] | None = None,
) -> SyntheticRecording:
    """Generate repeatable data in microvolts with known visual stress events."""

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    if config.channels < 1:
        raise ValueError("channels must be positive")
    if config.sample_rate_hz <= 0 or config.duration_seconds <= 0:
        raise ValueError("sample rate and duration must be positive")

    rng = np.random.default_rng(config.seed)
    channels = config.channels
    samples = config.samples
    sfreq = float(config.sample_rate_hz)
    report(f"Generating {channels} channels x {samples:,} samples")

    data = rng.standard_normal((channels, samples), dtype=np.float32)
    data *= np.float32(3.5)
    time = np.arange(samples, dtype=np.float32) / np.float32(sfreq)

    report("Adding correlated rhythms, drift, and mains interference")
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(channels, 4))
    weights = rng.uniform(0.35, 1.2, size=(channels, 4)).astype(np.float32)
    frequencies = (2.2, 6.0, 10.0, 19.0)
    amplitudes = (3.0, 2.5, 7.0, 1.2)
    common_alpha = np.sin(2.0 * np.pi * 10.0 * time, dtype=np.float32)
    spatial_alpha = np.linspace(-1.0, 1.0, channels, dtype=np.float32)

    # Per-channel construction avoids allocating another channel x sample array.
    for channel in range(channels):
        signal = data[channel]
        for component, (frequency, amplitude) in enumerate(
            zip(frequencies, amplitudes, strict=True)
        ):
            signal += (
                np.float32(amplitude)
                * weights[channel, component]
                * np.sin(
                    np.float32(2.0 * np.pi * frequency) * time
                    + np.float32(phases[channel, component]),
                    dtype=np.float32,
                )
            )
        signal += np.float32(2.5 * spatial_alpha[channel]) * common_alpha
        signal += np.float32(4.0) * np.sin(
            np.float32(2.0 * np.pi * 0.08) * time
            + np.float32(phases[channel, 0]),
            dtype=np.float32,
        )
        signal += np.float32(0.45) * np.sin(
            np.float32(2.0 * np.pi * config.mains_hz) * time
            + np.float32(phases[channel, 1]),
            dtype=np.float32,
        )

    events: list[SyntheticEvent] = []
    frontal = tuple(range(min(channels, max(4, channels // 12))))
    broad = tuple(range(min(channels, max(12, channels // 3))))

    report("Adding blink, ERP, spike, step, clipping, and flat-channel events")
    blink_times = np.arange(7.5, config.duration_seconds, 17.0)
    for blink_number, centre in enumerate(blink_times):
        start, stop = _sample_bounds(centre, 0.55, sfreq, samples)
        if stop <= start:
            continue
        local_time = (np.arange(start, stop, dtype=np.float32) / np.float32(sfreq))
        width = np.float32(0.11 + 0.015 * (blink_number % 3))
        pulse = np.exp(
            -0.5 * ((local_time - np.float32(centre)) / width) ** 2,
            dtype=np.float32,
        )
        amplitude = np.float32(80.0 + 12.0 * (blink_number % 4))
        for rank, channel in enumerate(frontal):
            data[channel, start:stop] += amplitude * np.float32(
                1.0 - 0.5 * rank / max(1, len(frontal) - 1)
            ) * pulse
        events.append(
            SyntheticEvent("blink", start, stop, frontal, float(amplitude))
        )

    erp_times = np.arange(12.0, config.duration_seconds, 23.0)
    for centre in erp_times:
        start, stop = _sample_bounds(centre, 0.8, sfreq, samples)
        local_time = np.arange(start, stop, dtype=np.float32) / np.float32(sfreq)
        negative = -np.float32(65.0) * np.exp(
            -0.5 * ((local_time - np.float32(centre + 0.17)) / np.float32(0.045)) ** 2,
            dtype=np.float32,
        )
        positive = np.float32(32.0) * np.exp(
            -0.5 * ((local_time - np.float32(centre + 0.31)) / np.float32(0.08)) ** 2,
            dtype=np.float32,
        )
        waveform = negative + positive
        for rank, channel in enumerate(broad):
            data[channel, start:stop] += waveform * np.float32(
                1.0 - 0.65 * rank / max(1, len(broad) - 1)
            )
        events.append(SyntheticEvent("large_erp", start, stop, broad, -65.0))

    stress_times = np.arange(5.0, config.duration_seconds, 11.0)
    for index, seconds in enumerate(stress_times):
        channel = int((index * 17 + 3) % channels)
        sample = min(samples - 1, int(round(seconds * sfreq)))
        data[channel, sample] += np.float32(280.0 if index % 2 == 0 else -260.0)
        events.append(
            SyntheticEvent(
                "single_sample_spike", sample, sample + 1, (channel,), float(data[channel, sample])
            )
        )

    if samples > int(20 * sfreq):
        start = int(0.41 * samples)
        stop = min(samples, start + int(0.65 * sfreq))
        step_channels = tuple(range(1, min(channels, 5)))
        data[list(step_channels), start:stop] += np.float32(75.0)
        events.append(SyntheticEvent("step", start, stop, step_channels, 75.0))

        start = int(0.63 * samples)
        stop = min(samples, start + int(1.2 * sfreq))
        flat_channel = min(channels - 1, max(0, channels // 2))
        data[flat_channel, start:stop] = np.float32(0.0)
        events.append(SyntheticEvent("flat", start, stop, (flat_channel,), 0.0))

        start = int(0.79 * samples)
        stop = min(samples, start + int(0.9 * sfreq))
        clip_channel = min(channels - 1, max(0, channels // 3))
        data[clip_channel, start:stop] = np.clip(
            data[clip_channel, start:stop] * np.float32(8.0), -45.0, 45.0
        )
        events.append(SyntheticEvent("clipping", start, stop, (clip_channel,), 45.0))

    if config.include_nans and samples > int(3 * sfreq):
        start = int(0.9 * samples)
        stop = min(samples, start + max(2, int(0.03 * sfreq)))
        channel = channels - 1
        data[channel, start:stop] = np.nan
        events.append(SyntheticEvent("nan_gap", start, stop, (channel,), None))

    labels = tuple(f"EEG {index + 1:03d}" for index in range(channels))
    report(f"Synthetic recording ready ({data.nbytes / 1024**2:.1f} MiB)")
    return SyntheticRecording(data, sfreq, labels, tuple(events), config)

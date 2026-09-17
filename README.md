# EEG Viewer 1.0

EEG Viewer is a fast, read-only desktop application for reviewing EEG waveforms,
trial-level artifact decisions, and interpolation state.

## Supported inputs

- FieldTrip continuous and segmented raw structures;
- FieldTrip timelock averages, including files containing several named
  conditions;
- FieldTrip grand averages with a grand mean and/or named individual subjects;
- conventional MATLAB MAT files and MATLAB 7.3/HDF5 files; and
- continuous EEGLAB `.set`/`.fdt` recordings, read lazily through MNE.

## Install and run without administrator access

Python 3.10 or newer is required. Run from this directory:

```text
python bootstrap.py
```

Use `python3 bootstrap.py` where Python is named `python3`. The bootstrap creates
`.venv` beside the application and installs packages there. It does not modify
system Python or require administrator/root access. The first run downloads
binary wheels and then opens a file chooser.

Files can subsequently be opened with **Open EEG…** or dropped onto an existing
viewer window. Supported files can also be supplied directly:

```text
python bootstrap.py --input path/to/data.mat --mode scalp
python bootstrap.py --input path/to/recording.set --mode chart
python bootstrap.py --input path/to/recording.fdt
```

Dropping an `.fdt` locates its same-named `.set` file. Import failures are shown
in a readable dialog and written to the optional log.

After an editable/development installation, the command is:

```text
eeg-viewer
```

PySide6 wheels include Qt on Windows, macOS, and mainstream Linux platforms. A
normal Linux desktop must provide its usual display/graphics system libraries.
Windows has been exercised directly; macOS and Linux still require release
smoke tests.

## Views and interaction

The **View** selector provides:

- `chart`: vertically stacked channels with channel scrolling;
- `scalp`: waveform tiles positioned by the resolved sensor layout; and
- `grid`: the same tile renderer in a deterministic regular grid.

Segmented data shows a trial overview below the waveforms. Each bar is one trial;
height is the number of unique channels marked by any artifact detector or as
impossible to interpolate. Click or drag across it to navigate. Average and
grand-average files expose named conditions or subjects in the **Series** menu.

For segmented data, the **Clean-trial average α** slider defaults to 0.50. A thicker, purple
trace beneath each channel's current trial shows the mean of that channel's
unflagged trials. Its visible range is independently fitted symmetrically around
zero within that channel's display height; the trial traces retain their shared
native-unit scale. Trials marked with any artifact, interpolation, or
cannot-interpolate flag for that channel are excluded. Trials of different
lengths are aligned by their recorded start times; missing or non-finite samples
do not contribute. When a trial contains timestamps below zero, their finite
per-channel mean is automatically subtracted before that trial contributes to
the clean average. Move the opacity slider to 0 to hide the overlay.

When segmented data has usable EEG sensor positions, two top-down scalp maps
appear beside the waveforms. The left map shows the same greedy, per-channel
clean-trial average as the traces; the right map shows the selected trial. Move
the pointer across any waveform cell to set their shared time centre. The
**Window** control sets the duration averaged around that centre. Both heads use
independent symmetric, zero-centred colour scales, so the lower-amplitude clean
average retains useful contrast. Sensor labels are available by hovering over
their dots. The checkable **Scalp maps** control beside the view selector shows
or hides the panel. A lightly regularized thin-plate spline produces the smooth
scalp field without visible triangulation facets. The selected-trial map receives
the same automatic negative-time baseline correction; raw trial traces remain
unaltered.

A small **Trial scale** inset at the lower-right of the plot shows paper-style
time and native-amplitude scale bars. Their labels and line lengths follow the
current time and amplitude zoom; they describe the trials, not the rescaled
average.

Hovering over a channel reports label, type, declared unit, artifact types,
interpolation state, cursor time/value, and visible-window min/max/peak-to-peak.
A short vertical cursor follows the pointer within that channel, with a small
time label in milliseconds; both disappear when the pointer leaves the channel.
The scalp maps retain the last selected cursor time.

- Mouse wheel: move through time.
- Ctrl+mouse wheel: zoom in or out around the cursor.
- Shift+mouse wheel: scroll chart channels.
- Left-drag: pan through time.
- Left/right: move by one quarter-window.
- Shift+left/right: move by a full window.
- Page Up/Page Down: previous/next segment; on continuous data, one time window.
- Up/down: move through chart channels.
- `+`/`=` and `-`/`_`: increase or decrease waveform amplitude.
- Home/End: jump to the beginning or end.

The hover panel always reserves two lines, so inspection never resizes the EEG
display.

## Layouts, units, and artifacts

Layout resolution remains independent of file import and follows:

1. embedded FieldTrip/EEGLAB positions when valid;
2. an explicit FieldTrip `.lay` supplied with `--layout`;
3. conservative label matching against MNE's standard montages; and
4. a deterministic grid.

Unmatched auxiliary channels are retained in a side grid. Three-dimensional
positions use an azimuthal scalp projection rather than discarding sensor height.

FieldTrip `.art`, `.art_type`, `.interp`, and `.cantInterp` are imported as
read-only state. Tile borders distinguish artifacts, interpolated channels, and
channels that cannot be interpolated; hover reports all simultaneous states.

Signal values stay in native stored units. Declared channel metadata is displayed
but never used for silent conversion because historical files sometimes contain
microvolt-scale values while declaring volts. Initial spacing is derived from the
signal and can be overridden with `--amplitude-spacing`.

## Correctness and performance

Display reduction retains each bin's minimum and maximum in temporal order, so
narrow spikes do not disappear when long windows are shown.

Automated timing controls are hidden from normal users. Developers can enable
them with `--developer-controls`, or run a forced-paint benchmark with:

```text
python bootstrap.py --synthetic --auto-benchmark 40 --developer-controls
```

The test suite covers adapters, MAT decoding fallback, multi-condition and grand
averages, lazy EEGLAB access/cache behavior, navigation, layouts, artifacts,
hover geometry, trial overview behavior, and headless Qt rendering:

```text
.venv/Scripts/python -m pip install -e ".[test]"  # Windows
.venv/Scripts/python -m pytest
```

Use `.venv/bin/python` on macOS/Linux.

The same suite runs in GitHub Actions on Windows, macOS, and Linux. Interactive
release smoke tests are still required because a headless test cannot validate
the feel of native windowing, mouse input, or every graphics driver.

## Reproducible local fixtures

Generated participant-derived files remain local and are ignored by Git:

```text
.venv/Scripts/python scripts/create_fieldtrip_fixture.py results/fieldtrip_fixture.mat
.venv/Scripts/python scripts/create_grand_average_fixture.py SOURCE_DIRECTORY results/grand_average.mat
```

The second command builds a FieldTrip `individual` grand-average structure from
compatible subject averages and also includes their grand mean.

## Deferred beyond Version 1

- artifact editing and manual review decisions;
- undo/redo and persistence/export;
- continuous interval annotations and EEGLAB event display;
- additional EEG/BIDS adapters; and
- further cache complexity unless larger recordings demonstrate a need.

The scalp view arranges waveform tiles at electrodes; the accompanying dual-head
panel provides the interpolated voltage topomaps.

## License

EEG Viewer is released under the [MIT License](LICENSE).

## Project provenance

Version 1 was extracted from the rendering work originally developed alongside
`ECKEEGVis.m` in `eeg_tools`. The source relationship, extraction commits, and
data-handling boundaries are recorded in
[docs/PROVENANCE.md](docs/PROVENANCE.md). No participant EEG data are committed.

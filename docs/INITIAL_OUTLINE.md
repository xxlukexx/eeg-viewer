# Python EEG Viewer: Initial Working Outline

> **Status:** working document; initial outline only
>
> **Date:** 6 August 2026
> **Source prototype:** `ECKEEGVis.m` and related MATLAB code in `eeg_tools`

This document captures an initial reading of the existing MATLAB prototype and
a possible direction for a Python replacement. It is deliberately provisional.
It is not a specification, an architectural commitment, or a promise to retain
the names, boundaries, libraries, file formats, or implementation sequence
described below.

The immediate purpose is to preserve the useful ideas in the prototype, make
the main uncertainties explicit, and define a small first experiment that will
give us evidence before we make larger design decisions.

## Current position

The first rendering spike was completed on 6 August 2026. It uses PySide6 and
PyQtGraph, with a single batched waveform item and an optional OpenGL viewport.
It includes deterministic synthetic EEG-like data, continuous navigation,
amplitude scaling, channel scrolling, ordered-extrema display reduction,
automated timings, and an interactive window.

Initial results support using PyQtGraph with OpenGL for the next increment:

- 32 visible channels over 10 seconds: 32.2 ms median synchronous frame;
- 129 visible channels over 10 seconds: 61.4 ms median synchronous frame; and
- 129 visible channels over 60 seconds: 99.9 ms median synchronous frame.

These measurements force a widget render and image readback, so they are more
conservative than ordinary screen presentation. The 60-second all-channel case
is primarily limited by CPU extrema preparation; submitting prepared geometry
to PyQtGraph remains approximately 1 ms. The next relevant performance work is
therefore a multiresolution display cache, not an immediate renderer rewrite.

This is a provisional implementation decision. The renderer remains behind a
small boundary so that it can be reconsidered if real-data or topographic-view
benchmarks reveal a materially different workload.

### FieldTrip/layout increment (6 August 2026)

The next working increment is now implemented in the same provisional
benchmark area. This remains exploratory code, not a frozen application
architecture or file-format specification.

It adds:

- a small internal contract for channels, variable-length segments, display
  series, montage candidates, and imported artifact state;
- FieldTrip import for raw continuous data, segmented trials, timelock averages,
  and `individual` grand-average data;
- conventional and MATLAB 7.3/HDF5 MAT reading via `pymatreader`;
- embedded `elec` import without coupling layout resolution to the FieldTrip
  adapter;
- an independent resolution chain of embedded montage, explicit `.lay`, MNE
  standard-label lookup, and deterministic grid fallback;
- preservation of unmatched channels using a hybrid sensor-plus-side-grid
  arrangement;
- chart, scalp-positioned small-multiple, and regular-grid views using the same
  signal source, navigation state, extrema reduction, and batched waveform;
- segment and series switching; and
- renderer-neutral import of `.art`, `.art_type`, `.interp`, and `.cantInterp`.

The artifact state is deliberately not encoded in channel geometry. Each view
can ask for the same per-channel/per-segment visual state and decide how to draw
it. Tile borders currently provide only a simple proof that this separation
works; their colours and interaction are not a UI decision. Continuous manual
annotation and editing remain deferred.

The validation suite currently contains 15 passing tests, including a
conventional MAT round trip and a headless Qt test that changes views and
variable-length segments. A generated 32-channel, 24-trial FieldTrip fixture
imports its embedded standard montage and three artifact layers successfully.

A first randomly selected existing LEAP file was also loaded successfully. It
contains 70 channels, 168 trials of 1,001 samples at 1 kHz, a complete embedded
layout, three artifact layers (`minmax`, `range`, and `eogstat`), and imported
interpolation state. A five-frame headless scalp-view smoke run measured 13.4 ms
median and 20.4 ms p95 synchronous frame time on the development machine.

This file exposed an important unit issue: its channel metadata declares volts,
but its stored values are on a scale around tens of units and appear already
scaled for EEG display. The viewer therefore does not silently convert values
from channel metadata. It labels the amplitude as native stored units, retains
the declared channel unit separately, and derives a readable initial spacing
from the signal values. A later format-specific unit policy will need evidence
from more files rather than a heuristic conversion.

The first visual review also exposed two interaction/layout defects. Directly
discarding the third sensor coordinate collapsed inferior and lateral channels;
the layout now uses an azimuthal 3-D scalp projection and sizes tiles against
all channel pairs. The 70-channel test layout now has no intersecting panels.
Page Up/Down now step through segments, while both shifted and unshifted forms
of the plus/minus keys change waveform amplitude even when a child control has
keyboard focus.

A second existing FieldTrip file exposed a `pymatreader` failure on nested
MATLAB metadata even though its waveform structure was conventional. Import now
falls back to SciPy's simplified conventional-MAT decoder when the primary
decoder fails, while retaining `pymatreader` for MATLAB 7.3/HDF5 support. The
fallback path has a regression test and records which decoder was used in the
dataset metadata.

### Read-only review polish (6 August 2026)

The viewer now has a clickable trial overview with one bar per segment. Bar
height is the number of unique channels marked by any imported artifact layer
or as impossible to interpolate; a white cursor identifies the current trial.
This deliberately summarises channel burden rather than summing detector hits,
which would double-count channels marked by several detectors.

Mouse hover works across chart, scalp, and grid views. It highlights the
channel and reports label, index, type, declared unit, artifact types,
interpolation state, cursor time/value, and visible-window signal range. A
compact legend explains the artifact/interpolation colours. The scalp view also
has a nose orientation marker and a constrained wide-head aspect ratio that
retains recognisable spatial structure without wasting most of a wide display.
Editing remains explicitly deferred.

### Version 1 hardening (6 August 2026)

The read-only feature set is frozen as Version 1.0. The application now opens a
file chooser on normal startup, provides **Open EEG…**, accepts `.mat`, `.set`,
and paired `.fdt` files by drag-and-drop, and reports import failures in a GUI
dialog. Benchmark controls are hidden unless explicitly requested by a developer.

Average handling was validated against LEAP subject-average files. Containers
with several timelock structures become named display series (for example,
`face_up` and `face_inv`). A reproducible local script builds a FieldTrip
grand-average fixture containing both a grand mean and named individuals; the
Series control displays those labels rather than numeric indices.

Continuous access was validated on a 61-channel, 31-minute EEGLAB `.set`/`.fdt`
recording at 256 Hz. The adapter uses MNE without preloading the 116 MB signal
and keeps a bounded 60-second all-channel read-ahead cache. Random uncached jumps
over the network measured roughly 150 ms, while adjacent cached 10-second window
preparation measured 3.7 ms median and 5.4 ms p95.

Because the sampled project files were all conventional MATLAB 5 files, MATLAB
R2026a was used to write a genuine `-v7.3` version of the derived grand-average
fixture. It imports through the HDF5 path with six named series (grand mean plus
five participants).

The local package is now `eeg-review-viewer` Version 1.0.0 with an `eeg-viewer`
command. Windows bootstrap installation and dependency consistency have been
verified. A headless test workflow covers Windows, macOS, and Linux when pushed;
manual visual smoke tests on macOS and Linux remain outstanding. Editing,
review decisions, undo/redo, and export remain Version 2 concerns.

## Purpose

The intended tool sits between fully manual EEG cleaning and unattended
automatic cleaning. A typical workflow is:

1. Choose artifact-detection parameters and clean a batch of data.
2. Plot distributions of included and excluded trial counts.
3. Inspect the tails and other suspicious cases in the viewer.
4. Identify false positives, false negatives, or systematic physiological
   features that resemble artifacts.
5. Adjust the parameters and repeat.

The viewer should make this review fast enough that scrolling through many
trials or long continuous recordings is practical. It is primarily a review
and visualisation tool, not an EEG analysis package.

## What the MATLAB prototype demonstrates

`ECKEEGVis.m` contains several ideas worth preserving:

- topographically arranged channel traces;
- fast keyboard navigation between trials;
- separate artifact detector layers;
- visual indication of interpolation and failed interpolation;
- a compact overview of artifact density across trials;
- channel, latency, voltage, and artifact-type hover information;
- manual marking;
- overlaying average waveforms; and
- batched, redraw-on-change rendering rather than conventional MATLAB axes.

The associated artifact convention is a logical
`channel x trial x detector` array in `.art`, with `.art_type` naming the third
dimension. This is useful as an import/export convention for existing data.

## Problems in the current implementation

The current visualiser combines data conversion, layout reconciliation,
artifact storage, interpolation state, rendering, keyboard and mouse input,
edit history, and persistence in one mutable class. This makes otherwise small
changes interact in surprising ways.

Some representative limitations and defects are:

- Continuous FieldTrip data is treated as one very large trial rather than as
  a navigable time stream.
- Timelock data is handled by temporarily disguising `avg` as a one-element
  trial array; grand averages are not modelled properly.
- Channels without layout positions can be removed from the loaded data.
- Periodic sample decimation can hide narrow extrema, including the spikes the
  reviewer is trying to find.
- Layout, render geometry, input handling, and source data are tightly coupled.
- The partially completed `EEGVis_*` rewrite starts separating responsibilities
  but remains tied to Psychtoolbox and FieldTrip.
- Undo/redo state is declared but not populated by edits.
- Several input and multidimensional indexing paths are fragile.
- The manual artifact layer can add a bad mark but cannot reliably override an
  automated mark as a false positive, because downstream status is formed by
  OR-ing all layers.

These observations describe reasons to replace the structure, not a requirement
to reproduce every historical behaviour.

## Provisional product direction

The replacement may eventually support:

- segmented trial data;
- continuous data in a vertically stacked channel chart;
- continuous or segmented data in a topographic small-multiple view;
- stepping by trial or by a configurable duration such as 1 or 10 seconds;
- subject averages, condition overlays, and grand averages;
- sensor-layout positioning with a deterministic grid fallback;
- artifact-type, interpolation, and review overlays;
- manual review with explicit false-positive and false-negative decisions;
- a timeline or overview showing artifact density and review state; and
- adapters for FieldTrip initially, followed potentially by MNE, BIDS,
  BrainVision, EEGLAB, and other formats.

This list is a direction of travel, not the scope of the first implementation.

## Provisional data boundary

FieldTrip should be an input adapter, not necessarily the application's native
data model. MNE may be useful for format readers and metadata handling, but it
also should not automatically become the viewer's domain model.

A small read-oriented interface is likely to be more useful:

```python
class SignalSource(Protocol):
    metadata: RecordingMetadata
    channels: ChannelTable
    segments: SegmentTable
    layout: SensorLayout | None

    def read(
        self,
        segment_id: str,
        channels: Sequence[int],
        start_sample: int,
        stop_sample: int,
    ) -> SignalBlock:
        ...
```

The important concept is random access to a requested channel and sample
window. A continuous recording can be one long segment; epoched data can expose
many segments; and averages can expose one or more display series. Exact class
names and shapes should wait until real use cases and benchmark code expose
what is convenient.

Integer sample positions should remain the authoritative coordinates. Floating
point time values are useful for display but are a poor identity for edits and
annotations.

Units must be explicit. Existing FieldTrip files may contain values in microvolts
without sufficient metadata, while MNE expects SI units. An adapter should
preserve known units and expose unknown units rather than silently guessing.

## Provisional annotation model

The dense FieldTrip `.art` array works for trial-level annotations but does not
generalise cleanly to continuous time. A possible native representation is a
set of interval annotations containing:

- recording or segment identity;
- half-open start and stop sample positions;
- one, several, or all channels;
- kind, such as artifact, interpolation, or rejection;
- label, such as blink, range, flat, or manual;
- source, such as detector, import, or reviewer; and
- review decision, such as no override, force bad, or force good.

The detector output should remain unchanged when a reviewer overrides it. The
effective reviewed state can then be derived without destroying provenance.
FieldTrip `.art`, `.art_type`, `.interp`, and `.cantInterp` can be imported into
this representation and exported through a compatibility path later.

This is an early proposal only. Annotation and persistence design should not be
allowed to complicate the initial rendering benchmark.

## Rendering options under consideration

The initial candidates are:

- **PySide6 + PyQtGraph:** comparatively mature, convenient scientific plotting,
  peak-preserving downsampling, clipping, and an existing precedent in the MNE
  Qt browser.
- **PySide6 + VisPy:** a more explicitly GPU-oriented canvas with batched line
  geometry, at the cost of more custom rendering work.
- **A lower-level Qt/OpenGL or Qt Quick renderer:** potentially the highest
  ceiling, but currently unlikely to justify the additional maintenance.

The benchmark should be structured so that drawing engines can be compared
without changing the synthetic signal generator, navigation state, windowing,
downsampling, or measurements.

The implementation should avoid creating one heavyweight plot widget per
electrode. A chart or topographic screen should batch waveform geometry into a
small number of drawable objects.

### Provisional decision after the first benchmark

Use **PySide6 + PyQtGraph with OpenGL enabled by default** for the next
increment, while retaining a CPU fallback. Do not implement the same benchmark
in VisPy yet. The current measurements do not show a rendering-engine bottleneck
large enough to justify the additional implementation and maintenance cost.

Revisit this decision if later evidence shows that:

- real-data window loading behaves materially differently from the synthetic
  array source;
- the batched topographic small-multiple view cannot meet interaction targets;
- platform testing exposes unacceptable Qt/OpenGL compatibility problems; or
- waveform submission and painting, rather than data preparation, becomes the
  dominant cost.

## First experiment: realistic synthetic rendering benchmark

**Status: completed for the initial PyQtGraph CPU/OpenGL comparison.** The code
became this standalone repository and the report is in
`benchmarks/RESULTS_2026-08-06.md`. The sections below retain the original
experimental intent and remain useful when repeating the benchmark on other
machines or against another renderer.

The first implementation should be a deliberately disposable rendering spike
with reusable foundations. It should answer whether candidate Python drawing
stacks can provide the required interaction latency on representative hardware.

It should not yet load real EEG, understand FieldTrip, edit artifacts, calculate
topographic projections, or establish a persistent application schema.

### Synthetic data

Generate deterministic, EEG-like multichannel data with configurable:

- channel counts such as 32, 64, 128, and 256;
- sample rates such as 250, 500, 1,000, and 2,000 Hz;
- recording duration and epoch count;
- oscillatory background activity;
- correlated spatial components;
- slow drift and mains interference;
- narrow spikes, steps, clipping, flat periods, and high-frequency bursts;
- blink-like broad frontal events;
- occasional very large ERP-like components; and
- NaNs or discontinuities where useful for stressing rendering paths.

The generator is not intended to be physiologically complete. It needs to make
visual defects, hidden extrema, scaling mistakes, and performance bottlenecks
obvious. It should use a fixed seed and return both signal data and known event
locations so rendered output can be checked.

### Reusable non-rendering components

Even in the spike, keep these independent of the drawing engine:

- a channel-by-sample source interface;
- a view-window and navigation state;
- channel selection and ordering;
- amplitude scaling;
- min/max or ordered-extrema reduction that cannot hide spikes;
- a small window cache and adjacent-window prefetch interface;
- deterministic synthetic data generation; and
- timing and frame-statistics collection.

These pieces are likely to survive even if the selected rendering library does
not.

### Initial views

Start with only a stacked chart view. It exercises the core workload without
introducing layout geometry. Useful interactions are:

- move left and right by a fraction or a full window;
- select 1, 10, 30, or 60 second windows;
- zoom and pan in time;
- adjust amplitude scale;
- change the number of visible channels; and
- resize the application window.

A synthetic small-multiple grid can be added after the chart benchmark if it
helps expose per-channel viewport overhead. It should not yet be described as a
real topographic view.

### Measurements

Record at least:

- application startup and first-draw time;
- uncached and cached window-change latency;
- pan/zoom frame time and dropped frames;
- geometry preparation time separately from drawing time;
- amount of data copied per update;
- peak and steady-state memory use;
- resize latency; and
- performance with CPU rendering and available accelerated modes.

Measurements should be written to a small machine-readable result file with
the machine, Python, Qt, library, and GPU/driver versions. Visual correctness
matters alongside speed: narrow synthetic extrema must remain visible at every
zoom level where their time bin is represented.

### Provisional targets

Targets are useful for interpreting the experiment but are not yet product
requirements:

- cached navigation should feel immediate, approximately 30-50 ms or less;
- an uncached window should normally appear within roughly 100 ms;
- direct manipulation should sustain smooth interaction on representative
  128/129-channel data; and
- memory use should remain bounded for long recordings rather than requiring
  the full recording to be copied into display buffers.

### Decision produced by the experiment

The spike should finish with an evidence-based choice among:

1. PyQtGraph is comfortably adequate and becomes the initial renderer.
2. PyQtGraph is adequate only with a custom batched waveform item.
3. VisPy offers a material improvement worth its added complexity.
4. Neither is adequate, justifying investigation of a lower-level renderer.

Only after that decision should the project commit to the application shell and
begin FieldTrip integration, sensor layouts, or annotation editing.

## Possible later architecture

If the benchmark supports continuing, a likely separation is:

```text
format adapters
    -> signal source and metadata
        -> window loading, reduction, and cache
            -> shared navigation/view state
                -> chart renderer
                -> topographic renderer

annotation adapters
    -> annotation/review store
        -> review commands
            -> overlays and compatibility export
```

The chart and topographic views should share navigation, selection, scaling,
and annotation state rather than becoming separate applications.

## Testing direction

The first spike should already include inexpensive tests for code that is
likely to survive:

- deterministic generation from a fixed seed;
- requested output shapes and channel/sample coordinates;
- navigation clamping and window boundaries;
- preservation of extrema during display reduction;
- cache correctness; and
- detection of known synthetic events in reduced display buffers.

Rendering benchmarks should be separate from ordinary unit tests. Later work
can add FieldTrip fixtures, annotation round trips, Qt interaction tests, and a
small headless rendering smoke test.

## Likely repository boundary

The Python application will probably deserve a new independent repository under
the canonical `F:/codex/te-pipeline/repos` workspace. This repository should
remain the historical MATLAB source and provenance reference. That decision
does not need to be made before the benchmark; the spike can be kept clearly
labelled until its value and destination are known.

## Open questions

- Which representative machine or machines define acceptable performance?
- What channel counts, sample rates, recording lengths, and trial sizes are most
  important in current studies?
- Is smooth continuous dragging required, or is rapid discrete window stepping
  sufficient for the first release?
- Should initial rendering preserve the exact visual density of ECKEEGVis, or
  optimise first for the stacked continuous chart?
- Which FieldTrip MATLAB versions and file encodings must be supported?
- How should unknown or mixed channel units be presented?
- What should force-good mean when several detector layers disagree?
- Should manual review be stored only in a sidecar, or also be exportable into a
  reviewed FieldTrip structure for existing MATLAB cleaning scripts?
- Which trial metrics should a later histogram/review-queue interface consume?

These questions are intentionally left open. The document should be revised as
the benchmark and subsequent small experiments provide evidence.

## Current proposed sequence

1. **Completed:** build the deterministic synthetic data generator and
   renderer-independent windowing/reduction code.
2. **Completed:** benchmark a stacked chart in PyQtGraph with CPU and OpenGL
   viewports.
3. **Completed:** record the results and select PyQtGraph/OpenGL provisionally.
4. **Completed provisionally:** introduce the internal channel, segment, series,
   montage-candidate, and artifact-state contract.
5. **Completed provisionally:** add common FieldTrip import while keeping layout
   resolution independent.
6. **Completed provisionally:** add embedded, explicit `.lay`, standard-label,
   hybrid, and grid layout resolution.
7. **Completed provisionally:** add batched scalp-positioned and grid waveform
   views alongside the chart view.
8. Validate imports against representative existing project files, including a
   MATLAB 7.3 file, unusual trial metadata, mixed sensor types, and the exact
   historical artifact variants encountered in current pipelines.
9. Exercise the current views interactively on larger real segmented and
   continuous recordings, then fix correctness or navigation issues exposed by
   real workflows.
10. Add a multiresolution extrema cache and measure continuous pan/zoom without
   synchronous readback, especially for long all-channel windows.
11. Refine artifact presentation as a separate overlay layer: legend, multiple
   simultaneous types, interpolation/cannot-interpolate priority, hover detail,
   and trial-level summaries.
12. Design the interval annotation model, manual force-good/force-bad semantics,
    persistence, and FieldTrip artifact compatibility.
13. Add averages, additional formats, and QC/review-queue integration as
    separate increments.

This sequence remains provisional. Real-data access, cache behaviour, the
topographic benchmark, or cross-platform testing may change the order or expose
the need for another rendering experiment.

## Reference implementations and documentation

- Existing prototype: `ECKEEGVis.m` in the separate `eeg_tools` repository
- Artifact contract: `eegAR_Detect.m` and `eegAR_UpdateArt.m`
- Incomplete component rewrite: `EEGVis_data.m`, `EEGVis_channel.m`, and
  `EEGVis_viewpane.m`
- [PyQtGraph PlotDataItem performance guidance](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotdataitem.html)
- [MNE Qt Browser](https://github.com/mne-tools/mne-qt-browser)
- [MNE supported data formats](https://mne.tools/stable/documentation/implementation.html#supported-data-formats)
- [VisPy Qt embedding example](https://vispy.org/gallery/scene/realtime_data/ex01_embedded_vispy.html)
- [VisPy LineVisual API](https://vispy.org/api/vispy.visuals.line.line.html)
- Initial implementation: this repository
- Initial measured results: `benchmarks/RESULTS_2026-08-06.md`
- Extraction record: `docs/PROVENANCE.md`

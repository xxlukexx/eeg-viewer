# Provenance

EEG Viewer began as a Python rendering and data-adapter prototype beside the
MATLAB `ECKEEGVis.m` code in the `eeg_tools` repository. It was promoted to a
standalone project after Version 1 proved useful and no longer depended on
`eeg_tools` at runtime.

The standalone history starts from a path-filtered form of:

- source repository: `xxlukexx/eeg_tools`;
- source commit: `d82f3dd` (`Add EEG Viewer 1.0`);
- source path: `benchmarks/python_rendering`;
- filtered root commit: `fbad798`.

The extraction retained the original implementation commit, then renamed the
internal Python package from `eegvis_benchmark` to `eeg_viewer`, moved the
benchmark report under `benchmarks/`, and adapted documentation and CI to the
standalone repository layout.

The MATLAB prototype remains historical design evidence, not a runtime or build
dependency. Real participant EEG files used for manual validation remain at
their original protected locations and are not part of this repository.
Generated fixtures and benchmark output under `results/` are ignored, apart
from the directory placeholder.

`docs/INITIAL_OUTLINE.md` is intentionally a working document rather than a
fixed specification. It records the reasoning that led to Version 1 and may be
revised as the viewer evolves.

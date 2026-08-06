from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6 import QtCore, QtTest, QtWidgets

from eeg_viewer.app import ViewerWindow
from eeg_viewer.fieldtrip import fieldtrip_from_mapping
from eeg_viewer.layout import resolve_layout
from eeg_viewer.model import DatasetViewSource


def test_window_switches_views_and_segments(tmp_path: Path, monkeypatch) -> None:
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset = fieldtrip_from_mapping(
        {
            "label": ["Fp1", "Cz", "O2"],
            "fsample": 100.0,
            "trial": [np.zeros((3, 100)), np.ones((3, 120))],
            "elec": {
                "label": ["Fp1", "Cz", "O2"],
                "chanpos": [[-0.5, 0.8, 0], [0, 0, 1], [0.4, -0.8, 0]],
            },
            "art": [[0, 1], [0, 0], [0, 0]],
            "art_type": "blink",
        }
    )
    source = DatasetViewSource(dataset)
    layout = resolve_layout(source.channel_labels, embedded=dataset.montage_candidate)
    window = ViewerWindow(
        source,
        dataset.metadata,
        resolved_layout=layout,
        initial_mode="chart",
        initial_window_seconds=0.5,
        visible_channels=3,
        amplitude_spacing_uv=100.0,
        opengl_requested=False,
        result_path=tmp_path / "smoke.json",
        automated_iterations=0,
        quit_after_automated=False,
    )

    window.show()
    application.processEvents()
    window.mode_control.setCurrentText("scalp")
    window.mode_control.setFocus()
    QtTest.QTest.keyClick(window.mode_control, QtCore.Qt.Key.Key_PageDown)
    assert window.trial_overview.current_segment == 1
    assert window.trial_overview.bad_channel_counts.tolist() == [0, 1]
    hover_height = window.hover_label.height()
    plot_height = window.plot.height()
    placement = window._current_placements[0]
    window._show_channel_hover(
        0,
        0,
        QtCore.QPointF(placement.center_x, placement.center_y),
        placement,
    )
    assert "blink" in window.hover_label.text()
    assert window.hover_item.isVisible()
    application.processEvents()
    assert window.hover_label.height() == hover_height
    assert window.plot.height() == plot_height
    original_spacing = window.state.amplitude_spacing_uv
    QtTest.QTest.keyClick(window.mode_control, QtCore.Qt.Key.Key_Equal)
    assert window.state.amplitude_spacing_uv < original_spacing
    QtTest.QTest.keyClick(window.mode_control, QtCore.Qt.Key.Key_Minus)
    assert np.isclose(window.state.amplitude_spacing_uv, original_spacing)
    QtTest.QTest.keyClick(
        window.mode_control,
        QtCore.Qt.Key.Key_Equal,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert window.state.amplitude_spacing_uv < original_spacing
    window.mode_control.setCurrentText("grid")
    metric = window.refresh("test")
    application.processEvents()

    assert source.segment_index == 1
    assert window.state.sample_count == 120
    assert metric.channels == 3
    assert metric.points > 0
    assert window.acceptDrops()
    assert window.open_button.text().startswith("Open EEG")
    shown_errors: list[str] = []
    monkeypatch.setattr(
        "eeg_viewer.app.prepare_viewer_input",
        lambda path: (_ for _ in ()).throw(ValueError("unsupported test data")),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda parent, title, message: shown_errors.append(message),
    )
    window._open_path(tmp_path / "bad.xyz")
    assert shown_errors and "unsupported test data" in shown_errors[0]
    window.close()

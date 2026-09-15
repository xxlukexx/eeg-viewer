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
            "time": [np.arange(100) / 100 - 0.1, np.arange(120) / 100 - 0.1],
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
    chart_point = window.view_box.mapViewToScene(QtCore.QPointF(0.1, 2.0))
    window._mouse_moved((chart_point,))
    assert window.hover_cursor.isVisible()
    assert window.hover_time_label.isVisible()
    assert window.hover_time_label.toPlainText() == "100.0 ms"
    assert np.isclose(window.hover_cursor.line().x1(), 0.1)
    assert np.isclose(window.hover_cursor.line().y1(), 1.54)
    assert np.isclose(window.hover_cursor.line().y2(), 2.46)
    moved_point = window.view_box.mapViewToScene(QtCore.QPointF(0.2, 2.0))
    window._mouse_moved((moved_point,))
    assert window.hover_time_label.toPlainText() == "200.0 ms"
    assert np.isclose(window.hover_cursor.line().x1(), 0.2)
    off_channel = window.view_box.mapViewToScene(QtCore.QPointF(0.2, -0.65))
    window._mouse_moved((off_channel,))
    assert not window.hover_cursor.isVisible()
    assert not window.hover_time_label.isVisible()
    window._mouse_moved((chart_point,))
    window.eventFilter(
        window.plot.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave)
    )
    assert not window.hover_cursor.isVisible()
    assert not window.hover_time_label.isVisible()
    assert window.average_alpha_slider.value() == 50
    assert window.average_alpha_slider.isVisible()
    assert window.average_alpha_value.text() == "0.50"
    assert window.average_curve.opts["pen"].color().alpha() == 128
    assert window.average_curve.opts["antialias"] is True
    assert window.average_curve.zValue() < window.curve.zValue()
    assert window.average_curve.opts["pen"].color() != window.curve.opts["pen"].color()
    assert window.trial_scale.isVisible()
    initial_time_scale = window.trial_scale.time_seconds
    initial_amplitude_scale = window.trial_scale.amplitude
    window.window_seconds.setValue(0.1)
    assert window.trial_scale.time_seconds < initial_time_scale
    window.window_seconds.setValue(0.5)
    window.amplitude.setValue(500)
    assert window.trial_scale.amplitude > initial_amplitude_scale
    window.amplitude.setValue(100)
    np.testing.assert_allclose(window._average_cache_data[:, :50], [[0] * 50, [0.5] * 50, [0.5] * 50])
    _, initial_average_y = window.average_curve.getData()
    assert np.isfinite(initial_average_y).any()
    window.average_alpha_slider.setValue(0)
    assert window.average_curve.opts["pen"].color().alpha() == 0
    assert window.average_curve.getData()[0].size == 0
    window.average_alpha_slider.setValue(75)
    assert window.average_curve.opts["pen"].color().alpha() == 191
    assert window.average_alpha_value.text() == "0.75"
    assert np.isfinite(window.average_curve.getData()[1]).any()
    restored_x = window.average_curve.getData()[0].copy()
    window.average_alpha_slider.setValue(25)
    assert window.average_curve.opts["pen"].color().alpha() == 64
    np.testing.assert_array_equal(
        window.average_curve.getData()[0], restored_x, strict=True
    )
    window.average_alpha_slider.setValue(7)
    assert window.average_curve.opts["pen"].color().alpha() == 18
    assert window.average_curve.opts["antialias"] is True
    window.average_alpha_slider.setValue(25)
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
    assert window.hover_cursor.isVisible()
    assert window.hover_time_label.isVisible()
    expected_tile_time = (
        source.time_start_seconds
        + (window.state.start_sample + (window._last_data.shape[1] - 1) / 2)
        / source.sample_rate_hz
    )
    assert window.hover_time_label.toPlainText() == f"{expected_tile_time * 1000:.1f} ms"
    assert np.isclose(window.hover_cursor.line().x1(), placement.center_x)
    assert np.isclose(
        window.hover_cursor.line().y2(), placement.center_y + placement.height / 2
    )
    outside_tiles = window.view_box.mapViewToScene(QtCore.QPointF(0.98, 0.98))
    window._mouse_moved((outside_tiles,))
    assert not window.hover_cursor.isVisible()
    assert not window.hover_time_label.isVisible()
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
    assert window.average_curve.opts["pen"].color().alpha() == 64
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


def test_average_pen_alpha_blends_in_rendered_pixels() -> None:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    plot = pg.PlotWidget(background=(11, 14, 20))
    plot.resize(360, 240)
    plot.setXRange(0, 1, padding=0)
    plot.setYRange(-1, 1, padding=0)
    curve = pg.PlotCurveItem(
        pen=pg.mkPen((167, 118, 217, 0), width=8), antialias=False
    )
    plot.addItem(curve)
    curve.setData(x=[0, 1], y=[0, 0])
    plot.show()
    application.processEvents()

    rendered_red = []
    for alpha in (0, 64, 128, 191, 255):
        curve.setPen(pg.mkPen((167, 118, 217, alpha), width=8))
        application.processEvents()
        image = plot.grab().toImage()
        x = image.width() // 2
        rendered_red.append(
            max(image.pixelColor(x, y).red() for y in range(40, image.height() - 40))
        )

    assert rendered_red[0] < rendered_red[1] < rendered_red[2]
    assert rendered_red[2] < rendered_red[3] < rendered_red[4]
    plot.close()

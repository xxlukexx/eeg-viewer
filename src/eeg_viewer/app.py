"""Interactive PySide6/PyQtGraph EEG viewer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import html
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import __version__
from .core import (
    ArraySignalSource,
    ViewState,
    fit_average_to_channel,
    reduce_ordered_extrema,
    stack_for_plot,
)
from .fieldtrip import read_fieldtrip
from .eeglab import read_eeglab
from .geometry import place_waveforms, placements_from_layout
from .layout import ResolvedLayout, grid_layout, resolve_layout
from .metrics import MetricRecorder, UpdateMetric
from .model import DatasetKind, DatasetViewSource
from .synthetic import SyntheticConfig, generate_recording
from .topomap import DualTopomapWidget
from .widgets import TrialOverviewWidget, TrialScaleWidget


LOGGER = logging.getLogger("eeg_viewer")
EEG_FILE_FILTER = (
    "EEG data (*.mat *.set *.fdt);;FieldTrip MATLAB (*.mat);;"
    "EEGLAB (*.set *.fdt);;All files (*)"
)


def _nice_scale_value(target: float) -> float:
    """Largest 1/2/5 decade step that does not exceed the target."""

    if not math.isfinite(target) or target <= 0:
        return 1.0
    decade = 10.0 ** math.floor(math.log10(target))
    for coefficient in (5.0, 2.0, 1.0):
        candidate = coefficient * decade
        if candidate <= target:
            return candidate
    return decade


@dataclass(frozen=True)
class PreparedViewerInput:
    source: DatasetViewSource
    layout: ResolvedLayout
    metadata: dict[str, Any]
    amplitude_spacing: float


class NavigationViewBox(pg.ViewBox):
    """A ViewBox whose wheel and drag gestures request data windows."""

    time_move_requested = QtCore.Signal(float)
    time_zoom_requested = QtCore.Signal(float, float)
    channel_move_requested = QtCore.Signal(int)

    def wheelEvent(self, event, axis=None):  # noqa: N802 - Qt/pyqtgraph API
        steps = float(event.delta()) / 120.0
        modifiers = event.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            anchor = 0.5
            width = max(1.0, float(self.width()))
            try:
                anchor = min(1.0, max(0.0, float(event.pos().x()) / width))
            except (AttributeError, TypeError):
                pass
            self.time_zoom_requested.emit(0.8**steps, anchor)
        elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.channel_move_requested.emit(-int(np.sign(steps)))
        else:
            self.time_move_requested.emit(-0.10 * steps)
        event.accept()

    def mouseDragEvent(self, event, axis=None):  # noqa: N802 - pyqtgraph API
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            width = max(1.0, float(self.width()))
            fraction = -float(event.pos().x() - event.lastPos().x()) / width
            self.time_move_requested.emit(fraction)
            event.accept()
            return
        event.ignore()


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        source: Any,
        metadata: dict[str, object],
        *,
        resolved_layout: ResolvedLayout,
        initial_mode: str,
        initial_window_seconds: float,
        visible_channels: int,
        amplitude_spacing_uv: float,
        opengl_requested: bool,
        result_path: Path,
        automated_iterations: int,
        quit_after_automated: bool,
        show_developer_controls: bool = False,
    ) -> None:
        super().__init__()
        self.source = source
        self._has_clean_average = (
            getattr(getattr(source, "dataset", None), "kind", None)
            == DatasetKind.SEGMENTED
        )
        self.metadata = metadata
        self.opengl_requested = opengl_requested
        self.show_developer_controls = show_developer_controls
        self.resolved_layout = resolved_layout
        self.grid_layout = grid_layout(source.channel_labels)
        self.mode = initial_mode
        self.state = ViewState(
            sample_count=source.sample_count,
            sample_rate_hz=source.sample_rate_hz,
            channel_count=source.channel_count,
            window_seconds=initial_window_seconds,
            visible_channels=visible_channels,
            amplitude_spacing_uv=amplitude_spacing_uv,
        )
        self.state.clamp()
        self.result_path = result_path
        self.automated_iterations = automated_iterations
        self.quit_after_automated = quit_after_automated
        self._updating_controls = False
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(lambda: self.refresh("resize"))
        self._overlay_items: list[pg.GraphicsObject] = []
        self._current_placements = ()
        self._current_visual_states = ()
        self._last_data = np.empty((0, 0), dtype=np.float32)
        self._last_channel_indices: tuple[int, ...] = ()
        self._average_cache_key: tuple[Any, ...] | None = None
        self._average_cache_data: np.ndarray | None = None
        self._average_cache_start = 0
        self._topomap_channel_indices = self._resolve_topomap_channels()
        self._topomap_cursor_sample: int | None = None
        self.topomap_panel: DualTopomapWidget | None = None
        self.setAcceptDrops(True)

        renderer = {
            "name": "PyQtGraph",
            "pyqtgraph_version": pg.__version__,
            "qt_version": QtCore.qVersion(),
            "pyside_version": QtCore.__version__,
            "opengl_requested": opengl_requested,
        }
        self.metrics = MetricRecorder(configuration=metadata, renderer=renderer)

        source_name = (
            Path(str(metadata["source_path"])).name
            if metadata.get("source_path")
            else "synthetic data"
        )
        self.setWindowTitle(f"EEG Viewer 1.0 — {source_name}")
        self.resize(1500, 920)
        self._build_ui()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self.refresh("initial")

        if automated_iterations > 0:
            QtCore.QTimer.singleShot(500, self.run_automated_benchmark)

    def _resolve_topomap_channels(self) -> tuple[int, ...]:
        """Return positioned EEG channels suitable for scalp interpolation."""

        positions = self.resolved_layout.scalp_positions
        dataset = getattr(self.source, "dataset", None)
        if not self._has_clean_average or positions is None or dataset is None:
            return ()
        return tuple(
            index
            for index, channel in enumerate(dataset.channels)
            if self.resolved_layout.matched[index]
            and np.isfinite(positions[index]).all()
            and channel.channel_type.casefold() == "eeg"
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        heading = QtWidgets.QLabel(
            "EEG viewer foundation — mouse wheel scrolls time, "
            "Ctrl+wheel zooms, Shift+wheel scrolls channels, left-drag pans"
        )
        heading.setWordWrap(True)
        outer.addWidget(heading)

        source_name = Path(str(self.metadata.get("source_path", "synthetic data"))).name
        kind = str(self.metadata.get("dataset_kind", "synthetic"))
        layout_source = str(self.metadata.get("layout_source", self.resolved_layout.source))
        summary = QtWidgets.QLabel(
            f"{source_name}  ·  {kind}  ·  {self.source.channel_count} channels  ·  "
            f"{getattr(self.source, 'segment_count', 1)} segment(s)  ·  "
            f"{self.source.sample_rate_hz:g} Hz  ·  layout: {layout_source}"
        )
        summary.setStyleSheet("color: #9caabd;")
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(summary)

        artifact_names = [
            layer.name
            for layer in getattr(getattr(self.source, "dataset", None), "artifacts", ()).layers
        ] if getattr(self.source, "dataset", None) is not None else []
        artifact_text = ", ".join(artifact_names) if artifact_names else "none imported"
        average_legend = (
            '<span style="color:#a776d9">━</span> clean mean (rescaled) &nbsp;&nbsp; '
            if self._has_clean_average else ""
        )
        legend = QtWidgets.QLabel(
            '<span style="color:#f0aa3c">■</span> artifact &nbsp;&nbsp; '
            '<span style="color:#46a0f5">■</span> interpolated &nbsp;&nbsp; '
            '<span style="color:#f54b5a">■</span> cannot interpolate &nbsp;&nbsp; '
            f'{average_legend}'
            f'<span style="color:#8795a8">layers: {artifact_text}</span>'
        )
        legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        outer.addWidget(legend)

        controls = QtWidgets.QHBoxLayout()
        outer.addLayout(controls)

        self.open_button = QtWidgets.QPushButton("Open EEG…")
        self.open_button.clicked.connect(self._open_file_dialog)
        controls.addWidget(self.open_button)

        controls.addWidget(QtWidgets.QLabel("View"))
        self.mode_control = QtWidgets.QComboBox()
        self.mode_control.addItems(("chart", "scalp", "grid"))
        self.mode_control.setCurrentText(self.mode)
        self.mode_control.currentTextChanged.connect(self._mode_changed)
        controls.addWidget(self.mode_control)

        self.topomap_toggle = QtWidgets.QToolButton()
        self.topomap_toggle.setText("Scalp maps")
        self.topomap_toggle.setCheckable(True)
        self.topomap_toggle.setChecked(len(self._topomap_channel_indices) >= 3)
        self.topomap_toggle.setEnabled(len(self._topomap_channel_indices) >= 3)
        self.topomap_toggle.setToolTip("Show or hide the dual scalp-map panel")
        self.topomap_toggle.toggled.connect(self._topomap_visibility_changed)
        controls.addWidget(self.topomap_toggle)

        segment_count = int(getattr(self.source, "segment_count", 1))
        controls.addWidget(QtWidgets.QLabel("Segment"))
        self.segment_control = QtWidgets.QSpinBox()
        self.segment_control.setRange(1, segment_count)
        self.segment_control.setEnabled(segment_count > 1)
        self.segment_control.valueChanged.connect(self._segment_changed)
        controls.addWidget(self.segment_control)

        self.average_alpha_label = QtWidgets.QLabel("Clean-trial average α")
        self.average_alpha_label.setVisible(self._has_clean_average)
        controls.addWidget(self.average_alpha_label)
        self.average_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.average_alpha_slider.setRange(0, 100)
        self.average_alpha_slider.setSingleStep(5)
        self.average_alpha_slider.setPageStep(10)
        self.average_alpha_slider.setFixedWidth(110)
        self.average_alpha_slider.setToolTip(
            "Opacity of the per-channel clean-trial mean; 0 hides it"
        )
        self.average_alpha_slider.setValue(50)
        self.average_alpha_slider.setVisible(self._has_clean_average)
        self.average_alpha_slider.valueChanged.connect(self._average_alpha_changed)
        controls.addWidget(self.average_alpha_slider)
        self.average_alpha_value = QtWidgets.QLabel("0.50")
        self.average_alpha_value.setMinimumWidth(30)
        self.average_alpha_value.setVisible(self._has_clean_average)
        controls.addWidget(self.average_alpha_value)

        series_count = int(getattr(self.source, "series_count", 1))
        controls.addWidget(QtWidgets.QLabel("Series"))
        self.series_control = QtWidgets.QComboBox()
        dataset = getattr(self.source, "dataset", None)
        series_labels = (
            dataset.signal.series_labels
            if dataset is not None
            else tuple(f"series {index + 1}" for index in range(series_count))
        )
        self.series_control.addItems(series_labels)
        self.series_control.setEnabled(series_count > 1)
        self.series_control.currentIndexChanged.connect(self._series_changed)
        controls.addWidget(self.series_control)

        controls.addWidget(QtWidgets.QLabel("Window"))
        self.window_seconds = QtWidgets.QDoubleSpinBox()
        self.window_seconds.setRange(0.05, self.source.duration_seconds)
        self.window_seconds.setDecimals(2)
        self.window_seconds.setSuffix(" s")
        self.window_seconds.setValue(self.state.window_seconds)
        self.window_seconds.valueChanged.connect(self._window_seconds_changed)
        controls.addWidget(self.window_seconds)

        controls.addWidget(QtWidgets.QLabel("Visible channels"))
        self.visible_channels = QtWidgets.QSpinBox()
        self.visible_channels.setRange(1, self.source.channel_count)
        self.visible_channels.setValue(self.state.visible_channels)
        self.visible_channels.valueChanged.connect(self._visible_channels_changed)
        controls.addWidget(self.visible_channels)

        controls.addWidget(QtWidgets.QLabel("Trace spacing"))
        self.amplitude = QtWidgets.QDoubleSpinBox()
        self.amplitude.setRange(
            max(1e-12, self.state.amplitude_spacing_uv * 1e-6),
            max(2_000.0, self.state.amplitude_spacing_uv * 1e6),
        )
        self.amplitude.setDecimals(9 if self.state.amplitude_spacing_uv < 1 else 3)
        self.amplitude.setSingleStep(self.state.amplitude_spacing_uv * 0.1)
        self.amplitude.setSuffix(f" {self.source.unit}")
        self.amplitude.setValue(self.state.amplitude_spacing_uv)
        self.amplitude.valueChanged.connect(self._amplitude_changed)
        controls.addWidget(self.amplitude)

        self.run_button = QtWidgets.QPushButton("Run 40-step benchmark")
        self.run_button.clicked.connect(lambda: self.run_automated_benchmark(40))
        controls.addWidget(self.run_button)

        self.save_button = QtWidgets.QPushButton("Save timings")
        self.save_button.clicked.connect(self.save_results)
        controls.addWidget(self.save_button)
        controls.addStretch(1)

        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setMinimumWidth(330)
        self.metrics_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        controls.addWidget(self.metrics_label)
        self.run_button.setVisible(self.show_developer_controls)
        self.save_button.setVisible(self.show_developer_controls)
        self.metrics_label.setVisible(self.show_developer_controls)

        self.view_box = NavigationViewBox(enableMenu=False)
        self.view_box.setMouseEnabled(x=False, y=False)
        self.view_box.time_move_requested.connect(self._move_time)
        self.view_box.time_zoom_requested.connect(self._zoom_time)
        self.view_box.channel_move_requested.connect(self._move_channels)

        self.plot = pg.PlotWidget(viewBox=self.view_box, background=(11, 14, 20))
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.showGrid(x=True, y=False, alpha=0.16)
        self.plot.getAxis("left").setWidth(86)
        self.curve = pg.PlotCurveItem(
            pen=pg.mkPen((110, 220, 190), width=1),
            antialias=False,
            connect="finite",
        )
        self.average_curve = pg.PlotCurveItem(
            pen=pg.mkPen((167, 118, 217, 128), width=2.7),
            # PyQtGraph's OpenGL curve path enables GL_BLEND for antialiased
            # lines. Without it, every nonzero pen alpha renders fully opaque.
            antialias=True,
            connect="finite",
        )
        self.average_curve.setZValue(-0.5)
        self.plot.addItem(self.average_curve)
        self.plot.addItem(self.curve)
        self.trial_scale = TrialScaleWidget(self.plot)
        self.trial_scale.raise_()
        self.hover_item = QtWidgets.QGraphicsRectItem()
        self.hover_item.setPen(pg.mkPen((190, 220, 255), width=1.3))
        self.hover_item.setBrush(pg.mkBrush(90, 135, 180, 35))
        self.hover_item.setZValue(-1)
        self.hover_item.hide()
        self.plot.addItem(self.hover_item)
        self.hover_cursor = QtWidgets.QGraphicsLineItem()
        self.hover_cursor.setPen(pg.mkPen((247, 194, 93, 220), width=1.2))
        self.hover_cursor.setZValue(3)
        self.hover_cursor.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.hover_cursor.hide()
        self.plot.addItem(self.hover_cursor)
        self.hover_time_label = pg.TextItem(
            color=(247, 216, 159),
            fill=(19, 26, 36, 225),
            border=pg.mkPen((96, 105, 119), width=0.7),
            anchor=(0, 0),
        )
        font = QtGui.QFont()
        font.setPointSize(8)
        self.hover_time_label.textItem.document().setDocumentMargin(1)
        self.hover_time_label.setFont(font)
        self.hover_time_label.setZValue(4)
        self.hover_time_label.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.hover_time_label.hide()
        self.plot.addItem(self.hover_time_label)
        self._hover_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._mouse_moved,
        )

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.plot, 1)
        self.channel_scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.channel_scroll.setRange(0, self.state.max_first_channel)
        self.channel_scroll.setPageStep(self.state.visible_channels)
        self.channel_scroll.valueChanged.connect(self._channel_scroll_changed)
        plot_row.addWidget(self.channel_scroll)
        if len(self._topomap_channel_indices) >= 3:
            scalp_positions = self.resolved_layout.scalp_positions
            assert scalp_positions is not None
            indices = np.asarray(self._topomap_channel_indices, dtype=int)
            positions = 2.0 * scalp_positions[indices] - 1.0
            labels = tuple(self.source.channel_labels[index] for index in indices)
            self.topomap_panel = DualTopomapWidget(
                positions,
                labels,
                self._topomap_channel_indices,
                self.source.sample_rate_hz,
            )
            self.topomap_panel.window_ms_changed.connect(
                self._topomap_window_changed
            )
            self.topomap_panel.setVisible(self.topomap_toggle.isChecked())
            plot_row.addWidget(self.topomap_panel)
        outer.addLayout(plot_row, 1)

        bad_counts, layer_counts = self._trial_overview_data()
        overview_row = QtWidgets.QVBoxLayout()
        overview_row.setSpacing(1)
        self.trial_overview_label = QtWidgets.QLabel(
            "Trials — bar height is the number of channels marked bad; click or drag to navigate"
        )
        self.trial_overview_label.setStyleSheet("color: #8795a8;")
        overview_row.addWidget(self.trial_overview_label)
        self.trial_overview = TrialOverviewWidget(
            bad_counts,
            self.source.channel_count,
            layer_counts,
        )
        self.trial_overview.segment_selected.connect(self._overview_segment_selected)
        overview_row.addWidget(self.trial_overview)
        overview_visible = int(getattr(self.source, "segment_count", 1)) > 1
        self.trial_overview_label.setVisible(overview_visible)
        self.trial_overview.setVisible(overview_visible)
        outer.addLayout(overview_row)

        self.time_scroll = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        self.time_scroll.setRange(0, 100_000)
        self.time_scroll.setPageStep(10_000)
        self.time_scroll.valueChanged.connect(self._time_scroll_changed)
        outer.addWidget(self.time_scroll)

        self.hover_label = QtWidgets.QLabel(
            "Hover over a channel to inspect signal and artifact details."
        )
        hover_height = self.hover_label.fontMetrics().lineSpacing() * 2 + 12
        self.hover_label.setFixedHeight(hover_height)
        self.hover_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.hover_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.hover_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.hover_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.hover_label.setStyleSheet(
            "background:#111721; color:#d7e0eb; padding:4px 7px; border:1px solid #2c3746;"
        )
        outer.addWidget(self.hover_label)

        self.status = QtWidgets.QLabel()
        self.status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        outer.addWidget(self.status)
        self.setCentralWidget(central)
        self._update_mode_chrome()

    def _trial_overview_data(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        segment_count = int(getattr(self.source, "segment_count", 1))
        union = np.zeros((self.source.channel_count, segment_count), dtype=bool)
        layer_counts: dict[str, np.ndarray] = {}
        dataset = getattr(self.source, "dataset", None)
        if dataset is not None:
            for layer in dataset.artifacts.layers:
                union |= layer.mask
                layer_counts[layer.name] = layer.mask.sum(axis=0, dtype=np.int32)
            if dataset.artifacts.cannot_interpolate is not None:
                union |= dataset.artifacts.cannot_interpolate
        return union.sum(axis=0, dtype=np.int32), layer_counts

    @QtCore.Slot(int)
    def _overview_segment_selected(self, index: int) -> None:
        self.segment_control.setValue(index + 1)

    @QtCore.Slot()
    def _open_file_dialog(self) -> None:
        current = self.metadata.get("source_path")
        start = str(Path(str(current)).parent) if current else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open EEG data",
            start,
            EEG_FILE_FILTER,
        )
        if path:
            self._open_path(Path(path))

    def _open_path(self, path: Path) -> None:
        self.status.setText(f"Opening {path.name}…")
        self.open_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            prepared = prepare_viewer_input(path)
        except Exception as error:
            LOGGER.exception("Could not open %s", path)
            QtWidgets.QMessageBox.critical(
                self,
                "Could not open EEG data",
                f"{path}\n\n{type(error).__name__}: {error}",
            )
            self.status.setText(f"Could not open {path.name}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.open_button.setEnabled(True)

        new_window = ViewerWindow(
            prepared.source,
            prepared.metadata,
            resolved_layout=prepared.layout,
            initial_mode=self.mode,
            initial_window_seconds=min(
                self.state.window_seconds, prepared.source.duration_seconds
            ),
            visible_channels=min(self.state.visible_channels, prepared.source.channel_count),
            amplitude_spacing_uv=prepared.amplitude_spacing,
            opengl_requested=self.opengl_requested,
            result_path=_default_result_path().resolve(),
            automated_iterations=0,
            quit_after_automated=False,
            show_developer_controls=self.show_developer_controls,
        )
        application = QtWidgets.QApplication.instance()
        if application is not None:
            windows = getattr(application, "_eeg_viewer_windows", [])
            windows.append(new_window)
            application._eeg_viewer_windows = windows
        new_window.show()
        self.close()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if any(path.suffix.casefold() in {".mat", ".set", ".fdt"} for path in paths):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        supported = [path for path in paths if path.suffix.casefold() in {".mat", ".set", ".fdt"}]
        if supported:
            event.acceptProposedAction()
            self._open_path(supported[0])
            return
        event.ignore()

    def _mode_changed(self, value: str) -> None:
        self.mode = value
        self.state.first_channel = 0
        self._update_mode_chrome()
        self.refresh("view-mode")

    def _average_alpha_changed(self, value: int) -> None:
        was_visible = self.average_curve.opts["pen"].color().alpha() > 0
        alpha = value / 100.0
        self.average_curve.setPen(
            pg.mkPen((167, 118, 217, round(alpha * 255)), width=2.7)
        )
        self.average_alpha_value.setText(f"{alpha:.2f}")
        if was_visible != (value > 0):
            self.refresh("average-alpha")

    @QtCore.Slot(bool)
    def _topomap_visibility_changed(self, visible: bool) -> None:
        if self.topomap_panel is None:
            return
        self.topomap_panel.setVisible(visible)
        self._average_cache_key = None
        self.refresh("topomap-visibility")

    @QtCore.Slot(float)
    def _topomap_window_changed(self, _value: float) -> None:
        self._average_cache_key = None
        self.refresh("topomap-window")

    def _segment_changed(self, value: int) -> None:
        if self._updating_controls or not hasattr(self.source, "select_segment"):
            return
        self.source.select_segment(value - 1)
        self.state.sample_count = self.source.sample_count
        self.state.start_sample = 0
        self._topomap_cursor_sample = None
        self.state.clamp()
        self.window_seconds.setMaximum(self.source.duration_seconds)
        self._rebuild_static_overlays()
        self.refresh("segment")

    def _series_changed(self, value: int) -> None:
        if self._updating_controls or not hasattr(self.source, "select_series"):
            return
        self.source.select_series(value)
        self.refresh("series")

    def _step_segment(self, amount: int) -> None:
        segment_count = int(getattr(self.source, "segment_count", 1))
        if segment_count > 1:
            target = max(1, min(segment_count, self.segment_control.value() + amount))
            self.segment_control.setValue(target)
        else:
            self.state.move_time_fraction(float(amount))
            self.refresh("page-time")

    def _zoom_vertical(self, factor: float) -> None:
        self.state.amplitude_spacing_uv *= factor
        self.state.clamp()
        self.refresh("amplitude-keyboard")

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if (
            watched is self.plot.viewport()
            and event.type() == QtCore.QEvent.Type.Leave
        ):
            self._hide_channel_hover(reset_label=True)
        is_ours = watched is self or (
            isinstance(watched, QtWidgets.QWidget) and self.isAncestorOf(watched)
        )
        if is_ours and event.type() == QtCore.QEvent.Type.KeyPress:
            key_event = event
            key = key_event.key()
            if key == QtCore.Qt.Key.Key_PageUp:
                self._step_segment(-1)
                return True
            if key == QtCore.Qt.Key.Key_PageDown:
                self._step_segment(1)
                return True
            if key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
                self._zoom_vertical(0.8)
                return True
            if key in (QtCore.Qt.Key.Key_Minus, QtCore.Qt.Key.Key_Underscore):
                self._zoom_vertical(1.25)
                return True
        return super().eventFilter(watched, event)

    def _update_mode_chrome(self) -> None:
        chart = self.mode == "chart"
        # Preserve recognisable scalp geometry without reducing waveform tiles
        # to a tiny central square on modern wide displays.
        self.view_box.setAspectLocked(self.mode == "scalp", ratio=1.8)
        self.visible_channels.setEnabled(chart)
        self.channel_scroll.setVisible(chart)
        self.plot.getAxis("left").setVisible(chart)
        self.plot.getAxis("bottom").setVisible(chart)
        self.plot.showGrid(x=chart, y=False, alpha=0.16)
        self._rebuild_static_overlays()

    def _clear_static_overlays(self) -> None:
        for item in self._overlay_items:
            self.plot.removeItem(item)
        self._overlay_items.clear()

    def _active_layout(self) -> ResolvedLayout:
        return self.grid_layout if self.mode == "grid" else self.resolved_layout

    def _visual_states(self) -> tuple[Any, ...]:
        dataset = getattr(self.source, "dataset", None)
        if dataset is None:
            return ()
        return dataset.artifacts.visual_states(
            getattr(self.source, "segment_index", 0), self.source.channel_count
        )

    def _rebuild_static_overlays(self) -> None:
        self._clear_static_overlays()
        self._current_placements = ()
        self._current_visual_states = self._visual_states()
        if self.mode == "chart" or not hasattr(self, "plot"):
            return
        placements = placements_from_layout(self._active_layout())
        self._current_placements = placements
        states = self._current_visual_states
        paths: dict[str, QtGui.QPainterPath] = {
            "normal": QtGui.QPainterPath(),
            "artifact": QtGui.QPainterPath(),
            "interpolated": QtGui.QPainterPath(),
            "cannot": QtGui.QPainterPath(),
        }
        for placement in placements:
            state = states[placement.channel_index] if states else None
            category = "normal"
            if state is not None:
                if state.cannot_interpolate:
                    category = "cannot"
                elif state.interpolated:
                    category = "interpolated"
                elif state.artifact_types:
                    category = "artifact"
            paths[category].addRect(
                placement.center_x - placement.width / 2,
                placement.center_y - placement.height / 2,
                placement.width,
                placement.height,
            )
            label = pg.TextItem(
                placement.label,
                color=(170, 180, 195),
                anchor=(0.5, 1.0),
            )
            label.setPos(
                placement.center_x,
                placement.center_y - placement.height / 2,
            )
            label.setFont(QtGui.QFont("Sans Serif", 6))
            self.plot.addItem(label)
            self._overlay_items.append(label)
        pens = {
            "normal": pg.mkPen((62, 72, 86), width=1),
            "artifact": pg.mkPen((240, 170, 60), width=2),
            "interpolated": pg.mkPen((70, 160, 245), width=2),
            "cannot": pg.mkPen((245, 75, 90), width=2),
        }
        for category, path in paths.items():
            item = QtWidgets.QGraphicsPathItem(path)
            item.setPen(pens[category])
            self.plot.addItem(item)
            self._overlay_items.append(item)
        if self.mode == "scalp" and self._active_layout().kind in ("sensor", "hybrid"):
            hybrid = self._active_layout().kind == "hybrid"
            head_width = 0.73 if hybrid else 0.95
            head = QtWidgets.QGraphicsEllipseItem(0.025, 0.025, head_width, 0.95)
            head.setPen(pg.mkPen((75, 88, 105), width=1.5))
            head.setZValue(-2)
            self.plot.addItem(head)
            self._overlay_items.append(head)
            centre_x = 0.025 + head_width / 2
            nose = QtGui.QPainterPath()
            nose.moveTo(centre_x - 0.035, 0.965)
            nose.lineTo(centre_x, 1.015)
            nose.lineTo(centre_x + 0.035, 0.965)
            nose_item = QtWidgets.QGraphicsPathItem(nose)
            nose_item.setPen(pg.mkPen((100, 116, 136), width=1.5))
            nose_item.setZValue(-2)
            self.plot.addItem(nose_item)
            self._overlay_items.append(nose_item)

    def _topomap_window_samples(self) -> int:
        if self.topomap_panel is None:
            return 1
        return max(
            1,
            int(
                round(
                    self.topomap_panel.window_ms.value()
                    * self.source.sample_rate_hz
                    / 1_000.0
                )
            ),
        )

    def _topomap_window_bounds(self, centre_sample: int) -> tuple[int, int]:
        count = min(self.source.sample_count, self._topomap_window_samples())
        start = int(centre_sample) - (count - 1) // 2
        stop = start + count
        if start < 0:
            stop -= start
            start = 0
        if stop > self.source.sample_count:
            start -= stop - self.source.sample_count
            stop = self.source.sample_count
        return max(0, start), max(0, stop)

    @staticmethod
    def _mean_rows(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        counts = finite.sum(axis=1)
        sums = np.where(finite, values, 0.0).sum(axis=1, dtype=np.float64)
        result = np.full(values.shape[0], np.nan, dtype=np.float64)
        np.divide(sums, counts, out=result, where=counts > 0)
        return result

    def _update_topomaps(self) -> None:
        if self.topomap_panel is None or self.topomap_panel.isHidden():
            return

        visible_stop = max(self.state.start_sample + 1, self.state.stop_sample)
        centre = self._topomap_cursor_sample
        if centre is None or not self.state.start_sample <= centre < visible_stop:
            centre = min(
                self.source.sample_count - 1,
                self.state.start_sample
                + max(0, self.state.stop_sample - self.state.start_sample - 1) // 2,
            )
            self._topomap_cursor_sample = centre

        start, stop = self._topomap_window_bounds(centre)
        current = np.asarray(
            self.source.read_baseline_corrected(slice(None), start, stop)
        )
        if (
            self._average_cache_data is not None
            and self._average_cache_start <= start
            and stop
            <= self._average_cache_start + self._average_cache_data.shape[1]
        ):
            local_start = start - self._average_cache_start
            local_stop = stop - self._average_cache_start
            average = self._average_cache_data[:, local_start:local_stop]
        else:
            average = self.source.read_clean_average(slice(None), start, stop)

        indices = self.topomap_panel.channel_indices
        clean_values = self._mean_rows(np.asarray(average))[indices]
        current_values = self._mean_rows(current)[indices]
        time_seconds = (
            float(getattr(self.source, "time_start_seconds", 0.0))
            + centre / self.source.sample_rate_hz
        )
        self.topomap_panel.set_maps(
            clean_values,
            current_values,
            trial_number=int(getattr(self.source, "segment_index", 0)) + 1,
            time_seconds=time_seconds,
            window_seconds=(stop - start) / self.source.sample_rate_hz,
            unit=str(self.source.unit),
        )

    def _mouse_moved(self, event: tuple[Any, ...]) -> None:
        if not event:
            return
        position = event[0]
        if isinstance(position, (tuple, list)):
            position = position[0]
        if not self.view_box.sceneBoundingRect().contains(position):
            self._hide_channel_hover(reset_label=True)
            return
        point = self.view_box.mapSceneToView(position)
        channel_index: int | None = None
        data_row: int | None = None
        placement = None
        if self.mode == "chart":
            visible_count = len(self._last_channel_indices)
            row = int(round((visible_count - 1) - float(point.y())))
            if 0 <= row < visible_count and abs((visible_count - 1 - row) - point.y()) <= 0.5:
                data_row = row
                channel_index = self._last_channel_indices[row]
        else:
            matches = [
                candidate
                for candidate in self._current_placements
                if abs(float(point.x()) - candidate.center_x) <= candidate.width / 2
                and abs(float(point.y()) - candidate.center_y) <= candidate.height / 2
            ]
            if matches:
                placement = min(
                    matches,
                    key=lambda candidate: (
                        float(point.x()) - candidate.center_x
                    ) ** 2
                    + (float(point.y()) - candidate.center_y) ** 2,
                )
                channel_index = placement.channel_index
                try:
                    data_row = self._last_channel_indices.index(channel_index)
                except ValueError:
                    data_row = None

        if channel_index is None or data_row is None:
            self._hide_channel_hover(reset_label=True)
            return
        self._show_channel_hover(channel_index, data_row, point, placement)

    def _hide_channel_hover(self, *, reset_label: bool) -> None:
        self.hover_item.hide()
        self.hover_cursor.hide()
        self.hover_time_label.hide()
        if reset_label:
            self.hover_label.setText(
                "Hover over a channel to inspect signal and artifact details."
            )

    def _show_channel_hover(
        self,
        channel_index: int,
        data_row: int,
        point: QtCore.QPointF,
        placement: Any,
    ) -> None:
        values = self._last_data[data_row]
        finite = values[np.isfinite(values)]
        if finite.size:
            signal_text = (
                f"window min {float(finite.min()):.4g}, max {float(finite.max()):.4g}, "
                f"p-p {float(np.ptp(finite)):.4g} {html.escape(self.source.unit)}"
            )
        else:
            signal_text = "window contains no finite samples"

        dataset = getattr(self.source, "dataset", None)
        if dataset is not None:
            channel = dataset.channels[channel_index]
            channel_text = (
                f"type {html.escape(channel.channel_type)}, declared unit "
                f"{html.escape(channel.unit)}"
            )
        else:
            channel = None
            channel_text = f"display unit {html.escape(self.source.unit)}"

        state = (
            self._current_visual_states[channel_index]
            if self._current_visual_states
            else None
        )
        review_parts: list[str] = []
        if state is not None:
            if state.artifact_types:
                review_parts.append("artifact: " + ", ".join(map(html.escape, state.artifact_types)))
            if state.interpolated:
                review_parts.append("interpolated")
            if state.cannot_interpolate:
                review_parts.append("cannot interpolate")
        review_text = "; ".join(review_parts) if review_parts else "no imported flags"

        if placement is None:
            time_seconds = float(point.x())
            left = float(getattr(self.source, "time_start_seconds", 0.0)) + self.state.start_sample / self.source.sample_rate_hz
            right = float(getattr(self.source, "time_start_seconds", 0.0)) + self.state.stop_sample / self.source.sample_rate_hz
            if not left <= time_seconds <= right:
                self._hide_channel_hover(reset_label=True)
                return
            baseline = len(self._last_channel_indices) - 1 - data_row
            bottom, top = baseline - 0.46, baseline + 0.46
            self.hover_item.setRect(left, bottom, max(1e-12, right - left), top - bottom)
        else:
            left = placement.center_x - placement.width / 2
            right = placement.center_x + placement.width / 2
            fraction = (
                (float(point.x()) - left)
                / max(1e-12, placement.width)
            )
            sample_position = self.state.start_sample + (
                min(1.0, max(0.0, fraction)) * max(0, values.size - 1)
            )
            time_seconds = (
                float(getattr(self.source, "time_start_seconds", 0.0))
                + sample_position / self.source.sample_rate_hz
            )
            bottom = placement.center_y - placement.height / 2
            top = placement.center_y + placement.height / 2
            self.hover_item.setRect(
                left,
                bottom,
                placement.width,
                placement.height,
            )
        self.hover_cursor.setLine(float(point.x()), bottom, float(point.x()), top)
        self.hover_time_label.setText(f"{time_seconds * 1_000:.1f} ms")
        self.hover_time_label.setAnchor(
            (1, 0) if float(point.x()) > left + 0.85 * (right - left) else (0, 0)
        )
        self.hover_time_label.setPos(float(point.x()), top)
        local_sample = int(
            round(
                (time_seconds - float(getattr(self.source, "time_start_seconds", 0.0)))
                * self.source.sample_rate_hz
            )
        ) - self.state.start_sample
        if 0 <= local_sample < values.size:
            self._topomap_cursor_sample = min(
                self.source.sample_count - 1,
                self.state.start_sample + local_sample,
            )
            self._update_topomaps()
        cursor_text = f"t {time_seconds:.4f} s"
        if 0 <= local_sample < values.size and np.isfinite(values[local_sample]):
            cursor_text += f", value {float(values[local_sample]):.4g} {html.escape(self.source.unit)}"

        trial_index = int(getattr(self.source, "segment_index", 0)) + 1
        label = html.escape(self.source.channel_labels[channel_index])
        self.hover_label.setText(
            f"<b>{label}</b> (channel {channel_index + 1}) &nbsp;·&nbsp; trial {trial_index} "
            f"&nbsp;·&nbsp; {channel_text} &nbsp;·&nbsp; {review_text}<br>"
            f"{cursor_text} &nbsp;·&nbsp; {signal_text}"
        )
        self.hover_item.show()
        self.hover_cursor.show()
        self.hover_time_label.show()

    def _time_scroll_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self.state.start_sample = int(
            round(value / 100_000.0 * self.state.max_start_sample)
        )
        self.state.clamp()
        self.refresh("time-scrollbar")

    def _channel_scroll_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self.state.first_channel = value
        self.state.clamp()
        self.refresh("channel-scrollbar")

    def _window_seconds_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        old_window = self.state.window_seconds
        self.state.zoom_time(value / old_window, 0.5)
        self.refresh("window-control")

    def _visible_channels_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self.state.visible_channels = value
        self.state.clamp()
        self.refresh("visible-channels")

    def _amplitude_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self.state.amplitude_spacing_uv = value
        self.state.clamp()
        self.refresh("amplitude")

    @QtCore.Slot(float)
    def _move_time(self, fraction: float) -> None:
        self.state.move_time_fraction(fraction)
        self.refresh("time-gesture")

    @QtCore.Slot(float, float)
    def _zoom_time(self, factor: float, anchor: float) -> None:
        self.state.zoom_time(factor, anchor)
        self.refresh("zoom-gesture")

    @QtCore.Slot(int)
    def _move_channels(self, amount: int) -> None:
        self.state.move_channels(amount)
        self.refresh("channel-gesture")

    def _sync_controls(self) -> None:
        self._updating_controls = True
        try:
            self.window_seconds.setValue(self.state.window_seconds)
            self.visible_channels.setValue(self.state.visible_channels)
            self.amplitude.setValue(self.state.amplitude_spacing_uv)
            self.channel_scroll.setRange(0, self.state.max_first_channel)
            self.channel_scroll.setPageStep(self.state.visible_channels)
            self.channel_scroll.setValue(self.state.first_channel)
            value = 0
            if self.state.max_start_sample:
                value = int(
                    round(
                        self.state.start_sample
                        / self.state.max_start_sample
                        * 100_000
                    )
                )
            self.time_scroll.setValue(value)
            if hasattr(self.source, "segment_index"):
                self.segment_control.setValue(self.source.segment_index + 1)
                self.trial_overview.set_current_segment(self.source.segment_index)
            if hasattr(self.source, "series_index"):
                self.series_control.setCurrentIndex(self.source.series_index)
        finally:
            self._updating_controls = False

    def refresh(self, reason: str, *, record: bool = True) -> UpdateMetric:
        self._hide_channel_hover(reset_label=False)
        self.state.clamp()
        geometry_started = time.perf_counter()
        channel_slice = self.state.channel_slice if self.mode == "chart" else slice(None)
        data = self.source.read(
            channel_slice,
            self.state.start_sample,
            self.state.stop_sample,
        )
        self._last_data = np.asarray(data)
        if self.mode == "chart":
            self._last_channel_indices = tuple(
                range(self.state.first_channel, self.state.first_channel + data.shape[0])
            )
        else:
            self._last_channel_indices = tuple(range(data.shape[0]))
        # Approximately one time bin per horizontal pixel, bounded for resize
        # spikes. Each bin emits two ordered extrema.
        time_bins = max(64, min(4_096, int(max(1, self.plot.width()) * 1.15)))
        reduced = reduce_ordered_extrema(
            data,
            start_sample=self.state.start_sample,
            max_time_bins=time_bins,
        )
        average_x = average_y = np.empty(0, dtype=np.float32)
        average_reduced = None
        needs_average = self._has_clean_average and (
            self.average_alpha_slider.value() > 0
            or (
                self.topomap_panel is not None
                and not self.topomap_panel.isHidden()
            )
        )
        if needs_average:
            padding = (
                self._topomap_window_samples() // 2
                if self.topomap_panel is not None
                and not self.topomap_panel.isHidden()
                else 0
            )
            cache_start = max(0, self.state.start_sample - padding)
            cache_stop = min(self.source.sample_count, self.state.stop_sample + padding)
            cache_key = (
                self.source.series_index,
                self.source.segment_index,
                cache_start,
                cache_stop,
            )
            if cache_key != self._average_cache_key:
                self._average_cache_data = self.source.read_clean_average(
                    slice(None),
                    cache_start,
                    cache_stop,
                )
                self._average_cache_key = cache_key
                self._average_cache_start = cache_start
            if self.average_alpha_slider.value() > 0:
                assert self._average_cache_data is not None
                view_start = self.state.start_sample - self._average_cache_start
                view_stop = view_start + data.shape[1]
                average_for_display = self._average_cache_data[
                    channel_slice, view_start:view_stop
                ]
                average_reduced = reduce_ordered_extrema(
                    fit_average_to_channel(average_for_display, half_height=0.42),
                    start_sample=self.state.start_sample,
                    max_time_bins=time_bins,
                )
        offsets = np.empty(0, dtype=np.float32)
        if self.mode == "chart":
            x, y, offsets = stack_for_plot(
                reduced,
                self.source.sample_rate_hz,
                self.state.amplitude_spacing_uv,
            )
            x += np.float32(getattr(self.source, "time_start_seconds", 0.0))
            if average_reduced is not None:
                average_x, average_y, _ = stack_for_plot(
                    average_reduced,
                    self.source.sample_rate_hz,
                    1.0,
                )
                average_x += np.float32(self.source.time_start_seconds)
        else:
            placements = self._current_placements or placements_from_layout(
                self._active_layout()
            )
            x, y = place_waveforms(
                reduced,
                placements,
                amplitude_spacing=self.state.amplitude_spacing_uv,
                sample_range=(self.state.start_sample, self.state.stop_sample),
            )
            if average_reduced is not None:
                average_x, average_y = place_waveforms(
                    average_reduced,
                    placements,
                    amplitude_spacing=1.0,
                    sample_range=(self.state.start_sample, self.state.stop_sample),
                )
        geometry_ms = (time.perf_counter() - geometry_started) * 1_000.0

        submit_started = time.perf_counter()
        self.curve.setData(x=x, y=y, connect="finite", skipFiniteCheck=False)
        self.average_curve.setData(
            x=average_x, y=average_y, connect="finite", skipFiniteCheck=False
        )
        if self.mode == "chart":
            time_origin = float(getattr(self.source, "time_start_seconds", 0.0))
            start_seconds = time_origin + self.state.start_sample / self.source.sample_rate_hz
            stop_seconds = time_origin + self.state.stop_sample / self.source.sample_rate_hz
            self.plot.setXRange(start_seconds, stop_seconds, padding=0)
            self.plot.setYRange(-0.7, max(0.7, len(offsets) - 0.3), padding=0)
            labels = self.source.channel_labels[
                self.state.first_channel : self.state.first_channel + len(offsets)
            ]
            ticks = [(float(offset), label) for offset, label in zip(offsets, labels, strict=True)]
            self.plot.getAxis("left").setTicks([ticks])
        else:
            self.plot.setXRange(0.0, 1.0, padding=0.02)
            self.plot.setYRange(0.0, 1.0, padding=0.02)
        self._update_trial_scale(len(offsets))
        self._sync_controls()
        submit_ms = (time.perf_counter() - submit_started) * 1_000.0

        metric = UpdateMetric(
            reason=reason,
            geometry_ms=geometry_ms,
            submit_ms=submit_ms,
            points=int(x.size),
            channels=int(data.shape[0]),
            samples_per_bin=reduced.samples_per_bin,
        )
        if record:
            self.metrics.add(metric)
        self.metrics_label.setText(
            f"prepare {geometry_ms:5.1f} ms  |  submit {submit_ms:5.1f} ms  |  "
            f"{x.size:,} points"
        )
        self.status.setText(
            f"Samples {self.state.start_sample:,}–{self.state.stop_sample:,} of "
            f"{self.source.sample_count:,}  |  {self.mode} view, "
            f"{data.shape[0]} of {self.source.channel_count} channels  |  "
            f"{reduced.samples_per_bin} source samples/display bin"
        )
        self._update_topomaps()
        return metric

    def _update_trial_scale(self, chart_channels: int) -> None:
        """Show nice-valued scale bars in the display's current pixel scale."""

        window_seconds = max(
            1.0 / self.source.sample_rate_hz,
            (self.state.stop_sample - self.state.start_sample) / self.source.sample_rate_hz,
        )
        view_width = max(1.0, float(self.view_box.width()))
        view_height = max(1.0, float(self.view_box.height()))
        if self.mode == "chart":
            pixels_per_second = view_width / window_seconds
            y_span = max(1.4, chart_channels + 0.4)
            pixels_per_unit = view_height / y_span / self.state.amplitude_spacing_uv
        else:
            placements = self._current_placements or placements_from_layout(
                self._active_layout()
            )
            tile = placements[0]
            pixels_per_second = tile.width * view_width / 1.04 / window_seconds
            pixels_per_unit = (
                tile.height * view_height / 1.04 / self.state.amplitude_spacing_uv
            )

        time_value = _nice_scale_value(68.0 / pixels_per_second)
        amplitude_value = _nice_scale_value(32.0 / pixels_per_unit)
        unit = str(self.source.unit)
        if unit.startswith("native [") and unit.endswith("]"):
            unit = unit[8:-1]
        elif unit == "native units":
            unit = "units"
        self.trial_scale.set_scales(
            time_value,
            amplitude_value,
            unit,
            time_value * pixels_per_second,
            amplitude_value * pixels_per_unit,
        )
        self.trial_scale.move(
            max(0, self.plot.width() - self.trial_scale.width() - 14),
            max(0, self.plot.height() - self.trial_scale.height() - 14),
        )

    @QtCore.Slot()
    def run_automated_benchmark(self, iterations: int | None = None) -> None:
        iterations = iterations or self.automated_iterations or 40
        LOGGER.info("Starting synchronous %d-step benchmark", iterations)
        self.run_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            rng = np.random.default_rng(20260806)
            starts = rng.integers(
                0,
                max(1, self.state.max_start_sample + 1),
                size=iterations + 4,
            )
            for start in starts[:4]:
                self.state.start_sample = int(start)
                self.refresh("warmup", record=False)
                QtWidgets.QApplication.processEvents()
                self.plot.grab()

            for index, start in enumerate(starts[4:], start=1):
                frame_started = time.perf_counter()
                self.state.start_sample = int(start)
                metric = self.refresh("automated", record=False)
                QtWidgets.QApplication.processEvents()
                # QWidget.grab forces the submitted scene through a synchronous
                # paint path, producing a conservative end-to-end measurement.
                image = self.plot.grab()
                _ = image.size()
                QtWidgets.QApplication.processEvents()
                metric.synchronous_frame_ms = (
                    time.perf_counter() - frame_started
                ) * 1_000.0
                self.metrics.add(metric)
                self.status.setText(
                    f"Automated benchmark {index}/{iterations}: "
                    f"{metric.synchronous_frame_ms:.1f} ms synchronous frame"
                )

            path = self.save_results()
            summary = self.metrics.to_dict()["summary"]["synchronous_frame"]
            if summary:
                message = (
                    f"Benchmark complete: median {summary['median_ms']:.1f} ms, "
                    f"p95 {summary['p95_ms']:.1f} ms. Saved to {path}"
                )
                LOGGER.info(message)
                self.status.setText(message)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)

        if self.quit_after_automated:
            QtCore.QTimer.singleShot(100, QtWidgets.QApplication.quit)

    @QtCore.Slot()
    def save_results(self) -> Path:
        path = self.metrics.write(self.result_path)
        LOGGER.info("Wrote benchmark results to %s", path)
        return path

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt API
        key = event.key()
        modifiers = event.modifiers()
        if key == QtCore.Qt.Key.Key_Left:
            self.state.move_time_fraction(-1.0 if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier else -0.25)
        elif key == QtCore.Qt.Key.Key_Right:
            self.state.move_time_fraction(1.0 if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier else 0.25)
        elif key == QtCore.Qt.Key.Key_Up:
            self.state.move_channels(-1)
        elif key == QtCore.Qt.Key.Key_Down:
            self.state.move_channels(1)
        elif key == QtCore.Qt.Key.Key_Home:
            self.state.start_sample = 0
        elif key == QtCore.Qt.Key.Key_End:
            self.state.start_sample = self.state.max_start_sample
        else:
            super().keyPressEvent(event)
            return
        self.state.clamp()
        self.refresh("keyboard")
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if hasattr(self, "plot"):
            self._resize_timer.start(120)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        if self.metrics.updates:
            self.save_results()
        super().closeEvent(event)


def _default_result_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return Path(__file__).resolve().parents[2] / "results" / f"viewer_{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive read-only EEG viewer"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="FieldTrip .mat or EEGLAB .set/.fdt file",
    )
    parser.add_argument("--fieldtrip", type=Path, default=None, help="FieldTrip .mat file")
    parser.add_argument("--variable", default=None, help="MATLAB variable to import")
    parser.add_argument("--layout", type=Path, default=None, help="explicit FieldTrip .lay file")
    parser.add_argument("--standard-montage", default=None, help="specific MNE standard montage name")
    parser.add_argument("--mode", choices=("chart", "scalp", "grid"), default="chart")
    parser.add_argument("--channels", type=int, default=129)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="open deterministic synthetic data instead of showing the file dialog",
    )
    parser.add_argument("--sample-rate", type=float, default=1_000.0)
    parser.add_argument("--duration", type=float, default=180.0, help="recording seconds")
    parser.add_argument("--window", type=float, default=10.0, help="initial visible seconds")
    parser.add_argument("--visible-channels", type=int, default=32)
    parser.add_argument(
        "--amplitude-spacing",
        type=float,
        default=None,
        help="native data units per channel/tile spacing (auto for FieldTrip, 100 for synthetic)",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mains", type=float, default=50.0)
    parser.add_argument("--include-nans", action="store_true")
    parser.add_argument(
        "--opengl",
        choices=("on", "off"),
        default="on",
        help="request Qt's OpenGL viewport (default: on)",
    )
    parser.add_argument(
        "--auto-benchmark",
        type=int,
        default=0,
        metavar="STEPS",
        help="run a synchronous benchmark after launch",
    )
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="keep the window open after --auto-benchmark",
    )
    parser.add_argument(
        "--developer-controls",
        action="store_true",
        help="show benchmark timing controls in the viewer",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _configure_logging(path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _automatic_fieldtrip_spacing(dataset: Any) -> float:
    """Choose a readable native-unit scale without making unit assumptions."""

    samples = []
    for segment_index in range(min(8, dataset.signal.segment_count)):
        sample_count = dataset.signal.sample_count(segment_index)
        stop = min(sample_count, 2_000)
        samples.append(
            np.asarray(
                dataset.signal.read(segment_index, 0, slice(None), 0, stop)
            ).ravel()
        )
    finite = np.concatenate(samples)
    finite = np.abs(finite[np.isfinite(finite)])
    robust = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    target = max(1e-12, robust * 2.0)
    exponent = math.floor(math.log10(target))
    magnitude = 10.0**exponent
    fraction = target / magnitude
    nice = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
    return nice * magnitude


def prepare_viewer_input(
    path: str | Path,
    *,
    variable: str | None = None,
    explicit_layout: Path | None = None,
    standard_montage: str | None = None,
    amplitude_spacing: float | None = None,
) -> PreparedViewerInput:
    """Load one supported file and prepare shared viewer metadata/layout."""

    imported_started = time.perf_counter()
    source_path = Path(path).expanduser().resolve()
    suffix = source_path.suffix.casefold()
    if suffix == ".mat":
        dataset = read_fieldtrip(source_path, variable=variable)
    elif suffix in {".set", ".fdt"}:
        dataset = read_eeglab(source_path)
    else:
        raise ValueError("Supported EEG files are FieldTrip .mat and EEGLAB .set/.fdt")
    source = DatasetViewSource(dataset)
    resolved_layout = resolve_layout(
        source.channel_labels,
        embedded=dataset.montage_candidate,
        explicit_layout=explicit_layout,
        standard_montage=standard_montage,
    )
    spacing = (
        amplitude_spacing
        if amplitude_spacing is not None
        else _automatic_fieldtrip_spacing(dataset)
    )
    metadata = dict(dataset.metadata)
    metadata.update(
        {
            "dataset_kind": dataset.kind.value,
            "shape": [
                dataset.signal.series_count,
                dataset.signal.channel_count,
                [segment.sample_count for segment in dataset.segments],
            ],
            "layout_source": resolved_layout.source,
            "layout_kind": resolved_layout.kind,
            "layout_coverage": resolved_layout.coverage,
            "import_seconds": time.perf_counter() - imported_started,
        }
    )
    LOGGER.info(
        "Imported %s via %s (%s, %d channels, %d segments); layout %s, %.0f%% matched",
        source_path,
        metadata.get("adapter", "unknown adapter"),
        dataset.kind.value,
        source.channel_count,
        source.segment_count,
        resolved_layout.source,
        resolved_layout.coverage * 100,
    )
    LOGGER.info(
        "Signal values remain in native stored units; channel metadata declares %s. "
        "Automatic spacing: %g native units",
        sorted({channel.unit for channel in dataset.channels}),
        spacing,
    )
    return PreparedViewerInput(source, resolved_layout, metadata, spacing)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_file)
    opengl = args.opengl == "on"
    pg.setConfigOptions(antialias=False, useOpenGL=opengl)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("EEG Viewer")
    app.setOrganizationName("ECK EEG tools")

    if args.input is not None and args.fieldtrip is not None:
        build_parser().error("use --input or --fieldtrip, not both")
    input_path = args.input or args.fieldtrip
    if input_path is None and not args.synthetic and not args.auto_benchmark:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open EEG data",
            str(Path.home()),
            EEG_FILE_FILTER,
        )
        if not selected:
            return 0
        input_path = Path(selected)
    if input_path is not None:
        try:
            prepared = prepare_viewer_input(
                input_path,
                variable=args.variable,
                explicit_layout=args.layout,
                standard_montage=args.standard_montage,
                amplitude_spacing=args.amplitude_spacing,
            )
        except Exception as error:
            LOGGER.exception("Could not open %s", input_path)
            QtWidgets.QMessageBox.critical(
                None,
                "Could not open EEG data",
                f"{input_path}\n\n{type(error).__name__}: {error}",
            )
            return 2
        source = prepared.source
        resolved_layout = prepared.layout
        metadata = prepared.metadata
        amplitude_spacing = prepared.amplitude_spacing
    else:
        config = SyntheticConfig(
            channels=args.channels,
            sample_rate_hz=args.sample_rate,
            duration_seconds=args.duration,
            seed=args.seed,
            mains_hz=args.mains,
            include_nans=args.include_nans,
        )
        generated_started = time.perf_counter()
        recording = generate_recording(config, progress=LOGGER.info)
        generation_seconds = time.perf_counter() - generated_started
        metadata = recording.metadata()
        metadata["generation_seconds"] = generation_seconds
        LOGGER.info("Synthetic generation completed in %.3f seconds", generation_seconds)
        source = ArraySignalSource(
            data=recording.data_uv,
            sample_rate_hz=recording.sample_rate_hz,
            channel_labels=recording.channel_labels,
        )
        amplitude_spacing = (
            args.amplitude_spacing if args.amplitude_spacing is not None else 100.0
        )
        resolved_layout = (
            resolve_layout(source.channel_labels, explicit_layout=args.layout)
            if args.layout is not None
            else grid_layout(source.channel_labels)
        )
    output = args.output or _default_result_path()
    window = ViewerWindow(
        source,
        metadata,
        resolved_layout=resolved_layout,
        initial_mode=args.mode,
        initial_window_seconds=args.window,
        visible_channels=args.visible_channels,
        amplitude_spacing_uv=amplitude_spacing,
        opengl_requested=opengl,
        result_path=output.resolve(),
        automated_iterations=args.auto_benchmark,
        quit_after_automated=bool(args.auto_benchmark and not args.stay_open),
        show_developer_controls=args.developer_controls,
    )
    window.show()
    app._eeg_viewer_windows = [window]
    exit_code = app.exec()
    if output.exists():
        try:
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            LOGGER.info("Final timing summary: %s", summary)
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return int(exit_code)

"""Fast, reusable 2-D scalp interpolation and Qt display widgets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.linalg import lu_factor, lu_solve


def _diverging_lut(size: int = 256) -> np.ndarray:
    """Return an opaque blue-white-red lookup table."""

    stops = np.asarray([0.0, 0.24, 0.5, 0.76, 1.0])
    colours = np.asarray(
        [
            [40, 62, 160],
            [67, 151, 210],
            [242, 242, 238],
            [235, 140, 72],
            [170, 34, 52],
        ],
        dtype=float,
    )
    positions = np.linspace(0.0, 1.0, size)
    rgb = np.column_stack(
        [np.interp(positions, stops, colours[:, index]) for index in range(3)]
    )
    alpha = np.full((size, 1), 255.0)
    return np.rint(np.column_stack((rgb, alpha))).astype(np.uint8)


@dataclass(frozen=True)
class _SplineGeometry:
    valid_indices: np.ndarray
    evaluation: np.ndarray | None = None
    factorization: tuple[np.ndarray, np.ndarray] | None = None
    fallback_weights: np.ndarray | None = None

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        if self.fallback_weights is not None:
            return self.fallback_weights @ values[self.valid_indices]
        assert self.evaluation is not None
        assert self.factorization is not None
        right_hand_side = np.zeros(len(self.valid_indices) + 3, dtype=float)
        right_hand_side[: len(self.valid_indices)] = values[self.valid_indices]
        coefficients = lu_solve(self.factorization, right_hand_side)
        return self.evaluation @ coefficients


class TopomapInterpolator:
    """Interpolate fixed sensor positions with a thin-plate scalp spline.

    The spline factorization and grid basis are cached by the set of finite
    sensors. A small regularization term damps sensor noise while retaining the
    low-curvature shape expected from scalp potentials.
    """

    def __init__(self, positions: np.ndarray, grid_size: int = 128) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("sensor positions must have shape sensors x 2")
        if not np.isfinite(positions).all():
            raise ValueError("sensor positions must be finite")
        if grid_size < 16:
            raise ValueError("topomap grid must be at least 16 pixels wide")

        self.positions = positions
        self.grid_size = int(grid_size)
        axis = np.linspace(-1.0, 1.0, self.grid_size)
        grid_x, grid_y = np.meshgrid(axis, axis)
        self.inside_mask = grid_x**2 + grid_y**2 <= 1.0
        self._inside_points = np.column_stack(
            (grid_x[self.inside_mask], grid_y[self.inside_mask])
        )
        self._cache: OrderedDict[bytes, _SplineGeometry] = OrderedDict()

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.positions.shape[0],):
            raise ValueError("topomap values must have one value per sensor")

        finite = np.isfinite(values)
        image = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        if not np.any(finite):
            return image

        key = finite.tobytes()
        prepared = self._cache.get(key)
        if prepared is None:
            prepared = self._prepare_weights(finite)
            self._cache[key] = prepared
            self._cache.move_to_end(key)
            while len(self._cache) > 8:
                self._cache.popitem(last=False)

        interpolated = prepared.interpolate(values)
        image[self.inside_mask] = interpolated.astype(np.float32, copy=False)
        return image

    def _prepare_weights(self, finite: np.ndarray) -> _SplineGeometry:
        valid_indices = np.flatnonzero(finite)
        positions = self.positions[valid_indices]
        polynomial = np.column_stack((np.ones(valid_indices.size), positions))
        if valid_indices.size < 3 or np.linalg.matrix_rank(polynomial) < 3:
            return _SplineGeometry(
                valid_indices,
                fallback_weights=self._inverse_distance_weights(positions),
            )

        pairwise_squared = np.sum(
            (positions[:, None, :] - positions[None, :, :]) ** 2,
            axis=2,
        )
        nonzero_distances = np.sqrt(pairwise_squared[pairwise_squared > 1e-12])
        typical_spacing = (
            float(np.median(nonzero_distances)) if nonzero_distances.size else 1.0
        )
        smoothing = max(1e-7, 0.002 * typical_spacing**2)

        sensor_kernel = self._thin_plate_kernel(pairwise_squared)
        system = np.zeros((valid_indices.size + 3, valid_indices.size + 3))
        system[: valid_indices.size, : valid_indices.size] = sensor_kernel
        system[: valid_indices.size, : valid_indices.size] += (
            np.eye(valid_indices.size) * smoothing
        )
        system[: valid_indices.size, valid_indices.size :] = polynomial
        system[valid_indices.size :, : valid_indices.size] = polynomial.T

        grid_squared = np.sum(
            (self._inside_points[:, None, :] - positions[None, :, :]) ** 2,
            axis=2,
        )
        evaluation = np.column_stack(
            (
                self._thin_plate_kernel(grid_squared),
                np.ones(self._inside_points.shape[0]),
                self._inside_points,
            )
        )
        return _SplineGeometry(
            valid_indices,
            evaluation=evaluation,
            factorization=lu_factor(system),
        )

    def _inverse_distance_weights(self, positions: np.ndarray) -> np.ndarray:
        distance_squared = np.sum(
            (self._inside_points[:, None, :] - positions[None, :, :]) ** 2,
            axis=2,
        )
        inverse = 1.0 / np.maximum(distance_squared, 1e-8)
        return (inverse / inverse.sum(axis=1, keepdims=True)).astype(np.float32)

    @staticmethod
    def _thin_plate_kernel(distance_squared: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = 0.5 * distance_squared * np.log(distance_squared)
        return np.where(distance_squared > 0.0, kernel, 0.0)


class ScalpMapWidget(QtWidgets.QWidget):
    """One top-down scalp image with sensor positions and orientation marks."""

    def __init__(
        self,
        title: str,
        interpolator: TopomapInterpolator,
        labels: Sequence[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.interpolator = interpolator
        self.lookup_table = _diverging_lut()
        self.last_values = np.full(len(labels), np.nan, dtype=float)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.title = QtWidgets.QLabel(title)
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("color:#d7e0eb; font-weight:600;")
        layout.addWidget(self.title)

        self.plot = pg.PlotWidget(background=(11, 14, 20))
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-1.2, 1.2, padding=0)
        self.plot.setYRange(-1.18, 1.24, padding=0)
        self.plot.setMinimumSize(170, 190)
        layout.addWidget(self.plot, 1)

        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setZValue(-10)
        self.plot.addItem(self.image)

        head_pen = pg.mkPen((205, 214, 225), width=1.4)
        head = QtWidgets.QGraphicsEllipseItem(-1.0, -1.0, 2.0, 2.0)
        head.setPen(head_pen)
        head.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self.plot.addItem(head)

        nose_path = QtGui.QPainterPath(QtCore.QPointF(-0.12, 0.99))
        nose_path.lineTo(0.0, 1.17)
        nose_path.lineTo(0.12, 0.99)
        nose = QtWidgets.QGraphicsPathItem(nose_path)
        nose.setPen(head_pen)
        self.plot.addItem(nose)

        for side in (-1.0, 1.0):
            ear_path = QtGui.QPainterPath(QtCore.QPointF(side, 0.24))
            ear_path.cubicTo(side * 1.13, 0.19, side * 1.13, -0.19, side, -0.24)
            ear = QtWidgets.QGraphicsPathItem(ear_path)
            ear.setPen(head_pen)
            self.plot.addItem(ear)

        positions = interpolator.positions
        self.sensors = pg.ScatterPlotItem(
            x=positions[:, 0],
            y=positions[:, 1],
            data=list(labels),
            size=5,
            pen=pg.mkPen((20, 24, 31), width=0.8),
            brush=pg.mkBrush((238, 242, 247, 210)),
            hoverable=True,
            tip=lambda _x, _y, label: str(label),
        )
        self.plot.addItem(self.sensors)

        self.scale = DivergingScaleWidget()
        layout.addWidget(self.scale)

    def set_values(self, values: np.ndarray, limit: float, unit: str) -> None:
        self.last_values = np.asarray(values, dtype=float).copy()
        field = self.interpolator.interpolate(self.last_values)
        finite = np.isfinite(field)
        rgba = np.zeros((*field.shape, 4), dtype=np.uint8)
        if np.any(finite):
            normalised = np.zeros(field.shape, dtype=float)
            normalised[finite] = np.clip(
                (field[finite] + limit) / (2.0 * limit), 0.0, 1.0
            )
            indices = np.rint(normalised * (len(self.lookup_table) - 1)).astype(int)
            rgba[finite] = self.lookup_table[indices[finite]]
        self.image.setImage(rgba, autoLevels=False)
        self.image.setRect(QtCore.QRectF(-1.0, -1.0, 2.0, 2.0))
        self.scale.set_range(limit, unit)


class DivergingScaleWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.lookup_table = _diverging_lut(128)
        self.limit = 1.0
        self.unit = ""
        self.setFixedHeight(34)

    def set_range(self, limit: float, unit: str) -> None:
        self.limit = float(limit)
        self.unit = str(unit)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        bar = QtCore.QRect(8, 2, max(1, self.width() - 16), 10)
        gradient = QtGui.QLinearGradient(bar.left(), 0, bar.right(), 0)
        for index, colour in enumerate(self.lookup_table[::8]):
            fraction = index / max(1, len(self.lookup_table[::8]) - 1)
            gradient.setColorAt(fraction, QtGui.QColor(*map(int, colour[:3])))
        painter.fillRect(bar, gradient)
        painter.setPen(QtGui.QColor(80, 91, 107))
        painter.drawRect(bar)
        painter.setPen(QtGui.QColor(186, 197, 211))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        left = f"−{self.limit:.3g}"
        right = f"+{self.limit:.3g} {self.unit}"
        painter.drawText(8, 29, left)
        painter.drawText(
            bar,
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom,
            "0",
        )
        painter.drawText(
            8,
            15,
            max(1, self.width() - 16),
            14,
            QtCore.Qt.AlignmentFlag.AlignRight,
            right,
        )


class DualTopomapWidget(QtWidgets.QGroupBox):
    """One signal scalp map, optionally paired with a clean-trial average."""

    window_ms_changed = QtCore.Signal(float)

    def __init__(
        self,
        positions: np.ndarray,
        labels: Sequence[str],
        channel_indices: Sequence[int],
        sample_rate_hz: float,
        parent: QtWidgets.QWidget | None = None,
        *,
        show_clean_average: bool = True,
    ) -> None:
        super().__init__("Scalp maps", parent)
        self.channel_indices = np.asarray(channel_indices, dtype=int)
        self.show_clean_average = bool(show_clean_average)
        self.last_clean_values = np.full(len(channel_indices), np.nan, dtype=float)
        self.last_current_values = np.full(len(channel_indices), np.nan, dtype=float)
        self.last_clean_limit = 1.0
        self.last_current_limit = 1.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(7, 8, 7, 7)
        layout.setSpacing(4)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Window"))
        self.window_ms = QtWidgets.QDoubleSpinBox()
        minimum_ms = 1_000.0 / float(sample_rate_hz)
        self.window_ms.setRange(minimum_ms, 2_000.0)
        self.window_ms.setDecimals(1)
        self.window_ms.setSingleStep(max(minimum_ms, 5.0))
        self.window_ms.setSuffix(" ms")
        self.window_ms.setValue(max(minimum_ms, 20.0))
        self.window_ms.setToolTip("Time window centred on the waveform cursor")
        self.window_ms.valueChanged.connect(self.window_ms_changed)
        controls.addWidget(self.window_ms)
        controls.addStretch(1)
        layout.addLayout(controls)

        interpolator = TopomapInterpolator(np.asarray(positions, dtype=float))
        map_row = QtWidgets.QHBoxLayout()
        map_row.setSpacing(5)
        self.clean_map = (
            ScalpMapWidget("Clean-trial average", interpolator, labels)
            if self.show_clean_average
            else None
        )
        self.current_map = ScalpMapWidget("Selected trial", interpolator, labels)
        if self.clean_map is not None:
            map_row.addWidget(self.clean_map, 1)
        map_row.addWidget(self.current_map, 1)
        layout.addLayout(map_row, 1)

        self.time_label = QtWidgets.QLabel("Move over a waveform to set the map centre")
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color:#9caabd;")
        layout.addWidget(self.time_label)
        self.setMinimumWidth(410 if self.show_clean_average else 210)
        self.setMaximumWidth(600 if self.show_clean_average else 320)

    def set_maps(
        self,
        clean_values: np.ndarray | None,
        current_values: np.ndarray,
        *,
        trial_number: int,
        current_title: str | None = None,
        time_seconds: float,
        window_seconds: float,
        unit: str,
    ) -> None:
        if clean_values is not None:
            self.last_clean_values = np.asarray(clean_values, dtype=float).copy()
        self.last_current_values = np.asarray(current_values, dtype=float).copy()
        if self.clean_map is not None:
            self.last_clean_limit = self._symmetric_limit(self.last_clean_values)
        self.last_current_limit = self._symmetric_limit(self.last_current_values)
        if self.clean_map is not None:
            self.clean_map.set_values(
                self.last_clean_values, self.last_clean_limit, unit
            )
        self.current_map.set_values(
            self.last_current_values, self.last_current_limit, unit
        )
        self.current_map.title.setText(current_title or f"Trial {trial_number}")
        self.time_label.setText(
            f"Centre {time_seconds * 1_000:.1f} ms  ·  "
            f"window {window_seconds * 1_000:.1f} ms"
        )

    @staticmethod
    def _symmetric_limit(values: np.ndarray) -> float:
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if finite.size else 1.0
        if not math.isfinite(limit) or limit <= 1e-30:
            return 1.0
        return limit

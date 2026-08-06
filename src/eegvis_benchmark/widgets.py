"""Small read-only Qt widgets used by the EEG review interface."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets


class TrialOverviewWidget(QtWidgets.QWidget):
    """Compact clickable artifact density overview with one bar per segment."""

    segment_selected = QtCore.Signal(int)

    def __init__(
        self,
        bad_channel_counts: NDArray[np.integer],
        channel_count: int,
        layer_counts: Mapping[str, NDArray[np.integer]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.bad_channel_counts = np.asarray(bad_channel_counts, dtype=np.int32)
        self.channel_count = max(1, int(channel_count))
        self.layer_counts = {
            str(name): np.asarray(counts, dtype=np.int32)
            for name, counts in (layer_counts or {}).items()
        }
        self.current_segment = 0
        self.setMouseTracking(True)
        self.setMinimumHeight(62)
        self.setToolTip("Click or drag to select a trial")

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        return QtCore.QSize(900, 76)

    @property
    def segment_count(self) -> int:
        return int(self.bad_channel_counts.size)

    @QtCore.Slot(int)
    def set_current_segment(self, index: int) -> None:
        if not self.segment_count:
            return
        self.current_segment = max(0, min(self.segment_count - 1, int(index)))
        self.update()

    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(7.0, 6.0, max(1.0, self.width() - 14.0), max(1.0, self.height() - 22.0))

    def _index_at(self, x: float) -> int | None:
        if not self.segment_count:
            return None
        rect = self._plot_rect()
        fraction = (float(x) - rect.left()) / max(1.0, rect.width())
        return max(0, min(self.segment_count - 1, int(fraction * self.segment_count)))

    def _select_at(self, x: float) -> None:
        index = self._index_at(x)
        if index is not None and index != self.current_segment:
            self.set_current_segment(index)
            self.segment_selected.emit(index)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._select_at(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        index = self._index_at(event.position().x())
        if index is not None:
            if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                self._select_at(event.position().x())
            parts = [
                f"Trial {index + 1}: {self.bad_channel_counts[index]}/{self.channel_count} bad channels"
            ]
            for name, counts in self.layer_counts.items():
                count = int(counts[index])
                if count:
                    parts.append(f"{name}: {count}")
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), " · ".join(parts), self)
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor(17, 21, 29))
        rect = self._plot_rect()
        painter.setPen(QtGui.QPen(QtGui.QColor(58, 67, 81), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        if not self.segment_count:
            return

        bar_width = rect.width() / self.segment_count
        for index, count in enumerate(self.bad_channel_counts):
            fraction = min(1.0, max(0.0, float(count) / self.channel_count))
            height = max(1.0, fraction * rect.height()) if count else 1.0
            x = rect.left() + index * bar_width
            if count:
                # Low artifact densities are amber; dense failures tend red.
                red = 225
                green = int(174 - 105 * fraction)
                colour = QtGui.QColor(red, green, 63)
            else:
                colour = QtGui.QColor(51, 65, 75)
            painter.fillRect(
                QtCore.QRectF(x, rect.bottom() - height, max(1.0, bar_width * 0.82), height),
                colour,
            )

        current_x = rect.left() + (self.current_segment + 0.5) * bar_width
        painter.setPen(QtGui.QPen(QtGui.QColor(245, 249, 255), 2))
        painter.drawLine(
            QtCore.QPointF(current_x, rect.top()),
            QtCore.QPointF(current_x, rect.bottom() + 2),
        )
        painter.setPen(QtGui.QColor(145, 157, 174))
        painter.drawText(7, self.height() - 3, "1")
        final_text = str(self.segment_count)
        width = painter.fontMetrics().horizontalAdvance(final_text)
        painter.drawText(self.width() - width - 7, self.height() - 3, final_text)

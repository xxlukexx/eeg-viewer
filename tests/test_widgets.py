from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtTest, QtWidgets

from eeg_viewer.widgets import TrialOverviewWidget


def test_trial_overview_click_selects_segment_and_renders() -> None:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = TrialOverviewWidget(
        np.array([0, 1, 2, 4]),
        channel_count=4,
        layer_counts={"blink": np.array([0, 1, 0, 2])},
    )
    widget.resize(400, 76)
    selected: list[int] = []
    widget.segment_selected.connect(selected.append)
    widget.show()
    application.processEvents()

    QtTest.QTest.mouseClick(
        widget,
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(255, 30),
    )
    image = widget.grab().toImage()

    assert selected == [2]
    assert widget.current_segment == 2
    assert not image.isNull()
    widget.close()

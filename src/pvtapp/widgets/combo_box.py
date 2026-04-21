"""Shared no-wheel selector helpers for the desktop UI."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QTabBar,
    QTabWidget,
    QWidget,
)


_ARROW_COLOR_DEFAULT = QColor("#e5e7eb")
_ARROW_COLOR_HOVER = QColor("#60a5fa")
_ARROW_COLOR_DISABLED = QColor("#64748b")


def _paint_triangle(
    widget: QWidget,
    *,
    cx: float,
    cy: float,
    width: float,
    height: float,
    pointing: str,
    color: QColor,
) -> None:
    """Paint a solid filled triangle centred on (cx, cy).

    This bypasses QSS ``::down-arrow`` / ``::up-arrow`` subcontrols, which Qt's
    stylesheet engine renders unreliably for dark themes on Windows — the
    border-triangle hack collapses to a blank rectangle and inline SVG URIs
    sometimes fail to resolve. Drawing the glyph directly in paintEvent is
    deterministic and theme-aware.
    """
    painter = QPainter(widget)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        half_w = width / 2.0
        half_h = height / 2.0
        if pointing == "down":
            triangle = QPolygonF(
                [
                    QPointF(cx - half_w, cy - half_h),
                    QPointF(cx + half_w, cy - half_h),
                    QPointF(cx, cy + half_h),
                ]
            )
        else:  # "up"
            triangle = QPolygonF(
                [
                    QPointF(cx - half_w, cy + half_h),
                    QPointF(cx + half_w, cy + half_h),
                    QPointF(cx, cy - half_h),
                ]
            )
        painter.drawPolygon(triangle)
    finally:
        painter.end()


class NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel selection changes.

    This keeps scroll-wheel navigation attached to the surrounding scroll area
    instead of silently mutating the hovered combo box. Also paints an explicit
    down-triangle glyph on top of Qt's drop-down area so the arrow is visible
    in the dark theme regardless of QSS rendering quirks.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def wheelEvent(self, event) -> None:  # pragma: no cover - simple UI guard
        event.ignore()

    def paintEvent(self, event) -> None:  # pragma: no cover - visual-only
        super().paintEvent(event)
        if not self.isEnabled():
            color = _ARROW_COLOR_DISABLED
        elif self.underMouse():
            color = _ARROW_COLOR_HOVER
        else:
            color = _ARROW_COLOR_DEFAULT
        # Drop-down subcontrol width from our QSS is ~22px at scale 1.0. We
        # centre the glyph horizontally over that band.
        arrow_band_width = 22
        arrow_band_right_pad = 6
        cx = self.width() - arrow_band_right_pad - (arrow_band_width - arrow_band_right_pad) / 2.0
        cy = self.height() / 2.0
        _paint_triangle(
            self,
            cx=cx,
            cy=cy,
            width=9.0,
            height=6.0,
            pointing="down",
            color=color,
        )


class _NoWheelAbstractSpinBox:
    """Mixin adding up/down arrow painting on top of Qt's default spin box."""

    def _paint_spin_arrows(self) -> None:  # pragma: no cover - visual-only
        enabled = self.isEnabled()
        color = _ARROW_COLOR_DEFAULT if enabled else _ARROW_COLOR_DISABLED
        width = self.width()
        height = self.height()
        # Spin buttons occupy a ~14px-wide strip on the right, split vertically.
        button_band = 14
        right_pad = 6
        cx = width - right_pad - (button_band - right_pad) / 2.0
        # Up arrow centred in the top half, down arrow in the bottom half.
        up_cy = height * 0.30
        down_cy = height * 0.70
        _paint_triangle(self, cx=cx, cy=up_cy, width=7.0, height=4.5, pointing="up", color=color)
        _paint_triangle(self, cx=cx, cy=down_cy, width=7.0, height=4.5, pointing="down", color=color)


class NoWheelSpinBox(QSpinBox, _NoWheelAbstractSpinBox):
    """Spin box that ignores mouse-wheel value changes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def wheelEvent(self, event) -> None:  # pragma: no cover - simple UI guard
        event.ignore()

    def paintEvent(self, event) -> None:  # pragma: no cover - visual-only
        super().paintEvent(event)
        self._paint_spin_arrows()


class NoWheelDoubleSpinBox(QDoubleSpinBox, _NoWheelAbstractSpinBox):
    """Double spin box that ignores mouse-wheel value changes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def wheelEvent(self, event) -> None:  # pragma: no cover - simple UI guard
        event.ignore()

    def paintEvent(self, event) -> None:  # pragma: no cover - visual-only
        super().paintEvent(event)
        self._paint_spin_arrows()


class CheckBox(QCheckBox):
    """QCheckBox that paints its own visible checkmark glyph.

    Qt's default checkmark relies on QSS ``image: url(...)`` for the
    ``::indicator:checked`` pseudo-state, which is not reliable on every
    Windows style — the override leaves the checkbox looking like a flat
    filled square with no actual check visible. Drawing the tick directly
    in ``paintEvent`` is deterministic and always shows up.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)

    def paintEvent(self, event) -> None:  # pragma: no cover - visual-only
        super().paintEvent(event)
        if not self.isChecked() and self.checkState() != Qt.CheckState.PartiallyChecked:
            return
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        indicator_rect = self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, opt, self
        )
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            stroke = QColor("#ffffff") if self.isEnabled() else QColor("#9ca3af")
            # Thicker stroke when the indicator is bigger so the tick stays
            # legible at all UI scales.
            stroke_width = max(1.6, indicator_rect.width() * 0.14)
            pen = QPen(
                stroke,
                stroke_width,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            x = float(indicator_rect.x())
            y = float(indicator_rect.y())
            w = float(indicator_rect.width())
            h = float(indicator_rect.height())
            # Classic three-point checkmark, anchored to indicator bounds
            # with a small margin so it doesn't kiss the border.
            path = QPainterPath()
            path.moveTo(x + w * 0.22, y + h * 0.52)
            path.lineTo(x + w * 0.43, y + h * 0.72)
            path.lineTo(x + w * 0.80, y + h * 0.30)
            painter.drawPath(path)
        finally:
            painter.end()


class NoWheelTabBar(QTabBar):
    """Tab bar that ignores mouse-wheel tab switching."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

    def wheelEvent(self, event) -> None:  # pragma: no cover - simple UI guard
        event.ignore()


class NoWheelTabWidget(QTabWidget):
    """Tab widget that ignores mouse-wheel tab switching."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTabBar(NoWheelTabBar(self))

    def wheelEvent(self, event) -> None:  # pragma: no cover - simple UI guard
        event.ignore()

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from models import ZoneDefinition


class VideoWidget(QWidget):
    zone_changed = Signal(object)
    mapping_changed = Signal(dict)
    frame_size_changed = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._frame_size = (0, 0)
        self._zones: list[ZoneDefinition] = []
        self._active_zone: ZoneDefinition | None = None
        self._drag_index: int | None = None
        self._video_rect = QRectF()

    def set_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if self._frame_size != (width, height):
            self._frame_size = (width, height)
            self.frame_size_changed.emit(width, height)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.update()

    def set_zones(self, zones: list[ZoneDefinition]) -> None:
        self._zones = zones
        if self._active_zone not in zones:
            self._active_zone = zones[0] if zones else None
        self.update()

    def set_active_zone(self, zone: ZoneDefinition | None) -> None:
        self._active_zone = zone
        self.update()

    def clear_active_zone(self) -> None:
        if self._active_zone is None:
            return
        self._active_zone.polygon.clear()
        self._active_zone.closed = False
        self.zone_changed.emit(self._active_zone)
        self.update()

    def close_active_zone(self) -> None:
        if self._active_zone is None or len(self._active_zone.polygon) < 3:
            return
        self._active_zone.closed = True
        self.zone_changed.emit(self._active_zone)
        self.update()

    def coordinate_mapping(self) -> dict[str, float]:
        if self._frame_size[0] == 0 or self._video_rect.width() == 0:
            return {"x": 1.0, "y": 1.0}
        return {
            "x": self._frame_size[0] / self._video_rect.width(),
            "y": self._frame_size[1] / self._video_rect.height(),
        }

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151A20"))
        self._video_rect = self._calculate_video_rect()
        if self._image is None:
            painter.setPen(QColor("#AAB4BE"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开视频源后将在此显示画面")
            return
        painter.drawImage(self._video_rect, self._image)
        self._draw_zones(painter)
        self.mapping_changed.emit(self.coordinate_mapping())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._video_rect = self._calculate_video_rect()
        self.mapping_changed.emit(self.coordinate_mapping())

    def mousePressEvent(self, event: object) -> None:
        if self._active_zone is None or self._image is None:
            return
        position = event.position()
        original = self._to_original(position)
        if original is None:
            return
        nearest = self._nearest_vertex(position)
        if event.button() == Qt.MouseButton.RightButton:
            if nearest is not None:
                del self._active_zone.polygon[nearest]
                self.zone_changed.emit(self._active_zone)
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if nearest is not None:
            self._drag_index = nearest
            return
        if self._active_zone.closed:
            return
        self._active_zone.polygon.append([original.x(), original.y()])
        self.zone_changed.emit(self._active_zone)
        self.update()

    def mouseMoveEvent(self, event: object) -> None:
        if self._active_zone is None or self._drag_index is None:
            return
        original = self._to_original(event.position())
        if original is None:
            return
        self._active_zone.polygon[self._drag_index] = [original.x(), original.y()]
        self.zone_changed.emit(self._active_zone)
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        self._drag_index = None

    def mouseDoubleClickEvent(self, event: object) -> None:
        if self._active_zone is None or self._active_zone.closed:
            return
        original = self._to_original(event.position())
        if original is None:
            return
        if self._active_zone.polygon:
            last = self._active_zone.polygon[-1]
            if abs(last[0] - original.x()) <= 1 and abs(last[1] - original.y()) <= 1:
                self._active_zone.polygon.pop()
        self.close_active_zone()

    def _calculate_video_rect(self) -> QRectF:
        frame_width, frame_height = self._frame_size
        if frame_width == 0 or frame_height == 0:
            return QRectF(self.rect())
        scale = min(self.width() / frame_width, self.height() / frame_height)
        width = frame_width * scale
        height = frame_height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _draw_zones(self, painter: QPainter) -> None:
        for zone in self._zones:
            points = [self._to_display(QPointF(point[0], point[1])) for point in zone.polygon]
            if not points:
                continue
            is_active = zone is self._active_zone
            pen = QPen(QColor(zone.color), 3 if is_active else 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if len(points) >= 2:
                painter.drawPolyline(points)
            if len(points) >= 3 and zone.closed:
                painter.drawLine(points[-1], points[0])
            for point in points:
                painter.setBrush(QColor("#FFFFFF") if is_active else QColor(zone.color))
                painter.drawEllipse(point, 5 if is_active else 4, 5 if is_active else 4)
            painter.setPen(QColor(zone.color))
            painter.drawText(points[0] + QPointF(8, -8), zone.name)

    def _to_original(self, point: QPointF) -> QPointF | None:
        if not self._video_rect.contains(point) or self._frame_size[0] == 0:
            return None
        x = (point.x() - self._video_rect.x()) * self._frame_size[0] / self._video_rect.width()
        y = (point.y() - self._video_rect.y()) * self._frame_size[1] / self._video_rect.height()
        return QPointF(x, y)

    def _to_display(self, point: QPointF) -> QPointF:
        if self._frame_size[0] == 0:
            return point
        x = self._video_rect.x() + point.x() * self._video_rect.width() / self._frame_size[0]
        y = self._video_rect.y() + point.y() * self._video_rect.height() / self._frame_size[1]
        return QPointF(x, y)

    def _nearest_vertex(self, position: QPointF) -> int | None:
        if self._active_zone is None:
            return None
        for index, point in enumerate(self._active_zone.polygon):
            display = self._to_display(QPointF(point[0], point[1]))
            if (display - position).manhattanLength() <= 12:
                return index
        return None

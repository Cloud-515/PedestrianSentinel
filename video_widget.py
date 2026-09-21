from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from models import AlarmEvent, ZoneDefinition


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
        # 上一次按下是否真的新增了顶点（存它的下标）。双击闭合时要靠它判断"要不要
        # 把刚才那一下加的点去掉" —— 用位置判断不可靠：双击落在画面外某个已有顶点上
        # 时，位置同样对得上，那样就会误删一个正常顶点。
        self._last_added_vertex: int | None = None
        self._video_rect = QRectF()
        self._pending_alarm_events: dict[tuple[str, str], AlarmEvent] = {}
        self._displayed_alarm_events: list[AlarmEvent] = []
        self._aggregation_timer = QTimer(self)
        self._aggregation_timer.setSingleShot(True)
        self._aggregation_timer.timeout.connect(self._flush_alarm_batch)
        self._alarm_timer = QTimer(self)
        self._alarm_timer.setSingleShot(True)
        self._alarm_timer.timeout.connect(self.clear_alarm)

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
        self._draw_alarm_banner(painter)
        self.mapping_changed.emit(self.coordinate_mapping())

    def show_alarm(self, event: AlarmEvent) -> None:
        key = (event.zone_name, event.track_id)
        self._pending_alarm_events[key] = event
        if not self._aggregation_timer.isActive():
            self._aggregation_timer.start(800)
        self._alarm_timer.start(5000)

    def clear_alarm(self) -> None:
        self._aggregation_timer.stop()
        self._alarm_timer.stop()
        if not self._pending_alarm_events and not self._displayed_alarm_events:
            return
        self._pending_alarm_events.clear()
        self._displayed_alarm_events.clear()
        self.update()

    def _flush_alarm_batch(self) -> None:
        if not self._pending_alarm_events:
            return
        self._displayed_alarm_events = list(self._pending_alarm_events.values())
        self._pending_alarm_events.clear()
        self.update()

    def _draw_alarm_banner(self, painter: QPainter) -> None:
        if not self._displayed_alarm_events or self._video_rect.isEmpty():
            return
        max_details = 4
        total_events = len(self._displayed_alarm_events)
        detail_events = self._displayed_alarm_events[:max_details]
        overflow_count = total_events - len(detail_events)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(12)
        metrics = QFontMetrics(font)
        row_height = metrics.height() + 4
        row_count = 1 + len(detail_events) + (1 if overflow_count else 0)
        banner_height = row_count * row_height + 20
        banner = QRectF(
            self._video_rect.x() + 12,
            self._video_rect.y() + 12,
            max(0.0, self._video_rect.width() - 24),
            min(banner_height, max(0.0, self._video_rect.height() - 24)),
        )
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(211, 47, 47, 179))
        painter.drawRoundedRect(banner, 4, 4)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(font)
        text_rect = banner.adjusted(16, 10, -16, -10)
        lines = [f"入侵报警  |  共 {total_events} 条报警"]
        lines.extend(
            f"目标 {event.track_id} 进入区域：{event.zone_name}  |  {event.wall_time}"
            for event in detail_events
        )
        if overflow_count:
            lines.append(f"另有 {overflow_count} 条报警")
        for index, line in enumerate(lines):
            row = QRectF(
                text_rect.x(),
                text_rect.y() + index * row_height,
                text_rect.width(),
                row_height,
            )
            elided = metrics.elidedText(
                line,
                Qt.TextElideMode.ElideRight,
                round(row.width()),
            )
            painter.drawText(
                row,
                Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine,
                elided,
            )
        painter.restore()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._video_rect = self._calculate_video_rect()
        self.mapping_changed.emit(self.coordinate_mapping())

    def mousePressEvent(self, event: object) -> None:
        if self._active_zone is None or self._image is None:
            return
        position = event.position()
        # 每次按下都重新判定"这一下有没有加点"，供紧随其后的双击闭合使用。
        self._last_added_vertex = None
        # 先看是不是点在某个顶点上，再看是不是落在画面内。
        #
        # 这个顺序是必须的：顶点可以落在画面之外（换过不同分辨率的视频源、或者把顶点
        # 拖出去过，都会这样），而"点是否在画面内"的判断会把画面外的顶点一并挡掉 ——
        # 于是那个顶点既拖不动也删不掉，永远卡在画面外。只有**新增**顶点才必须落在
        # 画面内（点在黑边上不该凭空加一个点）。
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
        original = self._to_original(position)
        if original is None:
            return
        self._active_zone.polygon.append([original.x(), original.y()])
        self._last_added_vertex = len(self._active_zone.polygon) - 1
        self.zone_changed.emit(self._active_zone)
        self.update()

    def mouseMoveEvent(self, event: object) -> None:
        if self._active_zone is None or self._drag_index is None:
            return
        # 不要求落在画面内：允许把顶点拖出画面，也允许把画面外的顶点拖回来。指针在
        # 画面外时按同一个线性映射外推即可，坐标会被夹在一个宽松范围内（见
        # _to_original），免得配置里出现天文数字。
        original = self._to_original(event.position(), clamp_to_video=False)
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
        original = self._to_original(event.position(), clamp_to_video=False)
        if original is None:
            return
        # 双击的第一下（press）若已经在落点处加过一个顶点，这里把它去掉再闭合。
        # 判据是"press 真的加过点"（_last_added_vertex），而不是位置对得上 —— 双击
        # 落在画面外某个已有顶点上时位置同样对得上，那样会误删一个正常顶点。
        added = self._last_added_vertex
        self._last_added_vertex = None
        polygon = self._active_zone.polygon
        if added is not None and polygon and added == len(polygon) - 1:
            last = polygon[-1]
            if abs(last[0] - original.x()) <= 1 and abs(last[1] - original.y()) <= 1:
                polygon.pop()
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
        """只画**编辑用的那一层**：激活区域的虚线轮廓与顶点手柄。

        区域本身（贴地效果、被行人遮挡）由引擎烧进画面里 —— 它必须画在人下面才能被
        遮挡，而控件只能画在图像之上。这里如果再把实线画一遍，就会出现两条错开的线：
        原来就是这样，线看着又粗又糊，而且横穿在人身上。

        虚线是刻意选的样式：它是界面元素（"你正在编辑这条边界"），不该被误读成画面里
        的实体边界。顶点手柄也只画激活区域的 —— 每个区域都挂一串圆点，监控时很吵。
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        zone = self._active_zone
        if zone is None or not zone.polygon:
            return
        handles = [self._handle_position(QPointF(point[0], point[1])) for point in zone.polygon]
        points = [handle for handle, _ in handles]

        def trace() -> None:
            if len(points) >= 2:
                painter.drawPolyline(points)
            if len(points) >= 3 and zone.closed:
                painter.drawLine(points[-1], points[0])

        painter.setBrush(Qt.BrushStyle.NoBrush)
        # 深色垫底 + 白色虚线：垫底保证在浅色地面上也看得见，白色让它一眼就是界面
        # 元素（"你正在编辑这条边界"），不会被误读成画面里的实体警戒线。
        painter.setPen(QPen(QColor(0, 0, 0, 150), 3.0))
        trace()
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.6, Qt.PenStyle.DashLine))
        trace()
        painter.setPen(QPen(QColor(0, 0, 0, 180), 1.2))
        for (position, clamped), handle in zip(handles, points):
            if not clamped:
                painter.setBrush(QColor("#FFFFFF"))
                painter.drawEllipse(position, 4, 4)
                continue
            # 被夹到边缘的顶点：画成指向真实方向的三角，提示"它还在更外面"。
            painter.setBrush(QColor("#FFFFFF"))
            direction = QPointF(handle.x() - position.x(), handle.y() - position.y())
            if direction.isNull():
                painter.drawEllipse(position, 4, 4)
                continue
            length = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
            unit = QPointF(direction.x() / length, direction.y() / length)
            perpendicular = QPointF(-unit.y(), unit.x())
            painter.drawPolygon(
                [
                    QPointF(position.x() + unit.x() * 8, position.y() + unit.y() * 8),
                    QPointF(
                        position.x() - unit.x() * 3 + perpendicular.x() * 5,
                        position.y() - unit.y() * 3 + perpendicular.y() * 5,
                    ),
                    QPointF(
                        position.x() - unit.x() * 3 - perpendicular.x() * 5,
                        position.y() - unit.y() * 3 - perpendicular.y() * 5,
                    ),
                ]
            )

    # 拖动时允许顶点离画面多远（画面尺寸的比例）。留一点余量是有意义的：警戒区一部分
    # 落到画面之外是合理的（比如想让它贴着画面边缘、不留缝）。但余量不能大 —— 顶点在
    # 屏幕上的位置是按比例外推的，允许拖出太远就会跑到控件可视区之外，那就真的抓不
    # 回来了（这正是"画面外的顶点拖不动"这类问题的另一半）。
    OFF_FRAME_MARGIN = 0.05

    def _to_original(self, point: QPointF, *, clamp_to_video: bool = True) -> QPointF | None:
        """把控件坐标换算成画面坐标。

        ``clamp_to_video=True``（默认）要求点落在画面区域内 —— 用于"新增顶点"，点在
        黑边上不该凭空加点。``False`` 时按同一个线性映射外推，于是画面外的顶点也能被
        拖动、也能被拖回画面里。
        """
        if self._frame_size[0] == 0 or self._frame_size[1] == 0:
            return None
        if self._video_rect.width() <= 0 or self._video_rect.height() <= 0:
            return None
        if clamp_to_video and not self._video_rect.contains(point):
            return None
        x = (point.x() - self._video_rect.x()) * self._frame_size[0] / self._video_rect.width()
        y = (point.y() - self._video_rect.y()) * self._frame_size[1] / self._video_rect.height()
        if not clamp_to_video:
            x = min(self._frame_size[0] * (1 + self.OFF_FRAME_MARGIN),
                    max(-self._frame_size[0] * self.OFF_FRAME_MARGIN, x))
            y = min(self._frame_size[1] * (1 + self.OFF_FRAME_MARGIN),
                    max(-self._frame_size[1] * self.OFF_FRAME_MARGIN, y))
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
            display, _ = self._handle_position(QPointF(point[0], point[1]))
            if (display - position).manhattanLength() <= 12:
                return index
        return None

    # 顶点标记离控件边缘的间距。
    HANDLE_INSET = 6.0

    def _handle_position(self, point: QPointF) -> tuple[QPointF, bool]:
        """顶点在控件里的落点，返回 (落点, 是否被夹到边缘)。

        为什么要夹：顶点坐标允许落在画面之外（换过不同分辨率的视频源、或者把顶点拖出去
        都会这样），而屏幕上的位置是按比例外推的 —— 窗口不大时，画面外的顶点会落到控件
        之外，那样它既看不见也点不到，就永远卡住了。夹到边缘并画成三角标记，至少让人
        看得见、抓得住、拖得回来。

        绘制与命中判定都走这里，两者必须用同一个位置，否则会出现"看得见却点不中"。
        """
        display = self._to_display(point)
        bounds = QRectF(self.rect()).adjusted(
            self.HANDLE_INSET,
            self.HANDLE_INSET,
            -self.HANDLE_INSET,
            -self.HANDLE_INSET,
        )
        if bounds.contains(display):
            return display, False
        return (
            QPointF(
                min(max(display.x(), bounds.left()), bounds.right()),
                min(max(display.y(), bounds.top()), bounds.bottom()),
            ),
            True,
        )

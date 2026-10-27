"""
calibration.py — Calibración manual de los puestos (sin detección automática de líneas).

Sobre un frame de la cámara, cada puesto se dibuja como un POLÍGONO (P01, P02, ...):

     N / "Nuevo puesto"         modo dibujo: cada clic izquierdo agrega un vértice
     Enter / clic derecho       cerrar el polígono (mínimo 3 vértices, se recomiendan 4)
     Backspace / Esc            deshacer el último vértice / cancelar el dibujo
     arrastrar un vértice       moverlo
     Ctrl + clic en un vértice  borrarlo (si quedan al menos 3)
     clic dentro de un puesto   seleccionarlo
     Supr                       eliminar el puesto seleccionado
     R                          rotar el eje del puesto (el eje contra el que se compara la línea de ruedas)
     "Dividir puesto"           parte el puesto seleccionado en N tramos iguales (uno por moto), con perspectiva
     Ctrl+S                     guardar en el JSON de calibración

Se puede abrir sola, sobre una foto o un video, y guardar en un archivo propio (no toca el de config.json):

     python calibration.py --imagen "foto_camara.jpg" --salida config/camara_seguridad.json

Zoom y desplazamiento (para marcar con precisión): rueda del ratón = zoom sobre el cursor · botón central o clic
sostenido en el fondo = desplazar · doble clic en el fondo o tecla 0 = ver la imagen completa.

Puntos del piso (opcional, mide los ángulos como si se mirara desde arriba):

     G / "Puntos del piso"      modo piso: clic en las 4 esquinas de un rectángulo REAL del piso, en orden
                                alrededor (1 → 2 → 3 → 4); luego se pide su tamaño real
                                (ancho = lado 1→2, largo = lado 2→3)
     arrastrar una esquina      moverla (se ve una cuadrícula que debe seguir las líneas del piso)
     Backspace / Esc            deshacer el último punto / cancelar
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from analysis.geometry import angular_error_deg
from analysis.parking_spaces import (ParkingLayout, ParkingSpace, clean_polygon_points, ground_quad_error,
                                     make_thumbnail, parallelogram_angle, split_space_points)

log = logging.getLogger("vimop.calibration")

GROUND_RGB = (0, 229, 255)                  # celeste: los puntos del piso se distinguen de los puestos
_PALETTE = [(0, 230, 118), (41, 182, 246), (255, 214, 0), (255, 112, 67), (171, 71, 188),
            (38, 198, 218), (156, 204, 101), (255, 167, 38), (236, 64, 122), (126, 87, 194)]


class CalibrationCanvas(QWidget):
    changed = Signal()
    selection_changed = Signal(str)
    ground_ready = Signal(list)                # 4 puntos válidos: falta pedir el tamaño real
    HANDLE = 9

    def __init__(self, image_bgr: np.ndarray, layout: ParkingLayout, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.layout_data = layout
        self.selected: str | None = None
        self.drawing = False
        self.current: list[tuple[float, float]] = []
        self.drag: tuple[str, int] | None = None
        self.hover: QPointF | None = None
        self.ground_mode = False                       # dibujando los 4 puntos del piso
        self.ground_current: list[tuple[float, float]] = []
        self.ground_drag: int | None = None            # esquina del piso que se arrastra
        self.zoom = 1.0                                # 1 = imagen completa en la ventana
        self.pan = QPointF(0, 0)                       # desplazamiento en píxeles de pantalla
        self.pan_drag: tuple[QPointF, QPointF] | None = None
        self.set_image(image_bgr)

    # ---- imagen / transformaciones ----------------------------------------
    def set_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.frame_bgr = image_bgr.copy()          # para guardar la miniatura del punto de vista
        self.img_w, self.img_h = w, h
        self.layout_data.scale_points((w, h))          # si la calibración era de otra resolución
        self.reset_view()

    # ---- zoom y desplazamiento ------------------------------------------------------
    ZOOM_MIN, ZOOM_MAX = 1.0, 12.0

    def reset_view(self) -> None:
        self.zoom, self.pan, self.pan_drag = 1.0, QPointF(0, 0), None
        self.update()

    def _base_scale(self) -> float:
        return min(self.width() / self.img_w, self.height() / self.img_h)

    def _tf(self) -> tuple[float, float, float]:
        s = self._base_scale() * self.zoom
        return (s, (self.width() - self.img_w * s) / 2 + self.pan.x(),
                (self.height() - self.img_h * s) / 2 + self.pan.y())

    def zoom_at(self, pos: QPointF, factor: float) -> None:
        """Zoom que deja quieto el punto de la imagen que está bajo el cursor."""
        new_zoom = min(max(self.zoom * factor, self.ZOOM_MIN), self.ZOOM_MAX)
        if new_zoom == self.zoom:
            return
        s0, ox0, oy0 = self._tf()
        ix, iy = (pos.x() - ox0) / s0, (pos.y() - oy0) / s0            # punto de la imagen bajo el cursor
        self.zoom = new_zoom
        s1 = self._base_scale() * self.zoom
        self.pan = QPointF(pos.x() - ix * s1 - (self.width() - self.img_w * s1) / 2,
                           pos.y() - iy * s1 - (self.height() - self.img_h * s1) / 2)
        self._clamp_pan()
        self.update()

    def _clamp_pan(self) -> None:
        """No dejar que la imagen se salga por completo de la ventana."""
        s = self._base_scale() * self.zoom
        max_x, max_y = abs(self.img_w * s - self.width()) / 2 + 40, abs(self.img_h * s - self.height()) / 2 + 40
        self.pan = QPointF(min(max(self.pan.x(), -max_x), max_x), min(max(self.pan.y(), -max_y), max_y))

    def wheelEvent(self, ev) -> None:
        self.zoom_at(ev.position(), 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2)
        ev.accept()

    def to_widget(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._tf()
        return QPointF(ox + x * s, oy + y * s)

    def to_image(self, p: QPointF) -> tuple[float, float]:
        s, ox, oy = self._tf()
        return (float(np.clip((p.x() - ox) / s, 0, self.img_w - 1)),
                float(np.clip((p.y() - oy) / s, 0, self.img_h - 1)))

    # ---- acciones ------------------------------------------------------------
    def start_space(self) -> None:
        self.drawing, self.current = True, []
        self.select(None)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()

    def cancel(self) -> None:
        self.drawing, self.current = False, []
        self.ground_mode, self.ground_current = False, []
        self.unsetCursor()
        self.update()

    # ---- puntos del piso -----------------------------------------------------------
    def start_ground(self) -> None:
        self.drawing, self.current = False, []
        self.ground_mode, self.ground_current = True, []
        self.select(None)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.update()

    def add_ground_point(self, x: float, y: float) -> str | None:
        """Añade una esquina; con la 4.ª valida el cuadrilátero. Devuelve el mensaje de error, si lo hay."""
        self.ground_current.append((float(x), float(y)))
        self.update()
        if len(self.ground_current) < 4:
            return None
        pts = list(self.ground_current)
        problem = ground_quad_error(pts)
        self.ground_current = []
        if problem:
            return problem
        self.ground_mode = False
        self.unsetCursor()
        self.ground_ready.emit(pts)
        return None

    def set_ground_quad(self, points, size: tuple[float, float], angle: float = 90.0) -> None:
        self.layout_data.ground_quad = ([(float(x), float(y)) for x, y in points], (float(size[0]), float(size[1])))
        self.layout_data.ground_angle = float(angle)
        self.layout_data.invalidate()
        self.changed.emit()
        self.update()

    def clear_ground(self) -> None:
        self.layout_data.ground_quad = None
        self.layout_data.ground_angle = 90.0
        self.layout_data.invalidate()
        self.ground_mode, self.ground_current = False, []
        self.unsetCursor()
        self.changed.emit()
        self.update()

    def _ground_vertex_at(self, pos: QPointF) -> int | None:
        quad = self.layout_data.ground_quad
        if not quad:
            return None
        for i, (x, y) in enumerate(quad[0]):
            if (self.to_widget(x, y) - pos).manhattanLength() <= self.HANDLE * 1.6:
                return i
        return None

    def finish_space(self) -> str | None:
        if not self.drawing:
            return None
        if len(self.current) < 3:
            QMessageBox.warning(self, "Puesto incompleto", "Un puesto necesita al menos 3 vértices (se recomiendan 4).")
            return None
        sid = self.layout_data.next_id()
        self.layout_data.add(ParkingSpace(sid, clean_polygon_points(self.current)))
        self.cancel()
        self.select(sid)
        self.changed.emit()
        return sid

    def select(self, sid: str | None) -> None:
        self.selected = sid
        self.selection_changed.emit(sid or "")
        self.update()

    def delete_selected(self) -> None:
        if self.selected:
            self.layout_data.remove(self.selected)
            self.select(None)
            self.changed.emit()

    def rotate_selected_axis(self) -> None:
        sp = self.layout_data.get(self.selected) if self.selected else None
        if sp:
            sp.axis_swap = not sp.axis_swap
            self.layout_data.invalidate()
            self.changed.emit()
            self.update()

    def split_selected(self, n: int) -> list[str]:
        """
        Parte el puesto seleccionado en `n` tramos iguales a lo largo de su eje (uno por moto). Los tramos se llaman
        P04-1, P04-2... y conservan el eje del puesto original. Devuelve los nombres nuevos.
        """
        sp = self.layout_data.get(self.selected) if self.selected else None
        if sp is None:
            raise ValueError("Seleccione primero el puesto que quiere dividir.")
        parent = sp.geometry()
        parts = split_space_points(sp.points, n, parent.angle)
        new_ids = []
        self.layout_data.remove(sp.space_id)
        for k, quad in enumerate(parts, start=1):
            part = ParkingSpace(f"{sp.space_id}-{k}", quad, False)
            # un tramo más corto que ancho tendría su "eje largo" cruzado: se fuerza la dirección del puesto original
            if angular_error_deg(part.geometry().angle, parent.angle) > 45.0:
                part.axis_swap = True
            self.layout_data.add(part)
            new_ids.append(part.space_id)
        self.select(new_ids[0])
        self.changed.emit()
        return new_ids

    def _vertex_at(self, pos: QPointF) -> tuple[str, int] | None:
        for sp in sorted(self.layout_data.spaces, key=lambda s: s.space_id != self.selected):
            for i, (x, y) in enumerate(sp.points):
                if (self.to_widget(x, y) - pos).manhattanLength() <= self.HANDLE * 1.4:
                    return sp.space_id, i
        return None

    def _space_at(self, pos: QPointF) -> str | None:
        x, y = self.to_image(pos)
        for sp in reversed(self.layout_data.spaces):
            if cv2.pointPolygonTest(np.asarray(sp.points, dtype=np.float32), (x, y), False) >= 0:
                return sp.space_id
        return None

    # ---- ratón / teclado --------------------------------------------------------
    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.pan_drag = (pos, QPointF(self.pan))
            return
        if self.ground_mode:
            if ev.button() == Qt.MouseButton.LeftButton:
                problem = self.add_ground_point(*self.to_image(pos))
                if problem:
                    QMessageBox.warning(self, "Puntos del piso", problem + "\n\nVuelva a marcar las 4 esquinas.")
            return
        if self.drawing:
            if ev.button() == Qt.MouseButton.LeftButton:
                self.current.append(self.to_image(pos))
                self.update()
            elif ev.button() == Qt.MouseButton.RightButton:
                self.finish_space()
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        ground_hit = self._ground_vertex_at(pos)
        hit = self._vertex_at(pos)
        # una esquina del piso gana si coincide con el vértice de un puesto, salvo que ese puesto esté seleccionado
        if ground_hit is not None and not (hit is not None and hit[0] == self.selected):
            self.ground_drag = ground_hit
            return
        if hit:
            sid, idx = hit
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                sp = self.layout_data.get(sid)
                if sp and len(sp.points) > 3:
                    del sp.points[idx]
                    self.layout_data.invalidate()
                    self.changed.emit()
                else:
                    QMessageBox.information(self, "Vértice", "Un puesto debe conservar al menos 3 vértices.")
                self.select(sid)
                return
            self.drag = hit
            self.select(sid)
            return
        target = self._space_at(pos)
        self.select(target)
        if target is None:                                              # clic sostenido en el fondo: desplazar
            self.pan_drag = (pos, QPointF(self.pan))

    def mouseMoveEvent(self, ev):
        self.hover = ev.position()
        if self.pan_drag is not None:
            start, pan0 = self.pan_drag
            self.pan = QPointF(pan0.x() + self.hover.x() - start.x(), pan0.y() + self.hover.y() - start.y())
            self._clamp_pan()
            self.update()
            return
        if self.ground_drag is not None and self.layout_data.ground_quad:
            pts, size = self.layout_data.ground_quad
            pts = list(pts)
            pts[self.ground_drag] = self.to_image(self.hover)
            self.layout_data.ground_quad = (pts, size)
            self.layout_data.invalidate()
        if self.drag:
            sid, idx = self.drag
            sp = self.layout_data.get(sid)
            if sp:
                sp.points[idx] = self.to_image(self.hover)
                self.layout_data.invalidate()
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.pan_drag is not None:
            self.pan_drag = None
            return
        if self.ground_drag is not None:
            self.ground_drag = None
            pts, size = self.layout_data.ground_quad
            if ground_quad_error(pts):                       # se arrastró hasta cruzar los lados: no guardar así
                QMessageBox.warning(self, "Puntos del piso", ground_quad_error(pts) + "\n\nMueva la esquina a otro lugar.")
            self.changed.emit()
            return
        if self.drag:
            self.drag = None
            self.changed.emit()

    def mouseDoubleClickEvent(self, ev):
        if self.drawing:
            self.finish_space()
        elif not self.ground_mode and self._space_at(ev.position()) is None:
            self.reset_view()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key.Key_0 and (self.drawing or self.ground_mode):
            self.reset_view()
            return
        if self.ground_mode:
            if key == Qt.Key.Key_Escape:
                self.cancel()
            elif key == Qt.Key.Key_Backspace and self.ground_current:
                self.ground_current.pop()
                self.update()
            return
        if self.drawing:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.finish_space()
            elif key == Qt.Key.Key_Escape:
                self.cancel()
            elif key == Qt.Key.Key_Backspace and self.current:
                self.current.pop()
                self.update()
            return
        if key == Qt.Key.Key_0:
            self.reset_view()
        elif key == Qt.Key.Key_N:
            self.start_space()
        elif key == Qt.Key.Key_G:
            self.start_ground()
        elif key == Qt.Key.Key_Delete:
            self.delete_selected()
        elif key == Qt.Key.Key_R:
            self.rotate_selected_axis()
        else:
            super().keyPressEvent(ev)

    # ---- dibujo -------------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        s, ox, oy = self._tf()
        p.drawImage(QRectF(ox, oy, self.img_w * s, self.img_h * s), self.image)
        font = QFont("Segoe UI")
        font.setPixelSize(15)
        font.setBold(True)
        p.setFont(font)
        for i, sp in enumerate(self.layout_data.spaces):
            rgb = _PALETTE[i % len(_PALETTE)]
            sel = sp.space_id == self.selected
            p.setPen(QPen(QColor(*rgb), 3 if sel else 2))
            p.setBrush(QBrush(QColor(*rgb, 90 if sel else 45)))
            p.drawPolygon(QPolygonF([self.to_widget(x, y) for x, y in sp.points]))
            g = sp.geometry()
            wa, wb = self.to_widget(*g.axis[0]), self.to_widget(*g.axis[1])
            p.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.PenStyle.DashLine))
            p.drawLine(wa, wb)
            self._arrow(p, wa, wb)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.setBrush(QBrush(QColor(255, 255, 255) if sel else QColor(*rgb)))
            r = self.HANDLE / 2 + (1 if sel else 0)
            for x, y in sp.points:
                p.drawEllipse(self.to_widget(x, y), r, r)
            c = self.to_widget(*g.center)
            box = QRectF(c.x() - 24, c.y() - 12, 48, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 170)))
            p.drawRoundedRect(box, 5, 5)
            p.setPen(QColor(*rgb))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, sp.space_id)

        self._paint_ground(p, font)

        if self.drawing and self.current:
            pts = [self.to_widget(x, y) for x, y in self.current]
            p.setPen(QPen(QColor(255, 235, 59), 2))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])
            if self.hover is not None:
                p.setPen(QPen(QColor(255, 235, 59, 150), 1, Qt.PenStyle.DashLine))
                p.drawLine(pts[-1], self.hover)
                if len(pts) >= 2:
                    p.drawLine(self.hover, pts[0])
            p.setBrush(QBrush(QColor(255, 235, 59)))
            for q in pts:
                p.drawEllipse(q, 5, 5)

        if self.ground_mode:
            help_text = (f"PISO: clic en la esquina {len(self.ground_current) + 1} de 4 de un rectángulo o paralelogramo "
                         "real del piso (en orden alrededor) · rueda = zoom · Backspace = deshacer · Esc = cancelar")
        elif self.drawing:
            help_text = ("DIBUJANDO: clic = vértice · rueda = zoom (botón central = mover) · Enter/clic derecho = cerrar · "
                         "Backspace = deshacer · Esc = cancelar")
        else:
            help_text = ("N = nuevo puesto · G = puntos del piso · rueda = zoom · arrastrar vértice = mover · Ctrl+clic "
                         "vértice = borrar vértice · Supr = borrar puesto · R = rotar eje · Ctrl+S = guardar")
        f2 = QFont("Segoe UI")
        f2.setPixelSize(13)
        p.setFont(f2)
        r = QRectF(8, self.height() - 30, self.width() - 16, 24)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 185)))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor(255, 235, 59) if (self.drawing or self.ground_mode) else QColor(230, 230, 230))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, help_text)
        p.end()

    def _paint_ground(self, p: QPainter, font: QFont) -> None:
        """Rectángulo del piso con su cuadrícula: si las líneas siguen las del suelo, la perspectiva está bien."""
        color = QColor(*GROUND_RGB)
        quad = self.layout_data.ground_quad
        if quad:
            pts, (gw, gh) = quad
            src = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32)
            try:
                h = cv2.getPerspectiveTransform(src, np.asarray(pts, dtype=np.float32))
                p.setPen(QPen(QColor(*GROUND_RGB, 150), 1, Qt.PenStyle.DashLine))
                for t in np.linspace(0.0, 1.0, 5)[1:-1]:
                    for a, b in (((t, 0.0), (t, 1.0)), ((0.0, t), (1.0, t))):
                        q = cv2.perspectiveTransform(np.array([[a, b]], dtype=np.float32), h)[0]
                        p.drawLine(self.to_widget(*q[0]), self.to_widget(*q[1]))
            except cv2.error:
                pass
            wp = [self.to_widget(x, y) for x, y in pts]
            p.setPen(QPen(color, 3))
            p.setBrush(QBrush(QColor(*GROUND_RGB, 28)))
            p.drawPolygon(QPolygonF(wp))
            p.setBrush(QBrush(color))
            for i, q in enumerate(wp):
                p.setPen(QPen(QColor(0, 0, 0), 1))
                p.drawEllipse(q, self.HANDLE / 2 + 2, self.HANDLE / 2 + 2)
                p.setPen(QColor(0, 0, 0))
                p.drawText(QRectF(q.x() - 8, q.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(i + 1))
            mid = (wp[0] + wp[2]) / 2
            box = QRectF(mid.x() - 70, mid.y() - 12, 140, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 175)))
            p.drawRoundedRect(box, 5, 5)
            p.setFont(font)
            p.setPen(color)
            skew = self.layout_data.ground_angle
            p.drawText(box, Qt.AlignmentFlag.AlignCenter,
                       f"PISO {gw:g} × {gh:g} m" + ("" if abs(skew - 90.0) < 0.05 else f" · {skew:.0f}°"))
        if self.ground_mode and self.ground_current:
            wp = [self.to_widget(x, y) for x, y in self.ground_current]
            p.setPen(QPen(color, 2))
            for i in range(len(wp) - 1):
                p.drawLine(wp[i], wp[i + 1])
            if self.hover is not None:
                p.setPen(QPen(QColor(*GROUND_RGB, 150), 1, Qt.PenStyle.DashLine))
                p.drawLine(wp[-1], self.hover)
            p.setBrush(QBrush(color))
            for i, q in enumerate(wp):
                p.setPen(QPen(QColor(0, 0, 0), 1))
                p.drawEllipse(q, 6, 6)
                p.setPen(QColor(0, 0, 0))
                p.drawText(QRectF(q.x() - 8, q.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(i + 1))

    @staticmethod
    def _arrow(p: QPainter, a: QPointF, b: QPointF) -> None:
        v = np.array([b.x() - a.x(), b.y() - a.y()])
        n = np.linalg.norm(v)
        if n < 1:
            return
        v /= n
        perp = np.array([-v[1], v[0]])
        tip = np.array([b.x(), b.y()])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 220)))
        p.drawPolygon(QPolygonF([QPointF(*tip), QPointF(*(tip - v * 12 + perp * 6)),
                                 QPointF(*(tip - v * 12 - perp * 6))]))


class CalibrationDialog(QDialog):
    """Ventana de calibración. Si se guarda, `result_layout` contiene la nueva calibración."""

    def __init__(self, image_bgr: np.ndarray, layout: ParkingLayout | None, save_path: Path, cfg,
                 frame_provider=None, source_desc: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("VIMOP — Calibrar parqueadero")
        self.resize(1440, 880)
        self.cfg = cfg
        self.save_path = Path(save_path)
        self.frame_provider = frame_provider
        self.saved = self.dirty = False
        self.result_layout: ParkingLayout | None = None
        work = ParkingLayout([ParkingSpace(s.space_id, list(s.points), s.axis_swap) for s in layout.spaces]
                             if layout else [], layout.reference_size if layout else None, source_desc)
        work.ground_quad = layout.ground_quad if layout else None            # los puntos del piso se conservan
        work.ground_angle = layout.ground_angle if layout else 90.0
        self.canvas = CalibrationCanvas(image_bgr, work)

        side = QVBoxLayout()
        t = QLabel("CALIBRACIÓN DE PUESTOS")
        t.setObjectName("sectionTitle")
        side.addWidget(t)
        info = QLabel("Dibuje cada puesto sobre las líneas blancas del piso: pulse N, haga clic en sus "
                      "4 esquinas y pulse Enter. Cada moto se evalúa con la línea de sus dos ruedas sobre estos polígonos.")
        info.setWordWrap(True)
        side.addWidget(info)
        box = QGroupBox("Puestos")
        bl = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.currentTextChanged.connect(lambda sid: self.canvas.select(sid) if sid else None)
        bl.addWidget(self.list, 1)
        for text, slot in (("➕  Nuevo puesto (N)", self.canvas.start_space),
                           ("✔  Terminar puesto (Enter)", self.canvas.finish_space),
                           ("↻  Rotar eje (R)", self.canvas.rotate_selected_axis),
                           ("✂  Dividir puesto en partes...", self._split),
                           ("✎  Renombrar puesto", self._rename),
                           ("🗑  Eliminar puesto (Supr)", self.canvas.delete_selected)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bl.addWidget(b)
        self.space_info = QLabel()
        self.space_info.setWordWrap(True)
        bl.addWidget(self.space_info)
        side.addWidget(box, 1)
        ground = QGroupBox("Puntos del piso (opcional)")
        gl = QVBoxLayout(ground)
        hint = QLabel("Sirven para medir los ángulos como si se mirara desde arriba. Use un puesto pintado de esquinas "
                      "conocidas: dibújelo con N (o seleccione uno ya dibujado) y pulse \"Usar como referencia\": el "
                      "mismo puesto queda calibrado y da las medidas del piso en un solo paso.")
        hint.setWordWrap(True)
        gl.addWidget(hint)
        use_space = QPushButton("📐  Usar el puesto seleccionado como referencia")
        use_space.clicked.connect(self._use_selected_as_ground)
        gl.addWidget(use_space)
        for text, slot in (("✥  Marcar 4 puntos sueltos (G)", self.canvas.start_ground),
                           ("🗑  Quitar puntos del piso", self.canvas.clear_ground)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            gl.addWidget(b)
        self.ground_info = QLabel()
        self.ground_info.setWordWrap(True)
        gl.addWidget(self.ground_info)
        side.addWidget(ground)
        for text, slot in (("📷  Tomar nuevo frame de la fuente", self._new_frame),
                           ("🖼  Cargar imagen...", self._load_image),
                           ("📂  Cargar calibración JSON...", self._load_json),
                           ("🧹  Borrar todos los puestos", self._clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            side.addWidget(b)
        save = QPushButton("💾  GUARDAR CALIBRACIÓN (Ctrl+S)")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._save)
        side.addWidget(save)
        close = QPushButton("Cerrar")
        close.clicked.connect(self.close)
        side.addWidget(close)

        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(330)
        root = QHBoxLayout(self)
        root.addWidget(self.canvas, 1)
        root.addWidget(panel)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        self.canvas.changed.connect(self._on_changed)
        self.canvas.selection_changed.connect(self._on_selection)
        self.canvas.ground_ready.connect(self._ask_ground_size)
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        self.list.blockSignals(True)
        self.list.clear()
        for sp in self.canvas.layout_data.spaces:
            self.list.addItem(sp.space_id)
        if self.canvas.selected:
            items = self.list.findItems(self.canvas.selected, Qt.MatchFlag.MatchExactly)
            if items:
                self.list.setCurrentItem(items[0])
        self.list.blockSignals(False)
        self._on_selection(self.canvas.selected or "")
        quad = self.canvas.layout_data.ground_quad
        angle = self.canvas.layout_data.ground_angle
        shape = "rectángulo" if abs(angle - 90.0) < 0.05 else f"paralelogramo (ángulo {angle:.1f}°)"
        self.ground_info.setText(
            f"Activos: {shape} de {quad[1][0]:g} × {quad[1][1]:g} m. Los ángulos se miden sin perspectiva."
            if quad else "Sin puntos del piso: los ángulos se miden en la imagen (aproximados).")

    def _on_changed(self):
        self.dirty = True
        self._refresh()

    def _use_selected_as_ground(self) -> None:
        """El puesto seleccionado (sus 4 esquinas, en el orden en que se dibujó) pasa a ser la referencia del piso."""
        sid = self.canvas.selected
        sp = self.canvas.layout_data.get(sid) if sid else None
        if sp is None:
            QMessageBox.information(self, "Puntos del piso", "Seleccione primero el puesto que quiere usar como "
                                                              "referencia (dibújelo con N si todavía no existe).")
            return
        pts = clean_polygon_points(sp.points)
        if len(pts) != 4:
            QMessageBox.warning(self, "Puntos del piso", f"{sid} tiene {len(pts)} esquinas: se necesitan exactamente "
                                                          "4 (un rectángulo o un romboide, sin dividir).")
            return
        problem = ground_quad_error(pts)
        if problem:
            QMessageBox.warning(self, "Puntos del piso", f"{sid} no sirve como referencia: {problem}")
            return
        self.canvas.select(sid)                                    # conserva la selección al reabrir el diálogo
        self._ask_ground_size(pts)

    def _ask_ground_size(self, points: list) -> None:
        """Pide el tamaño real del rectángulo marcado: ancho = lado 1→2, largo = lado 2→3."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Tamaño real del rectángulo del piso")
        form = QFormLayout(dlg)
        form.addRow(QLabel("Mida en el piso la figura que marcó (en metros).\n"
                           "Si es un paralelogramo (puesto en romboide), mida también sus diagonales;\n"
                           "si es un rectángulo, deje las diagonales en 0."))
        spins = []
        fields = (("Ancho: lado de la esquina 1 a la 2", 1.20, 0.05), ("Largo: lado de la esquina 2 a la 3", 2.20, 0.05),
                  ("Diagonal de la esquina 1 a la 3 (0 = rectángulo)", 0.0, 0.0),
                  ("Diagonal de la esquina 2 a la 4 (opcional)", 0.0, 0.0))
        for label, default, low in fields:
            sp = QDoubleSpinBox()
            sp.setRange(low, 1000.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.1)
            sp.setSuffix(" m")
            sp.setValue(default)
            form.addRow(label, sp)
            spins.append(sp)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        while dlg.exec() == QDialog.DialogCode.Accepted:
            width, length, d13, d24 = (s.value() for s in spins)
            try:
                angle = parallelogram_angle(width, length, d13 or None, d24 or None)
            except ValueError as exc:
                QMessageBox.warning(self, "Puntos del piso", f"Las medidas no sirven: {exc}.")
                continue
            self.canvas.set_ground_quad(points, (width, length), angle)
            return
        self.canvas.update()

    def _on_selection(self, sid: str):
        sp = self.canvas.layout_data.get(sid) if sid else None
        if not sp:
            self.space_info.setText("Seleccione un puesto para ver sus medidas.")
            return
        g = sp.geometry()
        self.space_info.setText(f"{sp.space_id}: bbox {g.width:.0f} x {g.height:.0f} px · "
                                f"ancho/alto {g.width / max(g.height, 1e-9):.2f} · eje {g.angle:.0f}°")

    def _split(self):
        sid = self.canvas.selected
        sp = self.canvas.layout_data.get(sid) if sid else None
        if sp is None:
            QMessageBox.information(self, "Dividir puesto", "Seleccione primero el puesto que quiere dividir.")
            return
        if len(sp.points) != 4:
            QMessageBox.information(self, "Dividir puesto", "Solo se pueden dividir puestos de 4 esquinas.")
            return
        n, ok = QInputDialog.getInt(self, "Dividir puesto", f"¿Cuántas motos caben en fila dentro de {sid}?\n"
                                    "(se parte en tramos iguales, uno por moto)", 2, 2, 20)
        if not ok:
            return
        try:
            self.canvas.split_selected(n)
        except ValueError as exc:
            QMessageBox.warning(self, "Dividir puesto", str(exc))

    def _rename(self):
        sid = self.canvas.selected
        if not sid:
            QMessageBox.information(self, "Renombrar", "Seleccione primero un puesto.")
            return
        new, ok = QInputDialog.getText(self, "Renombrar puesto", "Nuevo nombre:", text=sid)
        if ok and new:
            try:
                self.canvas.layout_data.rename(sid, new)
                self.canvas.select(new.strip().upper())
                self._on_changed()
            except ValueError as exc:
                QMessageBox.warning(self, "Renombrar", str(exc))

    def _new_frame(self):
        if self.frame_provider is None:
            QMessageBox.information(self, "Frame", "No hay una fuente de video configurada.")
            return
        try:
            self.canvas.set_image(self.frame_provider())
        except Exception as exc:
            QMessageBox.warning(self, "Frame", f"No se pudo obtener un frame:\n{exc}")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Imagen del parqueadero", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Imagen", "No se pudo leer la imagen.")
            return
        self.canvas.set_image(img)

    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar calibración", str(self.save_path.parent), "JSON (*.json)")
        if not path:
            return
        try:
            loaded = ParkingLayout.load(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Calibración", f"Archivo inválido:\n{exc}")
            return
        loaded.scale_points((self.canvas.img_w, self.canvas.img_h))
        self.canvas.layout_data = loaded
        self.canvas.select(None)
        self._on_changed()

    def _clear_all(self):
        if QMessageBox.question(self, "Borrar", "¿Borrar TODOS los puestos?") == QMessageBox.StandardButton.Yes:
            for sp in list(self.canvas.layout_data.spaces):
                self.canvas.layout_data.remove(sp.space_id)
            self.canvas.select(None)
            self._on_changed()

    def _save(self):
        lay = self.canvas.layout_data
        if lay.is_empty():
            QMessageBox.warning(self, "Guardar", "No hay puestos para guardar.")
            return
        lay.reference_thumbnail = make_thumbnail(self.canvas.frame_bgr)   # punto de vista (modo foto)
        try:
            lay.save(self.save_path)
        except OSError as exc:
            QMessageBox.critical(self, "Guardar", f"No se pudo guardar:\n{exc}")
            return
        self.saved, self.dirty, self.result_layout = True, False, lay
        QMessageBox.information(self, "Calibración guardada", f"{len(lay.spaces)} puestos guardados en:\n{self.save_path}")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape and not self.canvas.drawing and not self.canvas.ground_mode:
            self.close()
            return
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        if self.dirty and not self.canvas.layout_data.is_empty():
            ans = QMessageBox.question(self, "Cambios sin guardar", "¿Guardar los cambios antes de cerrar?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                       | QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Cancel:
                ev.ignore()
                return
            if ans == QMessageBox.StandardButton.Yes:
                self._save()
        if self.saved:
            self.accept()
        else:
            self.reject()
        ev.accept()


def load_reference_image(path: Path) -> np.ndarray:
    """Foto o primer fotograma de un video, para calibrar sobre una imagen de LA MISMA cámara."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"No se pudo leer el primer fotograma de {path.name}")
        return frame
    from core.image_io import ImageLoadError, read_image
    try:
        return read_image(path).image
    except ImageLoadError as exc:
        raise ValueError(str(exc)) from exc


def open_standalone(image_path: str | Path, out_path: str | Path, cfg=None) -> CalibrationDialog:
    """Ventana de calibración independiente. Si `out_path` ya existe, se continúa desde ahí."""
    from config import load_config
    cfg = cfg or load_config()
    image = load_reference_image(Path(image_path))
    out_path = Path(out_path)
    layout = None
    if out_path.exists():
        layout = ParkingLayout.load(out_path)
    return CalibrationDialog(image, layout, out_path, cfg, source_desc=f"Imagen: {Path(image_path).name}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Calibrar los puestos sobre una foto o video de la cámara")
    ap.add_argument("--imagen", required=True, help="foto o video de la cámara (con los puestos a la vista)")
    ap.add_argument("--salida", default="config/camara_seguridad.json", help="archivo de calibración a crear/continuar")
    args = ap.parse_args(argv)
    from PySide6.QtWidgets import QApplication
    from config import BASE_DIR
    from ui.assets import app_icon
    from ui.theme import STYLE
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setWindowIcon(app_icon())
    out = Path(args.salida)
    out = out if out.is_absolute() else BASE_DIR / out
    try:
        dlg = open_standalone(args.imagen, out)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    dlg.exec()
    if dlg.saved:
        print(f"Calibración guardada en {out}")
        print(f'Para usarla:  python main.py --headless --video --calibration "{out}"   (o cárguela desde la interfaz)')
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

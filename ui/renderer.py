"""
ui/renderer.py — Dibujo de resultados sobre el frame con QPainter.

Se usa Qt en vez de cv2.putText porque OpenCV no dibuja tildes ("PÉSIMO", "MOVIÉNDOSE").
El QImage resultante se muestra en la interfaz y también se guarda en capturas y videos.

    línea de ejes R–F de cada moto con el color de su categoría (BGR de utils.CLASS_COLORS_BGR):
        EXCELENTE verde · BUENO azul · REGULAR amarillo · MALO naranja · PÉSIMO rojo
        círculo lleno = rueda trasera · rombo = rueda delantera (con el respaldo por máscara no se
        distingue cuál es cuál: son los dos extremos)
    etiqueta:  "MOTO #07 · 87.4 · BUENO" en el punto medio M de la línea, desplazada para no tapar
               (en movimiento: gris, "MOTO #07 · MOVIÉNDOSE")
    bbox:      tenue, solo como referencia (SHOW_BBOX)
    puestos:   polígono + eje discontinuo tenue + "P04 LIBRE" / "P04 OCUPADO"
    alertas:   banner rojo arriba; banner y línea de las motos con alerta PARPADEAN
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF

from analysis.state_machine import EVALUATED_STATES, ST_PARKING
from core.pipeline import FrameResult, MotoView
from utils import FREE_SPACE_BGR, MOVING_BGR, OCCUPIED_SPACE_BGR, PERSON_BGR, class_color_bgr


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)
    return cv2.cvtColor(arr[:, : w * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)


def qcolor(bgr, alpha: int = 255) -> QColor:
    return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]), alpha)


def text_color_for(bgr) -> QColor:
    lum = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return QColor(20, 20, 20) if lum > 150 else QColor(255, 255, 255)


def moto_color(info: dict) -> tuple[int, int, int]:
    if info.get("clasificacion") and info.get("estado_movimiento") in EVALUATED_STATES:
        return class_color_bgr(info["clasificacion"])
    return MOVING_BGR


def moto_label(info: dict) -> str:
    label = f"MOTO #{info['id']:02d}"
    if info.get("puntaje") is not None and info["estado_movimiento"] in EVALUATED_STATES:
        label += f" · {info['puntaje']:.1f} · {info['clasificacion']}"
        if info["estado_movimiento"] == ST_PARKING:
            label += " (estacionando)"
    else:
        label += f" · {info['estado_movimiento']}"
    if not info.get("visible", True):
        label += " (oculta)"
    return label


class FrameRenderer:
    def __init__(self, show_masks: bool = True, blink: bool = True, show_bbox: bool = True):
        self.show_masks = show_masks
        self.blink = blink
        self.show_bbox = show_bbox

    def render(self, res: FrameResult, selected_id: int | None = None, fps: float = 0.0) -> QImage:
        img = bgr_to_qimage(res.frame)
        w, h = img.width(), img.height()
        k = max(0.55, w / 1280.0)
        blink_on = (not self.blink) or int(time.monotonic() * 2) % 2 == 0
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        for sp in res.spaces:
            self._space(p, sp, k, w, h)
        for det in res.persons:
            self._person(p, det, k, w, h)
        for view in sorted(res.motos, key=lambda v: v.info["id"] == selected_id):
            alert = view.info["id"] in res.alert_ids
            self._moto(p, view, k, w, h, view.info["id"] == selected_id, alert, blink_on)
        self._hud(p, res, k, w, h, fps, blink_on)
        p.end()
        return img

    # ------------------------------------------------------------------
    @staticmethod
    def _font(k: float, size: float, bold: bool = False) -> QFont:
        f = QFont("Segoe UI")
        f.setPixelSize(max(9, int(size * k)))
        f.setBold(bold)
        return f

    def _pill(self, p, x, y, lines, bg: QColor, fg: QColor, k, cw, ch) -> None:
        pad = 5 * k
        ms = [QFontMetricsF(f) for _, f in lines]
        width = max(m.horizontalAdvance(t) for (t, _), m in zip(lines, ms)) + 2 * pad
        height = sum(m.height() for m in ms) + 2 * pad
        x, y = min(max(0.0, x), cw - width), min(max(0.0, y), ch - height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(QRectF(x, y, width, height), 4 * k, 4 * k)
        p.setPen(fg)
        cy = y + pad
        for (text, font), m in zip(lines, ms):
            p.setFont(font)
            p.drawText(QPointF(x + pad, cy + m.ascent()), text)
            cy += m.height()

    def _space(self, p, sp: dict, k, w, h) -> None:
        occupied = bool(sp["occupants"])
        bgr = OCCUPIED_SPACE_BGR if occupied else FREE_SPACE_BGR
        p.setPen(QPen(qcolor(bgr, 230), max(1.5, 2 * k)))
        p.setBrush(QBrush(qcolor(bgr, 38 if occupied else 26)))
        p.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in sp["polygon"]]))
        (ax, ay), (bx, by) = sp["axis"]
        p.setPen(QPen(QColor(255, 255, 255, 110), max(1.0, 1.2 * k), Qt.PenStyle.DashLine))
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))
        cx, cy = sp["center"]
        self._pill(p, cx - 30 * k, cy - 16 * k, [(f"{sp['id']} {'OCUPADO' if occupied else 'LIBRE'}",
                                                 self._font(k, 12, True))],
                   qcolor(bgr, 205), QColor(15, 15, 15), k, w, h)

    def _person(self, p, det, k, w, h) -> None:
        x1, y1, x2, y2 = det.bbox
        p.setPen(QPen(qcolor(PERSON_BGR), max(1.5, 2 * k)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        self._pill(p, x1, y1 - 20 * k, [(f"PERSONA {det.confidence:.2f}", self._font(k, 11, True))],
                   qcolor(PERSON_BGR, 220), QColor(255, 255, 255), k, w, h)

    def _moto(self, p, view: MotoView, k, w, h, selected: bool, alert: bool, blink_on: bool) -> None:
        info, det, a = view.info, view.detection, view.analysis
        bgr = moto_color(info)
        x1, y1, x2, y2 = info["bbox"]
        if det is not None and self.show_masks and det.mask is not None:     # solo con modelos '-seg'
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            path = QPainterPath()
            for c in contours:
                if len(c) >= 3:
                    path.addPolygon(QPolygonF([QPointF(float(q[0][0] + x1), float(q[0][1] + y1)) for q in c]))
                    path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qcolor(bgr, 55)))
            p.drawPath(path)
        has_axis = det is not None and a is not None
        dim = alert and not blink_on                         # parpadeo de la alerta
        if not has_axis or self.show_bbox:
            # con línea de ejes el bbox es solo una referencia tenue; sin ella es lo único que se ve
            w_box = (2 if selected else 1.2) * k if has_axis else (5 if selected else 3) * k
            alpha = 95 if has_axis else 255
            p.setPen(QPen(qcolor(bgr, alpha), w_box * (0.4 if dim and not has_axis else 1.0),
                          Qt.PenStyle.SolidLine if det is not None else Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        if selected:
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.2 * k, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(x1 - 4 * k, y1 - 4 * k, x2 - x1 + 8 * k, y2 - y1 + 8 * k))
        anchor = None
        if has_axis:
            anchor = self._axis(p, a, bgr, k, selected, dim)
        font = self._font(k, 14, True)
        lines = [(("⚠ " if alert else "") + moto_label(info), font)]
        if info.get("avisos") and "orientación no confiable" in info["avisos"]:
            lines.append(("⚠ orientación no confiable", self._font(k, 11)))
        if anchor is not None:
            self._pill_near(p, anchor, a, lines, qcolor(bgr, 235), text_color_for(bgr), k, w, h)
        else:
            lh = QFontMetricsF(font).height() + 10 * k
            ly = y1 - lh - 2 if y1 - lh - 2 >= 0 else y1 + 2
            self._pill(p, x1, ly, lines, qcolor(bgr, 235), text_color_for(bgr), k, w, h)

    def _axis(self, p, a, bgr, k, selected: bool, dim: bool) -> QPointF:
        """Línea R–F con un círculo en la trasera y un rombo en la delantera. Devuelve el punto medio M."""
        r, f = QPointF(*a.rear), QPointF(*a.front)
        width = (6 if selected else 4) * k * (0.4 if dim else 1.0)
        p.setPen(QPen(QColor(0, 0, 0, 150), width + 2 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(r, f)                                    # contorno oscuro: se lee sobre cualquier fondo
        p.setPen(QPen(qcolor(bgr), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(r, f)
        rad = 5.5 * k
        p.setPen(QPen(QColor(255, 255, 255), 1.5 * k))
        p.setBrush(QBrush(qcolor(bgr)))
        p.drawEllipse(r, rad, rad)                          # trasera: círculo
        p.drawPolygon(QPolygonF([QPointF(f.x(), f.y() - 1.3 * rad), QPointF(f.x() + 1.3 * rad, f.y()),
                                 QPointF(f.x(), f.y() + 1.3 * rad), QPointF(f.x() - 1.3 * rad, f.y())]))
        return QPointF(*a.center_img)

    def _pill_near(self, p, m: QPointF, a, lines, bg, fg, k, cw, ch) -> None:
        """Rótulo junto al punto medio M, desplazado en la normal de la línea para no taparla."""
        dx, dy = a.front[0] - a.rear[0], a.front[1] - a.rear[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        nx, ny = -dy / norm, dx / norm
        if ny > 0 or (abs(ny) < 1e-6 and nx < 0):           # preferir arriba (y menor) de la línea
            nx, ny = -nx, -ny
        pad = 5 * k
        ms = [QFontMetricsF(f) for _, f in lines]
        pw = max(m_.horizontalAdvance(t) for (t, _), m_ in zip(lines, ms)) + 2 * pad
        ph = sum(m_.height() for m_ in ms) + 2 * pad
        reach = abs(nx) * pw / 2 + abs(ny) * ph / 2 + 8 * k     # que el rectángulo no toque la línea
        cx, cy = m.x() + nx * reach, m.y() + ny * reach
        self._pill(p, cx - pw / 2, cy - ph / 2, lines, bg, fg, k, cw, ch)

    def _hud(self, p, res: FrameResult, k, w, h, fps: float, blink_on: bool) -> None:
        if res.alerts:
            text = res.alerts[0] + (f"   (+{len(res.alerts) - 1})" if len(res.alerts) > 1 else "")
            font = self._font(k, 17, True)
            bar_h = QFontMetricsF(font).height() + 12 * k
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(210, 20, 30, 225) if blink_on else QColor(120, 10, 15, 200)))
            p.drawRect(QRectF(0, 0, w, bar_h))
            p.setPen(QColor(255, 255, 255))
            p.setFont(font)
            p.drawText(QRectF(10 * k, 0, w - 20 * k, bar_h), Qt.AlignmentFlag.AlignVCenter, text)
        if res.static:
            top = (QFontMetricsF(self._font(k, 17, True)).height() + 18 * k) if res.alerts else 8 * k
            self._pill(p, 10 * k, top, [("MODO FOTO — análisis estático", self._font(k, 15, True)),
                                        ("sin tracking, sin estados de movimiento, sin historial",
                                         self._font(k, 11))],
                       QColor(20, 20, 30, 215), QColor(0, 229, 255), k, w, h)
            if not res.motos:
                font = self._font(k, 26, True)
                fm = QFontMetricsF(font)
                text = "No se detectaron motocicletas"
                self._pill(p, (w - fm.horizontalAdvance(text)) / 2 - 10 * k, h / 2 - fm.height(), [(text, font)],
                           QColor(0, 0, 0, 200), QColor(255, 214, 0), k, w, h)
        if not res.calibrated:
            lines = [("PUESTOS NO CALIBRADOS" if not res.static else "SIN PUESTOS", self._font(k, 15, True)),
                     ("Use 'Calibrar parqueadero' o 'Cargar calibración'" if not res.static
                      else "Solo detección y placa", self._font(k, 12))]
            self._pill(p, 10 * k, h - 70 * k, lines, QColor(0, 0, 0, 190), QColor(255, 214, 0), k, w, h)
        if res.static:
            stamp = f"VIMOP · {res.origin} · {res.wall_time:%d-%m-%Y %H:%M:%S}"
        else:
            stamp = f"VIMOP · {res.wall_time:%d-%m-%Y %H:%M:%S} · {fps:4.1f} FPS"
        self._pill(p, w, h - 30 * k, [(stamp, self._font(k, 11))], QColor(0, 0, 0, 150), QColor(255, 255, 255),
                   k, w, h)

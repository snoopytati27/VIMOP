"""ui/widgets.py — Widgets reutilizables: visor de video, tarjetas de resumen y ficha de moto."""
from __future__ import annotations

import html

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from utils import MOVING_BGR, NO_PLATE, bgr_to_hex, class_color_bgr
from ui.theme import DANGER, INVADE, LINE, MUTED, PANEL, PANEL_DEEP, RADIUS_PANEL, TEXT, WARN


class VideoView(QLabel):
    """Visor del video / foto con zoom y paneo. Al hacer clic (sin arrastrar) emite la coordenada en píxeles
    del frame, ya corregida por el zoom.

    Rueda = zoom sobre el cursor · botón central (o clic izquierdo sostenido con zoom) = mover ·
    doble clic o 0 = ajustar a la ventana · + / − = zoom con el teclado (con el foco en el visor).
    El zoom se conserva entre frames (video en vivo) mientras no cambie el tamaño de la imagen."""
    clicked = Signal(float, float)
    zoom_changed = Signal(float)              # 1.0 = ajustada a la ventana
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.25
    DRAG_THRESHOLD = 5                        # px: menos que esto es un clic, más es arrastrar

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(480, 270)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)   # con el foco aquí, ← → navegan entre fotos
        self._image: QImage | None = None
        self.zoom = 1.0
        self._center = QPointF()              # punto de la imagen que queda en el centro del visor
        self._press: tuple[QPointF, QPointF, Qt.MouseButton] | None = None   # (pos, centro, botón)
        self._dragging = False
        self._hint = ""
        self._hint_timer = QTimer(self, singleShot=True, interval=3500, timeout=self._clear_hint)
        self._corner = ""
        self.show_message("VIMOP\n\nPulse  ▶ INICIAR")

    # ---------------------------------------------------------------- contenido
    def show_message(self, text: str) -> None:
        self._image = None
        self.setText(text)
        self.setStyleSheet(f"background:{PANEL_DEEP}; border:1px solid {LINE}; border-radius:{RADIUS_PANEL}px; "
                           f"color:{MUTED}; font-size:16pt;")
        self.reset_zoom()

    def set_image(self, img: QImage) -> None:
        same_size = self._image is not None and self._image.size() == img.size()
        if self._image is None:
            self.setText("")
        self._image = img
        if not same_size:
            self.reset_zoom()
        self.update()

    def show_hint(self, text: str) -> None:
        """Texto de ayuda breve dibujado sobre la imagen (p. ej. al entrar en pantalla completa)."""
        self._hint = text
        self._hint_timer.start()
        self.update()

    def _clear_hint(self) -> None:
        self._hint = ""
        self.update()

    def set_corner_text(self, text: str) -> None:
        """Etiqueta discreta y persistente en la esquina (p. ej. el contador de fotos en pantalla completa)."""
        if text == self._corner:
            return
        self._corner = text
        self.update()

    # ---------------------------------------------------------------- geometría
    def _scale(self) -> float:
        iw, ih = self._image.width(), self._image.height()
        return min(self.width() / iw, self.height() / ih) * self.zoom

    def widget_to_image(self, pos: QPointF) -> QPointF:
        s = self._scale()
        return QPointF(self._center.x() + (pos.x() - self.width() / 2) / s,
                       self._center.y() + (pos.y() - self.height() / 2) / s)

    def _clamp_center(self) -> None:
        """Que no se pueda mover la imagen más allá de sus bordes (y centrada si cabe entera)."""
        s, iw, ih = self._scale(), self._image.width(), self._image.height()
        half_w, half_h = self.width() / (2 * s), self.height() / (2 * s)
        x = iw / 2 if half_w * 2 >= iw else min(max(self._center.x(), half_w), iw - half_w)
        y = ih / 2 if half_h * 2 >= ih else min(max(self._center.y(), half_h), ih - half_h)
        self._center = QPointF(x, y)

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        if self._image is not None:
            self._center = QPointF(self._image.width() / 2, self._image.height() / 2)
        self.unsetCursor()
        self.update()
        self.zoom_changed.emit(self.zoom)

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Multiplica el zoom dejando fijo el punto de la imagen bajo `anchor` (por defecto, el centro)."""
        if self._image is None:
            return
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        fixed = self.widget_to_image(anchor)
        new_zoom = min(max(self.zoom * factor, 1.0), self.MAX_ZOOM)
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        s = self._scale()
        self._center = QPointF(fixed.x() - (anchor.x() - self.width() / 2) / s,
                               fixed.y() - (anchor.y() - self.height() / 2) / s)
        self._clamp_center()
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.0 else Qt.CursorShape.ArrowCursor)
        self.update()
        self.zoom_changed.emit(self.zoom)

    # ---------------------------------------------------------------- dibujo
    def paintEvent(self, ev):
        super().paintEvent(ev)                # fondo (hoja de estilos) y, sin imagen, el mensaje
        if self._image is None:
            return
        self._clamp_center()
        s = self._scale()
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        rounded = QPainterPath()                      # la imagen sigue las esquinas redondeadas del visor
        rounded.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), RADIUS_PANEL - 1, RADIUS_PANEL - 1)
        p.setClipPath(rounded)
        p.translate(self.width() / 2, self.height() / 2)
        p.scale(s, s)
        p.translate(-self._center.x(), -self._center.y())
        p.drawImage(0, 0, self._image)
        p.resetTransform()
        if self._hint:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            box = QRectF(0, 0, min(self.width() - 20, 760), 34)
            box.moveCenter(QPointF(self.width() / 2, 30))
            p.drawRoundedRect(box, 8, 8)
            p.setPen(QColor(TEXT))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, self._hint)
        if self._corner:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            metrics = p.fontMetrics()
            box = QRectF(0, 0, metrics.horizontalAdvance(self._corner) + 20, 26)
            box.moveTopRight(QPointF(self.width() - 12, 12))
            p.drawRoundedRect(box, 8, 8)
            p.setPen(QColor(TEXT))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, self._corner)
        p.end()

    # ---------------------------------------------------------------- ratón y teclado
    def wheelEvent(self, ev):
        if self._image is None or ev.angleDelta().y() == 0:
            return super().wheelEvent(ev)
        self.zoom_by(self.ZOOM_STEP if ev.angleDelta().y() > 0 else 1 / self.ZOOM_STEP, ev.position())
        ev.accept()

    def mousePressEvent(self, ev):
        self.setFocus()
        if self._image is not None and ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._press = (ev.position(), QPointF(self._center), ev.button())
            self._dragging = False

    def mouseMoveEvent(self, ev):
        if self._press is None:
            return
        start, center, button = self._press
        delta = ev.position() - start
        if not self._dragging:
            can_pan = self.zoom > 1.0 or button == Qt.MouseButton.MiddleButton
            if not can_pan or abs(delta.x()) + abs(delta.y()) < self.DRAG_THRESHOLD:
                return
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        s = self._scale()
        self._center = QPointF(center.x() - delta.x() / s, center.y() - delta.y() / s)
        self._clamp_center()
        self.update()

    def mouseReleaseEvent(self, ev):
        press, dragging = self._press, self._dragging
        self._press, self._dragging = None, False
        if dragging:
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.0 else Qt.CursorShape.ArrowCursor)
            return
        if press is None or press[2] != Qt.MouseButton.LeftButton or self._image is None:
            return
        pt = self.widget_to_image(ev.position())
        if 0 <= pt.x() < self._image.width() and 0 <= pt.y() < self._image.height():
            self.clicked.emit(pt.x(), pt.y())

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(self.ZOOM_STEP)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / self.ZOOM_STEP)
        elif key == Qt.Key.Key_0:
            self.reset_zoom()
        else:
            return super().keyPressEvent(ev)          # ← → siguen hacia la ventana (modo foto)
        ev.accept()


class GlowButton(QPushButton):
    """Botón con un brillo de color alrededor, encendido solo mientras el botón está habilitado
    (p. ej. DETENER: brilla en rojo mientras hay algo que detener)."""

    def __init__(self, text: str, color: str = DANGER, blur: int = 24):
        super().__init__(text)
        self.glow = QGraphicsDropShadowEffect(self, blurRadius=blur, offset=QPointF(0, 0))
        c = QColor(color)
        c.setAlpha(200)
        self.glow.setColor(c)
        self.setGraphicsEffect(self.glow)
        self.glow.setEnabled(self.isEnabled())

    def changeEvent(self, ev):
        if ev.type() == QEvent.Type.EnabledChange:
            self.glow.setEnabled(self.isEnabled())
        super().changeEvent(ev)


class Tile(QFrame):
    def __init__(self, title: str, color: str = TEXT):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self.value = QLabel("0")
        self.value.setObjectName("tileValue")
        self.value.setStyleSheet(f"color:{color};")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setMinimumWidth(self.value.fontMetrics().horizontalAdvance("0000"))   # hasta 3 dígitos sin cortes
        t = QLabel(title)
        t.setObjectName("tileTitle")
        lay.addWidget(self.value, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)

    def set(self, v) -> None:
        self.value.setText(str(v))


WHEEL_TEXT = {"pose": "modelo entrenado", "mask": "estimadas por máscara (aproximado)",
              "bbox": "sin ruedas (solo bbox)"}


def fmt_pct(v) -> str:
    return "—" if v is None else f"{v:.0f}%"


def moto_card_html(m: dict, big: bool) -> str:
    """Ficha HTML de una moto (panel lateral)."""
    esc = html.escape
    cls = m.get("clasificacion")
    color = bgr_to_hex(class_color_bgr(cls)) if cls else bgr_to_hex(MOVING_BGR)
    score = f"{m['puntaje']:.1f}" if m.get("puntaje") is not None else "—"
    estado = cls or m["estado_movimiento"]
    plate = esc(m["placa"])
    if m["placa"] != NO_PLATE:
        plate += f" <span style='color:{MUTED}'>({m['placa_lecturas']} lecturas)</span>"
    rows = [("Placa", plate), ("Puesto", esc(m["puesto"] or "—")), ("Movimiento", esc(m["estado_movimiento"]))]
    if not big:
        rows = rows[:2]
    body = "".join(f"<tr><td style='color:{MUTED};padding-right:10px'>{k}:</td><td><b>{v}</b></td></tr>"
                   for k, v in rows)
    if big:
        if m.get("orientacion") is None and m.get("orientacion_confiable") is False:
            ang = f" <span style='color:{WARN}'>no confiable</span>"
        else:
            ang = ""
        err_txt = "—" if m.get("error_angular") is None else f"{m['error_angular']:.1f}°"
        wheels = WHEEL_TEXT.get(m.get("ruedas_origen") or "", "")
        avisos = "".join(f"<tr><td colspan=2 style='color:{WARN}'>⚠ {esc(v)}</td></tr>" for v in (m.get("avisos") or []))
        inv = ""
        if m.get("puestos_invadidos"):
            inv = f" <span style='color:{INVADE}'>({esc(', '.join(m['puestos_invadidos']))})</span>"
        body += ("<tr><td colspan=2><hr></td></tr>"
                 f"<tr><td style='color:{MUTED}'>Orientación (35%):</td><td><b>{fmt_pct(m['orientacion'])}</b>{ang}</td></tr>"
                 f"<tr><td style='color:{MUTED}'>Error angular:</td><td><b>{err_txt}</b></td></tr>"
                 f"<tr><td style='color:{MUTED}'>Ubicación (35%):</td><td><b>{fmt_pct(m['ubicacion'])}</b></td></tr>"
                 f"<tr><td style='color:{MUTED}'>Centrado (20%):</td><td><b>{fmt_pct(m['centrado'])}</b></td></tr>"
                 f"<tr><td style='color:{MUTED}'>Invasión (10%):</td><td><b>{fmt_pct(m['invasion'])}</b> "
                 f"<span style='color:{MUTED}'>{esc(m.get('nivel_invasion') or '')}</span>{inv}</td></tr>"
                 + (f"<tr><td style='color:{MUTED}'>Ruedas:</td><td><b>{wheels}</b></td></tr>" if wheels else "")
                 + avisos)
    hidden = "" if m["visible"] else f" <span style='color:{MUTED};font-size:9pt'>(oculta)</span>"
    return (f"<div style='border-left:5px solid {color};padding:6px 10px;margin:6px 0;background:{PANEL}'>"
            f"<div style='font-size:{'16pt' if big else '12pt'};font-weight:800'>MOTO #{m['id']:02d}{hidden}</div>"
            f"<table cellspacing=2>{body}</table>"
            f"<div style='margin-top:6px;font-size:{'15pt' if big else '11pt'};font-weight:800'>"
            f"SCORE: <span style='color:{color}'>{score}</span> &nbsp; "
            f"<span style='background:{color};color:#111'>&nbsp;{esc(estado)}&nbsp;</span></div></div>")

"""
tools/label_wheels.py — Etiquetado semiautomático de las ruedas de las motos (dataset de puntos clave).

    python tools/label_wheels.py [carpeta_con_fotos_y_videos]

Flujo:
    1. Abrir carpeta: se leen las fotos y, de cada video, un fotograma cada N segundos (sin casi-duplicados).
    2. Preetiquetar: el detector de motos + el backend "mask" de WheelLocator colocan los 2 puntos de cada moto.
    3. Revisar: arrastre los puntos (trasera azul, delantera roja) y pulse Enter para aceptar y pasar a la siguiente.
    4. Guardar progreso (Ctrl+S) — se puede cerrar y reanudar después.
    5. Exportar: dataset YOLO-pose (solo lo revisado, partición 80/20 por fuente) + dataset.yaml + .zip para Drive.

Atajos:
    Enter aceptar y siguiente · ← → navegar · S intercambiar delantera/trasera · Supr descartar moto
    N añadir moto (arrastre un recuadro) · V visibilidad del punto (visible / oculto / fuera) · Tab moto siguiente
    U marcar como sin revisar · rueda del ratón = zoom · arrastrar fondo = desplazar · Ctrl+S guardar · Esc cancelar
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal  # noqa: E402
from PySide6.QtGui import (QAction, QBrush, QColor, QCursor, QFont, QImage, QKeySequence, QPainter, QPen,  # noqa: E402
                           QShortcut)
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,  # noqa: E402
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog, QToolBar, QVBoxLayout,
                               QWidget)

from config import load_config  # noqa: E402
from core.image_io import ImageLoadError  # noqa: E402
from tools.wheel_dataset import (NEXT_VIS, VIS_LABELS, VIS_OUTSIDE, VIS_VISIBLE, ImageEntry, MotoLabel,  # noqa: E402
                                 WheelProject, export_dataset, find_parent_project, load_entry_image, preannotate,
                                 scan_folder)

log = logging.getLogger("vimop.label_wheels")

REAR_COLOR, FRONT_COLOR = QColor("#2979ff"), QColor("#ff1744")
HIT_RADIUS = 12.0                       # píxeles de pantalla para "tocar" un punto
MIN_NEW_BBOX = 8.0                      # píxeles de imagen


def bgr_to_qimage(frame) -> QImage:
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# ==========================================================================
# Lienzo: foto grande, bbox, dos puntos arrastrables por moto, zoom y desplazamiento
# ==========================================================================
class Canvas(QWidget):
    edited = Signal()                   # cambió alguna etiqueta
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.image: QImage | None = None
        self.entry: ImageEntry | None = None
        self.message = "Abra una carpeta con fotos y videos."
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.sel_moto: int | None = None
        self.sel_point: str | None = None            # "rear" | "front"
        self.add_mode = False
        self._drag: tuple | None = None              # ("point", moto, key) | ("pan", start, offset0) | ("bbox", p0, p1)

    # ---- datos ---------------------------------------------------------------
    def set_entry(self, entry: ImageEntry | None, image: QImage | None, message: str = "") -> None:
        self.entry, self.image = entry, image
        self.message = message or ("Sin imagen." if entry is None else "")
        self.sel_moto = 0 if entry is not None and entry.motos else None
        self.sel_point = None
        self.add_mode = False
        self._drag = None
        self.fit()
        self.selection_changed.emit()

    def fit(self) -> None:
        if self.image is None:
            return
        s = min(self.width() / self.image.width(), self.height() / self.image.height())
        self.scale = max(s, 0.01)
        self.offset = QPointF((self.width() - self.image.width() * self.scale) / 2,
                              (self.height() - self.image.height() * self.scale) / 2)
        self.update()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.fit()

    def to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def to_image(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        if self.image is None:
            return x, y
        return min(max(x, 0.0), float(self.image.width())), min(max(y, 0.0), float(self.image.height()))

    @property
    def moto(self) -> MotoLabel | None:
        if self.entry is None or self.sel_moto is None or not (0 <= self.sel_moto < len(self.entry.motos)):
            return None
        return self.entry.motos[self.sel_moto]

    # ---- acciones sobre la moto seleccionada -------------------------------------
    def _target(self) -> MotoLabel | None:
        if self.moto is not None:
            return self.moto
        if self.entry is not None and len(self.entry.motos) == 1:
            self.sel_moto = 0
            return self.entry.motos[0]
        return None

    def swap_wheels(self) -> bool:
        m = self._target()
        if m is None:
            return False
        m.swap()
        self.sel_point = {"rear": "front", "front": "rear"}.get(self.sel_point or "", None)
        self._changed()
        return True

    def delete_moto(self) -> bool:
        m = self._target()
        if m is None or self.entry is None:
            return False
        self.entry.motos.remove(m)
        self.sel_moto = min(self.sel_moto or 0, len(self.entry.motos) - 1) if self.entry.motos else None
        self.sel_point = None
        self._changed()
        return True

    def cycle_visibility(self) -> bool:
        """visible -> oculto por oclusión -> fuera de imagen -> visible, del punto seleccionado (o el trasero)."""
        m = self._target()
        if m is None:
            return False
        point = getattr(m, self.sel_point or "rear")
        point[2] = NEXT_VIS[int(point[2])]
        self._changed()
        return True

    def next_moto(self) -> None:
        if self.entry and self.entry.motos:
            self.sel_moto = 0 if self.sel_moto is None else (self.sel_moto + 1) % len(self.entry.motos)
            self.sel_point = None
            self.selection_changed.emit()
            self.update()

    def start_add(self) -> None:
        self.add_mode = self.image is not None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if self.add_mode else Qt.CursorShape.ArrowCursor))
        self.update()

    def cancel_add(self) -> None:
        self.add_mode = False
        self._drag = None
        self.unsetCursor()
        self.update()

    def add_moto(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Moto nueva con los puntos en posiciones razonables (abajo, a 15 % y 85 % del ancho) para arrastrarlos."""
        if self.entry is None:
            return
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        w, h = x2 - x1, y2 - y1
        self.entry.motos.append(MotoLabel([x1, y1, x2, y2], [x1 + 0.15 * w, y2 - 0.08 * h, VIS_VISIBLE],
                                          [x2 - 0.15 * w, y2 - 0.08 * h, VIS_VISIBLE], manual=True))
        self.sel_moto = len(self.entry.motos) - 1
        self.sel_point = None
        self.cancel_add()
        self._changed()

    def _changed(self) -> None:
        self.edited.emit()
        self.selection_changed.emit()
        self.update()

    # ---- ratón ------------------------------------------------------------------
    def _hit_point(self, pos: QPointF) -> tuple[int, str] | None:
        if self.entry is None:
            return None
        best, best_d = None, HIT_RADIUS
        order = list(range(len(self.entry.motos)))
        if self.sel_moto is not None:                                   # la seleccionada tiene prioridad
            order.sort(key=lambda i: i != self.sel_moto)
        for i in order:
            for key in ("rear", "front"):
                p = getattr(self.entry.motos[i], key)
                sp = self.to_screen(p[0], p[1])
                d = ((sp.x() - pos.x()) ** 2 + (sp.y() - pos.y()) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = (i, key), d
        return best

    def _hit_bbox(self, pos: QPointF) -> int | None:
        if self.entry is None:
            return None
        ip = self.to_image(pos)
        hits = [i for i, m in enumerate(self.entry.motos) if m.bbox[0] <= ip.x() <= m.bbox[2] and m.bbox[1] <= ip.y() <= m.bbox[3]]
        if not hits:
            return None
        return min(hits, key=lambda i: (self.entry.motos[i].bbox[2] - self.entry.motos[i].bbox[0]) *
                                       (self.entry.motos[i].bbox[3] - self.entry.motos[i].bbox[1]))

    def mousePressEvent(self, ev) -> None:
        self.setFocus()
        if self.image is None:
            return
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._drag = ("pan", pos, QPointF(self.offset))
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self.add_mode:
            ip = self.to_image(pos)
            self._drag = ("bbox", ip, ip)
            return
        hit = self._hit_point(pos)
        if hit is not None:
            self.sel_moto, self.sel_point = hit
            self._drag = ("point", *hit)
            self.selection_changed.emit()
            self.update()
            return
        box = self._hit_bbox(pos)
        if box is not None:
            self.sel_moto, self.sel_point = box, None
            self.selection_changed.emit()
            self.update()
            return
        self._drag = ("pan", pos, QPointF(self.offset))                    # clic sostenido en el fondo: desplazar

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is None:
            return
        pos = ev.position()
        kind = self._drag[0]
        if kind == "pan":
            _, start, off0 = self._drag
            self.offset = QPointF(off0.x() + pos.x() - start.x(), off0.y() + pos.y() - start.y())
        elif kind == "point" and self.entry is not None:
            _, i, key = self._drag
            ip = self.to_image(pos)
            x, y = self.clamp(ip.x(), ip.y())
            p = getattr(self.entry.motos[i], key)
            p[0], p[1] = x, y
            self.edited.emit()
        elif kind == "bbox":
            ip = self.to_image(pos)
            self._drag = ("bbox", self._drag[1], QPointF(*self.clamp(ip.x(), ip.y())))
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag is not None and self._drag[0] == "bbox":
            _, p0, p1 = self._drag
            self._drag = None
            if abs(p1.x() - p0.x()) >= MIN_NEW_BBOX and abs(p1.y() - p0.y()) >= MIN_NEW_BBOX:
                self.add_moto(p0.x(), p0.y(), p1.x(), p1.y())
            return
        self._drag = None

    def wheelEvent(self, ev) -> None:
        if self.image is None:
            return
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        new_scale = min(max(self.scale * factor, 0.05), 40.0)
        pos = ev.position()
        ip = self.to_image(pos)                                            # el punto bajo el cursor no se mueve
        self.scale = new_scale
        self.offset = QPointF(pos.x() - ip.x() * self.scale, pos.y() - ip.y() * self.scale)
        self.update()

    def mouseDoubleClickEvent(self, ev) -> None:
        self.fit()

    # ---- dibujo -------------------------------------------------------------------
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 26, 34))
        if self.image is None:
            p.setPen(QColor(190, 195, 210))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        target = QRectF(self.offset.x(), self.offset.y(), self.image.width() * self.scale,
                        self.image.height() * self.scale)
        p.drawImage(target, self.image)
        if self.entry is not None:
            reviewed = self.entry.reviewed
            p.setPen(QPen(QColor("#00c853" if reviewed else "#ff9100"), 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(target)                                           # verde = revisada, naranja = solo preetiquetada
            for i, m in enumerate(self.entry.motos):
                self._paint_moto(p, i, m, i == self.sel_moto)
        if self._drag is not None and self._drag[0] == "bbox":
            a, b = self.to_screen(self._drag[1].x(), self._drag[1].y()), self.to_screen(self._drag[2].x(), self._drag[2].y())
            p.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(a, b).normalized())
        if self.add_mode:
            p.setPen(QColor(255, 235, 59))
            p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            p.drawText(12, 24, "AÑADIR MOTO: arrastre un recuadro sobre ella (Esc cancela)")

    def _paint_moto(self, p: QPainter, index: int, m: MotoLabel, selected: bool) -> None:
        x1, y1, x2, y2 = m.bbox
        a, b = self.to_screen(x1, y1), self.to_screen(x2, y2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 230 if selected else 110), 2 if selected else 1,
                      Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine))
        p.drawRect(QRectF(a, b))
        r, f = self.to_screen(m.rear[0], m.rear[1]), self.to_screen(m.front[0], m.front[1])
        p.setPen(QPen(QColor(0, 0, 0, 160), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(r, f)
        p.setPen(QPen(QColor(255, 235, 59), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(r, f)
        for key, pt, color in (("rear", r, REAR_COLOR), ("front", f, FRONT_COLOR)):
            v = int(getattr(m, key)[2])
            active = selected and self.sel_point == key
            radius = 8.0 if active else 6.5
            if v == VIS_VISIBLE:
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.setBrush(QBrush(color))
            elif v == VIS_OUTSIDE:
                p.setPen(QPen(QColor(150, 150, 150), 2, Qt.PenStyle.DashLine))
                p.setBrush(QBrush(QColor(90, 90, 90, 120)))
            else:                                                        # oculto por oclusión: aro hueco
                p.setPen(QPen(color, 3))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pt, radius, radius)
            if v != VIS_VISIBLE:
                p.setPen(QColor(255, 255, 255))
                p.setFont(QFont("Segoe UI", 8))
                p.drawText(pt + QPointF(radius + 3, 4), "oculto" if v == 1 else "fuera")
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.drawText(a + QPointF(4, 14), f"#{index + 1}" + (" ✎" if m.manual else ""))


# ==========================================================================
# Trabajos en segundo plano (la ventana no se congela)
# ==========================================================================
class ScanWorker(QThread):
    progress = Signal(str)
    done = Signal(int, int)
    failed = Signal(str)

    def __init__(self, project: WheelProject, every_seconds: float):
        super().__init__()
        self.project, self.every = project, every_seconds

    def run(self) -> None:
        try:
            photos, frames = scan_folder(self.project, self.every, self.progress.emit)
            self.done.emit(photos, frames)
        except Exception as exc:                                         # noqa: BLE001 - se informa a la persona
            log.exception("Error al leer la carpeta")
            self.failed.emit(str(exc))


class PrelabelWorker(QThread):
    progress = Signal(int, int)
    done = Signal(int, str)
    failed = Signal(str)

    def __init__(self, project: WheelProject, cfg):
        super().__init__()
        self.project, self.cfg = project, cfg

    def run(self) -> None:
        try:
            from analysis.wheels import WheelLocator
            from core.detector import ModelNotFoundError, YoloDetector, select_device
            device, is_gpu, desc = select_device(self.cfg.DEVICE)
            try:
                detector = YoloDetector(self.cfg, device, is_gpu)
            except ModelNotFoundError as exc:
                self.failed.emit(str(exc))
                return
            cfg = self.cfg.copy()
            cfg.WHEEL_BACKEND = "mask"                                   # el preetiquetado usa siempre el respaldo
            n = preannotate(self.project, detector, WheelLocator(cfg, device, False), cfg,
                            progress=lambda i, t: self.progress.emit(i, t))
            note = "" if detector.is_segmentation else (
                "\n\nAVISO: el modelo de detección no trae máscaras, así que los puntos son solo aproximados "
                "(segmento a lo largo del recuadro). Use un modelo '-seg' (MODEL_PATH) para un preetiquetado mejor.")
            self.done.emit(n, f"Dispositivo: {desc}{note}")
        except Exception as exc:                                         # noqa: BLE001
            log.exception("Error al preetiquetar")
            self.failed.emit(str(exc))


# ==========================================================================
# Ventana principal
# ==========================================================================
class LabelWindow(QMainWindow):
    SHORTCUTS = ("Enter aceptar · ←/→ navegar · S intercambiar · Supr descartar · N añadir moto · V visibilidad · "
                 "Tab siguiente moto · U sin revisar · rueda = zoom · Ctrl+S guardar")

    def __init__(self, folder: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("VIMOP — Etiquetado de ruedas")
        self.resize(1360, 860)
        self.cfg = load_config()
        self.project: WheelProject | None = None
        self.index = 0
        self._worker: QThread | None = None
        self._progress: QProgressDialog | None = None
        self._cache: dict[str, QImage] = {}

        self.canvas = Canvas()
        self.canvas.edited.connect(self._on_edited)
        self.canvas.selection_changed.connect(self._refresh_status)
        self.listing = QListWidget()
        self.listing.setMaximumWidth(280)
        self.listing.setFocusPolicy(Qt.FocusPolicy.NoFocus)      # las flechas y Tab son de la ventana
        self.listing.currentRowChanged.connect(self._on_row)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight:700;padding:2px 8px")
        self.detail = QLabel("")
        self.detail.setStyleSheet("color:#9aa0b4;padding:2px 8px")

        central = QWidget()
        lay = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(self.listing)
        row.addWidget(self.canvas, 1)
        lay.addLayout(row, 1)
        lay.addWidget(self.detail)
        hint = QLabel(self.SHORTCUTS)
        hint.setStyleSheet("color:#8a90a5;padding:2px 8px")
        lay.addWidget(hint)
        self.setCentralWidget(central)
        self._build_toolbar()
        self._build_shortcuts()
        if folder:
            self.open_folder(folder)

    # ---- construcción ---------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Principal")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.actions_by_name: dict[str, QAction] = {}

        def add(name: str, text: str, slot, tip: str = "") -> QAction:
            act = QAction(text, self)
            act.setToolTip(tip or text)
            act.triggered.connect(slot)
            tb.addAction(act)
            self.actions_by_name[name] = act
            return act

        add("open", "Abrir carpeta…", lambda: self.choose_folder())
        add("prelabel", "Preetiquetar", lambda: self.run_prelabel(), "Detectar motos y colocar los 2 puntos")
        tb.addSeparator()
        add("prev", "◀", lambda: self.go(self.index - 1))
        add("next", "▶", lambda: self.go(self.index + 1))
        add("accept", "✔ Aceptar (Enter)", lambda: self.accept_current())
        tb.addSeparator()
        add("save", "GUARDAR PROGRESO", lambda: self.save(), "Ctrl+S")
        add("export", "Exportar dataset…", lambda: self.export(), "YOLO-pose + dataset.yaml + .zip")
        tb.addSeparator()
        tb.addWidget(self.status_label)

    def _build_shortcuts(self) -> None:
        def sc(keys, slot):
            for k in (keys if isinstance(keys, (list, tuple)) else [keys]):
                s = QShortcut(QKeySequence(k), self)
                s.setContext(Qt.ShortcutContext.WindowShortcut)
                s.activated.connect(slot)

        sc(["Return", "Enter"], self.accept_current)
        sc("Right", lambda: self.go(self.index + 1))
        sc("Left", lambda: self.go(self.index - 1))
        sc("S", lambda: self.canvas.swap_wheels())
        sc("Delete", lambda: self.canvas.delete_moto())
        sc("N", lambda: self.canvas.start_add())
        sc("V", lambda: self.canvas.cycle_visibility())
        sc("Tab", lambda: self.canvas.next_moto())
        sc("U", self.unreview_current)
        sc("Ctrl+S", self.save)
        sc("Escape", lambda: self.canvas.cancel_add())

    # ---- carpeta / proyecto -------------------------------------------------------
    def choose_folder(self) -> None:
        if not self._confirm_leave():
            return
        folder = QFileDialog.getExistingDirectory(self, "Carpeta con fotos y videos")
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder: str | Path, every_seconds: float | None = None, blocking: bool = False) -> None:
        """Abre (o reanuda) una carpeta. `blocking=True` escanea en este hilo (pruebas y línea de comandos)."""
        parent = None if blocking else find_parent_project(folder)
        if parent is not None and Path(folder).resolve() != parent:
            box = QMessageBox(QMessageBox.Icon.Question, "Esta carpeta ya es parte de un proyecto",
                              f"'{Path(folder).name}' está dentro del proyecto:\n{parent}\n\nSi la abre por separado, sus "
                              "etiquetas quedarían en un proyecto aparte y NO entrarían al dataset principal.\n\n"
                              "¿Abrir la carpeta principal?", parent=self)
            main_btn = box.addButton("Abrir la principal (recomendado)", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Abrir esta de todos modos", QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            if box.clickedButton() is main_btn:
                folder = parent
        try:
            self.project = WheelProject.open(folder)
        except (FileNotFoundError, ValueError) as exc:
            self._error(str(exc))
            return
        self.cfg = load_config()
        if every_seconds is None:
            every_seconds = self.project.every_seconds
            if not blocking:
                value, ok = QInputDialog.getDouble(self, "Fotogramas de video", "Tomar un fotograma cada (segundos):",
                                                   every_seconds, 0.1, 600.0, 1)
                if not ok:
                    return
                every_seconds = value
        if blocking:
            scan_folder(self.project, every_seconds)
            self._after_scan(0, 0, quiet=True)
            return
        self._start_worker(ScanWorker(self.project, every_seconds), "Leyendo la carpeta…")
        self._worker.progress.connect(lambda t: self._progress and self._progress.setLabelText(t))
        self._worker.done.connect(self._after_scan)
        self._worker.failed.connect(self._worker_failed)
        self._worker.start()

    def _after_scan(self, photos: int, frames: int, quiet: bool = False) -> None:
        self._end_worker()
        if self.project is None:
            return
        self._cache.clear()
        self._fill_list()
        if not self.project.entries:
            self.canvas.set_entry(None, None, "La carpeta no tiene fotos ni videos.")
            self._refresh_status()
            return
        first = next((i for i, e in enumerate(self.project.entries) if not e.reviewed), 0)   # reanudar donde se quedó
        self.go(first)
        if not quiet:
            self.statusBar().showMessage(f"{photos} fotos y {frames} fotogramas nuevos. "
                                         f"Total: {len(self.project.entries)}.", 8000)

    def _fill_list(self) -> None:
        self.listing.blockSignals(True)
        self.listing.clear()
        assert self.project is not None
        for e in self.project.entries:
            self.listing.addItem(QListWidgetItem(self._item_text(e)))
        for i, e in enumerate(self.project.entries):
            self._style_item(i, e)
        self.listing.blockSignals(False)

    @staticmethod
    def _item_text(e: ImageEntry) -> str:
        mark = "✔" if e.reviewed else ("•" if e.prelabeled else "○")
        return f"{mark} {Path(e.path).name}  ({len(e.motos)})"

    def _style_item(self, i: int, e: ImageEntry) -> None:
        it = self.listing.item(i)
        if it is None:
            return
        it.setText(self._item_text(e))
        it.setForeground(QColor("#00c853") if e.reviewed else QColor("#ff9100") if e.prelabeled else QColor("#9aa0b4"))

    # ---- navegación -----------------------------------------------------------------
    def go(self, index: int) -> None:
        if self.project is None or not self.project.entries:
            return
        self.index = min(max(index, 0), len(self.project.entries) - 1)
        entry = self.project.entries[self.index]
        image, msg = self._load_image(entry)
        self.canvas.set_entry(entry, image, msg)
        self.listing.blockSignals(True)
        self.listing.setCurrentRow(self.index)
        self.listing.blockSignals(False)
        self._refresh_status()

    def _on_row(self, row: int) -> None:
        if row >= 0:
            self.go(row)

    def _load_image(self, entry: ImageEntry) -> tuple[QImage | None, str]:
        assert self.project is not None
        if entry.path in self._cache:
            return self._cache[entry.path], ""
        try:
            bgr = load_entry_image(self.project, entry, self.cfg)
        except (ImageLoadError, OSError) as exc:
            return None, f"No se pudo abrir la imagen:\n{exc}"
        entry.height, entry.width = bgr.shape[:2]
        img = bgr_to_qimage(bgr)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        self._cache[entry.path] = img
        return img, ""

    # ---- edición ----------------------------------------------------------------------
    def accept_current(self) -> None:
        if self.project is None or not self.project.entries:
            return
        entry = self.project.entries[self.index]
        entry.reviewed = True
        self.project.dirty = True
        self._style_item(self.index, entry)
        if self.index < len(self.project.entries) - 1:
            self.go(self.index + 1)
        else:
            self.canvas.update()
            self._refresh_status()

    def unreview_current(self) -> None:
        if self.project is None or not self.project.entries:
            return
        entry = self.project.entries[self.index]
        entry.reviewed = False
        self.project.dirty = True
        self._style_item(self.index, entry)
        self.canvas.update()
        self._refresh_status()

    def _on_edited(self) -> None:
        if self.project is not None:
            self.project.dirty = True
            if self.project.entries:
                self._style_item(self.index, self.project.entries[self.index])
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self.project is None or not self.project.entries:
            self.status_label.setText("")
            self.detail.setText("")
            return
        e = self.project.entries[self.index]
        state = "REVISADA" if e.reviewed else ("solo preetiquetada" if e.prelabeled else "sin etiquetar")
        self.status_label.setText(self.project.progress_text(self.index) + ("  ●" if self.project.dirty else ""))
        m = self.canvas.moto
        sel = ""
        if m is not None:
            sel = (f" · moto #{(self.canvas.sel_moto or 0) + 1}: trasera {VIS_LABELS[int(m.rear[2])]}, "
                   f"delantera {VIS_LABELS[int(m.front[2])]}")
        self.detail.setText(f"{Path(e.path).name} · fuente {e.source} · {state} · {len(e.motos)} motos{sel}")
        self.setWindowTitle(f"VIMOP — Etiquetado de ruedas — {self.project.folder.name}"
                            + (" *" if self.project.dirty else ""))

    # ---- guardar / exportar / cerrar ------------------------------------------------------
    def save(self) -> bool:
        if self.project is None:
            return False
        try:
            path = self.project.save()
        except OSError as exc:
            self._error(f"No se pudo guardar el progreso: {exc}")
            return False
        self.statusBar().showMessage(f"Progreso guardado en {path.name}", 5000)
        self._refresh_status()
        return True

    def export(self, out_dir: str | Path | None = None, quiet: bool = False):
        if self.project is None:
            return None
        if not self.save():
            return None
        try:
            res = export_dataset(self.project, self.cfg, out_dir)
        except (ValueError, OSError, ImageLoadError) as exc:
            self._error(str(exc))
            return None
        if not quiet:
            text = (f"Dataset exportado en:\n{res.out_dir}\n\n"
                    f"train: {res.train_images} · val: {res.val_images} · test: {res.test_images} imágenes · "
                    f"{res.motos} motos\n"
                    f"Fuentes train: {', '.join(res.train_sources)}\nFuentes val: {', '.join(res.val_sources)}\n"
                    f"Fuentes test: {', '.join(res.test_sources) or '—'}\n"
                    f"\nZip para Drive:\n{res.zip_path}")
            if res.warnings:
                text += "\n\nAVISOS:\n- " + "\n- ".join(res.warnings)
            QMessageBox.information(self, "Exportación lista", text)
        return res

    def _confirm_leave(self) -> bool:
        """Con cambios sin guardar: guardar, descartar o cancelar. True = se puede continuar."""
        if self.project is None or not self.project.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Cambios sin guardar")
        box.setText("Hay cambios sin guardar. ¿Qué desea hacer?")
        save = box.addButton("Guardar", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Descartar", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel:
            return False
        if clicked is save:
            return self.save()
        return True

    def closeEvent(self, ev) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Trabajo en curso", "Espere a que termine el preetiquetado o la lectura.")
            ev.ignore()
            return
        if self._confirm_leave():
            ev.accept()
        else:
            ev.ignore()

    # ---- preetiquetado ---------------------------------------------------------------------
    def run_prelabel(self) -> None:
        if self.project is None or not self.project.entries:
            self._error("Primero abra una carpeta con fotos o videos.")
            return
        pending = sum(not e.reviewed and not e.prelabeled for e in self.project.entries)
        if pending == 0:
            self.statusBar().showMessage("Todo lo pendiente ya está preetiquetado o revisado.", 6000)
            return
        self._start_worker(PrelabelWorker(self.project, self.cfg), f"Preetiquetando {pending} imágenes…\n"
                                                                  "(la primera vez carga el modelo)", maximum=pending)
        self._worker.progress.connect(lambda i, t: self._progress and self._progress.setValue(i))
        self._worker.done.connect(self._after_prelabel)
        self._worker.failed.connect(self._worker_failed)
        self._worker.start()

    def _after_prelabel(self, n: int, note: str) -> None:
        self._end_worker()
        self._fill_list()
        self.go(self.index)
        self._on_edited()
        QMessageBox.information(self, "Preetiquetado", f"Se preetiquetaron {n} imágenes. Revíselas: los puntos son "
                                                      f"una propuesta, no la verdad.\n{note}")

    # ---- utilidades de trabajos ----------------------------------------------------------------
    def _start_worker(self, worker: QThread, text: str, maximum: int = 0) -> None:
        self._worker = worker
        self._progress = QProgressDialog(text, None, 0, maximum, self)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.show()

    def _end_worker(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        if self._worker is not None:
            self._worker.wait(5000)
            self._worker = None

    def _worker_failed(self, message: str) -> None:
        self._end_worker()
        self._error(message)

    def _error(self, message: str) -> None:
        log.error(message)
        QMessageBox.warning(self, "VIMOP — Etiquetado de ruedas", message)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = QApplication.instance() or QApplication(argv)
    win = LabelWindow(argv[1] if len(argv) > 1 else None)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

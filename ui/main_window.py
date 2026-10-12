"""
ui/main_window.py — Ventana principal de VIMOP.

┌ cabecera: VIMOP · puestos · dispositivo (GPU/CPU) · cámara · FPS ─────────────────────┐
│ resumen: motos · estacionadas · moviéndose · excelentes · buenas · regulares · ...     │
│ ┌controles┐ ┌ video en tiempo real ──────────────────┐ ┌ moto seleccionada ─┐          │
│ │ fuente  │ │ (banner de alertas que parpadea)       │ │ placa, puesto, ... │          │
│ │ botones │ │                                        │ │ O / U / C / I      │          │
│ └─────────┘ └────────────────────────────────────────┘ │ alertas            │          │
│ [Motos en vivo] [Eventos] [Historial / consulta]                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import html
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QScroller, QSplitter, QStatusBar, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from analysis.parking_spaces import ParkingLayout, load_layout
from calibration import CalibrationDialog
from config import RESULTS_DIR, Config
from core.camera import SOURCE_TYPES, CameraError, describe_source, grab_single_frame
from core.image_io import FILE_FILTERS, _IMAGE_GLOB, _VIDEO_GLOB, detect_file_kind, list_images_in_folder
from database import ResultsStore, query_events, query_motos
from ui.dialogs import SettingsDialog
from ui.assets import WORDMARK, logo_label
from ui.help import show_manual
from ui.photo_worker import PhotoWorker
from ui.theme import ALERT, DIM, MUTED, OK, WARN_STRONG
from ui.widgets import GlowButton, Tile, VideoView, moto_card_html
from ui.worker import ProcessingThread
from utils import CLASS_COLORS_BGR, CLASS_NAMES, bgr_to_hex

log = logging.getLogger("vimop.ui")
SOURCE_LABELS = {"webcam": "Webcam USB", "ip": "Cámara IP (HTTP)", "rtsp": "Cámara RTSP", "video": "Video MP4",
                 "image": "Foto / imagen"}


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("VIMOP — Visión Inteligente para el Monitoreo y Ordenamiento del Parqueo")
        self.resize(1600, 980)
        self.thread: ProcessingThread | None = None
        self.store: ResultsStore | None = None
        self.selected_id: int | None = None
        self.last_payload: dict | None = None
        self.last_frame: np.ndarray | None = None
        self._last_table = 0.0
        self.layout_path = cfg.resolve_path(cfg.PARKING_SPACES_FILE)
        self.parking_layout, msg = load_layout(self.layout_path)
        self._build()
        self.statusBar().showMessage(msg)
        self._update_calibration_chip()
        QShortcut(QKeySequence("F1"), self, activated=lambda: show_manual(self))     # F1 = manual
        # ---- modo FOTO
        self.photo_worker: PhotoWorker | None = None
        self.photo_store: ResultsStore | None = None
        self.photo_paths: list[Path] = []
        self.photo_index = 0
        self.photo_layout: ParkingLayout | None = None
        self.photo_policy = "ask"                 # "ask" | "skip" (continuar sin puestos)
        self.photo_cache: dict[int, tuple[QImage, dict]] = {}
        self.photo_exporting = False
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._photo_step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._photo_step(+1))
        # ---- pantalla completa del visor (F11 entra/sale, Esc sale)
        self.video_fullscreen = False
        self._pre_fullscreen_state = Qt.WindowState.WindowNoState
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.toggle_video_fullscreen)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=lambda: self.set_video_fullscreen(False))
        self.setAcceptDrops(True)                 # arrastrar y soltar archivos o carpetas

    # ================================================================ construcción
    def _build(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)
        self._root_layout = root

        self.header_bar = QWidget()
        header = QHBoxLayout(self.header_bar)
        header.setContentsMargins(0, 0, 0, 0)
        title = logo_label(WORDMARK, 46)
        sub = QLabel("MONITOREO DE PARQUEO\nVisión Inteligente para el Monitoreo y Ordenamiento del Parqueo")
        sub.setObjectName("appSubtitle")
        header.addWidget(title)
        header.addSpacing(12)
        header.addWidget(sub)
        header.addStretch(1)
        self.chip_calib, self.chip_device = self._chip("PUESTOS: —"), self._chip("DISPOSITIVO: —")
        self.chip_camera, self.chip_fps = self._chip("CÁMARA: DETENIDA"), self._chip("FPS: —")
        for c in (self.chip_calib, self.chip_device, self.chip_camera, self.chip_fps):
            header.addWidget(c)
        root.addWidget(self.header_bar)

        self.tiles_bar = QWidget()
        tiles = QHBoxLayout(self.tiles_bar)
        tiles.setContentsMargins(0, 0, 0, 0)
        self.tiles = {"detectadas": Tile("MOTOS"), "estacionadas": Tile("ESTACIONADAS"),
                      "moviendose": Tile("MOVIÉNDOSE")}
        plural = {"EXCELENTE": "EXCELENTES", "BUENO": "BUENAS", "REGULAR": "REGULARES", "MALO": "MALAS",
                  "PÉSIMO": "PÉSIMAS"}
        for name in CLASS_NAMES:
            self.tiles[name] = Tile(plural[name], bgr_to_hex(CLASS_COLORS_BGR[name]))
        for t in self.tiles.values():
            tiles.addWidget(t)
        root.addWidget(self.tiles_bar)

        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        self.controls_panel = self._controls()
        tl.addWidget(self.controls_panel)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self._photo_bar())
        self.video = VideoView()                  # el banner de alertas se dibuja dentro del video
        self.video.clicked.connect(self._video_clicked)
        cl.addWidget(self._viewer_bar())
        cl.addWidget(self.video, 1)
        tl.addWidget(center, 1)
        self.side_panel = self._side_panel()
        tl.addWidget(self.side_panel)
        split.addWidget(top)
        self.tabs_panel = self._tabs()
        split.addWidget(self.tabs_panel)
        split.setSizes([650, 260])
        root.addWidget(split, 1)
        self.setStatusBar(QStatusBar())

    def _chip(self, text: str) -> QLabel:
        c = QLabel(text)
        c.setObjectName("chip")
        return c

    def _section(self, lay, text: str):
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        lay.addSpacing(6)
        lay.addWidget(lbl)

    def _button(self, lay, text: str, slot, name: str = "") -> QPushButton:
        b = QPushButton(text)
        if name:
            b.setObjectName(name)
        b.clicked.connect(slot)
        lay.addWidget(b)
        return b

    def _controls(self) -> QWidget:
        """Panel izquierdo. Va dentro de un QScrollArea: si la ventana es baja aparece una barra de
        desplazamiento en vez de aplastar los botones unos sobre otros."""
        w = QFrame()
        w.setObjectName("card")
        w.setFixedWidth(272)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 6, 5, 6)          # la barra de desplazamiento no pegada al borde del panel
        content = QWidget()
        content.setObjectName("controlsContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(12, 4, 14, 4)          # aire entre los botones y la barra de desplazamiento
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("controlsScroll")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(content)
        outer.addWidget(self.controls_scroll)
        self._section(lay, "FUENTE DE VIDEO")
        self.source_combo = QComboBox()
        for kind in SOURCE_TYPES:
            self.source_combo.addItem(SOURCE_LABELS[kind], kind)
        self.source_edit = QLineEdit()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        lay.addWidget(self.source_combo)
        lay.addWidget(self.source_edit)
        self._button(lay, "📂  EXAMINAR ARCHIVO", self.open_files_dialog, "primaryButton").setToolTip(
            "Abrir uno o varios videos o fotos (también puede arrastrarlos a la ventana)")
        self._button(lay, "🗂  EXAMINAR CARPETA (fotos)", self.open_folder_dialog)
        kind = str(self.cfg.SOURCE_TYPE) if str(self.cfg.SOURCE_TYPE) in SOURCE_TYPES else "webcam"
        self.source_combo.setCurrentIndex(list(SOURCE_TYPES).index(kind))
        self._source_changed()

        self.btn_start = self._button(lay, "▶  INICIAR", lambda: self.start_processing(), "primaryButton")
        self.btn_stop = GlowButton("■  DETENER")        # borde, letras e ícono rojos; brilla mientras está activo
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.clicked.connect(self.stop_processing)
        lay.addWidget(self.btn_stop)
        self.btn_stop.setEnabled(False)

        self._section(lay, "PARQUEADERO")
        self._button(lay, "📐  CALIBRAR PARQUEADERO", self.open_calibration)
        self._button(lay, "📂  CARGAR CALIBRACIÓN", self.load_calibration)

        self._section(lay, "RESULTADOS")
        self.btn_save = self._button(lay, "💾  Guardar", lambda: self._send("save"))
        self.btn_shot = self._button(lay, "📸  Captura de pantalla", lambda: self._send("screenshot"))
        self.btn_rec = QPushButton("⏺  Grabar video")
        self.btn_rec.setCheckable(True)
        self.btn_rec.toggled.connect(self._toggle_record)
        lay.addWidget(self.btn_rec)
        for b in (self.btn_save, self.btn_shot, self.btn_rec):
            b.setEnabled(False)
        self._button(lay, "📁  Abrir carpeta results", self._open_results)
        lay.addSpacing(6)
        self._button(lay, "⚙  Configuración", self.open_settings)
        self._button(lay, "❓  Ayuda (F1)", lambda: show_manual(self))
        lay.addStretch(1)
        self.models_label = QLabel("")
        self.models_label.setWordWrap(True)
        self.models_label.setStyleSheet(f"color:{DIM};font-size:8pt;")
        lay.addWidget(self.models_label)
        return w

    def _photo_bar(self) -> QWidget:
        """Barra del MODO FOTO: navegación, guardar resultado y exportar lote."""
        bar = QFrame()
        bar.setObjectName("card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        self.btn_prev = self._button(lay, "◀  Anterior (←)", lambda: self._photo_step(-1))
        self.photo_counter = QLabel("0 / 0")
        self.photo_counter.setObjectName("sectionTitle")
        lay.addWidget(self.photo_counter)
        self.btn_next = self._button(lay, "Siguiente (→)  ▶", lambda: self._photo_step(+1))
        lay.addStretch(1)
        self.photo_progress = QProgressBar()
        self.photo_progress.setVisible(False)
        self.photo_progress.setFixedWidth(220)
        lay.addWidget(self.photo_progress)
        self.btn_photo_save = self._button(lay, "💾  GUARDAR RESULTADO", self._save_photo_result, "primaryButton")
        self.btn_export = self._button(lay, "📦  EXPORTAR LOTE", self._export_batch)
        self._button(lay, "✕  Cerrar fotos", self._exit_photo_mode)
        bar.setVisible(False)
        self.photo_bar = bar
        return bar

    def _viewer_bar(self) -> QWidget:
        """Barra del visor: zoom (−, %, +, ajustar) y pantalla completa. Igual para cámara, video y foto."""
        bar = QFrame()
        bar.setObjectName("card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 2, 8, 2)
        hint = QLabel("Rueda: zoom · botón central o arrastrar: mover · doble clic: ajustar")
        hint.setStyleSheet(f"color:{DIM};font-size:8pt;")
        lay.addWidget(hint)
        lay.addStretch(1)
        self._button(lay, "−", lambda: self.video.zoom_by(1 / VideoView.ZOOM_STEP)).setToolTip("Alejar (−)")
        self.zoom_label = QLabel("100 %")
        self.zoom_label.setMinimumWidth(self.zoom_label.fontMetrics().horizontalAdvance("800 %") + 6)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.zoom_label)
        self._button(lay, "+", lambda: self.video.zoom_by(VideoView.ZOOM_STEP)).setToolTip("Acercar (+)")
        self._button(lay, "⤢  Ajustar", self.video.reset_zoom).setToolTip("Ver la imagen completa (0 o doble clic)")
        self._button(lay, "⛶  Pantalla completa (F11)", self.toggle_video_fullscreen)
        self.video.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))
        self.viewer_bar = bar
        return bar

    def _side_panel(self) -> QWidget:
        w = QFrame()
        w.setObjectName("card")
        w.setFixedWidth(360)
        lay = QVBoxLayout(w)
        t = QLabel("MOTO SELECCIONADA (clic en el video o en la tabla)")
        t.setObjectName("sectionTitle")
        t.setWordWrap(True)
        lay.addWidget(t)
        self.info = QTextBrowser()
        lay.addWidget(self.info, 3)
        t2 = QLabel("⚠ ALERTAS")
        t2.setObjectName("sectionTitle")
        lay.addWidget(t2)
        self.alerts_box = QTextBrowser()
        self.alerts_box.setMaximumHeight(170)
        lay.addWidget(self.alerts_box, 1)
        return w

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # arrastrar con el clic izquierdo desplaza la tabla (como en pantalla táctil, con inercia);
        # un clic corto sigue seleccionando la fila
        t.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        QScroller.grabGesture(t.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)
        return t

    def _tabs(self) -> QWidget:
        tabs = QTabWidget()
        self.table = self._table(["ID", "PLACA", "PUESTO", "SCORE", "ESTADO"])
        self.table.itemSelectionChanged.connect(self._table_selected)
        tabs.addTab(self.table, "🏍  Motos en vivo")
        self.events_table = self._table(["HORA", "ID", "EVENTO", "PUESTO", "SCORE", "ESTADO", "PLACA", "DETALLE"])
        tabs.addTab(self.events_table, "📋  Eventos")

        hist = QWidget()
        hl = QVBoxLayout(hist)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Placa o ID (también puesto, estado)...")
        self.search.returnPressed.connect(self.refresh_history)
        self.date_check = QCheckBox("Fecha:")
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        bar.addWidget(self.search, 1)
        bar.addWidget(self.date_check)
        bar.addWidget(self.date_edit)
        self._button(bar, "🔎  Buscar", self.refresh_history)
        hl.addLayout(bar)
        hs = QSplitter(Qt.Orientation.Horizontal)
        self.history_table = self._table(["SESIÓN", "ID", "PLACA", "PUESTO", "SCORE",
                                          "ESTADO", "PRIMERA VEZ", "ÚLTIMA VEZ"])
        self.history_table.itemSelectionChanged.connect(self._history_selected)
        self.history_events = self._table(["HORA", "EVENTO", "PUESTO", "SCORE", "ESTADO", "DETALLE"])
        hs.addWidget(self.history_table)
        hs.addWidget(self.history_events)
        hs.setSizes([750, 450])
        hl.addWidget(hs, 1)
        tabs.addTab(hist, "🗂  Historial / consulta")
        tabs.currentChanged.connect(lambda i: self.refresh_history() if i == 2 else None)
        return tabs

    # ================================================================ fuente
    def _source_changed(self):
        kind = self.source_combo.currentData()
        if kind == "video":
            value = self.cfg.CAMERA_SOURCE if str(self.cfg.SOURCE_TYPE) == "video" else self.cfg.VIDEO_PATH
        else:
            value = self.cfg.CAMERA_SOURCE if str(self.cfg.SOURCE_TYPE) == kind else ("0" if kind == "webcam" else "")
        self.source_edit.setText(str(value))
        self.source_edit.setPlaceholderText({"webcam": "0, 1, 2 ...", "ip": "http://192.168.1.50:8080/video",
                                             "rtsp": "rtsp://usuario:clave@IP:554/...",
                                             "video": "ruta del video .mp4",
                                             "image": "ruta de la foto .jpg / .png"}[kind])

    # ================================================================ pantalla completa del visor
    def _fullscreen_hidden_widgets(self) -> list[QWidget]:
        """Lo que se oculta en pantalla completa: solo queda el visor (la barra del modo foto se maneja aparte,
        ver `set_video_fullscreen`, porque su visibilidad normal depende de si hay modo foto activo)."""
        return [self.header_bar, self.tiles_bar, self.controls_panel, self.side_panel, self.tabs_panel,
                self.viewer_bar, self.statusBar()]

    def _update_fullscreen_counter(self):
        """Contador "n / total" discreto en la esquina del visor, solo en pantalla completa + modo foto
        (la barra con los botones de navegación/guardar/exportar queda oculta, ver `_fullscreen_hidden_widgets`)."""
        show = self.video_fullscreen and self.photo_paths
        n = len(self.photo_paths)
        self.video.set_corner_text(f"{self.photo_index + 1} / {n}" if show else "")

    def toggle_video_fullscreen(self):
        self.set_video_fullscreen(not self.video_fullscreen)

    def set_video_fullscreen(self, on: bool):
        """Pantalla completa: la ventana ocupa todo el monitor y solo se ve el visor (misma ventana, así que
        el clic para elegir moto y las flechas ← → del modo foto siguen funcionando)."""
        if on == self.video_fullscreen:
            return
        self.video_fullscreen = on
        for w in self._fullscreen_hidden_widgets():
            w.setVisible(not on)
        self.photo_bar.setVisible(not on and bool(self.photo_paths))
        if on:
            self._pre_fullscreen_state = self.windowState()
            self._root_layout.setContentsMargins(0, 0, 0, 0)
            self.showFullScreen()
            self.video.show_hint("Pantalla completa · F11 o Esc: salir · rueda: zoom · doble clic: ajustar")
        else:
            self._root_layout.setContentsMargins(12, 10, 12, 6)
            self.setWindowState(self._pre_fullscreen_state)
            self.show()
        self._update_fullscreen_counter()
        self.video.setFocus()

    # ================================================================ EXAMINAR ARCHIVO
    def open_files_dialog(self):
        """Selector nativo: uno o varios videos o fotos."""
        paths, _ = QFileDialog.getOpenFileNames(self, "EXAMINAR ARCHIVO (video o foto)",
                                                str(self.cfg.resolve_path(".")), FILE_FILTERS)
        if paths:
            self.open_paths([Path(p) for p in paths])

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "EXAMINAR CARPETA de fotos", str(self.cfg.resolve_path(".")))
        if not folder:
            return
        images = list_images_in_folder(folder)
        if not images:
            QMessageBox.information(self, "Examinar carpeta", f"La carpeta no contiene fotos soportadas:\n{folder}")
            return
        self.open_paths(images)

    def open_paths(self, paths: list[Path]):
        """Clasifica por extensión y, si no, por contenido. Fotos -> modo foto; video -> análisis de video."""
        images, videos, bad = [], [], []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                images += list_images_in_folder(p)
                continue
            kind = detect_file_kind(p)
            (images if kind == "image" else videos if kind == "video" else bad).append(p)
        if bad:
            QMessageBox.warning(self, "Archivo no soportado",
                                "Estos archivos no son videos ni fotos soportados, o están dañados:\n\n"
                                + "\n".join(f"• {b.name}" for b in bad[:10])
                                + ("\n..." if len(bad) > 10 else "")
                                + "\n\nFormatos: videos " + _VIDEO_GLOB + " · fotos " + _IMAGE_GLOB)
        if images:
            if videos:
                QMessageBox.information(self, "Examinar archivo", "Se abrirán las fotos seleccionadas. "
                                        "Los videos se abren de uno en uno con EXAMINAR ARCHIVO.")
            self._enter_photo_mode(images)
        elif videos:
            if len(videos) > 1:
                QMessageBox.information(self, "Examinar archivo", f"Se abrirá solo el primer video: {videos[0].name}")
            self._exit_photo_mode()
            if self.thread is not None:
                self.thread.request_stop()
                self.thread.wait(8000)
                self._on_stopped("Cambio de fuente")
            self.source_combo.setCurrentIndex(list(SOURCE_TYPES).index("video"))
            self.source_edit.setText(str(videos[0]))
            self.start_processing()

    def keyPressEvent(self, ev):
        # Respaldo de los atajos ← →: llega aquí cuando ningún control hijo usa la tecla
        # (p. ej. si Windows no reactivó la ventana tras cerrar un diálogo).
        if self._photo_mode() and ev.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self._photo_step(-1 if ev.key() == Qt.Key.Key_Left else +1)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [Path(u.toLocalFile()) for u in ev.mimeData().urls() if u.isLocalFile()]
        if paths:
            ev.acceptProposedAction()
            self.open_paths(paths)

    # ================================================================ MODO FOTO
    def _enter_photo_mode(self, images: list[Path]):
        if self.thread is not None:                    # se detiene la cámara/video en curso
            self.thread.request_stop()
            self.thread.wait(8000)
            self._on_stopped("Cambio a modo foto")
        self.photo_paths, self.photo_index = list(images), 0
        self.photo_layout, self.photo_policy = self.parking_layout, "ask"
        self.photo_cache.clear()
        self.photo_bar.setVisible(not self.video_fullscreen)
        self.btn_export.setEnabled(len(images) > 1)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.source_combo.setCurrentIndex(list(SOURCE_TYPES).index("image"))
        self.source_edit.setText(str(images[0]))
        self.events_table.setRowCount(0)
        if self.photo_worker is None:
            self.photo_store = ResultsStore(RESULTS_DIR, bool(self.cfg.USE_SQLITE))
            w = PhotoWorker(self.cfg.copy(), self.photo_store, parent=self)
            w.ready.connect(lambda d: self.chip_device.setText(f"DISPOSITIVO: {d}"))
            w.result.connect(self._on_photo_result)
            w.mismatch.connect(self._on_photo_mismatch)
            w.failed.connect(self._on_photo_failed)
            w.saved.connect(lambda p: (self.statusBar().showMessage(f"Resultado guardado: {p}"),
                                       QMessageBox.information(self, "GUARDAR RESULTADO",
                                                               f"Foto anotada guardada en:\n{p}\n\n"
                                                               "Filas añadidas a results/detections.csv")))
            w.progress.connect(self._on_batch_progress)
            w.batch_done.connect(self._on_batch_done)
            w.status.connect(self.statusBar().showMessage)
            w.fatal.connect(self._on_photo_fatal)
            self.photo_worker = w
            w.start()
            self.video.show_message("Cargando modelos para el modo foto...")
        self._show_photo(0)

    def _photo_mode(self) -> bool:
        return bool(self.photo_paths)

    def _photo_step(self, delta: int):
        if not self._photo_mode() or self.photo_exporting:
            return
        i = self.photo_index + delta
        if 0 <= i < len(self.photo_paths):
            self._show_photo(i)

    def _show_photo(self, i: int):
        self.photo_index = i
        n = len(self.photo_paths)
        self.photo_counter.setText(f"{i + 1} / {n}  ·  {self.photo_paths[i].name}")
        self._update_fullscreen_counter()
        self.source_edit.setText(str(self.photo_paths[i]))
        self.chip_camera.setText(f"CÁMARA: FOTO {i + 1}/{n}")
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setEnabled(i < n - 1)
        self.btn_photo_save.setEnabled(i in self.photo_cache)
        self._select(None)
        self.video.setFocus()          # que ← → no las capture la casilla de texto de la fuente
        if i in self.photo_cache:
            self._display_photo(*self.photo_cache[i])
        else:
            self.video.show_message(f"Analizando {self.photo_paths[i].name}...")
            self.photo_worker.analyze(i, str(self.photo_paths[i]), self.photo_layout, self.photo_policy)

    def _on_photo_result(self, i: int, img: QImage, p: dict):
        self.photo_cache[i] = (img, p)
        if i == self.photo_index and self._photo_mode():
            self.btn_photo_save.setEnabled(True)
            self._display_photo(img, p)

    def _display_photo(self, img: QImage, p: dict):
        self.last_payload, self.last_frame = p, p["frame"]
        self.video.set_image(img)
        s = p["summary"]
        for key in ("detectadas", "estacionadas", "moviendose"):
            self.tiles[key].set(s[key])
        for name in CLASS_NAMES:
            self.tiles[name].set(s["por_clase"][name])
        self.chip_fps.setText("MODO FOTO")
        self.models_label.setText(f"Detector: {p['detector']} (sin tracking)\n{p['wheels']}\nPlacas: {p['plate_status']}\n"
                                  f"Calibración: {p['calibration']}\n"
                                  f"Análisis: {p['process_ms']:.0f} ms")
        self._update_table(p["motos"])
        self._update_info(p["motos"])
        self._update_alerts(p["alerts"])
        msg = " · ".join(p["notes"]) if p["notes"] else ""
        if not p["motos"]:
            msg = "No se detectaron motocicletas. " + msg
        self.statusBar().showMessage(f"{Path(p['path']).name}: {msg}")

    def _on_photo_mismatch(self, i: int, message: str, frame):
        if i != self.photo_index or not self._photo_mode():
            return
        box = QMessageBox(QMessageBox.Icon.Warning, "La calibración no coincide con la foto",
                          f"{message}\n\nLos puestos solo se pueden aplicar a fotos con la misma resolución "
                          "(o proporción) y el mismo punto de vista que la calibración.", parent=self)
        recal = box.addButton("Recalibrar sobre esta foto", QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton("Continuar sin puestos", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is recal:
            save_path = self.cfg.resolve_path("config") / f"foto_{self.photo_paths[i].stem}.json"
            dlg = CalibrationDialog(frame, None, save_path, self.cfg, parent=self,
                                    source_desc=f"Foto: {self.photo_paths[i].name}")
            dlg.exec()
            if dlg.saved and dlg.result_layout is not None:
                self.photo_layout = dlg.result_layout
                self.photo_cache.clear()
                self.statusBar().showMessage(f"Calibración de la foto guardada en {save_path}")
            self._show_photo(i)
        elif box.clickedButton() is skip:
            self.photo_policy = "skip"             # para todo el lote: solo detección y placa
            self._show_photo(i)
        else:
            self.video.show_message("Análisis cancelado: la calibración no coincide con la foto.")

    def _on_photo_failed(self, i: int, message: str):
        if i == self.photo_index or i < 0:
            name = self.photo_paths[i].name if 0 <= i < len(self.photo_paths) else ""
            self.video.show_message(f"No se pudo analizar\n{name}\n\n{message}")
            self.statusBar().showMessage(message.replace("\n", " "))
            QMessageBox.warning(self, "Modo foto", message)

    def _on_photo_fatal(self, message: str):
        QMessageBox.critical(self, "Modo foto", message)
        self.photo_worker = None
        self._exit_photo_mode()

    def _save_photo_result(self):
        if self._photo_mode() and self.photo_index in self.photo_cache:
            self.photo_worker.save(self.photo_index)

    def _export_batch(self):
        if not self._photo_mode():
            return
        if self.photo_exporting:
            self.photo_worker.cancel_export()
            return
        default = RESULTS_DIR / f"lote_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "EXPORTAR LOTE (CSV resumen)", str(default), "CSV (*.csv)")
        if not path:
            return
        self.photo_exporting = True
        self.btn_export.setText("■  Cancelar exportación")
        for b in (self.btn_prev, self.btn_next, self.btn_photo_save):
            b.setEnabled(False)
        self.photo_progress.setRange(0, len(self.photo_paths))
        self.photo_progress.setValue(0)
        self.photo_progress.setVisible(True)
        self.photo_worker.export([str(p) for p in self.photo_paths], self.photo_layout, Path(path))

    def _on_batch_progress(self, i: int, n: int, name: str):
        self.photo_progress.setValue(i - 1)
        self.photo_progress.setFormat(f"{i}/{n}  {name}")
        self.statusBar().showMessage(f"Exportando lote: {i}/{n}  {name}")

    def _on_batch_done(self, csv_path: str, done: int, motos: int, errors):
        self.photo_exporting = False
        self.photo_progress.setValue(self.photo_progress.maximum())
        self.photo_progress.setVisible(False)
        self.btn_export.setText("📦  EXPORTAR LOTE")
        if self._photo_mode():
            self._show_photo(self.photo_index)
        detail = ("\n\nProblemas:\n" + "\n".join(f"• {e.splitlines()[0]}" for e in errors[:8])) if errors else ""
        QMessageBox.information(self, "EXPORTAR LOTE",
                                f"{done} de {len(self.photo_paths)} fotos procesadas, {motos} motos.\n"
                                f"CSV resumen:\n{csv_path}{detail}")

    def _exit_photo_mode(self):
        if self.photo_worker is not None:
            self.photo_worker.stop()
            self.photo_worker.wait(15000)
            self.photo_worker.deleteLater()
            self.photo_worker = None
        if self.photo_store is not None:
            self.photo_store.close()
            self.photo_store = None
        self.photo_paths, self.photo_cache = [], {}
        self.photo_exporting = False
        self._update_fullscreen_counter()
        if hasattr(self, "photo_bar"):
            self.photo_bar.setVisible(False)
            self.chip_fps.setText("FPS: —")
            self.video.show_message("VIMOP\n\nPulse  ▶ INICIAR  o  📂 EXAMINAR ARCHIVO")

    def _current_source(self) -> tuple[str, object] | None:
        kind, value = self.source_combo.currentData(), self.source_edit.text().strip()
        if not value:
            QMessageBox.warning(self, "Fuente", "Escriba el índice, la URL o la ruta de la fuente de video.")
            return None
        if kind == "webcam":
            if not value.isdigit():
                QMessageBox.warning(self, "Fuente", "Para webcam escriba el índice (0, 1, 2...).")
                return None
            return kind, int(value)
        return kind, value

    def use_configured_source(self):
        """Pone en el panel la fuente de config.json (pudo cambiar en Configuración desde el menú inicial)."""
        kind = str(self.cfg.SOURCE_TYPE)
        self.source_combo.setCurrentIndex(list(SOURCE_TYPES).index(kind if kind in SOURCE_TYPES else "webcam"))
        self._source_changed()

    # ================================================================ acciones
    def run_choice(self, choice: str | None, paths: list[Path] | None = None):
        """Acción elegida en el menú inicial."""
        if choice == "camera":
            self.use_configured_source()
            self.start_processing()
        elif choice == "video":
            self.source_combo.setCurrentIndex(list(SOURCE_TYPES).index("video"))
            self.source_edit.setText(str(self.cfg.VIDEO_PATH))
            self.start_processing()
        elif choice == "file":
            self.open_paths(paths) if paths else self.open_files_dialog()
        elif choice == "calibrate":
            self.open_calibration()
        elif choice == "load":
            self.load_calibration()

    def start_processing(self):
        if self.thread is not None:
            return
        src = self._current_source()
        if src is None:
            return
        if src[0] == "image":                          # una foto: modo foto
            self.open_paths([Path(str(src[1]))])
            return
        self._exit_photo_mode()
        if self.parking_layout is None:
            box = QMessageBox(QMessageBox.Icon.Warning, "Puestos no calibrados",
                              "No hay puestos calibrados: se detectarán y seguirán las motos, pero NO se podrá "
                              "evaluar el parqueo.", parent=self)
            cal = box.addButton("Calibrar ahora", QMessageBox.ButtonRole.AcceptRole)
            cont = box.addButton("Continuar sin calibrar", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is cal:
                self.open_calibration()
                if self.parking_layout is None:
                    return
            elif box.clickedButton() is not cont:
                return
        kind, value = src
        if kind == "video" and self.source_combo.currentData() == "video" and str(self.cfg.SOURCE_TYPE) != "video":
            self.cfg.VIDEO_PATH = value
        else:
            self.cfg.SOURCE_TYPE, self.cfg.CAMERA_SOURCE = kind, value
        try:
            self.cfg.save()
        except OSError as exc:
            log.warning("No se pudo guardar config.json: %s", exc)
        self.store = ResultsStore(RESULTS_DIR, bool(self.cfg.USE_SQLITE))
        self.thread = ProcessingThread(self.cfg.copy(), kind, value, self.parking_layout, self.store, parent=self)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.status.connect(self.statusBar().showMessage)
        self.thread.camera_status.connect(lambda s: self.chip_camera.setText(f"CÁMARA: {s}"))
        self.thread.error.connect(lambda m: QMessageBox.critical(self, "VIMOP", m))
        self.thread.stopped.connect(self._on_stopped)
        self.thread.selected_id = self.selected_id
        self.events_table.setRowCount(0)
        self.video.show_message(f"Cargando modelos...\n{describe_source(kind, value)}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        for b in (self.btn_save, self.btn_shot, self.btn_rec):
            b.setEnabled(True)
        self.thread.start()

    def stop_processing(self):
        if self.thread is not None:
            self.statusBar().showMessage("Deteniendo y guardando resultados...")
            self.thread.request_stop()

    def _send(self, cmd: str, arg=None):
        if self.thread is not None:
            self.thread.send(cmd, arg)

    def _toggle_record(self, on: bool):
        self.btn_rec.setText("⏹  Detener grabación" if on else "⏺  Grabar video")
        self._send("record", on)

    def _on_stopped(self, reason: str):
        if self.thread is not None:
            self.thread.wait(5000)
            self.thread.deleteLater()
        self.thread = None
        if self.store is not None:
            self.store.close()
            self.store = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_rec.blockSignals(True)
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("⏺  Grabar video")
        self.btn_rec.blockSignals(False)
        for b in (self.btn_save, self.btn_shot, self.btn_rec):
            b.setEnabled(False)
        self.chip_fps.setText("FPS: —")
        self.statusBar().showMessage(f"{reason}. Resultados guardados en {RESULTS_DIR}")
        if reason in ("Video terminado", "Cámara no disponible"):
            QMessageBox.information(self, "VIMOP", f"{reason}.\nResultados guardados en:\n{RESULTS_DIR}")

    def _on_frame(self, img: QImage, p: dict):
        self.last_payload, self.last_frame = p, p["frame"]
        self.video.set_image(img)
        s = p["summary"]
        for key in ("detectadas", "estacionadas", "moviendose"):
            self.tiles[key].set(s[key])
        for name in CLASS_NAMES:
            self.tiles[name].set(s["por_clase"][name])
        self.chip_fps.setText(f"FPS: {p['fps']:.1f}")
        self.chip_device.setText(f"DISPOSITIVO: {p['device']}")
        self.models_label.setText(
            f"Detector: {p['detector']} · tracker {p['tracker']}\n{p['wheels']}\nPlacas: {p['plate_status']}\n"
            f"Inferencia {p['inference_ms']:.0f} ms · total {p['process_ms']:.0f} ms")
        for ev in p["events"]:
            self._add_event(ev)
        if p["new_alerts"] and bool(self.cfg.ALERT_SOUND):
            self._beep()
        if time.monotonic() - self._last_table >= 0.25:           # tablas a 4 Hz
            self._last_table = time.monotonic()
            self._update_table(p["motos"])
            self._update_info(p["motos"])
            self._update_alerts(p["alerts"])

    @staticmethod
    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            else:
                QApplication.beep()
        except Exception as exc:          # el sonido es opcional: nunca debe detener el programa
            log.debug("No se pudo reproducir el sonido de alerta: %s", exc)

    # ================================================================ tablas y panel
    @staticmethod
    def _item(text, color: str | None = None, bold: bool = False) -> QTableWidgetItem:
        it = QTableWidgetItem("—" if text is None or text == "" else str(text))
        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if color:
            it.setBackground(QColor(color))
            it.setForeground(QColor("#111111"))
        if bold:
            f = it.font()
            f.setBold(True)
            it.setFont(f)
        return it

    @staticmethod
    def _class_hex(cls) -> str | None:
        return bgr_to_hex(CLASS_COLORS_BGR[cls]) if cls in CLASS_COLORS_BGR else None

    def _update_table(self, motos: list[dict]):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(motos))
        sel = -1
        for r, m in enumerate(motos):
            color = self._class_hex(m["clasificacion"])
            estado = m["clasificacion"] or m["estado_movimiento"]
            vals = [f"{m['id']:02d}", m["placa"], m["puesto"],
                    f"{m['puntaje']:.1f}" if m["puntaje"] is not None else None, estado]
            for c, v in enumerate(vals):
                t.setItem(r, c, self._item(v, color if c in (3, 4) else None, bold=c in (0, 3, 4)))
            if m["id"] == self.selected_id:
                sel = r
        if sel >= 0:
            t.selectRow(sel)
        else:
            t.clearSelection()
        t.blockSignals(False)

    def _update_info(self, motos: list[dict]):
        if not motos:
            self.info.setHtml(f"<p style='color:{DIM}'>No hay motos detectadas.</p>")
            return
        sel = next((m for m in motos if m["id"] == self.selected_id), None)
        parts = [moto_card_html(sel, True), f"<div style='color:{MUTED};margin-top:8px'>OTRAS MOTOS</div>"] if sel \
            else [f"<div style='color:{MUTED}'>Haga clic en una moto para ver su ficha completa.</div>"]
        parts += [moto_card_html(m, False) for m in motos if sel is None or m["id"] != sel["id"]]
        pos = self.info.verticalScrollBar().value()
        self.info.setHtml("".join(parts))
        self.info.verticalScrollBar().setValue(pos)

    def _update_alerts(self, alerts: list[str]):
        if not alerts:
            self.alerts_box.setHtml(f"<p style='color:{OK}'>✔ Sin alertas</p>")
        else:
            self.alerts_box.setHtml("".join(f"<p style='color:{ALERT};font-weight:700;margin:2px 0'>"
                                            f"{html.escape(a)}</p>" for a in alerts))

    def _add_event(self, ev: dict):
        t = self.events_table
        t.insertRow(0)
        color = self._class_hex(ev["estado"])
        for c, v in enumerate([ev["hora"], f"{ev['id']:02d}", ev["evento"], ev["puesto"], ev["score"], ev["estado"],
                               ev["placa"], ev["detalle"]]):
            t.setItem(0, c, self._item(v, color if c == 5 else None, bold=c == 2))
        if t.rowCount() > 500:
            t.removeRow(t.rowCount() - 1)

    def _video_clicked(self, x: float, y: float):
        if not self.last_payload:
            return
        hits = [m for m in self.last_payload["motos"]
                if m["bbox"][0] <= x <= m["bbox"][2] and m["bbox"][1] <= y <= m["bbox"][3]]
        if hits:          # la caja más pequeña que contiene el clic (motos superpuestas)
            m = min(hits, key=lambda m: (m["bbox"][2] - m["bbox"][0]) * (m["bbox"][3] - m["bbox"][1]))
            self._select(m["id"])
        else:
            self._select(None)

    def _table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows and self.table.item(rows[0].row(), 0):
            self._select(int(self.table.item(rows[0].row(), 0).text()))

    def _select(self, moto_id: int | None):
        self.selected_id = moto_id
        if self.thread is not None:
            self.thread.selected_id = moto_id
        if self.last_payload:
            self._update_info(self.last_payload["motos"])
            self._update_table(self.last_payload["motos"])

    # ================================================================ historial
    def refresh_history(self):
        date = self.date_edit.date().toString("yyyy-MM-dd") if self.date_check.isChecked() else ""
        rows = query_motos(RESULTS_DIR / "vimop.db", self.search.text(), date)
        t = self.history_table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for r, m in enumerate(rows):
            color = self._class_hex(m["classification"])
            vals = [m["session"], m["moto_id"], m["plate"], m["space"],
                    f"{m['score']:.1f}" if m["score"] is not None else None, m["classification"],
                    m["first_seen"], m["last_seen"]]
            for c, v in enumerate(vals):
                t.setItem(r, c, self._item(v, color if c in (4, 5) else None))
        t.blockSignals(False)
        self.history_events.setRowCount(0)
        self.statusBar().showMessage(f"Historial: {len(rows)} motos encontradas.")

    def _history_selected(self):
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        evs = query_events(RESULTS_DIR / "vimop.db", self.history_table.item(r, 0).text(),
                           int(self.history_table.item(r, 1).text()))
        t = self.history_events
        t.setRowCount(len(evs))
        for i, e in enumerate(evs):
            color = self._class_hex(e["classification"])
            vals = [e["timestamp"].replace("T", " "), e["event_type"], e["space"],
                    f"{e['score']:.1f}" if e["score"] is not None else None, e["classification"], e["detail"]]
            for c, v in enumerate(vals):
                t.setItem(i, c, self._item(v, color if c == 4 else None))

    def _open_results(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(RESULTS_DIR))
            else:
                QMessageBox.information(self, "Resultados", str(RESULTS_DIR))
        except OSError as exc:
            QMessageBox.warning(self, "Resultados", str(exc))

    # ================================================================ calibración
    def _update_calibration_chip(self):
        lay = self.parking_layout
        if lay is None:
            self.chip_calib.setText("PUESTOS: SIN CALIBRAR")
            self.chip_calib.setStyleSheet(f"color:{WARN_STRONG};")
        else:
            self.chip_calib.setText(f"PUESTOS: {len(lay.spaces)}")
            self.chip_calib.setStyleSheet(f"color:{OK};")

    def _grab_frame(self) -> np.ndarray:
        if (self.thread is not None or self._photo_mode()) and self.last_frame is not None:
            return self.last_frame.copy()
        src = self._current_source()
        if src is None:
            raise CameraError("No hay una fuente configurada.")
        return grab_single_frame(src[0], src[1], self.cfg, int(self.cfg.CALIBRATION_FRAME_INDEX))

    def open_calibration(self, parent=None) -> bool:
        """Abre la calibración. `parent` es la ventana para los diálogos (por defecto, esta misma); se usa
        cuando se llama desde el menú inicial, con la ventana principal aún oculta. Devuelve True si la
        calibración cambió (se guardaron puestos nuevos)."""
        parent = parent or self
        try:
            frame = self._grab_frame()
        except CameraError as exc:
            ans = QMessageBox.question(parent, "Calibración", f"No se pudo obtener un frame de la fuente:\n{exc}\n\n"
                                                              "¿Calibrar sobre una imagen guardada?")
            if ans != QMessageBox.StandardButton.Yes:
                return False
            path, _ = QFileDialog.getOpenFileName(parent, "Imagen del parqueadero", "",
                                                  "Imágenes (*.png *.jpg *.jpeg *.bmp)")
            if not path:
                return False
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                QMessageBox.warning(parent, "Calibración", "No se pudo leer la imagen.")
                return False
        src = self._current_source()
        dlg = CalibrationDialog(frame, self.parking_layout, self.layout_path, self.cfg, frame_provider=self._grab_frame,
                                source_desc=describe_source(*src) if src else "", parent=parent)
        dlg.exec()
        if dlg.saved and dlg.result_layout is not None:
            self._set_layout(dlg.result_layout)
            return True
        return False

    def load_calibration(self, parent=None) -> bool:
        """Carga una calibración guardada. `parent` es la ventana para los diálogos (ver open_calibration).
        Devuelve True si se cargó una calibración válida."""
        parent = parent or self
        path, _ = QFileDialog.getOpenFileName(parent, "CARGAR CALIBRACIÓN", str(self.layout_path.parent),
                                              "JSON (*.json)")
        if not path:
            return False
        layout, msg = load_layout(Path(path))
        if layout is None:
            QMessageBox.warning(parent, "Calibración", msg)
            return False
        self.layout_path = Path(path)
        try:
            self.cfg.PARKING_SPACES_FILE = self.layout_path.relative_to(self.cfg.resolve_path(".")).as_posix()
        except ValueError:
            self.cfg.PARKING_SPACES_FILE = str(self.layout_path)
        try:
            self.cfg.save()
        except OSError as exc:
            log.warning("No se pudo guardar config.json: %s", exc)
        self._set_layout(layout)
        QMessageBox.information(parent, "Calibración", msg)
        return True

    def _set_layout(self, layout: ParkingLayout):
        self.parking_layout = layout
        self._update_calibration_chip()
        self._send("layout", layout)
        self.statusBar().showMessage(f"Calibración activa: {len(layout.spaces)} puestos")

    def open_settings(self):
        if SettingsDialog(self.cfg, self).exec():
            self._source_changed()

    def closeEvent(self, ev):
        self._exit_photo_mode()
        if self.thread is not None:
            self.thread.request_stop()
            self.thread.wait(8000)
        if self.store is not None:
            self.store.close()
        ev.accept()

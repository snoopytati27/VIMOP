"""ui/dialogs.py — Menú inicial y diálogo de configuración (escribe en config.json)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from config import Config
from core.camera import describe_source
from core.image_io import FILE_FILTERS
from ui.assets import ICON, WORDMARK, logo_label
from ui.help import show_manual
from ui.theme import MUTED


class SettingsDialog(QDialog):
    FIELDS = [
        ("Fuente de video", [
            ("SOURCE_TYPE", "Tipo de fuente", "combo", ["webcam", "ip", "rtsp", "video", "image"]),
            ("MAX_IMAGE_SIDE", "Lado máximo de las fotos (px)", "int", (320, 8000)),
            ("CAMERA_SOURCE", "Cámara (índice / URL / ruta)", "text", None),
            ("VIDEO_PATH", "Video de prueba", "text", None),
            ("VIDEO_LOOP", "Repetir video", "bool", None),
            ("VIDEO_REALTIME", "Video a velocidad real", "bool", None),
            ("PROCESS_MAX_WIDTH", "Ancho máximo de procesamiento", "int", (320, 3840)),
        ]),
        ("Modelos y tracking", [
            ("MODEL_PATH", "Modelo YOLO ('auto' = según GPU/CPU)", "text", None),
            ("DEVICE", "Dispositivo", "combo", ["auto", "cpu", "cuda:0"]),
            ("IMG_SIZE", "Tamaño de inferencia", "int", (256, 1536)),
            ("CONFIDENCE", "Confianza mínima", "float", (0.05, 0.95)),
            ("TRACKER", "Tracker", "combo", ["bytetrack", "botsort"]),
            ("ID_RETENTION_SECONDS", "Conservar ID perdido (s)", "float", (1, 600)),
            ("PLATE_MODEL_PATH", "Detector de placas", "text", None),
        ]),
        ("Identificación", [
            ("OCR_INTERVAL", "OCR cada N frames", "int", (1, 300)),
            ("PLATE_REGEX", "Regex de placa", "text", None),
            ("HISTORY_LENGTH", "Historial de score (N)", "int", (1, 200)),
        ]),
        ("Análisis", [
            ("ORIENTATION_WEIGHT", "Peso orientación", "float", (0, 1)),
            ("LOCATION_WEIGHT", "Peso ubicación", "float", (0, 1)),
            ("CENTERING_WEIGHT", "Peso centrado", "float", (0, 1)),
            ("INVASION_WEIGHT", "Peso invasión", "float", (0, 1)),
            ("MIN_ASSIGN", "Mínimo de la línea dentro del puesto (asignación)", "float", (0.05, 0.95)),
            ("ERR_MAX", "ERR_MAX (°): error angular con el que O = 0", "float", (5, 90)),
            ("WHEEL_BACKEND", "Ruedas (pose = modelo entrenado, mask = máscara)", "combo", ["auto", "pose", "mask"]),
            ("MIN_AXIS_LEN_RATIO", "Longitud mínima de la línea (× alto del bbox)", "float", (0.05, 1.0)),
            ("SHOW_BBOX", "Dibujar el bbox tenue (referencia)", "bool", None),
            ("PARKED_STABLE_SECONDS", "Segundos quieta para ESTACIONADA", "float", (0.5, 30)),
            ("CLASS_HYSTERESIS", "Histéresis de categoría (puntos)", "float", (0, 10)),
            ("ALERT_SOUND", "Sonido en alertas", "bool", None),
        ]),
    ]

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VIMOP — Configuración")
        self.resize(640, 780)
        self.cfg = cfg
        self.widgets: dict[str, tuple[str, QWidget]] = {}
        content = QWidget()
        lay = QVBoxLayout(content)
        for section, fields in self.FIELDS:
            box = QGroupBox(section)
            form = QFormLayout(box)
            for key, label, kind, extra in fields:
                val = getattr(cfg, key)
                if kind == "combo":
                    w = QComboBox()
                    w.addItems(extra)
                    w.setCurrentText(str(val))
                elif kind == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(val))
                elif kind == "int":
                    w = QSpinBox()
                    w.setRange(*extra)
                    w.setValue(int(val))
                elif kind == "float":
                    w = QDoubleSpinBox()
                    w.setDecimals(3)
                    w.setSingleStep(0.05)
                    w.setRange(*extra)
                    w.setValue(float(val))
                else:
                    w = QLineEdit(str(val))
                form.addRow(label + ":", w)
                self.widgets[key] = (kind, w)
            lay.addWidget(box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        note = QLabel("Se guarda en config.json y se aplica la próxima vez que pulse ▶ INICIAR.")
        note.setObjectName("appSubtitle")
        root.addWidget(note)
        root.addWidget(scroll, 1)
        root.addWidget(buttons)

    def _save(self):
        for key, (kind, w) in self.widgets.items():
            if kind == "combo":
                val = w.currentText()
            elif kind == "bool":
                val = w.isChecked()
            elif kind in ("int", "float"):
                val = w.value() if kind == "int" else round(w.value(), 4)
            else:
                val = w.text().strip()
                if key == "CAMERA_SOURCE" and val.isdigit():
                    val = int(val)
            setattr(self.cfg, key, val)
        total = sum(float(getattr(self.cfg, k)) for k in
                    ("ORIENTATION_WEIGHT", "LOCATION_WEIGHT", "CENTERING_WEIGHT", "INVASION_WEIGHT"))
        if abs(total - 1.0) > 0.01:
            QMessageBox.warning(self, "Pesos", f"Los pesos suman {total:.2f}; se normalizarán para sumar 1.")
        try:
            path = self.cfg.save()
        except OSError as exc:
            QMessageBox.critical(self, "Configuración", f"No se pudo guardar:\n{exc}")
            return
        QMessageBox.information(self, "Configuración", f"Guardado en\n{path}")
        self.accept()


def _camera_source_text(cfg: Config) -> str:
    """Descripción corta de la fuente configurada actualmente para "Iniciar cámara" (webcam 0, video: x.mp4...)."""
    return describe_source(str(cfg.SOURCE_TYPE), cfg.CAMERA_SOURCE)


class StartMenu(QDialog):
    """Centro de VIMOP: MONITOREAR (abre la ventana principal) y PREPARAR EL PARQUEADERO (se queda en el menú)."""

    def __init__(self, cfg: Config, calibration_status: Callable[[], str],
                actions: dict[str, Callable[[QWidget], bool]], parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.calibration_status = calibration_status
        self.actions = actions                        # {"calibrate": fn(parent)->bool, "load": fn(parent)->bool}
        self.choice: str | None = None
        self.chosen_paths: list[Path] | None = None
        self.buttons: dict[str, QPushButton] = {}
        self.setWindowTitle("VIMOP")
        self.setFixedWidth(540)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        brand = QHBoxLayout()                          # ícono + nombre, centrados
        brand.addStretch(1)
        brand.addWidget(logo_label(ICON, 84, fallback=""))
        brand.addSpacing(10)
        brand.addWidget(logo_label(WORDMARK, 64))
        brand.addStretch(1)
        lay.addLayout(brand)
        sub = QLabel("Visión Inteligente para el Monitoreo\ny Ordenamiento del Parqueo")
        sub.setObjectName("appSubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)

        lay.addWidget(self._section_title("MONITOREAR"))
        self._add_button(lay, "camera", self._camera_button_text())
        self._add_button(lay, "file", "2.  🗂  Examinar archivo (video o foto)")

        lay.addSpacing(6)
        lay.addWidget(self._section_title("PREPARAR EL PARQUEADERO"))
        self.cal_label = QLabel(self.calibration_status())
        self.cal_label.setWordWrap(True)
        self.cal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cal_label.setStyleSheet(f"color:{MUTED};margin:2px 0 6px;")
        lay.addWidget(self.cal_label)
        self._add_button(lay, "calibrate", "1.  📐  Calibrar parqueadero")
        self._add_button(lay, "load", "2.  📂  Cargar calibración")
        self._add_button(lay, "settings", "3.  ⚙  Configuración")

        help_btn = QPushButton("❓  Ayuda / manual de usuario (F1)")
        help_btn.clicked.connect(lambda: show_manual(self))
        QShortcut(QKeySequence("F1"), self, activated=lambda: show_manual(self))
        exit_btn = QPushButton("Salir")
        exit_btn.clicked.connect(self.reject)
        lay.addSpacing(8)
        lay.addWidget(help_btn)
        lay.addWidget(exit_btn)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return lbl

    def _add_button(self, lay: QVBoxLayout, key: str, text: str) -> None:
        b = QPushButton(text)
        b.setObjectName("menuButton")
        b.clicked.connect(lambda _=False, k=key: self._pick(k))
        lay.addWidget(b)
        self.buttons[key] = b

    def _camera_button_text(self) -> str:
        return f"1.  📹  Iniciar cámara\n      ({_camera_source_text(self.cfg)})"

    def _refresh_camera_button(self) -> None:
        self.buttons["camera"].setText(self._camera_button_text())

    def _refresh_calibration_label(self) -> None:
        self.cal_label.setText(self.calibration_status())

    def _pick(self, key: str):
        if key == "settings":
            if SettingsDialog(self.cfg, self).exec():
                self._refresh_camera_button()
            return
        if key in self.actions:                        # "calibrate" / "load": se ejecutan aquí y se vuelve al menú
            self.actions[key](self)
            self._refresh_calibration_label()
            return
        if key == "file":
            paths, _ = QFileDialog.getOpenFileNames(self, "EXAMINAR ARCHIVO (video o foto)",
                                                     str(self.cfg.resolve_path(".")), FILE_FILTERS)
            if not paths:                              # canceló: se queda en el menú
                return
            self.chosen_paths = [Path(p) for p in paths]
            self.choice = key
            self.accept()
            return
        self.choice = key                              # "camera": abre la ventana principal
        self.accept()

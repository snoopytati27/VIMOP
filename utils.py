"""
utils.py — Utilidades compartidas: clasificación del score, colores, recortes y logging.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Clasificación (diapositiva 07). LISTA ordenada de mayor a menor umbral.
# Colores en BGR (convención de OpenCV).
# --------------------------------------------------------------------------
CLASSIFICATION_TABLE: list[tuple[float, str, tuple[int, int, int]]] = [
    (90.0, "EXCELENTE", (83, 200, 0)),     # verde
    (75.0, "BUENO", (255, 121, 41)),       # azul
    (60.0, "REGULAR", (0, 214, 255)),      # amarillo
    (40.0, "MALO", (0, 128, 255)),         # naranja
    (0.0, "PÉSIMO", (35, 28, 229)),        # rojo
]
CLASS_NAMES = [name for _, name, _ in CLASSIFICATION_TABLE]
CLASS_COLORS_BGR = {name: bgr for _, name, bgr in CLASSIFICATION_TABLE}
_CLASS_RANGES = {}          # nombre -> (mínimo, máximo) del rango de score
_upper = 100.0 + 1e-9
for _thr, _name, _ in CLASSIFICATION_TABLE:
    _CLASS_RANGES[_name] = (_thr, _upper)
    _upper = _thr

MOVING_BGR = (185, 185, 185)       # motos en movimiento: gris, sin score
PERSON_BGR = (255, 110, 190)       # personas (morado)
FREE_SPACE_BGR = (118, 230, 0)
OCCUPIED_SPACE_BGR = (82, 82, 255)

NO_PLATE = "No detectada"


def classify_score(score: float) -> str:
    """90–100 EXCELENTE · 75–89 BUENO · 60–74 REGULAR · 40–59 MALO · 0–39 PÉSIMO."""
    for threshold, name, _ in CLASSIFICATION_TABLE:
        if score >= threshold:
            return name
    return CLASSIFICATION_TABLE[-1][1]


def classify_with_hysteresis(score: float, previous: str | None, margin: float) -> str:
    """
    Igual que classify_score, pero la categoría anterior se conserva mientras el
    score no salga de su rango más un margen. Ej. con margen 2: una moto BUENO
    (75–89.9) no pasa a REGULAR hasta bajar de 73, ni a EXCELENTE hasta llegar a 92.
    Evita que el color del bbox parpadee cuando el score oscila en un corte.
    """
    if previous in _CLASS_RANGES:
        lo, hi = _CLASS_RANGES[previous]
        if lo - margin <= score < hi + margin:
            return previous
    return classify_score(score)


def class_color_bgr(name: str | None) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(name or "", MOVING_BGR)


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# Cajas e imágenes
# --------------------------------------------------------------------------
def bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def boxes_intersect(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        return cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return frame


def safe_crop(frame: np.ndarray, bbox, pad_ratio: float = 0.0):
    """Recorte con margen, sin salirse de la imagen. Devuelve (recorte, bbox_usado)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pw, ph = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# --------------------------------------------------------------------------
# Logging (consola + results/vimop.log)
# --------------------------------------------------------------------------
def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(getattr(h, "_vimop", False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._vimop = True  # type: ignore[attr-defined]
        root.addHandler(sh)
        try:
            fh = logging.FileHandler(results_dir / "vimop.log", encoding="utf-8")
            fh.setFormatter(fmt)
            fh._vimop = True  # type: ignore[attr-defined]
            root.addHandler(fh)
        except OSError as exc:
            root.warning("No se pudo crear results/vimop.log: %s", exc)
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

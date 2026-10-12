"""ui/assets.py — Logos de VIMOP (carpeta assets/).

Los PNG son "luz sobre negro" (fondo negro opaco). Al cargarlos, el negro se vuelve transparente sin perder
el brillo: alfa = el canal más brillante de cada píxel y el color se des-premultiplica. Así el logo se ve igual
sobre cualquier fondo oscuro (el degradado de la ventana, un panel) sin un rectángulo negro alrededor.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
from PySide6.QtWidgets import QLabel

from config import BASE_DIR

ASSETS_DIR = BASE_DIR / "assets"
WORDMARK = "logo_vimop_wordmark.png"      # el nombre "VIMOP"
ICON = "logo_vimop_icon.png"              # el ícono circular (moto + cámara)
BLACK_LEVEL = 6                           # por debajo de esto es ruido de compresión: totalmente transparente
TRIM_ALPHA = 12                           # para recortar el margen: píxeles con menos alfa cuentan como vacío
TRIM_PAD = 0.02                           # margen que se deja al recortar (fracción del lado mayor)


@lru_cache(maxsize=None)
def light_image(name: str) -> QImage | None:
    """Carga assets/<name> con el negro convertido en transparencia y el margen vacío recortado."""
    img = QImage(str(ASSETS_DIR / name))
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
    rgb = arr[..., :3].astype(np.float32)
    peak = rgb.max(axis=2)
    alpha = np.clip((peak - BLACK_LEVEL) * 255.0 / (255 - BLACK_LEVEL), 0, 255)
    color = np.where(peak[..., None] > 0, rgb * (255.0 / np.maximum(peak, 1))[..., None], 0)
    out = np.dstack([np.clip(color, 0, 255), alpha]).astype(np.uint8)
    ys, xs = np.nonzero(alpha > TRIM_ALPHA)
    if len(xs):
        pad = int(max(w, h) * TRIM_PAD)
        out = out[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1]
    out = np.ascontiguousarray(out)
    oh, ow = out.shape[:2]
    return QImage(out.data, ow, oh, ow * 4, QImage.Format.Format_RGBA8888).copy()


def logo_pixmap(name: str, height: int) -> QPixmap | None:
    """El logo escalado a `height` px lógicos (nítido en pantallas con escalado, p. ej. 125 %)."""
    img = light_image(name)
    if img is None:
        return None
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    pix = QPixmap.fromImage(img.scaledToHeight(round(height * dpr), Qt.TransformationMode.SmoothTransformation))
    pix.setDevicePixelRatio(dpr)
    return pix


def logo_label(name: str, height: int, fallback: str = "VIMOP") -> QLabel:
    """QLabel con el logo; si falta el archivo, el texto `fallback` con el estilo de título (appTitle)."""
    lbl = QLabel()
    pix = logo_pixmap(name, height)
    if pix is None:
        lbl.setText(fallback)
        lbl.setObjectName("appTitle")
    else:
        lbl.setPixmap(pix)
        lbl.setAccessibleName(fallback)
    return lbl


def app_icon() -> QIcon:
    """Ícono de la ventana / barra de tareas (vacío si falta el archivo). Aquí sí se usa el PNG original,
    con su fondo negro: en una barra de tareas clara el ícono transparente se vería lavado."""
    path = ASSETS_DIR / ICON
    return QIcon(str(path)) if path.is_file() else QIcon()

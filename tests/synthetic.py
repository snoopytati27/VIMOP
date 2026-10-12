"""Utilidades para construir escenas sintéticas (puestos y motos) en las pruebas."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.parking_spaces import ParkingLayout, ParkingSpace  # noqa: E402
from config import Config, defaults  # noqa: E402
from core.detector import Detection  # noqa: E402

IMG_W, IMG_H = 800, 600


def test_config(**overrides) -> Config:
    """Configuración por defecto (sin leer el config.json del usuario ni cargar wheels_pose.pt)."""
    values = defaults()
    values["WHEEL_BACKEND"] = "mask"
    values.update(overrides)
    return Config(values)


def rect(x1, y1, x2, y2) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def layout_px(spaces: dict[str, list]):
    return ParkingLayout([ParkingSpace(k, v) for k, v in spaces.items()], (IMG_W, IMG_H)).runtime(IMG_W, IMG_H)


def moto(x1, y1, x2, y2, mask=None) -> Detection:
    """Detección sintética de una moto con bbox (x1, y1, x2, y2)."""
    return Detection("motorcycle", "motorcycle", 0.9, (x1, y1, x2, y2), mask, 1)


def wheels(rear, front, source: str = "mask", confidence: float = 0.9) -> dict:
    """Resultado de WheelLocator.locate para pruebas: línea R–F ya conocida."""
    return {"rear": rear, "front": front, "source": source, "confidence": confidence}


# Escena del ejemplo numérico del manual: dos puestos de 100 x 200 px, uno al lado del otro.
# Cada moto = (bbox, rueda trasera R, rueda delantera F).
MANUAL_SPACES = {"P01": rect(100, 100, 200, 300), "P02": rect(200, 100, 300, 300)}
MANUAL_EXCELLENT = ((110, 115, 190, 295), (148, 128), (156, 282))
MANUAL_BAD = ((130, 150, 230, 300), (150, 160), (215, 285))

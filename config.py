"""
config.py — Parámetros de VIMOP.

Aquí están TODOS los valores por defecto. Para cambiarlos sin tocar código se
edita `config.json` (en la raíz del proyecto) o se usa el menú "Configuración"
de la interfaz, que escribe en ese mismo archivo. Al arrancar se cargan estos
valores y luego se aplican encima los de config.json.

Ejemplos de config.json:
    {"SOURCE_TYPE": "webcam", "CAMERA_SOURCE": 0}
    {"SOURCE_TYPE": "rtsp",   "CAMERA_SOURCE": "rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101"}
    {"SOURCE_TYPE": "ip",     "CAMERA_SOURCE": "http://192.168.1.50:8080/video"}
    {"SOURCE_TYPE": "video",  "CAMERA_SOURCE": "videos/prueba.mp4"}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("vimop.config")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_JSON = BASE_DIR / "config.json"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# ==========================================================================
# 1. FUENTE DE VIDEO
# ==========================================================================
SOURCE_TYPE = "webcam"          # "webcam" | "ip" | "rtsp" | "video" | "image" (foto)
CAMERA_SOURCE: Any = 0          # índice de webcam, URL http/rtsp o ruta de video
VIDEO_PATH = "WhatsApp Video 2026-09-22 at 16.39.37.mp4"   # usado por "4. Probar video"
VIDEO_LOOP = False
VIDEO_REALTIME = True           # no reproducir el video más rápido que sus FPS
VIDEO_FRAME_STEP = 1            # procesar 1 de cada N frames del video
CAMERA_RECONNECT_SECONDS = 3.0  # espera entre reintentos de conexión
CAMERA_MAX_RETRIES = 0          # 0 = reintentar indefinidamente
CAMERA_OPEN_TIMEOUT_MS = 8000
RTSP_TRANSPORT = "tcp"
PROCESS_MAX_WIDTH = 1280        # los frames más anchos se reducen antes de procesar
IR_CLAHE = False                # ecualizar (CLAHE) las imágenes infrarrojas antes de detectar; lo usa también
                                # la herramienta de etiquetado, para que el dataset vea lo mismo que el programa
IR_SATURATION_MAX = 12.0        # saturación media (0–255) bajo la cual una imagen se considera infrarroja

# ---- Modo FOTO (Examinar archivo) ----
MAX_IMAGE_SIDE = 1920           # fotos con un lado mayor se reducen (se avisa al usuario)
VIEWPOINT_MIN_SIMILARITY = 0.35 # similitud mínima entre la foto y la imagen de calibración (0–1)
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# ==========================================================================
# 2. MODELOS
# ==========================================================================
# Detector de motos. Por defecto un modelo "-seg": su máscara alimenta el respaldo "mask" de
# WheelLocator (línea de ejes). El bbox se usa solo para detección y tracking.
# Con un modelo sin segmentación (yolo11m.pt) y sin wheels_pose.pt no hay línea de ejes real:
# la orientación se marca "no confiable".
MODEL_PATH = "auto"             # "auto": GPU -> MODEL_PATH_GPU, CPU -> MODEL_PATH_CPU
MODEL_PATH_GPU = "models/yolo11s-seg.pt"
MODEL_PATH_CPU = "models/yolo11s-seg.pt"
DEVICE = "auto"                 # "auto" | "cpu" | "cuda:0"
HALF_PRECISION = True           # FP16 solo en GPU
IMG_SIZE = 640
CONFIDENCE = 0.35
TRACK_LOW_CONFIDENCE = 0.12     # detecciones débiles que ByteTrack usa para no perder tracks
IOU = 0.5
MOTORCYCLE_CLASSES = ["motorcycle"]
PERSON_CLASSES = ["person"]
DETECT_PERSONS = True

PLATE_MODEL_PATH = "models/plate.pt"                       # detector YOLO de placas (opcional)
PLATE_MODEL_CONFIDENCE = 0.30
PLATE_MODEL_IMGSZ = 640                                    # tamaño de inferencia sobre el recorte de la moto (640 = el del entrenamiento; más alto empeora)
PLATE_CROP_MARGIN = 0.5                                    # contexto extra alrededor de la moto (fracción del ancho/alto, por lado) para el detector YOLO de placas; el heurístico usa 0.05

# ==========================================================================
# 3. TRACKING / ID
# ==========================================================================
TRACKER = "bytetrack"           # "bytetrack" | "botsort"
TRACK_BUFFER = 90               # frames que el tracker recuerda un objeto perdido
TRACK_MIN_HITS = 3              # frames seguidos antes de asignar un "MOTO #n"
LOST_TIMEOUT_SECONDS = 1.5      # sin verse este tiempo -> se considera perdida
ID_RETENTION_SECONDS = 30.0     # el registro se conserva este tiempo por si el tracker la recupera
REID_IOU = 0.35
REID_APPEARANCE_MIN = 0.45

# ==========================================================================
# 4. IDENTIFICACIÓN
# ==========================================================================
OCR_INTERVAL = 15               # OCR de cada moto cada N frames
PLATE_REGEX = r"^[A-Z]{3}\d{2}[A-Z]$|^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$"     # colombiana de moto (ABC12D) | indonesia (L 6570 ABJ), varios formatos con "|"
PLATE_MIN_CONFIDENCE = 0.35
PLATE_MIN_VOTES = 2             # lecturas coincidentes mínimas
PLATE_MIN_CONSENSUS = 0.60      # fracción (ponderada) de votos que debe tener la placa ganadora
PLATE_LOCK_VOTES = 6            # con estas lecturas se deja de hacer OCR
PLATE_MIN_WIDTH_PX = 36         # placas más pequeñas son ilegibles: no se intenta OCR
PLATE_MIN_MOTO_HEIGHT_PX = 60   # motos más pequeñas en la imagen no se intentan
PLATE_OCR_BACKEND = "auto"      # "auto" (fast-plate-ocr si está instalado, si no EasyOCR) | "fast" | "easyocr"
PLATE_FAST_MODEL = "cct-s-v2-global-model"   # modelo de fast-plate-ocr (se descarga la 1.ª vez, ~5 MB)
PLATE_FAST_MIN_CONFIDENCE = 0.90             # confianza media por carácter mínima para aceptar su lectura
HISTORY_LENGTH = 15             # scores del promedio suavizado
IDENT_HISTORY_LENGTH = 30       # lecturas de placa por moto
POSITION_HISTORY_LENGTH = 90

# ==========================================================================
# 5. ANÁLISIS DEL PARQUEO (pesos definitivos 35/35/20/10)
#    Modelo: LÍNEA DE EJES. Cada moto es el segmento R–F entre el contacto de la rueda
#    trasera y el de la delantera con el piso. El bbox solo sirve para detección y tracking.
# ==========================================================================
ORIENTATION_WEIGHT = 0.35
LOCATION_WEIGHT = 0.35
CENTERING_WEIGHT = 0.20
INVASION_WEIGHT = 0.10
WHEEL_BACKEND = "auto"          # "pose" | "mask" | "auto" (pose si existe WHEEL_MODEL_PATH; si no, mask)
WHEEL_MODEL_PATH = "models/wheels_pose.pt"   # YOLO-pose con 2 puntos clave: rueda trasera y delantera
WHEEL_IMG_SIZE = 960            # tamaño de inferencia del modelo de ruedas (el del entrenamiento)
WHEEL_KPT_CONF = 0.30           # confianza mínima de cada punto clave; si no, se usa el respaldo mask
WHEEL_SMOOTH_FRAMES = 5         # media móvil de R y F por moto (la línea no tiembla)
BOTTOM_BAND = 0.25              # respaldo mask: franja inferior de la máscara (fracción de su altura)
INSET_FRACTION = 0.12           # respaldo mask: los extremos se meten esta fracción de la longitud
                                # (el extremo de la silueta es el borde de la rueda, no su contacto)
MIN_AXIS_LEN_RATIO = 0.35       # longitud de la línea / alto del bbox; menos = moto de frente/espaldas
SHOW_BBOX = True                # dibujar el bbox tenue, solo como referencia
MIN_ASSIGN = 0.30               # longitud(L ∩ S) / longitud(L) mínima para asignar el puesto
ERR_MAX = 45.0                  # error angular (°) con el que O = 0
INVASION_NONE = 0.02            # 0–2 %  ninguna
INVASION_SMALL = 0.10           # 2–10 % pequeña ; > 10 % considerable
INVASION_MAX = 0.30             # I = 100·(1 − min(1, ratio / 0.30))
CLASS_HYSTERESIS = 2.0          # puntos de margen para cambiar de categoría (evita parpadeo)
PARKING_SPACES_FILE = "config/parking_spaces.json"
CALIBRATION_FRAME_INDEX = 0

# ==========================================================================
# 6. ESTADOS DE LA MOTO
# ==========================================================================
MOTION_WINDOW_SECONDS = 0.8
MOVING_SPEED_HIGH = 0.30        # velocidad del punto medio M de la línea en diagonales del bbox por segundo
MOVING_SPEED_LOW = 0.12         # (independiente de la distancia a la cámara); histéresis alta/baja
PARKED_STABLE_SECONDS = 2.0
CAMERA_MOTION_COMPENSATION = True
CLASS_CHANGE_MIN_SECONDS = 1.5

# ==========================================================================
# 7. ALERTAS Y RESULTADOS
# ==========================================================================
CORRECT_MIN_SCORE = 60.0
ALERT_MIN_SCORE = 40.0
ALERT_MIN_LOCATION = 50.0
ALERT_SOUND = False
ALERT_BLINK = True
ALERT_REARM_SECONDS = 10.0      # una alerta que desaparece y vuelve antes de esto no se registra de nuevo
DETECTION_LOG_INTERVAL_SECONDS = 2.0
AUTOSAVE_SECONDS = 10.0
USE_SQLITE = True

# Claves que se escriben siempre en config.json para que sea fácil cambiar la fuente
_ALWAYS_SAVED = ("SOURCE_TYPE", "CAMERA_SOURCE", "VIDEO_PATH", "DEVICE", "TRACKER")
_NOT_SETTINGS = {"BASE_DIR", "CONFIG_JSON", "MODELS_DIR", "RESULTS_DIR"}


def defaults() -> dict[str, Any]:
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_") and k not in _NOT_SETTINGS}


class Config:
    """Parámetros con acceso por atributo: cfg.CONFIDENCE."""

    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k.isupper()}

    def copy(self) -> "Config":
        return Config(json.loads(json.dumps(self.as_dict())))

    @staticmethod
    def resolve_path(path: str | Path) -> Path:
        p = Path(str(path)).expanduser()
        return p if p.is_absolute() else BASE_DIR / p

    def weights(self) -> tuple[float, float, float, float]:
        """Pesos (O, U, C, I); si no suman 1 se normalizan."""
        w = [float(self.ORIENTATION_WEIGHT), float(self.LOCATION_WEIGHT),
             float(self.CENTERING_WEIGHT), float(self.INVASION_WEIGHT)]
        total = sum(w)
        if total <= 0:
            return 0.35, 0.35, 0.20, 0.10
        if abs(total - 1.0) > 1e-6:
            log.warning("Los pesos suman %.3f; se normalizan a 1.0", total)
        return w[0] / total, w[1] / total, w[2] / total, w[3] / total

    def save(self, path: Path = CONFIG_JSON) -> Path:
        base = defaults()
        data = {k: v for k, v in self.as_dict().items() if base.get(k) != v or k in _ALWAYS_SAVED}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def load_config(path: Path = CONFIG_JSON) -> Config:
    values = defaults()
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
            for key, val in overrides.items():
                if key in values:
                    values[key] = val
                else:
                    log.warning("Parámetro desconocido en %s: %s (ignorado)", path.name, key)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("No se pudo leer %s (%s); se usan valores por defecto.", path, exc)
    return Config(values)

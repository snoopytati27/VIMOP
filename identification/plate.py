"""
identification/plate.py — Reconocimiento de placas en 4 pasos (NO OCR sobre toda la moto).

    1. recorte de la moto
    2. detector de placa       YoloPlateDetector (models/plate.pt, si existe)
                               HeuristicPlateDetector (respaldo, MENOS FIABLE): contornos
                               rectangulares en la mitad inferior de la moto, con bonificación
                               si el fondo es amarillo (placas de moto en Colombia)
    3. preprocesado            ampliación, escala de grises, CLAHE y binarizado (Otsu)
    4. OCR                     EasyOCR, solo caracteres A-Z y 0-9

Validación: el texto debe cumplir PLATE_REGEX (por defecto ^[A-Z]{3}\\d{2}[A-Z]$, placa de
moto colombiana ABC12D). Antes se corrigen confusiones típicas SEGÚN LA POSICIÓN:
en posición de letra 0→O, 1→I, 8→B, 5→S...; en posición de número O→0, I→1, B→8, S→5...
Si no cumple el formato, la lectura se descarta. NUNCA se inventa una placa.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("vimop.plate")

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7",
            "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}


@dataclass
class PlateCandidate:
    image: np.ndarray
    box: tuple[int, int, int, int]      # en coordenadas del recorte de la moto
    score: float


@dataclass
class PlateRead:
    text: str
    confidence: float
    method: str
    raw: str = ""


# --------------------------------------------------------------------------
# 2. Detectores de placa
# --------------------------------------------------------------------------
class PlateDetector(ABC):
    name = "base"
    reliable = False

    @abstractmethod
    def detect(self, moto_bgr: np.ndarray) -> list[PlateCandidate]:
        ...


class YoloPlateDetector(PlateDetector):
    name = "YOLO placas (models/plate.pt)"
    reliable = True

    def __init__(self, model_path: str, device: str, conf: float):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device, self.conf = device, conf

    def detect(self, moto_bgr: np.ndarray) -> list[PlateCandidate]:
        res = self.model.predict(moto_bgr, imgsz=320, conf=self.conf, device=self.device, verbose=False)[0]
        if res.boxes is None:
            return []
        h, w = moto_bgr.shape[:2]
        out = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.12
            bx1, by1, bx2, by2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
            if bx2 - bx1 > 8 and by2 - by1 > 6:
                out.append(PlateCandidate(moto_bgr[by1:by2, bx1:bx2].copy(), (bx1, by1, bx2, by2), float(c)))
        return sorted(out, key=lambda c: -c.score)[:2]


class HeuristicPlateDetector(PlateDetector):
    """
    Respaldo sin modelo entrenado (menos fiable): busca en la MITAD INFERIOR de la moto
    contornos rectangulares con proporción de placa de moto (ancho/alto ≈ 1.2–2.8),
    con caracteres oscuros dentro. Si el fondo es amarillo suma puntos.
    """
    name = "heurística de contornos (menos fiable)"

    def detect(self, moto_bgr: np.ndarray) -> list[PlateCandidate]:
        h, w = moto_bgr.shape[:2]
        if h < 30 or w < 30:
            return []
        y0 = h // 2
        lower = moto_bgr[y0:]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(cv2.bilateralFilter(gray, 7, 50, 50), 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        yellow = cv2.inRange(hsv, (15, 120, 90), (38, 255, 255))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        area_ref = float(lower.shape[0] * lower.shape[1])
        cands = []
        for src in (edges, yellow):
            contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                if area < max(150.0, 0.004 * area_ref) or area > 0.30 * area_ref or bh < 8:
                    continue
                aspect = bw / float(bh)
                if not 1.2 <= aspect <= 2.8:
                    continue
                rect_fill = cv2.contourArea(c) / float(area)
                if rect_fill < 0.45:                            # poco rectangular
                    continue
                inner_v = hsv[y:y + bh, x:x + bw, 2]
                dark = float(np.mean(inner_v < 90))
                if not 0.05 <= dark <= 0.60:                     # sin "caracteres"
                    continue
                yellow_frac = float(np.mean(yellow[y:y + bh, x:x + bw] > 0))
                score = rect_fill * (1.0 - min(abs(aspect - 1.65) / 1.65, 1.0)) + 0.8 * yellow_frac
                px, py = int(bw * 0.08), int(bh * 0.12)
                bx1, by1 = max(0, x - px), max(0, y - py) + y0
                bx2, by2 = min(w, x + bw + px), min(h, y + bh + py + y0)
                cands.append(PlateCandidate(moto_bgr[by1:by2, bx1:bx2].copy(), (bx1, by1, bx2, by2), score))
        # supresión de duplicados (candidatos casi iguales de ambas fuentes)
        cands.sort(key=lambda c: -c.score)
        kept: list[PlateCandidate] = []
        for c in cands:
            if all(_iou(c.box, k.box) < 0.5 for k in kept):
                kept.append(c)
        return kept[:3]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# 3. Preprocesado
# --------------------------------------------------------------------------
def preprocess_plate(plate_bgr: np.ndarray, target_height: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (gris con CLAHE, binarizada con Otsu), ambas ampliadas a target_height."""
    h, w = plate_bgr.shape[:2]
    scale = target_height / float(max(h, 1))
    img = cv2.resize(plate_bgr, (max(1, int(w * scale)), target_height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 1.2), -0.6, 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary


# --------------------------------------------------------------------------
# 4. OCR
# --------------------------------------------------------------------------
class PlateOCR:
    """EasyOCR. Si no está instalado o no puede iniciar: available = False (sin cerrar el programa)."""

    def __init__(self, use_gpu: bool):
        self.available, self.error, self.reader = False, "", None
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
            self.available = True
        except ImportError:
            self.error = "EasyOCR no está instalado (pip install easyocr)"
        except Exception as exc:
            self.error = f"EasyOCR no pudo iniciar (¿sin internet para descargar sus modelos?): {exc}"
        if self.error:
            log.warning(self.error)

    def read(self, image: np.ndarray) -> list[tuple[tuple[float, float, float], str, float]]:
        """[((y_centro, x_centro, alto), texto, confianza), ...]"""
        if not self.available:
            return []
        tokens = []
        for box, text, conf in self.reader.readtext(image, allowlist=ALLOWLIST, detail=1, paragraph=False,
                                                    text_threshold=0.5, low_text=0.3, width_ths=0.8):
            pts = np.asarray(box, dtype=np.float32)
            text = re.sub(r"[^A-Z0-9]", "", str(text).upper())
            if text:
                tokens.append(((float(pts[:, 1].mean()), float(pts[:, 0].mean()),
                                float(pts[:, 1].max() - pts[:, 1].min())), text, float(conf)))
        return tokens


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def template_from_regex(regex: str) -> str | None:
    """'^[A-Z]{3}\\d{2}[A-Z]$' -> 'LLLDDL'. None si la regex es más compleja (no se corrige)."""
    body = regex.strip().lstrip("^").rstrip("$")
    token = re.compile(r"(\[A-Z\]|\\d|\[0-9\])(?:\{(\d+)\})?")
    out, pos = [], 0
    while pos < len(body):
        m = token.match(body, pos)
        if not m:
            return None
        out.append(("L" if m.group(1) == "[A-Z]" else "D") * int(m.group(2) or 1))
        pos = m.end()
    return "".join(out) or None


class PlateTextParser:
    def __init__(self, regex: str):
        self.regex = re.compile(regex)
        self.template = template_from_regex(regex)

    def fit(self, text: str) -> tuple[str, int] | None:
        """Corrige según la posición y valida. Devuelve (texto, nº de correcciones) o None."""
        if self.template is None:
            return (text, 0) if self.regex.match(text) else None
        if len(text) != len(self.template):
            return None
        out, fixes = [], 0
        for ch, kind in zip(text, self.template):
            if kind == "L" and not ch.isalpha():
                ch, fixes = TO_LETTER.get(ch, ""), fixes + 1
            elif kind == "D" and not ch.isdigit():
                ch, fixes = TO_DIGIT.get(ch, ""), fixes + 1
            if not ch:
                return None
            out.append(ch)
        fixed = "".join(out)
        return (fixed, fixes) if self.regex.match(fixed) else None

    def parse(self, tokens) -> tuple[str, float, str] | None:
        if not tokens:
            return None
        tokens = sorted(tokens, key=lambda t: t[0][0])          # por línea (arriba -> abajo)
        lines: list[list] = []
        for tok in tokens:
            if lines and abs(tok[0][0] - lines[-1][0][0][0]) < 0.5 * max(tok[0][2], 1.0):
                lines[-1].append(tok)
            else:
                lines.append([tok])
        ordered = [t for line in lines for t in sorted(line, key=lambda t: t[0][1])]
        texts, confs = [t[1] for t in ordered], [t[2] for t in ordered]
        n = len(texts)
        candidates = [("".join(texts[i:j]), float(np.mean(confs[i:j])))
                      for i in range(n) for j in range(i + 1, min(n, i + 3) + 1)]
        joined, mean_conf = "".join(texts), float(np.mean(confs))
        size = len(self.template) if self.template else 0
        if size:
            candidates += [(joined[k:k + size], mean_conf * 0.9) for k in range(max(0, len(joined) - size) + 1)]
        best = None
        for text, conf in candidates:
            fitted = self.fit(text)
            if fitted is None or fitted[1] > 2:
                continue
            score = conf * (0.9 ** fitted[1])
            if best is None or score > best[1]:
                best = (fitted[0], score, " ".join(texts))
        return best


class PlateReader:
    """Orquesta los 4 pasos. read(recorte_moto) -> PlateRead | None."""

    def __init__(self, cfg, device: str, use_gpu: bool):
        self.cfg = cfg
        path = cfg.resolve_path(cfg.PLATE_MODEL_PATH)
        self.detector: PlateDetector = HeuristicPlateDetector()
        self.warning = ""
        if path.exists():
            try:
                self.detector = YoloPlateDetector(str(path), device, float(cfg.PLATE_MODEL_CONFIDENCE))
            except Exception as exc:
                log.error("No se pudo cargar %s: %s", path.name, exc)
        if not self.detector.reliable:
            self.warning = (f"No existe {path.name}: se usa un detector de placas heurístico, MENOS FIABLE. "
                            "Entrene uno con tools/train_models.py plates.")
            log.warning(self.warning)
        self.ocr = PlateOCR(use_gpu)
        self.parser = PlateTextParser(str(cfg.PLATE_REGEX))

    @property
    def available(self) -> bool:
        return self.ocr.available

    @property
    def description(self) -> str:
        if not self.ocr.available:
            return f"No disponible: {self.ocr.error}"
        return f"{self.detector.name} + EasyOCR"

    def read(self, moto_bgr: np.ndarray) -> PlateRead | None:
        if not self.ocr.available or moto_bgr is None or moto_bgr.size == 0:
            return None
        best = None
        for cand in self.detector.detect(moto_bgr):
            # placas diminutas: caracteres de 2–3 px, el OCR solo daría basura
            if cand.image.shape[1] < int(self.cfg.PLATE_MIN_WIDTH_PX) or cand.image.shape[0] < 10:
                continue
            for variant in preprocess_plate(cand.image):
                parsed = self.parser.parse(self.ocr.read(variant))
                if parsed is None:
                    continue
                text, conf, raw = parsed
                if conf >= float(self.cfg.PLATE_MIN_CONFIDENCE) and (best is None or conf > best.confidence):
                    best = PlateRead(text, conf, self.detector.name, raw)
                break                          # la variante gris funcionó: no se prueba la binarizada
        return best

"""
tools/wheel_dataset.py — Lógica (sin interfaz) del dataset de puntos clave de las ruedas.

    project = WheelProject.open(carpeta)            # lee wheels_labels.json si existe
    scan_folder(project, every_seconds=2.0)         # fotos + fotogramas de los videos (sin casi-duplicados)
    preannotate(project, detector, locator, cfg)    # detector de motos + backend "mask" de WheelLocator
    project.save()                                  # guardar progreso (se puede reanudar)
    export_dataset(project, salida)                 # formato YOLO-pose + dataset.yaml + .zip

Cada moto = bbox + 2 puntos clave (rueda trasera, rueda delantera), cada uno con visibilidad:
    2 visible · 1 oculto por oclusión · 0 fuera de imagen
Formato de etiqueta YOLO-pose (una línea por moto, todo normalizado 0–1):
    0 cx cy w h  x1 y1 v1  x2 y2 v2
La partición train/val/test (70/15/15) se hace por FUENTE (un video, una sesión de fotos), nunca por fotograma:
los fotogramas casi idénticos de un mismo video no pueden caer en dos conjuntos. Las fotos de internet siempre
van a train, así que los porcentajes se calculan sobre las imágenes de la cámara.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.image_io import IMAGE_EXTS, VIDEO_EXTS, ImageLoadError, read_image, write_image
from core.preprocess import prepare_frame

log = logging.getLogger("vimop.wheel_dataset")

WORK_FILE = "wheels_labels.json"
FRAMES_DIR = "_frames"
EXPORT_DIR = "wheels_dataset"
SKIP_DIRS = {FRAMES_DIR, EXPORT_DIR, "runs", "__pycache__", ".git"}

VIS_VISIBLE, VIS_OCCLUDED, VIS_OUTSIDE = 2, 1, 0
VIS_LABELS = {VIS_VISIBLE: "visible", VIS_OCCLUDED: "oculto por oclusión", VIS_OUTSIDE: "fuera de imagen"}
NEXT_VIS = {VIS_VISIBLE: VIS_OCCLUDED, VIS_OCCLUDED: VIS_OUTSIDE, VIS_OUTSIDE: VIS_VISIBLE}


# ==========================================================================
# Modelo de datos
# ==========================================================================
@dataclass
class MotoLabel:
    bbox: list[float]                                       # x1, y1, x2, y2 en píxeles
    rear: list[float]                                       # x, y, v  (rueda trasera)
    front: list[float]                                      # x, y, v  (rueda delantera)
    manual: bool = False                                    # añadida a mano (no preetiquetada)

    def swap(self) -> None:
        self.rear, self.front = self.front, self.rear

    def points(self) -> list[list[float]]:
        return [self.rear, self.front]


@dataclass
class ImageEntry:
    path: str                                               # relativo a la carpeta del proyecto
    source: str                                             # fuente para la partición (video:x / sesion:x)
    width: int = 0
    height: int = 0
    reviewed: bool = False                                  # revisada por una persona
    prelabeled: bool = False                                # ya se le corrió el preetiquetado automático
    motos: list[MotoLabel] = field(default_factory=list)


@dataclass
class ExportResult:
    out_dir: Path
    zip_path: Path | None
    train_images: int
    val_images: int
    motos: int
    train_sources: list[str]
    val_sources: list[str]
    warnings: list[str] = field(default_factory=list)
    test_images: int = 0
    test_sources: list[str] = field(default_factory=list)


class WheelProject:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.entries: list[ImageEntry] = []
        self.every_seconds = 2.0
        self.dirty = False

    # ---- persistencia ----------------------------------------------------
    @property
    def work_file(self) -> Path:
        return self.folder / WORK_FILE

    @classmethod
    def open(cls, folder: str | Path) -> "WheelProject":
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"No existe la carpeta: {folder}")
        project = cls(folder)
        if project.work_file.exists():
            try:
                data = json.loads(project.work_file.read_text(encoding="utf-8"))
                project.every_seconds = float(data.get("every_seconds", 2.0))
                for raw in data.get("entries", []):
                    motos = [MotoLabel(**m) for m in raw.pop("motos", [])]
                    project.entries.append(ImageEntry(**raw, motos=motos))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"El archivo de trabajo {project.work_file.name} está dañado: {exc}") from exc
        return project

    def save(self) -> Path:
        data = {"version": 1, "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "every_seconds": self.every_seconds, "entries": [asdict(e) for e in self.entries]}
        tmp = self.work_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.work_file)                       # escritura atómica: un corte no deja el archivo a medias
        self.dirty = False
        return self.work_file

    # ---- consultas -------------------------------------------------------
    def abs_path(self, entry: ImageEntry) -> Path:
        return self.folder / entry.path

    def reviewed_count(self) -> int:
        return sum(e.reviewed for e in self.entries)

    def progress_text(self, index: int) -> str:
        return f"{index + 1} / {len(self.entries)} · revisadas {self.reviewed_count()}"

    def known_paths(self) -> set[str]:
        return {e.path for e in self.entries}


# ==========================================================================
# Carga de imágenes (mismo preprocesado que la aplicación)
# ==========================================================================
def load_entry_image(project: WheelProject, entry: ImageEntry, cfg) -> np.ndarray:
    """
    Imagen tal como la ve el programa: lectura con EXIF/alfa/tamaño (modo foto) + prepare_frame.
    Las coordenadas de las etiquetas son sobre ESTA imagen, y esta imagen es la que se exporta.
    """
    loaded = read_image(project.abs_path(entry), int(cfg.MAX_IMAGE_SIDE))
    return prepare_frame(loaded.image, cfg)


# ==========================================================================
# Escaneo de la carpeta: fotos y fotogramas de videos
# ==========================================================================
def photo_session(path: Path, root: Path) -> str:
    """Sesión de una foto: subcarpeta, si no la fecha del nombre (WhatsApp/cámara), si no la fecha del archivo."""
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return f"sesion:{rel.parts[0]}"
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", path.stem)
    if m:
        return f"sesion:{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return "sesion:" + datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def find_parent_project(folder: str | Path) -> Path | None:
    """Carpeta ancestro que ya es un proyecto de etiquetado (tiene wheels_labels.json), si la hay."""
    folder = Path(folder).resolve()
    for parent in folder.parents:
        if (parent / WORK_FILE).exists():
            return parent
    return None


def video_key(path: Path, root: Path) -> str:
    """Nombre único de un video: su nombre, y si está en una subcarpeta, la subcarpeta como prefijo."""
    rel = path.relative_to(root)
    return "_".join([*rel.parts[:-1], path.stem])


def dhash(frame: np.ndarray) -> int:
    """Hash perceptual de diferencia (64 bits): frames casi iguales dan hashes casi iguales."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
    return float(np.abs(ga.astype(np.int16) - gb.astype(np.int16)).mean())


def is_near_duplicate(frame: np.ndarray, kept: list[tuple[int, np.ndarray]], hash_bits: int = 5,
                      min_diff: float = 3.0) -> bool:
    """Casi duplicado del último fotograma conservado: hash perceptual parecido Y poca diferencia de píxeles."""
    if not kept:
        return False
    last_hash, last_small = kept[-1]
    return hamming(dhash(frame), last_hash) <= hash_bits and mean_abs_diff(frame, last_small) < min_diff


def extract_frames(video: Path, out_dir: Path, every_seconds: float = 2.0,
                   progress: Callable[[int, int], None] | None = None) -> list[Path]:
    """
    Guarda un fotograma cada `every_seconds` y descarta los casi duplicados del anterior conservado.
    Idempotente: no vuelve a escribir los que ya existen. Devuelve las rutas de los fotogramas conservados.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ImageLoadError(f"No se pudo abrir el video: {video.name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps * max(every_seconds, 0.05))))
        out_dir.mkdir(parents=True, exist_ok=True)
        kept: list[tuple[int, np.ndarray]] = []
        saved: list[Path] = []
        index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    if not is_near_duplicate(frame, kept):
                        target = out_dir / f"{video.stem}_t{index / fps:09.2f}.jpg"
                        if not target.exists() and not write_image(target, frame):
                            log.warning("No se pudo guardar %s", target)
                        else:
                            saved.append(target)
                            kept.append((dhash(frame), cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)))
                            kept = kept[-1:]                    # solo se compara con el último conservado
            if progress and total and index % 30 == 0:
                progress(index, total)
            index += 1
        return saved
    finally:
        cap.release()


def scan_folder(project: WheelProject, every_seconds: float | None = None,
                progress: Callable[[str], None] | None = None) -> tuple[int, int]:
    """
    Añade al proyecto las fotos de la carpeta (recursivo) y los fotogramas de sus videos.
    Es acumulativo: lo que ya estaba (con sus etiquetas) no se toca. Devuelve (fotos nuevas, fotogramas nuevos).
    """
    if every_seconds is not None:
        project.every_seconds = float(every_seconds)
    root = project.folder
    known = project.known_paths()
    new_photos = new_frames = 0
    files = sorted(p for p in root.rglob("*") if p.is_file() and not (set(p.relative_to(root).parts[:-1]) & SKIP_DIRS))
    for path in files:
        ext = path.suffix.lower()
        rel = path.relative_to(root).as_posix()
        if ext in IMAGE_EXTS and rel not in known:
            if progress:
                progress(f"Foto: {path.name}")
            project.entries.append(ImageEntry(rel, photo_session(path, root)))
            known.add(rel)
            new_photos += 1
        elif ext in VIDEO_EXTS:
            key = video_key(path, root)
            done_prefixes = [f"{FRAMES_DIR}/{key}/", (path.parent.relative_to(root) / FRAMES_DIR / path.stem).as_posix() + "/"]
            if any(e.path.startswith(pref) for e in project.entries for pref in done_prefixes):
                continue                                            # ya extraído: no se repite con otro intervalo
            if progress:
                progress(f"Video: {path.name}")
            try:
                frames = extract_frames(path, root / FRAMES_DIR / key, project.every_seconds)
            except ImageLoadError as exc:
                log.warning("%s", exc)
                continue
            for frame_path in frames:
                frel = frame_path.relative_to(root).as_posix()
                if frel not in known:
                    project.entries.append(ImageEntry(frel, f"video:{key}"))
                    known.add(frel)
                    new_frames += 1
    project.dirty = project.dirty or bool(new_photos or new_frames)
    return new_photos, new_frames


def merge_project(dest: WheelProject, src_folder: str | Path, source_map: dict[str, str] | None = None) -> int:
    """
    Trae a `dest` las imágenes (con sus etiquetas y su estado de revisión) de otro proyecto que vive en una
    subcarpeta de `dest.folder`, por ejemplo cuando se abrió `etiquetado/celular` como proyecto propio.
    Las rutas se reescriben relativas a `dest`; `source_map` permite renombrar o unir fuentes. No duplica.
    Devuelve cuántas imágenes se añadieron. El llamador guarda (`dest.save()`).
    """
    src_folder = Path(src_folder).resolve()
    prefix = src_folder.relative_to(dest.folder.resolve()).as_posix()
    other = WheelProject.open(src_folder)
    known = dest.known_paths()
    added = 0
    for entry in other.entries:
        new_path = f"{prefix}/{entry.path}"
        if new_path in known:
            continue
        entry.path = new_path
        entry.source = (source_map or {}).get(entry.source, entry.source)
        dest.entries.append(entry)
        known.add(new_path)
        added += 1
    dest.dirty = dest.dirty or bool(added)
    return added


# ==========================================================================
# Preetiquetado automático
# ==========================================================================
def preannotate(project: WheelProject, detector, locator, cfg, only_new: bool = True,
                progress: Callable[[int, int], None] | None = None) -> int:
    """
    Detecta las motos de cada imagen y coloca los 2 puntos con el backend "mask" de WheelLocator.
    No toca las imágenes ya revisadas ni las que ya se preetiquetaron (salvo only_new=False).
    Devuelve cuántas imágenes se preetiquetaron.
    """
    todo = [e for e in project.entries if not e.reviewed and not (only_new and e.prelabeled)]
    done = 0
    for i, entry in enumerate(todo):
        try:
            img = load_entry_image(project, entry, cfg)
        except ImageLoadError as exc:
            log.warning("%s", exc)
            continue
        entry.height, entry.width = img.shape[:2]
        entry.motos = []
        for det in detector.detect(img).motorcycles:
            det.track_id = None
            found = locator.locate(None, det)                     # sin frame: solo el respaldo por máscara
            entry.motos.append(MotoLabel([float(v) for v in det.bbox],
                                         [float(found["rear"][0]), float(found["rear"][1]), VIS_VISIBLE],
                                         [float(found["front"][0]), float(found["front"][1]), VIS_VISIBLE]))
        entry.prelabeled = True
        done += 1
        if progress:
            progress(i + 1, len(todo))
    project.dirty = project.dirty or bool(done)
    return done


# ==========================================================================
# Exportación YOLO-pose
# ==========================================================================
def yolo_pose_line(moto: MotoLabel, w: int, h: int) -> str | None:
    """Línea de etiqueta YOLO-pose de una moto; None si su bbox es degenerado."""
    x1, y1, x2, y2 = (float(v) for v in moto.bbox)
    x1, x2 = sorted((min(max(x1, 0.0), w), min(max(x2, 0.0), w)))
    y1, y2 = sorted((min(max(y1, 0.0), h), min(max(y2, 0.0), h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    parts = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"]
    for x, y, v in moto.points():
        v = int(v)
        if v == VIS_OUTSIDE:
            parts.append("0.000000 0.000000 0")                    # fuera de imagen: sin coordenadas
        else:
            parts.append(f"{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f} {v}")
    return " ".join(parts)


TRAIN_ONLY_PREFIXES = ("sesion:internet",)         # fotos ajenas a la cámara: solo entrenan, nunca validan


def split_sources(sources: dict[str, int], val_fraction: float = 0.2, seed: int = 0,
                  train_only_prefixes: tuple[str, ...] = TRAIN_ONLY_PREFIXES) -> tuple[set[str], set[str]]:
    """
    Reparte FUENTES completas entre train y val (nunca fotogramas sueltos).
    `sources` = {fuente: nº de imágenes}. Determinista con la semilla.
    Las fuentes que empiezan por `train_only_prefixes` (fotos de internet) siempre van a train: la validación debe
    medir el modelo con imágenes de la cámara. Con una sola fuente candidata a validar, todo va a train
    (el llamador avisa de que val no es independiente). La fracción de validación se calcula sobre las imágenes
    candidatas, no sobre las de internet.
    """
    candidates = {k: v for k, v in sources.items() if not k.startswith(train_only_prefixes)}
    names = sorted(candidates, key=lambda s: hashlib.md5(f"{seed}:{s}".encode()).hexdigest())
    total = sum(candidates.values())
    val: set[str] = set()
    count = 0
    for name in names:
        if len(val) >= len(names) - 1 or count >= val_fraction * total:
            break
        val.add(name)
        count += candidates[name]
    return set(sources) - val, val


def split_sources3(sources: dict[str, int], val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 0,
                   train_only_prefixes: tuple[str, ...] = TRAIN_ONLY_PREFIXES) -> tuple[set[str], set[str], set[str]]:
    """
    Reparte FUENTES completas entre train, val y test (70/15/15 por defecto). Las fuentes de internet van siempre
    a train y los porcentajes se miden sobre las imágenes candidatas (las de la cámara). Con pocas fuentes el
    reparto es grueso: se elige la combinación que más se acerca a lo pedido (determinista con la semilla), con
    val y test no vacíos. Con menos de 3 fuentes candidatas no hay test independiente (test vacío).
    """
    candidates = {k: v for k, v in sources.items() if not k.startswith(train_only_prefixes)}
    if len(candidates) < 3:
        train, val = split_sources(sources, val_fraction, seed, train_only_prefixes)
        return train, val, set()
    names = sorted(candidates, key=lambda s: hashlib.md5(f"{seed}:{s}".encode()).hexdigest())
    total = sum(candidates.values())
    targets = (1.0 - val_fraction - test_fraction, val_fraction, test_fraction)

    def cost(groups: tuple[int, ...]) -> float | None:
        sums = [0, 0, 0]
        for name, g in zip(names, groups):
            sums[g] += candidates[name]
        if not all(any(g == k for g in groups) for k in range(3)):
            return None                                             # train, val y test deben tener alguna fuente
        return sum(abs(s / total - t) for s, t in zip(sums, targets))

    if len(names) <= 10:
        best, best_cost = None, float("inf")
        for groups in itertools.product(range(3), repeat=len(names)):
            c = cost(groups)
            if c is not None and c < best_cost - 1e-12:
                best, best_cost = groups, c
    else:                                                           # muchas fuentes: reparto voraz
        groups, sums = [], [0, 0, 0]
        for name in names:
            g = 1 if sums[1] < val_fraction * total else 2 if sums[2] < test_fraction * total else 0
            groups.append(g)
            sums[g] += candidates[name]
        if 0 not in groups:
            groups[-1] = 0
        best = tuple(groups)
    val = {n for n, g in zip(names, best) if g == 1}
    test = {n for n, g in zip(names, best) if g == 2}
    return set(sources) - val - test, val, test


def clear_dir(path: Path, attempts: int = 6) -> None:
    """
    Borra una carpeta de exportación anterior. En Windows, OneDrive o el Explorador pueden bloquearla un instante
    ("Acceso denegado"): se reintenta y, si sigue bloqueada, se avisa con un mensaje claro en vez de dejar el
    dataset a medias sin explicación.
    """
    import os
    import stat
    import time

    def unlock(func, p, _exc):                                  # archivos de solo lectura (sincronización)
        os.chmod(p, stat.S_IWRITE)
        func(p)

    last: OSError | None = None
    for i in range(attempts):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=unlock)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.5 * (i + 1))
    raise OSError(f"No se pudo borrar la exportación anterior ({path}): {last}. Cierre el Explorador de archivos "
                  "en esa carpeta o pause la sincronización de OneDrive y vuelva a exportar.")


def export_dataset(project: WheelProject, cfg, out_dir: str | Path | None = None, val_fraction: float = 0.15,
                   seed: int = 0, make_zip: bool = True, test_fraction: float = 0.15) -> ExportResult:
    """Exporta SOLO las imágenes revisadas: imágenes + etiquetas YOLO-pose + dataset.yaml (+ .zip para Drive)."""
    out_dir = Path(out_dir) if out_dir else project.folder / EXPORT_DIR
    reviewed = [e for e in project.entries if e.reviewed]
    if not reviewed:
        raise ValueError("No hay imágenes revisadas: revise (Enter) al menos una antes de exportar.")
    counts: dict[str, int] = {}
    for e in reviewed:
        counts[e.source] = counts.get(e.source, 0) + 1
    train_src, val_src, test_src = split_sources3(counts, val_fraction, test_fraction, seed)
    warnings: list[str] = []
    if not val_src:
        val_src = set(train_src)
        warnings.append("Solo hay UNA fuente (un video o una sesión de fotos): no se puede separar validación "
                        "independiente. val = train; las métricas serán optimistas. Añada otra fuente.")
    elif not test_src:
        warnings.append("Hay menos de 3 fuentes de la cámara (videos o sesiones de fotos): no se puede separar un "
                        "conjunto de prueba (test) independiente. Añada más fuentes para repartir 70/15/15.")

    clear_dir(out_dir)
    split_kinds = ("train", "val", "test") if test_src else ("train", "val")
    for kind in split_kinds:
        (out_dir / "images" / kind).mkdir(parents=True)
        (out_dir / "labels" / kind).mkdir(parents=True)

    n_train = n_val = n_test = n_motos = 0
    used: set[str] = set()
    for e in reviewed:
        img = load_entry_image(project, e, cfg)
        h, w = img.shape[:2]
        lines = [ln for m in e.motos if (ln := yolo_pose_line(m, w, h))]
        n_motos += len(lines)
        stem = re.sub(r"[^\w.\-]+", "_", Path(e.path).stem)
        while stem in used:                                         # nombres únicos entre carpetas
            stem += "_"
        used.add(stem)
        kinds = [k for k, srcs in (("train", train_src), ("val", val_src), ("test", test_src)) if e.source in srcs]
        for kind in kinds:
            if not write_image(out_dir / "images" / kind / f"{stem}.jpg", img):
                raise OSError(f"No se pudo escribir la imagen {stem}.jpg")
            (out_dir / "labels" / kind / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                   encoding="utf-8")
        n_train += "train" in kinds
        n_val += "val" in kinds
        n_test += "test" in kinds

    # sin "path": Ultralytics toma la carpeta del yaml, así el dataset funciona en cualquier PC (o en Colab)
    (out_dir / "dataset.yaml").write_text(
        "# Dataset de ruedas de motos (YOLO-pose): 2 puntos clave = rueda trasera, rueda delantera\n"
        "train: images/train\nval: images/val\n" + ("test: images/test\n" if test_src else "") +
        "kpt_shape: [2, 3]     # 2 puntos, cada uno x y visibilidad\n"
        "flip_idx: [0, 1]      # al voltear la imagen las ruedas NO se intercambian\n"
        "names:\n  0: moto\n", encoding="utf-8")

    zip_path = None
    if make_zip:
        zip_path = out_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, Path(out_dir.name) / f.relative_to(out_dir))
    return ExportResult(out_dir, zip_path, n_train, n_val, n_motos, sorted(train_src), sorted(val_src), warnings,
                        n_test, sorted(test_src))

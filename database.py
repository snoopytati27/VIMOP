"""
database.py — Persistencia de resultados con ESCRITURA ASÍNCRONA.

results/
├── detections.csv       ID, Fecha, Hora, Placa, Puesto,
│                        Orientacion, Ubicacion, Centrado, Invasion, Score, Estado,
│                        Origen (camara | video:<archivo> | foto:<archivo>)
├── lote_<fecha>.csv     resumen de EXPORTAR LOTE (modo foto)
├── parking_events.csv   entradas, cambios de puesto, cambios de estado, salidas, alertas
├── motos.json           ficha de cada moto por ID, agrupada por sesión
├── vimop.db             SQLite para consultar motos anteriores (placa, ID o fecha)
├── screenshots/         capturas y fotos analizadas (<archivo>_vimop.jpg)
├── videos/

El hilo de video NUNCA escribe en disco directamente: encola la operación en una
queue.Queue y un hilo escritor la ejecuta. Así un disco lento (u OneDrive
sincronizando) no baja los FPS. El hilo escritor es el único que usa la conexión
SQLite. Todo archivo se abre con encoding="utf-8" dentro de un `with` (se cierra siempre).
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.image_io import write_image
from core.registry import ParkingEvent

log = logging.getLogger("vimop.database")

DETECTION_FIELDS = ["ID", "Fecha", "Hora", "Placa", "Puesto",
                    "Orientacion", "Ubicacion", "Centrado", "Invasion", "Score", "Estado", "Origen"]
BATCH_FIELDS = ["Archivo", "Origen", "Calibracion", "ID", "Placa", "Puesto",
                "Orientacion", "Ubicacion", "Centrado", "Invasion", "Score", "Estado", "Fecha", "Hora"]
EVENT_FIELDS = ["Sesion", "Fecha", "Hora", "ID", "Evento", "Puesto", "Score", "Estado", "Placa", "Detalle"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS motos (
    session TEXT, moto_id INTEGER, first_seen TEXT, last_seen TEXT, plate TEXT, space TEXT, state TEXT,
    orientation REAL, location REAL, centering REAL, invasion REAL, score REAL, classification TEXT,
    PRIMARY KEY (session, moto_id));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, timestamp TEXT, moto_id INTEGER,
    event_type TEXT, space TEXT, score REAL, classification TEXT, plate TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, timestamp TEXT, moto_id INTEGER, plate TEXT,
    space TEXT, state TEXT, orientation REAL, location REAL,
    centering REAL, invasion REAL, score REAL, classification TEXT);
CREATE INDEX IF NOT EXISTS idx_events_moto ON events(session, moto_id);
CREATE INDEX IF NOT EXISTS idx_motos_plate ON motos(plate);
"""


def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    """Si el CSV existe con otras columnas (versión anterior), se archiva y se empieza uno nuevo."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != fields:
        backup = path.with_name(f"{path.stem}_anterior_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        path.replace(backup)
        log.info("%s tenía otro formato; se archivó como %s", path.name, backup.name)


class ResultsStore:
    def __init__(self, results_dir: Path, use_sqlite: bool = True, session_id: str | None = None):
        self.dir = Path(results_dir)
        self.session = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.screens_dir, self.videos_dir = self.dir / "screenshots", self.dir / "videos"
        for d in (self.dir, self.screens_dir, self.videos_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.detections_csv = self.dir / "detections.csv"
        self.events_csv = self.dir / "parking_events.csv"
        self.motos_json = self.dir / "motos.json"
        self.db_path = self.dir / "vimop.db"
        self.use_sqlite = use_sqlite
        self._queue: queue.Queue = queue.Queue()      # COLA de escrituras pendientes
        self._writer: cv2.VideoWriter | None = None
        self.video_path: Path | None = None
        _ensure_csv_header(self.detections_csv, DETECTION_FIELDS)
        _ensure_csv_header(self.events_csv, EVENT_FIELDS)
        self._thread = threading.Thread(target=self._run, name="vimop-writer", daemon=True)
        self._thread.start()

    # ---- API (no bloqueante) ------------------------------------------------
    def log_detection(self, moto: dict, when: datetime, origin: str = "camara") -> None:
        self._queue.put(("detection", dict(moto), (when, origin)))

    def log_event(self, ev: ParkingEvent) -> None:
        self._queue.put(("event", ev, None))

    def save_motos(self, motos: list[dict]) -> None:
        self._queue.put(("motos", [dict(m) for m in motos], None))

    def save_annotated(self, image_bgr: np.ndarray, source_name: str) -> Path:
        """Foto analizada: results/screenshots/<archivo>_vimop.jpg (escritura síncrona para poder avisar)."""
        path = self.screens_dir / f"{Path(source_name).stem}_vimop.jpg"
        if not write_image(path, image_bgr):
            raise OSError(f"No se pudo guardar {path}")
        return path

    def save_screenshot(self, image_bgr: np.ndarray) -> Path:
        path = self.screens_dir / f"vimop_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]
        path = path.with_suffix(".jpg")
        self._queue.put(("image", image_bgr.copy(), path))
        return path

    def flush(self, timeout: float = 10.0) -> None:
        done = threading.Event()
        self._queue.put(("sync", done, None))
        done.wait(timeout)

    def close(self) -> None:
        self.stop_video()
        self._queue.put(None)
        self._thread.join(timeout=15.0)

    # ---- video (se escribe en el hilo de procesamiento: VideoWriter no es costoso) ----
    def start_video(self, width: int, height: int, fps: float) -> Path | None:
        self.stop_video()
        path = self.videos_dir / f"vimop_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, min(fps, 30.0)),
                                 (width, height))
        if not writer.isOpened():
            log.error("No se pudo crear el video %s", path)
            return None
        self._writer, self.video_path = writer, path
        return path

    def write_video(self, image_bgr: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(image_bgr)

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def stop_video(self) -> Path | None:
        path = self.video_path
        if self._writer is not None:
            self._writer.release()
        self._writer, self.video_path = None, None
        return path

    # ---- hilo escritor ------------------------------------------------------
    def _run(self) -> None:
        db = None
        if self.use_sqlite:
            try:
                db = sqlite3.connect(str(self.db_path))
                db.executescript(_SCHEMA)
                cols = {row[1] for row in db.execute("PRAGMA table_info(detections)")}
                if "origin" not in cols:                   # base de una versión anterior
                    db.execute("ALTER TABLE detections ADD COLUMN origin TEXT")
                db.execute("PRAGMA journal_mode=WAL")
                db.commit()
            except sqlite3.Error as exc:
                log.error("SQLite no disponible (%s): solo se guardarán CSV/JSON.", exc)
                db = None
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, payload, extra = item
            try:
                if kind == "detection":
                    self._write_detection(db, payload, extra)
                elif kind == "event":
                    self._write_event(db, payload)
                elif kind == "motos":
                    self._write_motos(db, payload)
                elif kind == "image":
                    write_image(extra, payload)             # admite rutas con tildes/espacios
                elif kind == "sync":
                    payload.set()
            except OSError as exc:
                log.error("No se pudo escribir en results/ (¿archivo abierto en Excel?): %s", exc)
            except sqlite3.Error as exc:
                log.error("Error SQLite: %s", exc)
            except Exception:
                log.exception("Error inesperado guardando resultados")
        if db is not None:
            db.close()

    @staticmethod
    def _append_csv(path: Path, fields: list[str], row: dict) -> None:
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def _write_detection(self, db, m: dict, extra) -> None:
        when, origin = extra
        self._append_csv(self.detections_csv, DETECTION_FIELDS, {
            "ID": m["id"], "Fecha": f"{when:%Y-%m-%d}", "Hora": f"{when:%H:%M:%S}", "Placa": m["placa"],
            "Puesto": m["puesto"],
            "Orientacion": m["orientacion"], "Ubicacion": m["ubicacion"], "Centrado": m["centrado"],
            "Invasion": m["invasion"], "Score": m["puntaje"],
            "Estado": m["clasificacion"] or m["estado_movimiento"], "Origen": origin})
        if db is not None:
            db.execute("INSERT INTO detections (session, timestamp, moto_id, plate, space, "
                       "state, orientation, location, centering, invasion, score, classification, origin) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (self.session, when.isoformat(timespec="seconds"), m["id"], m["placa"], m["puesto"],
                        m["estado_movimiento"], m["orientacion"], m["ubicacion"], m["centrado"], m["invasion"],
                        m["puntaje"], m["clasificacion"], origin))
            db.commit()

    def _write_event(self, db, ev: ParkingEvent) -> None:
        d = ev.to_dict()
        self._append_csv(self.events_csv, EVENT_FIELDS, {
            "Sesion": self.session, "Fecha": d["fecha"], "Hora": d["hora"], "ID": d["id"], "Evento": d["evento"],
            "Puesto": d["puesto"], "Score": d["score"], "Estado": d["estado"], "Placa": d["placa"],
            "Detalle": d["detalle"]})
        if db is not None:
            db.execute("INSERT INTO events (session, timestamp, moto_id, event_type, space, score, classification,"
                       " plate, detail) VALUES (?,?,?,?,?,?,?,?,?)",
                       (self.session, ev.timestamp.isoformat(timespec="seconds"), ev.moto_id, ev.event_type,
                        ev.space, ev.score, ev.classification, ev.plate, ev.detail))
            db.commit()

    def _write_motos(self, db, motos: list[dict]) -> None:
        data = {}
        if self.motos_json.exists():
            try:
                with open(self.motos_json, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                data = {}
        data[self.session] = {str(m["id"]): m for m in motos}
        tmp = self.motos_json.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        tmp.replace(self.motos_json)
        if db is not None:
            # columnas con nombre: también sirve con una base antigua que aún tenga marca/modelo/color
            db.executemany("INSERT OR REPLACE INTO motos (session, moto_id, first_seen, last_seen, plate, space, "
                           "state, orientation, location, centering, invasion, score, classification) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [(self.session, m["id"], m["primera_vez"], m["ultima_vez"], m["placa"], m["puesto"],
                             m["estado_movimiento"], m["orientacion"], m["ubicacion"], m["centrado"], m["invasion"],
                             m["puntaje"], m["clasificacion"]) for m in motos])
            db.commit()


# --------------------------------------------------------------------------
# Exportación en lote (modo foto)
# --------------------------------------------------------------------------
def batch_row(file_name: str, origin: str, calibration: str, moto: dict | None, when: datetime) -> dict:
    """Fila del CSV resumen del lote. moto = None -> la foto no tenía motos."""
    row = {"Archivo": file_name, "Origen": origin, "Calibracion": calibration,
           "Fecha": f"{when:%Y-%m-%d}", "Hora": f"{when:%H:%M:%S}"}
    if moto is None:
        row["Estado"] = "No se detectaron motocicletas"
        return row
    row.update({"ID": moto["id"], "Placa": moto["placa"], "Puesto": moto["puesto"],
                "Orientacion": moto["orientacion"],
                "Ubicacion": moto["ubicacion"], "Centrado": moto["centrado"], "Invasion": moto["invasion"],
                "Score": moto["puntaje"], "Estado": moto["clasificacion"] or moto["estado_movimiento"]})
    return row


def write_batch_csv(path: Path, rows: list[dict]) -> Path:
    """Un único CSV con todas las motos de todas las fotos del lote."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BATCH_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# Consultas (conexión de solo lectura, desde la interfaz)
# --------------------------------------------------------------------------
def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    if not Path(db_path).exists():
        return None
    try:
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def query_motos(db_path: Path, text: str = "", date: str = "", limit: int = 1000) -> list[dict]:
    """Busca por placa, ID, puesto o estado (text) y/o por fecha 'AAAA-MM-DD' (date)."""
    con = _connect_ro(db_path)
    if con is None:
        return []
    try:
        where, params = [], []
        text = (text or "").strip()
        if text:
            like = f"%{text}%"
            where.append("(plate LIKE ? OR CAST(moto_id AS TEXT) = ? OR space LIKE ? "
                         "OR classification LIKE ? OR session LIKE ?)")
            params += [like, text, like, like, like]
        if date:
            where.append("(first_seen LIKE ? OR last_seen LIKE ?)")
            params += [f"{date}%", f"{date}%"]
        sql = "SELECT * FROM motos" + (" WHERE " + " AND ".join(where) if where else "")
        rows = con.execute(sql + " ORDER BY last_seen DESC LIMIT ?", (*params, limit)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        log.error("Error consultando motos: %s", exc)
        return []
    finally:
        con.close()


def query_events(db_path: Path, session: str, moto_id: int) -> list[dict]:
    con = _connect_ro(db_path)
    if con is None:
        return []
    try:
        return [dict(r) for r in con.execute(
            "SELECT * FROM events WHERE session = ? AND moto_id = ? ORDER BY id", (session, moto_id)).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        con.close()

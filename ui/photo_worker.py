"""
ui/photo_worker.py — Hilo del MODO FOTO (la interfaz nunca se congela).

Es un QThread de larga vida: carga YOLO, OCR y clasificador UNA sola vez y luego atiende
órdenes que llegan por una cola (queue.Queue):

    ("analyze", índice, ruta, layout, política)  -> analiza una foto y emite `result`
    ("save",    índice)                          -> guarda <archivo>_vimop.jpg + filas en detections.csv
    ("export",  rutas, layout, csv)              -> lote completo con `progress` y `batch_done`
    ("stop",)

Si la foto no coincide con la calibración y la política es "ask", emite `mismatch` para que
la ventana pregunte al usuario (recalibrar sobre la foto o continuar sin puestos).
"""
from __future__ import annotations

import logging
import queue
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from config import Config
from core.detector import ModelNotFoundError
from core.photo import analyze_photo, export_batch
from core.pipeline import VimopPipeline
from database import ResultsStore
from ui.renderer import FrameRenderer, qimage_to_bgr

log = logging.getLogger("vimop.ui.photo")


class PhotoWorker(QThread):
    ready = Signal(str)                         # descripción del dispositivo
    result = Signal(int, QImage, object)        # índice, imagen anotada, payload
    mismatch = Signal(int, str, object)         # índice, mensaje, frame (para recalibrar)
    failed = Signal(int, str)                   # índice, mensaje de error de esa foto
    saved = Signal(str)                         # ruta de la foto anotada guardada
    progress = Signal(int, int, str)            # i, total, archivo
    batch_done = Signal(str, int, int, object)  # csv, fotos, motos, errores
    status = Signal(str)
    fatal = Signal(str)                         # no se pudo iniciar (modelo ausente)

    def __init__(self, cfg: Config, store: ResultsStore, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.store = store
        self.jobs: queue.Queue = queue.Queue()
        self._last: dict[int, tuple] = {}       # índice -> (FrameResult, QImage, ruta)
        self._cancel = False
        self.renderer = FrameRenderer(blink=False, show_bbox=bool(cfg.SHOW_BBOX))

    # ---- API usada por la ventana ---------------------------------------
    def analyze(self, index: int, path: str, layout, policy: str = "ask") -> None:
        self.jobs.put(("analyze", index, path, layout, policy))

    def save(self, index: int) -> None:
        self.jobs.put(("save", index))

    def export(self, paths: list[str], layout, csv_path: Path) -> None:
        self._cancel = False
        self.jobs.put(("export", paths, layout, csv_path))

    def cancel_export(self) -> None:
        self._cancel = True

    def stop(self) -> None:
        self._cancel = True
        self.jobs.put(("stop",))

    # ---- hilo ---------------------------------------------------------------
    def run(self) -> None:
        try:
            pipeline = VimopPipeline(self.cfg, None, self.store, status_cb=self.status.emit)
        except ModelNotFoundError as exc:
            self.fatal.emit(str(exc))
            return
        except Exception as exc:
            log.exception("No se pudo iniciar el modo foto")
            self.fatal.emit(f"No se pudo iniciar el modo foto: {exc}")
            return
        self.ready.emit(pipeline.device_desc)
        try:
            while True:
                job = self.jobs.get()
                kind = job[0]
                if kind == "stop":
                    break
                try:
                    if kind == "analyze":
                        self._analyze(pipeline, *job[1:])
                    elif kind == "save":
                        self._save(job[1])
                    elif kind == "export":
                        self._export(pipeline, *job[1:])
                except Exception as exc:           # ninguna foto debe tumbar el hilo
                    log.exception("Error en el modo foto")
                    self.failed.emit(job[1] if kind in ("analyze", "save") else -1, f"Error inesperado: {exc}")
        finally:
            pipeline.shutdown()

    def _analyze(self, pipeline, index: int, path: str, layout, policy: str) -> None:
        self.status.emit(f"Analizando {Path(path).name}...")
        out = analyze_photo(pipeline, path, layout, policy)
        if out.error:
            self.failed.emit(index, out.error)
            return
        if out.needs_decision:
            self.mismatch.emit(index, out.compat.message, out.frame)
            return
        res = out.result
        qimg = self.renderer.render(res)
        self._last[index] = (res, qimg, path)
        self.result.emit(index, qimg, {
            "motos": [v.info for v in res.motos], "summary": res.summary, "alerts": res.alerts,
            "notes": res.notes, "frame": res.frame, "calibrated": res.calibrated, "origin": res.origin,
            "calibration": out.calibration_label, "path": path, "device": pipeline.device_desc,
            "detector": pipeline.detector.description, "plate_status": pipeline.ident.plate_status,
            "wheels": pipeline.wheels.description, "process_ms": res.process_ms})

    def _save(self, index: int) -> None:
        if index not in self._last:
            self.failed.emit(index, "Primero debe analizarse la foto.")
            return
        res, qimg, path = self._last[index]
        out_path = self.store.save_annotated(qimage_to_bgr(qimg), Path(path).name)
        for v in res.motos:                                    # una fila por moto, con Origen
            self.store.log_detection(v.info, res.wall_time, res.origin)
        self.store.flush()
        self.saved.emit(str(out_path))

    def _export(self, pipeline, paths: list[str], layout, csv_path: Path) -> None:
        csv_path, done, motos, errors = export_batch(
            pipeline, paths, layout, csv_path,
            progress=lambda i, n, name: self.progress.emit(i, n, name),
            cancelled=lambda: self._cancel)
        self.batch_done.emit(str(csv_path), done, motos, errors)

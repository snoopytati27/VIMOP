"""
identification/worker.py — Hilo trabajador para la lectura de placas (OCR, pesada).

Patrón productor–consumidor con dos COLAS (queue.Queue, seguras entre hilos):
    pipeline --(jobs, acotada)-->  IdentificationWorker  --(results)--> pipeline
Si el OCR va atrasado, la cola llena rechaza trabajos nuevos (se reintentan N frames
después) en vez de acumular retraso. Los modelos se cargan dentro del hilo para no
congelar la interfaz al iniciar.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np

from identification.plate import PlateRead, PlateReader

log = logging.getLogger("vimop.ident")


@dataclass
class IdentJob:
    moto_id: int
    crop: np.ndarray
    want_plate: bool = True
    moto_box: tuple | None = None          # caja real de la moto dentro del recorte (el recorte trae margen de contexto)


@dataclass
class IdentResult:
    moto_id: int
    did_plate: bool = False
    plate: PlateRead | None = None


class IdentificationWorker(threading.Thread):
    def __init__(self, cfg, device: str, use_gpu: bool, max_jobs: int = 8):
        super().__init__(name="vimop-ident", daemon=True)
        self.cfg, self.device, self.use_gpu = cfg, device, use_gpu
        self.jobs: queue.Queue[IdentJob] = queue.Queue(maxsize=max_jobs)
        self.results: queue.Queue[IdentResult] = queue.Queue()
        self._stop_event = threading.Event()
        self.ready = threading.Event()
        self.plate_reader: PlateReader | None = None
        self.plate_status = "Cargando OCR..."
        self.plate_warning = ""

    @property
    def plate_available(self) -> bool:
        return self.plate_reader is not None and self.plate_reader.available

    def submit(self, job: IdentJob) -> bool:
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def drain(self) -> list[IdentResult]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self.plate_reader = PlateReader(self.cfg, self.device, self.use_gpu)
            self.plate_status = self.plate_reader.description
            self.plate_warning = self.plate_reader.warning
        except Exception as exc:
            log.exception("Fallo iniciando el lector de placas")
            self.plate_status = f"No disponible ({exc})"
        self.ready.set()
        log.info("Identificación lista. Placas: %s", self.plate_status)
        while not self._stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            res = IdentResult(job.moto_id)
            try:
                if job.want_plate and self.plate_available:
                    res.plate, res.did_plate = self.plate_reader.read(job.crop, job.moto_box), True
            except Exception:
                log.exception("Error identificando MOTO #%d", job.moto_id)
            self.results.put(res)

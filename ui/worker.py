"""
ui/worker.py — Hilo de procesamiento (QThread). La interfaz nunca ejecuta YOLO ni OCR.

    VideoSource (hilo de captura) --Queue--> ProcessingThread --señal Qt--> MainWindow
                                                  │
                                                  └--Queue--> ResultsStore (hilo escritor)

Las señales Qt entre hilos son "queued connections": Qt las pone en la cola de
eventos del hilo de la interfaz, que las procesa cuando está libre.
Las órdenes de la interfaz (capturar, grabar, guardar, cambiar calibración)
llegan por otra queue.Queue que el hilo revisa en cada frame.
"""
from __future__ import annotations

import logging
import queue
import time

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from analysis.parking_spaces import ParkingLayout
from config import Config
from core.camera import CameraError, VideoSource
from core.detector import ModelNotFoundError
from core.pipeline import VimopPipeline
from database import ResultsStore
from ui.renderer import FrameRenderer, qimage_to_bgr

log = logging.getLogger("vimop.ui.worker")


class ProcessingThread(QThread):
    frame_ready = Signal(QImage, object)
    status = Signal(str)
    camera_status = Signal(str)
    error = Signal(str)
    stopped = Signal(str)

    def __init__(self, cfg: Config, source_kind: str, source_value, layout: ParkingLayout | None,
                 store: ResultsStore, parent=None):
        # Con parent, Qt es dueño del hilo y no se destruye mientras reparte sus señales.
        super().__init__(parent)
        self.cfg = cfg
        self.source_kind, self.source_value = source_kind, source_value
        self.layout = layout
        self.store = store
        self.selected_id: int | None = None
        self.commands: queue.Queue = queue.Queue()
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def send(self, command: str, arg=None) -> None:
        self.commands.put((command, arg))

    def run(self) -> None:
        reason, pipeline, source = "Detenido por el usuario", None, None
        try:
            try:
                pipeline = VimopPipeline(self.cfg, self.layout, self.store, status_cb=self.status.emit)
            except ModelNotFoundError as exc:
                self.error.emit(str(exc))
                reason = "Error de modelo"
                return
            source = VideoSource(self.source_kind, self.source_value, self.cfg)
            self.status.emit(f"Conectando a {source.describe()}...")
            self.camera_status.emit("CONECTANDO")
            try:
                source.start()
            except CameraError as exc:
                self.error.emit(str(exc))
                reason = "No se pudo abrir la fuente"
                return
            self.camera_status.emit(source.status)
            self.status.emit(f"Procesando {source.describe()} en {pipeline.device_desc}")
            renderer = FrameRenderer(blink=bool(self.cfg.ALERT_BLINK), show_bbox=bool(self.cfg.SHOW_BBOX))
            fps, last_t, last_status, recording = 0.0, time.perf_counter(), source.status, False
            warned_plate = False

            while not self._stop_requested:
                screenshot = False
                while True:
                    try:
                        cmd, arg = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "layout":
                        pipeline.set_layout(arg)
                    elif cmd == "screenshot":
                        screenshot = True
                    elif cmd == "record":
                        recording = bool(arg)
                        if not recording:
                            path = self.store.stop_video()
                            if path:
                                self.status.emit(f"Video guardado: {path}")
                    elif cmd == "save":
                        pipeline.save_all()
                        self.store.flush()
                        self.status.emit(f"Resultados guardados en {self.store.dir}")

                packet = source.read(timeout=1.0)
                if source.status != last_status:
                    last_status = source.status
                    self.camera_status.emit(source.status)
                    if source.message:
                        self.status.emit(source.message.splitlines()[0])
                if packet is None:
                    if source.finished:
                        reason = "Video terminado" if not source.is_live else "Cámara no disponible"
                        break
                    continue
                if not warned_plate and pipeline.ident.ready.is_set():
                    warned_plate = True
                    if pipeline.ident.plate_warning:
                        self.status.emit(pipeline.ident.plate_warning)

                try:
                    result = pipeline.process(packet)
                except Exception as exc:                 # un frame defectuoso no tumba el sistema
                    log.exception("Error procesando frame")
                    self.status.emit(f"Error procesando frame: {exc}")
                    continue
                now = time.perf_counter()
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                last_t = now
                qimg = renderer.render(result, self.selected_id, fps)
                if screenshot or recording:
                    bgr = qimage_to_bgr(qimg)
                    if screenshot:
                        self.status.emit(f"Captura guardada: {self.store.save_screenshot(bgr)}")
                    if recording:
                        if not self.store.recording:
                            path = self.store.start_video(bgr.shape[1], bgr.shape[0], source.fps)
                            self.status.emit(f"Grabando en {path}" if path else "No se pudo iniciar la grabación")
                            recording = path is not None
                        self.store.write_video(bgr)
                self.frame_ready.emit(qimg, {
                    "motos": [v.info for v in result.motos], "summary": result.summary, "alerts": result.alerts,
                    "new_alerts": result.new_alerts, "events": [e.to_dict() for e in result.events],
                    "fps": fps, "inference_ms": result.inference_ms, "process_ms": result.process_ms,
                    "frame": result.frame, "calibrated": result.calibrated,
                    "device": pipeline.device_desc, "detector": pipeline.detector.description,
                    "tracker": str(self.cfg.TRACKER), "plate_status": pipeline.ident.plate_status,
                    "wheels": pipeline.wheels.description})
        except Exception as exc:
            log.exception("Error inesperado en el hilo de procesamiento")
            self.error.emit(f"Error inesperado: {exc}")
            reason = "Error"
        finally:
            if source is not None:
                source.stop()
            if pipeline is not None:
                try:
                    pipeline.shutdown()
                except Exception:
                    log.exception("Error cerrando el pipeline")
            self.store.stop_video()
            self.camera_status.emit(source.status if source is not None else "DESCONECTADA")
            self.stopped.emit(reason)

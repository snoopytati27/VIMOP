"""VIMOP cada esta lo que es 
    1- menú: MONITOREAR (Iniciar cámara · Examinar archivo) y PREPARAR EL PARQUEADERO (Calibrar · Cargar calibración · Configuración) 
    2- video [RUTA] abrir directamente el video de prueba
    python main.py --camera                iniciar directamente la cámara de config.json
    python main.py --calibrate             abrir la calibración
    python main.py --headless --video      pipeline SIN interfaz (imprime resultados en consola)
        opciones: --max-frames N  --save-video  --calibration RUTA.json  --results CARPETA
    python main.py --headless --photos F1.jpg F2.png CARPETA/   modo FOTO por lotes -> results/lote_<fecha>.csv
        opción: --save-images  (guarda cada foto anotada como <archivo>_vimop.jpg)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

try:                                             
    import cv2  # noqa: F401
except ImportError:
    sys.exit(f"ERROR: este Python no tiene OpenCV (cv2):\n  {sys.executable}\n" "Use el Python global donde está instalado el proyecto (en VS Code: Ctrl+Shift+P > "
             "'Python: Select Interpreter' > Python314), o instale: pip install -r requirements.txt")

from config import RESULTS_DIR, load_config
from utils import setup_logging

def parse_args():
    ap = argparse.ArgumentParser(description="VIMOP - monitoreo del parqueo de motos")
    ap.add_argument("--video", nargs="?", const="", help="usar un video (sin ruta = VIDEO_PATH)")
    ap.add_argument("--camera", action="store_true", help="usar la cámara de config.json")
    ap.add_argument("--calibrate", action="store_true", help="abrir la calibración")
    ap.add_argument("--headless", action="store_true", help="procesar sin interfaz gráfica")
    ap.add_argument("--max-frames", type=int, default=0, help="(headless) detener tras N frames")
    ap.add_argument("--save-video", action="store_true", help="(headless) guardar el video anotado")
    ap.add_argument("--calibration", help="archivo de calibración a usar (por defecto PARKING_SPACES_FILE)")
    ap.add_argument("--results", help="carpeta de resultados (por defecto results/)")
    ap.add_argument("--photos", nargs="+", help="(headless) fotos o carpetas a analizar en modo foto")
    ap.add_argument("--save-images", action="store_true", help="(headless --photos) guardar las fotos anotadas")
    return ap.parse_args()


def run_headless(cfg, args) -> int:
    
    from analysis.parking_spaces import load_layout
    from core.camera import CameraError, VideoSource
    from core.detector import ModelNotFoundError
    from core.pipeline import VimopPipeline
    from database import ResultsStore

    log = logging.getLogger("vimop.headless")
    renderer = app = None
    if args.save_video:
        app = _offscreen_qt()
        from ui.renderer import FrameRenderer, qimage_to_bgr
        renderer = FrameRenderer(show_bbox=bool(cfg.SHOW_BBOX))

    cal_path = Path(args.calibration) if args.calibration else cfg.resolve_path(cfg.PARKING_SPACES_FILE)
    layout, msg = load_layout(cal_path)
    log.info(msg)
    results_dir = Path(args.results) if args.results else RESULTS_DIR
    store = ResultsStore(results_dir, bool(cfg.USE_SQLITE))
    try:
        pipeline = VimopPipeline(cfg, layout, store)
    except ModelNotFoundError as exc:
        log.error("%s", exc)
        store.close()
        return 2
    source = VideoSource("video", cfg.VIDEO_PATH, cfg) if args.video is not None else VideoSource.from_config(cfg)
    try:
        source.start()
    except CameraError as exc:
        log.error("%s", exc)
        pipeline.shutdown()
        store.close()
        return 2
    log.info("Procesando %s en %s", source.describe(), pipeline.device_desc)
    n, t0, last_print, res = 0, time.time(), 0.0, None
    try:
        while True:
            packet = source.read(timeout=2.0)
            if packet is None:
                if source.finished:
                    log.info("%s", source.message or "Fuente terminada")
                    break
                continue
            res = pipeline.process(packet)
            n += 1
            if renderer is not None:
                bgr = qimage_to_bgr(renderer.render(res, fps=n / max(time.time() - t0, 1e-6)))
                if not store.recording:
                    store.start_video(bgr.shape[1], bgr.shape[0], source.fps)
                store.write_video(bgr)
            if time.time() - last_print >= 2.0:
                last_print = time.time()
                s = res.summary
                log.info("frame %d | %.0f ms | motos %d · estacionadas %d · moviéndose %d | %s",
                         packet.index, res.process_ms, s["detectadas"], s["estacionadas"], s["moviendose"],
                         " ".join(f"#{v.info['id']}:{v.info['puesto'] or '-'}:"
                                  f"{v.info['puntaje'] if v.info['puntaje'] is not None else v.info['estado_movimiento']}"
                                  for v in res.motos))
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario")
    finally:
        source.stop()
        pipeline.shutdown()
        video = store.stop_video()
        store.close()
    elapsed = max(time.time() - t0, 1e-6)
    log.info("Terminado: %d frames en %.1f s (%.1f FPS). IDs asignados: %d", n, elapsed, n / elapsed,
             pipeline.identities.next_id - 1)
    if res is not None:
        log.info("Resumen final: %s", res.summary)
    log.info("Resultados en %s%s", results_dir, f" · video: {video}" if video else "")
    if app is not None:
        app.quit()
    return 0


def _offscreen_qt():
   
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if fonts.exists():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.instance() or QGuiApplication(sys.argv[:1])


def run_photos(cfg, args) -> int:
    
    from datetime import datetime
    from analysis.parking_spaces import load_layout
    from core.detector import ModelNotFoundError
    from core.image_io import detect_file_kind, list_images_in_folder
    from core.photo import export_batch
    from core.pipeline import VimopPipeline
    from database import ResultsStore

    log = logging.getLogger("vimop.photos")
    paths = []
    for item in args.photos:
        p = Path(item)
        if p.is_dir():
            paths += list_images_in_folder(p)
        elif detect_file_kind(p) == "image":
            paths.append(p)
        else:
            log.warning("Se omite %s: no es una imagen soportada o no existe.", p)
    if not paths:
        log.error("No hay fotos válidas para analizar.")
        return 2
    renderer = None
    if args.save_images:
        _offscreen_qt()
        from ui.renderer import FrameRenderer, qimage_to_bgr
        renderer = FrameRenderer(blink=False, show_bbox=bool(cfg.SHOW_BBOX))
    cal_path = Path(args.calibration) if args.calibration else cfg.resolve_path(cfg.PARKING_SPACES_FILE)
    layout, msg = load_layout(cal_path)
    log.info(msg)
    results_dir = Path(args.results) if args.results else RESULTS_DIR
    store = ResultsStore(results_dir, bool(cfg.USE_SQLITE))
    try:
        pipeline = VimopPipeline(cfg, layout, store)
    except ModelNotFoundError as exc:
        log.error("%s", exc)
        store.close()
        return 2

    def on_outcome(out):
        if out.error:
            log.error("%s", out.error)
            return
        res = out.result
        log.info("%s | %s | %d motos%s", out.path.name, out.calibration_label, len(res.motos),
                 "".join(f"\n    MOTO #{v.info['id']}: puesto {v.info['puesto'] or '-'} · score "
                         f"{v.info['puntaje'] if v.info['puntaje'] is not None else '-'} · "
                         f"{v.info['clasificacion'] or v.info['estado_movimiento']} · placa {v.info['placa']}"
                         for v in res.motos) or "  (No se detectaron motocicletas)")
        for note in res.notes:
            log.info("    nota: %s", note)
        if renderer is not None:
            path = store.save_annotated(qimage_to_bgr(renderer.render(res)), out.path.name)
            for v in res.motos:
                store.log_detection(v.info, res.wall_time, res.origin)
            log.info("    guardada: %s", path)

    csv_path = results_dir / f"lote_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        csv_path, done, motos, errors = export_batch(
            pipeline, paths, layout, csv_path,
            progress=lambda i, n, name: log.info("[%d/%d] %s", i, n, name), on_outcome=on_outcome)
    finally:
        pipeline.shutdown()
        store.close()
    log.info("Lote terminado: %d/%d fotos, %d motos, %d errores. CSV: %s", done, len(paths), motos, len(errors),
             csv_path)
    return 0


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results) if args.results else RESULTS_DIR
    setup_logging(results_dir)
    log = logging.getLogger("vimop")
    cfg = load_config()
    if args.headless:
        return run_photos(cfg, args) if args.photos else run_headless(cfg, args)

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print("ERROR: PySide6 no está instalado. Ejecute: pip install -r requirements.txt")
        return 1
    from ui.assets import app_icon
    from ui.dialogs import StartMenu
    from ui.main_window import MainWindow
    from ui.theme import STYLE

    app = QApplication(sys.argv)
    app.setApplicationName("VIMOP")
    app.setWindowIcon(app_icon())                 # ventana, diálogos y barra de tareas
    app.setStyleSheet(STYLE)

    def excepthook(exc_type, exc, tb):          # ningún error inesperado cierra el programa
        log.error("Excepción no controlada:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        QMessageBox.critical(None, "VIMOP - error", f"{exc_type.__name__}: {exc}\n\nDetalle en results/vimop.log")
    sys.excepthook = excepthook

    if args.video:
        cfg.VIDEO_PATH = args.video
    window = MainWindow(cfg)                   # se construye oculta: calibrar/cargar la usan como fuente de datos
    paths = None
    if args.video is not None:
        choice = "video"
    elif args.camera:
        choice = "camera"
    elif args.calibrate:
        choice = "calibrate"
    else:
        from analysis.parking_spaces import load_layout

        def calibrate_from_menu(parent):
            window.use_configured_source()       # la fuente pudo cambiar en Configuración desde el menú
            return window.open_calibration(parent)

        actions = {"calibrate": calibrate_from_menu, "load": lambda parent: window.load_calibration(parent)}
        menu = StartMenu(cfg, lambda: load_layout(window.layout_path)[1], actions)
        if not menu.exec():
            return 0
        choice, paths = menu.choice, menu.chosen_paths
    window.show()
    window.run_choice(choice, paths)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

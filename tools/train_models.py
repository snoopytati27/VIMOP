"""
tools/train_models.py — Entrena el detector de placas (YOLO detección, 1 clase "placa").

    python tools/train_models.py plates --data dataset_placas/data.yaml --epochs 80 --no-install

data.yaml en formato Ultralytics (imágenes + etiquetas YOLO txt). Se puede etiquetar con Roboflow, CVAT o
Label Studio sobre recortes de motos. Sin --no-install, el resultado se copia a models/plate.pt y VIMOP lo usa
solo; conviene comprobar antes (con fotos reales de la cámara) que le gana al detector heurístico.
(El clasificador de marca/modelo y el color se quitaron del proyecto el 3-oct-2026.)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenar el detector de placas de VIMOP")
    ap.add_argument("task", choices=["plates"])
    ap.add_argument("--data", required=True, help="data.yaml del dataset de placas")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8, help="baje esto si la GPU se queda sin memoria (4 GB)")
    ap.add_argument("--device", default=None, help="'cpu', '0' (GPU)... por defecto automático")
    ap.add_argument("--no-install", action="store_true",
                    help="no copiar el resultado a models/: revíselo primero antes de que VIMOP lo use")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        print(f"ERROR: no existe {data}")
        return 1
    model, imgsz, dest = YOLO("yolo11n.pt"), args.imgsz or 640, ROOT / "models" / "plate.pt"
    kwargs = dict(data=str(data), epochs=args.epochs, imgsz=imgsz, batch=args.batch, project=str(ROOT / "runs"),
                  name=args.task, exist_ok=True)
    if args.device is not None:
        kwargs["device"] = args.device
    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print("El entrenamiento terminó pero no se encontró best.pt")
        return 1
    print(f"\nMejor modelo: {best}")
    if args.no_install:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print(f"Copiado a {dest}. VIMOP lo usará automáticamente al pulsar INICIAR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

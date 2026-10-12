"""
tools/train_wheels.py — Entrena el modelo de ruedas (YOLO-pose, 2 puntos clave) en ESTE equipo.

    python tools/train_wheels.py --data wheels_dataset/dataset.yaml
    python tools/train_wheels.py --data wheels_dataset.zip --model s --epochs 150

Por defecto: yolo11n-pose, imgsz=960, 100 épocas, dispositivo automático (GPU si hay CUDA), aumentos con
escala de grises (simula infrarrojo), brillo variable y aumentos tipo cámara fija (ojo de pez, baja resolución,
desenfoque, JPEG, ruido, noche; ver tools/wheel_training.py; --camera-aug 0 los apaga).
Al terminar copia best.pt a models/wheels_pose.pt,
que VIMOP usa solo (WHEEL_BACKEND = "auto"). Si la GPU tiene poca memoria (4 GB), baje --batch o --imgsz,
o entrene en Colab (colab/train_wheels_colab.ipynb).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = {"n": "yolo11n-pose", "s": "yolo11s-pose"}


def resolve_dataset(data: str, work_dir: Path) -> Path:
    """dataset.yaml, carpeta con dataset.yaml o .zip (se descomprime en work_dir). Devuelve el dataset.yaml."""
    p = Path(data)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    if p.suffix.lower() == ".zip":
        dest = work_dir / p.stem
        if dest.exists():
            shutil.rmtree(dest)
        with zipfile.ZipFile(p) as zf:
            zf.extractall(dest)
        found = sorted(dest.rglob("dataset.yaml"))
        if not found:
            raise FileNotFoundError(f"El zip no contiene dataset.yaml: {p}")
        return found[0]
    if p.is_dir():
        p = p / "dataset.yaml"
    if p.suffix.lower() not in (".yaml", ".yml") or not p.exists():
        raise FileNotFoundError(f"No se encontró dataset.yaml en {data}")
    return p


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO
    from tools.wheel_training import make_trainer

    data_yaml = resolve_dataset(args.data, ROOT / "runs" / "wheels_data")
    device = pick_device(args.device)
    base = MODELS[args.model]
    try:
        weights = ROOT / "models" / f"{base}.pt"          # se descarga aquí la primera vez (requiere internet)
        weights.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(f"{base}.yaml" if args.from_scratch else str(weights))
    except Exception as exc:                        # sin internet: no se pudieron bajar los pesos preentrenados
        print(f"AVISO: no se pudieron cargar los pesos preentrenados ({exc}). Se entrena DESDE CERO: hacen falta "
              "muchas más imágenes.")
        model = YOLO(f"{base}.yaml")
    kwargs = dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                  project=str(ROOT / "runs"), name=args.name, exist_ok=True, patience=args.patience,
                  workers=args.workers, plots=False,
                  hsv_v=0.5, hsv_s=0.5, degrees=5.0, scale=0.4, fliplr=0.5, mosaic=1.0,
                  trainer=make_trainer(args.gray_prob, camera_aug=args.camera_aug))
    print(f"Entrenando {base} en {device} · {args.epochs} épocas · imgsz {args.imgsz} · lote {args.batch} · "
          f"aumentos tipo cámara {args.camera_aug:g}")
    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("El entrenamiento terminó pero no se encontró best.pt")
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entrenar el modelo de ruedas de VIMOP (YOLO-pose)")
    ap.add_argument("--data", required=True, help="dataset.yaml, carpeta del dataset o .zip exportado")
    ap.add_argument("--model", choices=sorted(MODELS), default="n", help="n = nano (rápido), s = small")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="'cpu', '0'... por defecto automático")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gray-prob", type=float, default=0.25, help="fracción de imágenes convertidas a grises")
    ap.add_argument("--camera-aug", type=float, default=1.0,
                    help="fuerza de los aumentos tipo cámara (ojo de pez, baja resolución, desenfoque, JPEG, ruido, "
                         "noche): 1 = normal, 2 = el doble de frecuentes, 0 = apagados")
    ap.add_argument("--name", default="wheels")
    ap.add_argument("--from-scratch", action="store_true", help="sin pesos preentrenados (para pruebas)")
    ap.add_argument("--no-install", action="store_true", help="no copiar best.pt a models/wheels_pose.pt")
    args = ap.parse_args(argv)
    try:
        best = train(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Mejor modelo: {best}")
    if not args.no_install:
        dest = ROOT / "models" / "wheels_pose.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"Copiado a {dest}: VIMOP lo usará automáticamente (Ruedas: modelo entrenado).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

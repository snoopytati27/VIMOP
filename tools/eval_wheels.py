"""
tools/eval_wheels.py — Compara el modelo entrenado (pose) con el respaldo por máscara en el conjunto de validación.

    python tools/eval_wheels.py --data wheels_dataset/dataset.yaml [--model models/wheels_pose.pt]

Para cada moto de validación con las dos ruedas visibles se mide, por método:
    · error de cada rueda en píxeles y normalizado por la diagonal del bbox (0,05 = 5 % de la diagonal)
    · error angular del eje R–F en grados (0–90; el eje no tiene sentido)
    · cuántas motos no se encontraron
El respaldo por máscara no distingue trasera de delantera, así que el error de cada rueda se mide con el
emparejamiento que minimiza la distancia total; para el modelo entrenado se informa además el % de motos
con trasera/delantera en el orden correcto.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.geometry import angular_error_deg, segment_axis_angle_deg  # noqa: E402
from core.image_io import ImageLoadError, read_image  # noqa: E402
from utils import bbox_iou  # noqa: E402

MATCH_IOU = 0.30


@dataclass
class GroundTruth:
    bbox: tuple[float, float, float, float]
    rear: np.ndarray
    front: np.ndarray


@dataclass
class MethodStats:
    name: str
    total: int = 0
    found: int = 0
    rear_px: list[float] = field(default_factory=list)
    front_px: list[float] = field(default_factory=list)
    rear_norm: list[float] = field(default_factory=list)
    front_norm: list[float] = field(default_factory=list)
    angle: list[float] = field(default_factory=list)
    order_ok: int = 0

    def add(self, gt: GroundTruth, rear: np.ndarray | None, front: np.ndarray | None) -> None:
        self.total += 1
        if rear is None or front is None:
            return
        self.found += 1
        diag = float(np.hypot(gt.bbox[2] - gt.bbox[0], gt.bbox[3] - gt.bbox[1]))
        straight = np.linalg.norm(rear - gt.rear) + np.linalg.norm(front - gt.front)
        crossed = np.linalg.norm(rear - gt.front) + np.linalg.norm(front - gt.rear)
        if crossed < straight:                                   # mejor emparejamiento (el respaldo no distingue)
            rear, front = front, rear
        else:
            self.order_ok += 1
        er, ef = float(np.linalg.norm(rear - gt.rear)), float(np.linalg.norm(front - gt.front))
        self.rear_px.append(er)
        self.front_px.append(ef)
        self.rear_norm.append(er / diag)
        self.front_norm.append(ef / diag)
        self.angle.append(angular_error_deg(segment_axis_angle_deg(rear, front), segment_axis_angle_deg(gt.rear, gt.front)))

    def summary(self) -> dict:
        def mean(v):
            return round(float(np.mean(v)), 3) if v else None
        return {"metodo": self.name, "motos": self.total, "encontradas": self.found,
                "error_rueda_trasera_px": mean(self.rear_px), "error_rueda_delantera_px": mean(self.front_px),
                "error_rueda_trasera_norm": mean(self.rear_norm), "error_rueda_delantera_norm": mean(self.front_norm),
                "error_angular_grados": mean(self.angle),
                "orden_trasera_delantera_correcto_pct": round(100.0 * self.order_ok / self.found, 1) if self.found else None}


def load_ground_truth(data_yaml: Path, split: str = "val") -> list[tuple[Path, list[GroundTruth]]]:
    """Imágenes de `split` (val o test) con sus motos de referencia (solo las que tienen las 2 ruedas visibles)."""
    import yaml
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"]) if cfg.get("path") else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if split not in cfg:
        raise ValueError(f"El dataset no tiene conjunto '{split}' (¿faltan fuentes para separarlo? ver AVISOS al exportar)")
    val_dir = root / cfg[split]
    out = []
    for img_path in sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        label = Path(str(img_path).replace("images", "labels", 1)).with_suffix(".txt")
        try:
            h, w = read_image(img_path).image.shape[:2]
        except ImageLoadError:
            continue
        motos = []
        if label.exists():
            for line in label.read_text(encoding="utf-8").splitlines():
                v = [float(t) for t in line.split()]
                if len(v) < 11 or int(v[7]) != 2 or int(v[10]) != 2:
                    continue
                cx, cy, bw, bh = v[1] * w, v[2] * h, v[3] * w, v[4] * h
                motos.append(GroundTruth((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                                         np.array([v[5] * w, v[6] * h]), np.array([v[8] * w, v[9] * h])))
        out.append((img_path, motos))
    return out


def predict_pose(model, img: np.ndarray, imgsz: int, kpt_conf: float):
    """[(bbox, rear, front)] del modelo de puntos clave; descarta puntos por debajo de kpt_conf."""
    res = model.predict(img, imgsz=imgsz, conf=0.20, verbose=False)[0]
    out = []
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return out
    boxes = res.boxes.xyxy.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy()
    kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
    for b, k, c in zip(boxes, kxy, kc):
        if c[0] >= kpt_conf and c[1] >= kpt_conf:
            out.append((tuple(float(x) for x in b), k[0].astype(float), k[1].astype(float)))
    return out


def best_match(gt_bbox, candidates):
    best, best_iou = None, MATCH_IOU
    for cand in candidates:
        iou = bbox_iou(gt_bbox, cand[0])
        if iou >= best_iou:
            best, best_iou = cand, iou
    return best


def evaluate(data_yaml: Path, model_path: Path, cfg, split: str = "val") -> tuple[MethodStats, MethodStats]:
    from analysis.wheels import WheelLocator
    from core.detector import YoloDetector, select_device
    from ultralytics import YOLO

    device, is_gpu, _ = select_device(cfg.DEVICE)
    pose_model = YOLO(str(model_path))
    detector = YoloDetector(cfg, device, is_gpu)
    mask_cfg = cfg.copy()
    mask_cfg.WHEEL_BACKEND = "mask"
    locator = WheelLocator(mask_cfg, device, False)
    pose, mask = MethodStats("pose (modelo entrenado)"), MethodStats("mask (respaldo)")
    for img_path, motos in load_ground_truth(data_yaml, split):
        if not motos:
            continue
        img = read_image(img_path).image
        pose_preds = predict_pose(pose_model, img, int(cfg.WHEEL_IMG_SIZE), float(cfg.WHEEL_KPT_CONF))
        dets = detector.detect(img).motorcycles
        for gt in motos:
            p = best_match(gt.bbox, pose_preds)
            pose.add(gt, *(p[1:] if p else (None, None)))
            d = best_match(gt.bbox, [(det.bbox, det) for det in dets])
            if d is None:
                mask.add(gt, None, None)
            else:
                d[1].track_id = None
                found = locator.locate(None, d[1])
                ok = found["source"] == "mask"                          # sin máscara no hay estimación de ruedas
                mask.add(gt, np.asarray(found["rear"]) if ok else None, np.asarray(found["front"]) if ok else None)
    return pose, mask


def format_table(stats: list[MethodStats]) -> str:
    rows = [s.summary() for s in stats]
    cols = [("metodo", "método"), ("encontradas", "encontradas"), ("error_rueda_trasera_px", "trasera px"),
            ("error_rueda_delantera_px", "delantera px"), ("error_rueda_trasera_norm", "trasera/diag"),
            ("error_rueda_delantera_norm", "delantera/diag"), ("error_angular_grados", "ángulo °")]
    lines = [" | ".join(h for _, h in cols)]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r[k]
            cells.append("—" if v is None else (f"{r['encontradas']}/{r['motos']}" if k == "encontradas" else str(v)))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from config import load_config
    ap = argparse.ArgumentParser(description="Comparar pose vs máscara en el conjunto de validación")
    ap.add_argument("--data", required=True, help="dataset.yaml exportado por label_wheels")
    ap.add_argument("--model", default=None, help="por defecto WHEEL_MODEL_PATH (models/wheels_pose.pt)")
    ap.add_argument("--out", default=None, help="guardar el resultado en este .json")
    ap.add_argument("--split", choices=("val", "test"), default="val",
                    help="val para decidir/ajustar; test SOLO al final, una vez, para el resultado que se reporta")
    args = ap.parse_args(argv)
    cfg = load_config()
    data = Path(args.data)
    if data.is_dir():
        data = data / "dataset.yaml"
    model = Path(args.model) if args.model else cfg.resolve_path(cfg.WHEEL_MODEL_PATH)
    for path, what in ((data, "el dataset"), (model, "el modelo de ruedas")):
        if not path.exists():
            print(f"ERROR: no se encontró {what}: {path}")
            return 1
    try:
        pose, mask = evaluate(data, model, cfg, args.split)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Conjunto evaluado: {args.split}")
    print(format_table([pose, mask]))
    if pose.total == 0:
        print(f"\nAVISO: el conjunto '{args.split}' no tiene motos con las dos ruedas visibles.")
    if args.out:
        Path(args.out).write_text(json.dumps([pose.summary(), mask.summary()], indent=1, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nGuardado en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

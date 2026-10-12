"""
tools/wheel_training.py — Entrenador YOLO-pose con aumentos que simulan la cámara fija real (y la infrarroja).

Ultralytics no trae "pasar a escala de grises" ni los defectos de una cámara de seguridad como aumentos de datos.
Las fotos de internet y de celular son nítidas y de cerca; la cámara fija ve motos pequeñas, borrosas, comprimidas,
con las líneas curvas por la lente y en infrarrojo de noche. Aquí, sobre cada lote ya armado (en la GPU):
ojo de pez (mueve también cajas y puntos), baja resolución, desenfoque, compresión JPEG, ruido y noche/infrarrojo,
más la fracción en grises y el brillo variable de siempre.

Este mismo código (el bloque entre las marcas "aumentos tipo cámara") está dentro del cuaderno
colab/train_wheels_colab.ipynb (Colab no tiene este repositorio); una prueba comprueba que sigan iguales.
"""
from __future__ import annotations

# --- aumentos tipo cámara (inicio) ---
import math
import random

import numpy as np
import torch
import torch.nn.functional as F


def fisheye_points(xy, k):
    """Puntos (..., 2) en 0–1 de la imagen original -> dónde caen tras el ojo de pez de fuerza k (coincide con fisheye_image)."""
    s = np.asarray(xy, dtype=np.float64) * 2.0 - 1.0
    r_s = np.linalg.norm(s, axis=-1, keepdims=True)
    r = r_s.copy()
    for _ in range(12):                                       # Newton: r (1 + k r²) / (1 + k) = r_s
        g = r * (1 + k * r * r) / (1 + k) - r_s
        r = r - g * (1 + k) / (1 + 3 * k * r * r)
    scale = np.where(r_s > 1e-9, r / np.maximum(r_s, 1e-9), 1.0 + k)
    return (s * scale + 1.0) / 2.0


def fisheye_boxes(xywh, k):
    """Cajas (N, 4) cx, cy, w, h en 0–1 -> la caja que contiene a la caja ya distorsionada."""
    t = np.linspace(0.0, 1.0, 9)
    out = np.empty((len(xywh), 4), dtype=np.float64)
    for i, (cx, cy, bw, bh) in enumerate(np.asarray(xywh, dtype=np.float64)):
        x1, x2, y1, y2 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
        xs, ys = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        edge = np.concatenate([np.stack([xs, np.full_like(t, y1)], 1), np.stack([xs, np.full_like(t, y2)], 1),
                               np.stack([np.full_like(t, x1), ys], 1), np.stack([np.full_like(t, x2), ys], 1)])
        p = fisheye_points(edge, k)
        lo, hi = p.min(axis=0), p.max(axis=0)
        out[i] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[0] - lo[0], hi[1] - lo[1]]
    return out


def fisheye_image(img, k):
    """img (3, H, W): efecto ojo de pez (el centro se agranda y los bordes se comprimen). k ~ 0.1 suave, 0.4 fuerte."""
    _, h, w = img.shape
    ys = ((torch.arange(h, device=img.device, dtype=img.dtype) + 0.5) / h) * 2 - 1
    xs = ((torch.arange(w, device=img.device, dtype=img.dtype) + 0.5) / w) * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    f = (1 + k * (gx * gx + gy * gy)) / (1 + k)
    grid = torch.stack((gx * f, gy * f), dim=-1)[None]
    return F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def low_resolution(img, factor):
    """Achica la imagen a `factor` (0–1) y la vuelve a agrandar: una moto lejana con pocos píxeles."""
    _, h, w = img.shape
    small = F.interpolate(img[None], scale_factor=factor, mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)[0]


def gaussian_blur(img, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k1 = torch.exp(-(x * x) / (2 * sigma * sigma))
    k1 = k1 / k1.sum()
    c = img.shape[0]
    pad = F.pad(img[None], (r, r, r, r), mode="reflect")
    out = F.conv2d(pad, k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(out, k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)[0]


def jpeg_roundtrip(img, quality):
    """Comprime y descomprime como JPEG de calidad `quality` (los cuadritos de un video comprimido)."""
    import cv2
    arr = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(dec).to(img.device).permute(2, 0, 1).to(img.dtype) / 255.0


def _warp_labels(batch, i, k):
    """Mueve las cajas y los puntos de la imagen i del lote con el mismo ojo de pez que se le aplicó a la imagen."""
    sel = batch["batch_idx"] == i
    if not bool(sel.any()):
        return
    boxes = batch["bboxes"]
    boxes[sel] = torch.as_tensor(fisheye_boxes(boxes[sel].double().cpu().numpy(), k), dtype=boxes.dtype,
                                 device=boxes.device)
    kpts = batch.get("keypoints")
    if kpts is not None and kpts.shape[-1] >= 3:
        kp = kpts[sel].double().cpu().numpy()
        moved = fisheye_points(kp[..., :2].reshape(-1, 2), k).reshape(kp[..., :2].shape)
        kp[..., :2] = np.where((kp[..., 2] > 0)[..., None], moved, kp[..., :2])   # los "fuera de imagen" no se mueven
        kpts[sel] = torch.as_tensor(kp, dtype=kpts.dtype, device=kpts.device)


def camera_augment_batch(batch, strength=1.0):
    """
    Aumentos que imitan la cámara fija real, imagen por imagen (cada una se aplica con su probabilidad × strength;
    strength 0 = nada). Orden: ojo de pez (mueve las etiquetas), baja resolución, desenfoque, JPEG, ruido, noche.
    """
    if strength <= 0:
        return
    img = batch["img"]                                            # (B, 3, H, W) float 0–1
    dev = img.device

    def hit(p):
        return random.random() < min(1.0, p * strength)

    for i in range(img.shape[0]):
        x = img[i]
        if hit(0.30):
            k = random.uniform(0.10, 0.40)
            x = fisheye_image(x, k)
            _warp_labels(batch, i, k)
        if hit(0.35):
            x = low_resolution(x, random.uniform(0.25, 0.60))
        if hit(0.30):
            x = gaussian_blur(x, random.uniform(0.6, 2.2))
        if hit(0.30):
            x = jpeg_roundtrip(x, random.randint(15, 60))
        if hit(0.25):
            x = x + torch.randn_like(x) * random.uniform(0.01, 0.05)
        if hit(0.12):                                              # noche / infrarrojo: oscuro, gris y con grano
            x = x.mean(dim=0, keepdim=True).expand(3, -1, -1) * random.uniform(0.25, 0.55)
            x = x + torch.randn(1, *x.shape[1:], device=dev, dtype=x.dtype) * random.uniform(0.03, 0.07)
        img[i] = x.clamp(0.0, 1.0)
# --- aumentos tipo cámara (fin) ---


def make_trainer(gray_prob: float = 0.25, brightness: tuple[float, float] = (0.6, 1.4), camera_aug: float = 1.0):
    """Clase entrenadora para `model.train(trainer=...)`. camera_aug = fuerza de los aumentos tipo cámara (0 = apagados)."""
    from ultralytics.models.yolo.pose import PoseTrainer

    class WheelTrainer(PoseTrainer):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            camera_augment_batch(batch, camera_aug)
            img = batch["img"]                                   # (B, 3, H, W) float 0–1
            b = img.shape[0]
            gray = torch.rand(b, device=img.device) < gray_prob
            if bool(gray.any()):
                lum = img[gray].mean(dim=1, keepdim=True)
                img[gray] = lum.expand(-1, 3, -1, -1)
            lo, hi = brightness
            gain = lo + (hi - lo) * torch.rand(b, 1, 1, 1, device=img.device)
            batch["img"] = (img * gain).clamp_(0.0, 1.0)
            return batch

    return WheelTrainer

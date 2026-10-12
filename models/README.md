# Carpeta `models/`

| Archivo | ¿Obligatorio? | Uso |
|---|---|---|
| `yolo11s.pt` | Opcional | Detector normal (solo bounding boxes, sin máscara). Ya no es el de fábrica. |
| `yolo11m.pt` | No | Detector sin segmentación (más preciso en cajas). Ya no es el de fábrica: sin máscara no hay línea de ejes por máscara (poner su ruta en `MODEL_PATH` solo si tiene `wheels_pose.pt`). |
| `yolo11s-seg.pt` | Sí (por defecto) | Detector con segmentación: su máscara alimenta el respaldo de ruedas (`WHEEL_BACKEND = mask`). |
| `yolov8m-seg.pt` | Opcional | Otro modelo `-seg` (poner su ruta en `MODEL_PATH`). |
| `wheels_pose.pt` | Opcional | Modelo de 2 puntos clave (rueda trasera y delantera) entrenado con `tools/train_wheels.py` o en Colab. Si existe, VIMOP lo usa solo. |
| `plate.pt` | Opcional | YOLO entrenado para detectar **placas de moto**. Sin él se usa una heurística de contornos, menos fiable. |

Entrenamiento:

```bat
python tools/train_models.py plates --data dataset_placas/data.yaml
```

VIMOP no identifica color ni marca/modelo de las motos (se quitó el 3-oct-2026): solo ID, placa y puntaje de parqueo.

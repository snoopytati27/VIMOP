# VIMOP — Visión Inteligente para el Monitoreo y Ordenamiento del Parqueo

Universidad Industrial de Santander (UIS) · Visión Artificial y Estructuras de Datos · 2026
Juan Diego Garzón Cordero (2254270) · Yusmary Tatiana Castañeda García (2254254)

VIMOP observa el parqueadero de motos con una cámara en tiempo real y, para cada moto: la detecta y la
sigue con un **ID persistente** (`MOTO #n`), intenta leer su **placa**, decide en qué **puesto** está y califica su parqueo de 0 a 100 con
**orientación, ubicación, centrado e invasión**. No identifica color ni marca/modelo (se quitaron el 3-oct-2026).

> **Usuarios (no programadores):** lean el **[Manual de usuario](docs/MANUAL_USUARIO.md)**.
> Dentro del programa se abre con el botón **Ayuda** o la tecla **F1**.
> Este README es la documentación técnica.

---

## 1. Instalación (Windows 10/11, VS Code, Python 3.10 – 3.12)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

- **Solo CPU:** no hay que hacer nada más; VIMOP usa `models/yolo11s-seg.pt`.
- **GPU NVIDIA:** después, con el venv activado:
  `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
  VIMOP detecta CUDA (`torch.cuda.is_available()`), usa `models/yolo11s-seg.pt` en la GPU y lo muestra en la barra superior.
- La primera ejecución descarga pesos de YOLO y modelos de EasyOCR (requiere internet).

## 2. Ejecución

| Comando | Qué hace |
|---|---|
| `python main.py` | Menú: **MONITOREAR** (Iniciar cámara · Examinar archivo, abren la ventana principal) y **PREPARAR EL PARQUEADERO** (Calibrar · Cargar calibración · Configuración, se quedan en el menú) · Ayuda (F1) |
| `python main.py --video [RUTA]` / `--camera` / `--calibrate` | Accesos directos |
| `python main.py --headless --video --save-video` | Pipeline sin interfaz (consola + video anotado en `results/videos/`) |
| `python main.py --headless --photos F1.jpg CARPETA/ --save-images` | Modo foto por lotes sin interfaz → `results/lote_<fecha>.csv` (+ `<archivo>_vimop.jpg`) |
| `python -m unittest -v` | 142 pruebas unitarias (+1 lenta opcional: `VIMOP_SLOW_TESTS=1`, entrena 1 época de humo) |

Opciones del modo sin interfaz: `--max-frames N`, `--calibration archivo.json`, `--results carpeta`.

**Fuente de video** (solo editando `config.json`):

```json
{"SOURCE_TYPE": "webcam", "CAMERA_SOURCE": 0}
{"SOURCE_TYPE": "rtsp",   "CAMERA_SOURCE": "rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101"}
{"SOURCE_TYPE": "ip",     "CAMERA_SOURCE": "http://192.168.1.50:8080/video"}
{"SOURCE_TYPE": "video",  "CAMERA_SOURCE": "videos/prueba.mp4"}
```

Todos los parámetros están comentados en `config.py`; lo que se cambie en *Configuración* se guarda en `config.json`.

## 3. Arquitectura

```
Cámara → Captura → Detección → Tracking/ID → Identificación → Ruedas (WheelLocator) → Análisis (línea de ejes) → Puntuación → Clasificación → Visual + BD
```

```
proyecto_motos/
├── main.py                       menú de inicio y modo --headless
├── config.py / config.json       parámetros (por defecto / del usuario)
├── core/
│   ├── camera.py                 VideoSource: webcam / ip / rtsp / video / image, hilo lector + reconexión
│   ├── image_io.py               lectura de fotos: Unicode, alfa, grises, EXIF, MAX_IMAGE_SIDE, tipo de archivo
│   ├── photo.py                  modo foto sin interfaz: analizar una foto y exportar un lote
│   ├── detector.py               YOLO + model.track(persist=True) (ByteTrack / BoT-SORT)
│   ├── tracker.py                IdentityManager: track_id -> MOTO #n, retención, re-identificación
│   ├── registry.py               fichas, historiales (deque), votaciones, eventos
│   ├── preprocess.py             preprocesado compartido con la herramienta de etiquetado (CLAHE en infrarrojo)
│   └── pipeline.py               orquestador por frame (sin dependencias de la interfaz)
├── identification/               plate.py · worker.py (hilo de OCR)
├── analysis/
│   ├── geometry.py               ángulos de ejes, PCA, eje de ruedas por máscara, homografía (Shapely)
│   ├── wheels.py                 WheelLocator: ruedas por modelo de puntos clave (pose) o por máscara
│   ├── parking_spaces.py         puestos (polígonos Shapely), JSON, eje/largo/ancho del puesto, homografía del piso
│   ├── parking_analyzer.py       O, U, C, I y score con la línea de ejes R–F
│   └── state_machine.py          ENTRANDO / MOVIÉNDOSE / ESTACIONANDO / ESTACIONADA
├── calibration.py                herramienta gráfica de calibración de puestos
├── database.py                   CSV + JSON + SQLite, escritura asíncrona
├── ui/                           PySide6: main_window, worker y photo_worker (QThread), renderer, widgets, dialogs, help, theme
├── docs/                         MANUAL_USUARIO.md + img/ (capturas)
├── tests/                        pruebas unitarias
├── tools/                        train_models.py (placas) · label_wheels.py (etiquetado semiautomático de ruedas)
│                                 · wheel_dataset.py · train_wheels.py · wheel_training.py · eval_wheels.py
├── colab/train_wheels_colab.ipynb entrenamiento del modelo de ruedas en Google Colab (avance en Drive)
├── models/                       pesos (ver models/README.md)
└── results/                      detections.csv, parking_events.csv, motos.json, vimop.db, vimop.log, screenshots/, videos/
```

Hilos: captura (`VideoSource`) → *Queue* → procesamiento (`QThread`) → señales Qt → interfaz; OCR en
`IdentificationWorker`; escritura a disco en `ResultsStore`. La interfaz nunca ejecuta YOLO ni OCR.

## 4. Modelo geométrico: línea de ejes

Cada moto se representa con un **segmento L = R–F**: R es el punto de contacto de la rueda trasera con el piso y F el
de la delantera. El bbox se usa solo para detección y tracking (se dibuja tenue con `SHOW_BBOX`). La etiqueta
(`MOTO #07 · 87.4 · BUENO`) va en el punto medio **M = (R + F) / 2**.

**Obtención de las ruedas** (`analysis/wheels.py`, `WheelLocator.locate(frame, det) -> {rear, front, source, confidence}`):

| `WHEEL_BACKEND` | Qué hace |
|---|---|
| `pose` | YOLO-pose con 2 puntos clave en `models/wheels_pose.pt` (una inferencia por frame para todas las motos). Interfaz: *Ruedas: modelo entrenado*. |
| `mask` | Respaldo sin entrenar: franja inferior de la máscara (`BOTTOM_BAND` = 0.25), PCA, extremos metidos `INSET_FRACTION` = 0.12. Interfaz: *Ruedas: estimadas por máscara (aproximado)*. Necesita un modelo `-seg`. |
| `auto` (por defecto) | `pose` si el archivo existe; si no, `mask`. Si un punto del modelo no llega a `WHEEL_KPT_CONF`, esa moto usa `mask`. |

R y F se suavizan con una media móvil (`WHEEL_SMOOTH_FRAMES` = 5, `deque` por track). Sin modelo ni máscara solo queda un
segmento a lo largo del bbox (fuente `bbox`): no es una orientación real y se marca **no confiable**.

**Caso degenerado (moto de frente o de espaldas):** si `|L| < MIN_AXIS_LEN_RATIO` (0.35) × alto del bbox, la orientación
es **no confiable**: no se inventa ningún ángulo (O = `None`), el score se calcula con los pesos restantes renormalizados
y se muestra el aviso *orientación no confiable*.

**Perspectiva:** si la calibración trae `ground_quad` (4 esquinas de un rectángulo o paralelogramo real del piso + su
tamaño + `angle`, 90 por omisión), R, F y los puestos se pasan a la vista cenital con una homografía antes de medir.
La homografía se calibra con UN puesto; lejos de él, una lente con distorsión (fisheye) la puede disparar (un puesto
de ~2 m² sale midiendo miles). Por eso cada puesto se mide en el piso solo si su área ahí no se aleja demasiado del
área real esperada (`GROUND_AREA_TOLERANCE` en `analysis/parking_spaces.py`); si no, mide en la imagen, como si no
hubiera piso. Sin `ground_quad`, o en un puesto que no pasó ese chequeo, se avisa *ángulos aproximados*.

Sea S el puesto (eje mayor de ángulo θ_S, calculado una vez en la calibración; largo L_S y ancho W_S) y Sj los demás:

| Factor | Fórmula | Peso |
|---|---|---|
| Puesto asignado | el que contiene el mayor `long(L ∩ Si) / long(L)`, si ≥ `MIN_ASSIGN` (30 %); si no, FUERA | — |
| Orientación O | `θ_M = atan2(F − R)`, `d = abs(θ_M − θ_S) mod 180`, `err = min(d, 180 − d)`, `O = 100 · max(0, 1 − err/ERR_MAX)` (`ERR_MAX` = 45°) | 35 % |
| Ubicación U | `long(L ∩ S) / long(L) · 100` (`shapely.LineString.intersection`) | 35 % |
| Centrado C | `(dl, dt)` = componentes de `M − centro(S)` a lo largo y perpendicular al eje; `d = √((dl/(L_S/2))² + (dt/(W_S/2))²)`; `C = 100 · max(0, 1 − d)` | 20 % |
| Invasión I | `ratio = max_{j ≠ puesto} long(L ∩ Sj) / long(L)`, `I = 100 · (1 − min(1, ratio/0.30))`; niveles 0–2 % ninguna, 2–10 % pequeña, > 10 % considerable | 10 % |

**Score** = `0.35·O + 0.35·U + 0.20·C + 0.10·I` ∈ [0, 100], promedio de las últimas `HISTORY_LENGTH` = 15 evaluaciones
(deque; los fotogramas con orientación no confiable no entran en el promedio de O). **Clasificación** con histéresis de 2 puntos:

| Score | Estado | Color de la línea |
|---|---|---|
| 90–100 | EXCELENTE | verde |
| 75–89 | BUENO | azul |
| 60–74 | REGULAR | amarillo |
| 40–59 | MALO | naranja |
| 0–39 | PÉSIMO | rojo |

El manual (sección 8) trae un ejemplo numérico resuelto (96,4 EXCELENTE y 49,1 MALO) que
`tests/test_manual_example.py` compara con el código y con el texto del manual.

### Ruedas entrenadas: etiquetar, entrenar, activar

1. `python tools/label_wheels.py CARPETA` — ventana PySide6: extrae fotogramas de los videos (sin casi-duplicados),
   **preetiqueta** con el detector + el backend `mask`, permite arrastrar los puntos, guarda y reanuda el progreso
   (`wheels_labels.json`) y **exporta** solo lo revisado en formato YOLO-pose (`kpt_shape: [2, 3]`, `flip_idx: [0, 1]`),
   con partición 80/20 **por fuente** (video o sesión de fotos) y un `.zip` para Drive.
2. Entrenar: `python tools/train_wheels.py --data wheels_dataset/dataset.yaml` (local, GPU si hay) o
   `colab/train_wheels_colab.ipynb` (Colab, avance en Drive). Aumentos con escala de grises y brillo variable.
3. `models/wheels_pose.pt` → VIMOP lo usa solo. `python tools/eval_wheels.py --data ...` compara pose y máscara
   (error de cada rueda en px y normalizado, error angular del eje).

## 5. Detección, tracking, ID y placa

- **YOLO** (COCO, clases `motorcycle` y `person`): una sola inferencia por frame; entrega bbox, confianza y clase.
- **Tracking:** `model.track(persist=True, tracker=bytetrack|botsort)` (Kalman + asociación por IoU con el
  algoritmo húngaro; ByteTrack usa también detecciones de baja confianza para no perder motos tapadas).
- **ID:** `IdentityManager` asigna `MOTO #n` tras `TRACK_MIN_HITS` = 3 frames, conserva el registro
  `ID_RETENTION_SECONDS` = 30 s y re-identifica por IoU con la última caja + histograma de color.
- **Placa:** recorte de la moto → detector (`models/plate.pt`, o heurística de contornos en la mitad inferior,
  **menos fiable**) → escala + CLAHE + binarizado → EasyOCR → regex `PLATE_REGEX` con corrección por posición
  (0/O, 1/I, 8/B, 5/S) → votación ponderada en deque. Sin consenso: **No detectada**. Solo con motos quietas.
- **Estados:** velocidad del punto medio M de la línea de ejes en diagonales del bbox por segundo (> 0.30 moviéndose, < 0.12 quieta);
  2 s quieta en un puesto → ESTACIONADA (4 s fuera). Score definitivo, eventos y alertas solo en ESTACIONADA.

## 6. Modo foto (Examinar archivo)

- **Entrada:** botón **EXAMINAR ARCHIVO** / menú 4 (`QFileDialog.getOpenFileNames`, filtros Videos · Imágenes ·
  Todos los soportados), **EXAMINAR CARPETA**, o arrastrar y soltar. Tipo por extensión y, si no, por contenido
  (`cv2.imdecode` / `cv2.VideoCapture`). Archivos dañados o no soportados → mensaje, sin cerrar.
- **Lectura** (`core/image_io.py`): `np.fromfile` + `cv2.imdecode` (rutas Unicode), alfa → compuesto sobre blanco,
  grises/16 bits → BGR 8 bits, orientación EXIF corregida (Pillow), lado mayor limitado a `MAX_IMAGE_SIDE` (aviso).
- **Fuente:** `VideoSource("image", ruta)` con la misma interfaz (`start/read/release`); entrega un único
  `FramePacket(static=True)`.
- **Pipeline estático** (`VimopPipeline._process_static`): `detector.detect()` (sin tracker), IDs secuenciales
  MOTO #1, #2…, sin máquina de estados (todas ESTACIONADA), placa sobre ese frame sin votación,
  mismas fórmulas O/U/C/I y clasificación sin histéresis, **sin eventos**. Rótulo "MODO FOTO — análisis estático".
- **Calibración** (`ParkingLayout.check_compatibility`): misma resolución → aplicada; misma proporción (±2 %) →
  polígonos escalados; otra proporción u otro punto de vista (correlación de una miniatura 96×54 guardada al
  calibrar, umbral `VIEWPOINT_MIN_SIMILARITY` = 0.35) → el usuario elige *recalibrar sobre la foto*
  (`config/foto_<archivo>.json`) o *continuar sin puestos*. Medido con los datos del proyecto: la misma escena da
  0.87–1.0 (incluso con la cámara en mano o con otro brillo) y la foto de otra cámara −0.12.
- **Resultados:** GUARDAR RESULTADO → `results/screenshots/<archivo>_vimop.jpg` + filas en `detections.csv` con
  columna **Origen** (`camara`, `video:<archivo>`, `foto:<archivo>`); EXPORTAR LOTE → un CSV resumen con barra de
  progreso. Todo en un `QThread` (`ui/photo_worker.py`) que carga los modelos una sola vez.

## 7. Estructuras de datos

| Estructura | Dónde | Por qué |
|---|---|---|
| **dict** `motos[id]`, `track_to_moto` | registry, tracker | acceso O(1) a la ficha de cada moto |
| **list** de puestos + dict `by_id` | parking_spaces | orden P01, P02… para dibujar/listar; O(1) por nombre |
| **set** `active_ids`, `visible_ids` | registry, tracker | pertenencia O(1); aparecidas/desaparecidas/alertas nuevas por diferencia de conjuntos |
| **deque(maxlen=N)** placa, score, posiciones | registry, state_machine | ventana deslizante: insertar/descartar en O(1) |
| **Counter** | votaciones | placa más frecuente |
| **OrderedDict** motos perdidas | tracker | vencidas al inicio → expiración O(1) |
| **Queue** | captura→inferencia, →OCR, →disco, UI→hilo | comunicación segura entre hilos, sin acumular latencia |

| Operación (por frame; n motos, P puestos, k historial) | Complejidad |
|---|---|
| Buscar/actualizar moto por ID | O(1) |
| Asociación del tracker (húngaro) | O(n³), n ≤ ~30 |
| Intersección L ∩ S (segmento con polígono de v vértices) | O(v) por puesto; filtro por bbox del segmento → O(P) comparaciones baratas |
| Análisis de todas las motos | O(n · P) |
| Historial (insertar / votar / promediar) | O(1) / O(k) / O(k) |
| Consulta SQLite por placa | O(log N) con índice |

## 8. Persistencia

`results/detections.csv` (ID, Fecha, Hora, Placa, Puesto, Orientacion, Ubicacion,
Centrado, Invasion, Score, Estado), `parking_events.csv` (ENTRADA, CAMBIO_ESTADO, CAMBIO_PUESTO, SALIDA,
ALERTA…), `motos.json`, `vimop.db` (SQLite; pestaña *Historial / consulta* por placa, ID o fecha). Escritura
asíncrona en un hilo, UTF-8, archivos siempre cerrados.

## 9. Verificación realizada

- `compileall` y `pyflakes` limpios; sin `TODO` ni `pass`.
- **142 pruebas en verde** (Python 3.14, RTX 3050; 1 opcional omitida), entre ellas los criterios de aceptación de la
  línea de ejes: moto alineada → error ≈ 0° y O ≈ 100; girada 90° → O = 0; cruzando dos puestos → U ≈ 50 e I bajo;
  ángulos alrededor de 0°/180° sin saltos; moto de frente → orientación no confiable y score con pesos renormalizados;
  sin `wheels_pose.pt` funciona el respaldo `mask` (y lo indica) y con el modelo lo usa; línea que no toca puestos
  lejanos; homografía del piso; herramienta de etiquetado (arrastrar puntos, atajos, progreso, guardar/reanudar,
  diálogo de cambios sin guardar); exportación YOLO-pose con partición por fuente sin fugas; cuaderno de Colab válido;
  además ejemplo del manual = código = texto, F1 sin `docs/` → mensaje claro y el modo foto.
- Dataset exportado aceptado por `yolo train` (1–2 épocas de humo en la GPU con el entrenador de escala de grises).
- Con el video de prueba, `yolo11s-seg.pt` y respaldo `mask` en una RTX 3050 Laptop: 300 frames sin interfaz a ~6.4 FPS
  (~30 ms de inferencia por frame; incluye OCR y guardado), 12 motos con IDs estables.
- Modo foto en la interfaz: 5 fotos con navegación ← →, diálogo de calibración que no coincide, GUARDAR RESULTADO,
  EXPORTAR LOTE (5/5 fotos, 37 motos, un CSV), archivo dañado sin cierre, arrastrar y soltar, y un `.mp4` después.
- Interfaz probada: menú, ventana principal (procesando el video de prueba con la barra *Ruedas: estimadas por máscara*),
  calibración, Ayuda/F1 (con manual, sin archivo y sin carpeta `docs/`) y `tools/label_wheels.py` sin pantalla (offscreen).
- **El modelo `wheels_pose.pt` NO está entrenado todavía** (hacen falta imágenes revisadas por una persona): hoy se usa el respaldo por máscara.
- **No probado con una cámara física** (este equipo no tiene webcam ni cámara IP/RTSP conectada).

## 10. Limitaciones y mejoras

Moto vista de frente o de espaldas (orientación no confiable); el respaldo por máscara es una estimación aproximada y
no distingue trasera de delantera; oclusión de ruedas; noche/contraluz (infrarrojo); sin los puntos del piso (`ground_quad`) los ángulos se
miden con perspectiva; placas pequeñas ilegibles; el video de prueba
está grabado con la cámara en la mano. Los 4 puntos del piso se marcan en la ventana de calibración
(tecla **G**, `python main.py --calibrate`) y se guardan en el JSON de la calibración. Mejoras: entrenar `wheels_pose.pt` con fotos propias, detector de placas de
moto propio (`tools/train_models.py plates`) y OCR especializado.

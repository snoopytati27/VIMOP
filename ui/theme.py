"""ui/theme.py — Tema de la interfaz: estilo HUD (azul noche + cian) como hoja de estilos Qt.

Aquí están SOLO los colores del "chrome" (fondos, bordes, paneles, texto). Los colores del puntaje
(EXCELENTE verde, BUENO azul, REGULAR amarillo, MALO naranja, PÉSIMO rojo y el gris de "moviéndose")
son otra cosa: vienen de utils.py (CLASSIFICATION_TABLE / MOVING_BGR) y no se cambian desde el tema.
"""

FONT = "Bahnschrift"          # fuente de Windows de aire técnico; para volver a la anterior: "Segoe UI"

# ---- chrome
BG = "#03070c"                # fondo general (casi negro azulado)
BG_GLOW = "#07162a"           # centro del degradado del fondo
PANEL = "#08161f"             # fondo de paneles / tarjetas
PANEL_DEEP = "#040b12"        # fondo de tablas, cuadros de texto y del visor
LINE = "#123543"              # bordes finos
ACCENT = "#00e5ff"            # cian principal (títulos, bordes activos, pestaña elegida)
ACCENT_DIM = "#0b6f80"        # cian apagado (bordes de botones, barras de desplazamiento)
TEXT = "#d4f6fb"              # texto normal
MUTED = "#5f8c98"             # texto secundario (etiquetas, subtítulos)
DIM = "#3f6470"               # texto terciario (ayudas pequeñas)

# ---- esquinas redondeadas (las tablas de datos, QTableWidget, se quedan rectas a propósito)
RADIUS_PANEL = 10             # paneles / tarjetas / visor
RADIUS_CONTROL = 7            # botones, campos de texto, listas, pestañas
RADIUS_CHIP = 11              # chips de la cabecera (forma de píldora)

# ---- colores con significado (no decorativos)
OK = "#00e676"                # "sin alertas", calibración cargada
WARN = "#ffb74d"              # avisos (ángulo no confiable, etc.)
WARN_STRONG = "#ffd600"       # "PUESTOS: SIN CALIBRAR"
INVADE = "#ff8a65"            # puestos invadidos
ALERT = "#ff5252"             # texto de alertas
DANGER = "#ff3b3b"            # botón DETENER: borde, letras e ícono rojos (con brillo) sobre azul

STYLE = f"""
* {{ font-family: '{FONT}'; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#central {{ background: qradialgradient(cx:0.5, cy:0.3, radius:0.9, fx:0.5, fy:0.3,
    stop:0 {BG_GLOW}, stop:1 {BG}); }}
QFrame#card, QGroupBox {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: {RADIUS_PANEL}px; }}
QGroupBox {{ margin-top: 14px; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; font-weight: 600;
    letter-spacing: 2px; }}
QLabel#appTitle {{ font-size: 24pt; font-weight: 700; color: {ACCENT}; letter-spacing: 6px; }}
QLabel#appSubtitle {{ color: {MUTED}; font-size: 9pt; letter-spacing: 1px; }}
QLabel#sectionTitle {{ color: {ACCENT}; font-weight: 600; font-size: 9pt; letter-spacing: 2px; }}
QLabel#tileValue {{ font-size: 20pt; font-weight: 700; }}
QLabel#tileTitle {{ color: {MUTED}; font-size: 8pt; letter-spacing: 1px; padding-top: 3px; }}
QLabel#chip {{ background: rgba(0, 229, 255, 18); border: 1px solid {ACCENT_DIM}; border-radius: {RADIUS_CHIP}px;
    padding: 3px 10px; color: {ACCENT}; letter-spacing: 1px; }}
QPushButton {{ background: rgba(0, 229, 255, 10); border: 1px solid {ACCENT_DIM}; border-radius: {RADIUS_CONTROL}px;
    padding: 7px 12px; letter-spacing: 1px; }}
QPushButton:hover {{ background: rgba(0, 229, 255, 40); border-color: {ACCENT}; }}
QPushButton:pressed {{ background: rgba(0, 229, 255, 70); }}
QPushButton:checked {{ background: rgba(0, 229, 255, 60); border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #2f4a52; border-color: #0f2630; background: transparent; }}
QPushButton#primaryButton {{ background: rgba(0, 229, 255, 55); color: #eaffff; font-weight: 700;
    border: 1px solid {ACCENT}; }}
QPushButton#primaryButton:hover {{ background: rgba(0, 229, 255, 90); }}
QPushButton#dangerButton {{ background: rgba(0, 229, 255, 40); color: {DANGER}; font-weight: 700;
    border: 2px solid {DANGER}; }}
QPushButton#dangerButton:hover {{ background: rgba(255, 59, 59, 45); }}
QPushButton#dangerButton:pressed {{ background: rgba(255, 59, 59, 80); }}
QPushButton#dangerButton:disabled {{ color: #5a2a30; border: 1px solid #3a1c25; background: rgba(0, 229, 255, 8); }}
QPushButton#menuButton {{ font-size: 13pt; padding: 14px; text-align: left; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{ background: {PANEL_DEEP}; border: 1px solid {LINE};
    border-radius: {RADIUS_CONTROL}px; padding: 5px; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {ACCENT_DIM}; border-radius: {RADIUS_CONTROL}px;
    selection-background-color: rgba(0, 229, 255, 60); }}
QTableWidget {{ background: {PANEL_DEEP}; gridline-color: #0c2530; border: 1px solid {LINE}; border-radius: 0px;
    alternate-background-color: #06111a; selection-background-color: rgba(0, 229, 255, 60); }}
QHeaderView::section {{ background: {PANEL}; color: {ACCENT}; padding: 5px; border: none;
    border-bottom: 1px solid {ACCENT_DIM}; font-weight: 600; letter-spacing: 2px; }}
QTabWidget::pane {{ border: none; }}                 /* el borde lo pone la tabla (recta) */
QTabBar::tab {{ background: transparent; padding: 7px 16px; color: {MUTED}; border-bottom: 2px solid transparent;
    border-top-left-radius: {RADIUS_CONTROL}px; border-top-right-radius: {RADIUS_CONTROL}px; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; background: rgba(0, 229, 255, 18); }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTextBrowser, QListWidget {{ background: {PANEL_DEEP}; border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px; }}
QListWidget::item:selected {{ background: rgba(0, 229, 255, 70); color: #eaffff; }}
QProgressBar {{ background: {PANEL_DEEP}; border: 1px solid {LINE}; border-radius: {RADIUS_CONTROL}px;
    text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: {RADIUS_CONTROL}px; }}
QStatusBar {{ background: #02050a; color: {MUTED}; border-top: 1px solid {LINE}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {ACCENT_DIM}; border-radius: 6px; }}
QScrollArea {{ border: none; }}
QScrollArea#controlsScroll, QScrollArea#controlsScroll > QWidget, QWidget#controlsContent {{
    background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {ACCENT_DIM}; min-height: 30px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: {ACCENT_DIM}; min-width: 30px; border-radius: 4px; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""

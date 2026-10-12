"""
ui/help.py — Manual de usuario dentro de la aplicación (botón "Ayuda" y tecla F1).

Renderiza docs/MANUAL_USUARIO.md (Markdown) en una ventana. Si la carpeta docs/ o el
archivo no existen, muestra un mensaje claro en lugar de fallar.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QFont, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser, QVBoxLayout

from config import BASE_DIR
from ui.theme import ACCENT

log = logging.getLogger("vimop.help")

MANUAL_PATH = BASE_DIR / "docs" / "MANUAL_USUARIO.md"


def load_manual(path: Path = MANUAL_PATH) -> tuple[str | None, str]:
    """(texto Markdown o None, mensaje para el usuario). Nunca lanza excepciones."""
    if not path.parent.exists():
        return None, (f"No se encontró la carpeta del manual:\n{path.parent}\n\n"
                      "Copie la carpeta 'docs' del proyecto junto a main.py.")
    if not path.exists():
        return None, (f"No se encontró el manual de usuario:\n{path}\n\n"
                      "Copie el archivo MANUAL_USUARIO.md dentro de la carpeta 'docs'.")
    try:
        return path.read_text(encoding="utf-8"), ""
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"No se pudo leer el manual ({path.name}): {exc}"


class HelpDialog(QDialog):
    def __init__(self, markdown: str, source: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VIMOP — Manual de usuario")
        self.resize(1000, 820)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        pal = self.view.palette()
        pal.setColor(QPalette.ColorRole.Link, QColor("#6ab7ff"))       # enlaces legibles en tema oscuro
        self.view.setPalette(pal)
        base_dir = source.parent
        self.view.setSearchPaths([str(base_dir)])                       # para las imágenes docs/img/...
        self.view.document().setBaseUrl(QUrl.fromLocalFile(str(base_dir) + "/"))
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setFamilies(["Consolas", "Courier New", "Lucida Console"])   # fuente fija real (no símbolos)
        self.view.document().setDefaultStyleSheet(
            f"h1,h2,h3{{color:{ACCENT};}} a{{color:#6ab7ff;}} "
            "code,pre{font-family:Consolas,'Courier New',monospace; color:#9ad0ff;}")
        self.view.setMarkdown(markdown)
        # los bloques de código usan la fuente fija del documento
        doc = self.view.document()
        cursor = QTextCursor(doc)
        block = doc.begin()
        while block.isValid():
            if block.blockFormat().nonBreakableLines() or block.charFormat().fontFixedPitch():
                fmt = QTextCharFormat()
                fmt.setFont(mono)
                cursor.setPosition(block.position())
                cursor.setPosition(block.position() + block.length() - 1, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(fmt)
            else:
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid() and frag.charFormat().fontFixedPitch():
                        fmt = QTextCharFormat()
                        fmt.setFont(mono)
                        cursor.setPosition(frag.position())
                        cursor.setPosition(frag.position() + frag.length(), QTextCursor.MoveMode.KeepAnchor)
                        cursor.mergeCharFormat(fmt)
                    it += 1
            block = block.next()
        top = QPushButton("⬆  Inicio")
        top.clicked.connect(lambda: self.view.verticalScrollBar().setValue(0))
        close = QPushButton("Cerrar")
        close.clicked.connect(self.accept)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(f"Archivo: {source}"))
        bar.addStretch(1)
        bar.addWidget(top)
        bar.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)


def show_manual(parent=None, path: Path | None = None) -> bool:
    """Abre el manual. Devuelve False (y muestra un mensaje) si no está disponible."""
    path = path or MANUAL_PATH
    text, msg = load_manual(path)
    if text is None:
        log.warning(msg.replace("\n", " "))
        QMessageBox.warning(parent, "Manual no disponible", msg)
        return False
    HelpDialog(text, path, parent).exec()
    return True

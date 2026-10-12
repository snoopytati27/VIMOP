"""El manual (F1) debe cargarse, y si falta docs/ debe devolverse un mensaje claro sin errores."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.synthetic import ROOT

from ui.help import MANUAL_PATH, load_manual


class ManualLoadingTests(unittest.TestCase):
    def test_manual_exists_and_loads(self):
        self.assertEqual(MANUAL_PATH, ROOT / "docs" / "MANUAL_USUARIO.md")
        text, msg = load_manual()
        self.assertIsNotNone(text, msg)
        for section in ("Qué es VIMOP", "Calibración", "Examinar video o foto", "Solución de problemas", "Glosario"):
            self.assertIn(section, text)

    def test_images_referenced_by_manual_exist(self):
        text, _ = load_manual()
        import re
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            self.assertTrue((MANUAL_PATH.parent / rel).exists(), rel)

    def test_missing_docs_folder_gives_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, msg = load_manual(Path(tmp) / "no_existe" / "MANUAL_USUARIO.md")
        self.assertIsNone(text)
        self.assertIn("No se encontró la carpeta del manual", msg)

    def test_missing_manual_file_gives_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, msg = load_manual(Path(tmp) / "MANUAL_USUARIO.md")
        self.assertIsNone(text)
        self.assertIn("No se encontró el manual", msg)


if __name__ == "__main__":
    unittest.main()

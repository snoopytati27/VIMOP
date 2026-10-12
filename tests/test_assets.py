"""Logos (ui/assets.py): el fondo negro se vuelve transparente, se recorta el margen y, si falta el archivo,
se vuelve al texto "VIMOP"."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class LogoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_black_background_becomes_transparent_and_margin_is_trimmed(self):
        from ui.assets import ASSETS_DIR, WORDMARK, light_image
        original = QImage(str(ASSETS_DIR / WORDMARK))
        img = light_image(WORDMARK)
        self.assertIsNotNone(img)
        self.assertEqual(img.pixelColor(0, 0).alpha(), 0, "la esquina (fondo negro) debe quedar transparente")
        self.assertLess(img.height(), original.height(), "el margen negro de arriba y abajo se recorta")
        alphas = [img.pixelColor(x, img.height() // 2).alpha() for x in range(0, img.width(), 7)]
        self.assertEqual(max(alphas), 255, "las letras (cian brillante) siguen opacas")

    def test_labels_show_the_logo_or_fall_back_to_text(self):
        from ui import assets
        lbl = assets.logo_label(assets.ICON, 40)
        self.assertFalse(lbl.pixmap().isNull())
        self.assertAlmostEqual(lbl.pixmap().deviceIndependentSize().height(), 40, delta=1)
        with patch.object(assets, "light_image", return_value=None):
            lbl = assets.logo_label(assets.WORDMARK, 40)
        self.assertEqual(lbl.text(), "VIMOP")
        self.assertEqual(lbl.objectName(), "appTitle")

    def test_app_icon(self):
        from ui.assets import app_icon
        self.assertFalse(app_icon().isNull())


if __name__ == "__main__":
    unittest.main()

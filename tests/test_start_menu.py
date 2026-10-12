"""Menú inicial (PySide6, sin pantalla): grupos MONITOREAR / PREPARAR EL PARQUEADERO.

- Calibrar / Cargar calibración: se ejecutan con el menú como parent y el menú NO se cierra.
- Iniciar cámara: acepta con choice "camera".
- Examinar archivo: si se cancela el selector, el menú se queda abierto; si se eligen archivos,
  acepta con choice "file" y las rutas elegidas.
- El texto de "Iniciar cámara" refleja la fuente configurada (describe_source).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.synthetic import test_config  # noqa: E402

try:
    from PySide6.QtWidgets import QApplication, QDialog, QFileDialog
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class StartMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _menu(self, calibration_status=None, actions=None, **cfg_overrides):
        from ui.dialogs import StartMenu
        cfg = test_config(**cfg_overrides)
        calibration_status = calibration_status or (lambda: "Calibración cargada: 3 puestos (puestos.json).")
        actions = actions if actions is not None else {"calibrate": Mock(return_value=True),
                                                        "load": Mock(return_value=True)}
        menu = StartMenu(cfg, calibration_status, actions)
        self.addCleanup(menu.deleteLater)
        return menu, cfg, actions

    # -------------------------------------------------------------- PREPARAR EL PARQUEADERO
    def test_calibrate_runs_with_menu_as_parent_and_stays_open(self):
        menu, _, actions = self._menu()
        menu._pick("calibrate")
        actions["calibrate"].assert_called_once_with(menu)
        self.assertIsNone(menu.choice)
        self.assertNotEqual(menu.result(), QDialog.DialogCode.Accepted)

    def test_load_runs_with_menu_as_parent_and_stays_open(self):
        menu, _, actions = self._menu()
        menu._pick("load")
        actions["load"].assert_called_once_with(menu)
        self.assertIsNone(menu.choice)
        self.assertNotEqual(menu.result(), QDialog.DialogCode.Accepted)

    def test_calibrate_refreshes_calibration_label(self):
        status = Mock(side_effect=["Sin calibrar.", "Calibración cargada: 1 puesto (a.json)."])
        menu, _, _ = self._menu(calibration_status=status)
        self.assertEqual(menu.cal_label.text(), "Sin calibrar.")
        menu._pick("calibrate")
        self.assertEqual(menu.cal_label.text(), "Calibración cargada: 1 puesto (a.json).")

    def test_load_even_when_cancelled_keeps_menu_open(self):
        actions = {"calibrate": Mock(return_value=True), "load": Mock(return_value=False)}
        menu, _, actions = self._menu(actions=actions)
        menu._pick("load")
        actions["load"].assert_called_once_with(menu)
        self.assertIsNone(menu.choice)

    # -------------------------------------------------------------- MONITOREAR
    def test_camera_choice_accepts(self):
        menu, _, _ = self._menu()
        menu._pick("camera")
        self.assertEqual(menu.choice, "camera")
        self.assertEqual(menu.result(), QDialog.DialogCode.Accepted)

    def test_camera_button_text_reflects_cfg_source(self):
        menu, _, _ = self._menu(SOURCE_TYPE="webcam", CAMERA_SOURCE=3)
        self.assertIn("Webcam 3", menu.buttons["camera"].text())

    def test_camera_button_text_reflects_video_source(self):
        menu, _, _ = self._menu(SOURCE_TYPE="video", CAMERA_SOURCE="videos/prueba.mp4")
        self.assertIn("prueba.mp4", menu.buttons["camera"].text())

    def test_file_dialog_cancelled_keeps_menu_open(self):
        menu, _, _ = self._menu()
        with patch.object(QFileDialog, "getOpenFileNames", return_value=([], "")):
            menu._pick("file")
        self.assertIsNone(menu.choice)
        self.assertIsNone(menu.chosen_paths)

    def test_file_chosen_accepts_with_choice_and_paths(self):
        menu, _, _ = self._menu()
        chosen = ["C:/videos/uno.mp4", "C:/fotos/dos.jpg"]
        with patch.object(QFileDialog, "getOpenFileNames", return_value=(chosen, "")):
            menu._pick("file")
        self.assertEqual(menu.choice, "file")
        self.assertEqual(menu.chosen_paths, [Path(p) for p in chosen])
        self.assertEqual(menu.result(), QDialog.DialogCode.Accepted)


if __name__ == "__main__":
    unittest.main()

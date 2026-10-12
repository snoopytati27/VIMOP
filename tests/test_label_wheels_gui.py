"""Herramienta de etiquetado (PySide6, sin pantalla): arrastrar puntos, atajos, progreso, guardar/reanudar."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.synthetic import test_config  # noqa: E402
from tests.test_wheel_dataset import FakeDetector, scene  # noqa: E402

try:
    from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False

from analysis.wheels import WheelLocator  # noqa: E402
from core.image_io import write_image  # noqa: E402
from tools.wheel_dataset import VIS_OCCLUDED, VIS_OUTSIDE, VIS_VISIBLE, preannotate  # noqa: E402


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class LabelWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tools.label_wheels import LabelWindow
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for i in range(3):
            write_image(self.root / f"foto{i}.jpg", scene(i, 320, 200))
        self.win = LabelWindow()
        self.win.resize(900, 600)
        self.win.show()
        QTest.qWaitForWindowExposed(self.win)
        self.win.open_folder(self.root, every_seconds=1.0, blocking=True)
        cfg = test_config()
        preannotate(self.win.project, FakeDetector(), WheelLocator(cfg), cfg)
        self.win._fill_list()
        self.win.go(0)
        self.canvas = self.win.canvas

    def tearDown(self):
        if self.win.project is not None:
            self.win.project.dirty = False
        self.win.close()
        self.tmp.cleanup()

    # ---- utilidades -----------------------------------------------------------
    def screen_of(self, key: str) -> QPoint:
        m = self.canvas.moto
        p = self.canvas.to_screen(getattr(m, key)[0], getattr(m, key)[1])
        return QPoint(round(p.x()), round(p.y()))

    def drag(self, start: QPoint, end: QPoint) -> None:
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.canvas, end)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)

    def press_key(self, key) -> None:
        self.win.activateWindow()
        QTest.qWaitForWindowActive(self.win)
        QTest.keyClick(self.win, key)

    # ---- pruebas ----------------------------------------------------------------
    def test_opens_folder_preloads_and_shows_progress(self):
        self.assertEqual(len(self.win.project.entries), 3)
        self.assertEqual(self.win.status_label.text().split("  ")[0], "1 / 3 · revisadas 0")
        self.assertEqual(len(self.canvas.entry.motos), 1)
        self.assertIsNotNone(self.canvas.image)
        self.assertTrue(self.win.listing.item(0).text().startswith("•"))      # preetiquetada, no revisada

    def test_dragging_a_point_moves_it_and_marks_dirty(self):
        self.win.project.dirty = False
        before = list(self.canvas.moto.rear)
        start = self.screen_of("rear")
        target = start + QPoint(25, -12)
        self.drag(start, target)
        after = self.canvas.moto.rear
        expected = self.canvas.to_image(QPointF(target))
        self.assertAlmostEqual(after[0], expected.x(), delta=1.0)
        self.assertAlmostEqual(after[1], expected.y(), delta=1.0)
        self.assertNotEqual((after[0], after[1]), (before[0], before[1]))
        self.assertTrue(self.win.project.dirty)
        self.assertEqual(self.canvas.sel_point, "rear")

    def test_dragging_outside_the_image_is_clamped(self):
        self.drag(self.screen_of("front"), QPoint(self.canvas.width() + 400, -300))
        f = self.canvas.moto.front
        self.assertLessEqual(f[0], self.canvas.image.width())
        self.assertGreaterEqual(f[1], 0.0)

    def test_swap_key_exchanges_rear_and_front(self):
        rear, front = list(self.canvas.moto.rear), list(self.canvas.moto.front)
        self.press_key(Qt.Key.Key_S)
        self.assertEqual(self.canvas.moto.rear, front)
        self.assertEqual(self.canvas.moto.front, rear)

    def test_visibility_cycles_visible_occluded_outside(self):
        self.canvas.sel_point = "front"
        seen = [self.canvas.moto.front[2]]
        for _ in range(3):
            self.press_key(Qt.Key.Key_V)
            seen.append(self.canvas.moto.front[2])
        self.assertEqual(seen, [VIS_VISIBLE, VIS_OCCLUDED, VIS_OUTSIDE, VIS_VISIBLE])

    def test_add_moto_with_key_n_and_delete_it_with_del(self):
        self.press_key(Qt.Key.Key_N)
        self.assertTrue(self.canvas.add_mode)
        a = self.canvas.to_screen(20, 20)
        b = self.canvas.to_screen(90, 70)
        self.drag(QPoint(round(a.x()), round(a.y())), QPoint(round(b.x()), round(b.y())))
        self.assertEqual(len(self.canvas.entry.motos), 2)
        new = self.canvas.moto
        self.assertTrue(new.manual)
        self.assertTrue(new.bbox[0] < new.rear[0] < new.front[0] < new.bbox[2])
        self.assertFalse(self.canvas.add_mode)
        self.press_key(Qt.Key.Key_Delete)
        self.assertEqual(len(self.canvas.entry.motos), 1)

    def test_escape_cancels_add_mode(self):
        self.press_key(Qt.Key.Key_N)
        self.press_key(Qt.Key.Key_Escape)
        self.assertFalse(self.canvas.add_mode)

    def test_enter_accepts_and_advances_navigation_keys_work(self):
        self.press_key(Qt.Key.Key_Return)
        self.assertTrue(self.win.project.entries[0].reviewed)
        self.assertEqual(self.win.index, 1)
        self.assertTrue(self.win.listing.item(0).text().startswith("✔"))
        self.assertTrue(self.win.status_label.text().startswith("2 / 3 · revisadas 1"))
        self.press_key(Qt.Key.Key_Left)
        self.assertEqual(self.win.index, 0)
        self.press_key(Qt.Key.Key_Right)
        self.press_key(Qt.Key.Key_Right)
        self.press_key(Qt.Key.Key_Right)                                    # no pasa del final
        self.assertEqual(self.win.index, 2)

    def test_zoom_keeps_the_point_under_the_cursor(self):
        pos = QPointF(300, 200)
        before = self.canvas.to_image(pos)
        ev = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        scale0 = self.canvas.scale
        self.canvas.wheelEvent(ev)
        self.assertGreater(self.canvas.scale, scale0)
        after = self.canvas.to_image(pos)
        self.assertAlmostEqual(before.x(), after.x(), places=6)
        self.assertAlmostEqual(before.y(), after.y(), places=6)

    def test_pan_with_background_drag(self):
        off0 = QPointF(self.canvas.offset)
        self.drag(QPoint(5, 5), QPoint(45, 35))
        self.assertAlmostEqual(self.canvas.offset.x() - off0.x(), 40, delta=1)
        self.assertAlmostEqual(self.canvas.offset.y() - off0.y(), 30, delta=1)

    def test_save_and_resume_where_it_was_left(self):
        self.press_key(Qt.Key.Key_Return)                                   # revisa la 0
        self.canvas.sel_point = "rear"
        self.press_key(Qt.Key.Key_V)                                        # foto 1: trasera oculta
        self.assertTrue(self.win.project.dirty)
        self.assertTrue(self.win.save())
        self.assertFalse(self.win.project.dirty)
        from tools.label_wheels import LabelWindow
        other = LabelWindow()
        other.open_folder(self.root, blocking=True)
        try:
            self.assertEqual(other.index, 1)                                # primera sin revisar
            self.assertEqual(other.project.entries[1].motos[0].rear[2], VIS_OCCLUDED)
            self.assertTrue(other.project.entries[0].reviewed)
        finally:
            other.project.dirty = False
            other.close()

    def click_dialog_button(self, text: str) -> None:
        def click():
            box = QApplication.activeModalWidget()
            for b in box.buttons():
                if b.text() == text:
                    b.click()
                    return
            box.close()
        QTimer.singleShot(150, click)

    def test_unsaved_changes_dialog_has_save_discard_cancel(self):
        self.canvas.swap_wheels()                                           # deja cambios sin guardar
        self.assertTrue(self.win.project.dirty)
        self.click_dialog_button("Cancelar")
        self.assertFalse(self.win._confirm_leave())
        self.assertTrue(self.win.project.dirty)
        self.click_dialog_button("Descartar")
        self.assertTrue(self.win._confirm_leave())
        self.assertFalse(self.win.project.work_file.exists())
        self.click_dialog_button("Guardar")
        self.assertTrue(self.win._confirm_leave())
        self.assertTrue(self.win.project.work_file.exists())
        self.assertFalse(self.win.project.dirty)

    def test_export_from_the_window_produces_a_dataset(self):
        for _ in range(3):
            self.press_key(Qt.Key.Key_Return)
        res = self.win.export(self.root / "out", quiet=True)
        self.assertIsNotNone(res)
        self.assertTrue((self.root / "out" / "dataset.yaml").exists())
        self.assertTrue(res.zip_path.exists())

    def test_export_without_reviewed_images_is_refused_with_a_message(self):
        shown = []
        self.win._error = shown.append
        self.assertIsNone(self.win.export(self.root / "out", quiet=True))
        self.assertEqual(len(shown), 1)
        self.assertIn("revisadas", shown[0])

    def test_empty_folder_shows_a_message(self):
        with tempfile.TemporaryDirectory() as empty:
            self.win.project.dirty = False
            self.win.open_folder(empty, every_seconds=1.0, blocking=True)
            self.assertIsNone(self.canvas.image)
            self.assertIn("no tiene fotos", self.canvas.message)


if __name__ == "__main__":
    unittest.main()

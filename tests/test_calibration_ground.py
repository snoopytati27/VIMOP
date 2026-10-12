"""Calibración: puntos del piso (homografía) dibujados y guardados desde la ventana, sin pantalla."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from tests.synthetic import moto, test_config, wheels  # noqa: E402

try:
    from PySide6.QtCore import QPoint, QTimer, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QMessageBox
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False

from analysis.parking_analyzer import WARN_APPROX, ParkingAnalyzer  # noqa: E402
from analysis.parking_spaces import (ParkingLayout, ParkingSpace, ground_quad_error,  # noqa: E402
                                     parallelogram_angle)

QUAD = [(100, 100), (300, 100), (340, 400), (60, 400)]                   # trapecio: un rectángulo visto en perspectiva


class GroundQuadValidationTests(unittest.TestCase):
    def test_valid_quads_in_either_direction(self):
        self.assertIsNone(ground_quad_error(QUAD))
        self.assertIsNone(ground_quad_error(QUAD[::-1]))
        self.assertIsNone(ground_quad_error(QUAD[2:] + QUAD[:2]))         # empezando por cualquier esquina

    def test_rejects_crossed_collinear_and_tiny(self):
        self.assertIn("convexo", ground_quad_error([(100, 100), (300, 400), (300, 100), (100, 400)]))    # "corbatín"
        self.assertIn("convexo", ground_quad_error([(0, 0), (100, 0), (200, 0), (300, 0)]))              # alineados
        self.assertIn("pequeño", ground_quad_error([(0, 0), (10, 0), (10, 10), (0, 10)]))
        self.assertIn("4 puntos", ground_quad_error([(0, 0), (100, 0), (100, 100)]))


class ParallelogramGroundTests(unittest.TestCase):
    """Puesto en romboide como referencia del piso: lados 1.0 × 2.0 m y ángulo de 60° en la esquina 1."""
    W, L, ANGLE = 1.0, 2.0, 60.0

    def ground_corners(self):
        t = np.radians(self.ANGLE)
        p4 = (self.L * np.cos(t), self.L * np.sin(t))
        return np.array([(0, 0), (self.W, 0), (self.W + p4[0], p4[1]), p4], dtype=np.float64)

    def image_corners(self):
        """El paralelogramo del piso visto por una cámara inclinada (homografía cualquiera)."""
        import cv2
        h = np.array([[210.0, 40.0, 100.0], [15.0, 120.0, 60.0], [0.0004, 0.0021, 1.0]])
        pts = cv2.perspectiveTransform(self.ground_corners().reshape(-1, 1, 2), h).reshape(-1, 2)
        return [tuple(p) for p in pts]

    def test_angle_from_the_diagonals(self):
        d13, d24 = np.sqrt(7.0), np.sqrt(3.0)                    # d13² = 1+4+2·2·cos60 ; d24² = 1+4−2·2·cos60
        self.assertAlmostEqual(parallelogram_angle(self.W, self.L, d13), 60.0, places=6)
        self.assertAlmostEqual(parallelogram_angle(self.W, self.L, None, d24), 60.0, places=6)
        self.assertAlmostEqual(parallelogram_angle(self.W, self.L, d13, d24), 60.0, places=6)
        self.assertEqual(parallelogram_angle(self.W, self.L), 90.0)                   # sin diagonales: rectángulo
        self.assertAlmostEqual(parallelogram_angle(1.0, 2.0, np.sqrt(5.0)), 90.0, places=6)

    def test_the_stall_measured_in_camera_02(self):
        angle = parallelogram_angle(1.14, 1.94, 2.90, 1.40)
        self.assertAlmostEqual(angle, 43.2, delta=0.1)                                # las diagonales se promedian
        self.assertGreater(parallelogram_angle(1.14, 1.94, None, 1.40), angle)        # 45.4° con la corta
        self.assertLess(parallelogram_angle(1.14, 1.94, 2.90), angle)                 # 40.8° con la larga

    def test_impossible_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            parallelogram_angle(1.0, 2.0, 4.0)                    # la diagonal no puede superar la suma de los lados
        with self.assertRaises(ValueError):
            parallelogram_angle(1.0, 2.0, 0.5)                    # ni ser menor que su diferencia
        with self.assertRaises(ValueError):
            parallelogram_angle(1.0, 2.0, 2.999)                  # casi una línea (≈ 180°)
        with self.assertRaises(ValueError):
            parallelogram_angle(0.0, 2.0, 1.0)

    def make_layout(self, corners):
        layout = ParkingLayout([ParkingSpace("P01", list(QUAD))], (640, 480))
        layout.ground_quad = (list(corners), (self.W, self.L))
        layout.ground_angle = self.ANGLE
        return layout

    def test_homography_maps_the_image_corners_to_the_real_parallelogram(self):
        from analysis.geometry import transform_points
        img = self.image_corners()
        hom = self.make_layout(img).runtime(640, 480).homography
        np.testing.assert_allclose(transform_points(img, hom), self.ground_corners(), atol=1e-4)

    def test_corners_marked_in_the_opposite_direction_are_not_mirrored(self):
        from analysis.geometry import transform_points
        img = self.image_corners()[::-1]                          # sentido contrario: la esquina 1 pasa a ser la 4 original
        hom = self.make_layout(img).runtime(640, 480).homography
        ground = transform_points(img, hom)
        self.assertAlmostEqual(float(np.linalg.norm(ground[0] - ground[1])), self.W, places=4)   # 1→2 = ancho
        self.assertAlmostEqual(float(np.linalg.norm(ground[1] - ground[2])), self.L, places=4)   # 2→3 = largo
        # sin reflejar: el giro de las esquinas en la vista cenital es el mismo que en la imagen
        turn = lambda p: float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
        self.assertGreater(turn(ground) * turn(np.asarray(img)), 0)

    def test_save_and_reload_keep_the_angle_and_a_rectangle_defaults_to_90(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            self.make_layout(self.image_corners()).save(path)
            self.assertAlmostEqual(ParkingLayout.load(path).ground_angle, self.ANGLE, places=2)
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["ground_quad"]["angle"]                                          # archivo de una versión anterior
            path.write_text(json.dumps(data), encoding="utf-8")
            old = ParkingLayout.load(path)
            self.assertEqual(old.ground_angle, 90.0)
            self.assertIsNotNone(old.ground_quad)
            data["ground_quad"]["angle"] = 3.0                                        # ángulo imposible: no se usa el piso
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(ParkingLayout.load(path).ground_quad)

    def test_a_rectangle_is_still_a_rectangle(self):
        from analysis.parking_spaces import ground_destination
        dst = ground_destination((1.0, 2.0), 90.0, QUAD)
        np.testing.assert_allclose(dst, [(0, 0), (1, 0), (1, 2), (0, 2)], atol=1e-9)


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class GroundCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from calibration import CalibrationDialog
        self.tmp = tempfile.TemporaryDirectory()
        self.save_path = Path(self.tmp.name) / "cal.json"
        self.image = np.full((480, 640, 3), 90, dtype=np.uint8)
        layout = ParkingLayout([ParkingSpace("P01", list(QUAD))], (640, 480))
        self.dlg = CalibrationDialog(self.image, layout, self.save_path, test_config())
        self.dlg.resize(1200, 700)
        self.dlg.show()
        QTest.qWaitForWindowExposed(self.dlg)
        self.canvas = self.dlg.canvas

    def tearDown(self):
        self.dlg.dirty = False
        self.dlg.close()
        self.tmp.cleanup()

    def widget_pos(self, x, y) -> QPoint:
        p = self.canvas.to_widget(x, y)
        return QPoint(round(p.x()), round(p.y()))

    def click_points(self, pts):
        for x, y in pts:
            QTest.mouseClick(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, self.widget_pos(x, y))

    def accept_size_dialog(self, width=1.2, length=2.2):
        def fill():
            box = QApplication.activeModalWidget()
            found = box.findChildren(QDoubleSpinBox)
            found[0].setValue(width)
            found[1].setValue(length)
            box.accept()
        QTimer.singleShot(150, fill)

    # ---- pruebas -------------------------------------------------------------------
    def test_drawing_four_corners_asks_the_real_size_and_stores_the_quad(self):
        self.canvas.clear_ground()
        self.assertIsNone(self.canvas.layout_data.ground_quad)
        self.canvas.start_ground()
        self.assertTrue(self.canvas.ground_mode)
        self.accept_size_dialog(1.2, 2.2)
        self.click_points([(120, 110), (280, 110), (320, 380), (80, 380)])
        quad = self.canvas.layout_data.ground_quad
        self.assertIsNotNone(quad)
        self.assertEqual(quad[1], (1.2, 2.2))
        self.assertEqual(len(quad[0]), 4)
        self.assertFalse(self.canvas.ground_mode)
        self.assertIsNotNone(self.canvas.layout_data.runtime(640, 480).homography)   # ya hay homografía
        self.assertIn("1.2 × 2.2", self.dlg.ground_info.text())
        self.assertTrue(self.dlg.dirty)

    def test_dialog_with_diagonals_stores_a_parallelogram(self):
        self.canvas.clear_ground()
        self.canvas.start_ground()

        def fill():
            box = QApplication.activeModalWidget()
            for spin, value in zip(box.findChildren(QDoubleSpinBox), (1.14, 1.94, 2.90, 1.40)):
                spin.setValue(value)
            box.accept()
        QTimer.singleShot(150, fill)
        self.click_points([(120, 110), (280, 110), (320, 380), (80, 380)])
        self.assertEqual(self.canvas.layout_data.ground_quad[1], (1.14, 1.94))
        self.assertAlmostEqual(self.canvas.layout_data.ground_angle, 43.2, delta=0.1)
        self.assertIn("paralelogramo", self.dlg.ground_info.text())
        self.assertAlmostEqual(self.dlg.canvas.layout_data.to_json()["ground_quad"]["angle"], 43.19, delta=0.01)

    def test_dialog_with_impossible_diagonal_warns_and_keeps_asking(self):
        self.canvas.clear_ground()
        self.canvas.start_ground()
        calls = []

        def fill():
            box = QApplication.activeModalWidget()
            spins = box.findChildren(QDoubleSpinBox)
            first = not calls
            for spin, value in zip(spins, (1.14, 1.94, 9.0 if first else 0.0, 0.0)):   # 1.ª vez: diagonal imposible
                spin.setValue(value)
            if first:
                calls.append(1)
                QTimer.singleShot(150, fill)                                    # el cuadro se vuelve a abrir
            box.accept()
        QTimer.singleShot(150, fill)
        with mock.patch.object(QMessageBox, "warning") as warn:
            self.click_points([(120, 110), (280, 110), (320, 380), (80, 380)])
        self.assertEqual(warn.call_count, 1)
        self.assertIn("diagonales", warn.call_args[0][2])
        self.assertIsNotNone(self.canvas.layout_data.ground_quad)                # la 2.ª vez se aceptó (rectángulo 1.14 × 1.94)
        self.assertEqual(self.canvas.layout_data.ground_angle, 90.0)

    def test_cancelling_the_size_dialog_discards_the_points(self):
        self.canvas.clear_ground()
        self.canvas.start_ground()
        QTimer.singleShot(150, lambda: QApplication.activeModalWidget().reject())
        self.click_points([(120, 110), (280, 110), (320, 380), (80, 380)])
        self.assertIsNone(self.canvas.layout_data.ground_quad)

    def test_crossed_points_are_rejected_with_a_message_and_restart(self):
        self.canvas.clear_ground()
        self.canvas.start_ground()
        with mock.patch.object(QMessageBox, "warning") as warn:
            self.click_points([(120, 110), (300, 380), (280, 110), (80, 380)])       # lados cruzados
        self.assertEqual(warn.call_count, 1)
        self.assertIn("convexo", warn.call_args[0][2])
        self.assertIsNone(self.canvas.layout_data.ground_quad)
        self.assertEqual(self.canvas.ground_current, [])                             # vuelve a empezar
        self.assertTrue(self.canvas.ground_mode)

    def test_backspace_and_escape_in_ground_mode(self):
        self.canvas.start_ground()
        self.click_points([(120, 110), (280, 110)])
        self.assertEqual(len(self.canvas.ground_current), 2)
        QTest.keyClick(self.canvas, Qt.Key.Key_Backspace)
        self.assertEqual(len(self.canvas.ground_current), 1)
        QTest.keyClick(self.canvas, Qt.Key.Key_Escape)
        self.assertFalse(self.canvas.ground_mode)
        self.assertEqual(self.canvas.ground_current, [])
        self.assertTrue(self.dlg.isVisible())                                         # Esc no cerró la ventana

    def test_g_key_starts_ground_mode(self):
        self.canvas.setFocus()
        QTest.keyClick(self.canvas, Qt.Key.Key_G)
        self.assertTrue(self.canvas.ground_mode)

    def test_using_the_selected_space_as_ground_asks_the_same_size_dialog(self):
        """Dibujar (o elegir) un puesto y usarlo como referencia: un solo paso, sin marcar 4 puntos sueltos."""
        self.canvas.layout_data.add(ParkingSpace("P99", [(120, 110), (280, 110), (320, 380), (80, 380)]))
        self.dlg._refresh()
        self.canvas.select("P99")

        def fill():
            box = QApplication.activeModalWidget()
            for spin, value in zip(box.findChildren(QDoubleSpinBox), (1.14, 1.94, 2.90, 1.40)):
                spin.setValue(value)
            box.accept()
        QTimer.singleShot(150, fill)
        self.dlg._use_selected_as_ground()
        quad = self.canvas.layout_data.ground_quad
        self.assertEqual(quad[0], [(120.0, 110.0), (280.0, 110.0), (320.0, 380.0), (80.0, 380.0)])
        self.assertEqual(quad[1], (1.14, 1.94))
        self.assertAlmostEqual(self.canvas.layout_data.ground_angle, 43.2, delta=0.1)

    def test_using_the_selected_space_without_a_selection_asks_to_pick_one(self):
        self.canvas.select(None)
        with mock.patch.object(QMessageBox, "information") as info:
            self.dlg._use_selected_as_ground()
        self.assertEqual(info.call_count, 1)
        self.assertIsNone(self.canvas.layout_data.ground_quad)

    def test_using_a_space_with_more_than_four_corners_is_rejected(self):
        self.canvas.layout_data.add(ParkingSpace("P98", [(50, 50), (150, 50), (200, 200), (150, 300), (50, 300)]))
        self.dlg._refresh()
        self.canvas.select("P98")
        with mock.patch.object(QMessageBox, "warning") as warn:
            self.dlg._use_selected_as_ground()
        self.assertEqual(warn.call_count, 1)
        self.assertIn("4", warn.call_args[0][2])
        self.assertIsNone(self.canvas.layout_data.ground_quad)

    def test_a_corner_can_be_dragged_and_the_homography_follows(self):
        self.canvas.set_ground_quad(QUAD, (1.0, 2.0))
        h0 = self.canvas.layout_data.runtime(640, 480).homography.copy()
        start = self.widget_pos(*QUAD[0])
        end = start + QPoint(-12, 9)
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(self.canvas, end)
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        moved = self.canvas.layout_data.ground_quad[0][0]
        self.assertNotEqual(moved, QUAD[0])
        self.assertFalse(np.allclose(h0, self.canvas.layout_data.runtime(640, 480).homography))
        self.assertEqual(self.canvas.layout_data.ground_quad[1], (1.0, 2.0))          # el tamaño no cambia

    def test_dragging_a_corner_across_the_others_warns(self):
        self.canvas.set_ground_quad(QUAD, (1.0, 2.0))
        start = self.widget_pos(*QUAD[0])
        end = self.widget_pos(320, 390)                                               # más allá de la esquina 3
        with mock.patch.object(QMessageBox, "warning") as warn:
            QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
            QTest.mouseMove(self.canvas, end)
            QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        self.assertEqual(warn.call_count, 1)

    def test_clear_ground_removes_the_homography(self):
        self.canvas.set_ground_quad(QUAD, (1.0, 2.0))
        self.canvas.clear_ground()
        self.assertIsNone(self.canvas.layout_data.ground_quad)
        self.assertIsNone(self.canvas.layout_data.runtime(640, 480).homography)
        self.assertIn("aproximados", self.dlg.ground_info.text())

    def test_paint_with_quad_and_partial_points_does_not_crash(self):
        self.canvas.set_ground_quad(QUAD, (1.0, 2.0))
        self.assertFalse(self.canvas.grab().isNull())
        self.canvas.start_ground()
        self.click_points([(120, 110), (280, 110)])
        self.assertFalse(self.canvas.grab().isNull())

    def test_save_reload_and_the_analyzer_uses_it(self):
        self.canvas.set_ground_quad(QUAD, (1.0, 2.0))
        with mock.patch.object(QMessageBox, "information"):
            self.dlg._save()
        self.assertTrue(self.save_path.exists())
        loaded = ParkingLayout.load(self.save_path)
        self.assertEqual(loaded.ground_quad[1], (1.0, 2.0))
        rt = loaded.runtime(640, 480)
        self.assertIsNotNone(rt.homography)
        an = ParkingAnalyzer(test_config())
        # línea sobre el borde izquierdo del trapecio (recta en el piso, torcida en la imagen)
        a, b = np.array(QUAD[0], float), np.array(QUAD[3], float)
        r, f = a + 0.1 * (b - a) + [4, 0], a + 0.9 * (b - a) + [4, 0]
        res = an.analyze(moto(60, 100, 340, 400), rt, wheels=wheels(tuple(r), tuple(f)))
        self.assertTrue(res.ground_plane)
        self.assertNotIn(WARN_APPROX, res.warnings)
        self.assertLess(res.angle_error, 0.5)

    # ---- zoom y desplazamiento -----------------------------------------------------------
    def wheel(self, pos, delta):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QWheelEvent
        p = QPointF(pos)
        self.canvas.wheelEvent(QWheelEvent(p, p, QPoint(0, 0), QPoint(0, delta), Qt.MouseButton.NoButton,
                                           Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))

    def test_wheel_zooms_keeping_the_point_under_the_cursor_and_is_bounded(self):
        pos = QPoint(400, 300)
        before = self.canvas.to_image(pos)
        self.wheel(pos, 120)
        self.assertGreater(self.canvas.zoom, 1.0)
        after = self.canvas.to_image(pos)
        self.assertAlmostEqual(before[0], after[0], delta=0.5)
        self.assertAlmostEqual(before[1], after[1], delta=0.5)
        for _ in range(60):
            self.wheel(pos, 120)
        self.assertEqual(self.canvas.zoom, self.canvas.ZOOM_MAX)
        for _ in range(120):
            self.wheel(pos, -120)
        self.assertEqual(self.canvas.zoom, self.canvas.ZOOM_MIN)                      # no se aleja más que "todo visible"

    def test_points_marked_while_zoomed_land_where_the_cursor_is(self):
        self.canvas.clear_ground()
        self.wheel(QPoint(400, 300), 120)
        self.wheel(QPoint(400, 300), 120)
        self.canvas.start_ground()
        target = QPoint(420, 310)
        expected = self.canvas.to_image(target)
        QTest.mouseClick(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
        x, y = self.canvas.ground_current[0]
        self.assertAlmostEqual(x, expected[0], delta=0.6)
        self.assertAlmostEqual(y, expected[1], delta=0.6)

    def test_middle_button_pans_and_key_0_resets(self):
        self.wheel(QPoint(400, 300), 120)
        self.wheel(QPoint(400, 300), 120)
        pan0 = (self.canvas.pan.x(), self.canvas.pan.y())
        QTest.mousePress(self.canvas, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(300, 300))
        QTest.mouseMove(self.canvas, QPoint(340, 320))
        QTest.mouseRelease(self.canvas, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(340, 320))
        self.assertNotEqual(pan0, (self.canvas.pan.x(), self.canvas.pan.y()))
        QTest.keyClick(self.canvas, Qt.Key.Key_0)
        self.assertEqual((self.canvas.zoom, self.canvas.pan.x(), self.canvas.pan.y()), (1.0, 0.0, 0.0))

    def test_left_drag_on_the_background_pans_but_drawing_still_adds_points(self):
        self.canvas.clear_ground()
        self.wheel(QPoint(400, 300), 120)
        self.wheel(QPoint(400, 300), 120)
        empty = QPoint(30, 40)                                                        # fuera de cualquier puesto
        self.assertIsNone(self.canvas._space_at(empty))
        pan0 = (self.canvas.pan.x(), self.canvas.pan.y())
        QTest.mousePress(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, empty)
        QTest.mouseMove(self.canvas, empty + QPoint(30, 25))
        QTest.mouseRelease(self.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, empty + QPoint(30, 25))
        self.assertNotEqual(pan0, (self.canvas.pan.x(), self.canvas.pan.y()))
        self.canvas.start_ground()
        pan1 = (self.canvas.pan.x(), self.canvas.pan.y())
        self.click_points([(200, 200)])
        self.assertEqual(len(self.canvas.ground_current), 1)                          # dibujando: el clic agrega punto
        self.assertEqual(pan1, (self.canvas.pan.x(), self.canvas.pan.y()))

    def test_new_image_resets_the_zoom(self):
        self.wheel(QPoint(400, 300), 120)
        self.canvas.set_image(self.image)
        self.assertEqual(self.canvas.zoom, 1.0)

    def test_reopening_keeps_the_ground_quad(self):
        from calibration import CalibrationDialog
        lay = ParkingLayout([ParkingSpace("P01", list(QUAD))], (640, 480))
        lay.ground_quad = (list(QUAD), (1.0, 2.0))
        dlg = CalibrationDialog(self.image, lay, self.save_path, test_config())
        try:
            self.assertEqual(dlg.canvas.layout_data.ground_quad[1], (1.0, 2.0))
            self.assertIn("1 × 2", dlg.ground_info.text())
        finally:
            dlg.dirty = False
            dlg.close()


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class StandaloneCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_opens_on_a_photo_and_continues_from_an_existing_file(self):
        import calibration
        from core.image_io import write_image
        with tempfile.TemporaryDirectory() as tmp:
            photo, out = Path(tmp) / "foto ñ.jpg", Path(tmp) / "cal.json"
            write_image(photo, np.full((300, 500, 3), 120, dtype=np.uint8))
            dlg = calibration.open_standalone(photo, out, test_config())
            self.assertEqual((dlg.canvas.img_w, dlg.canvas.img_h), (500, 300))
            self.assertTrue(dlg.canvas.layout_data.is_empty())
            dlg.canvas.layout_data.add(ParkingSpace("P01", [(50, 50), (150, 50), (150, 250), (50, 250)]))
            dlg.canvas.set_ground_quad([(50, 50), (150, 50), (150, 250), (50, 250)], (1.0, 2.0))
            with mock.patch.object(QMessageBox, "information"):
                dlg._save()
            dlg.dirty = False
            dlg.close()
            again = calibration.open_standalone(photo, out, test_config())                 # continúa donde quedó
            self.assertEqual([sp.space_id for sp in again.canvas.layout_data.spaces], ["P01"])
            self.assertEqual(again.canvas.layout_data.ground_quad[1], (1.0, 2.0))
            again.dirty = False
            again.close()

    def test_missing_or_broken_image_gives_a_clear_error(self):
        import calibration
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                calibration.load_reference_image(Path(tmp) / "no_existe.jpg")
            bad = Path(tmp) / "roto.jpg"
            bad.write_bytes(b"no es una imagen")
            with self.assertRaises(ValueError):
                calibration.load_reference_image(bad)
            self.assertEqual(calibration.main(["--imagen", str(Path(tmp) / "no_existe.jpg")]), 1)


if __name__ == "__main__":
    unittest.main()

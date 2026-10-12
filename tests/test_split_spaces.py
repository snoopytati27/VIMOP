"""Dividir un puesto en N tramos iguales (uno por moto), con perspectiva."""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from analysis.geometry import angular_error_deg  # noqa: E402
from analysis.parking_spaces import ParkingLayout, ParkingSpace, clean_polygon_points, split_space_points  # noqa: E402

try:
    from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
    HAVE_QT = True
except ImportError:                                                     # pragma: no cover
    HAVE_QT = False

STRIP = [(0, 0), (400, 0), (400, 100), (0, 100)]
TRAPEZOID = [(100, 400), (500, 400), (380, 150), (220, 150)]              # un rectángulo del piso visto en perspectiva


def area(pts) -> float:
    return Polygon(pts).area


class SplitFunctionTests(unittest.TestCase):
    def test_equal_parts_of_a_rectangle_cover_the_original_without_overlap(self):
        for n in (2, 3, 4, 5):
            parts = split_space_points(STRIP, n, 0.0)
            self.assertEqual(len(parts), n)
            self.assertAlmostEqual(sum(area(p) for p in parts), area(STRIP), places=3)
            for p in parts:
                self.assertAlmostEqual(area(p), area(STRIP) / n, places=3)
            for a, b in zip(parts, parts[1:]):
                self.assertAlmostEqual(Polygon(a).intersection(Polygon(b)).area, 0.0, places=3)

    def test_cuts_run_across_the_long_direction_whichever_way_the_corners_were_clicked(self):
        vertical = [(0, 0), (100, 0), (100, 400), (0, 400)]               # franja vertical
        parts = split_space_points(vertical, 2, 90.0)
        self.assertAlmostEqual(area(parts[0]), 100 * 200, places=3)
        np.testing.assert_allclose(Polygon(parts[0]).bounds, (0.0, 0.0, 100.0, 200.0), atol=1e-6)
        for start in range(4):                                             # empezando por cualquier esquina
            rotated = STRIP[start:] + STRIP[:start]
            parts = split_space_points(rotated, 2, 0.0)
            self.assertAlmostEqual(area(parts[0]), 200 * 100, places=3)
            x1, _, x2, _ = Polygon(parts[0]).bounds
            self.assertAlmostEqual(x2 - x1, 200.0, places=3)

    def test_perspective_far_parts_are_smaller_and_the_cut_is_at_the_real_middle(self):
        # el corte real por la mitad de un rectángulo en perspectiva pasa por el cruce de sus diagonales
        parts = split_space_points(TRAPEZOID, 2, 90.0)
        cut_a, cut_b = np.array(parts[0][1], float), np.array(parts[0][2], float)
        p1, p2, p3, p4 = (np.array(p, float) for p in TRAPEZOID)
        d1, d2 = p3 - p1, p4 - p2
        t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        center = p1 + t[0] * d1
        v, w = cut_b - cut_a, center - cut_a
        cross = v[0] * w[1] - v[1] * w[0]
        self.assertAlmostEqual(float(cross), 0.0, delta=1e-3 * np.linalg.norm(v))
        near, far = (parts[0], parts[1]) if area(parts[0]) > area(parts[1]) else (parts[1], parts[0])
        self.assertGreater(area(near), area(far))                          # el tramo lejano se ve más chico

    def test_axis_angle_chooses_the_cutting_direction(self):
        square_ish = [(0, 0), (300, 0), (300, 280), (0, 280)]
        along_x = split_space_points(square_ish, 2, 0.0)
        along_y = split_space_points(square_ish, 2, 90.0)
        np.testing.assert_allclose(Polygon(along_x[0]).bounds, (0.0, 0.0, 150.0, 280.0), atol=1e-6)
        np.testing.assert_allclose(Polygon(along_y[0]).bounds, (0.0, 0.0, 300.0, 140.0), atol=1e-6)

    def test_duplicate_closing_vertex_is_removed_so_five_points_still_split(self):
        five = STRIP + [(1.0, 0.5)]                                        # el último cae casi sobre el primero
        self.assertEqual(len(clean_polygon_points(five)), 4)
        self.assertEqual(len(clean_polygon_points(STRIP + [STRIP[-1]])), 4)  # repetido seguido
        self.assertEqual(len(split_space_points(five, 2, 0.0)), 2)
        self.assertEqual(clean_polygon_points(STRIP), [(float(x), float(y)) for x, y in STRIP])   # los buenos no cambian
        self.assertEqual(len(clean_polygon_points([(0, 0), (1, 0), (2, 0)])), 3)                # nunca menos de 3

    def test_loading_a_file_with_the_duplicate_vertex_gives_four_corners(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"spaces": {"P01": {"points": [[207.4, 412.6], [469.5, 310.8], [658.1, 315.5],
                                                                     [415.8, 421.1], [207.4, 411.7]]}}}), encoding="utf-8")
            self.assertEqual(len(ParkingLayout.load(path).get("P01").points), 4)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            split_space_points([(0, 0), (10, 0), (10, 10)], 2)               # no son 4 esquinas
        with self.assertRaises(ValueError):
            split_space_points(STRIP, 1)
        with self.assertRaises(ValueError):
            split_space_points(STRIP, 21)
        with self.assertRaises(ValueError):
            split_space_points([(0, 0), (400, 100), (400, 0), (0, 100)], 2)  # lados cruzados


@unittest.skipUnless(HAVE_QT, "PySide6 no está instalado")
class SplitDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make(self, spaces):
        import tempfile
        from pathlib import Path
        from calibration import CalibrationDialog
        from tests.synthetic import test_config
        tmp = tempfile.TemporaryDirectory()
        layout = ParkingLayout([ParkingSpace(k, list(v)) for k, v in spaces.items()], (640, 480))
        dlg = CalibrationDialog(np.full((480, 640, 3), 90, dtype=np.uint8), layout, Path(tmp.name) / "c.json",
                                test_config())

        def cleanup():
            dlg.dirty = False
            dlg.close()
            tmp.cleanup()
        self.addCleanup(cleanup)
        return dlg

    def test_split_replaces_the_space_by_named_parts_keeping_the_axis(self):
        dlg = self.make({"P04": [(20, 100), (420, 100), (420, 180), (20, 180)]})          # franja horizontal
        dlg.canvas.select("P04")
        parent_angle = dlg.canvas.layout_data.get("P04").geometry().angle
        ids = dlg.canvas.split_selected(5)                                                # tramos de 80 x 80 (cuadrados)
        self.assertEqual(ids, ["P04-1", "P04-2", "P04-3", "P04-4", "P04-5"])
        self.assertIsNone(dlg.canvas.layout_data.get("P04"))
        self.assertEqual(dlg.canvas.selected, "P04-1")
        self.assertTrue(dlg.dirty)
        for sid in ids:
            g = dlg.canvas.layout_data.get(sid).geometry()
            self.assertLess(angular_error_deg(g.angle, parent_angle), 1.0)               # cada tramo conserva el eje

    def test_parts_shorter_than_wide_do_not_flip_their_axis(self):
        dlg = self.make({"P01": [(20, 100), (420, 100), (420, 180), (20, 180)]})
        dlg.canvas.select("P01")
        for sid in dlg.canvas.split_selected(8):                                          # tramos de 50 x 80 (más anchos)
            g = dlg.canvas.layout_data.get(sid).geometry()
            self.assertLess(angular_error_deg(g.angle, 0.0), 1.0)
            self.assertAlmostEqual(g.length, 50.0, delta=1.0)                            # L_S: a lo largo de la franja
            self.assertAlmostEqual(g.breadth, 80.0, delta=1.0)

    def test_button_asks_how_many_and_handles_no_selection_and_bad_shapes(self):
        dlg = self.make({"P01": [(20, 100), (420, 100), (420, 180), (20, 180)],
                         "P02": [(20, 250), (200, 250), (300, 300), (200, 350), (20, 350)]})         # 5 vértices
        with mock.patch.object(QMessageBox, "information") as info:
            dlg.canvas.select(None)
            dlg._split()
            dlg.canvas.select("P02")
            dlg._split()
        self.assertEqual(info.call_count, 2)
        dlg.canvas.select("P01")
        with mock.patch.object(QInputDialog, "getInt", return_value=(3, True)):
            dlg._split()
        self.assertEqual(sorted(s.space_id for s in dlg.canvas.layout_data.spaces),
                         ["P01-1", "P01-2", "P01-3", "P02"])
        dlg.canvas.select("P01-1")
        with mock.patch.object(QInputDialog, "getInt", return_value=(2, False)):              # cancelar no cambia nada
            dlg._split()
        self.assertEqual(len(dlg.canvas.layout_data.spaces), 4)

    def test_split_survives_save_and_reload(self):
        dlg = self.make({"P01": [(20, 100), (420, 100), (420, 180), (20, 180)]})
        dlg.canvas.select("P01")
        dlg.canvas.split_selected(2)
        with mock.patch.object(QMessageBox, "information"):
            dlg._save()
        loaded = ParkingLayout.load(dlg.save_path)
        self.assertEqual([s.space_id for s in loaded.spaces], ["P01-1", "P01-2"])
        self.assertEqual(len(loaded.runtime(640, 480).spaces), 2)


if __name__ == "__main__":
    unittest.main()

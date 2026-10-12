"""Pruebas del analizador de parqueo (línea de ejes R–F) con segmentos y polígonos sintéticos."""
from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from tests.synthetic import MANUAL_SPACES, layout_px, moto, rect, test_config, wheels

from analysis.parking_analyzer import (INV_BIG, INV_NONE, INV_SMALL, WARN_APPROX, WARN_UNRELIABLE,
                                       ParkingAnalyzer)
from analysis.parking_spaces import ParkingLayout, ParkingSpace


class AnalyzerTests(unittest.TestCase):
    """P01 x 100–200, P02 x 200–300, y 100–300: puestos verticales (eje a 90°, L_S = 200, W_S = 100)."""

    def setUp(self):
        self.cfg = test_config()
        self.an = ParkingAnalyzer(self.cfg)
        self.layout = layout_px(MANUAL_SPACES)

    def analyze(self, r, f, bbox=(100, 100, 300, 300), layout=None, **kw):
        return self.an.analyze(moto(*bbox), layout or self.layout, wheels=wheels(r, f), **kw)

    # ---- criterios de aceptación ----------------------------------------------
    def test_moto_aligned_with_axis(self):
        a = self.analyze((150, 120), (150, 280), bbox=(110, 110, 190, 290))
        self.assertEqual(a.space_id, "P01")
        self.assertAlmostEqual(a.angle_error, 0.0, places=6)         # error ≈ 0°
        self.assertAlmostEqual(a.orientation, 100.0, places=6)       # O ≈ 100
        self.assertAlmostEqual(a.location, 100.0)
        self.assertAlmostEqual(a.centering, 100.0, places=6)
        self.assertEqual(a.invasion, 100.0)
        self.assertEqual(a.invasion_level, INV_NONE)

    def test_moto_rotated_90_degrees(self):
        a = self.analyze((110, 200), (190, 200), bbox=(105, 150, 195, 250))
        self.assertEqual(a.space_id, "P01")
        self.assertAlmostEqual(a.angle_error, 90.0, places=6)
        self.assertEqual(a.orientation, 0.0)                         # O = 0

    def test_moto_crossing_two_spaces(self):
        a = self.analyze((150, 200), (250, 200), bbox=(140, 150, 260, 250))
        self.assertIn(a.space_id, ("P01", "P02"))
        self.assertAlmostEqual(a.location, 50.0, delta=1)            # U ≈ 50
        self.assertLess(a.invasion, 10.0)                            # I bajo
        self.assertEqual(a.invasion_level, INV_BIG)

    def test_degenerate_moto_seen_from_the_front(self):
        # segmento muy corto respecto al alto del bbox (160): 15 px < 0.35 · 160
        a = self.analyze((150, 200), (150, 215), bbox=(130, 120, 170, 280))
        self.assertFalse(a.orientation_reliable)
        self.assertIsNone(a.orientation)                             # no se inventa un ángulo
        self.assertIsNone(a.angle_error)
        self.assertIn(WARN_UNRELIABLE, a.warnings)
        expected = (a.location * 0.35 + a.centering * 0.20 + a.invasion * 0.10) / 0.65   # pesos renormalizados
        self.assertAlmostEqual(a.score, expected, places=3)

    def test_bbox_only_source_is_never_reliable(self):
        a = self.an.analyze(moto(110, 110, 190, 290), self.layout, wheels=wheels((150, 120), (150, 280), "bbox"))
        self.assertIsNone(a.orientation)
        self.assertEqual(a.wheel_source, "bbox")

    def test_angles_around_0_and_180_have_no_jumps(self):
        lay = layout_px({"H": rect(100, 100, 300, 200)})             # puesto horizontal: eje a 0° ≡ 180°
        prev = None
        for deg in np.arange(-6.0, 6.01, 0.5):
            rad = math.radians(deg)
            r = (200 - 80 * math.cos(rad), 150 - 80 * math.sin(rad))
            f = (200 + 80 * math.cos(rad), 150 + 80 * math.sin(rad))
            a = self.an.analyze(moto(110, 110, 290, 190), lay, wheels=wheels(r, f))
            self.assertAlmostEqual(a.angle_error, abs(deg), places=6)
            if prev is not None:
                self.assertLess(abs(a.orientation - prev), 2.0)      # 0,5° ≈ 1,1 puntos de O: sin saltos
            prev = a.orientation
        flipped = self.an.analyze(moto(110, 110, 290, 190), lay, wheels=wheels((280, 150), (120, 150)))
        self.assertAlmostEqual(flipped.angle_error, 0.0, places=6)   # el eje no tiene sentido: R↔F da igual

    def test_axis_does_not_touch_far_spaces(self):
        # moto pegada al andén, lejos de P02/P03: la línea no toca los puestos lejanos
        lay = layout_px({"P01": rect(100, 100, 200, 300), "P02": rect(200, 100, 300, 300),
                         "P03": rect(300, 100, 400, 300)})
        a = self.analyze((105, 130), (105, 270), bbox=(95, 120, 175, 280), layout=lay)
        self.assertEqual(a.space_id, "P01")
        self.assertEqual(list(a.overlaps), ["P01"])
        self.assertEqual(a.invasion_level, INV_NONE)
        self.assertEqual(a.invaded_spaces, [])

    # ---- otros casos ------------------------------------------------------------
    def test_moto_outside_all_spaces(self):
        a = self.analyze((520, 120), (520, 280), bbox=(500, 100, 540, 300))
        self.assertIsNone(a.space_id)                                # FUERA
        self.assertEqual((a.orientation, a.location, a.centering), (0.0, 0.0, 0.0))
        self.assertEqual(a.classification, "PÉSIMO")

    def test_min_assign_threshold(self):
        # solo el 20 % de la línea cae en P01 (< 30 %) -> fuera de puestos
        a = self.analyze((60, 200), (140, 200), bbox=(50, 150, 150, 250))
        self.assertAlmostEqual(a.overlaps["P01"], 0.5)               # 100..140 de 60..140
        b = self.analyze((20, 200), (100 + 20, 200), bbox=(10, 150, 130, 250))
        self.assertAlmostEqual(b.overlaps["P01"], 0.2)
        self.assertIsNone(b.space_id)

    def test_invasion_levels(self):
        small = self.analyze((125, 200), (205, 200), bbox=(120, 150, 210, 250))    # 5/80 = 6,25 % en P02
        self.assertEqual(small.invasion_level, INV_SMALL)
        self.assertAlmostEqual(small.invasion, 100 * (1 - (5 / 80) / 0.30), delta=0.1)
        big = self.analyze((120, 200), (220, 200), bbox=(110, 150, 230, 250))      # 20 % en P02
        self.assertEqual(big.invasion_level, INV_BIG)

    def test_invasion_uses_max_over_other_spaces(self):
        lay = layout_px({"P01": rect(100, 100, 200, 300), "P02": rect(200, 100, 300, 300),
                         "P03": rect(0, 100, 100, 300)})
        a = self.analyze((80, 200), (230, 200), bbox=(70, 150, 240, 250), layout=lay)   # 20 px en P03, 30 en P02
        self.assertEqual(a.space_id, "P01")
        self.assertAlmostEqual(a.invasion_ratio, 30 / 150)
        self.assertEqual(a.invaded_spaces, ["P02", "P03"])

    def test_centering_formula(self):
        a = self.analyze((150, 150), (170, 290), bbox=(140, 140, 180, 300))
        mx, my = 160.0, 220.0                                        # M; el puesto: centro (150, 200), eje vertical
        dl, dt = my - 200.0, mx - 150.0
        d_n = math.hypot(dl / (200 / 2), dt / (100 / 2))
        self.assertAlmostEqual(a.centering, 100 * (1 - d_n), places=6)
        self.assertEqual(a.center_img, (mx, my))                     # el rótulo va en M = (R + F) / 2

    def test_centering_is_zero_beyond_the_ellipse(self):
        a = self.analyze((101, 101), (101, 103), bbox=(95, 95, 110, 110))     # M en la esquina: d = 1,39 > 1
        self.assertEqual(a.space_id, "P01")
        self.assertEqual(a.centering, 0.0)

    def test_orientation_uses_err_max(self):
        an = ParkingAnalyzer(test_config(ERR_MAX=30.0))
        a = an.analyze(moto(100, 100, 200, 300), self.layout, wheels=wheels((150, 120), (150 + 160 * math.tan(math.radians(15)), 280)))
        self.assertAlmostEqual(a.angle_error, 15.0, places=4)
        self.assertAlmostEqual(a.orientation, 50.0, places=3)

    def test_score_formula_weights(self):
        self.assertAlmostEqual(self.an.score(100, 100, 100, 100), 100.0)
        self.assertAlmostEqual(self.an.score(100, 0, 0, 0), 35.0)
        self.assertAlmostEqual(self.an.score(0, 100, 0, 0), 35.0)
        self.assertAlmostEqual(self.an.score(0, 0, 100, 0), 20.0)
        self.assertAlmostEqual(self.an.score(0, 0, 0, 100), 10.0)
        self.assertAlmostEqual(self.an.score(None, 100, 100, 100), 100.0)      # renormalizado: 0.65 → 1
        self.assertAlmostEqual(self.an.score(None, 65, 0, 0), 35.0)

    def test_warns_when_measuring_in_the_image(self):
        a = self.analyze((150, 120), (150, 280), bbox=(110, 110, 190, 290))
        self.assertIn(WARN_APPROX, a.warnings)
        self.assertFalse(a.ground_plane)


class GroundHomographyTests(unittest.TestCase):
    """Con 4 puntos del piso los ángulos se miden en la vista cenital, no en la imagen."""

    # rectángulo real de 100 x 200 que la cámara ve como un trapecio
    QUAD = [(100, 100), (300, 100), (340, 400), (60, 400)]

    def make_layout(self, with_quad: bool) -> ParkingLayout:
        lay = ParkingLayout([ParkingSpace("P01", list(self.QUAD))], (800, 600))
        if with_quad:
            lay.ground_quad = (list(self.QUAD), (100.0, 200.0))
        return lay

    def test_axis_along_the_floor_edge_has_zero_error_with_homography(self):
        # línea sobre el borde izquierdo del trapecio (recta en el piso, inclinada en la imagen)
        left_a, left_b = np.array(self.QUAD[0], float), np.array(self.QUAD[3], float)
        r = tuple(left_a + 0.1 * (left_b - left_a) + np.array([4.0, 0]))         # 4 px hacia adentro (izq. -> der.)
        f = tuple(left_a + 0.9 * (left_b - left_a) + np.array([4.0, 0]))
        cfg = test_config()
        with_h = ParkingAnalyzer(cfg).analyze(moto(60, 100, 340, 400), self.make_layout(True).runtime(800, 600),
                                              wheels=wheels(r, f))
        without = ParkingAnalyzer(cfg).analyze(moto(60, 100, 340, 400), self.make_layout(False).runtime(800, 600),
                                               wheels=wheels(r, f))
        self.assertTrue(with_h.ground_plane)
        self.assertNotIn(WARN_APPROX, with_h.warnings)
        self.assertIn(WARN_APPROX, without.warnings)
        self.assertLess(with_h.angle_error, 0.5)                     # en el piso va paralela al eje del puesto
        self.assertGreater(without.angle_error, 5.0)                 # en la imagen se ve torcida

    def test_ground_quad_round_trip_and_scaling(self):
        lay = self.make_layout(True)
        data = lay.to_json()
        self.assertEqual(data["ground_quad"]["size"], [100.0, 200.0])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            lay.save(path)
            loaded = ParkingLayout.load(path)
        self.assertEqual(loaded.ground_quad[1], (100.0, 200.0))
        h1 = loaded.runtime(800, 600).homography
        h2 = loaded.runtime(1600, 1200).homography                    # otra resolución, misma escena
        pt = np.array([[[200.0, 250.0]]], dtype=np.float64)
        a = cv2.perspectiveTransform(pt, h1)
        b = cv2.perspectiveTransform(pt * 2, h2)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_invalid_ground_quad_falls_back_to_the_image(self):
        self.assertIsNone(ParkingLayout._parse_ground_quad({"image_points": [[0, 0], [1, 1]], "size": [1, 1]}, "x"))
        self.assertIsNone(ParkingLayout._parse_ground_quad({"image_points": [[0, 0]] * 4, "size": [1, 1]}, "x"))
        self.assertIsNone(ParkingLayout._parse_ground_quad({"image_points": self.QUAD, "size": [0, 5]}, "x"))


class GroundReliabilityTests(unittest.TestCase):
    """Un puesto lejano puede 'explotar' en la vista cenital (lente con distorsión): no debe robarle la
    asignación a un puesto real más cercano. Caso real: config/camara_03.json, calibrado con P01 (26-sep-2026)."""

    # puntos (px) tal como quedaron calibrados en camara_03.json; P04 es cercano, P07/P09 son lejanos
    P01 = [(206.9, 412.4), (434.6, 603.0), (757.0, 603.0), (414.9, 421.7)]
    P04 = [(207.4, 412.6), (469.5, 310.8), (658.1, 315.5), (415.8, 421.1)]
    P07 = [(348.5, 209.9), (486.7, 261.7), (628.9, 264.9), (465.1, 212.6)]
    P09 = [(582.2, 216.2), (762.5, 269.2), (886.2, 274.7), (696.5, 218.9)]
    # rueda trasera/delantera reales, detectadas sobre la moto de etiquetado_ruedas/WhatsApp...15.58.53.jpeg
    REAR, FRONT = (292.6, 370.7), (441.0, 357.2)

    def make_layout(self) -> ParkingLayout:
        lay = ParkingLayout([ParkingSpace(sid, pts) for sid, pts in
                             (("P01", self.P01), ("P04", self.P04), ("P07", self.P07), ("P09", self.P09))],
                            (1322, 604))
        lay.ground_quad = (list(self.P01), (1.14, 1.94))
        lay.ground_angle = 43.19                                   # puesto en romboide, no rectángulo
        return lay

    def test_only_the_space_close_to_the_reference_is_ground_reliable(self):
        rt = self.make_layout().runtime(1322, 604)
        self.assertTrue(rt.by_id["P01"].ground_reliable)           # es el propio puesto de referencia
        for sid in ("P04", "P07", "P09"):                          # se disparan al extrapolar (ver runtime())
            self.assertFalse(rt.by_id[sid].ground_reliable, sid)

    def test_a_motorcycle_over_p04_is_assigned_to_p04_not_to_the_exploded_p09(self):
        an = ParkingAnalyzer(test_config())
        rt = self.make_layout().runtime(1322, 604)
        a = an.analyze(moto(200, 300, 460, 430), rt, wheels=wheels(self.REAR, self.FRONT))
        self.assertEqual(a.space_id, "P04")                        # antes del arreglo salía "P09"
        self.assertFalse(a.ground_plane)                           # P04 no es fiable en el piso: mide en la imagen
        self.assertIn(WARN_APPROX, a.warnings)
        without_ground = an.analyze(moto(200, 300, 460, 430), ParkingLayout(
            [ParkingSpace(sid, pts) for sid, pts in (("P01", self.P01), ("P04", self.P04), ("P07", self.P07),
                                                     ("P09", self.P09))], (1322, 604)).runtime(1322, 604),
            wheels=wheels(self.REAR, self.FRONT))
        self.assertEqual(a.score, without_ground.score)            # mismo resultado que sin piso: P09 no interfiere


if __name__ == "__main__":
    unittest.main()

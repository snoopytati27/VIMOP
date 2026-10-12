"""Escalado de polígonos y compatibilidad calibración / foto (modo foto)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic import moto, rect, test_config, wheels

from analysis.parking_analyzer import ParkingAnalyzer
from analysis.parking_spaces import ParkingLayout, ParkingSpace, make_thumbnail


def scene(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Imagen sintética con estructura (para la comparación de punto de vista)."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (9, 16, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(img, (w // 8, h // 3), (w // 2, h - 10), (255, 255, 255), 6)
    return img


def layout_800x400() -> ParkingLayout:
    return ParkingLayout([ParkingSpace("P01", rect(100, 100, 200, 300)),
                          ParkingSpace("P02", rect(200, 100, 300, 300))], (800, 400))


class PolygonScalingTests(unittest.TestCase):
    def test_polygons_are_scaled_to_the_photo_resolution(self):
        rt = layout_800x400().runtime(1600, 800)                     # misma proporción, doble tamaño
        p01 = rt.by_id["P01"]
        self.assertEqual(p01.shape.bounds, (200.0, 200.0, 400.0, 600.0))
        self.assertAlmostEqual(p01.area, 4 * 100 * 200)
        self.assertEqual(p01.geom.center, (300.0, 400.0))
        self.assertAlmostEqual(p01.aspect, 0.5)                       # la proporción del puesto no cambia

    def test_same_moto_scores_the_same_after_scaling(self):
        an = ParkingAnalyzer(test_config())
        small = an.analyze(moto(110, 115, 190, 295), layout_800x400().runtime(800, 400),
                           wheels=wheels((150, 130), (158, 280)))
        big = an.analyze(moto(220, 230, 380, 590), layout_800x400().runtime(1600, 800),
                         wheels=wheels((300, 260), (316, 560)))
        self.assertEqual(small.space_id, big.space_id)
        self.assertAlmostEqual(small.score, big.score, places=3)

    def test_thumbnail_is_saved_and_loaded(self):
        lay = layout_800x400()
        lay.reference_thumbnail = make_thumbnail(scene(800, 400))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal ñ.json"
            lay.save(path)
            loaded = ParkingLayout.load(path)
        self.assertEqual(loaded.reference_thumbnail, lay.reference_thumbnail)
        self.assertEqual(loaded.reference_size, (800, 400))


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.lay = layout_800x400()
        self.lay.reference_thumbnail = make_thumbnail(scene(800, 400, seed=1))

    def test_same_resolution_ok(self):
        c = self.lay.check_compatibility(scene(800, 400, seed=1))
        self.assertEqual(c.status, "ok")
        self.assertTrue(c.usable)
        self.assertGreater(c.similarity, 0.95)

    def test_same_aspect_other_resolution_is_scaled(self):
        c = self.lay.check_compatibility(scene(1600, 800, seed=1))
        self.assertEqual(c.status, "scaled")
        self.assertTrue(c.usable)

    def test_different_aspect_is_mismatch(self):
        c = self.lay.check_compatibility(scene(800, 600, seed=1))
        self.assertEqual(c.status, "mismatch")
        self.assertFalse(c.usable)
        self.assertIn("proporción", c.message)

    def test_different_viewpoint_is_mismatch(self):
        c = self.lay.check_compatibility(scene(800, 400, seed=7))       # otra "escena"
        self.assertEqual(c.status, "mismatch")
        self.assertIn("PUNTO DE VISTA", c.message)

    def test_brightness_change_is_still_the_same_viewpoint(self):
        darker = cv2.convertScaleAbs(scene(800, 400, seed=1), alpha=0.6, beta=15)
        self.assertTrue(self.lay.check_compatibility(darker).usable)

    def test_old_calibration_without_thumbnail(self):
        self.lay.reference_thumbnail = None
        c = self.lay.check_compatibility(scene(800, 400, seed=5))
        self.assertEqual(c.status, "ok")
        self.assertIn("no se pudo verificar el punto de vista", c.message)

    def test_empty_layout(self):
        self.assertEqual(ParkingLayout([], (800, 400)).check_compatibility(scene(800, 400)).status, "none")


if __name__ == "__main__":
    unittest.main()

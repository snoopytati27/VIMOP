"""Pruebas de geometría básica y de la clasificación con histéresis."""
from __future__ import annotations

import unittest

import numpy as np

from analysis.geometry import (angular_error_deg, axis_angle_deg, bbox_axis, principal_axis, segment_axis_angle_deg,
                               space_polygon, transform_points, wheel_axis_from_mask)
from utils import classify_score, classify_with_hysteresis


class GeometryTests(unittest.TestCase):
    def test_angular_error_is_axis_based(self):
        self.assertAlmostEqual(angular_error_deg(0, 180), 0)
        self.assertAlmostEqual(angular_error_deg(10, 170), 20)
        self.assertAlmostEqual(angular_error_deg(0, 90), 90)
        self.assertAlmostEqual(angular_error_deg(179, 1), 2)

    def test_axis_angle_range(self):
        self.assertAlmostEqual(axis_angle_deg(0, 1), 90)
        self.assertAlmostEqual(axis_angle_deg(0, -1), 90)
        self.assertAlmostEqual(axis_angle_deg(-1, 0), 0)

    def test_principal_axis_of_vertical_cloud(self):
        pts = np.stack([np.random.default_rng(0).normal(0, 1, 500), np.linspace(-50, 50, 500)], axis=1)
        _, v, _, elong = principal_axis(pts)
        self.assertAlmostEqual(axis_angle_deg(v[0], v[1]), 90, delta=2)
        self.assertGreater(elong, 5)

    def test_segment_angle_ignores_direction(self):
        self.assertAlmostEqual(segment_axis_angle_deg((0, 0), (10, 10)), 45)
        self.assertAlmostEqual(segment_axis_angle_deg((10, 10), (0, 0)), 45)
        self.assertAlmostEqual(segment_axis_angle_deg((5, 5), (5, 0)), 90)

    def test_wheel_axis_from_horizontal_mask(self):
        mask = np.zeros((60, 200), dtype=bool)
        mask[10:60, 4:196] = True                          # moto de lado: silueta ancha
        r, f = wheel_axis_from_mask(mask, (100, 50), bottom_band=0.25, inset=0.12)
        self.assertAlmostEqual(segment_axis_angle_deg(r, f), 0, delta=1)
        self.assertLess(r[0], f[0])                        # de izquierda a derecha
        length = f[0] - r[0]
        self.assertAlmostEqual(length, 0.76 * 188, delta=8)             # 76 % de la silueta (12 % por lado)
        self.assertGreater(r[0], 100 + 4 + 0.10 * 192)                 # los extremos están metidos hacia adentro
        self.assertGreater(r[1], 50 + 40)                              # dentro de la franja inferior (y ≥ 75 %)

    def test_wheel_axis_from_diagonal_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        for i in range(100):
            mask[max(0, i - 6):min(100, i + 7), i] = True   # diagonal a 45° (en imagen: y crece hacia abajo)
        axis = wheel_axis_from_mask(mask, (0, 0), bottom_band=1.0, inset=0.0)
        self.assertAlmostEqual(segment_axis_angle_deg(*axis), 45, delta=2)

    def test_wheel_axis_needs_enough_pixels(self):
        self.assertIsNone(wheel_axis_from_mask(np.zeros((10, 10), dtype=bool), (0, 0)))
        tiny = np.zeros((10, 10), dtype=bool)
        tiny[9, 3:5] = True
        self.assertIsNone(wheel_axis_from_mask(tiny, (0, 0)))

    def test_bbox_axis_follows_the_longer_side(self):
        r, f = bbox_axis((0, 0, 100, 40))
        self.assertAlmostEqual(segment_axis_angle_deg(r, f), 0)
        r, f = bbox_axis((0, 0, 40, 100))
        self.assertAlmostEqual(segment_axis_angle_deg(r, f), 90)

    def test_transform_points_identity_and_translation(self):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        np.testing.assert_allclose(transform_points(pts, None), pts)
        h = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(transform_points(pts, h), [(11, -3), (13, -1)])

    def test_self_intersecting_polygon_is_repaired(self):
        poly = space_polygon([(0, 0), (10, 10), (10, 0), (0, 10)])              # "corbatín"
        self.assertTrue(poly.is_valid)
        self.assertGreater(poly.area, 0)


class ClassificationTests(unittest.TestCase):
    def test_cuts(self):
        cases = {100: "EXCELENTE", 90: "EXCELENTE", 89.9: "BUENO", 75: "BUENO", 74.9: "REGULAR",
                 60: "REGULAR", 59.9: "MALO", 40: "MALO", 39.9: "PÉSIMO", 0: "PÉSIMO"}
        for score, name in cases.items():
            self.assertEqual(classify_score(score), name, score)

    def test_hysteresis_keeps_previous_category_near_the_cut(self):
        self.assertEqual(classify_with_hysteresis(73.5, "BUENO", 2.0), "BUENO")
        self.assertEqual(classify_with_hysteresis(72.9, "BUENO", 2.0), "REGULAR")
        self.assertEqual(classify_with_hysteresis(91.0, "BUENO", 2.0), "BUENO")
        self.assertEqual(classify_with_hysteresis(92.0, "BUENO", 2.0), "EXCELENTE")
        self.assertEqual(classify_with_hysteresis(50.0, None, 2.0), "MALO")


if __name__ == "__main__":
    unittest.main()

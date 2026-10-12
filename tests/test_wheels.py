"""WheelLocator: respaldo por máscara, modelo de puntos clave (simulado), suavizado y degradación."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.synthetic import layout_px, moto, rect, test_config

from analysis.geometry import segment_axis_angle_deg
from analysis.parking_analyzer import ParkingAnalyzer
from analysis.wheels import SRC_BBOX, SRC_MASK, SRC_POSE, WheelLocator


def horizontal_moto(track_id=1, bbox=(100, 100, 300, 160)):
    x1, y1, x2, y2 = bbox
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    mask[8:, 4:-4] = True
    det = moto(*bbox, mask=mask)
    det.track_id = track_id
    return det


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakePoseModel:
    """Imita model.predict de YOLO-pose: una moto con sus dos puntos clave."""
    task = "pose"

    def __init__(self, box, rear, front, conf=(0.9, 0.9)):
        self.calls = 0
        self.box, self.kxy, self.kconf = box, [rear, front], list(conf)

    def predict(self, frame, **kw):
        self.calls += 1

        class Res:
            pass

        class Boxes(Res):
            def __len__(self):
                return 1
        r = Res()
        r.boxes = Boxes()
        r.boxes.xyxy = FakeTensor([self.box])
        r.keypoints = Res()
        r.keypoints.xy = FakeTensor([self.kxy])
        r.keypoints.conf = FakeTensor([self.kconf])
        return [r]


class BackendSelectionTests(unittest.TestCase):
    def test_without_model_file_uses_mask_and_says_so(self):
        loc = WheelLocator(test_config(WHEEL_BACKEND="auto", WHEEL_MODEL_PATH="models/no_existe.pt"))
        self.assertFalse(loc.has_model)
        self.assertEqual(loc.description, "Ruedas: estimadas por máscara (aproximado)")

    def test_pose_requested_but_missing_falls_back_to_mask(self):
        loc = WheelLocator(test_config(WHEEL_BACKEND="pose", WHEEL_MODEL_PATH="models/no_existe.pt"))
        self.assertFalse(loc.has_model)
        res = loc.locate(None, horizontal_moto())
        self.assertEqual(res["source"], SRC_MASK)

    def test_corrupt_model_file_falls_back_to_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "wheels_pose.pt"
            bad.write_bytes(b"esto no es un modelo")
            loc = WheelLocator(test_config(WHEEL_BACKEND="auto", WHEEL_MODEL_PATH=str(bad)))
        self.assertFalse(loc.has_model)

    def test_unknown_backend_is_treated_as_auto(self):
        loc = WheelLocator(test_config(WHEEL_BACKEND="raro", WHEEL_MODEL_PATH="models/no_existe.pt"))
        self.assertEqual(loc.requested, "auto")


class MaskBackendTests(unittest.TestCase):
    def setUp(self):
        self.loc = WheelLocator(test_config())

    def test_horizontal_moto_gives_a_horizontal_line_inside_the_bbox(self):
        res = self.loc.locate(None, horizontal_moto())
        self.assertEqual(res["source"], SRC_MASK)
        self.assertAlmostEqual(segment_axis_angle_deg(res["rear"], res["front"]), 0, delta=1)
        self.assertTrue(100 < res["rear"][0] < res["front"][0] < 300)

    def test_no_mask_falls_back_to_bbox_source(self):
        res = self.loc.locate(None, moto(100, 100, 300, 160))
        self.assertEqual(res["source"], SRC_BBOX)

    def test_analyzer_marks_bbox_source_as_unreliable_orientation(self):
        an = ParkingAnalyzer(test_config())
        a = an.analyze(moto(110, 110, 190, 290), layout_px({"P01": rect(100, 100, 200, 300)}))
        self.assertEqual(a.wheel_source, SRC_BBOX)
        self.assertFalse(a.orientation_reliable)
        self.assertIsNone(a.orientation)

    def test_analyzer_end_to_end_with_mask_backend(self):
        # moto horizontal en un puesto horizontal: alineada; en un puesto vertical: atravesada
        an = ParkingAnalyzer(test_config())
        det = horizontal_moto(bbox=(110, 120, 290, 180))
        horizontal = layout_px({"H": rect(100, 100, 300, 200)})
        vertical = layout_px({"V": rect(150, 50, 250, 250)})
        good = an.analyze(det, horizontal)
        self.assertEqual(good.wheel_source, SRC_MASK)
        self.assertLess(good.angle_error, 2.0)
        self.assertGreater(good.orientation, 95.0)
        det2 = horizontal_moto(track_id=2, bbox=(110, 120, 290, 180))
        bad = an.analyze(det2, vertical)
        self.assertGreater(bad.angle_error, 80.0)
        self.assertEqual(bad.orientation, 0.0)


class PoseBackendTests(unittest.TestCase):
    def make(self, model, **cfg):
        loc = WheelLocator(test_config(WHEEL_MODEL_PATH="models/no_existe.pt", **cfg))
        loc.model = model                                        # como si wheels_pose.pt existiera
        return loc

    def test_uses_the_trained_model_when_present(self):
        model = FakePoseModel((100, 100, 300, 160), (120, 150), (280, 150))
        loc = self.make(model)
        self.assertEqual(loc.description, "Ruedas: modelo entrenado")
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        res = loc.locate(frame, horizontal_moto())
        self.assertEqual(res["source"], SRC_POSE)
        np.testing.assert_allclose(res["rear"], (120, 150))
        np.testing.assert_allclose(res["front"], (280, 150))
        self.assertAlmostEqual(res["confidence"], 0.9, places=5)

    def test_one_inference_per_frame_for_all_motos(self):
        model = FakePoseModel((100, 100, 300, 160), (120, 150), (280, 150))
        loc = self.make(model)
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        loc.locate(frame, horizontal_moto(1))
        loc.locate(frame, horizontal_moto(2))
        self.assertEqual(model.calls, 1)
        loc.locate(np.full((300, 400, 3), 9, dtype=np.uint8), horizontal_moto(1))    # otro frame
        self.assertEqual(model.calls, 2)

    def test_low_confidence_keypoint_falls_back_to_mask(self):
        model = FakePoseModel((100, 100, 300, 160), (120, 150), (280, 150), conf=(0.9, 0.1))
        res = self.make(model).locate(np.zeros((300, 400, 3), dtype=np.uint8), horizontal_moto())
        self.assertEqual(res["source"], SRC_MASK)

    def test_pose_of_another_moto_is_not_used(self):
        model = FakePoseModel((500, 400, 700, 460), (520, 450), (680, 450))       # lejos de la moto analizada
        res = self.make(model).locate(np.zeros((600, 800, 3), dtype=np.uint8), horizontal_moto())
        self.assertEqual(res["source"], SRC_MASK)

    def test_without_frame_pose_is_skipped(self):
        model = FakePoseModel((100, 100, 300, 160), (120, 150), (280, 150))
        res = self.make(model).locate(None, horizontal_moto())
        self.assertEqual(res["source"], SRC_MASK)
        self.assertEqual(model.calls, 0)


class SmoothingTests(unittest.TestCase):
    def test_line_jitter_is_reduced(self):
        loc = WheelLocator(test_config(WHEEL_SMOOTH_FRAMES=5))
        rng = np.random.default_rng(3)
        raw, smooth = [], []
        for _ in range(40):
            jr, jf = rng.normal(0, 3, 2), rng.normal(0, 3, 2)
            r, f = np.array([120.0, 150.0]) + jr, np.array([280.0, 150.0]) + jf
            out = loc._smooth(7, {"rear": r, "front": f, "source": SRC_MASK, "confidence": 0.5})
            raw.append(r[1])
            smooth.append(out["rear"][1])
        self.assertLess(np.std(smooth[5:]), 0.7 * np.std(raw[5:]))

    def test_swapped_extremes_do_not_flip_the_line(self):
        loc = WheelLocator(test_config(WHEEL_SMOOTH_FRAMES=5))
        for _ in range(3):
            loc._smooth(1, {"rear": np.array([120.0, 150.0]), "front": np.array([280.0, 150.0]),
                            "source": SRC_MASK, "confidence": 0.5})
        out = loc._smooth(1, {"rear": np.array([280.0, 150.0]), "front": np.array([120.0, 150.0]),
                              "source": SRC_MASK, "confidence": 0.5})
        self.assertLess(out["rear"][0], out["front"][0])
        self.assertAlmostEqual(out["rear"][0], 120.0, delta=1)

    def test_pose_keeps_rear_and_front_semantics(self):
        loc = WheelLocator(test_config(WHEEL_SMOOTH_FRAMES=5))
        loc._smooth(1, {"rear": np.array([120.0, 150.0]), "front": np.array([280.0, 150.0]),
                        "source": SRC_POSE, "confidence": 0.9})
        out = loc._smooth(1, {"rear": np.array([280.0, 150.0]), "front": np.array([120.0, 150.0]),
                              "source": SRC_POSE, "confidence": 0.9})       # la moto dio la vuelta de verdad
        self.assertGreater(out["rear"][0], 150)

    def test_no_track_id_means_no_smoothing(self):
        loc = WheelLocator(test_config())
        res = {"rear": np.array([1.0, 1.0]), "front": np.array([9.0, 1.0]), "source": SRC_MASK, "confidence": 0.5}
        self.assertIs(loc._smooth(None, res), res)

    def test_history_is_bounded(self):
        loc = WheelLocator(test_config())
        for tid in range(1000):
            loc._smooth(tid, {"rear": np.zeros(2), "front": np.ones(2), "source": SRC_MASK, "confidence": 0.5})
        self.assertLessEqual(len(loc._history), 256)


if __name__ == "__main__":
    unittest.main()

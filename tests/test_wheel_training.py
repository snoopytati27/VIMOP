"""Evaluación (pose vs mask), lectura del dataset exportado y entrenamiento de humo (opcional, lento)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.synthetic import test_config
from tests.test_wheel_dataset import scene

from core.image_io import write_image
from tools.eval_wheels import GroundTruth, MethodStats, best_match, format_table, load_ground_truth
from tools.train_wheels import resolve_dataset
from tools.wheel_dataset import ImageEntry, MotoLabel, WheelProject, export_dataset


def gt(rear=(100.0, 100.0), front=(200.0, 100.0), bbox=(80.0, 60.0, 220.0, 130.0)) -> GroundTruth:
    return GroundTruth(bbox, np.array(rear), np.array(front))


class MethodStatsTests(unittest.TestCase):
    def test_perfect_prediction_has_zero_error(self):
        s = MethodStats("x")
        s.add(gt(), np.array([100.0, 100.0]), np.array([200.0, 100.0]))
        out = s.summary()
        self.assertEqual((out["error_rueda_trasera_px"], out["error_rueda_delantera_px"]), (0.0, 0.0))
        self.assertEqual(out["error_angular_grados"], 0.0)
        self.assertEqual(out["orden_trasera_delantera_correcto_pct"], 100.0)

    def test_errors_in_pixels_normalized_and_angle(self):
        s = MethodStats("x")
        g = gt(bbox=(0.0, 0.0, 30.0, 40.0))                                  # diagonal 50
        s.add(g, np.array([103.0, 104.0]), np.array([200.0, 100.0]))         # trasera a 5 px
        out = s.summary()
        self.assertAlmostEqual(out["error_rueda_trasera_px"], 5.0)
        self.assertAlmostEqual(out["error_rueda_trasera_norm"], 0.1)
        expected_angle = np.degrees(np.arctan2(4, 97))
        self.assertAlmostEqual(out["error_angular_grados"], expected_angle, delta=0.01)

    def test_swapped_wheels_are_paired_by_distance_and_flagged_as_wrong_order(self):
        s = MethodStats("mask")
        s.add(gt(), np.array([200.0, 101.0]), np.array([100.0, 101.0]))     # R y F al revés
        out = s.summary()
        self.assertAlmostEqual(out["error_rueda_trasera_px"], 1.0)
        self.assertEqual(out["orden_trasera_delantera_correcto_pct"], 0.0)

    def test_missing_detections_are_counted(self):
        s = MethodStats("x")
        s.add(gt(), None, None)
        s.add(gt(), np.array([100.0, 100.0]), np.array([200.0, 100.0]))
        out = s.summary()
        self.assertEqual((out["motos"], out["encontradas"]), (2, 1))
        empty = MethodStats("y")
        empty.add(gt(), None, None)
        self.assertIsNone(empty.summary()["error_angular_grados"])
        self.assertIn("—", format_table([empty]))

    def test_best_match_uses_iou(self):
        cands = [((0, 0, 10, 10), "a"), ((80, 60, 220, 130), "b")]
        self.assertEqual(best_match((85, 62, 215, 128), cands)[1], "b")
        self.assertIsNone(best_match((500, 500, 600, 600), cands))


class DatasetRoundTripTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        project = WheelProject.open(root)
        for src in ("video:a", "video:b"):
            for i in range(3):
                rel = f"{src[-1]}{i}.jpg"
                write_image(root / rel, scene(i + (0 if src == "video:a" else 9), 200, 120))
                project.entries.append(ImageEntry(rel, src, reviewed=True, motos=[
                    MotoLabel([20, 30, 180, 100], [50, 90, 2], [150, 92, 2])]))
        export_dataset(project, test_config(), root / "ds")
        return root / "ds"

    def test_ground_truth_is_read_back_in_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.make_dataset(Path(tmp))
            items = load_ground_truth(ds / "dataset.yaml")
            self.assertTrue(items)
            _, motos = items[0]
            self.assertEqual(len(motos), 1)
            m = motos[0]
            np.testing.assert_allclose(m.bbox, (20, 30, 180, 100), atol=0.5)
            np.testing.assert_allclose(m.rear, (50, 90), atol=0.5)
            np.testing.assert_allclose(m.front, (150, 92), atol=0.5)

    def test_relative_dataset_path_is_resolved_once(self):
        # regresión: con una ruta relativa y sin "path:" en el yaml la carpeta se duplicaba
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            ds = self.make_dataset(Path(tmp))
            rel = Path(os.path.relpath(ds / "dataset.yaml", Path.cwd()))
            self.assertFalse(rel.is_absolute())
            self.assertTrue(load_ground_truth(rel))

    def test_export_replaces_a_previous_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = self.make_dataset(root)
            (ds / "images" / "train" / "vieja.jpg").write_bytes(b"x")
            project = WheelProject.open(root)
            for i, src in enumerate(("video:a", "video:b")):
                write_image(root / f"n{i}.jpg", scene(20 + i, 200, 120))
                project.entries.append(ImageEntry(f"n{i}.jpg", src, reviewed=True, motos=[
                    MotoLabel([20, 30, 180, 100], [50, 90, 2], [150, 92, 2])]))
            export_dataset(project, test_config(), ds)
            self.assertFalse((ds / "images" / "train" / "vieja.jpg").exists())
            self.assertTrue((ds / "dataset.yaml").exists())

    def test_occluded_ground_truth_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = WheelProject.open(root)
            for src in ("video:a", "video:b"):
                write_image(root / f"{src[-1]}.jpg", scene(1, 200, 120))
                project.entries.append(ImageEntry(f"{src[-1]}.jpg", src, reviewed=True, motos=[
                    MotoLabel([20, 30, 180, 100], [50, 90, 2], [150, 92, 1])]))       # delantera oculta
            export_dataset(project, test_config(), root / "ds")
            self.assertTrue(all(not motos for _, motos in load_ground_truth(root / "ds" / "dataset.yaml")))

    def test_resolve_dataset_accepts_yaml_folder_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = self.make_dataset(root)
            self.assertEqual(resolve_dataset(str(ds / "dataset.yaml"), root / "w"), ds / "dataset.yaml")
            self.assertEqual(resolve_dataset(str(ds), root / "w"), ds / "dataset.yaml")
            from_zip = resolve_dataset(str(root / "ds.zip"), root / "w")
            self.assertEqual(from_zip.name, "dataset.yaml")
            self.assertTrue((from_zip.parent / "images" / "train").is_dir())
            with self.assertRaises(FileNotFoundError):
                resolve_dataset(str(root / "no_existe.zip"), root / "w")


class CameraAugmentTests(unittest.TestCase):
    def test_fisheye_points_match_the_warped_image(self):
        import torch
        from tools.wheel_training import fisheye_image, fisheye_points
        size = 160
        for k in (0.1, 0.25, 0.4):
            for (px, py) in ((0.25, 0.30), (0.70, 0.40), (0.50, 0.85), (0.15, 0.80)):
                img = torch.zeros(3, size, size)
                cx, cy = int(px * size), int(py * size)
                img[:, cy - 1:cy + 2, cx - 1:cx + 2] = 1.0                     # punto blanco de 3×3 px
                out = fisheye_image(img, k)[0]
                ys, xs = torch.nonzero(out > 0.5 * out.max(), as_tuple=True)
                found = np.array([xs.float().mean().item() + 0.5, ys.float().mean().item() + 0.5]) / size
                expected = fisheye_points(np.array([(cx + 0.5) / size, (cy + 0.5) / size]), k)
                self.assertLess(np.abs(found - expected).max() * size, 1.6, (k, px, py))   # < 1,6 px de 160

    def test_fisheye_keeps_the_center_and_zero_strength_is_identity(self):
        from tools.wheel_training import fisheye_boxes, fisheye_points
        np.testing.assert_allclose(fisheye_points(np.array([0.5, 0.5]), 0.3), [0.5, 0.5], atol=1e-9)
        pts = np.array([[0.1, 0.2], [0.9, 0.7], [0.5, 0.05]])
        np.testing.assert_allclose(fisheye_points(pts, 0.0), pts, atol=1e-9)
        far = fisheye_points(np.array([[0.02, 0.02], [0.98, 0.98]]), 0.4)
        self.assertTrue(((far >= 0.0) & (far <= 1.0)).all())                    # las esquinas no se salen del cuadro
        box = np.array([[0.5, 0.5, 0.2, 0.1]])
        np.testing.assert_allclose(fisheye_boxes(box, 0.0), box, atol=1e-9)
        grown = fisheye_boxes(np.array([[0.5, 0.5, 0.2, 0.1]]), 0.4)            # el centro se agranda
        self.assertGreater(grown[0, 2], 0.2)

    def make_batch(self, size=96):
        import torch
        img = torch.rand(3, 3, size, size)
        return {"img": img.clone(),
                "batch_idx": torch.tensor([0.0, 0.0, 1.0, 2.0]),
                "cls": torch.zeros(4, 1),
                "bboxes": torch.tensor([[0.3, 0.5, 0.3, 0.2], [0.7, 0.3, 0.2, 0.2], [0.5, 0.5, 0.6, 0.4],
                                        [0.2, 0.8, 0.2, 0.2]]),
                "keypoints": torch.tensor([[[0.2, 0.5, 2.0], [0.4, 0.5, 2.0]], [[0.65, 0.3, 2.0], [0.0, 0.0, 0.0]],
                                           [[0.3, 0.5, 2.0], [0.7, 0.5, 1.0]], [[0.15, 0.8, 2.0], [0.25, 0.8, 2.0]]])}

    def test_strength_zero_changes_nothing(self):
        import torch
        from tools.wheel_training import camera_augment_batch
        batch = self.make_batch()
        ref = {k: v.clone() for k, v in batch.items()}
        camera_augment_batch(batch, 0.0)
        for k in ref:
            self.assertTrue(torch.equal(batch[k], ref[k]), k)

    def test_full_strength_keeps_shapes_ranges_and_hidden_points(self):
        import random
        import torch
        from tools.wheel_training import camera_augment_batch
        random.seed(3)
        torch.manual_seed(3)
        moved = False
        for _ in range(20):
            batch = self.make_batch()
            ref = {k: v.clone() for k, v in batch.items()}
            camera_augment_batch(batch, 5.0)                                   # todas las imágenes, todos los aumentos
            self.assertEqual(batch["img"].shape, ref["img"].shape)
            self.assertTrue(bool(((batch["img"] >= 0) & (batch["img"] <= 1)).all()))
            self.assertEqual(batch["bboxes"].shape, ref["bboxes"].shape)
            self.assertTrue(bool((batch["bboxes"] >= 0).all()) and bool((batch["bboxes"][:, :2] <= 1).all()))
            self.assertTrue(torch.equal(batch["keypoints"][1, 1], ref["keypoints"][1, 1]))   # punto fuera de imagen: intacto
            self.assertTrue(torch.equal(batch["keypoints"][..., 2], ref["keypoints"][..., 2]))   # visibilidad intacta
            moved = moved or not torch.equal(batch["keypoints"], ref["keypoints"])
        self.assertTrue(moved, "con fuerza 5 el ojo de pez debió mover algún punto")

    def test_blur_lowres_and_jpeg_keep_shape_and_range(self):
        import torch
        from tools.wheel_training import gaussian_blur, jpeg_roundtrip, low_resolution
        img = torch.rand(3, 64, 80)
        for out in (gaussian_blur(img, 1.5), low_resolution(img, 0.3), jpeg_roundtrip(img, 20)):
            self.assertEqual(out.shape, img.shape)
            self.assertTrue(bool(((out >= -1e-4) & (out <= 1 + 1e-4)).all()))
        self.assertLess(gaussian_blur(img, 2.0).std().item(), img.std().item())            # suaviza
        flat = torch.full((3, 32, 32), 0.4)
        torch.testing.assert_close(gaussian_blur(flat, 2.0), flat)                          # no cambia el brillo medio


class ColabNotebookTests(unittest.TestCase):
    PATH = Path(__file__).resolve().parents[1] / "colab" / "train_wheels_colab.ipynb"

    def setUp(self):
        import json
        self.nb = json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_is_a_valid_v4_notebook(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertGreater(len(self.nb["cells"]), 5)
        for cell in self.nb["cells"]:
            self.assertIn(cell["cell_type"], ("code", "markdown"))
            self.assertIsInstance(cell["source"], list)
            self.assertTrue(all(isinstance(line, str) for line in cell["source"]))
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_code_cells_are_valid_python(self):
        import ast
        for i, cell in enumerate(c for c in self.nb["cells"] if c["cell_type"] == "code"):
            src = "".join(line for line in "".join(cell["source"]).splitlines(keepends=True)
                          if not line.lstrip().startswith(("!", "%")))          # magias de Colab
            try:
                ast.parse(src)
            except SyntaxError as exc:                                          # pragma: no cover
                self.fail(f"celda de código {i}: {exc}")

    def test_mounts_drive_keeps_progress_there_and_resumes(self):
        text = "".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code")
        for needle in ("drive.mount", "resume=True", "project=os.path.join(DRIVE_DIR",
                       "best.pt", "wheels_pose.pt", "zipfile.ZipFile", "imgsz=IMGSZ"):
            self.assertIn(needle, text)

    def test_trainer_matches_the_one_in_the_project(self):
        import inspect
        from tools import wheel_training
        text = "".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code")
        for line in inspect.getsource(wheel_training.make_trainer).splitlines():
            stripped = line.strip()
            if stripped.startswith(("gray = torch.rand", "lum = img[gray]", "img[gray] = lum", "batch = super()")):
                self.assertIn(stripped.replace("gray_prob", "GRAY_PROB"), text)
        self.assertIn("camera_augment_batch(batch, CAMERA_AUG)", text)

    def test_camera_augment_block_is_identical_in_the_notebook(self):
        import re
        from tools import wheel_training
        source = Path(wheel_training.__file__).read_text(encoding="utf-8")
        block = re.search(r"# --- aumentos tipo cámara \(inicio\) ---\n(.*?)# --- aumentos tipo cámara \(fin\) ---",
                          source, re.S).group(1)
        text = "".join("".join(c["source"]) for c in self.nb["cells"] if c["cell_type"] == "code")
        self.assertIn(block.rstrip("\n"), text)                    # si cambia el módulo, hay que actualizar el cuaderno
        self.assertIn("CAMERA_AUG = ", text)


@unittest.skipUnless(os.environ.get("VIMOP_SLOW_TESTS") == "1", "lento: VIMOP_SLOW_TESTS=1 para activarlo")
class SmokeTrainingTests(unittest.TestCase):
    def test_yolo_train_accepts_the_exported_dataset(self):
        from tools.train_wheels import main as train_main
        with tempfile.TemporaryDirectory() as tmp:
            ds = DatasetRoundTripTests().make_dataset(Path(tmp))
            code = train_main(["--data", str(ds / "dataset.yaml"), "--from-scratch", "--epochs", "1", "--imgsz", "64",
                               "--batch", "2", "--workers", "0", "--device", "cpu", "--no-install",
                               "--name", "smoke_test"])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()

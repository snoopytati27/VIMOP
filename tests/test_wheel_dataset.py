"""Dataset de ruedas: extracción de fotogramas, preetiquetado, progreso, partición por fuente y exportación."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic import test_config

from analysis.wheels import WheelLocator
from core.detector import Detection, DetectionBatch
from core.image_io import write_image
from tools.wheel_dataset import (FRAMES_DIR, find_parent_project, merge_project, VIS_OCCLUDED, VIS_OUTSIDE, VIS_VISIBLE, ImageEntry, MotoLabel,
                                 WheelProject, dhash, export_dataset, extract_frames, hamming, is_near_duplicate,
                                 photo_session, preannotate, scan_folder, split_sources, split_sources3,
                                 yolo_pose_line)

RNG = np.random.default_rng(0)


def scene(seed: int, w: int = 160, h: int = 96) -> np.ndarray:
    """Imagen con estructura, distinta para cada semilla."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (6, 10, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(img, (10 + seed % 30, 20), (90, 70), (255, 255, 255), 3)
    return img


def make_video(path: Path, segments: list[tuple[np.ndarray, int]], fps: int = 10) -> None:
    h, w = segments[0][0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for frame, n in segments:
        for _ in range(n):
            vw.write(frame)
    vw.release()


def moto_label(x1=20, y1=30, x2=120, y2=80) -> MotoLabel:
    return MotoLabel([x1, y1, x2, y2], [x1 + 20, y2 - 5, VIS_VISIBLE], [x2 - 20, y2 - 5, VIS_VISIBLE])


class FakeDetector:
    """Devuelve una moto horizontal con máscara, como un modelo -seg."""

    def detect(self, img):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(w * .2), int(h * .3), int(w * .8), int(h * .8)
        mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        mask[6:, 3:-3] = True
        return DetectionBatch(motorcycles=[Detection("motorcycle", "motorcycle", 0.9, (x1, y1, x2, y2), mask, None)])


class FrameExtractionTests(unittest.TestCase):
    def test_hash_and_duplicates(self):
        a, b = scene(1), scene(2)
        self.assertEqual(hamming(dhash(a), dhash(a.copy())), 0)
        self.assertGreater(hamming(dhash(a), dhash(b)), 5)
        small = cv2.resize(a, (64, 36))
        self.assertTrue(is_near_duplicate(a.copy(), [(dhash(a), small)]))
        self.assertFalse(is_near_duplicate(b, [(dhash(a), small)]))
        self.assertFalse(is_near_duplicate(a, []))

    def test_extracts_every_n_seconds_and_drops_near_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "carpeta ñ con espacios"
            folder.mkdir()
            video = folder / "cámara 1.mp4"
            make_video(video, [(scene(1), 50), (scene(2), 50)])          # 10 s a 10 fps: 5 s de una escena, 5 s de otra
            frames = extract_frames(video, folder / FRAMES_DIR / video.stem, every_seconds=1.0)
            self.assertEqual(len(frames), 2)                              # 10 candidatos, solo 2 no duplicados
            self.assertTrue(all(f.exists() for f in frames))
            again = extract_frames(video, folder / FRAMES_DIR / video.stem, every_seconds=1.0)
            self.assertEqual(frames, again)                               # idempotente

    def test_unreadable_video_raises_a_clear_error(self):
        from core.image_io import ImageLoadError
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "roto.mp4"
            bad.write_bytes(b"no es un video")
            with self.assertRaises(ImageLoadError):
                extract_frames(bad, Path(tmp) / "out")


class ScanTests(unittest.TestCase):
    def test_scan_is_cumulative_and_skips_generated_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_image(root / "WhatsApp Image 2026-09-24 at 17.47.41.jpeg", scene(1))
            write_image(root / "sesion_b" / "foto1.jpg", scene(2))
            make_video(root / "v1.mp4", [(scene(3), 30), (scene(4), 30)])
            project = WheelProject.open(root)
            photos, frames = scan_folder(project, every_seconds=1.0)
            self.assertEqual((photos, frames), (2, 2))
            sources = {e.source for e in project.entries}
            self.assertEqual(sources, {"sesion:2026-09-24", "sesion:sesion_b", "video:v1"})
            project.entries[0].reviewed = True
            project.entries[0].motos = [moto_label()]
            project.save()
            again = WheelProject.open(root)
            self.assertEqual(scan_folder(again), (0, 0))                   # no duplica ni pierde etiquetas
            self.assertTrue(again.entries[0].reviewed)
            write_image(root / "nueva.jpg", scene(5))
            self.assertEqual(scan_folder(again)[0], 1)
            self.assertNotIn(FRAMES_DIR, {Path(e.path).parts[0] for e in again.entries if e.source != "video:v1"})

    def test_video_already_extracted_is_not_extracted_again_and_same_names_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "celular").mkdir()
            (root / "camara").mkdir()
            make_video(root / "celular" / "v.mp4", [(scene(1), 30), (scene(2), 30)])
            make_video(root / "camara" / "v.mp4", [(scene(3), 30), (scene(4), 30)])     # mismo nombre, otra carpeta
            project = WheelProject.open(root)
            self.assertEqual(scan_folder(project, every_seconds=1.0), (0, 4))
            self.assertEqual({e.source for e in project.entries}, {"video:celular_v", "video:camara_v"})
            self.assertEqual(scan_folder(project, every_seconds=3.0), (0, 0))            # otro intervalo: no repite
            make_video(root / "celular" / "nuevo.mp4", [(scene(5), 30), (scene(6), 30)])
            self.assertEqual(scan_folder(project, every_seconds=3.0)[1], 2)              # el nuevo sí, con 3 s

    def test_merge_a_subfolder_project_into_the_parent_without_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "celular").mkdir()
            make_video(root / "celular" / "a.mp4", [(scene(1), 30), (scene(2), 30)])
            make_video(root / "celular" / "b.mp4", [(scene(3), 30), (scene(4), 30)])
            sub = WheelProject.open(root / "celular")                     # se abrió la subcarpeta como proyecto
            scan_folder(sub, 1.0)
            for e in sub.entries:
                e.reviewed = True
                e.motos = [moto_label()]
            sub.save()
            parent = WheelProject.open(root)
            n = merge_project(parent, root / "celular", {"video:a": "sesion:celular", "video:b": "sesion:celular"})
            self.assertEqual(n, 4)
            self.assertEqual({e.source for e in parent.entries}, {"sesion:celular"})
            self.assertTrue(all(e.reviewed and e.motos for e in parent.entries))
            self.assertTrue(all((root / e.path).exists() for e in parent.entries))     # las rutas apuntan a los archivos
            self.assertEqual(merge_project(parent, root / "celular"), 0)               # idempotente
            self.assertEqual(scan_folder(parent, 3.0), (0, 0))            # y el escaneo no vuelve a extraer esos videos
            self.assertEqual(len(parent.entries), 4)

    def test_finds_the_parent_project_of_a_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "internet").mkdir()
            self.assertIsNone(find_parent_project(root / "internet"))
            WheelProject.open(root).save()
            self.assertEqual(find_parent_project(root / "internet"), root.resolve())
            self.assertIsNone(find_parent_project(root.parent / "otra"))

    def test_photo_session_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.jpg", "IMG_20260101_120000.jpg", "sub/b.jpg"):
                write_image(root / name, scene(1))
            self.assertEqual(photo_session(root / "IMG_20260101_120000.jpg", root), "sesion:2026-01-01")
            self.assertEqual(photo_session(root / "sub" / "b.jpg", root), "sesion:sub")
            self.assertTrue(photo_session(root / "a.jpg", root).startswith("sesion:20"))


class PreannotationTests(unittest.TestCase):
    def test_preannotates_with_mask_backend_and_keeps_reviewed_untouched(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                write_image(root / f"f{i}.jpg", scene(i, 200, 120))
            project = WheelProject.open(root)
            scan_folder(project)
            project.entries[0].reviewed = True                              # ya revisada por una persona
            project.entries[0].motos = [moto_label()]
            seen = []
            n = preannotate(project, FakeDetector(), WheelLocator(cfg), cfg, progress=lambda i, t: seen.append((i, t)))
            self.assertEqual(n, 2)
            self.assertEqual(seen[-1], (2, 2))
            self.assertEqual(len(project.entries[0].motos), 1)              # la revisada no se tocó
            for e in project.entries[1:]:
                self.assertTrue(e.prelabeled)
                self.assertFalse(e.reviewed)                                # preetiquetada != revisada
                self.assertEqual(len(e.motos), 1)
                m = e.motos[0]
                self.assertEqual((m.rear[2], m.front[2]), (VIS_VISIBLE, VIS_VISIBLE))
                self.assertLess(m.rear[0], m.front[0])
                self.assertTrue(m.bbox[0] < m.rear[0] < m.front[0] < m.bbox[2])
            self.assertEqual(preannotate(project, FakeDetector(), WheelLocator(cfg), cfg), 0)   # ya hechas: nada nuevo


class ProgressTests(unittest.TestCase):
    def test_progress_text_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = WheelProject.open(tmp)
            project.entries = [ImageEntry(f"{i}.jpg", "sesion:x") for i in range(230)]
            for e in project.entries[:30]:
                e.reviewed = True
            self.assertEqual(project.progress_text(44), "45 / 230 · revisadas 30")
            project.entries[3].motos = [moto_label()]
            project.entries[3].motos[0].front[2] = VIS_OCCLUDED
            project.save()
            self.assertFalse(project.dirty)
            back = WheelProject.open(tmp)
            self.assertEqual(back.reviewed_count(), 30)
            self.assertEqual(back.entries[3].motos[0].front[2], VIS_OCCLUDED)

    def test_corrupt_work_file_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "wheels_labels.json").write_text("{ esto no es json", encoding="utf-8")
            with self.assertRaises(ValueError):
                WheelProject.open(tmp)

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            WheelProject.open("C:/no/existe/para/nada")


class YoloFormatTests(unittest.TestCase):
    def test_line_is_normalized_and_has_two_keypoints(self):
        m = MotoLabel([20, 30, 120, 80], [40, 75, VIS_VISIBLE], [100, 75, VIS_OCCLUDED])
        parts = yolo_pose_line(m, 200, 100).split()
        self.assertEqual(len(parts), 5 + 2 * 3)
        vals = [float(p) for p in parts]
        self.assertEqual(vals[0], 0)
        np.testing.assert_allclose(vals[1:5], [0.35, 0.55, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(vals[5:], [0.2, 0.75, 2, 0.5, 0.75, 1], atol=1e-6)

    def test_outside_keypoint_has_no_coordinates_and_values_are_clipped(self):
        m = MotoLabel([-10, 30, 250, 80], [-5, 75, VIS_VISIBLE], [100, 75, VIS_OUTSIDE])
        vals = [float(p) for p in yolo_pose_line(m, 200, 100).split()]
        coords = vals[1:5] + [vals[5], vals[6], vals[8], vals[9]]           # sin las visibilidades
        self.assertTrue(all(0.0 <= v <= 1.0 for v in coords))
        self.assertEqual(vals[-3:], [0.0, 0.0, 0])
        self.assertEqual(vals[5], 0.0)                                      # x = -5 se recorta a 0

    def test_degenerate_bbox_is_dropped(self):
        self.assertIsNone(yolo_pose_line(MotoLabel([10, 10, 11, 50], [10, 10, 2], [10, 20, 2]), 100, 100))


class SplitTests(unittest.TestCase):
    def test_sources_are_never_split(self):
        sources = {f"video:{i}": 10 for i in range(10)}
        train, val = split_sources(sources, 0.2, seed=1)
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(sources))
        self.assertGreaterEqual(sum(sources[s] for s in val), 0.2 * 100)
        self.assertEqual((train, val), split_sources(sources, 0.2, seed=1))   # determinista
        self.assertNotEqual(val, split_sources(sources, 0.2, seed=2)[1])

    def test_internet_photos_never_go_to_validation(self):
        sources = {"sesion:internet": 13, "video:a": 15, "sesion:celular": 16, "sesion:2026-09-24": 2}
        for seed in range(20):
            train, val = split_sources(sources, 0.2, seed=seed)
            self.assertIn("sesion:internet", train)
            self.assertNotIn("sesion:internet", val)
            self.assertTrue(val)
        only_internet_plus_one = split_sources({"sesion:internet": 10, "video:a": 5}, 0.2)
        self.assertEqual(only_internet_plus_one, ({"sesion:internet", "video:a"}, set()))   # no hay con qué validar

    def test_single_source_all_goes_to_train(self):
        train, val = split_sources({"video:a": 40}, 0.2)
        self.assertEqual((train, val), ({"video:a"}, set()))

    def test_two_sources_leave_one_in_each(self):
        train, val = split_sources({"a": 50, "b": 50}, 0.2)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)


class Split3Tests(unittest.TestCase):
    def test_many_sources_approach_70_15_15(self):
        sources = {f"video:{i}": 10 for i in range(10)}
        train, val, test = split_sources3(sources, 0.15, 0.15, seed=1)
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(train | val | test, set(sources))
        share = lambda group: sum(sources[s] for s in group) / 100
        self.assertAlmostEqual(share(train), 0.70, delta=0.06)
        self.assertAlmostEqual(share(val), 0.15, delta=0.06)
        self.assertAlmostEqual(share(test), 0.15, delta=0.06)
        self.assertEqual((train, val, test), split_sources3(sources, 0.15, 0.15, seed=1))   # determinista

    def test_internet_always_train_and_percentages_use_camera_images(self):
        sources = {"sesion:internet": 80, **{f"video:{i}": 10 for i in range(10)}}
        for seed in range(10):
            train, val, test = split_sources3(sources, 0.15, 0.15, seed=seed)
            self.assertIn("sesion:internet", train)
            self.assertTrue(val and test)
            self.assertNotIn("sesion:internet", val | test)
            camera = sum(sources[s] for s in val | test)
            self.assertAlmostEqual(camera / 100, 0.30, delta=0.06)       # sobre las 100 imágenes de la cámara

    def test_four_real_sources_give_a_coarse_but_valid_split(self):
        sources = {"sesion:2026-09-24": 2, "sesion:celular_2026-09-25": 16, "video:a": 15, "video:b": 6,
                   "sesion:internet": 18}
        train, val, test = split_sources3(sources, 0.15, 0.15)
        self.assertTrue(train - {"sesion:internet"} and val and test)
        self.assertEqual(train | val | test, set(sources))

    def test_fewer_than_three_camera_sources_has_no_test(self):
        train, val, test = split_sources3({"a": 50, "b": 50, "sesion:internet": 10}, 0.15, 0.15)
        self.assertEqual(test, set())
        self.assertEqual(len(val), 1)
        self.assertEqual(split_sources3({"video:a": 40}, 0.15, 0.15), ({"video:a"}, set(), set()))

    def test_more_than_ten_sources_still_covers_all_three(self):
        sources = {f"video:{i}": 5 for i in range(14)}
        train, val, test = split_sources3(sources, 0.15, 0.15, seed=3)
        self.assertTrue(train and val and test)
        self.assertEqual(train | val | test, set(sources))


class ExportTests(unittest.TestCase):
    def build(self, root: Path, sources: dict[str, int], reviewed_each: int) -> WheelProject:
        project = WheelProject.open(root)
        seed = 0
        for src, n in sources.items():
            for i in range(n):
                rel = f"{src.replace(':', '_')}_{i}.jpg"
                write_image(root / rel, scene(seed, 160, 96))
                seed += 1
                e = ImageEntry(rel, src, reviewed=i < reviewed_each, motos=[moto_label()])
                project.entries.append(e)
        return project

    def test_export_only_reviewed_split_by_source_and_valid_yaml(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self.build(root, {"video:a": 6, "video:b": 6, "sesion:c": 6}, reviewed_each=4)
            res = export_dataset(project, cfg, root / "salida")
            self.assertEqual(res.train_images + res.val_images + res.test_images, 12)   # solo las 4 + 4 + 4 revisadas
            self.assertEqual((res.train_images, res.val_images, res.test_images), (4, 4, 4))   # 3 fuentes: una a cada uno
            self.assertEqual(res.motos, 12)
            self.assertFalse(set(res.train_sources) & set(res.val_sources) or set(res.train_sources) & set(res.test_sources)
                             or set(res.val_sources) & set(res.test_sources))
            self.assertEqual(sorted(res.train_sources + res.val_sources + res.test_sources),
                             ["sesion:c", "video:a", "video:b"])
            self.assertEqual(res.warnings, [])
            out = root / "salida"
            train_imgs = {p.stem for p in (out / "images" / "train").glob("*.jpg")}
            val_imgs = {p.stem for p in (out / "images" / "val").glob("*.jpg")}
            test_imgs = {p.stem for p in (out / "images" / "test").glob("*.jpg")}
            self.assertFalse(train_imgs & val_imgs or train_imgs & test_imgs or val_imgs & test_imgs)   # ningún fotograma en dos conjuntos
            for kind, stems in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
                labels = {p.stem for p in (out / "labels" / kind).glob("*.txt")}
                self.assertEqual(labels, stems)
            yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8")
            self.assertIn("test: images/test", yaml_text)
            self.assertIn("kpt_shape: [2, 3]", yaml_text)
            self.assertIn("flip_idx: [0, 1]", yaml_text)
            self.assertNotIn("\npath:", "\n" + yaml_text)                     # portátil: usa la carpeta del yaml
            with zipfile.ZipFile(res.zip_path) as zf:
                names = zf.namelist()
            self.assertIn("salida/dataset.yaml", names)
            self.assertTrue(any(n.startswith("salida/images/train/") for n in names))

    def test_two_sources_export_without_test_and_warn(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self.build(root, {"video:a": 5, "video:b": 5}, reviewed_each=5)
            res = export_dataset(project, cfg, root / "salida", make_zip=False)
            self.assertEqual((res.train_images, res.val_images, res.test_images), (5, 5, 0))
            self.assertEqual(len(res.warnings), 1)
            self.assertFalse((root / "salida" / "images" / "test").exists())
            self.assertNotIn("test:", (root / "salida" / "dataset.yaml").read_text(encoding="utf-8"))

    def test_single_source_warns_and_val_equals_train(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self.build(root, {"video:solo": 5}, reviewed_each=5)
            res = export_dataset(project, cfg, root / "salida", make_zip=False)
            self.assertEqual(res.train_images, 5)
            self.assertEqual(res.val_images, 5)
            self.assertEqual(len(res.warnings), 1)
            self.assertIsNone(res.zip_path)

    def test_nothing_reviewed_is_an_error(self):
        cfg = test_config()
        with tempfile.TemporaryDirectory() as tmp:
            project = self.build(Path(tmp), {"video:a": 3}, reviewed_each=0)
            with self.assertRaises(ValueError):
                export_dataset(project, cfg, Path(tmp) / "salida")

    def test_labels_are_json_serializable_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = self.build(Path(tmp), {"video:a": 2}, reviewed_each=1)
            project.save()
            raw = json.loads(project.work_file.read_text(encoding="utf-8"))
            self.assertEqual(len(raw["entries"]), 2)
            self.assertEqual(raw["entries"][0]["motos"][0]["rear"][2], VIS_VISIBLE)


if __name__ == "__main__":
    unittest.main()

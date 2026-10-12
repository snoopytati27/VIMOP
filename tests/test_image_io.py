"""Carga de fotos: rutas Unicode, alfa, grises, EXIF, tamaño máximo, archivos dañados y fuente 'image'."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic import test_config

from core.camera import CameraError, VideoSource
from core.image_io import (ImageLoadError, detect_file_kind, list_images_in_folder, read_image, to_bgr,
                           write_image)


def _save(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    buf.tofile(str(path))


class ImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "fotos con tildes ñandú áéíóú"
        self.dir.mkdir()
        self.img = np.zeros((60, 80, 3), dtype=np.uint8)
        self.img[:, :40] = (0, 0, 255)                      # mitad izquierda roja (BGR)

    def tearDown(self):
        self._tmp.cleanup()

    def test_unicode_path(self):
        path = self.dir / "parqueadero cámara Nº3.jpg"
        _save(path, self.img)
        loaded = read_image(path)
        self.assertEqual(loaded.image.shape, (60, 80, 3))
        self.assertEqual(loaded.image.dtype, np.uint8)
        self.assertGreater(loaded.image[30, 10, 2], 200)     # sigue roja

    def test_png_with_alpha_becomes_bgr(self):
        bgra = np.zeros((40, 40, 4), dtype=np.uint8)
        bgra[:, :, 1] = 200                                   # verde
        bgra[:, :20, 3] = 255                                 # izquierda opaca
        bgra[:, 20:, 3] = 0                                   # derecha transparente
        path = self.dir / "transparente ñ.png"
        _save(path, bgra)
        img = read_image(path).image
        self.assertEqual(img.shape, (40, 40, 3))
        np.testing.assert_array_equal(img[5, 5], [0, 200, 0])           # opaco: su color
        np.testing.assert_array_equal(img[5, 35], [255, 255, 255])      # transparente: fondo blanco

    def test_grayscale_becomes_bgr(self):
        path = self.dir / "gris.png"
        _save(path, np.full((30, 30), 128, dtype=np.uint8))
        img = read_image(path).image
        self.assertEqual(img.shape, (30, 30, 3))
        self.assertTrue((img == 128).all())

    def test_to_bgr_16_bits(self):
        img = to_bgr(np.full((4, 4), 65535, dtype=np.uint16))
        self.assertEqual((img.dtype, img.shape), (np.uint8, (4, 4, 3)))
        self.assertTrue((img == 255).all())

    def test_exif_orientation_is_corrected(self):
        from PIL import Image
        stored = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)          # guardada "de lado"
        im = Image.fromarray(cv2.cvtColor(stored, cv2.COLOR_BGR2RGB))
        exif = im.getexif()
        exif[0x0112] = 6                                                         # "rotar 90° horario"
        path = self.dir / "celular.jpg"
        im.save(path, exif=exif.tobytes(), quality=95)
        loaded = read_image(path)
        self.assertEqual(loaded.exif_orientation, 6)
        self.assertEqual(loaded.image.shape[:2], (60, 80))                       # vuelve a quedar horizontal
        self.assertGreater(loaded.image[30, 10, 2], 180)                         # rojo a la izquierda

    def test_large_image_is_resized(self):
        path = self.dir / "grande.jpg"
        _save(path, np.zeros((1000, 3000, 3), dtype=np.uint8))
        loaded = read_image(path, max_side=1920)
        self.assertTrue(loaded.resized)
        self.assertEqual(loaded.size, (1920, 640))
        self.assertEqual(loaded.original_size, (3000, 1000))
        self.assertTrue(any("reducida" in n for n in loaded.notes()))

    def test_corrupt_and_missing_files_raise_clear_errors(self):
        bad = self.dir / "dañada.jpg"
        bad.write_bytes(b"\xff\xd8 no es una imagen")
        with self.assertRaises(ImageLoadError) as ctx:
            read_image(bad)
        self.assertIn("dañad", str(ctx.exception))
        with self.assertRaises(ImageLoadError):
            read_image(self.dir / "no_existe.png")

    def test_detect_file_kind(self):
        _save(self.dir / "a.jpg", self.img)
        self.assertEqual(detect_file_kind(self.dir / "a.jpg"), "image")
        self.assertEqual(detect_file_kind(self.dir / "clip.mp4"), "video")      # por extensión
        no_ext = self.dir / "foto_sin_extension"
        _save(self.dir / "tmp.png", self.img)
        (self.dir / "tmp.png").rename(no_ext)
        self.assertEqual(detect_file_kind(no_ext), "image")                     # por contenido
        txt = self.dir / "notas.txt"
        txt.write_text("hola", encoding="utf-8")
        self.assertIsNone(detect_file_kind(txt))

    def test_write_image_unicode_and_folder_listing(self):
        out = self.dir / "salida ñ" / "resultado_vimop.jpg"
        self.assertTrue(write_image(out, self.img))
        self.assertTrue(out.exists())
        _save(self.dir / "b.PNG", self.img)
        names = [p.name for p in list_images_in_folder(self.dir)]
        self.assertIn("b.PNG", names)

    def test_video_source_image_mode(self):
        path = self.dir / "foto ñ.jpg"
        _save(path, self.img)
        src = VideoSource("image", str(path), test_config())
        src.start()
        packet = src.read(timeout=1.0)
        src.release()
        self.assertIsNotNone(packet)
        self.assertTrue(packet.static)
        self.assertEqual(packet.origin, "foto:foto ñ.jpg")
        self.assertEqual(packet.frame.shape, (60, 80, 3))
        self.assertTrue(src.finished)
        self.assertIsNone(src.read(timeout=0.05))                 # una foto = un solo frame
        bad = VideoSource("image", str(self.dir / "no_existe.jpg"), test_config())
        with self.assertRaises(CameraError):
            bad.start()


if __name__ == "__main__":
    unittest.main()

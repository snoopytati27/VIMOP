"""Pruebas de placa (validación del formato y votación)."""
from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from tests.synthetic import test_config
from core.registry import MotoRegistry
from identification.plate import FastPlateOCR, PlateCandidate, PlateReader, PlateTextParser, template_from_regex
from utils import NO_PLATE

REGEX = r"^[A-Z]{3}\d{2}[A-Z]$"


def tok(y, x, text, conf=0.9, h=20):
    return ((y, x, h), text, conf)


class PlateParserTests(unittest.TestCase):
    def setUp(self):
        self.p = PlateTextParser(REGEX)

    def test_template_from_regex(self):
        self.assertEqual(template_from_regex(REGEX), "LLLDDL")
        self.assertEqual(template_from_regex(r"^[A-Z]{3}\d{3}$"), "LLLDDD")
        self.assertIsNone(template_from_regex(r"^(ABC|XYZ)\d+$"))

    def test_valid_plate(self):
        self.assertEqual(self.p.fit("ABC12D"), ("ABC12D", 0))

    def test_position_aware_corrections(self):
        self.assertEqual(self.p.fit("A8C12D")[0], "ABC12D")      # 8 en posición de letra -> B
        self.assertEqual(self.p.fit("ABCI2D")[0], "ABC12D")      # I en posición de número -> 1
        self.assertEqual(self.p.fit("ABC1S0")[0], "ABC15O")      # S -> 5 (número), 0 -> O (letra)

    def test_invalid_plates_rejected(self):
        self.assertIsNone(self.p.fit("AB12D"))       # longitud
        self.assertIsNone(self.p.fit("ABC12#"))
        self.assertIsNone(self.p.parse([tok(10, 10, "BUCARAMANGA")]))

    def test_two_line_moto_plate(self):
        # placa de moto: "ABC" arriba, "12D" abajo, ciudad debajo
        tokens = [tok(60, 50, "12D", 0.8), tok(20, 50, "ABC", 0.9), tok(95, 50, "BUCARAMANGA", 0.5, h=10)]
        text, conf, _ = self.p.parse(tokens)
        self.assertEqual(text, "ABC12D")
        self.assertGreater(conf, 0.5)


class PlateVotingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config()
        self.reg = MotoRegistry(self.cfg)
        self.rec = self.reg.observe(1, (0, 0, 10, 10), 0.9, datetime.now())

    def test_single_read_is_not_enough(self):
        self.reg.add_plate(self.rec, "ABC12D", 0.9, "test", datetime.now())
        self.assertEqual(self.rec.plate(self.cfg), NO_PLATE)

    def test_consensus_wins_over_one_bad_frame(self):
        for text in ("ABC12D", "ABC12D", "A8C12O", "ABC12D"):
            self.reg.add_plate(self.rec, text, 0.8, "test", datetime.now())
        self.assertEqual(self.rec.plate(self.cfg), "ABC12D")

    def test_no_consensus_means_not_detected(self):
        for text in ("ABC12D", "XYZ98K", "ABC12D", "XYZ98K"):
            self.reg.add_plate(self.rec, text, 0.8, "test", datetime.now())
        self.assertEqual(self.rec.plate(self.cfg), NO_PLATE)

    def test_low_confidence_means_not_detected(self):
        for _ in range(4):
            self.reg.add_plate(self.rec, "ABC12D", 0.2, "test", datetime.now())
        self.assertEqual(self.rec.plate(self.cfg), NO_PLATE)


BOTH = r"^[A-Z]{3}\d{2}[A-Z]$|^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$"


class MultiFormatTests(unittest.TestCase):
    def setUp(self):
        self.p = PlateTextParser(BOTH)

    def test_colombian_still_corrected_by_position(self):
        self.assertEqual(self.p.fit("A8C12D"), ("ABC12D", 1))        # 8 en posición de letra -> B
        self.assertEqual(self.p.fit("ABC1ZD"), ("ABC12D", 1))        # Z en posición de número -> 2

    def test_indonesian_exact(self):
        self.assertEqual(self.p.fit("L6570ABJ"), ("L6570ABJ", 0))
        self.assertEqual(self.p.fit("AG6204KCU"), ("AG6204KCU", 0))

    def test_garbage_rejected(self):
        for bad in ("", "ABCDEF", "123456", "ABCD12E", "L65700ABJ"):
            self.assertIsNone(self.p.fit(bad), bad)

    def test_single_format_unchanged(self):
        single = PlateTextParser(REGEX)
        self.assertEqual(single.template, "LLLDDL")
        self.assertIsNone(single.fit("L6570ABJ"))

    def test_parse_tokens_both_formats(self):
        self.assertEqual(self.p.parse([tok(10, 10, "ABC", 0.9), tok(10, 60, "12D", 0.9)])[0], "ABC12D")
        self.assertEqual(self.p.parse([tok(10, 10, "L", 0.9), tok(10, 40, "6570", 0.9), tok(10, 100, "ABJ", 0.9)])[0], "L6570ABJ")


class _FakeFast:
    """Sustituye a FastPlateOCR sin cargar el modelo."""
    def __init__(self, text, conf):
        self.text, self.conf = text, conf

    def read(self, _img):
        return self.text, self.conf


class _FakeDetector:
    name, reliable = "fake", True

    def detect(self, _moto):
        return [PlateCandidate(np.full((60, 120, 3), 255, np.uint8), (0, 0, 120, 60), 0.9)]


class _FakeEasy:
    def __init__(self, tokens):
        self.available, self.error, self._tokens = True, "", tokens

    def read(self, _img):
        return self._tokens


def make_reader(fast=None, easy=None, regex=REGEX):
    r = PlateReader.__new__(PlateReader)
    r.cfg = SimpleNamespace(PLATE_MIN_WIDTH_PX=36, PLATE_MIN_CONFIDENCE=0.35, PLATE_FAST_MIN_CONFIDENCE=0.90, PLATE_CROP_MARGIN=0.5)
    r.use_gpu, r.detector, r.fast, r._easy = False, _FakeDetector(), fast, easy
    r.parser, r.warning = PlateTextParser(regex), ""
    return r


class FastPlateReaderTests(unittest.TestCase):
    moto = np.zeros((200, 200, 3), np.uint8)

    def test_fast_reading_is_used_and_corrected_by_position(self):
        r = make_reader(fast=_FakeFast("A8C12D", 0.97)).read(self.moto)      # 8 en posición de letra -> B
        self.assertEqual(r.text, "ABC12D")
        self.assertIn("fast-plate-ocr", r.method)

    def test_low_confidence_fast_falls_back_to_easyocr(self):
        easy = _FakeEasy([tok(10, 10, "XYZ98K", 0.8)])
        r = make_reader(fast=_FakeFast("ABC12D", 0.5), easy=easy).read(self.moto)
        self.assertEqual(r.text, "XYZ98K")
        self.assertIn("EasyOCR", r.method)

    def test_wrong_format_is_discarded_never_invented(self):
        self.assertIsNone(make_reader(fast=_FakeFast("L6570ABJ", 0.99), easy=_FakeEasy([])).read(self.moto))

    def test_other_country_regex(self):
        r = make_reader(fast=_FakeFast("L6570ABJ", 0.99), regex=r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$").read(self.moto)
        self.assertEqual(r.text, "L6570ABJ")

    def test_plate_outside_moto_box_is_ignored(self):
        r = make_reader(fast=_FakeFast("ABC12D", 0.97))                # la placa falsa ocupa (0,0,120,60)
        self.assertIsNotNone(r.read(self.moto, (0, 0, 200, 200)))      # dentro de la caja de la moto
        self.assertIsNone(r.read(self.moto, (150, 100, 200, 200)))     # es de la moto vecina

    def test_crop_margin_depends_on_detector(self):
        r = make_reader(fast=_FakeFast("ABC12D", 0.97))
        self.assertEqual(r.crop_margin, 0.5)                       # detector YOLO: con contexto
        r.detector = SimpleNamespace(name="heur", reliable=False)
        self.assertEqual(r.crop_margin, 0.05)                      # heurístico: recorte ajustado

    def test_fast_ocr_without_package_is_unavailable_not_a_crash(self):
        import sys
        saved = sys.modules.get("fast_plate_ocr")
        sys.modules["fast_plate_ocr"] = None                      # simula que no está instalado
        try:
            f = FastPlateOCR("cct-s-v2-global-model")
        finally:
            if saved is None:
                sys.modules.pop("fast_plate_ocr", None)
            else:
                sys.modules["fast_plate_ocr"] = saved
        self.assertFalse(f.available)
        self.assertEqual(f.read(np.zeros((40, 100, 3), np.uint8)), ("", 0.0))


if __name__ == "__main__":
    unittest.main()

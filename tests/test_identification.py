"""Pruebas de placa (validación del formato y votación)."""
from __future__ import annotations

import unittest
from datetime import datetime


from tests.synthetic import test_config
from core.registry import MotoRegistry
from identification.plate import PlateTextParser, template_from_regex
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


if __name__ == "__main__":
    unittest.main()

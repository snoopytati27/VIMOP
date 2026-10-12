"""
El ejemplo numérico resuelto del manual (docs/MANUAL_USUARIO.md, sección 8) debe coincidir
con lo que calcula el código. Si alguien cambia las fórmulas o el manual, esta prueba falla.
"""
from __future__ import annotations

import re
import unittest

from tests.synthetic import MANUAL_BAD, MANUAL_EXCELLENT, MANUAL_SPACES, ROOT, layout_px, moto, test_config, wheels

from analysis.parking_analyzer import ParkingAnalyzer

MANUAL = ROOT / "docs" / "MANUAL_USUARIO.md"

# valores publicados en el manual (1 decimal) -> (O, U, C, I, score, estado)
EXPECTED = {
    "EXCELENTE": (MANUAL_EXCELLENT, 93.4, 100.0, 93.6, 100.0, 96.4),
    "MALO": (MANUAL_BAD, 38.9, 76.9, 31.2, 23.1, 49.1),
}


class ManualExampleTests(unittest.TestCase):
    def setUp(self):
        self.an = ParkingAnalyzer(test_config())
        self.layout = layout_px(MANUAL_SPACES)

    def test_code_matches_published_numbers(self):
        for state, ((bbox, rear, front), o, u, c, i, score) in EXPECTED.items():
            a = self.an.analyze(moto(*bbox), self.layout, wheels=wheels(rear, front))
            got = tuple(round(v, 1) for v in (a.orientation, a.location, a.centering, a.invasion, a.score))
            self.assertEqual(got, (o, u, c, i, score), state)
            self.assertEqual(a.classification, state)
            self.assertEqual(a.space_id, "P01")

    def test_manual_contains_the_same_numbers(self):
        self.assertTrue(MANUAL.exists(), "falta docs/MANUAL_USUARIO.md")
        text = MANUAL.read_text(encoding="utf-8")
        for state, (_, o, u, c, i, score) in EXPECTED.items():
            line = next((ln for ln in text.splitlines() if ln.startswith(f"| **Resultado {state}**")), None)
            self.assertIsNotNone(line, f"no se encontró la fila de resultado {state} en el manual")
            numbers = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", line)]
            self.assertEqual(numbers[:5], [o, u, c, i, score], state)


if __name__ == "__main__":
    unittest.main()

"""Pruebas de la máquina de estados de movimiento."""
from __future__ import annotations

import unittest

from analysis.state_machine import (ST_ENTERING, ST_MOVING, ST_PARKED, ST_PARKING, MotionParams,
                                    MotionStateMachine)

FPS = 10.0


def run(sm: MotionStateMachine, positions, in_space=True, t0=0.0):
    states = []
    for i, p in enumerate(positions):
        states.append(sm.update(t0 + i / FPS, p, in_space)[1])
    return states


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.params = MotionParams(window=0.8, speed_high=30, speed_low=12, stable_seconds=2.0)

    def test_starts_entering(self):
        sm = MotionStateMachine(self.params)
        self.assertEqual(run(sm, [(0, 0)] * 2)[-1], ST_ENTERING)

    def test_still_inside_space_becomes_parked_after_stable_time(self):
        sm = MotionStateMachine(self.params)
        states = run(sm, [(100, 100)] * 35)          # 3.5 s quieta
        self.assertIn(ST_PARKING, states)
        self.assertEqual(states[-1], ST_PARKED)
        first_parked = states.index(ST_PARKED) / FPS
        self.assertGreaterEqual(first_parked, 2.0)

    def test_moving_moto(self):
        sm = MotionStateMachine(self.params)
        states = run(sm, [(i * 10.0, 0) for i in range(20)])   # 100 cm/s
        self.assertEqual(states[-1], ST_MOVING)

    def test_parked_then_leaves(self):
        sm = MotionStateMachine(self.params)
        run(sm, [(0, 0)] * 30)
        self.assertEqual(sm.state, ST_PARKED)
        states = run(sm, [(i * 10.0, 0) for i in range(1, 15)], t0=3.0)
        self.assertEqual(states[-1], ST_MOVING)

    def test_outside_space_needs_double_time(self):
        sm = MotionStateMachine(self.params)
        states = run(sm, [(0, 0)] * 35, in_space=False)
        self.assertNotIn(ST_PARKED, states)          # 3.5 s < 4 s
        states = run(sm, [(0, 0)] * 15, in_space=False, t0=3.5)
        self.assertEqual(states[-1], ST_PARKED)

    def test_small_jitter_does_not_count_as_moving(self):
        sm = MotionStateMachine(self.params)
        jitter = [(100 + (i % 2) * 3.0, 100) for i in range(30)]   # ±3 cm de ruido
        self.assertEqual(run(sm, jitter)[-1], ST_PARKED)


if __name__ == "__main__":
    unittest.main()

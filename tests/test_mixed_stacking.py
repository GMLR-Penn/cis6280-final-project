"""Task-contract tests for terminal height, materials, geometry and deadline."""
import copy
from dataclasses import asdict
import unittest

import mujoco
import numpy as np

from cube_pick.mixed_stacking import (MATERIALS, MixedSceneConfig, MixedStackEnv,
                                     ObjectSpec, stable_height, world_bounds)


def records():
    row = dict(object_velocity=np.zeros((2, 6)),
               object_hand_contacts=np.zeros(2, int), object_ground_contacts=np.zeros(2, int),
               object_table_contacts=np.array([1, 0]), object_contacts=np.array([[0, 1], [1, 0]]),
               object_bounds=np.array([[[-.03, -.03, 0], [.03, .03, .06]],
                                       [[-.01, -.025, .06], [.04, .025, .11]]]))
    return [copy.deepcopy(row) for _ in range(31)]


class MixedHeight(unittest.TestCase):
    def score(self, rows):
        return stable_height(rows, [0, 0])

    def test_physical_height_allows_offset_without_cube_alignment_gate(self):
        result = self.score(records())
        self.assertEqual(result['object_count'], 2)
        self.assertAlmostEqual(result['height_m'], .11)

    def test_single_tall_object_is_a_valid_solution(self):
        rows = records()
        for r in rows:
            r['object_contacts'][:] = 0
            r['object_bounds'][0, 1, 2] = .15
        self.assertEqual(self.score(rows)['object_indices'], [0])
        self.assertAlmostEqual(self.score(rows)['height_m'], .15)

    def test_full_window_required(self):
        self.assertEqual(self.score(records()[:-1])['height_m'], 0)

    def test_hand_touch_disqualifies_connected_structure(self):
        rows = records()
        rows[5]['object_hand_contacts'][1] = 1
        self.assertEqual(self.score(rows)['height_m'], 0)

    def test_airborne_or_unconnected_object_does_not_increase_score(self):
        rows = records()
        for r in rows:
            r['object_contacts'][:] = 0
        self.assertAlmostEqual(self.score(rows)['height_m'], .06)
        for r in rows:
            r['object_table_contacts'][:] = 0
        self.assertEqual(self.score(rows)['height_m'], 0)

    def test_supply_area_and_ground_are_not_scored(self):
        rows = records()
        for r in rows:
            r['object_bounds'][:, :, 0] += .25
        self.assertEqual(self.score(rows)['height_m'], 0)
        rows = records()
        rows[0]['object_ground_contacts'][0] = 1
        self.assertEqual(self.score(rows)['height_m'], 0)

    def test_falling_and_slow_drift_are_not_stable(self):
        rows = records()
        rows[-1]['object_velocity'][1, 5] = -.5
        self.assertEqual(self.score(rows)['height_m'], 0)
        rows = records()
        rows[-1]['object_bounds'][1, :, 0] += .006
        self.assertEqual(self.score(rows)['height_m'], 0)

    def test_past_height_cannot_be_banked(self):
        high = records()
        low = records()
        for r in low:
            r['object_contacts'][:] = 0
        self.assertAlmostEqual(self.score(high+low)['height_m'], .06)

    def test_contact_disappearing_during_window_is_not_persistent(self):
        rows = records()
        rows[-1]['object_contacts'][:] = 0
        self.assertEqual(self.score(rows)['height_m'], 0)


class MixedPhysics(unittest.TestCase):
    def test_seed_roundtrip_and_fixed_color_material_mapping(self):
        a = MixedSceneConfig.from_seed(7)
        b = MixedSceneConfig(**asdict(a))
        self.assertEqual(asdict(a), asdict(b))
        self.assertEqual(asdict(a), asdict(MixedSceneConfig.from_seed(7)))
        self.assertNotEqual(asdict(a), asdict(MixedSceneConfig.from_seed(8)))
        self.assertEqual({s.shape for s in a.objects}, {'cube', 'cuboid', 'cylinder', 'rhombic_prism'})
        e = MixedStackEnv(config=b)
        try:
            obs = e.observation(rgb=False)
            self.assertEqual(set(obs), {'proprio', 'time_remaining', 'goal_xy'})
            self.assertEqual(obs['proprio'].shape, (75,))
            self.assertEqual(e.model.nq, 25+7*4)
            self.assertEqual(e.model.neq, 0)
            self.assertEqual(e.model.nmocap, 0)
            self.assertFalse(set(e.object_geoms) & e.hand_geoms)
            for s, body, geom in zip(b.objects, e.object_ids, e.object_geoms):
                self.assertAlmostEqual(e.model.body_mass[body], s.mass)
                np.testing.assert_allclose(e.model.geom_rgba[geom, :3], MATERIALS[s.material]['rgb'])
                np.testing.assert_allclose(e.data.xpos[body, 2], s.size[2]/2, atol=.002)
            for a, b, fr in zip(e.model.pair_geom1, e.model.pair_geom2, e.model.pair_friction):
                if a in e.object_geoms and b in e.object_geoms:
                    m1 = e.config.objects[e.object_geoms.index(a)].material
                    m2 = e.config.objects[e.object_geoms.index(b)].material
                    self.assertAlmostEqual(fr[0], np.sqrt(MATERIALS[m1]['friction']*MATERIALS[m2]['friction']))
        finally:
            e.close()

    def test_shape_volumes_and_tilted_bounds(self):
        s = ObjectSpec('rhombic_prism', (.08, .06, .05), 'blue', (0, 0))
        self.assertAlmostEqual(s.volume, .08*.06*.05/2)
        c = ObjectSpec('cylinder', (.06, .06, .10), 'orange', (0, 0))
        self.assertAlmostEqual(c.volume, np.pi*.03**2*.10)
        rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        lo, hi = world_bounds(c, np.zeros(3), rotation)
        np.testing.assert_allclose(hi-lo, [.10, .06, .06])
        rotation = np.array([[np.cos(.4), -np.sin(.4), 0], [np.sin(.4), np.cos(.4), 0], [0, 0, 1]])
        lo, hi = world_bounds(s, np.zeros(3), rotation)
        np.testing.assert_allclose(hi, (s.vertices()@rotation.T).max(axis=0))

    def test_timeout_stop_and_reset(self):
        e = MixedStackEnv(time_limit=.2)
        try:
            with self.assertRaises(RuntimeError):
                e.outcome()
            with self.assertRaises(ValueError):
                e.step(np.full(25, np.nan))
            self.assertEqual(e.elapsed_steps, 0)
            e.step(np.zeros(25))
            e.step(np.zeros(25))
            self.assertEqual(e.status, 'finalizing')
            self.assertEqual(e.time_remaining, 0)
            with self.assertRaises(RuntimeError):
                e.step(np.zeros(25))
            while e.status == 'finalizing':
                e.finalize_step()
            outcome = e.outcome()
            self.assertEqual(outcome['stop_reason'], 'time_limit')
            self.assertEqual(outcome['elapsed_seconds'], .2)
            self.assertEqual(outcome['final_stable_height_m'], 0)
            self.assertEqual(e.elapsed_steps, 2)
            e.reset()
            self.assertEqual(e.status, 'running')
            self.assertEqual(e.elapsed_steps, 0)
            e.stop()
            with self.assertRaises(RuntimeError):
                e.stop()
        finally:
            e.close()

    def test_bad_configuration(self):
        for limit in [0, float('inf'), .15]:
            with self.assertRaises(ValueError):
                MixedSceneConfig.from_seed(0, limit)
        with self.assertRaises(ValueError):
            ObjectSpec('cylinder', (.06, .07, .05), 'blue', (0, 0))


if __name__ == '__main__':
    unittest.main()
